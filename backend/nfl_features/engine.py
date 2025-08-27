# backend/nfl_features/engine.py
"""
NFLFeatureEngine – master orchestrator for all NFL feature modules.

Key behaviors:
  • Canonicalize inputs (dates, teams, season) once, up front.
  • Call modules in a leakage-safe order with FULL historical context.
  • Merge strictly by game_id; each module must return one row per game_id.
  • Wire in advanced, drive, and map stages (with safe fallbacks).
  • Rich debug instrumentation: per-module timing, new cols, nulls, bucket counts.
  • Drop constant columns (all-NaN or single value) at the end to stabilize models.
"""

from __future__ import annotations

import time
import re
from inspect import signature
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.exc import NoSuchModuleError
import os

import logging
log = logging.getLogger(__name__)

# --- Supabase safe client loader ------------------------------------------------
try:
    from supabase import create_client as _sb_create_client, Client as _SBClient  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    try:
        from supabase_py import create_client as _sb_create_client, Client as _SBClient  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        _sb_create_client, _SBClient = None, None  # type: ignore


class _SupabaseStub:
    def __getattr__(self, _name: str) -> Any:
        def _stub(*_a, **_kw): ...
        return _stub


_VALID_URL_RE = re.compile(r"^(https?)://.+")


def _safe_create_client(url: str, key: str):
    if _sb_create_client is None or not _VALID_URL_RE.match(url or ""):
        return _SupabaseStub()
    try:
        return _sb_create_client(url, key)
    except Exception:  # pragma: no cover
        return _SupabaseStub()

# --- Feature modules -----------------------------------------------------------
# Precompute helpers on historical (ok to keep)
from .advanced import compute_advanced_metrics
from .drive_metrics import compute_drive_metrics

# Canonical transforms (these MUST return one row per game_id with prefixed cols)
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .momentum import transform as momentum_transform
from .rest import transform as rest_transform
from .rolling import load_rolling_features as rolling_transform
from .season import transform as season_transform
from .situational import compute_situational_features

# Optional transforms (advanced/drive/map) – tolerate absence gracefully
try:
    from .advanced import transform as advanced_transform  # type: ignore
except Exception:  # pragma: no cover
    advanced_transform = None  # type: ignore

try:
    from .drive import transform as drive_transform  # type: ignore
except Exception:  # pragma: no cover
    drive_transform = None  # type: ignore

try:
    from .map import compute_map_features  # type: ignore
except Exception:  # pragma: no cover
    compute_map_features = None  # type: ignore

from .utils import determine_season, normalize_team_name

# --- Logging -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["NFLFeatureEngine"]


# ==============================================================================
# Utility / Debug helpers
# ==============================================================================

def _supports_kwarg(func: Any, kw: str) -> bool:
    try:
        return kw in signature(func).parameters
    except (ValueError, TypeError):
        return False


def _profile_df(df: pd.DataFrame, name: str, top_n: int = 10) -> Dict[str, Any]:
    if df.empty:
        return {"name": name, "rows": 0, "cols": 0, "null_top": {}, "constant_cols": []}
    null_counts = df.isna().sum().sort_values(ascending=False)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": null_counts.head(top_n).to_dict(),
        "constant_cols": const_cols[:top_n],
    }


def _log_profile(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    p = _profile_df(df, name)
    logger.debug(
        "[PROFILE] %-20s rows=%-6d cols=%-4d | null_top=%s | const=%s",
        p["name"], p["rows"], p["cols"], p["null_top"], p["constant_cols"],
    )


def _merge_and_log(left: pd.DataFrame, right: pd.DataFrame, key: str, how: str, module: str, debug: bool) -> pd.DataFrame:
    # Be loud if a whole stage produced nothing (can silently kill a feature family)
    if right is None or right.empty:
        logger.warning("ENGINE:%s produced EMPTY output; no features merged for this stage", module.upper())
        return left

    if key not in right.columns:
        logger.warning("ENGINE:%s output lacks key '%s' – skipping", module.upper(), key)
        return left

    # enforce one row per key on RHS to avoid duplication
    if right.duplicated(key).any():
        dup_ct = int(right.duplicated(key).sum())
        logger.warning("ENGINE:%s output has %d duplicate %s rows – keeping last", module.upper(), dup_ct, key)
        right = right.sort_values(key).drop_duplicates(key, keep="last")

    # avoid overwriting existing columns except the key
    dup_cols = [c for c in right.columns if c in left.columns and c != key]
    if dup_cols:
        right = right.drop(columns=dup_cols)

    before_cols = set(left.columns)
    t0 = time.time()
    out = left.merge(right, on=key, how=how)
    dt = time.time() - t0

    new_cols = [c for c in out.columns if c not in before_cols]
    if debug:
        logger.debug("ENGINE:%s merge in %.3fs | +%d cols | sample=%s",
                     module, dt, len(new_cols), new_cols[:25])
        if new_cols:
            sub = out[new_cols]
            nulls = sub.isna().sum().sort_values(ascending=False).head(10).to_dict()
            zeros = (sub.select_dtypes(include=np.number) == 0).sum().sort_values(ascending=False).head(10).to_dict()
            logger.debug("ENGINE:%s new_col nulls(top10)=%s", module, nulls)
            logger.debug("ENGINE:%s new_col zeros(top10)=%s", module, zeros)

    return out


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize dates, teams, season."""
    out = df.copy()

    # kickoff_ts → UTC datetime
    if "kickoff_ts" in out.columns:
        out["kickoff_ts"] = pd.to_datetime(out["kickoff_ts"], errors="coerce", utc=True)

    # game_date present (naive UTC date); derive from kickoff_ts if missing
    if "game_date" not in out.columns or out["game_date"].isna().all():
        if "kickoff_ts" in out.columns and out["kickoff_ts"].notna().any():
            out["game_date"] = out["kickoff_ts"].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            out["game_date"] = pd.NaT
    else:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    # normalized teams
    if "home_team_norm" not in out.columns:
        if "home_team_id" in out.columns:
            out["home_team_norm"] = out["home_team_id"].apply(normalize_team_name)
        elif "home_team" in out.columns:
            out["home_team_norm"] = out["home_team"].apply(normalize_team_name)

    if "away_team_norm" not in out.columns:
        if "away_team_id" in out.columns:
            out["away_team_norm"] = out["away_team_id"].apply(normalize_team_name)
        elif "away_team" in out.columns:
            out["away_team_norm"] = out["away_team"].apply(normalize_team_name)

    for c in ("home_team_norm", "away_team_norm"):
        if c in out.columns:
            out[c] = out[c].astype(str).str.lower()

    # season
    if "season" not in out.columns:
        out["season"] = out["game_date"].apply(determine_season)

    return out

def create_pg_engine_safely(db_url_env: str | None = None):
    """
    Create a SQLAlchemy engine for Postgres.
    - Normalizes 'postgres://' -> 'postgresql://'
    - Tries bare 'postgresql://' first (auto-uses installed DBAPI)
    - If the DBAPI isn't found, retries with 'postgresql+psycopg2://'
    """
    raw = db_url_env or os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not raw:
        raise RuntimeError("No DATABASE_URL/SUPABASE_DB_URL provided")

    # Normalize deprecated scheme (common in Supabase strings)
    url = raw.replace("postgres://", "postgresql://", 1)

    # If user already supplied an explicit driver (e.g., postgresql+psycopg2://), use as-is
    if "postgresql+" in url:
        return create_engine(url)

    # Try with whatever DBAPI SQLAlchemy finds (requires an installed driver)
    try:
        return create_engine(url)
    except NoSuchModuleError as e:
        # Likely no driver installed for 'postgresql://'
        log.warning("No Postgres DBAPI found for URL '%s' (%s). Retrying with psycopg2...", url, e)

    # Fallback: force psycopg2
    forced = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(forced)

def _build_kickoff_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Try to derive a UTC time window [lo, hi] from season_games.
    """
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if "scheduled_time" in df.columns:
        try:
            ts = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True).fillna(ts)
        except Exception:
            pass

    if "kickoff_ts" in df.columns:
        try:
            kt = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
            ts = ts.fillna(kt)
        except Exception:
            pass

    if ts.isna().all() and ("game_date" in df.columns):
        date_str = df.get("game_date").astype(str)
        time_str = df.get("game_time", pd.Series("00:00:00", index=df.index)).astype(str)
        combo = (date_str + " " + time_str).str.strip()
        try:
            et = pd.to_datetime(combo, errors="coerce").dt.tz_localize(
                "America/New_York", ambiguous="infer", nonexistent="shift_forward"
            )
            ts = et.dt.tz_convert("UTC")
        except Exception:
            pass

    if ts.notna().any():
        lo = ts.min() - pd.Timedelta(days=7)
        hi = ts.max() + pd.Timedelta(days=7)
        return lo, hi
    return None, None

def fetch_team_snapshot_weather(engine) -> pd.DataFrame:
    """
    Pulls team-level snapshot rows from public.weather_forecast_snapshots for NFL.
    Expected columns: team_name_norm, temperature_f, humidity_pct, wind_speed_mph, wind_deg
    """
    table = "public.weather_forecast_snapshots"
    cols = """
        team_name_norm,
        temperature_f,
        humidity_pct,
        wind_speed_mph,
        wind_deg
    """
    q = text(f"""
        SELECT {cols}
        FROM {table}
        WHERE sport = 'NFL'
        ORDER BY team_name_norm, COALESCE(temperature_f, NULL) NULLS LAST
    """)
    try:
        with engine.begin() as conn:
            df = pd.read_sql_query(q, conn)
        # Keep newest / last per team_name_norm (the ORDER BY above is just a hint; drop_dupes is the guardrail)
        if "team_name_norm" in df.columns:
            df["team_name_norm"] = df["team_name_norm"].astype(str).str.strip().str.lower()
            df = df.drop_duplicates("team_name_norm", keep="last")
        return df
    except Exception as e:
        log.warning("ENGINE: snapshot weather fetch failed: %s", e)
        return pd.DataFrame()


# engine.py
def fetch_latest_weather_for_games(engine, season_games: pd.DataFrame, *, live_window_days: int = 30) -> pd.DataFrame:
    """
    Fetch live forecasts ONLY when the target games are near 'now'.
    Otherwise, return empty so map.py uses climo/snapshot fallbacks.
    """
    cols = "*"
    table = "public.weather_nfl_latest_forecast_per_game"
    now = pd.Timestamp.now(tz="UTC")

    lo, hi = _build_kickoff_bounds(season_games)

    # If we can’t infer a window, don’t fetch (prevents grabbing the whole MV).
    if lo is None or hi is None:
        return pd.DataFrame()

    # If the latest target kickoff is clearly historical, skip live weather.
    if hi < now - pd.Timedelta(days=live_window_days):
        return pd.DataFrame()

    with engine.begin() as conn:
        # Strict time-window query only (no “whole MV” fallback here)
        q = text(f"""
            SELECT {cols}
            FROM {table}
            WHERE scheduled_time BETWEEN :lo AND :hi
            ORDER BY scheduled_time
        """)
        df = pd.read_sql_query(q, conn, params={"lo": lo, "hi": hi})
        return df

def _bucket_for(col: str) -> str:
    c = col.lower()

    # Explicit prev-season routing (do this FIRST so it doesn't get caught by home_/away_)
    if c.startswith("prev_season_") or "_prev_season_" in c \
       or c.startswith("home_prev_season_") or c.startswith("away_prev_season_"):
        return "season"

    if c.startswith("h2h_"): return "h2h"
    if c.startswith("rolling_") or "rolling_" in c: return "rolling"
    if c.startswith("rest_") or "rest_days" in c or "short_week" in c or "off_bye" in c: return "rest"
    if c.startswith("form_") or "form_win_pct" in c or "momentum" in c: return "form/momentum"
    if c.startswith("drive_"): return "drive"
    if c.startswith("adv_"): return "advanced"
    if c.startswith("season_"): return "season"
    if c.startswith("situational_") or c in ("week", "day_of_week", "is_division_game", "is_conference_game"): return "situational"
    if c.startswith("total_"): return "engineered_total"
    if c.startswith("home_") or c.startswith("away_"): return "raw_home_away"
    if c.startswith("map_"): return "map"
    return "other"


def _summarize_by_bucket(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    numcols = [c for c in df.columns if c != "game_id"]
    counts = {}
    for c in numcols:
        b = _bucket_for(c)
        counts[b] = counts.get(b, 0) + 1
    logger.debug("BUCKETS[%s]: %s", name, counts)


def _drop_constant_columns(df: pd.DataFrame, *, exclude: tuple[str, ...] = ("game_id",), debug: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in df.columns if c not in exclude]
    nunq = {c: df[c].nunique(dropna=False) for c in cols}

    # ⬇️ Only drop constants that are NOT boolean dtype
    const_cols = [
        c for c, k in nunq.items()
        if k <= 1 and not pd.api.types.is_bool_dtype(df[c])
    ]

    if const_cols:
        if debug:
            logger.debug("ENGINE: dropping %d constant cols (sample=%s)", len(const_cols), const_cols[:18])
        else:
            logger.info("ENGINE: dropping %d constant cols", len(const_cols))
        df = df.drop(columns=const_cols, errors="ignore")
    return df



# ==============================================================================
# Engine
# ==============================================================================

class NFLFeatureEngine:
    """Runs the full NFL feature-engineering pipeline using RPCs for data."""

    DEFAULT_ORDER: List[str] = [
        "situational",
        "rest",
        "h2h",
        "form",      # uses historical_df; leakage-safe (asof/shift)
        "rolling",   # uses recent_form_df (already lagged at source)
        "season",    # season-to-date anchors (use prior-season baseline early)
        "advanced",  # adjusted efficiency/quality signals
        "drive",     # per-drive rates
        "momentum",  # optional; after core signals
        "map",       # final mapping/enrichment stage, if any
    ]

    def __init__(
        self,
        supabase_url: str,
        supabase_service_key: str,
        execution_order: Optional[List[str]] = None,
        *,
        # ---- DRIVE stage knobs ----
        drive_window: int = 5,
        drive_reset_by_season: bool = True,
        drive_min_prior_games: int = 0,
        drive_soft_fail: bool = True,
        disable_drive_stage: bool = False,
    ):
        """
        Initialize the feature engine.

        Parameters:
        execution_order: Optional override for module order.
        drive_window: Rolling window size for DRIVE (games).
        drive_reset_by_season: If True, rolling in DRIVE resets each season.
        drive_min_prior_games: If >0, DRIVE masks *_avg to NaN when prior games < threshold.
        drive_soft_fail: If True, DRIVE returns a sentinel frame on error and keeps pipeline running.
        disable_drive_stage: If True, the DRIVE module is removed from execution order.
        """
        # Order
        self.execution_order = execution_order or self.DEFAULT_ORDER[:]
        self.disable_drive_stage = bool(disable_drive_stage)
        if self.disable_drive_stage and "drive" in self.execution_order:
            self.execution_order = [m for m in self.execution_order if m != "drive"]

        # Supabase client (safe/stubbed)
        self.supabase = _safe_create_client(supabase_url, supabase_service_key)

        # DRIVE config
        self.drive_window = int(drive_window)
        self.drive_reset_by_season = bool(drive_reset_by_season)
        self.drive_min_prior_games = int(drive_min_prior_games)
        self.drive_soft_fail = bool(drive_soft_fail)

        self.TRANSFORMS = {
            "situational": compute_situational_features,
            "rest":        rest_transform,
            "h2h":         h2h_transform,
            "form":        form_transform,
            "rolling":     rolling_transform,
            "season":      season_transform,
            "advanced":    advanced_transform,   # may be None; handled below
            "drive":       drive_transform,      # may be None; handled below
            "momentum":    momentum_transform,
            "map":       compute_map_features,
        }


        logger.info(
            "NFLFeatureEngine initialized | order=%s | DRIVE{window=%s, reset_by_season=%s, min_prior=%s, soft_fail=%s, disabled=%s}",
            self.execution_order,
            self.drive_window,
            self.drive_reset_by_season,
            self.drive_min_prior_games,
            self.drive_soft_fail,
            self.disable_drive_stage,
        )

    # --- RPC helpers (optional) ------------------------------------------------
    def _fetch_data_via_rpc(self, rpc_name: str, params: Dict) -> pd.DataFrame:
        try:
            resp = self.supabase.rpc(rpc_name, params).execute()
            return pd.DataFrame(resp.data or [])
        except Exception as e:
            logger.error("RPC '%s' failed: %s", rpc_name, e)
            return pd.DataFrame()

    def _fetch_season_stats_via_rpc(self, season: int) -> pd.DataFrame:
        return self._fetch_data_via_rpc("rpc_get_nfl_all_season_stats", {"p_season": season})

    def _fetch_recent_form_via_rpc(self) -> pd.DataFrame:
        return self._fetch_data_via_rpc("rpc_get_nfl_all_recent_form", {})

    def _fetch_advanced_stats_via_rpc(self, season: int) -> pd.DataFrame:
        return self._fetch_data_via_rpc("rpc_get_nfl_all_advanced_stats", {"p_season": season})

    # --- MAP stage helpers (optional tables/views) ----------------------------
    def _fetch_latest_weather_table(self) -> pd.DataFrame:
        """Expect one-most-recent forecast row per game_id."""
        try:
            # Use a view/materialized view that resolves to latest per game_id.
            resp = self.supabase.table("weather_nfl_latest_forecast_per_game").select("*").execute()
            return pd.DataFrame(resp.data or [])
        except Exception as e:
            logger.warning("ENGINE: weather latest fetch failed: %s", e)
            return pd.DataFrame()

    # --- Main entrypoint -------------------------------------------------------
    def build_features(
        self,
        games_df: pd.DataFrame,
        *,
        historical_games_df: pd.DataFrame,
        historical_team_stats_df: pd.DataFrame,
        # Optional pre-fetched frames (engine will fall back to RPC if desired)
        season_stats_df: Optional[pd.DataFrame] = None,
        recent_form_df: Optional[pd.DataFrame] = None,
        advanced_stats_df: Optional[pd.DataFrame] = None,
        debug: bool = False,
        flag_imputations: bool = True,
        rolling_window: int = 3,
        form_lookback: int = 5,
        momentum_span: int = 5,
        h2h_max_games: int = 10,
        # ------------ DRIVE stage controls (NEW) ------------
        window: int = 10,
        reset_by_season: bool = True,
        min_prior_games: int = 0,
        soft_fail: bool = True,
        disable_drive_stage: bool = False,
        # ------------ Finalization ------------
        drop_constant_cols: bool = True,
        ) -> pd.DataFrame:

        if debug:
            logger.setLevel(logging.DEBUG)

        if games_df.empty:
            logger.warning("ENGINE: games_df empty; nothing to do.")
            return pd.DataFrame()

        t0 = time.time()
        logger.debug("ENGINE: build_features START | rows=%d cols=%d", *games_df.shape)

        # Canonicalize
        games = _canon(games_df)
        hist_games = _canon(historical_games_df)
        hist_team_stats = historical_team_stats_df.copy()

        # Seasons present in this batch
        all_seasons = sorted(games["season"].dropna().unique())
        logger.debug("ENGINE: seasons in batch → %s", all_seasons)

        # Prefetch auxiliary data (use provided frames if present)
        if season_stats_df is None:
            parts = []
            prev_seasons = sorted({int(s) - 1 for s in all_seasons if pd.notna(s)})
            for ps in prev_seasons:
                try:
                    prior = self._fetch_season_stats_via_rpc(int(ps))
                    if not prior.empty:
                        parts.append(prior)
                except Exception:
                    continue
            season_stats_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if recent_form_df is None:
            recent_form_df = self._fetch_recent_form_via_rpc()

        if advanced_stats_df is None:
            parts = []
            for s in all_seasons:
                try:
                    s_int = int(s)
                except Exception:
                    continue
                parts.append(self._fetch_advanced_stats_via_rpc(s_int))
            advanced_stats_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        _log_profile(season_stats_df, "season_stats_df (priors)", debug)
        if debug and season_stats_df is not None and not season_stats_df.empty and "season" in season_stats_df.columns:
            avail_prev = sorted(pd.Series(season_stats_df["season"]).dropna().astype(int).unique().tolist())
            logger.debug("ENGINE: priors available for seasons → %s", avail_prev)

        _log_profile(recent_form_df, "recent_form_df", debug)
        _log_profile(advanced_stats_df, "advanced_stats_df", debug)

        # Optional: fetch MAP inputs once (safe if empty)
        weather_latest_prefetched = self._fetch_latest_weather_table()  # not used for map; kept for visibility only
        _log_profile(weather_latest_prefetched, "weather_latest_prefetched", debug)

        # Build per-season to respect within-season chronology
        chunks: List[pd.DataFrame] = []
        for season_token in all_seasons:
            season_int = int(season_token)
            logger.info("ENGINE: season %s start", season_int)

            # Schedule spine for this season
            spine_cols = [
                "game_id", "game_date", "game_time", "kickoff_ts",
                "season", "week",
                "home_team_norm", "away_team_norm",
                "home_score", "away_score",
            ]
            for _idcol in ("home_team_id", "away_team_id"):
                if _idcol in games.columns and _idcol not in spine_cols:
                    spine_cols.append(_idcol)

            season_spine = games.loc[
                games["season"] == season_token,
                [c for c in spine_cols if c in games.columns]
            ].copy()
            season_games = season_spine.copy()

            # Full historical context (modules handle leakage internally)
            full_hist = hist_games.copy()

            # Execute modules in order
            for module in self.execution_order:
                t_mod = time.time()

                # Resolve from registry (tests monkeypatch eng.TRANSFORMS[mod])
                fn = self.TRANSFORMS.get(module)

                # Respect disable flag for drive
                if module == "drive" and disable_drive_stage:
                    logger.info("ENGINE:DRIVE disabled via flag; skipping.")
                    continue

                if fn is None:
                    logger.warning("ENGINE:%s transform not available – skipping", module.upper())
                    continue

                # ---- Consolidated kwargs handshake (no reassign) ----
                kw: Dict[str, Any] = {}

                # Common flags
                if _supports_kwarg(fn, "debug"):
                    kw["debug"] = debug
                if _supports_kwarg(fn, "flag_imputations"):
                    kw["flag_imputations"] = flag_imputations

                # Historical context (safe copies)
                if _supports_kwarg(fn, "historical_df"):
                    kw["historical_df"] = full_hist.copy(deep=True)
                if _supports_kwarg(fn, "historical_team_stats_df"):
                    kw["historical_team_stats_df"] = hist_team_stats.copy(deep=True)

                # Module-specific
                if module == "season":
                    # Provide priors and missing-prev hint
                    if season_stats_df is not None and not season_stats_df.empty:
                        kw["season_stats_df"] = season_stats_df
                    if season_stats_df is not None and "season" in season_stats_df.columns:
                        available_prev = set(season_stats_df["season"].dropna().astype(int).unique())
                        kw["known_missing_prev"] = (season_int - 1) not in available_prev
                        if debug:
                            logger.debug(
                                "ENGINE: season %s → required_prev=%s | available_prev=%s | known_missing_prev=%s",
                                season_int, season_int - 1, sorted(available_prev), kw["known_missing_prev"]
                            )
                    # Safety: prefer league averages over global default if priors missing
                    if _supports_kwarg(fn, "prefer_league_avg_if_missing"):
                        kw["prefer_league_avg_if_missing"] = True
                    # Pass season series if the transform supports it
                    if _supports_kwarg(fn, "season") and "season" in season_games.columns:
                        kw["season"] = season_games["season"]

                elif module == "rolling":
                    if recent_form_df is not None and not recent_form_df.empty:
                        kw["recent_form_df"] = recent_form_df
                    kw["window"] = rolling_window

                elif module == "form":
                    kw["lookback_window"] = form_lookback

                elif module == "momentum":
                    kw["span"] = momentum_span

                elif module == "h2h":
                    kw["max_games"] = h2h_max_games

                elif module == "advanced":
                    if (
                        advanced_stats_df is not None
                        and not advanced_stats_df.empty
                        and _supports_kwarg(fn, "advanced_stats_df")
                    ):
                        kw["advanced_stats_df"] = advanced_stats_df.loc[
                            advanced_stats_df.get("season", pd.Series(dtype=int)).isin([season_int - 1, season_int])
                        ]

                elif module == "drive":
                    # Respect disable flag was handled above; pass controls if supported
                    if _supports_kwarg(fn, "window"):
                        kw["window"] = int(window)
                    if _supports_kwarg(fn, "reset_by_season"):
                        kw["reset_by_season"] = bool(reset_by_season)
                    if _supports_kwarg(fn, "min_prior_games"):
                        kw["min_prior_games"] = int(min_prior_games)
                    if _supports_kwarg(fn, "soft_fail"):
                        kw["soft_fail"] = bool(soft_fail)

                elif module == "map":
                    # --- fresh per-run weather for the current working frame ---
                    try:
                        engine = create_pg_engine_safely()
                    except Exception as e:
                        logger.warning("ENGINE:MAP could not init DB engine: %s", e)
                        engine = None

                    try:
                        if engine is not None:
                            weather_latest_df = fetch_latest_weather_for_games(engine, season_games, live_window_days=30)
                            snapshot_df = fetch_team_snapshot_weather(engine)  # <— NEW
                        else:
                            weather_latest_df = pd.DataFrame()
                            snapshot_df = pd.DataFrame()                     # <— NEW
                    except Exception as e:
                        logger.warning("ENGINE:MAP weather fetch failed: %s", e)
                        weather_latest_df = pd.DataFrame()
                        snapshot_df = pd.DataFrame()     
                                        # quick, high-signal diagnostics (no-op if empty)
                    if debug:
                        try:
                            kick_lo, kick_hi = _build_kickoff_bounds(season_games)
                            logger.debug("ENGINE:MAP kickoff window UTC → [%s, %s]", kick_lo, kick_hi)
                            if not weather_latest_df.empty and "scheduled_time" in weather_latest_df.columns:
                                st = pd.to_datetime(weather_latest_df["scheduled_time"], errors="coerce", utc=True)
                                logger.debug("ENGINE:MAP weather rows=%d | sched[min=%s, max=%s]",
                                             len(weather_latest_df), st.min(), st.max())
                        except Exception:
                            pass

                    # keep ids visually consistent in downstream logs
                    try:
                        season_games["game_id"] = season_games["game_id"].astype(str).str.strip()
                    except Exception:
                        pass

                    # Provide optional exogenous inputs; compute_map_features tolerates empties
                    if _supports_kwarg(fn, "weather_latest_df") and not weather_latest_df.empty:
                        kw["weather_latest_df"] = weather_latest_df
                    if _supports_kwarg(fn, "snapshot_df") and not snapshot_df.empty:     # <— NEW
                        kw["snapshot_df"] = snapshot_df
                
                # Call transform on the current running frame (season_games)
                try:
                    mod_out = fn(season_games, **kw)
                except TypeError as e:
                    logger.error("ENGINE:%s call failed (TypeError): %s", module.upper(), e)
                    continue
                except Exception as e:
                    if module == "drive" and soft_fail:
                        logger.error("ENGINE:DRIVE soft-fail: %s", e)
                        continue
                    logger.error("ENGINE:%s call failed: %s", module.upper(), e)
                    continue

                if mod_out is None or mod_out.empty:
                    # TINY TWEAK already handled in _merge_and_log, but also guard here
                    logger.info("ENGINE:%s returned empty – skipped", module.upper())
                    continue

                # Merge by game_id only
                season_games = _merge_and_log(
                    season_games, mod_out, key="game_id", how="left", module=module, debug=debug
                )
                logger.debug("ENGINE:%s done in %.3fs", module, time.time() - t_mod)

            chunks.append(season_games)

        if not chunks:
            logger.error("ENGINE: no season chunks produced – returning empty frame.")
            return pd.DataFrame()

        final = pd.concat(chunks, ignore_index=True)
        # Drop helper/private cols
        final.drop(columns=[c for c in final.columns if c.startswith("_")], inplace=True, errors="ignore")

        # Drop constant columns (stabilize downstream selection/fit)
        if drop_constant_cols:
            final = _drop_constant_columns(final, exclude=("game_id",), debug=debug)

        _log_profile(final, "FINAL_FEATURES", debug)
        _summarize_by_bucket(final.drop(columns=["game_id"], errors="ignore"), "FINAL_FEATURES", debug)
        logger.info("ENGINE: complete in %.2fs | final shape=%s", time.time() - t0, final.shape)
        return final
