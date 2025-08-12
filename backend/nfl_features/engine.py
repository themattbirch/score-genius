# backend/nfl_features/engine.py
"""
NFLFeatureEngine – master orchestrator for all NFL feature modules.

Key behaviors:
  • Canonicalize inputs (dates, teams, season) once, up front.
  • Call modules in a leakage-safe order with FULL historical context.
  • Merge strictly by game_id; each module must return one row per game_id.
  • Wire in advanced, drive, and map stages (with safe fallbacks).
  • Rich debug instrumentation: per-module timing, new cols, nulls, bucket counts.
"""

from __future__ import annotations

import logging
import time
import re
from inspect import signature
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
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
    if right is None or right.empty:
        logger.warning("ENGINE:%s produced empty output – skipping merge", module.upper())
        return left

    if key not in right.columns:
        logger.warning("ENGINE:%s output lacks key '%s' – skipping", module.upper(), key)
        return left

    # enforce one row per key on RHS to avoid duplication
    if right.duplicated(key).any():
        dup_ct = int(right.duplicated(key).sum())
        logger.warning("ENGINE:%s output has %d duplicate %s rows – keeping last", module.upper(), dup_ct, key)
        right = right.sort_values(key).drop_duplicates(key, keep="last")

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
    ):
        self.execution_order = execution_order or self.DEFAULT_ORDER
        self.supabase = _safe_create_client(supabase_url, supabase_service_key)
        logger.info("NFLFeatureEngine initialized | order=%s", self.execution_order)

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

        # Precompute advanced/drive columns on historical (idempotent if repeated)
        if "team_id" in hist_games.columns:
            logger.debug("ENGINE: precomputing ADV/DRIVE on historical frame")
            hist_games = compute_advanced_metrics(hist_games)
            hist_games = compute_drive_metrics(hist_games)

        # Prefetch auxiliary data (use provided frames if present)
        if season_stats_df is None:
            parts = []
            # we need priors for S-1 for every S in the batch
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
            # Carry IDs if present so season.py can join by team_id
            for _idcol in ("home_team_id", "away_team_id"):
                if _idcol in games.columns and _idcol not in spine_cols:
                    spine_cols.append(_idcol)

            season_spine = games.loc[games["season"] == season_token, [c for c in spine_cols if c in games.columns]].copy()
            season_games = season_spine.copy()

            # Full historical context (modules handle leakage internally)
            full_hist = hist_games.copy()

            # Execute modules in order
            for module in self.execution_order:
                t_mod = time.time()

                # Resolve transform
                if module == "situational":
                    fn = compute_situational_features
                elif module == "rest":
                    fn = rest_transform
                elif module == "h2h":
                    fn = h2h_transform
                elif module == "form":
                    fn = form_transform
                elif module == "rolling":
                    fn = rolling_transform
                elif module == "season":
                    fn = season_transform
                elif module == "advanced":
                    fn = advanced_transform
                elif module == "drive":
                    fn = drive_transform
                elif module == "momentum":
                    fn = momentum_transform
                elif module == "map":
                    fn = compute_map_features
                else:
                    logger.warning("ENGINE: unknown module '%s' – skipping", module)
                    continue

                if fn is None:
                    logger.warning("ENGINE:%s transform not available – skipping", module.upper())
                    continue

                # Kwargs handshake
                kw: Dict[str, Any] = {}
                if _supports_kwarg(fn, "debug"): kw["debug"] = debug
                if _supports_kwarg(fn, "flag_imputations"): kw["flag_imputations"] = flag_imputations
                if _supports_kwarg(fn, "historical_df"): kw["historical_df"] = full_hist
                if _supports_kwarg(fn, "historical_team_stats_df"): kw["historical_team_stats_df"] = hist_team_stats

                if module == "season":
                    # Pass full priors; let season.transform handle (season-1) selection and self-heal.
                    if season_stats_df is not None and not season_stats_df.empty:
                        kw["season_stats_df"] = season_stats_df
                    # Coverage hint so season.transform can downgrade logs (e.g., 2021 needs 2020 which may be missing)
                    if season_stats_df is not None and "season" in season_stats_df.columns:
                        available_prev = set(season_stats_df["season"].dropna().astype(int).unique())
                        kw["known_missing_prev"] = (season_int - 1) not in available_prev
                        if debug:
                            logger.debug(
                                "ENGINE: season %s → required_prev=%s | available_prev=%s | known_missing_prev=%s",
                                season_int, season_int - 1, sorted(available_prev), kw["known_missing_prev"]
                            )
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
                    if advanced_stats_df is not None and not advanced_stats_df.empty and _supports_kwarg(fn, "advanced_stats_df"):
                        kw["advanced_stats_df"] = advanced_stats_df.loc[advanced_stats_df.get("season", pd.Series(dtype=int)).isin([season_int - 1, season_int])]
                    # many advanced transforms can also read from historical_df (already precomputed)

                # Call transform on the current running frame (season_games)
                try:
                    mod_out = fn(season_games, **kw)
                except TypeError as e:
                    logger.error("ENGINE:%s call failed (TypeError): %s", module.upper(), e)
                    continue
                except Exception as e:
                    logger.error("ENGINE:%s call failed: %s", module.upper(), e)
                    continue

                if mod_out is None or mod_out.empty:
                    logger.warning("ENGINE:%s returned empty – skipped", module.upper())
                    continue

                # Merge by game_id only
                season_games = _merge_and_log(season_games, mod_out, key="game_id", how="left", module=module, debug=debug)
                logger.debug("ENGINE:%s done in %.3fs", module, time.time() - t_mod)

            chunks.append(season_games)

        if not chunks:
            logger.error("ENGINE: no season chunks produced – returning empty frame.")
            return pd.DataFrame()

        final = pd.concat(chunks, ignore_index=True)
        # Drop helper/private cols
        final.drop(columns=[c for c in final.columns if c.startswith("_")], inplace=True, errors="ignore")

        _log_profile(final, "FINAL_FEATURES", debug)
        _summarize_by_bucket(final.drop(columns=["game_id"], errors="ignore"), "FINAL_FEATURES", debug)
        logger.info("ENGINE: complete in %.2fs | final shape=%s", time.time() - t0, final.shape)
        return final
