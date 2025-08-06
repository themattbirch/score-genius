# backend/nfl_features/engine.py
"""
NFLFeatureEngine – master orchestrator for all NFL feature modules.

Now with aggressive DEBUG instrumentation:
  • Module-by-module timing, shape deltas, and new-column lists.
  • Null/zero/constant diagnostics per module merge.
  • Optional basic outlier scan (z-score|MAD) to flag extreme values.
  • Clear logging around dropped / non-numeric / helper columns.
"""

from __future__ import annotations

import logging
import time
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

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
    if _sb_create_client is None or not _VALID_URL_RE.match(url):
        return _SupabaseStub()
    try:
        return _sb_create_client(url, key)
    except Exception:  # pragma: no cover
        return _SupabaseStub()

# --- Feature modules -----------------------------------------------------------
from .advanced import compute_advanced_metrics
from .drive_metrics import compute_drive_metrics
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .momentum import transform as momentum_transform
from .rest import transform as rest_transform
from .rolling import load_rolling_features
from .season import transform as season_transform
from .situational import compute_situational_features

from .utils import (
    determine_season,
    normalize_team_name,
)

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
    """Return a quick profile dict for debug logging."""
    if df.empty:
        return {"name": name, "rows": 0, "cols": 0}

    numeric = df.select_dtypes(include=np.number)
    null_counts = df.isna().sum().sort_values(ascending=False)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    prof = {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": null_counts.head(top_n).to_dict(),
        "all_null_cols": [c for c, v in null_counts.items() if v > 0],
        "constant_cols": const_cols,
    }
    return prof

def _log_profile(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    prof = _profile_df(df, name)
    logger.debug(
        "[PROFILE] %s → rows=%d, cols=%d, top_null=%s, const=%s",
        prof["name"], prof["rows"], prof["cols"],
        prof.get("null_top"), prof.get("constant_cols"),
    )

def _detect_outliers(df: pd.DataFrame, z_thresh: float = 6.0) -> List[str]:
    """Return list of columns with extreme z-score values (very rough heuristic)."""
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return []
    z = (num - num.mean()) / num.std(ddof=0)
    mask = (np.abs(z) > z_thresh).any(axis=0)
    return list(mask[mask].index)

def _merge_and_log(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    how: str,
    module: str,
    debug: bool,
) -> pd.DataFrame:
    """Merge two frames and emit diagnostics about new columns, nulls, etc."""
    if right.empty:
        logger.warning("ENGINE:%s produced empty output – skipping merge", module.upper())
        return left

    before_cols = set(left.columns)
    t_start = time.time()
    out = left.merge(right, on=key, how=how)
    elapsed = time.time() - t_start

    new_cols = [c for c in out.columns if c not in before_cols]
    if debug:
        logger.debug("ENGINE:%s merge done in %.3fs; new_cols=%d → %s",
                     module, elapsed, len(new_cols), new_cols[:25])

        # Null / zero checks for just the new columns
        if new_cols:
            sub = out[new_cols]
            nulls = sub.isna().sum().sort_values(ascending=False)
            zeros = (sub == 0).sum().sort_values(ascending=False)
            logger.debug("ENGINE:%s new_col nulls (top10): %s", module, nulls.head(10).to_dict())
            logger.debug("ENGINE:%s new_col zeros  (top10): %s", module, zeros.head(10).to_dict())

            outliers = _detect_outliers(sub)
            if outliers:
                logger.debug("ENGINE:%s potential outlier cols (z>6): %s", module, outliers[:25])

    return out


# ==============================================================================
# Engine
# ==============================================================================

class NFLFeatureEngine:
    """Runs the full NFL feature-engineering pipeline using RPCs for data."""

    DEFAULT_ORDER: List[str] = [
        "situational",
        "season",
        "rest",
        "rolling",  # Note: This module will consume 'recent_form' data
        "form",
        "momentum",
        "h2h",
    ]

    TRANSFORMS: Dict[str, Any] = {
        "situational": compute_situational_features,
        "season": season_transform,
        "rest": rest_transform,
        "rolling": load_rolling_features,
        "form": form_transform,
        "momentum": momentum_transform,
        "h2h": h2h_transform,
    }

    def __init__(
        self,
        supabase_url: str,
        supabase_service_key: str,
        execution_order: Optional[List[str]] = None,
    ):
        self.execution_order = execution_order or self.DEFAULT_ORDER
        self.supabase = _safe_create_client(supabase_url, supabase_service_key)
        logger.info("NFLFeatureEngine initialized | order=%s", self.execution_order)
        self._ext_tracker: Optional[List[Dict[str, Any]]] = None # Tracker for tests

    # --- NEW: Private helper methods to call our bulk RPCs ---
    def _fetch_data_via_rpc(self, rpc_name: str, params: Dict) -> pd.DataFrame:
        logger.debug("Calling RPC '%s' with params %s", rpc_name, params)
        try:
            resp = self.supabase.rpc(rpc_name, params).execute()
            return pd.DataFrame(resp.data or [])
        except Exception as e:
            logger.error("RPC call '%s' failed: %s", rpc_name, e)
            return pd.DataFrame()

    def _fetch_season_stats_via_rpc(self, season: int) -> pd.DataFrame:
        logger.info("Fetching NFL season stats for season %d via RPC...", season)
        return self._fetch_data_via_rpc('rpc_get_nfl_all_season_stats', {'p_season': season})

    def _fetch_recent_form_via_rpc(self) -> pd.DataFrame:
        logger.info("Fetching NFL recent form data for all teams via RPC...")
        return self._fetch_data_via_rpc('rpc_get_nfl_all_recent_form', {})

    def _fetch_advanced_stats_via_rpc(self, season: int) -> pd.DataFrame:
        logger.info("Fetching NFL advanced stats for season %d via RPC...", season)
        return self._fetch_data_via_rpc('rpc_get_nfl_all_advanced_stats', {'p_season': season})


    def build_features(
        self,
        games_df: pd.DataFrame,
        *,
        historical_games_df: pd.DataFrame,
        historical_team_stats_df: pd.DataFrame,
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
            logger.warning("ENGINE: games_df is empty; nothing to do.")
            return pd.DataFrame()

        t0 = time.time()
        logger.debug("ENGINE: build_features start | rows=%d cols=%d", *games_df.shape)

        # --- MODIFIED: Canonicalization (which creates the 'season' column) now happens FIRST ---
        def _canon(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "home_team_norm" not in out.columns and "home_team_id" in out.columns:
                 out["home_team_norm"] = out["home_team_id"].apply(normalize_team_name)
            if "away_team_norm" not in out.columns and "away_team_id" in out.columns:
                 out["away_team_norm"] = out["away_team_id"].apply(normalize_team_name)
            if "game_date" in out.columns:
                out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
                if "season" not in out.columns:
                    out["season"] = out["game_date"].apply(determine_season)
            return out
        
        games = _canon(games_df)
        hist_games = _canon(historical_games_df)
        hist_team_stats = historical_team_stats_df.copy()

        # --- MODIFIED: This line can now safely access the 'season' column ---
        all_seasons = sorted(games["season"].dropna().unique())
        
        # Pre-fetch all necessary data for all seasons in the batch via RPCs
        all_season_stats_df = pd.concat(
            [self._fetch_season_stats_via_rpc(int(season)) for season in all_seasons],
            ignore_index=True
        )
        all_recent_form_df = self._fetch_recent_form_via_rpc()
        
        _log_profile(all_season_stats_df, "all_season_stats (RPC)", debug)
        _log_profile(all_recent_form_df, "all_recent_form (RPC)", debug)
        
        # (Precompute logic remains the same)
        if "team_id" in hist_games.columns:
            logger.debug("ENGINE: precomputing advanced & drive metrics on historical")
            hist_games = compute_advanced_metrics(hist_games)
            hist_games = compute_drive_metrics(hist_games)

        # (The rest of the function remains the same...)
        chunks: List[pd.DataFrame] = []
        for season_str in all_seasons:
            season = int(season_str)
            logger.info("ENGINE: season %s start", season)
            season_games = games.loc[games["season"] == season_str].copy()
            cutoff = season_games["game_date"].min()
            past_games = hist_games.loc[hist_games["game_date"] < cutoff]

            for module in self.execution_order:
                fn = self.TRANSFORMS[module]
                logger.debug("ENGINE:%s running…", module)

                kw: Dict[str, Any] = {"flag_imputations": flag_imputations, "debug": debug}
                if _supports_kwarg(fn, "historical_df"): kw["historical_df"] = past_games
                
                if module == "season":
                    kw["season_stats_df"] = all_season_stats_df[all_season_stats_df['season'] == (season - 1)]
                elif module == "rolling":
                    kw["recent_form_df"] = all_recent_form_df
                    kw["window"] = rolling_window
                elif module == "form":
                    kw["lookback_window"] = form_lookback
                elif module == "momentum":
                    kw["span"] = momentum_span
                elif module == "h2h":
                    kw["max_games"] = h2h_max_games
                
                if _supports_kwarg(fn, "historical_team_stats_df"):
                    kw["historical_team_stats_df"] = hist_team_stats

                mod_out = fn(season_games, **kw)

                if mod_out.empty:
                    logger.warning("ENGINE:%s returned empty DF – skipped", module.upper())
                    continue
                
                season_games = _merge_and_log(season_games, mod_out, "game_id", "left", module, debug)

            chunks.append(season_games)

        if not chunks:
            logger.error("ENGINE: no season chunks produced; returning empty frame.")
            return pd.DataFrame()

        final = pd.concat(chunks, ignore_index=True)
        final.drop(columns=[c for c in final.columns if c.startswith("_")], inplace=True, errors="ignore")
        
        _log_profile(final, "FINAL_FEATURES", debug)
        logger.info("ENGINE: complete in %.2fs | final shape=%s", time.time() - t0, final.shape)
        return final
