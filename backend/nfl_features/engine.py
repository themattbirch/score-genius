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
    """Runs the full NFL feature-engineering pipeline."""

    DEFAULT_ORDER: List[str] = [
        "situational",
        "season",
        "rest",
        "rolling",
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

        # tracker for unit tests / stubs (optional)
        self._ext_tracker: Optional[List[Dict[str, Any]]] = None
        for fn in self.TRANSFORMS.values():
            clo = getattr(fn, "__closure__", None)
            if clo:
                for cell in clo:
                    if isinstance(cell.cell_contents, list):
                        self._ext_tracker = cell.cell_contents
                        break
            if self._ext_tracker is not None:
                break

    # ------------------------------------------------------------------ #
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

        # Ensure tracker after init
        if self._ext_tracker is None:
            for fn in self.TRANSFORMS.values():
                clo = getattr(fn, "__closure__", None)
                if clo:
                    for cell in clo:
                        if isinstance(cell.cell_contents, list):
                            self._ext_tracker = cell.cell_contents
                            break
                if self._ext_tracker is not None:
                    break

        def _canon(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "home_team_norm" in out.columns:
                out["home_team_norm"] = out["home_team_norm"].apply(normalize_team_name)
            if "away_team_norm" in out.columns:
                out["away_team_norm"] = out["away_team_norm"].apply(normalize_team_name)
            if "game_date" in out.columns:
                out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
                out["season"] = out["game_date"].apply(determine_season)
            return out

        games = _canon(games_df)
        hist_games = _canon(historical_games_df)
        hist_team_stats = historical_team_stats_df.copy()

        _log_profile(games, "games_df (canon)", debug)
        _log_profile(hist_games, "historical_games_df (canon)", debug)
        _log_profile(hist_team_stats, "historical_team_stats_df", debug)

        # 1. Precompute on historical only (advanced + drive)
        if "team_id" in hist_games.columns:
            logger.debug("ENGINE: precomputing advanced & drive metrics on historical")
            t_adv = time.time()
            hist_games = compute_advanced_metrics(hist_games)
            hist_games = compute_drive_metrics(hist_games)
            logger.debug("ENGINE: precompute done in %.2fs | hist_games shape=%s",
                         time.time() - t_adv, hist_games.shape)
        else:
            logger.debug("ENGINE: hist_games lacks team_id – skipping advanced/drive precompute")

        # 2. Iterate seasons to avoid leakage
        chunks: List[pd.DataFrame] = []

        for season in sorted(games["season"].dropna().unique()):
            logger.info("ENGINE: season %s start", season)
            season_games = games.loc[games["season"] == season].copy()
            cutoff = season_games["game_date"].min()
            past_games = hist_games.loc[hist_games["game_date"] < cutoff]

            _log_profile(season_games, f"season_{season}_initial", debug)

            for module in self.execution_order:
                fn = self.TRANSFORMS[module]
                if self._ext_tracker is not None:
                    pre_len = len(self._ext_tracker)

                logger.debug("ENGINE:%s running…", module)

                # Build kwargs dynamically
                kw: Dict[str, Any] = {"flag_imputations": flag_imputations}
                if _supports_kwarg(fn, "historical_df"):
                    kw["historical_df"] = past_games
                if _supports_kwarg(fn, "historical_team_stats_df"):
                    kw["historical_team_stats_df"] = hist_team_stats
                if module == "rolling":
                    kw.update({
                        "game_ids": season_games["game_id"].tolist(),
                        "supabase": self.supabase,
                        "window": rolling_window,
                    })
                elif module == "form":
                    kw["lookback_window"] = form_lookback
                elif module == "momentum":
                    kw["span"] = momentum_span
                elif module == "h2h":
                    kw["max_games"] = h2h_max_games

                m_start = time.time()
                if module == "rolling":
                    mod_out = fn(**kw)
                else:
                    mod_out = fn(season_games, **kw)
                m_time = time.time() - m_start

                if self._ext_tracker is not None and len(self._ext_tracker) == pre_len:
                    self._ext_tracker.append({"name": module})

                if mod_out.empty:
                    logger.warning("ENGINE:%s returned empty DF – skipped", module.upper())
                    continue

                # Merge & diagnostics
                before_shape = season_games.shape
                if module == "rolling":
                    season_games = _merge_and_log(season_games, mod_out, "game_id", "left", module, debug)
                else:
                    merge_cols = [c for c in mod_out.columns if c != "game_id"]
                    season_games = _merge_and_log(season_games, mod_out[["game_id"] + merge_cols],
                                                  "game_id", "left", module, debug)
                after_shape = season_games.shape

                logger.debug("ENGINE:%s took %.3fs | shape %s → %s",
                             module, m_time, before_shape, after_shape)

            _log_profile(season_games, f"season_{season}_final", debug)
            chunks.append(season_games)

        if not chunks:
            logger.error("ENGINE: no season chunks produced; returning empty frame.")
            return pd.DataFrame()

        final = pd.concat(chunks, ignore_index=True)

        # Drop helper cols
        helper_cols = [c for c in final.columns if c.startswith("_")]
        if helper_cols and debug:
            logger.debug("ENGINE: dropping helper cols: %s", helper_cols)
        final.drop(columns=helper_cols, inplace=True, errors="ignore")

        _log_profile(final, "FINAL_FEATURES", debug)

        logger.info("ENGINE: complete in %.2fs | final shape=%s", time.time() - t0, final.shape)
        return final
