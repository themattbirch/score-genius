# backend/features/rolling.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
import pandas as pd

from .utils import DEFAULTS
from .base_windows import fetch_rolling
from backend.features.legacy.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)
__all__ = ["transform"]


@lru_cache(maxsize=1)
def _fe() -> FeatureEngine:
    """Singleton legacy engine for fallback."""
    return FeatureEngine(supabase_client=None, debug=False)


def _legacy_fallback(
    out: pd.DataFrame,
    window_sizes: list[int],
    reason: str,
    debug: bool = False
) -> pd.DataFrame:
    if debug:
        logger.warning("rolling.transform: falling back to legacy (reason=%s)", reason)
    try:
        return _fe().add_rolling_features(out, window_sizes=window_sizes)
    except Exception as e:
        logger.exception("rolling.transform: legacy fallback failed, returning input. %s", e)
        return out


def transform(
    df: pd.DataFrame,
    *,
    conn: Any | None = None,
    window_sizes: list[int] = [5, 10, 20],
    debug: bool = False
) -> pd.DataFrame:
    """
    Primary: fetch precomputed rolling stats via `fetch_rolling(conn, game_ids)`.
    Fallback: call legacy Python implementation.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("rolling.transform: input empty, returning unchanged.")
        return df

    # Ensure required keys
    for col in ("game_id", "home_team_norm", "away_team_norm"):
        if col not in df.columns:
            return _legacy_fallback(df.copy(deep=False), window_sizes, f"missing column '{col}'", debug)

    # No DB? skip straight to legacy
    if conn is None:
        return _legacy_fallback(df.copy(deep=False), window_sizes, "no DB connection", debug)

    try:
        out = df.copy(deep=False)
        out["game_id"] = out["game_id"].astype(str)
        ids = out["game_id"].unique().tolist()

        rolled = fetch_rolling(conn, ids)
        if rolled is None or rolled.empty:
            return _legacy_fallback(out, window_sizes, "SQL returned no rows", debug)

        # Prepare for merge
        rolled["game_id"] = rolled["game_id"].astype(str)
        rolled["team_norm"] = rolled["team_norm"].astype(str)

        # Identify stats columns
        stats_cols = [c for c in rolled.columns if c not in ("game_id", "team_norm", "game_date")]

        # Home merge
        home_map = {c: f"home_{c}" for c in stats_cols}
        home_df = rolled.rename(columns=home_map).drop(columns=["team_norm", "game_date"], errors="ignore")
        out = out.merge(
            home_df,
            left_on=["game_id", "home_team_norm"],
            right_on=["game_id", "team_norm"],
            how="left",
            suffixes=("", "_hdup")
        )

        # Away merge
        away_map = {c: f"away_{c}" for c in stats_cols}
        away_df = rolled.rename(columns=away_map).drop(columns=["team_norm", "game_date"], errors="ignore")
        out = out.merge(
            away_df,
            left_on=["game_id", "away_team_norm"],
            right_on=["game_id", "team_norm"],
            how="left",
            suffixes=("", "_adup")
        )

        # Clean up duplicates
        drop_cols = [c for c in out.columns if c.endswith("_hdup") or c.endswith("_adup") or c == "team_norm"]
        out = out.drop(columns=drop_cols, errors="ignore")

        # Fill NaNs with defaults
        new_cols = list(home_map.values()) + list(away_map.values())
        for col in new_cols:
            key = col.split("_", 1)[-1]  # strip prefix
            default = DEFAULTS.get(key, 0.0)
            out[col] = out[col].fillna(default)
            if key.endswith("_std"):
                out[col] = out[col].clip(lower=0.0)

        if debug:
            logger.debug("rolling.transform: SQL merge succeeded, added %d cols.", len(new_cols))

        return out

    except Exception as e:
        logger.exception("rolling.transform: SQL path error (%s), falling back.", e)
        return _legacy_fallback(df.copy(deep=False), window_sizes, f"sql error {e}", debug)
