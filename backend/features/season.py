# backend/features/season.py

from __future__ import annotations
import pandas as pd
import logging
from typing import Optional

from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Instantiate legacy engine once for fallback
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("Season: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("Season: Failed to instantiate Legacy engine.")
    _legacy_engine = None

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    team_stats: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add season‐context features (win_pct, avg_pts_for/against, current_form, net_rating_diff, etc.)
    using the legacy engine under the hood.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Season.transform: received empty DataFrame, returning as-is.")
        return df

    if _legacy_engine is None:
        logger.error("Season.transform: legacy engine unavailable, returning original DataFrame.")
        return df

    try:
        if debug:
            logger.debug("Season.transform: invoking legacy add_season_context_features.")
        # Note: legacy expects (df, team_stats_df)
        result = _legacy_engine.add_season_context_features(
            df.copy(),
            team_stats_df=team_stats
        )
        if debug:
            logger.debug(f"Season.transform: result shape = {result.shape}")
        return result
    except Exception as e:
        logger.exception(f"Season.transform: fallback error: {e}")
        return df
