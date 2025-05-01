# backend/features/h2h.py

from __future__ import annotations
import pandas as pd
import logging
from typing import Optional

from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Instantiate legacy engine once for fallback
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("H2H: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("H2H: Failed to instantiate Legacy engine.")
    _legacy_engine = None

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    historical: Optional[pd.DataFrame] = None,
    max_games: int = 7,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add head-to-head matchup features using the legacy engine.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("H2H.transform: received empty DataFrame, returning unchanged.")
        return df

    if _legacy_engine is None:
        logger.error("H2H.transform: legacy engine unavailable, returning original DataFrame.")
        return df

    try:
        if debug:
            logger.debug(f"H2H.transform: running legacy H2H with max_games={max_games}.")
        result = _legacy_engine.add_matchup_history_features(
            df.copy(),
            historical_df=historical,
            max_games=max_games
        )
        if debug:
            logger.debug(f"H2H.transform: result shape = {result.shape}")
        return result
    except Exception as e:
        logger.exception(f"H2H.transform: fallback error: {e}")
        return df
