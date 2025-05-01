# backend/features/advanced.py

from __future__ import annotations
import pandas as pd
import logging

from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Instantiate legacy engine once for reuse/fallback
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("Advanced: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("Advanced: Failed to instantiate Legacy engine.")
    _legacy_engine = None

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Add advanced metrics (eFG%, FT rate, rebounds, pace, ratings, TOV%)
    by delegating to the legacy integrate_advanced_features method.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Advanced.transform: received empty DataFrame, returning as-is.")
        return df

    if _legacy_engine is None:
        logger.error("Advanced.transform: legacy engine unavailable, returning original DataFrame.")
        return df

    try:
        if debug:
            logger.debug("Advanced.transform: invoking legacy integrate_advanced_features.")
        result = _legacy_engine.integrate_advanced_features(df.copy())
        if debug:
            logger.debug(f"Advanced.transform: result shape = {result.shape}")
        return result
    except Exception as e:
        logger.exception(f"Advanced.transform: fallback error: {e}")
        return df