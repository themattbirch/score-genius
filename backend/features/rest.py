# backend/features/rest.py

from __future__ import annotations
import pandas as pd
import logging
from typing import Any

# Import the legacy FeatureEngine for fallback
from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Instantiate a single legacy engine at module load for reuse
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("Rest: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("Rest: Failed to instantiate LegacyFeatureEngine.")
    _legacy_engine = None

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Add rest days, games in last N days, back-to-back flags, and schedule advantages.
    Uses the legacy Python method add_rest_features_vectorized under the hood.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Rest.transform: received empty DataFrame, returning as-is.")
        return df

    if _legacy_engine is None:
        logger.error("Rest.transform: legacy engine unavailable, returning original DataFrame.")
        return df

    if debug:
        logger.debug("Rest.transform: adding rest features via legacy engine.")

    try:
        # Work on a copy to avoid modifying the original
        result = _legacy_engine.add_rest_features_vectorized(df.copy())
        if debug:
            logger.debug(f"Rest.transform: resulting DataFrame shape {result.shape}")
        return result
    except Exception as e:
        logger.exception(f"Rest.transform: error in legacy fallback: {e}")
        return df
