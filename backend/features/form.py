# backend/features/form.py

from __future__ import annotations
import pandas as pd
import logging
from typing import Optional

from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Instantiate legacy engine once for fallback
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("Form: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("Form: Failed to instantiate Legacy engine.")
    _legacy_engine = None

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Add form-string derived features (home/away form metrics, diffs)
    using the legacy engine under the hood.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Form.transform: received empty DataFrame, returning as-is.")
        return df

    if _legacy_engine is None:
        logger.error("Form.transform: legacy engine unavailable, returning original DataFrame.")
        return df

    try:
        if debug:
            logger.debug("Form.transform: invoking legacy add_form_string_features.")
        result = _legacy_engine.add_form_string_features(df.copy())
        if debug:
            logger.debug(f"Form.transform: result shape = {result.shape}")
        return result
    except Exception as e:
        logger.exception(f"Form.transform: fallback error: {e}")
        return df