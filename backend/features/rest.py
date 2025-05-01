# backend/features/rest.py
from __future__ import annotations
import logging
from functools import lru_cache
import pandas as pd

from backend.features.legacy.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)
__all__ = ["transform"]


@lru_cache(maxsize=1)
def _fe() -> FeatureEngine:
    """Singleton legacy engine (no DB, no debug)."""
    return FeatureEngine(supabase_client=None, debug=False)


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Thin proxy to legacy `add_rest_features_vectorized`.
    Returns a new DataFrame with rest & schedule features added.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("rest.transform: input empty, returning unchanged.")
        return df

    out = df.copy(deep=False)
    try:
        if debug:
            logger.debug("rest.transform: invoking legacy add_rest_features_vectorized.")
        result = _fe().add_rest_features_vectorized(out)
        if debug:
            logger.debug("rest.transform: done, output shape=%s", result.shape)
        return result
    except Exception as e:
        logger.error("rest.transform: legacy call failed, returning original df. %s", e, exc_info=True)
        return out
