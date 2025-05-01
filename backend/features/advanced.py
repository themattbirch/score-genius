# backend/features/advanced.py

from __future__ import annotations
import logging
from functools import lru_cache
import pandas as pd

from backend.features.legacy.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)
__all__ = ["transform"]


@lru_cache(maxsize=1)
def _fe() -> FeatureEngine:
    """Singleton legacy FeatureEngine (no Supabase, no debug)."""
    return FeatureEngine(supabase_client=None, debug=False)


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Thin proxy to legacy `integrate_advanced_features`.
    Adds advanced metrics (eFG%, FT rate, rebounds, pace, ratings, TOV%).
    """
    if df is None or df.empty:
        if debug:
            logger.debug("advanced.transform: empty input, returning unchanged.")
        return df

    if debug:
        logger.debug("advanced.transform: calling legacy integrate_advanced_features")
    out = _fe().integrate_advanced_features(df.copy())

    if debug:
        logger.debug("advanced.transform: done, shape=%s", out.shape)
    return out
