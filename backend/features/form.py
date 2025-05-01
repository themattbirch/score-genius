# backend/features/form.py

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
    Thin proxy for legacy `add_form_string_features`.
    Returns a **new** DataFrame with form features added.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Form.transform: input empty, returning unchanged.")
        return df

    out = df.copy()

    if debug:
        logger.debug("Form.transform: invoking legacy add_form_string_features.")

    try:
        result = _fe().add_form_string_features(out)
        if debug:
            logger.debug("Form.transform: done, output shape = %s", result.shape)
        return result
    except Exception as e:
        logger.exception("Form.transform: legacy proxy error: %s", e)
        # on failure, return original copy
        return out
