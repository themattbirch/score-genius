# backend/features/h2h.py

from __future__ import annotations
import logging
from functools import lru_cache

import pandas as pd

from backend.features.legacy.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)
__all__ = ["transform"]


@lru_cache(maxsize=1)
def _fe() -> FeatureEngine:
    """Singleton legacy engine (no Supabase, no debug)."""
    return FeatureEngine(supabase_client=None, debug=False)


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    historical_df: pd.DataFrame | None = None,
    window: int = 7,
) -> pd.DataFrame:
    """
    Thin proxy for legacy add_matchup_history_features.

    Parameters
    ----------
    df
        DataFrame of target games.
    historical_df
        Historical games DataFrame (head-to-head source).
    window
        Lookback window (number of games) for h2h features.
    debug
        If True, emit debug logs.

    Returns
    -------
    DataFrame
        New DataFrame with head-to-head features added.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("h2h.transform: input empty, returning unchanged.")
        return df

    out = df.copy()
    try:
        if debug:
            logger.debug(
                "h2h.transform: invoking legacy add_matchup_history_features "
                "(window=%d)", window
            )
        result = _fe().add_matchup_history_features(
            out,
            historical_df if historical_df is not None else pd.DataFrame(),
            max_games=window
        )
        if debug:
            logger.debug("h2h.transform: done, output shape = %s", result.shape)
        return result
    except Exception as e:
        logger.exception("h2h.transform: legacy proxy error: %s", e)
        return out
