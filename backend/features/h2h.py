# backend/features/h2h.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

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
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int | None = None,
    window: int | None = None,            # ← kept for backward-compat
    debug: bool = False,
) -> pd.DataFrame:
    """
    Thin proxy around FeatureEngine.add_matchup_history_features().

    Parameters
    ----------
    df : pd.DataFrame
        Target games.
    historical_df : pd.DataFrame | None
        Source games for head-to-head look-ups.
    max_games / window : int
        Look-back depth; `max_games` is preferred, `window` is legacy.
    debug : bool
        Verbose logging toggle.

    Returns
    -------
    pd.DataFrame
        `df` plus matchup-history features.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("h2h.transform: input empty → returning unchanged.")
        return df

    lookback = max_games if max_games is not None else window or 7

    if debug:
        logger.debug(
            "h2h.transform: calling legacy add_matchup_history_features "
            "(max_games=%d)", lookback
        )

    try:
        result = _fe().add_matchup_history_features(
            df.copy(),
            historical_df if historical_df is not None else pd.DataFrame(),
            max_games=lookback,
        )
        if debug:
            logger.debug("h2h.transform: success – output shape %s", result.shape)
        return result
    except Exception as e:
        logger.exception("h2h.transform: legacy proxy error: %s", e)
        return df
