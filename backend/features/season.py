# backend/features/season.py
from __future__ import annotations

import logging
from functools import lru_cache
import pandas as pd
from typing import Optional

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
    team_stats_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Thin proxy to legacy `add_season_context_features`.

    Parameters
    ----------
    df
        DataFrame of games to augment.
    team_stats_df
        Optional DataFrame of team-season stats for context features.

    Returns
    -------
    pd.DataFrame
        New DataFrame with season-context features added.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("season.transform: input empty, returning unchanged.")
        return df

    out = df.copy(deep=False)
    if debug:
        logger.debug("season.transform: invoking legacy add_season_context_features()")

    try:
        result = _fe().add_season_context_features(
            out,
            team_stats_df=team_stats_df
        )
        if debug:
            logger.debug("season.transform: done, result shape=%s", result.shape)
        return result

    except Exception as e:
        logger.exception("season.transform: legacy proxy failed, returning original DataFrame. %s", e)
        return out
