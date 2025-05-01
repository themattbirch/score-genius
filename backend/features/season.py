# backend/features/season.py
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
    team_stats: Optional[pd.DataFrame] = None,
    team_stats_df: Optional[pd.DataFrame] = None,  # ← legacy alias
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add season-to-date context features.

    Parameters
    ----------
    df : pd.DataFrame
        Games to augment.
    team_stats / team_stats_df : pd.DataFrame | None
        Season-aggregate per-team stats (optional).
    debug : bool
        Verbose logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with season-context features added.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("season.transform: input empty → returning unchanged.")
        return df

    # Resolve which stats DataFrame the caller supplied
    stats_df = team_stats if team_stats is not None else team_stats_df

    if debug:
        logger.debug(
            "season.transform: calling legacy add_season_context_features "
            "(stats rows = %s)", 0 if stats_df is None else len(stats_df)
        )

    try:
        result = _fe().add_season_context_features(df.copy(), stats_df)
        if debug:
            logger.debug("season.transform: success – output shape %s", result.shape)
        return result
    except Exception as e:
        logger.exception("season.transform: legacy proxy failed: %s", e)
        return df
