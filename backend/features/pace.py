# backend/features/pace.py
from __future__ import annotations
import logging
from functools import lru_cache
import pandas as pd

from .utils import DEFAULTS, convert_and_fill
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
    defaults: dict[str, float] = DEFAULTS,
) -> pd.DataFrame:
    """
    Proxy to legacy `integrate_advanced_features` for just the 'pace' stats.

    Ensures necessary raw columns exist, delegates to legacy, then
    returns only the pace-related features.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("pace.transform: empty input, returning unchanged.")
        return df

    out = df.copy(deep=False)

    # Ensure requisite raw columns are numeric
    needed = [
        "home_fg_attempted", "away_fg_attempted",
        "home_ft_attempted", "away_ft_attempted",
        "home_off_reb", "away_off_reb",
        "home_turnovers", "away_turnovers",
    ]
    out = convert_and_fill(out, needed, default=0.0)

    # Delegate to the legacy advanced-features method
    try:
        if debug:
            logger.debug("pace.transform: invoking legacy integrate_advanced_features.")
        legacy_df = _fe().integrate_advanced_features(out)
    except Exception as e:
        logger.error("pace.transform: legacy call failed, returning original df. %s", e, exc_info=True)
        return out

    # Select only the pace columns
    pace_cols = ["home_pace", "away_pace", "game_pace"]
    missing = [c for c in pace_cols if c not in legacy_df.columns]
    if missing:
        logger.warning("pace.transform: missing expected pace cols %s; returning what’s available.", missing)
    result = legacy_df[[c for c in pace_cols if c in legacy_df.columns]].copy()

    if debug:
        logger.debug("pace.transform: done, output shape=%s", result.shape)
    return result
