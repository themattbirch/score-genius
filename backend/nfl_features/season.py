# backend/mlb_features/season.py
"""
Attaches previous-season statistical context for MLB home and away teams.
Assumes 'home_team_norm', 'away_team_norm', and 'season' columns exist on input.
"""
from __future__ import annotations
import logging
import time
from typing import Optional, Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("Could not import DEFAULTS; using local fallbacks in season.py")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]

def transform(
    df: pd.DataFrame,
    *,
    historical_team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Index-join based transform: attaches previous season stats only.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    start = time.time()
    logger.info("season: starting transform – input shape %s", df.shape)

    result = df.copy()
    required = {"game_id", "season", "home_team_norm", "away_team_norm"}
    if not required.issubset(result.columns):
        logger.error("season: missing columns %s – skipping", required - set(result.columns))
        return result

    # Map original stat column names to feature names
    feature_map = {
        "wins_all_percentage":   "prev_season_win_pct",
        "runs_for_avg_all":      "prev_season_avg_runs_for",
        "runs_against_avg_all":  "prev_season_avg_runs_against",
    }
    default_map = {
        "prev_season_win_pct":        MLB_DEFAULTS.get("win_pct", 0.5),
        "prev_season_avg_runs_for":   MLB_DEFAULTS.get("avg_runs_for", 4.5),
        "prev_season_avg_runs_against": MLB_DEFAULTS.get("avg_runs_against", 4.5),
    }

    # If no stats, fill defaults
    if historical_team_stats_df is None or historical_team_stats_df.empty:
        logger.warning("season: no historical stats; using defaults")
        for side in ("home", "away"):
            for feat, val in default_map.items():
                col = f"{side}_{feat}"
                result[col] = val
                if flag_imputations:
                    result[f"{col}_imputed"] = 1
        duration = time.time() - start
        logger.info("season: complete in %.2f s – output %s", duration, result.shape)
        return result

    # Prepare tiny lookup table (should only contain previous season rows)
    ts = (
        historical_team_stats_df
        .rename(columns=feature_map)
        .set_index("team_norm")[list(feature_map.values())]
    )

    # Join home and away sides by index
    res = (
        result
        .join(ts.add_prefix("home_"), on="home_team_norm")
        .join(ts.add_prefix("away_"), on="away_team_norm")
    )

    # Flag missing and fill defaults
    for side in ("home", "away"):
        for feat, dval in default_map.items():
            col = f"{side}_{feat}"
            if flag_imputations:
                res[f"{col}_imputed"] = res[col].isna().astype(int)
            res[col] = res[col].fillna(dval)

    # Derived differences
    res["prev_season_win_pct_diff"] = (
        res["home_prev_season_win_pct"] - res["away_prev_season_win_pct"]
    )
    res["prev_season_net_rating_diff"] = (
        (res["home_prev_season_avg_runs_for"] - res["home_prev_season_avg_runs_against"])
      - (res["away_prev_season_avg_runs_for"] - res["away_prev_season_avg_runs_against"])
    )

    duration = time.time() - start
    logger.info("season: complete in %.2f s – output %s", duration, res.shape)
    return res
