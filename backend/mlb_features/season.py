# backend/mlb_features/season.py
"""
Attaches previous-season statistical context for MLB home and away teams.
Assumes 'home_team_norm', 'away_team_norm', and 'season' columns exist on input.
"""
from __future__ import annotations
import logging
import time
from typing import Optional, Any, Dict
import numpy as np

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("Could not import DEFAULTS; using local fallbacks in season.py")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]

# (in backend/mlb_features/season.py)

def transform(
    df: pd.DataFrame,
    *,
    historical_team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
    **kwargs, # Accept and ignore other args
) -> pd.DataFrame:
    """
    Attaches previous-season statistical context for MLB home and away teams.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    start = time.time()
    logger.info("season: starting transform – input shape %s", df.shape)

    result = df.copy()
    
    # Define the columns we want to fetch and the names we want to give them
    feature_map = {
        "wins_all_percentage": "prev_season_win_pct",
        "runs_for_avg_all": "prev_season_avg_runs_for",
        "runs_against_avg_all": "prev_season_avg_runs_against",
    }
    default_map = {
        "prev_season_win_pct": MLB_DEFAULTS.get("win_pct", 0.5),
        "prev_season_avg_runs_for": MLB_DEFAULTS.get("avg_runs_for", 4.5),
        "prev_season_avg_runs_against": MLB_DEFAULTS.get("avg_runs_against", 4.5),
    }

    if historical_team_stats_df is None or historical_team_stats_df.empty:
        logger.warning("season: no historical stats; using defaults")
        for side in ("home", "away"):
            for feat, val in default_map.items():
                result[f"{side}_{feat}"] = val
                if flag_imputations:
                    result[f"{side}_{feat}_imputed"] = True
    else:
        # Prepare the historical stats lookup table
        stats_lookup = historical_team_stats_df.copy()
        if 'team_norm' not in stats_lookup.columns:
            from .utils import normalize_team_name
            stats_lookup['team_norm'] = stats_lookup['team_id'].apply(normalize_team_name)

        # Select and rename only the columns we need for the lookup
        cols_to_keep = ['team_norm', 'season'] + list(feature_map.keys())
        stats_lookup = stats_lookup[[col for col in cols_to_keep if col in stats_lookup.columns]].rename(columns=feature_map)

        # Create a previous_season key for merging
        result['prev_season'] = result['season'] - 1

        # Merge for home teams
        result = pd.merge(
            result,
            stats_lookup.add_prefix('home_'),
            left_on=['home_team_norm', 'prev_season'],
            right_on=['home_team_norm', 'home_season'],
            how='left'
        )

        # Merge for away teams
        result = pd.merge(
            result,
            stats_lookup.add_prefix('away_'),
            left_on=['away_team_norm', 'prev_season'],
            right_on=['away_team_norm', 'away_season'],
            how='left',
            suffixes=('', '_away_dup')
        )
        
        # Clean up columns from merge
        result = result.loc[:, ~result.columns.str.endswith('_away_dup')]
        result = result.drop(columns=['home_season', 'away_season'], errors='ignore')

        # Fill NaNs and create imputation flags
        for side in ("home", "away"):
            for feat, dval in default_map.items():
                col = f"{side}_{feat}"
                if col not in result.columns: result[col] = np.nan
                if flag_imputations:
                    result[f"{col}_imputed"] = result[col].isnull()
                result[col] = result[col].fillna(dval)

    # --- Final Derived Columns ---
    result["prev_season_win_pct_diff"] = result["home_prev_season_win_pct"] - result["away_prev_season_win_pct"]
    result["prev_season_net_rating_diff"] = (
        (result["home_prev_season_avg_runs_for"] - result["home_prev_season_avg_runs_against"]) -
        (result["away_prev_season_avg_runs_for"] - result["away_prev_season_avg_runs_against"])
    )
    
    logger.info("season: complete in %.2f s – output shape %s", time.time() - start, result.shape)
    return result