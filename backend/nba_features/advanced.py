# backend/nba_features/advanced.py
"""
Attaches pre-calculated historical/seasonal advanced statistical splits to game data.

This module takes an input DataFrame of games (which includes a column indicating
which season's advanced stats to look up for each game) and another DataFrame
containing a collection of pre-calculated seasonal advanced stats for multiple seasons.
It looks up the relevant prior season's home performance stats for the home team and
away performance stats for the away team, for each game, and attaches them.
Differentials between these historical stats are also calculated.
"""

from __future__ import annotations
import logging
from typing import Sequence # Removed Dict as it's not directly used in signature

import pandas as pd # numpy is used by pandas
import numpy as np

from .utils import normalize_team_name, DEFAULTS, profile_time 

logger = logging.getLogger(__name__)
__all__ = ["transform"]

EXPECTED_STATS: Sequence[str] = [
    "pace", "off_rtg", "def_rtg", "net_rtg", "efg_pct", "tov_pct", "oreb_pct", "ft_rate"
]

# _STAT_MAP is not used in the provided transform, can be removed if truly unused.
# _STAT_MAP = {
#     "pace":               "pace",
#     "off_rtg":            "offensive_rating",
#     # ...
# }

@profile_time
def transform(
    df: pd.DataFrame, # Input df, must now contain 'adv_stats_lookup_season'
    *,
    all_historical_splits_df: pd.DataFrame, # Concatenated DF of all relevant prior seasons' stats
                                          # Must contain 'season' (season stats pertain to),
                                          # 'team_name' or 'team_norm', and stat_home/stat_away cols.
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    if 'adv_stats_lookup_season' not in out.columns:
        logger.error("'adv_stats_lookup_season' column is missing from input df. Cannot perform advanced stat lookup.")
        return out # Or raise error

    # 1) Normalize team names in the main DataFrame
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)

    # 2) Prepare the historical seasonal splits DataFrame
    if all_historical_splits_df.empty:
        logger.warning("Received empty 'all_historical_splits_df'. Advanced features will be NaNs or defaults.")
        # Create empty columns to prevent key errors later
        for side, prefix in (("home", "h"), ("away", "a")):
            for stat in EXPECTED_STATS:
                tgt_col = f"{prefix}_{stat}_{side}"
                out[tgt_col] = pd.NA
                if flag_imputations:
                    out[f"{tgt_col}_imputed"] = True # Mark all as imputed
    else:
        seasonal_stats_processed = all_historical_splits_df.copy()
        if "team_norm" not in seasonal_stats_processed.columns and "team_name" in seasonal_stats_processed.columns:
            seasonal_stats_processed["team_norm"] = seasonal_stats_processed["team_name"].map(normalize_team_name)
        elif "team_norm" not in seasonal_stats_processed.columns:
            logger.error("'team_norm' or 'team_name' column missing in 'all_historical_splits_df'.")
            # Handle error or return 'out' with NaNs for these features
            return out # Or fill with NaNs as above

        # Ensure 'season' column (season the stats pertain to) exists for merging
        if "season" not in seasonal_stats_processed.columns:
            logger.error("'season' column (identifying the season of the stats) missing in 'all_historical_splits_df'.")
            return out

        # 3) Merge stats for home teams
        home_stats_to_join = seasonal_stats_processed.copy()
        home_rename_mapping = {}
        for stat in EXPECTED_STATS:
            home_rename_mapping[f"{stat}_home"] = f"h_{stat}_home" # e.g., pace_home -> h_pace_home
        home_stats_to_join = home_stats_to_join.rename(columns=home_rename_mapping)
        
        relevant_home_cols = ['team_norm', 'season'] + [f"h_{stat}_home" for stat in EXPECTED_STATS]
        home_stats_to_join = home_stats_to_join[home_stats_to_join.columns.intersection(relevant_home_cols)]

        out = pd.merge(
            out,
            home_stats_to_join,
            left_on=['home_norm', 'adv_stats_lookup_season'],
            right_on=['team_norm', 'season'], # 'season' in home_stats_to_join is the season stats are FOR
            how='left',
            suffixes=('', '_drop_home') # Suffix for duplicated columns from right if any
        )
        cols_to_drop_home = [col for col in out.columns if col.endswith('_drop_home')]
        cols_to_drop_home.extend(['team_norm_drop_home', 'season_drop_home']) # Explicitly add if they exist with this exact suffix
        out.drop(columns=list(set(cols_to_drop_home).intersection(out.columns)), inplace=True, errors='ignore')


        # 4) Merge stats for away teams
        away_stats_to_join = seasonal_stats_processed.copy()
        away_rename_mapping = {}
        for stat in EXPECTED_STATS:
            away_rename_mapping[f"{stat}_away"] = f"a_{stat}_away" # e.g., pace_away -> a_pace_away
        away_stats_to_join = away_stats_to_join.rename(columns=away_rename_mapping)

        relevant_away_cols = ['team_norm', 'season'] + [f"a_{stat}_away" for stat in EXPECTED_STATS]
        away_stats_to_join = away_stats_to_join[away_stats_to_join.columns.intersection(relevant_away_cols)]

        out = pd.merge(
            out,
            away_stats_to_join,
            left_on=['away_norm', 'adv_stats_lookup_season'],
            right_on=['team_norm', 'season'],
            how='left',
            suffixes=('', '_drop_away')
        )
        cols_to_drop_away = [col for col in out.columns if col.endswith('_drop_away')]
        cols_to_drop_away.extend(['team_norm_drop_away', 'season_drop_away'])
        out.drop(columns=list(set(cols_to_drop_away).intersection(out.columns)), inplace=True, errors='ignore')

    # 5) Imputation for all generated h_... and a_... columns
    for side, prefix in (("home", "h"), ("away", "a")):
        for stat in EXPECTED_STATS:
            # Determine the correct target column name based on merge results
            if side == "home":
                tgt_col = f"h_{stat}_home"
            else: # away
                tgt_col = f"a_{stat}_away"

            impute_col = f"{tgt_col}_imputed"
            default = DEFAULTS.get(stat, 0.0)

            if tgt_col not in out.columns: # If merge failed to create the column (e.g. no relevant stats)
                out[tgt_col] = pd.NA    # Ensure column exists before imputation

            if flag_imputations:
                out[impute_col] = out[tgt_col].isna()
            
            out[tgt_col] = pd.to_numeric(out[tgt_col], errors='coerce').fillna(default).astype(float)
            
            if flag_imputations:
                # Ensure impute_col also exists if tgt_col was just created as all NA
                if impute_col not in out.columns:
                     out[impute_col] = True # If tgt_col was all NA and then filled, it was all imputed
                out[impute_col] = out[impute_col].astype(bool)


    # 6) Compute split diffs (using the now correctly sourced h_... and a_... columns)
    for stat in EXPECTED_STATS:
        home_col_name = f"h_{stat}_home"
        away_col_name = f"a_{stat}_away"
        # Ensure columns exist before trying to subtract
        if home_col_name in out.columns and away_col_name in out.columns:
            out[f"hist_{stat}_split_diff"] = out[home_col_name] - out[away_col_name]
        else:
            out[f"hist_{stat}_split_diff"] = np.nan


    # 7) Mirror the specific rating columns into the names rolling expects
    # Ensure source columns exist before assignment
    ratings_map = {
        "home_offensive_rating": "h_off_rtg_home",
        "away_offensive_rating": "a_off_rtg_away",
        "home_defensive_rating": "h_def_rtg_home",
        "away_defensive_rating": "a_def_rtg_away",
        "home_net_rating": "h_net_rtg_home",
        "away_net_rating": "a_net_rtg_away",
    }
    for target_rating_col, source_adv_col in ratings_map.items():
        if source_adv_col in out.columns:
            out[target_rating_col] = out[source_adv_col]
        else:
            out[target_rating_col] = np.nan


    # 8) Cleanup
    columns_to_drop_final = ["home_norm", "away_norm"]
    # The 'adv_stats_lookup_season' column was added by engine.py.
    # If it's not needed downstream from advanced.py, it can be dropped here too.
    # For now, let's assume it might be useful for debugging or later stages,
    # or it can be dropped by engine.py after all modules run.
    # If you want advanced.py to explicitly drop it:
    # columns_to_drop_final.append('adv_stats_lookup_season')

    out.drop(columns=[col for col in columns_to_drop_final if col in out.columns], inplace=True, errors='ignore')
    
    return out