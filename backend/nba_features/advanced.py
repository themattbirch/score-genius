# backend/nba_features/advanced.py
"""
Attaches pre-calculated historical/seasonal advanced statistical splits to game data.

This module takes an input DataFrame of games and another DataFrame containing
pre-calculated seasonal advanced stats (with home/away splits, often from an RPC
or a processed data source like 'nba_team_seasonal_advanced_splits').
It looks up the relevant season's home performance stats for the home team and
away performance stats for the away team, and attaches them to each game.
Differentials between these historical stats are also calculated.
"""

from __future__ import annotations
import logging
from typing import Mapping, Sequence, Dict # Added Dict

import numpy as np # Not strictly needed here now, but pandas uses it
import pandas as pd

# Assuming profile_time is for performance measurement, keep if used
from .utils import normalize_team_name, DEFAULTS, profile_time 

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Expected base advanced stat names (these should align with keys in DEFAULTS
# and the base names in the columns of stats_df, e.g., 'pace' for 'pace_home')
EXPECTED_STATS: Sequence[str] = [
    "pace", "off_rtg", "def_rtg", "net_rtg", "efg_pct", "tov_pct", "oreb_pct", "ft_rate"
]

@profile_time # If you want to profile this function
def transform(
    df: pd.DataFrame,
    *,
    stats_df: pd.DataFrame, # DataFrame from 'nba_team_seasonal_advanced_splits' table
    season: str | None = None,            # The season year for which stats_df is filtered/relevant
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Attaches seasonal advanced splits (home vs. away from stats_df for the given season) 
    to the input df and computes differentials.

    Args:
        df: Games DataFrame with ['game_id','home_team','away_team'].
        stats_df: DataFrame containing pre-calculated seasonal advanced stats. Expected
                  columns include 'team_name' (or 'team_norm'), 'season', and split stats
                  like 'pace_home', 'off_rtg_away', etc., for the specified 'season'.
        season: The specific season year (e.g., 2022 for 2022-23) for which the stats
                in `stats_df` should be used. This function will filter `stats_df` by this season.
        flag_imputations: If True, add boolean '_imputed' flags for attached stats.
        debug: If True, enable DEBUG logging for this function call.
    """
    original_logger_level = logger.level # Renamed to avoid conflict
    if season is None and "game_date" in stats_df:
        from .utils import determine_season
        season = determine_season(pd.to_datetime(stats_df["game_date"].iloc[0]))
    
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"historical_advanced_splits.transform: Attaching stats for season={season}")

    if df.empty:
        logger.warning("historical_advanced_splits.transform: Input 'df' is empty. Returning copy.")
        if debug: logger.setLevel(original_logger_level)
        return df.copy()
    
    if stats_df is None or stats_df.empty: # Added check for stats_df being None
        logger.warning("historical_advanced_splits.transform: Input 'stats_df' is empty. Cannot attach stats. Returning copy of df.")
        # Optionally, could add placeholder columns with NaNs or defaults to df here if a strict output schema is always needed.
        if debug: logger.setLevel(original_logger_level)
        return df.copy()

    out_df = df.copy() # Renamed to avoid confusion with 'df' parameter
    out_df['home_norm'] = out_df['home_team'].astype(str).map(normalize_team_name)
    out_df['away_norm'] = out_df['away_team'].astype(str).map(normalize_team_name)

    # Prepare and filter stats_df for the specified season
    # Ensure 'season' column exists for filtering
    if 'season' not in stats_df.columns:
        logger.error("historical_advanced_splits.transform: 'stats_df' is missing 'season' column. Cannot proceed.")
        if debug: logger.setLevel(original_logger_level)
        return out_df # Return df as is, without new features
        
    # Ensure 'team_name' or 'team_norm' exists for normalization
    if 'team_norm' not in stats_df.columns and 'team_name' not in stats_df.columns:
        logger.error("historical_advanced_splits.transform: 'stats_df' must have 'team_name' or 'team_norm'. Cannot proceed.")
        if debug: logger.setLevel(original_logger_level)
        return out_df

    # Use a copy for modifications
    seasonal_stats_filtered = stats_df[stats_df['season'] == season].copy()
    if 'team_norm' not in seasonal_stats_filtered.columns: # Ensure team_norm exists for merging
        seasonal_stats_filtered['team_norm'] = seasonal_stats_filtered['team_name'].astype(str).map(normalize_team_name)
    
    if seasonal_stats_filtered.empty:
        logger.warning(f"historical_advanced_splits.transform: No data in 'stats_df' for season {season}. Features will be defaults.")
        # Proceeding will result in NaNs for all lookups, which will then be default-filled.

    # Merge home and away splits
    for side_type in ('home', 'away'): # 'home' performance for home_team, 'away' performance for away_team
        # Determine the prefix for new columns ('h_' for home team's stats, 'a_' for away team's stats)
        output_col_prefix = 'h' if side_type == 'home' else 'a'
        # Determine the team column from out_df to join on
        join_on_team_col = 'home_norm' if side_type == 'home' else 'away_norm'
        
        # Define how to rename columns from seasonal_stats_filtered for this side
        # Source columns in seasonal_stats_filtered are like 'pace_home', 'off_rtg_home', 'pace_away', 'off_rtg_away'
        # Target columns in out_df are like 'h_pace_home', 'a_pace_away'
        rename_map_for_merge: Dict[str,str] = {}
        cols_to_select_from_seasonal = ['team_norm'] # Start with the join key

        for stat_base_name in EXPECTED_STATS:
            # Source column from seasonal_stats_filtered (e.g., 'pace_home' if side_type is 'home')
            source_col = f"{stat_base_name}_{side_type}" 
            # Target column in out_df (e.g., 'h_pace_home' if side_type is 'home' and output_col_prefix is 'h')
            target_col = f"{output_col_prefix}_{stat_base_name}_{side_type}" 
            
            if source_col in seasonal_stats_filtered.columns:
                rename_map_for_merge[source_col] = target_col
                cols_to_select_from_seasonal.append(source_col)
            else:
                logger.warning(f"Expected column '{source_col}' not found in stats_df for season {season}.")
        
        if len(cols_to_select_from_seasonal) > 1: # If there are actual stats to merge (beyond just team_norm)
            temp_df_to_merge = seasonal_stats_filtered[cols_to_select_from_seasonal].rename(columns=rename_map_for_merge)
            
            out_df = out_df.merge(
                temp_df_to_merge,
                left_on=join_on_team_col,
                right_on='team_norm', # team_norm from temp_df_to_merge
                how='left',
                suffixes=('', '_drop') # Suffix for the 'team_norm' from right if it conflicts
            )
            if 'team_norm_drop' in out_df.columns:
                out_df = out_df.drop(columns=['team_norm_drop'])
        else:
            logger.warning(f"No source columns found in stats_df to merge for side '{side_type}' and stats {EXPECTED_STATS}")


    # Impute defaults and set flags for all newly created columns
    for stat_base_name in EXPECTED_STATS:
        for side_type, output_col_prefix in (('home', 'h'), ('away', 'a')):
            # This is the column name in out_df, e.g. h_pace_home, a_off_rtg_away
            final_col_name = f"{output_col_prefix}_{stat_base_name}_{side_type}"
            
            # Use the base stat name (e.g., "pace", "off_rtg") for default lookup
            default_value = DEFAULTS.get(stat_base_name, 0.0) 

            if final_col_name not in out_df.columns: # If merge didn't create the column
                out_df[final_col_name] = default_value # Create with default
                if flag_imputations:
                    out_df[f"{final_col_name}_imputed"] = True # Fully imputed
            else:
                if flag_imputations:
                    out_df[f"{final_col_name}_imputed"] = out_df[final_col_name].isnull()
                out_df[final_col_name] = pd.to_numeric(out_df[final_col_name], errors='coerce').fillna(default_value)
            
            # Ensure types
            out_df[final_col_name] = out_df[final_col_name].astype(float)
            if flag_imputations and f"{final_col_name}_imputed" in out_df.columns:
                 out_df[f"{final_col_name}_imputed"] = out_df[f"{final_col_name}_imputed"].astype(bool)


    # Compute differentials
    logger.debug("Calculating differentials for historical seasonal splits...")
    for stat_base_name in EXPECTED_STATS:
        # Home team's historical home stat vs Away team's historical away stat
        home_team_stat_col = f"h_{stat_base_name}_home" 
        away_team_stat_col = f"a_{stat_base_name}_away"
        diff_col_name = f"hist_{stat_base_name}_split_diff" # More descriptive diff name

        if home_team_stat_col in out_df.columns and away_team_stat_col in out_df.columns:
            out_df[diff_col_name] = (out_df[home_team_stat_col] - out_df[away_team_stat_col]).astype(float)
        else:
            logger.warning(f"Cannot calculate {diff_col_name}, missing components: {home_team_stat_col} or {away_team_stat_col}.")
            out_df[diff_col_name] = 0.0 # Default diff to 0.0

    # Drop helper columns
    helper_cols = ['home_norm', 'away_norm']
    # Drop team_norm if it was added by merge and not original, or if only one was kept after merge
    if 'team_norm' in out_df.columns and 'team_norm' not in df.columns: 
        helper_cols.append('team_norm')
    out_df = out_df.drop(columns=[c for c in helper_cols if c in out_df.columns], errors='ignore')
    
    if debug:
        logger.setLevel(original_logger_level)
    logger.debug(f"historical_advanced_splits.transform completed. Output df shape: {out_df.shape}")
    return out_df