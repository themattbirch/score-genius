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
from typing import Sequence

import pandas as pd
import numpy as np

from .utils import normalize_team_name, DEFAULTS, profile_time

logger = logging.getLogger(__name__)
__all__ = ["transform"]

EXPECTED_STATS: Sequence[str] = [
    "pace", "off_rtg", "def_rtg", "net_rtg", "efg_pct", "tov_pct", "oreb_pct", "ft_rate"
]

@profile_time
def transform(
    df: pd.DataFrame, # Input df, must now contain 'adv_stats_lookup_season'
    *,
    all_historical_splits_df: pd.DataFrame, # Concatenated DF of all relevant prior seasons' stats
                                          # Must contain 'season' (season stats pertain to),
                                          # 'team_name' or 'team_norm', and stat_home/stat_away cols.
    flag_imputations: bool = True,
    debug: bool = False, # debug parameter is present but not explicitly used in this version
) -> pd.DataFrame:
    out = df.copy()

    if 'adv_stats_lookup_season' not in out.columns:
        logger.error("'adv_stats_lookup_season' column is missing from input df. Cannot perform advanced stat lookup.")
        # Consider raising an error or returning 'out' as is, based on desired behavior
        # For now, returning 'out' to prevent downstream crashes if this module is optional.
        return out

    # 1) Normalize team names in the main DataFrame
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)

    # 2) Prepare the historical seasonal splits DataFrame
    if all_historical_splits_df.empty:
        logger.warning("Received empty 'all_historical_splits_df'. Advanced features will be NaNs or defaults.")
        # Create placeholder columns for all expected stats to ensure downstream consistency
        for side_label, team_prefix in (("home", "h"), ("away", "a")):
            for stat in EXPECTED_STATS:
                # Standard column names like h_pace_home, a_efg_pct_away
                tgt_col = f"{team_prefix}_{stat}_{side_label}"
                out[tgt_col] = pd.NA # These will be filled by default values in the imputation step
                if flag_imputations:
                    out[f"{tgt_col}_imputed"] = True # Mark all as imputed
    else:
        seasonal_stats_processed = all_historical_splits_df.copy()
        if "team_norm" not in seasonal_stats_processed.columns and "team_name" in seasonal_stats_processed.columns:
            seasonal_stats_processed["team_norm"] = seasonal_stats_processed["team_name"].map(normalize_team_name)
        elif "team_norm" not in seasonal_stats_processed.columns:
            logger.error("'team_norm' or 'team_name' column missing in 'all_historical_splits_df'. Cannot proceed with merge.")
            return out # Or fill with NaNs/defaults as in the empty case

        if "season" not in seasonal_stats_processed.columns:
            logger.error("'season' column (identifying the season of the stats) missing in 'all_historical_splits_df'. Cannot proceed with merge.")
            return out # Or fill with NaNs/defaults

        # 3) Merge stats for home teams
        home_stats_to_join = seasonal_stats_processed.copy()
        home_rename_mapping = {f"{stat}_home": f"h_{stat}_home" for stat in EXPECTED_STATS}
        home_stats_to_join = home_stats_to_join.rename(columns=home_rename_mapping)
        
        # Select only relevant columns for merging (keys + desired stats)
        relevant_home_cols = ['team_norm', 'season'] + [f"h_{stat}_home" for stat in EXPECTED_STATS]
        # Use intersection to robustly handle cases where some expected stats might not be in home_stats_to_join
        home_stats_to_join = home_stats_to_join[list(set(home_stats_to_join.columns).intersection(relevant_home_cols))]

        out = pd.merge(
            out,
            home_stats_to_join,
            left_on=['home_norm', 'adv_stats_lookup_season'],
            right_on=['team_norm', 'season'], # 'season' in home_stats_to_join is the season stats are FOR
            how='left',
            suffixes=('', '_home_overlap') # Suffix for other overlapping columns from the right DataFrame
        )
        # Drop helper columns from merge:
        # 1. Columns created due to overlap (suffixed)
        # 2. Key columns from the right DataFrame ('team_norm', 'season')
        home_cols_to_drop = [col for col in out.columns if col.endswith('_home_overlap')]
        home_cols_to_drop.extend(['team_norm', 'season']) # Add right-side key names
        out.drop(columns=[col for col in home_cols_to_drop if col in out.columns], inplace=True, errors='ignore')

        # 4) Merge stats for away teams
        away_stats_to_join = seasonal_stats_processed.copy()
        away_rename_mapping = {f"{stat}_away": f"a_{stat}_away" for stat in EXPECTED_STATS}
        away_stats_to_join = away_stats_to_join.rename(columns=away_rename_mapping)

        relevant_away_cols = ['team_norm', 'season'] + [f"a_{stat}_away" for stat in EXPECTED_STATS]
        away_stats_to_join = away_stats_to_join[list(set(away_stats_to_join.columns).intersection(relevant_away_cols))]

        out = pd.merge(
            out,
            away_stats_to_join,
            left_on=['away_norm', 'adv_stats_lookup_season'],
            right_on=['team_norm', 'season'],
            how='left',
            suffixes=('', '_away_overlap')
        )
        away_cols_to_drop = [col for col in out.columns if col.endswith('_away_overlap')]
        away_cols_to_drop.extend(['team_norm', 'season'])
        out.drop(columns=[col for col in away_cols_to_drop if col in out.columns], inplace=True, errors='ignore')

    # 5) Imputation for all generated h_..._home and a_..._away columns
    for side_label, team_prefix in (("home", "h"), ("away", "a")):
        for stat in EXPECTED_STATS:
            tgt_col = f"{team_prefix}_{stat}_{side_label}" # e.g., h_pace_home or a_efg_pct_away
            impute_col = f"{tgt_col}_imputed"
            default = DEFAULTS.get(stat, 0.0) # Get default value for the stat

            # If target column doesn't exist (e.g., merge failed or stat was missing from source)
            if tgt_col not in out.columns:
                out[tgt_col] = default # Create column with default value
                if flag_imputations:
                    out[impute_col] = True # All values are considered imputed
            else:
                # Column exists, may contain NaNs or non-numeric values that need coercion
                if flag_imputations:
                    # Determine NAs: either existing or resulting from coercion to numeric
                    # This must be done *before* fillna
                    out[impute_col] = pd.to_numeric(out[tgt_col], errors='coerce').isna()
                
                # Convert to numeric (coercing errors to NaN) and then fill with default
                out[tgt_col] = pd.to_numeric(out[tgt_col], errors='coerce').fillna(default)

            # Ensure final types
            out[tgt_col] = out[tgt_col].astype(float)
            if flag_imputations:
                # Ensure imputation flag column exists and is boolean
                if impute_col not in out.columns: # Should have been created if tgt_col was missing or if it existed
                    out[impute_col] = False # If somehow missed, assume no imputation occurred for safety
                out[impute_col] = out[impute_col].astype(bool)

    # 6) Compute split diffs (using the now correctly sourced and imputed h_... and a_... columns)
    for stat in EXPECTED_STATS:
        home_col_name = f"h_{stat}_home" # e.g. h_pace_home
        away_col_name = f"a_{stat}_away" # e.g. a_pace_away
        
        # Ensure columns exist before trying to subtract (they should, due to imputation step)
        if home_col_name in out.columns and away_col_name in out.columns:
            out[f"hist_{stat}_split_diff"] = out[home_col_name] - out[away_col_name]
        else:
            # This case should ideally not be reached if imputation logic is comprehensive
            logger.warning(f"Missing one or both columns for diff: {home_col_name}, {away_col_name}. Diff will be NaN.")
            out[f"hist_{stat}_split_diff"] = np.nan


    # 7) Mirror specific rating columns to names expected by downstream processes (e.g., rolling features)
    ratings_map = {
        "home_offensive_rating": "h_off_rtg_home",
        "away_offensive_rating": "a_off_rtg_away",
        "home_defensive_rating": "h_def_rtg_home",
        "away_defensive_rating": "a_def_rtg_away",
        "home_net_rating": "h_net_rtg_home",
        "away_net_rating": "a_net_rtg_away",
    }
    for target_rating_col, source_adv_col in ratings_map.items():
        if source_adv_col in out.columns: # Source columns (e.g., h_off_rtg_home) should exist
            out[target_rating_col] = out[source_adv_col]
        else:
            # This implies an issue in earlier steps if source_adv_col is missing
            logger.warning(f"Source column '{source_adv_col}' for rating mirroring is missing. '{target_rating_col}' will be NaN.")
            out[target_rating_col] = np.nan


    # 8) Cleanup helper columns
    columns_to_drop_final = ["home_norm", "away_norm"]
    # 'adv_stats_lookup_season' is kept by default unless specified to be dropped.
    # It might be useful for debugging or if other modules also key off it.
    # If it must be dropped here:
    # columns_to_drop_final.append('adv_stats_lookup_season')

    out.drop(columns=[col for col in columns_to_drop_final if col in out.columns], inplace=True, errors='ignore')
    
    return out