# backend/nba_features/season.py

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, determine_season

# Get logger for this module
logger = logging.getLogger(__name__)

__all__ = ["transform"]

# Helper to compute previous season string
def _previous_season(season_str: Optional[str]) -> Optional[str]: # Added Optional
    if not season_str or pd.isna(season_str):
        logger.warning("Received None or NaN season_str in _previous_season.")
        return None
    try:
        start, end = season_str.split('-')
        prev_start = int(start) - 1
        prev_end = (int(end) -1 + 100) % 100 # Handles '00' from '1999-00' -> '1998-99' correctly
                                          # and 'YY' from 'YYYY-YY'
        # Ensure prev_end corresponds to prev_start+1 logic for standard NBA seasons
        # e.g. 2023-24 -> prev_start=2022, end_of_prev_start_year=22. prev_end should be (22+1)%100 = 23
        expected_prev_end = (prev_start + 1) % 100

        return f"{prev_start}-{expected_prev_end:02d}"
    except Exception as e:
        logger.warning(f"Could not parse season_str '{season_str}' in _previous_season: {e}. Returning None.")
        return None


def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Attach previous-season statistical context for home and away teams.
    - Falls back to defaults if stats are unavailable or missing.
    - Creates boolean flags '<feature>_imputed' when values come from DEFAULTS.

    Args:
        df: Game-level DataFrame with columns ['game_id','game_date','home_team','away_team']
        team_stats_df: Optional DataFrame with ['team_name','season','wins_all_percentage',
                       'points_for_avg_all','points_against_avg_all','current_form']
        flag_imputations: If True, create boolean flags marking default-filled features.
        debug: If True, enable debug logging for this function.

    Returns:
        DataFrame with added season context features, optional flags, and diffs/net ratings.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Adding season context features...")
    result = df.copy()

    for col in ['game_id', 'game_date', 'home_team', 'away_team']:
        if col not in result.columns:
            logger.error(f"Missing essential column: {col}, skipping season features.")
            if debug: logger.setLevel(orig_level)
            return df # Return original df if requirements not met

    result['game_date'] = pd.to_datetime(result['game_date'], errors='coerce')
    result = result.dropna(subset=['game_date'])
    if result.empty:
        logger.warning("DataFrame empty after game_date processing.")
        if debug: logger.setLevel(orig_level)
        return result # Return empty if all dates were invalid

    placeholder_defaults = {
        'home_season_win_pct': DEFAULTS.get('win_pct', 0.5), # Ensure DEFAULTS values are appropriate type
        'away_season_win_pct': DEFAULTS.get('win_pct', 0.5),
        'home_season_avg_pts_for': DEFAULTS.get('avg_pts_for', 100.0),
        'away_season_avg_pts_for': DEFAULTS.get('avg_pts_for', 100.0),
        'home_season_avg_pts_against': DEFAULTS.get('avg_pts_against', 100.0),
        'away_season_avg_pts_against': DEFAULTS.get('avg_pts_against', 100.0),
        'home_current_form': DEFAULTS.get('current_form', 'N/A'), # Ensure DEFAULTS['current_form'] is str
        'away_current_form': DEFAULTS.get('current_form', 'N/A')
    }

    if team_stats_df is None or team_stats_df.empty:
        logger.warning("No team_stats_df provided - using defaults for all season features.")
        for feat, default_val in placeholder_defaults.items():
            result[feat] = default_val
            if flag_imputations:
                result[f"{feat}_imputed"] = True
        
        numeric_zeros = ['season_win_pct_diff','season_pts_for_diff','season_pts_against_diff',
                         'home_season_net_rating','away_season_net_rating','season_net_rating_diff']
        for z_col in numeric_zeros:
            result[z_col] = 0.0
        if debug: logger.setLevel(orig_level)
        return result

    result['season'] = result['game_date'].apply(determine_season)
    result['season_stats_lookup'] = result['season'].apply(_previous_season)
    result['home_norm'] = result['home_team'].astype(str).map(normalize_team_name)
    result['away_norm'] = result['away_team'].astype(str).map(normalize_team_name)

    ts = team_stats_df.copy()
    ts['team_norm'] = ts['team_name'].astype(str).map(normalize_team_name)
    ts['season'] = ts['season'].astype(str) # Ensure season is string for key creation
    ts['lookup_key'] = ts['team_norm'] + '_' + ts['season']
    ts = ts.drop_duplicates(subset=['lookup_key'], keep='last').set_index('lookup_key')
    
    # Select only the necessary columns from team_stats_df
    stats_cols_to_merge = ['wins_all_percentage','points_for_avg_all','points_against_avg_all','current_form']
    # Check if all expected stats cols exist in ts
    missing_stats_cols = [sc for sc in stats_cols_to_merge if sc not in ts.columns]
    if missing_stats_cols:
        logger.warning(f"Missing columns in team_stats_df: {missing_stats_cols}. These stats can't be merged.")
        # Remove missing columns from the list to avoid KeyError
        stats_cols_to_merge = [sc for sc in stats_cols_to_merge if sc in ts.columns]
    
    sub_ts = ts[stats_cols_to_merge] if stats_cols_to_merge else pd.DataFrame(index=ts.index) # Empty df if no cols

    for side in ['home', 'away']:
        prefix = f"{side}_"
        renamed_cols_map = {
            'wins_all_percentage': f"{prefix}season_win_pct",
            'points_for_avg_all': f"{prefix}season_avg_pts_for",
            'points_against_avg_all': f"{prefix}season_avg_pts_against",
            'current_form': f"{prefix}current_form"
        }
        # Filter map for only columns present in sub_ts
        actual_renamed_cols_map = {k: v for k, v in renamed_cols_map.items() if k in sub_ts.columns}
        renamed_sub_ts = sub_ts.rename(columns=actual_renamed_cols_map)
        
        # Create merge key, handling potential None from _previous_season
        # If season_stats_lookup is None, key_col will contain NaN for that part if not str
        result['merge_key_temp'] = result[f"{side}_norm"].astype(str) + '_' + result['season_stats_lookup'].astype(str)
        
        result = result.merge(
            renamed_sub_ts,
            how='left',
            left_on='merge_key_temp',
            right_index=True
        )
        result = result.drop(columns=['merge_key_temp'])

        if flag_imputations:
            for original_name, suffixed_name in actual_renamed_cols_map.items():
                if suffixed_name in result.columns: # Check if column was actually added
                    result[f"{suffixed_name}_imputed"] = result[suffixed_name].isnull() # isna() is fine, astype(bool) later if needed

    # Fill defaults and ensure types
    for feat_col, default_val in placeholder_defaults.items():
        is_string_col = isinstance(default_val, str) # e.g. current_form
        
        if feat_col not in result.columns: # Column wasn't created by merge (e.g. a stat was missing in team_stats_df)
            result[feat_col] = default_val
            if flag_imputations:
                result[f"{feat_col}_imputed"] = True # This is a full imputation
        else:
            # Column exists, fill NaNs from failed lookups or NaNs in source data
            if flag_imputations and f"{feat_col}_imputed" not in result.columns: # Create imputation flag if not already made
                 result[f"{feat_col}_imputed"] = result[feat_col].isnull() # Mark rows where original value was NaN

            if is_string_col:
                result[feat_col] = result[feat_col].fillna(default_val)
            else:
                # For numeric columns, ensure they are numeric before fillna, then fill, then type
                result[feat_col] = pd.to_numeric(result[feat_col], errors='coerce').fillna(default_val)
        
        # Ensure final type
        if is_string_col:
            result[feat_col] = result[feat_col].astype(str)
        else:
            result[feat_col] = result[feat_col].astype(float) # All numeric stats become float
        
        if flag_imputations and f"{feat_col}_imputed" in result.columns:
            result[f"{feat_col}_imputed"] = result[f"{feat_col}_imputed"].astype(bool)


    result['season_win_pct_diff'] = result['home_season_win_pct'] - result['away_season_win_pct']
    result['season_pts_for_diff'] = result['home_season_avg_pts_for'] - result['away_season_avg_pts_for']
    result['season_pts_against_diff'] = result['home_season_avg_pts_against'] - result['away_season_avg_pts_against']
    result['home_season_net_rating'] = result['home_season_avg_pts_for'] - result['home_season_avg_pts_against']
    result['away_season_net_rating'] = result['away_season_avg_pts_for'] - result['away_season_avg_pts_against']
    result['season_net_rating_diff'] = result['home_season_net_rating'] - result['away_season_net_rating']
    
    # Ensure diff/rating columns are float
    diff_rating_cols = ['season_win_pct_diff','season_pts_for_diff','season_pts_against_diff',
                        'home_season_net_rating','away_season_net_rating','season_net_rating_diff']
    for dr_col in diff_rating_cols:
        if dr_col in result.columns: # Should always be true if base stats were created
            result[dr_col] = result[dr_col].astype(float)


    result.drop(columns=['season', 'season_stats_lookup', 'home_norm', 'away_norm'], inplace=True, errors='ignore')

    original_cols = list(df.columns) # Columns from the original input df
    seasonal_stat_cols = list(placeholder_defaults.keys())
    imputation_flag_cols = [c for c in result.columns if c.endswith('_imputed')]
    # diffs_and_ratings_cols already defined as diff_rating_cols

    # Build the final column order
    final_ordered_cols = []
    # Add original columns first, in their original order
    for col in original_cols:
        if col in result.columns and col not in final_ordered_cols:
            final_ordered_cols.append(col)
    
    # Add new feature groups, ensuring no duplicates and only existing columns
    for group in [seasonal_stat_cols, imputation_flag_cols, diff_rating_cols]:
        for col in group:
            if col in result.columns and col not in final_ordered_cols:
                final_ordered_cols.append(col)
    
    # Add any other new columns that might have been created and not in above groups
    # (e.g. if a new ungrouped feature was added)
    # for col in result.columns:
    #     if col not in final_ordered_cols:
    #         final_ordered_cols.append(col)

    result = result[final_ordered_cols]

    logger.info("Finished adding season context features.")
    if debug:
        logger.setLevel(orig_level)
    return result
