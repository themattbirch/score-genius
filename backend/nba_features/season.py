# backend/features/season.py

from __future__ import annotations
import logging
from typing import Any, Optional # Keep Any for DEFAULTS typing if needed

import numpy as np
import pandas as pd

# Import necessary components from the utils module
from .utils import DEFAULTS, normalize_team_name, determine_season # Import required utils

# --- Logger Configuration ---
# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly export the transform function
__all__ = ["transform"]

# -- Constants --
# EPSILON = 1e-6 # Moved to utils.py if needed globally

def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None, # Standardized input name
    debug: bool = False,
) -> pd.DataFrame:
    """
    Merges seasonal team statistics onto the game data and calculates derived season diffs.

    Args:
        df: Input DataFrame containing 'game_id', 'game_date', 'home_team', 'away_team'.
        team_stats_df: Optional DataFrame containing pre-calculated seasonal stats per team.
                       Expected columns: 'team_name', 'season', 'wins_all_percentage',
                       'points_for_avg_all', 'points_against_avg_all', 'current_form'.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added seasonal context features.
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for season.transform")

    logger.info("Adding season context features...")
    result_df = df.copy() # Work on a copy

    # Define placeholder columns to ensure they exist even if merge fails
    placeholder_cols = [
        'home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for', 'away_season_avg_pts_for',
        'home_season_avg_pts_against', 'away_season_avg_pts_against', 'current_form',
        'season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff',
        'home_season_net_rating', 'away_season_net_rating', 'season_net_rating_diff'
    ]
    # Required columns from the optional team_stats_df
    req_ts_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
    # Columns needed for diff calculations
    diff_input_cols = [
        'home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for', 'away_season_avg_pts_for',
        'home_season_avg_pts_against', 'away_season_avg_pts_against'
        # Net rating columns are calculated first, then used
    ]


    # --- Input Validation and Preparation ---
    team_stats_available = False
    ts_df = None

    if team_stats_df is not None and not team_stats_df.empty:
        ts_df = team_stats_df.copy()
        missing_ts_cols = [col for col in req_ts_cols if col not in ts_df.columns]
        if missing_ts_cols:
            logger.warning(f"Provided team_stats_df missing required columns: {missing_ts_cols}. Attempting to fill with defaults.")
            # Fill missing columns in the provided stats df with defaults
            for col in missing_ts_cols:
                if col == 'current_form':
                    ts_df[col] = 'N/A'
                elif 'percentage' in col:
                    ts_df[col] = DEFAULTS.get('win_pct', 0.5) # Use default win_pct
                elif 'points_for' in col:
                     ts_df[col] = DEFAULTS.get('avg_pts_for', 115.0)
                elif 'points_against' in col:
                     ts_df[col] = DEFAULTS.get('avg_pts_against', 115.0)
                else:
                    ts_df[col] = 0.0 # Generic numeric default
        team_stats_available = True
    else:
        logger.warning("`team_stats_df` is empty or None. Season context features will use defaults.")

    # Check essential columns in the main dataframe
    essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if not all(col in result_df.columns for col in essential_cols):
         missing_main = set(essential_cols) - set(result_df.columns)
         logger.error(f"Input `df` missing essential columns: {missing_main}. Cannot proceed with season merge.")
         team_stats_available = False # Cannot merge without keys

    # Check game_date validity
    if 'game_date' not in result_df.columns or pd.to_datetime(result_df['game_date'], errors='coerce').isnull().all():
        logger.error("`game_date` missing or all invalid in input df. Cannot determine season for merging.")
        team_stats_available = False

    # --- Merge Logic or Default Filling ---
    if not team_stats_available:
        logger.warning("Adding season context placeholders with defaults as stats are unavailable or input df is invalid.")
        # Fill ALL placeholder columns (including diffs) if merge is skipped
        for col in placeholder_cols:
            if col not in result_df.columns:
                # Determine default key (e.g., 'win_pct' from 'home_season_win_pct')
                base_key = col.replace('home_', '').replace('away_', '').replace('season_', '')
                diff_key = base_key.replace('_diff', '').replace('_net_rating','')
                form_key = base_key.replace('current_', '')

                default_val = 0.0 # Default numeric value
                if 'win_pct' in diff_key: default_val = DEFAULTS.get('win_pct', 0.5)
                elif 'form' in form_key: default_val = 'N/A' # String default for form
                elif 'avg_pts_for' in diff_key: default_val = DEFAULTS.get('avg_pts_for', 115.0)
                elif 'avg_pts_against' in diff_key: default_val = DEFAULTS.get('avg_pts_against', 115.0)
                elif 'diff' in base_key or 'net_rating' in base_key: default_val = 0.0 # Diffs default to 0
                else: default_val = DEFAULTS.get(diff_key, 0.0) # Fallback lookup

                result_df[col] = default_val

            # Ensure correct types after potential default fill
            if 'form' in col:
                result_df[col] = result_df[col].fillna('N/A').astype(str)
            else:
                # Use a known numeric default for filling NaNs before conversion
                numeric_default = 0.5 if 'win_pct' in col else 0.0
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(numeric_default)

    else: # Proceed with merging
        try:
            logger.debug("Preparing data for season stats merge...")
            # Prepare main DataFrame
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date']) # Drop rows where date conversion failed
            if result_df.empty:
                raise ValueError("No valid game dates in target df after cleaning.")

            # Determine season for each game using utility function
            result_df['season'] = result_df['game_date'].apply(determine_season)
            # Normalize team names using utility function
            result_df['home_team_norm'] = result_df['home_team'].astype(str).map(normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].astype(str).map(normalize_team_name)

            # Prepare team_stats DataFrame
            ts_df['team_name_norm'] = ts_df['team_name'].astype(str).map(normalize_team_name)
            # Select necessary columns and ensure types
            ts_merge = ts_df[['team_name_norm', 'season'] + [c for c in req_ts_cols if c not in ['team_name', 'season']]].copy()
            for col in ['wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all']:
                ts_merge[col] = pd.to_numeric(ts_merge.get(col), errors='coerce')
            ts_merge['season'] = ts_merge['season'].astype(str)
            ts_merge['current_form'] = ts_merge.get('current_form', 'N/A').astype(str).fillna('N/A')

            # Create merge key: team_normalized_Boston Celtics_2023-2024
            ts_merge['merge_key'] = ts_merge['team_name_norm'] + "_" + ts_merge['season']
            # Keep only the last entry per team per season if duplicates exist
            ts_merge = ts_merge.drop(columns=['team_name_norm', 'season']).drop_duplicates(subset=['merge_key'], keep='last')

            # Create merge keys in the main DataFrame
            result_df['merge_key_home'] = result_df['home_team_norm'] + "_" + result_df['season']
            result_df['merge_key_away'] = result_df['away_team_norm'] + "_" + result_df['season']

            # Define rename mappings for merging
            home_rename = {
                'wins_all_percentage': 'home_season_win_pct',
                'points_for_avg_all': 'home_season_avg_pts_for',
                'points_against_avg_all': 'home_season_avg_pts_against',
            }
            away_rename = {
                'wins_all_percentage': 'away_season_win_pct',
                'points_for_avg_all': 'away_season_avg_pts_for',
                'points_against_avg_all': 'away_season_avg_pts_against',
            }

            logger.debug("Attempting to merge season stats using keys...")
            # Merge for home team
            result_df = pd.merge(
                result_df,
                ts_merge.rename(columns=home_rename),
                how='left',
                left_on='merge_key_home',
                right_on='merge_key',
                indicator='_merge_home' # Add indicator column
            )
            result_df = result_df.drop(columns=['merge_key'], errors='ignore') # Drop key from right side

            # Merge for away team
            result_df = pd.merge(
                result_df,
                ts_merge.rename(columns=away_rename),
                how='left',
                left_on='merge_key_away',
                right_on='merge_key',
                suffixes=('', '_away_dup'), # Add suffix for potentially duplicated columns if merge key wasn't dropped properly
                indicator='_merge_away' # Add indicator column
            )
            result_df = result_df.drop(columns=['merge_key'], errors='ignore') # Drop key from right side

            # Log merge success rate
            if '_merge_home' in result_df.columns and '_merge_away' in result_df.columns:
                home_success = (result_df['_merge_home'] == 'both').mean()
                away_success = (result_df['_merge_away'] == 'both').mean()
                logger.info(f"Season stats merge success rate: Home={home_success:.1%}, Away={away_success:.1%}")
                if home_success < 0.9 or away_success < 0.9:
                    logger.warning("Low merge success rate for season stats. Check team name normalization, season formats in both dataframes, and coverage of team_stats_df.")
                # Drop merge indicators
                result_df = result_df.drop(columns=['_merge_home', '_merge_away'], errors='ignore')
            else:
                logger.warning("Merge indicators not found after merging season stats. Cannot report success rate.")

            # Clean up any duplicated columns from merge (shouldn't happen with proper key drops)
            result_df = result_df.drop(columns=[c for c in result_df.columns if '_away_dup' in c], errors='ignore')

        except Exception as merge_e:
            logger.error(f"Error during season stats merge process: {merge_e}", exc_info=debug)
            # If merge fails, ensure placeholder columns exist and fill with NaN for now
            for col in placeholder_cols:
                if col not in result_df.columns:
                    result_df[col] = np.nan

    # --- Final Processing (Filling NaNs and Calculating Diffs) ---
    # This block runs whether merge succeeded, failed, or was skipped
    logger.debug("Finalizing season context features (filling NaNs, calculating diffs)...")

    # ** NEW: Ensure input columns for diffs exist and are numeric BEFORE calculation **
    logger.debug("Ensuring numeric types for diff input columns...")
    for col in diff_input_cols:
        # Determine default based on column name
        default_val = 0.0
        if 'win_pct' in col: default_val = DEFAULTS.get('win_pct', 0.5)
        elif 'avg_pts_for' in col: default_val = DEFAULTS.get('avg_pts_for', 115.0)
        elif 'avg_pts_against' in col: default_val = DEFAULTS.get('avg_pts_against', 115.0)

        if col not in result_df.columns:
            logger.warning(f"Column '{col}' needed for diff calculation is missing. Adding with default: {default_val}")
            result_df[col] = default_val
        else:
            # Fill NaNs first, then coerce to numeric
            result_df[col] = result_df[col].fillna(default_val)
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)


    # Calculate differential features using the (now guaranteed numeric) columns
    try:
        logger.debug("Calculating seasonal differentials...")
        result_df['season_win_pct_diff'] = result_df['home_season_win_pct'] - result_df['away_season_win_pct']
        result_df['season_pts_for_diff'] = result_df['home_season_avg_pts_for'] - result_df['away_season_avg_pts_for']
        result_df['season_pts_against_diff'] = result_df['home_season_avg_pts_against'] - result_df['away_season_avg_pts_against']
        # Calculate net ratings before calculating their difference
        result_df['home_season_net_rating'] = result_df['home_season_avg_pts_for'] - result_df['home_season_avg_pts_against']
        result_df['away_season_net_rating'] = result_df['away_season_avg_pts_for'] - result_df['away_season_avg_pts_against']
        result_df['season_net_rating_diff'] = result_df['home_season_net_rating'] - result_df['away_season_net_rating']
    except Exception as diff_e:
         logger.error(f"Error calculating seasonal differentials: {diff_e}", exc_info=debug)
         # Ensure diff columns exist even if calculation fails, filling with 0
         diff_output_cols = ['season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff', 'home_season_net_rating', 'away_season_net_rating', 'season_net_rating_diff']
         for diff_col in diff_output_cols:
              if diff_col not in result_df.columns: result_df[diff_col] = 0.0
              else: result_df[diff_col] = result_df[diff_col].fillna(0.0) # Fill NaNs if calc partially failed


    # Final check and type enforcement for ALL placeholder columns
    logger.debug("Final type enforcement for all season module columns...")
    for col in placeholder_cols:
        # Determine default key (e.g., 'win_pct' from 'home_season_win_pct')
        base_key = col.replace('home_', '').replace('away_', '').replace('season_', '')
        diff_key = base_key.replace('_diff', '').replace('_net_rating','')
        form_key = base_key.replace('current_', '')

        default_val = 0.0 # Default numeric value
        if 'win_pct' in diff_key: default_val = DEFAULTS.get('win_pct', 0.5)
        elif 'form' in form_key: default_val = 'N/A' # String default for form
        elif 'avg_pts_for' in diff_key: default_val = DEFAULTS.get('avg_pts_for', 115.0)
        elif 'avg_pts_against' in diff_key: default_val = DEFAULTS.get('avg_pts_against', 115.0)
        elif 'diff' in base_key or 'net_rating' in base_key: default_val = 0.0 # Diffs default to 0
        else: default_val = DEFAULTS.get(diff_key, 0.0) # Fallback lookup

        # Ensure column exists and fill NaNs with the determined default
        if col not in result_df.columns:
             logger.warning(f"Column '{col}' missing before final fill. Adding with default: {default_val}")
             result_df[col] = default_val
        else:
             # Use fillna with the specific default for this column type
             result_df[col] = result_df[col].fillna(default_val)

        # Ensure correct final types
        if 'form' in col:
            result_df[col] = result_df[col].astype(str)
        else:
            # Use the determined default_val for filling NaNs during numeric conversion
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)


    # --- Clean Up ---
    # Drop intermediate columns used for merging
    result_df = result_df.drop(columns=['season', 'home_team_norm', 'away_team_norm', 'merge_key_home', 'merge_key_away'], errors='ignore')

    logger.info("Finished adding season context features.")
    logger.debug("season.transform: done, output shape=%s", result_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return result_df
