# backend/features/rest.py

from __future__ import annotations
import logging
from typing import Any # Keep Any for DEFAULTS typing if needed

import numpy as np
import pandas as pd

# Import necessary components from the utils module
# Note: normalize_team_name and TEAMS_TO_WATCH are not directly used by the active logic,
# but might be needed if the commented-out debug sections are re-enabled.
from .utils import DEFAULTS #, normalize_team_name, TEAMS_TO_WATCH

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
    debug: bool = False,
) -> pd.DataFrame:
    """
    Adds rest days, games in last N days, back-to-back flags, and schedule advantages.

    Args:
        df: Input DataFrame containing 'game_id', 'game_date', 'home_team', 'away_team'.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added rest and schedule-related feature columns.
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for rest.transform")

    essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    placeholder_cols = [
        'rest_days_home', 'games_last_7_days_home', 'games_last_14_days_home',
        'rest_days_away', 'games_last_7_days_away', 'games_last_14_days_away',
        'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage'
    ]
    expected_schedule_cols = [
        'games_last_7_days_home', 'games_last_14_days_home',
        'games_last_7_days_away', 'games_last_14_days_away'
    ]

    # --- Input Validation ---
    if df is None or df.empty:
        logger.warning("rest.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return pd.DataFrame()

    if not all(col in df.columns for col in essential_cols):
        missing = set(essential_cols) - set(df.columns)
        logger.warning(f"rest.transform: Input df missing essential columns {missing}. Returning DataFrame with defaults.")
        output_df = df.copy()
        # Add default placeholders if essential columns are missing
        for col in placeholder_cols:
            if col not in output_df.columns:
                # Extract base key for default lookup (e.g., 'rest_days' from 'rest_days_home')
                default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '').replace('is_back_to_back', 'rest_days') # Map B2B to rest_days default? Or add specific B2B default? Using rest_days for now.
                # Handle potential missing keys in DEFAULTS gracefully
                default_val = DEFAULTS.get(default_key, 0.0 if 'games' in col or 'advantage' in col or 'streak' in col else 3.0) # Basic type guessing
                output_df[col] = default_val
        if debug: logger.setLevel(current_level) # Restore logger level
        return output_df

    logger.debug("Adding rest features (vectorized)...")
    # Work on a copy
    df_copy = df.copy().reset_index(drop=True)

    # --- Data Preparation ---
    try:
        # Ensure correct data types and handle timezones
        df_copy['game_date'] = pd.to_datetime(df_copy['game_date'], errors='coerce').dt.tz_localize(None)
        df_copy['home_team'] = df_copy['home_team'].astype(str)
        df_copy['away_team'] = df_copy['away_team'].astype(str)
        df_copy['game_id'] = df_copy['game_id'].astype(str)
        # Check if date conversion failed significantly
        if df_copy['game_date'].isnull().all():
            logger.warning("All game_date values invalid/missing after conversion. Cannot calculate rest features accurately.")
            raise ValueError("Invalid game dates")
    except Exception as e:
        logger.error(f"Error processing essential columns for rest features: {e}. Returning DataFrame with defaults.", exc_info=debug)
        # Add default placeholders if processing fails
        for col in placeholder_cols:
            if col not in df_copy.columns:
                default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '').replace('is_back_to_back', 'rest_days')
                default_val = DEFAULTS.get(default_key, 0.0 if 'games' in col or 'advantage' in col or 'streak' in col else 3.0)
                df_copy[col] = default_val
        if debug: logger.setLevel(current_level) # Restore logger level
        return df_copy

    # --- Calculate Rest Days ---
    logger.debug("Calculating rest days...")
    try:
        # Create a long format log of games per team
        team_log = (
            pd.concat([
                df_copy[['game_date', 'home_team']].rename(columns={'home_team': 'team'}),
                df_copy[['game_date', 'away_team']].rename(columns={'away_team': 'team'})
            ], ignore_index=True)
            .sort_values(['team', 'game_date'])
            .drop_duplicates(subset=['team', 'game_date'], keep='first') # Keep first instance on a given day if duplicates exist
        )
        # Calculate previous game date for each team
        team_log['prev_game_date'] = team_log.groupby('team', observed=True)['game_date'].shift(1)

        # Prepare for merging
        temp_rest = team_log[['team', 'game_date', 'prev_game_date']].drop_duplicates(subset=['team', 'game_date'])

        # Merge previous game date for home teams
        df_copy = pd.merge(
            df_copy,
            temp_rest,
            how='left',
            left_on=['home_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one' # Each game date for a team should be unique in temp_rest
        ).rename(columns={'prev_game_date': 'prev_home_game_date'}).drop(columns='team', errors='ignore')

        # Merge previous game date for away teams
        df_copy = pd.merge(
            df_copy,
            temp_rest,
            how='left',
            left_on=['away_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'
        ).rename(columns={'prev_game_date': 'prev_away_game_date'}).drop(columns='team', errors='ignore')

        # Calculate rest days and fill NaNs (first game) with default
        df_copy['rest_days_home'] = (df_copy['game_date'] - df_copy['prev_home_game_date']).dt.days.fillna(DEFAULTS['rest_days'])
        df_copy['rest_days_away'] = (df_copy['game_date'] - df_copy['prev_away_game_date']).dt.days.fillna(DEFAULTS['rest_days'])

    except Exception as e_rest:
         logger.error(f"Error calculating rest days: {e_rest}. Filling rest days with defaults.", exc_info=debug)
         df_copy['rest_days_home'] = df_copy.get('rest_days_home', DEFAULTS['rest_days']).fillna(DEFAULTS['rest_days'])
         df_copy['rest_days_away'] = df_copy.get('rest_days_away', DEFAULTS['rest_days']).fillna(DEFAULTS['rest_days'])


    # --- Calculate Schedule Density (Games in Last 7/14 Days) ---
    logger.debug("Calculating games in last 7/14 days...")
    try:
        # Need game_id associated with each team/date entry for accurate rolling count
        home_log_ids = df_copy[['game_date', 'home_team', 'game_id']].rename(columns={'home_team': 'team'})
        away_log_ids = df_copy[['game_date', 'away_team', 'game_id']].rename(columns={'away_team': 'team'})
        game_ids_log = pd.concat([home_log_ids, away_log_ids], ignore_index=True)
        # Ensure unique game_id per team per date (using keep='first' consistent with team_log)
        game_ids_log = game_ids_log.drop_duplicates(subset=['team', 'game_date'], keep='first')

        # Merge the unique game_ids back to team_log
        # Use the already created team_log which has unique team-date pairs
        team_log_with_id = pd.merge(
            team_log.drop(columns='prev_game_date', errors='ignore'), # Drop prev_game_date if still there
            game_ids_log,
            on=['team', 'game_date'],
            how='left', # Keep all team-date entries, match game_id
            validate='one_to_one' # Should be unique after deduplication steps
        )
        if team_log_with_id['game_id'].isnull().any():
            # This might happen if df_copy had games not present in the original team_log build (unlikely with current logic)
            logger.warning("Could not associate game_id with all team-date entries in team_log. Rolling counts might be affected.")
            # Fill missing game_ids with a placeholder to avoid erroring the rolling count, though counts might be off
            team_log_with_id['game_id'] = team_log_with_id['game_id'].fillna('missing_id')


        # Set date as index for time-based rolling window
        team_log_indexed = team_log_with_id.set_index('game_date')
        # Calculate rolling count of games (using game_id) in the preceding 7/14 days (closed='left')
        counts_7d_series = team_log_indexed.groupby('team', observed=True)['game_id'].rolling('7D', closed='left').count()
        counts_14d_series = team_log_indexed.groupby('team', observed=True)['game_id'].rolling('14D', closed='left').count()
        counts_7d_series.name = 'count_7d'
        counts_14d_series.name = 'count_14d'

        # Reset index to merge back
        counts_7d_df = counts_7d_series.reset_index()
        counts_14d_df = counts_14d_series.reset_index()

        # Merge counts back based on team and game_date
        team_log_with_counts = pd.merge(
            team_log.drop(columns='prev_game_date', errors='ignore'), # Start fresh merge target
             counts_7d_df, on=['team', 'game_date'], how='left', validate='one_to_one'
        )
        team_log_with_counts = pd.merge(
            team_log_with_counts, counts_14d_df, on=['team', 'game_date'], how='left', validate='one_to_one'
        )
        # Fill NaNs (for first few games in window) with 0 and ensure integer type
        team_log_with_counts['count_7d'] = team_log_with_counts['count_7d'].fillna(0).astype(int)
        team_log_with_counts['count_14d'] = team_log_with_counts['count_14d'].fillna(0).astype(int)
        # Ensure uniqueness again before final merge
        team_log_with_counts = team_log_with_counts.drop_duplicates(subset=['team', 'game_date'])

        # Prepare a deduplicated temp df for merging schedule counts
        temp_sched = team_log_with_counts[['team', 'game_date', 'count_7d', 'count_14d']]

        # Merge schedule counts for home teams
        df_copy = pd.merge(
            df_copy,
            temp_sched,
            how='left',
            left_on=['home_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'
        ).rename(columns={'count_7d': 'games_last_7_days_home', 'count_14d': 'games_last_14_days_home'}).drop(columns='team', errors='ignore')

        # Merge schedule counts for away teams
        df_copy = pd.merge(
            df_copy,
            temp_sched,
            how='left',
            left_on=['away_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'
        ).rename(columns={'count_7d': 'games_last_7_days_away', 'count_14d': 'games_last_14_days_away'}).drop(columns='team', errors='ignore')

        # Fill NaNs in the final schedule columns with appropriate defaults
        for col in expected_schedule_cols:
             # Extract base key for default lookup (e.g., 'games_last_7_days' from 'games_last_7_days_home')
            default_key = col.replace('_home', '').replace('_away', '')
            default_val = DEFAULTS.get(default_key, 0) # Default to 0 for game counts
            df_copy[col] = df_copy[col].fillna(default_val).astype(int)

    except Exception as rolling_e:
        logger.error(f"Error during time-based rolling count for schedule density: {rolling_e}. Filling with defaults.", exc_info=debug)
        # Fill schedule columns with defaults if calculation fails
        for col in expected_schedule_cols:
            default_key = col.replace('_home', '').replace('_away', '')
            default_val = DEFAULTS.get(default_key, 0)
            if col not in df_copy.columns:
                df_copy[col] = default_val
            else:
                # Fill only NaNs that might exist from partial merge failures
                df_copy[col] = df_copy[col].fillna(default_val)
            df_copy[col] = df_copy[col].astype(int)

    # --- Calculate Final Derived Features ---
    # Calculate back-to-back flags (rest days == 1)
    df_copy['is_back_to_back_home'] = (df_copy['rest_days_home'] == 1).astype(int)
    df_copy['is_back_to_back_away'] = (df_copy['rest_days_away'] == 1).astype(int)
    # Calculate rest advantage (positive means home team has more rest)
    df_copy['rest_advantage'] = df_copy['rest_days_home'] - df_copy['rest_days_away']
    # Calculate schedule advantage (positive means home team played FEWER games recently)
    df_copy['schedule_advantage'] = df_copy['games_last_7_days_away'] - df_copy['games_last_7_days_home']

    # --- Clean Up ---
    # Drop intermediate columns used for calculation
    df_copy = df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date'], errors='ignore')

    # Ensure all placeholder columns exist in the final df, filling if necessary
    # (This is a safety check, should be populated by calculations or error handling)
    for col in placeholder_cols:
        if col not in df_copy.columns:
            logger.warning(f"Column '{col}' was unexpectedly missing at the end. Filling with default.")
            default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '').replace('is_back_to_back', 'rest_days')
            default_val = DEFAULTS.get(default_key, 0.0 if 'games' in col or 'advantage' in col or 'streak' in col else 3.0)
            df_copy[col] = default_val
        else:
             # Ensure no NaNs remain in the final columns (e.g., from edge cases)
             default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '').replace('is_back_to_back', 'rest_days')
             default_val = DEFAULTS.get(default_key, 0.0 if 'games' in col or 'advantage' in col or 'streak' in col else 3.0)
             df_copy[col] = df_copy[col].fillna(default_val)


    logger.debug("Finished adding rest features.")
    logger.debug("rest.transform: done, output shape=%s", df_copy.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    # Return the DataFrame with original columns plus the new rest features
    return df_copy