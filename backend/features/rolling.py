# backend/features/rolling.py

from __future__ import annotations
import logging
from typing import Any, List # Keep Any for DEFAULTS typing if needed

import numpy as np
import pandas as pd

# Import necessary components from the utils module
from .utils import DEFAULTS, normalize_team_name # Import DEFAULTS and normalize_team_name

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

# -- Helper Functions specific to rolling features --

def generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """
    Generate standardized rolling feature column name.
    Prefix can be 'home', 'away', or ''.
    Example: generate_rolling_column_name('home', 'score_for', 'mean', 5) -> 'home_rolling_score_for_mean_5'
    """
    prefix_part = f"{prefix}_" if prefix else ""
    return f"{prefix_part}rolling_{base}_{stat_type}_{window}"

# -- Main Transformation Function --

def transform(
    df: pd.DataFrame,
    *,
    window_sizes: List[int] = [5, 10, 20], # Default window sizes
    debug: bool = False,
) -> pd.DataFrame:
    """
    Adds rolling mean/std features for specified stats, calculated using past games only.

    This implementation uses pandas rolling operations with careful shifting and
    date checking to prevent data leakage from the current game into the rolling window.

    Args:
        df: Input DataFrame containing 'game_id', 'game_date', 'home_team', 'away_team',
            and the base statistics required (scores, ratings).
        window_sizes: A list of integers representing the rolling window sizes.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added rolling feature columns (e.g., 'home_rolling_score_for_mean_5').
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for rolling.transform")

    logger.debug(f"Calculating rolling features with windows: {window_sizes}")

    # --- Input Validation ---
    if df is None or df.empty:
        logger.warning("rolling.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return pd.DataFrame()

    essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if not all(col in df.columns for col in essential_cols):
        missing = set(essential_cols) - set(df.columns)
        logger.error(f"rolling.transform: Input df missing essential columns {missing}. Cannot proceed.")
        if debug: logger.setLevel(current_level) # Restore logger level
        # Returning original df might be problematic, better to return empty or raise error
        return df # Or raise ValueError(f"Missing essential columns: {missing}")

    # --- 1. Preparation and Normalization ---
    logger.debug("Preparing data: copying, normalizing dates and team names...")
    local = df.copy()
    try:
        # Ensure game_date is datetime and timezone-naive
        local['game_date'] = (
            pd.to_datetime(local['game_date'], errors='coerce')
            .dt.tz_localize(None)
        )
        # Check for invalid dates after conversion
        if local['game_date'].isnull().any():
             # Log how many NaTs were introduced
             nat_count = local['game_date'].isnull().sum()
             logger.warning(f"{nat_count} rows have invalid game_date after conversion. These rows might be dropped or cause issues.")
             # Option 1: Drop rows with NaT dates
             # local = local.dropna(subset=['game_date'])
             # Option 2: Proceed but be aware calculations might fail for these rows
             # (Current logic proceeds)

        # Normalize team names using the utility function
        # Add normalized columns; keep original team names as well
        local['home_norm'] = local['home_team'].astype(str).map(normalize_team_name)
        local['away_norm'] = local['away_team'].astype(str).map(normalize_team_name)
        local['game_id'] = local['game_id'].astype(str) # Ensure game_id is string

    except Exception as e_prep:
        logger.error(f"Error during data preparation: {e_prep}", exc_info=debug)
        if debug: logger.setLevel(current_level) # Restore logger level
        return df # Return original df on critical prep error

    # Define the mapping from raw stat columns to generic stat names used in rolling calculations
    # Ensure these raw columns exist in the input 'df'
    stat_map = {
        # Raw Column Name         : Generic Name for Rolling Calc
        'home_score':             'score_for',
        'away_score':             'score_for', # Away score becomes 'score_for' from away team's perspective
        'home_offensive_rating':  'off_rating',
        'away_offensive_rating':  'off_rating',
        'home_defensive_rating':  'def_rating', # Home def rating is what away team scored (off_rating)
        'away_defensive_rating':  'def_rating', # Away def rating is what home team scored (off_rating)
        'home_net_rating':        'net_rating',
        'away_net_rating':        'net_rating',
    }

    # Check which required raw stat columns are actually present
    available_raw_cols = [col for col in stat_map.keys() if col in local.columns]
    if not available_raw_cols:
         logger.error("Rolling: None of the required base stat columns found in input DataFrame. Cannot calculate rolling features.")
         if debug: logger.setLevel(current_level) # Restore logger level
         return local # Return df with normalized names but no rolling features

    # Filter stat_map to only include available columns
    filtered_stat_map = {k: v for k, v in stat_map.items() if k in available_raw_cols}
    logger.debug(f"Found available base stats for rolling: {list(filtered_stat_map.keys())}")


    # --- 2. Build Long Format DataFrame ---
    # Reshape data so each row represents one team's stats for a game
    logger.debug("Building long format DataFrame for team-based rolling calculations...")
    pieces = []
    for raw_col, gen_stat_name in filtered_stat_map.items():
        # Determine if it's a home or away stat
        side = 'home' if raw_col.startswith('home_') else 'away'
        # Select relevant columns and rename for generic long format
        piece = local[['game_id', 'game_date', f"{side}_norm", raw_col]].copy()
        piece = piece.rename(columns={f"{side}_norm": 'team_norm', raw_col: gen_stat_name})
        pieces.append(piece)

    if not pieces:
        # This case should be caught earlier, but as a safeguard:
        logger.error("Rolling: No data pieces generated for long format. Cannot proceed.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return local

    # Concatenate all pieces into the long format DataFrame
    team_view_df = pd.concat(pieces, ignore_index=True)

    # Ensure game_id is string type
    team_view_df['game_id'] = team_view_df['game_id'].astype(str)

    # --- 3. Perform Rolling Calculations ---
    logger.debug("Applying rolling calculations with stable sort and shift...")
    # **CRITICAL STEP**: Sort stably by team and date BEFORE grouping.
    # This ensures that .shift(1) correctly accesses the immediately preceding game for that team.
    team_view_df = team_view_df.sort_values(['team_norm', 'game_date'], kind='mergesort')

    all_rolled_stats = [] # Collect DataFrames for each stat/window combination

    # Iterate through specified window sizes
    for window in window_sizes:
        # Minimum number of periods required within the window to produce a result
        min_periods = max(1, window // 2) # Require at least half the window size

        # Iterate through the unique generic stat names (e.g., 'score_for', 'off_rating')
        for gen_stat_name in set(filtered_stat_map.values()):
            logger.debug(f"Calculating rolling {gen_stat_name} for window={window}...")

            # Group by team to perform rolling calculations independently for each team
            grouped_by_team = team_view_df[['team_norm', 'game_id', 'game_date', gen_stat_name]] \
                                    .groupby('team_norm', observed=True)

            total_leaks_detected = 0 # Counter for leaks within this group/stat/window

            # Define the function to apply to each team's group
            def calculate_rolling_for_group(group_df: pd.DataFrame) -> pd.DataFrame:
                nonlocal total_leaks_detected # Allow modification of the outer counter

                # **CRITICAL**: Ensure stable sort *within* the group again (belt-and-suspenders)
                group_df = group_df.sort_values('game_date', kind='mergesort').reset_index(drop=True)

                # Extract series for dates and the statistic values
                dates = group_df['game_date']
                values = group_df[gen_stat_name]

                # --- Leakage Prevention Mechanism ---
                # 1. Shift dates by 1 to get the date of the *previous* game in the sorted group.
                shifted_date_ints = dates.view('int64').shift(1)
                # 2. Calculate the maximum *shifted* date within the rolling window ending at the previous game.
                #    Using min_periods=1 ensures we get a max date even for the first few games.
                max_shifted_date_ints_in_window = shifted_date_ints.rolling(window, min_periods=1).max()
                # 3. Convert the max timestamp back to datetime. This represents the latest game date *included* in the rolling window.
                latest_date_in_window = pd.to_datetime(max_shifted_date_ints_in_window)

                # --- Calculate Rolling Mean and Std ---
                # Apply rolling calculation on the original values
                # **CRITICAL**: Apply .shift(1) *after* .rolling() to exclude the current row's value.
                mean_series = values.rolling(window, min_periods=min_periods).mean().shift(1)
                std_series = values.rolling(window, min_periods=min_periods).std().clip(lower=0).shift(1) # Clip stddev >= 0

                # Get default values from the DEFAULTS dictionary
                default_mean = DEFAULTS.get(gen_stat_name, 0.0)
                default_std = DEFAULTS.get(f"{gen_stat_name}_std", 0.0) # Assumes std defaults exist (e.g., 'score_for_std')

                # --- Detect and Handle Residual Leakage ---
                # Check if the latest date included in the window is >= the current game's date.
                # This should NOT happen with the stable sort + shift(1) logic, but acts as a final check.
                leak_mask = latest_date_in_window >= dates
                if leak_mask.any():
                    num_leaks = int(leak_mask.sum())
                    total_leaks_detected += num_leaks
                    # If leakage is detected, overwrite the calculated values with defaults for affected rows
                    mean_series[leak_mask] = default_mean
                    std_series[leak_mask] = default_std
                    logger.warning(f"Potential rolling leakage detected and reset for {num_leaks} rows (Team: {group_df['team_norm'].iloc[0]}, Stat: {gen_stat_name}, Window: {window})")


                # Fill any remaining NaNs (e.g., at the start of a team's history before min_periods is met) with defaults
                mean_series = mean_series.fillna(default_mean)
                std_series = std_series.fillna(default_std)

                # Return a DataFrame with the calculated rolling stats for this group
                return pd.DataFrame({
                    'team_norm': group_df['team_norm'],
                    'game_id': group_df['game_id'],
                    # Use generate_rolling_column_name for consistent naming
                    generate_rolling_column_name('', gen_stat_name, 'mean', window): mean_series,
                    generate_rolling_column_name('', gen_stat_name, 'std', window): std_series,
                }, index=group_df.index)

            # Apply the calculation function to each group (team)
            # Use try-except block for robustness during apply
            try:
                 rolled_stats_df = grouped_by_team.apply(calculate_rolling_for_group).reset_index(drop=True)
            except Exception as apply_err:
                 logger.error(f"Error applying rolling calculation for stat '{gen_stat_name}', window {window}: {apply_err}", exc_info=debug)
                 # Create an empty df with expected columns if apply fails, to avoid breaking concatenation
                 rolled_stats_df = pd.DataFrame(columns=[
                      'team_norm', 'game_id',
                      generate_rolling_column_name('', gen_stat_name, 'mean', window),
                      generate_rolling_column_name('', gen_stat_name, 'std', window)
                 ])


            # Log if any leaks were detected and reset during the apply step
            if total_leaks_detected > 0:
                logger.warning(
                    f"Total rolling leakage reset count for '{gen_stat_name}' w={window}: {total_leaks_detected} values"
                )

            # Append the results for this stat/window to the list
            all_rolled_stats.append(rolled_stats_df)

    # --- 4. Reassemble and Merge Back ---
    logger.debug("Reassembling and merging rolling features back to the original DataFrame structure...")
    if not all_rolled_stats:
        logger.warning("No rolling statistics were successfully calculated. Returning DataFrame without rolling features.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return local # Return the prepared 'local' df

    # Concatenate all calculated rolling stats DataFrames
    # Use merge to combine results based on game_id and team_norm, handling potential duplicates
    final_roll_df = all_rolled_stats[0] # Start with the first df
    for next_roll_df in all_rolled_stats[1:]:
         # Ensure unique keys before merging
         next_roll_df = next_roll_df.loc[:, ~next_roll_df.columns.duplicated()]
         final_roll_df = pd.merge(
              final_roll_df,
              next_roll_df,
              on=['game_id', 'team_norm'],
              how='outer', # Use outer merge to keep all rows
              suffixes=('', '_dup') # Add suffix for duplicate columns from merge
         )
         # Drop duplicate columns created by merge
         final_roll_df = final_roll_df.drop(columns=[col for col in final_roll_df if '_dup' in col], errors='ignore')


    # Prepare the original DataFrame ('local') and the final rolling stats ('final_roll_df') for merging
    out_df = local.copy() # Use the prepared 'local' df as the base for merging
    # Create merge keys for home and away teams in the output df
    out_df['merge_key_home'] = out_df['game_id'] + "_" + out_df['home_norm']
    out_df['merge_key_away'] = out_df['game_id'] + "_" + out_df['away_norm']
    # Create merge key in the rolling stats df
    final_roll_df['merge_key'] = final_roll_df['game_id'] + "_" + final_roll_df['team_norm']

    # Identify the columns containing the calculated rolling stats (they start with 'rolling_')
    rolling_stat_cols = [col for col in final_roll_df if col.startswith('rolling_')]
    # Prepare the rolling data for merging, keeping only the merge key and stat columns
    # Drop duplicates on merge_key to ensure one row per team per game
    merge_ready_rolling_df = final_roll_df[['merge_key'] + rolling_stat_cols].drop_duplicates('merge_key')

    # Merge rolling stats for the home team
    home_rename_map = {col: generate_rolling_column_name('home', *col.split('_')[1:]) for col in rolling_stat_cols} # e.g., rolling_score_for_mean_5 -> home_rolling_score_for_mean_5
    out_df = pd.merge(
        out_df,
        merge_ready_rolling_df.rename(columns=home_rename_map),
        how='left',
        left_on='merge_key_home',
        right_on='merge_key'
    ).drop(columns=['merge_key'], errors='ignore') # Drop the merge key from the rolling df side

    # Merge rolling stats for the away team
    away_rename_map = {col: generate_rolling_column_name('away', *col.split('_')[1:]) for col in rolling_stat_cols} # e.g., rolling_score_for_mean_5 -> away_rolling_score_for_mean_5
    out_df = pd.merge(
        out_df,
        merge_ready_rolling_df.rename(columns=away_rename_map), # Use the same merge_ready_rolling_df
        how='left',
        left_on='merge_key_away',
        right_on='merge_key'
    ).drop(columns=['merge_key'], errors='ignore') # Drop the merge key

    # Clean up merge keys and normalized team names from the final DataFrame
    out_df = out_df.drop(columns=['merge_key_home', 'merge_key_away', 'home_norm', 'away_norm'], errors='ignore')

    # --- 5. Final NaN Filling ---
    # Fill any remaining NaNs in the newly added rolling columns using defaults
    # This might happen if a team had no historical data for a merge
    logger.debug("Filling any remaining NaNs in rolling columns with defaults...")
    newly_added_rolling_cols = list(home_rename_map.values()) + list(away_rename_map.values())
    for col in newly_added_rolling_cols:
        if col in out_df.columns:
            # Determine the base stat name and type (mean/std) to find the correct default
            parts = col.split('_') # e.g., ['home', 'rolling', 'score', 'for', 'mean', '5']
            base_stat_key = "_".join(parts[2:-2]) # e.g., 'score_for'
            stat_type = parts[-2] # e.g., 'mean' or 'std'

            # Construct the key for the DEFAULTS dictionary
            default_key = f"{base_stat_key}_std" if stat_type == 'std' else base_stat_key
            default_val = DEFAULTS.get(default_key, 0.0) # Default to 0.0 if specific default not found

            # Fill NaNs and ensure correct type
            out_df[col] = out_df[col].fillna(default_val)
            if stat_type == 'std':
                 # Ensure std deviation is not negative
                 out_df[col] = pd.to_numeric(out_df[col], errors='coerce').fillna(default_val).clip(lower=0.0)
            else:
                 out_df[col] = pd.to_numeric(out_df[col], errors='coerce').fillna(default_val)
        else:
             # This case indicates an issue with the merge or renaming logic
             logger.warning(f"Expected rolling column '{col}' not found after merges. Skipping final fill.")


    logger.debug("Finished adding rolling features.")
    logger.debug("rolling.transform: done, output shape=%s", out_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return out_df