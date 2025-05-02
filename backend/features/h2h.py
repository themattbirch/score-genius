# backend/features/h2h.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List # Keep Any for DEFAULTS typing

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
# Define placeholder columns at the module level for consistency
H2H_PLACEHOLDER_COLS: List[str] = [
    'matchup_num_games', 'matchup_avg_point_diff', 'matchup_home_win_pct',
    'matchup_avg_total_score', 'matchup_avg_home_score',
    'matchup_avg_away_score', 'matchup_last_date', 'matchup_streak'
]
# EPSILON = 1e-6 # Moved to utils.py if needed globally

# -- Helper Function --

def _get_matchup_history_single(
    *, # Enforce keyword arguments for clarity
    home_team_norm: str,
    away_team_norm: str,
    historical_subset: pd.DataFrame,
    max_games: int = 7, # Default lookback window
    current_game_date: Optional[pd.Timestamp] = None,
    loop_index: Optional[int] = None, # Optional index for debugging logs
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Calculates Head-to-Head (H2H) statistics for one specific upcoming game,
    based on a provided subset of historical games between the two teams involved.
    Stats are calculated from the perspective of the `home_team_norm`.

    Args:
        home_team_norm: Normalized name of the home team for the upcoming game.
        away_team_norm: Normalized name of the away team for the upcoming game.
        historical_subset: DataFrame containing ONLY past games between these two teams.
        max_games: The maximum number of most recent historical games to consider.
        current_game_date: The date of the upcoming game (used to filter history).
        loop_index: Optional original index of the game row (for debug logging).
        debug: Flag to enable detailed debug logging.

    Returns:
        A dictionary containing calculated H2H statistics (keys match H2H_PLACEHOLDER_COLS).
    """
    # Build default result dictionary using placeholder keys and DEFAULTS values
    default_result: Dict[str, Any] = {}
    for col in H2H_PLACEHOLDER_COLS:
        # Special handling for date column default
        if col == 'matchup_last_date':
            default_result[col] = pd.NaT
        else:
            # Extract base key (e.g., 'avg_point_diff' from 'matchup_avg_point_diff')
            base_key = col.replace('matchup_', '')
            default_result[col] = DEFAULTS.get(base_key, 0.0) # Default to 0.0 if key not in DEFAULTS

    # --- Input Validation ---
    if historical_subset is None or historical_subset.empty or max_games <= 0 or pd.isna(current_game_date):
        if debug: logger.debug(f"H2H Helper[{loop_index}]: Returning defaults due to empty history, max_games<=0, or invalid date.")
        return default_result

    # --- Filter and Prepare Historical Data ---
    # (Filter logic moved to main transform function before calling this helper)
    # Assume historical_subset is already filtered for past games of the correct matchup

    if historical_subset.empty: # Check again after potential filtering in caller
        if debug: logger.debug(f"H2H Helper[{loop_index}]: historical_subset is empty.")
        return default_result

    # Select the most recent 'max_games' from the past encounters
    # Sort by date descending to get the most recent games first
    recent_matchups = historical_subset.sort_values('game_date', ascending=False).head(max_games)
    if recent_matchups.empty: # Should not happen if historical_subset wasn't empty, but safety check
         if debug: logger.debug(f"H2H Helper[{loop_index}]: No recent games found after head({max_games}).")
         return default_result

    # Ensure score columns are numeric and drop rows where conversion fails
    recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce')
    recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
    # Also ensure normalized team names exist for comparison
    recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score', 'home_team_norm', 'away_team_norm'])
    if recent_matchups.empty:
        if debug: logger.debug(f"H2H Helper[{loop_index}]: No valid scores/teams in recent matchups after dropna.")
        return default_result

    # --- Calculate H2H Statistics ---
    # Initialize lists/variables to store results
    diffs_list: List[float] = []           # Point difference from home_team_norm's perspective
    total_scores_list: List[float] = []    # Total score (home + away)
    home_persp_scores: List[float] = []    # Score of home_team_norm in each game
    away_persp_scores: List[float] = []    # Score of away_team_norm in each game
    home_persp_wins: int = 0               # Count of wins for home_team_norm
    current_streak: int = 0                # Current H2H winning/losing streak for home_team_norm
    last_winner_norm: Optional[str] = None # Tracks the winner of the previous game in the loop

    # Iterate through the recent matchups *chronologically* (oldest to newest) to calculate streak correctly
    for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
        h_score = game['home_score']
        a_score = game['away_score']
        g_home_norm = game['home_team_norm'] # Use pre-normalized name
        g_away_norm = game['away_team_norm'] # Use pre-normalized name

        # Determine point difference and winner from the perspective of the *target* home team
        if g_home_norm == home_team_norm: # Target home team played at home in this historical game
            diff = h_score - a_score
            won = h_score > a_score
            home_perspective_score = h_score
            away_perspective_score = a_score
        elif g_away_norm == home_team_norm: # Target home team played away in this historical game
            diff = a_score - h_score # Difference is from their perspective (away_score - home_score)
            won = a_score > h_score
            home_perspective_score = a_score # Their score when they were away
            away_perspective_score = h_score # Opponent's score when they were away
        else:
            # This should not happen if historical_subset is correctly pre-filtered
            logger.warning(f"H2H Helper[{loop_index}]: Team mismatch in historical game! Target: {home_team_norm} vs {away_team_norm}. Game: {g_home_norm} vs {g_away_norm}. Skipping game.")
            continue # Skip this game

        # Append calculated stats for this game
        diffs_list.append(diff)
        total_scores_list.append(h_score + a_score)
        home_persp_scores.append(home_perspective_score)
        away_persp_scores.append(away_perspective_score)
        if won:
            home_persp_wins += 1

        # Calculate streak
        game_winner_norm = home_team_norm if won else away_team_norm
        if last_winner_norm is None: # First game in the sequence
            current_streak = 1 if won else -1
        elif game_winner_norm == last_winner_norm: # Winner is the same as the previous game
            current_streak += (1 if won else -1) # Extend the streak
        else: # Streak broken
            current_streak = 1 if won else -1 # Start new streak
        last_winner_norm = game_winner_norm # Update winner for next iteration

    # --- Aggregate Results ---
    num_valid_games = len(diffs_list)
    if num_valid_games == 0:
        # Should be caught earlier, but final safety check
        if debug: logger.debug(f"H2H Helper[{loop_index}]: No valid games found in loop.")
        return default_result

    # Calculate final aggregated statistics
    final_stats: Dict[str, Any] = {
        'matchup_num_games': num_valid_games,
        'matchup_avg_point_diff': float(np.mean(diffs_list)),
        'matchup_home_win_pct': float(home_persp_wins / num_valid_games),
        'matchup_avg_total_score': float(np.mean(total_scores_list)),
        'matchup_avg_home_score': float(np.mean(home_persp_scores)),
        'matchup_avg_away_score': float(np.mean(away_persp_scores)),
        'matchup_last_date': recent_matchups['game_date'].max(), # Date of the most recent game considered
        'matchup_streak': int(current_streak), # Final streak value
    }

    # Ensure all placeholder columns are present, falling back to defaults if calculation failed for some reason
    for col in H2H_PLACEHOLDER_COLS:
        final_stats.setdefault(col, default_result[col])

    if debug: logger.debug(f"H2H Helper[{loop_index}]: Calculated stats: {final_stats}")
    return final_stats


# -- Main Transformation Function --

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 7, # Default lookback window for H2H stats
    debug: bool = False,
) -> pd.DataFrame:
    """
    Calculates and adds Head-to-Head (H2H) matchup history features to the input DataFrame.

    It looks at past games between the specific home and away teams for each
    game row in `df`, using `historical_df` as the source of past game data.

    Args:
        df: Input DataFrame containing 'game_id', 'game_date', 'home_team', 'away_team'.
        historical_df: DataFrame containing historical game results, including
                       'game_date', 'home_team', 'away_team', 'home_score', 'away_score'.
                       Should ideally contain data prior to the dates in `df`.
        max_games: The maximum number of recent H2H games to consider for stats.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added H2H feature columns (defined in H2H_PLACEHOLDER_COLS).
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for h2h.transform")
        logger.debug(f"H2H lookback window (max_games): {max_games}")

    # --- Input Validation ---
    if df is None or df.empty:
        logger.warning("h2h.transform: Input DataFrame `df` is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return pd.DataFrame()

    result_df = df.copy() # Work on a copy

    # Check for essential columns in the target DataFrame
    essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if not all(col in result_df.columns for col in essential_cols):
         missing_main = set(essential_cols) - set(result_df.columns)
         logger.error(f"Input `df` missing essential columns: {missing_main}. Cannot calculate H2H features.")
         # Fill placeholders with defaults if essential keys are missing
         for col in H2H_PLACEHOLDER_COLS:
             if col not in result_df.columns:
                 default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                 result_df[col] = default_val
         if debug: logger.setLevel(current_level) # Restore logger level
         return result_df

    # Handle case where no historical data is provided
    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: No `historical_df` provided. Filling H2H features with defaults.")
        for col in H2H_PLACEHOLDER_COLS:
            if col not in result_df.columns:
                default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                result_df[col] = default_val
        if debug: logger.setLevel(current_level) # Restore logger level
        return result_df

    # --- Prepare Historical Data ---
    logger.debug("Preparing historical data for H2H lookup...")
    try:
        # Clean and prepare historical data copy
        hist_df = historical_df.dropna(subset=['game_date', 'home_score', 'away_score', 'home_team', 'away_team']).copy()
        hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
        hist_df['home_score'] = pd.to_numeric(hist_df['home_score'], errors='coerce')
        hist_df['away_score'] = pd.to_numeric(hist_df['away_score'], errors='coerce')
        # Drop rows where essential conversions failed
        hist_df = hist_df.dropna(subset=['game_date', 'home_score', 'away_score'])

        if hist_df.empty:
             raise ValueError("Historical DataFrame is empty after cleaning.")

        # Normalize team names and create a matchup key (sorted team names)
        # *** FIX: Ensure .astype(str) is used directly before .map() ***
        hist_df['home_team_norm'] = hist_df['home_team'].astype(str).map(normalize_team_name)
        hist_df['away_team_norm'] = hist_df['away_team'].astype(str).map(normalize_team_name)
        hist_df['matchup_key'] = hist_df.apply(
            lambda r: "_vs_".join(sorted([r['home_team_norm'], r['away_team_norm']])), axis=1
        )
        # Create a lookup dictionary: {matchup_key: sorted_historical_games_df}
        # Sorting by date within each group is crucial for the helper function
        historical_lookup = {
            key: group_df.sort_values('game_date')
            for key, group_df in hist_df.groupby('matchup_key', observed=True)
        }
        logger.debug(f"Created historical lookup with {len(historical_lookup)} matchup keys.")

    except Exception as e_hist_prep:
        logger.error(f"Error preparing historical data: {e_hist_prep}. Filling H2H features with defaults.", exc_info=debug)
        for col in H2H_PLACEHOLDER_COLS:
             if col not in result_df.columns:
                 default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                 result_df[col] = default_val
        if debug: logger.setLevel(current_level) # Restore logger level
        return result_df

    # --- Prepare Target DataFrame ---
    logger.debug("Preparing target DataFrame for H2H calculation...")
    try:
        result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
        # Drop rows with invalid dates in the target df as H2H calc is impossible
        result_df = result_df.dropna(subset=['game_date'])
        if result_df.empty:
             logger.warning("Target DataFrame is empty after dropping invalid dates. Returning empty DataFrame.")
             if debug: logger.setLevel(current_level) # Restore logger level
             return pd.DataFrame()

        # *** FIX: Ensure .astype(str) is used directly before .map() ***
        result_df['home_team_norm'] = result_df['home_team'].astype(str).map(normalize_team_name)
        result_df['away_team_norm'] = result_df['away_team'].astype(str).map(normalize_team_name)
        result_df['matchup_key'] = result_df.apply(
            lambda r: "_vs_".join(sorted([r['home_team_norm'], r['away_team_norm']])), axis=1
        )
    except Exception as e_target_prep:
         logger.error(f"Error preparing target DataFrame: {e_target_prep}. Cannot calculate H2H features.", exc_info=debug)
         # Fill placeholders if target prep fails
         for col in H2H_PLACEHOLDER_COLS:
              if col not in result_df.columns:
                  default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                  result_df[col] = default_val
         if debug: logger.setLevel(current_level) # Restore logger level
         return result_df


    # --- Calculate H2H Features Row by Row ---
    logger.info(f"Calculating H2H features for {len(result_df)} games...")
    h2h_results: List[Dict[str, Any]] = []
    # Iterate through each game in the target DataFrame
    for idx, row in result_df.iterrows():
        # Find the relevant historical games using the matchup key
        matchup_key = row['matchup_key']
        hist_subset_for_game = historical_lookup.get(matchup_key, pd.DataFrame()) # Get df or empty df

        # Filter the subset for games strictly before the current game date
        # Doing the date filter here, just before calling the helper
        if not hist_subset_for_game.empty:
             hist_subset_for_game = hist_subset_for_game[hist_subset_for_game['game_date'] < row['game_date']]

        # Call the helper function to calculate H2H stats for this specific game
        stats = _get_matchup_history_single(
            home_team_norm=row['home_team_norm'],
            away_team_norm=row['away_team_norm'],
            historical_subset=hist_subset_for_game, # Pass the pre-filtered and sorted subset
            max_games=max_games,
            current_game_date=row['game_date'], # Pass current date for context (though filtering is done above now)
            loop_index=idx, # Pass index for logging
            debug=debug,
        )
        h2h_results.append(stats)

    # --- Merge Results Back ---
    logger.debug("Merging calculated H2H features back to the DataFrame...")
    try:
        # Create a DataFrame from the list of calculated H2H stats dictionaries
        h2h_stats_df = pd.DataFrame(h2h_results, index=result_df.index) # Use original index
        # Join the calculated H2H stats back to the result DataFrame
        result_df = result_df.join(h2h_stats_df, how='left')

        # Final check and fill for any missing H2H columns (e.g., if join failed unexpectedly)
        for col in H2H_PLACEHOLDER_COLS:
            if col not in result_df.columns:
                 logger.warning(f"H2H column '{col}' missing after join. Adding with default.")
                 default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                 result_df[col] = default_val
            else:
                 # Fill any NaNs that might remain (e.g., from games with no history)
                 default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                 result_df[col] = result_df[col].fillna(default_val)

            # Ensure correct types for final columns
            if col == 'matchup_last_date':
                 result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
            elif col in ['matchup_num_games', 'matchup_streak']:
                 # Ensure the column exists before trying to access/convert it
                 if col in result_df:
                      result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
            else: # Other stats are floats
                 if col in result_df:
                      result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0).astype(float)


    except Exception as e_merge:
        logger.error(f"Error merging H2H results back: {e_merge}. Filling H2H features with defaults.", exc_info=debug)
        # Fill placeholders if merge fails
        for col in H2H_PLACEHOLDER_COLS:
             if col not in result_df.columns:
                 default_val = pd.NaT if col == 'matchup_last_date' else DEFAULTS.get(col.replace('matchup_',''), 0.0)
                 result_df[col] = default_val

    # --- Clean Up ---
    # Drop intermediate columns
    result_df = result_df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')

    logger.info("Finished adding head-to-head features.")
    logger.debug("h2h.transform: done, output shape=%s", result_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return result_df