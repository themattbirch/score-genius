# backend/features/advanced.py

from __future__ import annotations
import logging
from typing import Any # Keep Any for DEFAULTS typing if needed

import numpy as np
import pandas as pd

# Import necessary components from the utils module
from .utils import safe_divide, DEFAULTS # Assuming DEFAULTS and safe_divide are in utils.py

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
# EPSILON = 1e-6 # Moved to utils.py if needed globally, otherwise define here if only used here

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Calculates advanced metrics: eFG%, FT%, Reb%, Possessions, Pace, Ratings, TOV%.

    Args:
        df: Input DataFrame containing necessary base stats (scores, fg, ft, reb, tov, ot).
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added advanced feature columns.
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for advanced.transform")

    if df is None or df.empty:
        logger.warning("advanced.transform: Input DataFrame is empty. Returning empty DataFrame.")
        # Restore logger level
        if debug: logger.setLevel(current_level)
        return pd.DataFrame() # Return empty DF to avoid errors downstream

    logger.debug("Integrating advanced metrics (Revised Pace/Poss/Ratings)...")
    # Work on a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # --- 1. Ensure Required Base Stats Exist & Are Numeric ---
    # Define the essential base statistics needed for calculations
    stat_cols = [
        'home_score', 'away_score', 'home_fg_made', 'home_fg_attempted',
        'away_fg_made', 'away_fg_attempted', 'home_3pm', 'home_3pa',
        'away_3pm', 'away_3pa', 'home_ft_made', 'home_ft_attempted',
        'away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb',
        'home_total_reb', 'away_off_reb', 'away_def_reb', 'away_total_reb',
        'home_turnovers', 'away_turnovers', 'home_ot', 'away_ot'
    ]
    # Check for missing columns and fill with 0
    for col in stat_cols:
        if col not in result_df.columns:
            logger.warning(f"Advanced Metrics: Column '{col}' not found. Adding with default value 0.")
            result_df[col] = 0
        # Convert to numeric, coercing errors and filling NaNs with 0
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    # --- 2. Calculate Basic Shooting / Rebounding / FT Rates ---
    logger.debug("Calculating basic rates (eFG%, FT%, Reb%)...")
    # Use safe_divide imported from utils and DEFAULTS constant
    result_df['home_efg_pct'] = safe_divide(result_df['home_fg_made'] + 0.5 * result_df['home_3pm'], result_df['home_fg_attempted'], DEFAULTS['efg_pct'])
    result_df['away_efg_pct'] = safe_divide(result_df['away_fg_made'] + 0.5 * result_df['away_3pm'], result_df['away_fg_attempted'], DEFAULTS['efg_pct'])
    result_df['home_ft_rate'] = safe_divide(result_df['home_ft_attempted'], result_df['home_fg_attempted'], DEFAULTS['ft_rate'])
    result_df['away_ft_rate'] = safe_divide(result_df['away_ft_attempted'], result_df['away_fg_attempted'], DEFAULTS['ft_rate'])
    result_df['home_oreb_pct'] = safe_divide(result_df['home_off_reb'], result_df['home_off_reb'] + result_df['away_def_reb'], DEFAULTS['oreb_pct'])
    result_df['away_dreb_pct'] = safe_divide(result_df['away_def_reb'], result_df['away_def_reb'] + result_df['home_off_reb'], DEFAULTS['dreb_pct'])
    result_df['away_oreb_pct'] = safe_divide(result_df['away_off_reb'], result_df['away_off_reb'] + result_df['home_def_reb'], DEFAULTS['oreb_pct'])
    result_df['home_dreb_pct'] = safe_divide(result_df['home_def_reb'], result_df['home_def_reb'] + result_df['away_off_reb'], DEFAULTS['dreb_pct'])
    result_df['home_trb_pct'] = safe_divide(result_df['home_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], DEFAULTS['trb_pct'])
    result_df['away_trb_pct'] = safe_divide(result_df['away_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], DEFAULTS['trb_pct'])

    # --- 3. Calculate Pace and Possessions ---
    logger.debug("Calculating raw possessions, average possessions, and Pace...")
    # Calculate Raw Possessions per team using standard formula
    home_poss_raw = (
        result_df['home_fg_attempted']
        + 0.44 * result_df['home_ft_attempted']
        - result_df['home_off_reb']
        + result_df['home_turnovers']
    )
    away_poss_raw = (
        result_df['away_fg_attempted']
        + 0.44 * result_df['away_ft_attempted']
        - result_df['away_off_reb']
        + result_df['away_turnovers']
    )
    result_df['home_poss_raw'] = home_poss_raw
    result_df['away_poss_raw'] = away_poss_raw
    # Create denominators for safe division, replacing 0 with NaN
    home_poss_safe_denom = home_poss_raw.replace(0, np.nan)
    away_poss_safe_denom = away_poss_raw.replace(0, np.nan)

    # Calculate Average Possessions per Team (for Pace calculation)
    avg_poss_per_team = 0.5 * (home_poss_raw + away_poss_raw)

    # Clip and Store Average Possessions Estimate
    possessions_est = avg_poss_per_team.clip(lower=50, upper=120) # Apply reasonable clipping
    result_df['possessions_est'] = possessions_est.fillna(DEFAULTS['estimated_possessions'])

    # Calculate Game Minutes Played, accounting for overtime
    num_ot = np.maximum(result_df.get('home_ot', 0), result_df.get('away_ot', 0)).clip(lower=0)
    game_minutes_calc = 48.0 + num_ot * 5.0
    result_df['game_minutes_played'] = np.maximum(48.0, game_minutes_calc) # Ensure minimum 48 mins

    # Calculate Game Pace (Pace per 48 minutes, using average possessions estimate)
    result_df['game_pace'] = safe_divide(
        result_df['possessions_est'] * 48.0,
        result_df['game_minutes_played'],
        DEFAULTS['pace']
    )
    # Assign the same game pace to both teams
    result_df['home_pace'] = result_df['game_pace']
    result_df['away_pace'] = result_df['game_pace']

    # --- 4. Calculate Efficiency (Ratings) and TOV% using RAW possessions ---
    logger.debug("Calculating ratings and TOV% using raw possessions...")
    # Calculate Offensive Ratings
    result_df['home_offensive_rating'] = safe_divide(result_df['home_score'] * 100, home_poss_safe_denom, DEFAULTS['offensive_rating'])
    result_df['away_offensive_rating'] = safe_divide(result_df['away_score'] * 100, away_poss_safe_denom, DEFAULTS['offensive_rating'])
    # Defensive Rating is simply the opponent's Offensive Rating
    result_df['home_defensive_rating'] = result_df['away_offensive_rating']
    result_df['away_defensive_rating'] = result_df['home_offensive_rating']

    # Apply clipping to ratings to keep them within reasonable bounds
    rating_cols = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']
    for col in rating_cols:
        # Use the generic default rating if specific one isn't found (shouldn't happen with comprehensive DEFAULTS)
        default_key = col.replace('home_', '').replace('away_', '')
        result_df[col] = result_df[col].clip(lower=70, upper=150).fillna(DEFAULTS.get(default_key, 115.0))

    # Calculate Net Ratings
    result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
    result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']

    # Calculate TOV% using raw possessions
    result_df['home_tov_rate'] = safe_divide(result_df['home_turnovers'] * 100, home_poss_safe_denom, DEFAULTS['tov_rate'])
    result_df['away_tov_rate'] = safe_divide(result_df['away_turnovers'] * 100, away_poss_safe_denom, DEFAULTS['tov_rate'])

    # Apply clipping to TOV%
    result_df['home_tov_rate'] = result_df['home_tov_rate'].clip(lower=5, upper=25).fillna(DEFAULTS['tov_rate'])
    result_df['away_tov_rate'] = result_df['away_tov_rate'].clip(lower=5, upper=25).fillna(DEFAULTS['tov_rate'])

    # --- 5. Calculate Differentials & Convenience Columns ---
    logger.debug("Calculating differential features...")
    result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']
    result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
    result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
    result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
    result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct']
    result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']
    # 'pace_differential' is intentionally omitted as home_pace == away_pace
    result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate'] # Note: Away - Home

    # Add total score and point diff if they don't exist (useful for analysis)
    if 'total_score' not in result_df.columns:
        result_df['total_score'] = result_df['home_score'] + result_df['away_score']
    if 'point_diff' not in result_df.columns:
        result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

    # --- 6. Clean Up ---
    # Remove intermediate columns if they exist (e.g., from older versions)
    result_df = result_df.drop(columns=['home_possessions', 'away_possessions'], errors='ignore')

    logger.debug("Finished integrating advanced features with revised Pace/Poss/Ratings.")
    logger.debug("advanced.transform: done, output shape=%s", result_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return result_df