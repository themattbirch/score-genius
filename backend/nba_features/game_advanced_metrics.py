# backend/nba_features/game_advanced_metrics.py
"""
Calculates single-game advanced metrics from box score data.

Features include: eFG%, FT Rate, Rebounding Percentages, Possessions,
Pace, Offensive/Defensive/Net Ratings, and Turnover Percentage for each team
in a game.
"""

from __future__ import annotations
import logging
from typing import Any 

import numpy as np
import pandas as pd

from .utils import safe_divide, DEFAULTS 

# --- Logger Configuration ---
# Remove basicConfig from here; should be in main application entry point
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
# )
logger = logging.getLogger(__name__)

__all__ = ["transform"]

# No EPSILON constant needed here unless used by this specific file's logic

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Calculates advanced metrics for each game: eFG%, FT%, Reb%, Possessions, Pace, Ratings, TOV%.

    Args:
        df: Input DataFrame containing necessary base stats (scores, fg, ft, reb, tov, ot).
            Expected columns are listed in 'stat_cols'.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added single-game advanced feature columns.
    """
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for game_advanced_metrics.transform")

    if df is None or df.empty:
        logger.warning("game_advanced_metrics.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level)
        return pd.DataFrame() 

    logger.debug("Calculating single-game advanced metrics (e.g., Pace, Possessions, Ratings)...")
    result_df = df.copy()

    # --- 1. Ensure Required Base Stats Exist & Are Numeric ---
    stat_cols = [
        'home_score', 'away_score', 'home_fg_made', 'home_fg_attempted',
        'away_fg_made', 'away_fg_attempted', 'home_3pm', 'home_3pa', # Assuming 3pa might be needed by some models if not eFG
        'away_3pm', 'away_3pa', 'home_ft_made', 'home_ft_attempted',
        'away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb',
        'home_total_reb', 'away_off_reb', 'away_def_reb', 'away_total_reb',
        'home_turnovers', 'away_turnovers', 'home_ot', 'away_ot'
    ]
    for col in stat_cols:
        if col not in result_df.columns:
            logger.warning(f"Game Advanced Metrics: Column '{col}' not found. Adding with default value 0.")
            result_df[col] = 0
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    # --- 2. Calculate Basic Shooting / Rebounding / FT Rates ---
    # (This section is already excellent as you provided it)
    result_df['home_efg_pct'] = safe_divide(result_df['home_fg_made'] + 0.5 * result_df['home_3pm'], result_df['home_fg_attempted'], DEFAULTS.get('efg_pct', 0.54))
    result_df['away_efg_pct'] = safe_divide(result_df['away_fg_made'] + 0.5 * result_df['away_3pm'], result_df['away_fg_attempted'], DEFAULTS.get('efg_pct', 0.54))
    result_df['home_ft_rate'] = safe_divide(result_df['home_ft_attempted'], result_df['home_fg_attempted'], DEFAULTS.get('ft_rate', 0.20))
    result_df['away_ft_rate'] = safe_divide(result_df['away_ft_attempted'], result_df['away_fg_attempted'], DEFAULTS.get('ft_rate', 0.20))
    result_df['home_oreb_pct'] = safe_divide(result_df['home_off_reb'], result_df['home_off_reb'] + result_df['away_def_reb'], DEFAULTS.get('oreb_pct', 0.23))
    result_df['away_dreb_pct'] = safe_divide(result_df['away_def_reb'], result_df['away_def_reb'] + result_df['home_off_reb'], DEFAULTS.get('dreb_pct', 0.77))
    result_df['away_oreb_pct'] = safe_divide(result_df['away_off_reb'], result_df['away_off_reb'] + result_df['home_def_reb'], DEFAULTS.get('oreb_pct', 0.23))
    result_df['home_dreb_pct'] = safe_divide(result_df['home_def_reb'], result_df['home_def_reb'] + result_df['away_off_reb'], DEFAULTS.get('dreb_pct', 0.77))
    result_df['home_trb_pct'] = safe_divide(result_df['home_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], DEFAULTS.get('trb_pct', 0.50))
    result_df['away_trb_pct'] = safe_divide(result_df['away_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], DEFAULTS.get('trb_pct', 0.50))


    # --- 3. Calculate Pace and Possessions ---
    # (This section is already excellent)
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
    result_df['home_poss_raw'] = home_poss_raw # Keep if useful, or drop later
    result_df['away_poss_raw'] = away_poss_raw # Keep if useful, or drop later
    
    home_poss_safe_denom = home_poss_raw.replace(0, np.nan)
    away_poss_safe_denom = away_poss_raw.replace(0, np.nan)

    avg_poss_per_team = 0.5 * (home_poss_raw + away_poss_raw)
    possessions_est = avg_poss_per_team.clip(lower=DEFAULTS.get('min_possessions', 70), upper=DEFAULTS.get('max_possessions', 120))
    result_df['possessions_est'] = possessions_est.fillna(DEFAULTS.get('estimated_possessions', 95.0))

    num_ot = np.maximum(result_df.get('home_ot', 0), result_df.get('away_ot', 0)).clip(lower=0)
    game_minutes_calc = 48.0 + num_ot * 5.0
    result_df['game_minutes_played'] = np.maximum(48.0, game_minutes_calc)

    result_df['game_pace'] = safe_divide(
        result_df['possessions_est'] * 48.0,
        result_df['game_minutes_played'],
        DEFAULTS.get('pace', 100.0)
    )
    result_df['home_pace'] = result_df['game_pace']
    result_df['away_pace'] = result_df['game_pace']

    # --- 4. Calculate Efficiency (Ratings) and TOV% using RAW possessions ---
    # (This section is already excellent)
    result_df['home_offensive_rating'] = safe_divide(result_df['home_score'] * 100, home_poss_safe_denom, DEFAULTS.get('offensive_rating', 115.0))
    result_df['away_offensive_rating'] = safe_divide(result_df['away_score'] * 100, away_poss_safe_denom, DEFAULTS.get('offensive_rating', 115.0))
    result_df['home_defensive_rating'] = result_df['away_offensive_rating']
    result_df['away_defensive_rating'] = result_df['home_offensive_rating']

    rating_cols = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']
    for col in rating_cols:
        default_key = col.replace('home_', '').replace('away_', '')
        default_rating_val = DEFAULTS.get(default_key, DEFAULTS.get('offensive_rating',115.0)) # Fallback to generic offensive_rating if specific (defensive) not found
        result_df[col] = result_df[col].clip(lower=DEFAULTS.get('min_rating', 70), upper=DEFAULTS.get('max_rating', 150)).fillna(default_rating_val)

    result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
    result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']

    result_df['home_tov_rate'] = safe_divide(result_df['home_turnovers'] * 100, home_poss_safe_denom, DEFAULTS.get('tov_rate', 13.0))
    result_df['away_tov_rate'] = safe_divide(result_df['away_turnovers'] * 100, away_poss_safe_denom, DEFAULTS.get('tov_rate', 13.0))
    
    result_df['home_tov_rate'] = result_df['home_tov_rate'].clip(lower=DEFAULTS.get('min_tov_rate', 5.0), upper=DEFAULTS.get('max_tov_rate', 25.0)).fillna(DEFAULTS.get('tov_rate', 13.0))
    result_df['away_tov_rate'] = result_df['away_tov_rate'].clip(lower=DEFAULTS.get('min_tov_rate', 5.0), upper=DEFAULTS.get('max_tov_rate', 25.0)).fillna(DEFAULTS.get('tov_rate', 13.0))


    # --- 5. Calculate Differentials & Convenience Columns ---
    # (This section is already excellent)
    result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']
    result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
    result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
    result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
    result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct'] # Note: home_dreb_pct uses away_off_reb
    result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']
    result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate']

    if 'total_score' not in result_df.columns:
        result_df['total_score'] = result_df['home_score'] + result_df['away_score']
    if 'point_diff' not in result_df.columns:
        result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

    # --- 6. Clean Up ---
    cols_to_drop_intermediate = ['home_possessions', 'away_possessions'] # From potential older versions
    # Add home_poss_raw, away_poss_raw if they are not desired final features
    # cols_to_drop_intermediate.extend(['home_poss_raw', 'away_poss_raw']) 
    result_df = result_df.drop(columns=[c for c in cols_to_drop_intermediate if c in result_df.columns], errors='ignore')

    logger.debug("Finished calculating single-game advanced metrics.")
    if debug: logger.setLevel(current_level)

    return result_df