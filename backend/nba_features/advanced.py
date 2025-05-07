# backend/nba_features/advanced.py

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
        df: Input DataFrame containing necessary base stats.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added advanced feature columns.
    """
    # Adjust logging level
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for advanced.transform")

    if df is None or df.empty:
        logger.warning("advanced.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug:
            logger.setLevel(orig_level)
        return pd.DataFrame()

    result_df = df.copy()

    # --- 1. Ensure Required Base Stats Exist & Are Numeric ---
    stat_cols = [
        'home_score', 'away_score',
        'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
        'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
        'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
        'home_off_reb', 'home_def_reb', 'home_total_reb',
        'away_off_reb', 'away_def_reb', 'away_total_reb',
        'home_turnovers', 'away_turnovers', 'home_ot', 'away_ot'
    ]
    for col in stat_cols:
        if col not in result_df.columns:
            logger.warning(f"Advanced Metrics: Column '{col}' not found. Adding with default value 0.")
            result_df[col] = 0
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    # --- 2. Basic Shooting/Rebounding/FT Rates ---
    result_df['home_efg_pct'] = safe_divide(
        result_df['home_fg_made'] + 0.5 * result_df['home_3pm'],
        result_df['home_fg_attempted'],
        DEFAULTS['efg_pct']
    )
    result_df['away_efg_pct'] = safe_divide(
        result_df['away_fg_made'] + 0.5 * result_df['away_3pm'],
        result_df['away_fg_attempted'],
        DEFAULTS['efg_pct']
    )
    result_df['home_ft_rate'] = safe_divide(
        result_df['home_ft_attempted'],
        result_df['home_fg_attempted'],
        DEFAULTS['ft_rate']
    )
    result_df['away_ft_rate'] = safe_divide(
        result_df['away_ft_attempted'],
        result_df['away_fg_attempted'],
        DEFAULTS['ft_rate']
    )
    result_df['home_oreb_pct'] = safe_divide(
        result_df['home_off_reb'],
        result_df['home_off_reb'] + result_df['away_def_reb'],
        DEFAULTS['oreb_pct']
    )
    result_df['away_dreb_pct'] = safe_divide(
        result_df['away_def_reb'],
        result_df['away_def_reb'] + result_df['home_off_reb'],
        DEFAULTS['dreb_pct']
    )
    result_df['away_oreb_pct'] = safe_divide(
        result_df['away_off_reb'],
        result_df['away_off_reb'] + result_df['home_def_reb'],
        DEFAULTS['oreb_pct']
    )
    result_df['home_dreb_pct'] = safe_divide(
        result_df['home_def_reb'],
        result_df['home_def_reb'] + result_df['away_off_reb'],
        DEFAULTS['dreb_pct']
    )
    result_df['home_trb_pct'] = safe_divide(
        result_df['home_total_reb'],
        result_df['home_total_reb'] + result_df['away_total_reb'],
        DEFAULTS['trb_pct']
    )
    result_df['away_trb_pct'] = safe_divide(
        result_df['away_total_reb'],
        result_df['home_total_reb'] + result_df['away_total_reb'],
        DEFAULTS['trb_pct']
    )

    # --- 3. Possessions & Pace ---
    # Raw possessions
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

    # Estimated possessions
    avg_poss = 0.5 * (home_poss_raw + away_poss_raw)
    poss_est = avg_poss.clip(lower=50, upper=120).fillna(DEFAULTS['estimated_possessions'])
    result_df['possessions_est'] = poss_est

    # Game minutes
    num_ot = np.maximum(result_df['home_ot'], result_df['away_ot'])
    result_df['game_minutes_played'] = np.maximum(48.0, 48.0 + num_ot * 5.0)

    # Pace
    result_df['game_pace'] = safe_divide(
        result_df['possessions_est'] * 48.0,
        result_df['game_minutes_played'],
        DEFAULTS['pace']
    )
    result_df['home_pace'] = result_df['game_pace']
    result_df['away_pace'] = result_df['game_pace']

    # --- 4. Ratings & Turnover Rate ---
    result_df['home_offensive_rating'] = safe_divide(
        result_df['home_score'] * 100,
        home_poss_raw.replace(0, np.nan),
        DEFAULTS['offensive_rating']
    )
    result_df['away_offensive_rating'] = safe_divide(
        result_df['away_score'] * 100,
        away_poss_raw.replace(0, np.nan),
        DEFAULTS['offensive_rating']
    )
    result_df['home_defensive_rating'] = result_df['away_offensive_rating']
    result_df['away_defensive_rating'] = result_df['home_offensive_rating']

    # Clip ratings
    for col in ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']:
        base_key = col.replace('home_', '').replace('away_', '')
        result_df[col] = result_df[col].clip(lower=70, upper=150).fillna(DEFAULTS.get(base_key, 115.0))

    # Net rating
    result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
    result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']

    # Turnover rate
    result_df['home_tov_rate'] = safe_divide(
        result_df['home_turnovers'] * 100,
        home_poss_raw.replace(0, np.nan),
        DEFAULTS['tov_rate']
    ).clip(lower=5, upper=25).fillna(DEFAULTS['tov_rate'])
    result_df['away_tov_rate'] = safe_divide(
        result_df['away_turnovers'] * 100,
        away_poss_raw.replace(0, np.nan),
        DEFAULTS['tov_rate']
    ).clip(lower=5, upper=25).fillna(DEFAULTS['tov_rate'])

    # --- 5. Differentials ---
    result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']
    result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
    result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
    result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
    result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct']
    result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']
    result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate']

    # Total score and point diff
    if 'total_score' not in result_df.columns:
        result_df['total_score'] = result_df['home_score'] + result_df['away_score']
    if 'point_diff' not in result_df.columns:
        result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

    logger.debug("advanced.transform finished; output shape=%s", result_df.shape)
    if debug:
        logger.setLevel(orig_level)
    return result_df
