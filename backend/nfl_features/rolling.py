# backend/nfl_features/rolling.py
"""Rolling‑window (recent form) feature loader.

-- MODIFIED to use a pre-fetched DataFrame instead of its own Supabase queries --

This module takes a pre-fetched DataFrame of recent-form stats, reshapes it
into home/away columns, applies sensible defaults, and derives differentials.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional

import pandas as pd

from .utils import DEFAULTS, prefix_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Configuration constants (column naming contract with the DB view)
# ---------------------------------------------------------------------------
# Note: The view name is no longer used here, but keeping constants is good practice.
_ROLLING_VIEW: str = "nfl_recent_form" # (formerly mv_nfl_recent_form)
_BASE_FEATURE_COLS: Mapping[str, str] = {
    # DB column → logical key used for defaults & naming
    "rolling_points_for_avg": "points_for_avg",
    "rolling_points_against_avg": "points_against_avg",
    "rolling_yards_per_play_avg": "yards_per_play_avg",
    "rolling_turnover_differential_avg": "turnover_differential_avg",
}
# Derived columns computed locally after fetch
_DERIVED_COLS = {
    "rolling_point_differential_avg": (
        "rolling_points_for_avg",
        "rolling_points_against_avg",
    )
}


def load_rolling_features(
    games: pd.DataFrame,
    *,
    recent_form_df: Optional[pd.DataFrame] = None,
    **kwargs # Absorbs unused kwargs from the engine
) -> pd.DataFrame:
    """
    Attaches recent-form metrics to a games DataFrame using a pre-fetched
    DataFrame of stats.
    """
    if games.empty:
        return pd.DataFrame()

    if recent_form_df is None or recent_form_df.empty:
        logger.warning("rolling: No recent_form_df provided. Cannot add rolling features.")
        return games.copy()

    games_df = games.copy()
    stats_df = recent_form_df.copy()

    # --- ADDED: Robustly ensure normalized team columns exist ---
    from .utils import normalize_team_name
    if 'home_team_norm' not in games_df.columns and 'home_team_id' in games_df.columns:
        games_df['home_team_norm'] = games_df['home_team_id'].apply(normalize_team_name)
    if 'away_team_norm' not in games_df.columns and 'away_team_id' in games_df.columns:
        games_df['away_team_norm'] = games_df['away_team_id'].apply(normalize_team_name)
    
    # Ensure the columns now exist before proceeding
    required_cols = ['game_id', 'home_team_norm', 'away_team_norm']
    if not all(col in games_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in games_df.columns]
        logger.error(f"rolling.py: DataFrame is missing critical columns after normalization: {missing}")
        return games.copy()
    
    if 'team_norm' not in stats_df.columns and 'team_id' in stats_df.columns:
        stats_df['team_norm'] = stats_df['team_id'].apply(normalize_team_name)
    # --- END ADDED BLOCK ---

    # Compute derived columns
    for new_col, (num_col, den_col) in _DERIVED_COLS.items():
        if num_col in stats_df.columns and den_col in stats_df.columns:
            stats_df[new_col] = stats_df[num_col] - stats_df[den_col]

    # Merge for home team
    home_stats = stats_df.add_prefix('home_')
    result = pd.merge(
        games_df,
        home_stats,
        left_on='home_team_norm',
        right_on='home_team_norm',
        how='left'
    )

    # Merge for away team
    away_stats = stats_df.add_prefix('away_')
    result = pd.merge(
        result,
        away_stats,
        left_on='away_team_norm',
        right_on='away_team_norm',
        how='left',
        suffixes=('', '_away_dup')
    )
    result = result.loc[:, ~result.columns.str.endswith('_away_dup')]
    
    # Fill missing values with defaults
    for db_col, generic_key in _BASE_FEATURE_COLS.items():
        for side in ['home', 'away']:
            col = f"{side}_{db_col}"
            default_val = DEFAULTS.get(generic_key, 0.0)
            if col in result.columns:
                result[col].fillna(default_val, inplace=True)

    # Compute final differentials
    for db_col in list(_BASE_FEATURE_COLS.keys()) + list(_DERIVED_COLS.keys()):
        h, a = f"home_{db_col}", f"away_{db_col}"
        if h in result.columns and a in result.columns:
            result[f"{db_col}_diff"] = result[h] - result[a]

    # Return only the game_id and the newly created features
    feature_cols = [c for c in result.columns if c.startswith(('home_rolling_', 'away_rolling_')) or c.endswith('_diff')]

    # Return ONLY the game_id and the new feature columns. The engine will do the merge.
    final_cols = ['game_id'] + feature_cols
    return result[final_cols]
