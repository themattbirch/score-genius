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
    **kwargs # Absorbs unused kwargs from the engine like 'window'
) -> pd.DataFrame:
    """
    Attaches recent-form metrics to a games DataFrame using a pre-fetched
    DataFrame of stats.
    """
    if games.empty:
        return pd.DataFrame()

    if recent_form_df is None or recent_form_df.empty:
        logger.warning("rolling: No recent_form_df provided. Cannot add rolling features.")
        # Return a DataFrame with game_id to prevent merge errors downstream
        return games[['game_id']].copy()

    games_df = games.copy()
    stats_df = recent_form_df.copy()

    # --- The original logic from Step 3 onwards is preserved ---
    # It now operates on the DataFrames passed into the function.

    # 3. Compute any derived columns locally
    for new_col, (num_col, den_col) in _DERIVED_COLS.items():
        stats_df[new_col] = (
            stats_df.get(num_col, 0.0) - stats_df.get(den_col, 0.0)
        )

    # 4. Merge in home/away and prefix
    home_join = games_df.merge(
        stats_df,
        how="left",
        left_on="home_team_id",
        right_on="team_id",
    )
    home_pref = prefix_columns(
        home_join.drop(columns=["team_id"], errors="ignore"),
        "home",
        exclude=["game_id", "home_team_id", "away_team_id"],
    )

    away_join = games_df.merge(
        stats_df,
        how="left",
        left_on="away_team_id",
        right_on="team_id",
    )
    away_pref = prefix_columns(
        away_join.drop(columns=["team_id"], errors="ignore"),
        "away",
        exclude=["game_id", "home_team_id", "away_team_id"],
    )

    combined = (
        home_pref
        .drop(columns=["home_team_id", "away_team_id"], errors="ignore")
        .merge(
            away_pref.drop(columns=["home_team_id", "away_team_id"], errors="ignore"),
            on="game_id",
        )
    )

    # 5. Fill missing with defaults
    for db_col, generic_key in _BASE_FEATURE_COLS.items():
        home_col = f"home_{db_col}"
        away_col = f"away_{db_col}"
        default_val = DEFAULTS.get(generic_key, 0.0)
        if home_col in combined:
            combined[home_col] = combined[home_col].astype(float).fillna(default_val)
        if away_col in combined:
            combined[away_col] = combined[away_col].astype(float).fillna(default_val)
    combined.fillna(0.0, inplace=True)

    # 6. Compute differentials
    for db_col in list(_BASE_FEATURE_COLS.keys()) + list(_DERIVED_COLS.keys()):
        h, a = f"home_{db_col}", f"away_{db_col}"
        if h in combined and a in combined:
            combined[f"{db_col}_diff"] = combined[h] - combined[a]

    # 7. Return only game_id + newly created feature columns
    keep = ["game_id"] + [c for c in combined if c.startswith(('home_rolling_', 'away_rolling_')) or c.endswith('_diff')]
    
    # Ensure all columns to keep actually exist
    final_cols = [c for c in keep if c in combined.columns]
    
    return combined[final_cols].copy()