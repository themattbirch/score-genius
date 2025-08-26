# backend/tests/nfl_features/test_rolling.py
from __future__ import annotations
import pandas as pd
import pytest
import numpy as np

import os
import sys

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.nfl_features import rolling
from backend.nfl_features.utils import DEFAULTS

# --- Fixtures ---
@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """Mock of the games DataFrame the function receives."""
    return pd.DataFrame({
        "game_id": ["G1", "G2"],
        "home_team_id": [10, 30],
        "away_team_id": [20, 10],
    })

@pytest.fixture
def sample_recent_form_df() -> pd.DataFrame:
    """Mock of the pre-fetched recent_form_df."""
    return pd.DataFrame({
        "team_id": [10, 20], # Team 30 is intentionally missing
        "rolling_points_for_avg": [28.5, 24.0],
        "rolling_points_against_avg": [21.0, 25.5],
        "rolling_yards_per_play_avg": [6.2, 5.1],
        "rolling_turnover_differential_avg": [0.5, -1.0],
    })

# --- Tests ---

def test_merge_and_diffs(sample_games_df, sample_recent_form_df):
    """Tests that stats are correctly merged and differentials are calculated."""
    result = rolling.load_rolling_features(
        games=sample_games_df,
        recent_form_df=sample_recent_form_df
    )
    
    # Check that the output has the correct shape
    assert len(result) == len(sample_games_df)
    # Check for a few expected columns
    assert "home_rolling_points_for_avg" in result.columns
    assert "away_rolling_yards_per_play_avg" in result.columns
    assert "rolling_point_differential_avg_diff" in result.columns

    # Check values for Game 1 (team 10 vs 20)
    game1 = result[result['game_id'] == 'G1'].iloc[0]
    assert np.isclose(game1['home_rolling_points_for_avg'], 28.5)
    assert np.isclose(game1['away_rolling_points_for_avg'], 24.0)
    
    # Check derived diff calculation
    home_diff = 28.5 - 21.0 # 7.5
    away_diff = 24.0 - 25.5 # -1.5
    expected_total_diff = home_diff - away_diff # 7.5 - (-1.5) = 9.0
    assert np.isclose(game1['rolling_point_differential_avg_diff'], expected_total_diff)

def test_defaults_when_team_missing(sample_games_df, sample_recent_form_df):
    """Tests that defaults are used when a team is missing from the recent_form_df."""
    result = rolling.load_rolling_features(
        games=sample_games_df,
        recent_form_df=sample_recent_form_df
    )
    
    # Check values for Game 2 (team 30 vs 10)
    game2 = result[result['game_id'] == 'G2'].iloc[0]
    
    # Home team (30) is missing, should get defaults
    assert np.isclose(game2['home_rolling_points_for_avg'], DEFAULTS.get("points_for_avg", 0.0))
    
    # Away team (10) has data
    assert np.isclose(game2['away_rolling_points_for_avg'], 28.5)

def test_empty_input_returns_empty():
    """Tests that an empty input games df results in an empty df."""
    assert rolling.load_rolling_features(games=pd.DataFrame()).empty
    
    # Test that an empty recent_form_df also results in an empty df (with just game_id)
    out = rolling.load_rolling_features(
        games=pd.DataFrame({"game_id": [1]}), 
        recent_form_df=pd.DataFrame()
    )
    assert "game_id" in out.columns
    assert "home_rolling_points_for_avg" not in out.columns