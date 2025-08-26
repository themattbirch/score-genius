# backend/tests/nfl_features/test_situational.py
from __future__ import annotations

"""
Unit-tests for backend.nfl_features.situational
"""

import os
import sys

import pandas as pd
import pytest

# Ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

from backend.nfl_features.situational import compute_situational_features

# --------------------------------------------------------------------------- #
# Test Fixture for Sample Game Data                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sample_games_df() -> pd.DataFrame:
    """Provides a diverse sample of game scenarios for testing."""
    data = [
        # 1. Regular Sunday Afternoon (Divisional)
        {
            "game_id": 101, "home_team_norm": "packers", "away_team_norm": "bears",
            "game_date": "2024-10-20", "game_time": "13:00:00", "stage": "REG",
            "game_timestamp": None
        },
        # 2. Sunday Night Football (Conference, non-divisional)
        {
            "game_id": 102, "home_team_norm": "49ers", "away_team_norm": "packers",
            "game_date": "2024-11-17", "game_time": "20:20:00", "stage": "REG",
            "game_timestamp": None
        },
        # 3. Monday Night Football (Inter-conference)
        {
            "game_id": 103, "home_team_norm": "chiefs", "away_team_norm": "lions",
            "game_date": "2024-09-23", "game_time": "20:15:00", "stage": "REG",
            "game_timestamp": None
        },
        # 4. Playoff Game (using UTC timestamp)
        {
            "game_id": 104, "home_team_norm": "ravens", "away_team_norm": "texans",
            "game_date": None, "game_time": None, "stage": "PST",
            "game_timestamp": "2025-01-18T21:30:00Z"  # Corresponds to 4:30 PM ET
        },
        # 5. Thursday Night Football
        {
            "game_id": 105, "home_team_norm": "seahawks", "away_team_norm": "rams",
            "game_date": "2024-10-10", "game_time": "20:15:00", "stage": "REG",
            "game_timestamp": None
        },
    ]
    return pd.DataFrame(data)

# --------------------------------------------------------------------------- #
# Tests for compute_situational_features                                      #
# --------------------------------------------------------------------------- #

def test_output_schema_and_shape(sample_games_df):
    """Tests that the output contains the correct columns and shape."""
    result = compute_situational_features(sample_games_df)
    expected_cols = {
        "game_id", "is_primetime", "is_weekend", "is_regular_season",
        "is_playoffs", "is_division_game", "is_conference_game",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(sample_games_df)

def test_primetime_and_weekend_flags(sample_games_df):
    """Tests the is_primetime and is_weekend flags."""
    result = compute_situational_features(sample_games_df).set_index("game_id")
    
    # Sunday afternoon: Weekend, not primetime
    assert result.loc[101, "is_weekend"] == 1
    assert result.loc[101, "is_primetime"] == 0
    
    # Sunday night: Weekend and primetime
    assert result.loc[102, "is_weekend"] == 1
    assert result.loc[102, "is_primetime"] == 1
    
    # Monday night: Not weekend, but primetime
    assert result.loc[103, "is_weekend"] == 0
    assert result.loc[103, "is_primetime"] == 1
    
    # Saturday playoff game: Weekend, but not primetime by our heuristic (afternoon)
    assert result.loc[104, "is_weekend"] == 1
    assert result.loc[104, "is_primetime"] == 0
    
    # Thursday night: Not weekend, but primetime
    assert result.loc[105, "is_weekend"] == 0
    assert result.loc[105, "is_primetime"] == 1

def test_season_stage_flags(sample_games_df):
    """Tests the is_regular_season and is_playoffs flags."""
    result = compute_situational_features(sample_games_df).set_index("game_id")
    
    # Regular season game
    assert result.loc[101, "is_regular_season"] == 1
    assert result.loc[101, "is_playoffs"] == 0
    
    # Playoff game
    assert result.loc[104, "is_regular_season"] == 0
    assert result.loc[104, "is_playoffs"] == 1

def test_rivalry_flags(sample_games_df):
    """Tests the is_division_game and is_conference_game flags."""
    result = compute_situational_features(sample_games_df).set_index("game_id")
    
    # Divisional game (Packers vs Bears are both NFC North)
    assert result.loc[101, "is_division_game"] == 1
    assert result.loc[101, "is_conference_game"] == 0
    
    # Conference game, non-divisional (49ers and Packers are both NFC)
    assert result.loc[102, "is_division_game"] == 0
    assert result.loc[102, "is_conference_game"] == 1
    
    # Inter-conference game (Chiefs are AFC, Lions are NFC)
    assert result.loc[103, "is_division_game"] == 0
    assert result.loc[103, "is_conference_game"] == 0

def test_empty_and_missing_column_inputs():
    """Tests edge cases like empty inputs or missing required columns."""
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    result_empty = compute_situational_features(empty_df)
    assert result_empty.empty
    
    # Test DataFrame missing a required column (e.g., 'stage')
    missing_col_df = pd.DataFrame([{
        "game_id": 201, "home_team_norm": "saints", "away_team_norm": "falcons",
        "game_date": "2024-12-01", "game_time": "13:00:00"
    }])
    # The function should still run and produce output, likely with default values for the missing logic
    result_missing = compute_situational_features(missing_col_df)
    assert "is_regular_season" in result_missing.columns
    assert result_missing.loc[0, "is_regular_season"] == 0 # Should default to 0