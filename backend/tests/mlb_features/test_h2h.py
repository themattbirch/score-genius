# backend/tests/mlb_features/test_h2h.py
import os
import sys
import numpy as np
import pandas as pd
import pytest
import logging

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.mlb_features import h2h
from backend.mlb_features.utils import normalize_team_name, DEFAULTS as MLB_DEFAULTS

# --- Test Constants ---
# These are the four columns the h2h.py module actually produces
H2H_COLS_MLB = [
    "matchup_num_games",
    "matchup_avg_run_diff",
    "matchup_home_win_pct",
    "matchup_avg_total_runs",
]

# --- Fixtures ---
@pytest.fixture(scope="module")
def historical_df_mlb() -> pd.DataFrame:
    """A consistent historical DataFrame for h2h tests."""
    data = [
        # A vs B history before July 15
        {"game_id": "h1", "game_date_et": "2023-04-01", "home_team_norm": "teama", "away_team_norm": "teamb", "home_score": 5, "away_score": 2}, # A wins, diff +3, total 7
        {"game_id": "h2", "game_date_et": "2023-04-02", "home_team_norm": "teamb", "away_team_norm": "teama", "home_score": 3, "away_score": 4}, # A wins, diff +1, total 7
        {"game_id": "h3", "game_date_et": "2023-05-10", "home_team_norm": "teama", "away_team_norm": "teamb", "home_score": 2, "away_score": 8}, # B wins, diff -6, total 10
        # A vs C history
        {"game_id": "h4", "game_date_et": "2023-06-01", "home_team_norm": "teamc", "away_team_norm": "teama", "home_score": 1, "away_score": 9}, # A wins
    ]
    df = pd.DataFrame(data)
    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    return df

@pytest.fixture(scope="module")
def upcoming_df_mlb() -> pd.DataFrame:
    """A consistent upcoming games DataFrame for h2h tests."""
    data = [
        {"game_id": "u1", "game_date_et": "2023-07-15", "home_team_norm": "teama", "away_team_norm": "teamb", "home_score": np.nan, "away_score": np.nan},
        {"game_id": "u2", "game_date_et": "2023-07-16", "home_team_norm": "teama", "away_team_norm": "teamc"},
        {"game_id": "u3", "game_date_et": "2023-07-17", "home_team_norm": "teamd", "away_team_norm": "teame"}, # No history
    ]
    df = pd.DataFrame(data)
    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    return df

# --- Tests ---

def test_h2h_with_history(historical_df_mlb, upcoming_df_mlb):
    """Test H2H calculation for a matchup with history."""
    game_to_test = upcoming_df_mlb[upcoming_df_mlb['game_id'] == 'u1'].copy()
    
    result = h2h.transform(
        df=game_to_test,
        historical_df=historical_df_mlb,
        max_games=10
    )
    
    # Expected stats for TeamA vs TeamB based on 3 historical games
    # Home Wins: A won as home team once (h1), lost once (h3). B never won as home team. Home win pct = 1/2 = 0.5
    # Run Diffs (from A's perspective as home team): +3 (h1), -1 (h2), -6 (h3). Avg = (3-1-6)/3 = -4/3
    # Total Runs: 7, 7, 10. Avg = 24/3 = 8.0
    
    game_row = result.iloc[0]
    assert game_row['matchup_num_games'] == 3
    assert np.isclose(game_row['matchup_avg_run_diff'], -4.0 / 3.0)
    assert np.isclose(game_row['matchup_home_win_pct'], 1.0 / 3.0)
    assert np.isclose(game_row['matchup_avg_total_runs'], 8.0)

def test_h2h_with_no_history(historical_df_mlb, upcoming_df_mlb):
    """Test H2H calculation for a matchup with no history."""
    game_to_test = upcoming_df_mlb[upcoming_df_mlb['game_id'] == 'u3'].copy()
    
    result = h2h.transform(
        df=game_to_test,
        historical_df=historical_df_mlb,
        max_games=10
    )
    
    # Expect default values
    game_row = result.iloc[0]
    assert game_row['matchup_num_games'] == 0
    assert np.isclose(game_row['matchup_avg_run_diff'], 0.0)
    assert np.isclose(game_row['matchup_home_win_pct'], 0.5)
    assert np.isclose(game_row['matchup_avg_total_runs'], 9.0)

def test_h2h_no_historical_df_provided(upcoming_df_mlb):
    """Test that defaults are returned if no historical_df is provided."""
    result = h2h.transform(df=upcoming_df_mlb.copy(), historical_df=None)
    
    # All rows should have default values
    assert (result['matchup_num_games'] == 0).all()
    assert (result['matchup_home_win_pct'] == 0.5).all()

def test_h2h_max_games_limit(historical_df_mlb, upcoming_df_mlb):
    """Test that the max_games parameter correctly limits the history."""
    game_to_test = upcoming_df_mlb[upcoming_df_mlb['game_id'] == 'u1'].copy()
    
    # Limit to only the 2 most recent games (h2 and h3)
    result = h2h.transform(
        df=game_to_test,
        historical_df=historical_df_mlb,
        max_games=2
    )
    
    # Expected stats for TeamA vs TeamB based on last 2 games
    # Home Wins: A lost once (h3). B never won as home team. Home win pct = 0/1 = 0.0
    # Run Diffs (from A's perspective as home team): -1 (h2), -6 (h3). Avg = (-1-6)/2 = -3.5
    # Total Runs: 7, 10. Avg = 8.5
    
    game_row = result.iloc[0]
    assert game_row['matchup_num_games'] == 2
    assert np.isclose(game_row['matchup_avg_run_diff'], -3.5)
    assert np.isclose(game_row['matchup_home_win_pct'], 0.0)
    assert np.isclose(game_row['matchup_avg_total_runs'], 8.5)