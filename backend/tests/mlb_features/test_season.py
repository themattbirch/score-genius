# backend/tests/mlb_features/test_season.py
import pandas as pd
import numpy as np
import pytest
import logging

import os
import sys

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.mlb_features import season # Module to test

# --- Fixtures ---

@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """A mock of the main games DataFrame passed to transform."""
    return pd.DataFrame({
        "game_id": ["G1", "G2"],
        "season": [2023, 2024],
        "home_team_norm": ["nyy", "bos"],
        "away_team_norm": ["bos", "nyy"],
    })

@pytest.fixture
def sample_historical_stats_df() -> pd.DataFrame:
    """Mock of the historical team stats with the necessary columns."""
    return pd.DataFrame({
        "team_norm": ["nyy", "bos", "nyy", "bos"],
        "season": [2022, 2022, 2023, 2023],
        "wins_all_percentage": [0.611, 0.481, 0.506, 0.481],
        "runs_for_avg_all": [4.98, 4.78, 4.41, 4.77],
        "runs_against_avg_all": [3.50, 4.86, 4.15, 4.42],
    })

# --- Tests ---

def test_transform_attaches_previous_historical_stats(sample_games_df, sample_historical_stats_df):
    """
    Tests that the transform correctly looks up and attaches stats from the PREVIOUS season.
    """
    result = season.transform(
        df=sample_games_df,
        historical_team_stats_df=sample_historical_stats_df
    )
    
    # Check for game G1 (season 2023, should use 2022 stats)
    game_g1 = result[result['game_id'] == 'G1'].iloc[0]
    
    # NYY (home) 2022 stats
    assert np.isclose(game_g1['home_prev_season_win_pct'], 0.611)
    assert np.isclose(game_g1['home_prev_season_avg_runs_for'], 4.98)
    
    # BOS (away) 2022 stats
    assert np.isclose(game_g1['away_prev_season_win_pct'], 0.481)
    assert np.isclose(game_g1['away_prev_season_avg_runs_against'], 4.86)
    
    # Check imputation flags should be False since data was found
    assert not game_g1['home_prev_season_win_pct_imputed']
    assert not game_g1['away_prev_season_win_pct_imputed']

def test_transform_handles_missing_teams_with_defaults(sample_games_df, sample_historical_stats_df):
    """
    Tests that if a team's previous season stats are missing, it falls back to defaults.
    """
    # Game G2 is in season 2024, so it looks for 2023 stats, which exist for both teams.
    # Let's create a new game where one team is missing.
    games_with_missing_team = pd.DataFrame({
        "game_id": ["G3"],
        "season": [2023], # Looks for 2022 stats
        "home_team_norm": ["nyy"], # NYY has 2022 stats
        "away_team_norm": ["unknown_team"], # This team is not in our historical mock
    })

    result = season.transform(
        df=games_with_missing_team,
        historical_team_stats_df=sample_historical_stats_df
    )
    
    game_g3 = result.iloc[0]

    # Home team (NYY) should have its real 2022 stats
    assert not game_g3['home_prev_season_win_pct_imputed']
    assert np.isclose(game_g3['home_prev_season_win_pct'], 0.611)

    # Away team (unknown) should have default values and imputed flag
    assert game_g3['away_prev_season_win_pct_imputed'] == True
    assert np.isclose(game_g3['away_prev_season_win_pct'], season.MLB_DEFAULTS.get("win_pct", 0.5))

def test_transform_handles_no_historical_data(sample_games_df):
    """
    Tests that if no historical_team_stats_df is provided, all stats are defaulted.
    """
    result = season.transform(df=sample_games_df, historical_team_stats_df=pd.DataFrame())

    # All imputed flags should be True
    assert result['home_prev_season_win_pct_imputed'].all()
    assert result['away_prev_season_win_pct_imputed'].all()

    # All values should be the default
    default_win_pct = season.MLB_DEFAULTS.get("win_pct", 0.5)
    assert np.isclose(result['home_prev_season_win_pct'], default_win_pct).all()
    assert np.isclose(result['away_prev_season_win_pct'], default_win_pct).all()

def test_transform_calculates_diffs_correctly(sample_games_df, sample_historical_stats_df):
    """
    Tests that the final differential columns are calculated correctly.
    """
    result = season.transform(
        df=sample_games_df,
        historical_team_stats_df=sample_historical_stats_df
    )
    
    game_g1 = result[result['game_id'] == 'G1'].iloc[0]

    # For G1 (2023), NYY (H) vs BOS (A), using 2022 stats
    # NYY win pct: 0.611, BOS win pct: 0.481
    expected_diff = 0.611 - 0.481
    assert np.isclose(game_g1['prev_season_win_pct_diff'], expected_diff)

    # NYY net rating: 4.98 - 3.50 = 1.48
    # BOS net rating: 4.78 - 4.86 = -0.08
    expected_net_rating_diff = 1.48 - (-0.08)
    assert np.isclose(game_g1['prev_season_net_rating_diff'], expected_net_rating_diff)