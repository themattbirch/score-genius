# backend/tests/mlb_features/test_advanced.py
import pandas as pd
import pytest
import numpy as np
import logging

import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.mlb_features import advanced

# Fixtures
@pytest.fixture
def mock_historical_team_stats():
    """A realistic mock of historical_team_stats_df."""
    return pd.DataFrame({
        'team_norm': ['nyy', 'bos', 'nyy', 'bos'],
        'season': [2022, 2022, 2023, 2023],
        'wins_all_total': [99, 78, 82, 78],
        'games_played_all': [162, 162, 162, 162],
        'runs_for_total_all': [807, 735, 715, 772],
        'runs_against_total_all': [567, 787, 672, 716],
        'wins_home_total': [57, 43, 41, 41],
        'games_played_home': [81, 81, 81, 81],
        'wins_away_total': [42, 35, 41, 37],
        'games_played_away': [81, 81, 81, 81]
    })

@pytest.fixture
def sample_games_df():
    """A mock of the main games DataFrame passed to transform."""
    return pd.DataFrame({
        'game_id': ['G1', 'G2'],
        'season': [2023, 2023],
        'home_team_norm': ['nyy', 'bos'],
        'away_team_norm': ['bos', 'nyy']
    })

# --- Tests for the new structure ---

def test_precompute_two_season_aggregates(mock_historical_team_stats):
    """Tests the heavy-lifting pre-computation function directly."""
    precomputed = advanced._precompute_two_season_aggregates(mock_historical_team_stats)
    
    # Check for correct columns
    expected_cols = [
        'team_norm', 'season', 'season_win_pct', 'season_runs_for_avg', 
        'season_runs_against_avg', 'venue_win_pct_home', 'venue_win_pct_away'
    ]
    assert all(col in precomputed.columns for col in expected_cols)
    
    # Check a specific calculation for the 2023 season
    # For NYY in 2023, it should aggregate 2022 and 2023 stats
    nyy_2023 = precomputed[(precomputed['team_norm'] == 'nyy') & (precomputed['season'] == 2023)].iloc[0]
    
    # Expected win % for NYY over 2022-2023: (99+82) / (162+162) = 181 / 324
    expected_win_pct = 181 / 324
    assert np.isclose(nyy_2023['season_win_pct'], expected_win_pct)

def test_transform_merges_correctly(sample_games_df, mock_historical_team_stats):
    """Tests that the transform function correctly merges the precomputed stats."""
    # 1. Generate the precomputed data, just like the engine does
    precomputed_stats = advanced._precompute_two_season_aggregates(mock_historical_team_stats)

    # 2. Pass the precomputed data to the transform function
    result = advanced.transform(
        df=sample_games_df,
        precomputed_stats=precomputed_stats,
        debug=True
    )
    
    # 3. Check if the columns were added and prefixed correctly
    assert 'home_season_win_pct' in result.columns
    assert 'away_season_win_pct' in result.columns
    assert 'home_venue_win_advantage' in result.columns

    # 4. Check a specific value
    # Home team for G1 is 'nyy', season 2023. We expect its 2022-2023 aggregated win pct.
    expected_nyy_win_pct = 181 / 324
    assert np.isclose(result.loc[0, 'home_season_win_pct'], expected_nyy_win_pct)

def test_transform_skips_if_no_precomputed_stats(sample_games_df, caplog):
    """Tests that the transform function skips its logic if no precomputed data is provided."""
    with caplog.at_level(logging.WARNING):
        # Call without the precomputed_stats argument
        result = advanced.transform(df=sample_games_df)
    
    # The result should be the original dataframe, unmodified
    assert result.equals(sample_games_df)
    # A warning should be logged
    assert "No precomputed_stats provided. Skipping." in caplog.text

def test_transform_handles_teams_not_in_precomputed(sample_games_df, mock_historical_team_stats):
    """Tests that teams not in the precomputed stats get default (0) values."""
    # Modify the game df to include a team not in our historical mock
    sample_games_df.loc[0, 'home_team_norm'] = 'unknown_team'
    
    precomputed_stats = advanced._precompute_two_season_aggregates(mock_historical_team_stats)
    result = advanced.transform(df=sample_games_df, precomputed_stats=precomputed_stats)
    
    # The home_season_win_pct for the unknown team should be filled with 0.0
    assert np.isclose(result.loc[0, 'home_season_win_pct'], 0.0)