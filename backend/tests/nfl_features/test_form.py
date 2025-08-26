# backend/tests/nfl_features/test_form.py

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

from backend.nfl_features import form, utils

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sample_games_df() -> pd.DataFrame:
    """Provides upcoming games to be featurized."""
    return pd.DataFrame([
        # Game 1: Team A vs Team B. Both have full history.
        {"game_id": 101, "game_date": "2024-10-20", "home_team_norm": "team_a", "away_team_norm": "team_b"},
        # Game 2: Team C vs Team D. C has short history, D has no history.
        {"game_id": 102, "game_date": "2024-10-20", "home_team_norm": "team_c", "away_team_norm": "team_d"},
    ])

@pytest.fixture(scope="module")
def sample_historical_games_df() -> pd.DataFrame:
    """Provides a controlled game history for form/streak calculations."""
    data = [
        # History for Team A: Last 5 games are L, W, W, W, W. Enters with a 4-game win streak.
        {"game_date": "2024-09-08", "home_team_norm": "team_a", "away_team_norm": "opponent1", "home_score": 20, "away_score": 21}, # L
        {"game_date": "2024-09-15", "home_team_norm": "team_a", "away_team_norm": "opponent2", "home_score": 24, "away_score": 10}, # W
        {"game_date": "2024-09-22", "home_team_norm": "opponent3", "away_team_norm": "team_a", "home_score": 17, "away_score": 30}, # W
        {"game_date": "2024-09-29", "home_team_norm": "team_a", "away_team_norm": "opponent4", "home_score": 28, "away_score": 27}, # W
        {"game_date": "2024-10-13", "home_team_norm": "opponent5", "away_team_norm": "team_a", "home_score": 10, "away_score": 14}, # W
        
        # History for Team B: Last 5 games are W, L, L, L. Enters with a 3-game losing streak.
        {"game_date": "2024-09-08", "home_team_norm": "team_b", "away_team_norm": "opponent1", "home_score": 31, "away_score": 21}, # W
        {"game_date": "2024-09-15", "home_team_norm": "opponent2", "away_team_norm": "team_b", "home_score": 24, "away_score": 10}, # L
        {"game_date": "2024-09-22", "home_team_norm": "team_b", "away_team_norm": "opponent3", "home_score": 17, "away_score": 30}, # L
        {"game_date": "2024-10-06", "home_team_norm": "opponent4", "away_team_norm": "team_b", "home_score": 28, "away_score": 27}, # L
        
        # History for Team C: Only 3 games of history.
        {"game_date": "2024-09-22", "home_team_norm": "team_c", "away_team_norm": "opponent3", "home_score": 10, "away_score": 17}, # L
        {"game_date": "2024-10-06", "home_team_norm": "team_c", "away_team_norm": "opponent4", "home_score": 21, "away_score": 20}, # W
        {"game_date": "2024-10-13", "home_team_norm": "opponent5", "away_team_norm": "team_c", "home_score": 35, "away_score": 14}, # L
        
        # Team D has no history in this DataFrame.
    ]
    return pd.DataFrame(data)

# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

def test_form_and_streak_calculations(sample_games_df, sample_historical_games_df):
    """Tests that form win % and current streak are calculated correctly."""
    result = form.transform(sample_games_df, historical_df=sample_historical_games_df, lookback_window=5)
    
    # Isolate Game 101: Team A vs Team B
    row = result[result["game_id"] == 101].iloc[0]

    # Team A: History is L, W, W, W, W. Enters game with 4-game win streak, 80% win pct in last 5.
    assert np.isclose(row["home_form_win_pct_5"], 0.8)
    assert np.isclose(row["home_current_streak"], 4.0)

    # Team B: History is W, L, L, L. Enters game with 3-game losing streak, 25% win pct in last 4.
    assert np.isclose(row["away_form_win_pct_5"], 0.25) # 1 win in last 4 games
    assert np.isclose(row["away_current_streak"], -3.0)
    
    # Check diffs
    assert np.isclose(row["form_win_pct_5_diff"], 0.8 - 0.25)
    assert np.isclose(row["current_streak_diff"], 4.0 - (-3.0))

def test_short_history_handling(sample_games_df, sample_historical_games_df):
    """Tests calculations for a team with fewer games than the lookback window."""
    result = form.transform(sample_games_df, historical_df=sample_historical_games_df, lookback_window=5)
    
    # Isolate Game 102: Team C
    row = result[result["game_id"] == 102].iloc[0]

    # Team C: History is L, W, L. Enters with 1-game losing streak. Win pct is 1/3.
    assert np.isclose(row["home_form_win_pct_5"], 1/3)
    assert np.isclose(row["home_current_streak"], -1.0)

def test_imputation_for_new_team(sample_games_df, sample_historical_games_df):
    """Tests that a team with no history receives default values and is flagged."""
    result = form.transform(sample_games_df, historical_df=sample_historical_games_df, flag_imputations=True)
    
    # Isolate Game 102: Team D
    row = result[result["game_id"] == 102].iloc[0]

    # Team D has no history, should get defaults
    assert row["away_form_win_pct_5"] == utils.DEFAULTS["form_win_pct"]
    assert row["away_current_streak"] == utils.DEFAULTS["current_streak"]
    assert row["away_form_win_pct_5_imputed"] == 1
    assert row["away_current_streak_imputed"] == 1

def test_all_defaults_with_no_history(sample_games_df):
    """Tests the bail-early path when no historical DataFrame is provided."""
    result = form.transform(sample_games_df, historical_df=pd.DataFrame(), flag_imputations=True)
    
    assert (result["home_form_win_pct_5"] == utils.DEFAULTS["form_win_pct"]).all()
    assert (result["away_current_streak"] == utils.DEFAULTS["current_streak"]).all()
    assert (result.filter(like="_imputed") == 1).all().all()

def test_flag_imputations_false(sample_games_df, sample_historical_games_df):
    """Tests that no '_imputed' columns are created when the flag is False."""
    result = form.transform(
        sample_games_df,
        historical_df=sample_historical_games_df,
        flag_imputations=False
    )
    imputed_cols = [col for col in result.columns if col.endswith("_imputed")]
    assert not imputed_cols