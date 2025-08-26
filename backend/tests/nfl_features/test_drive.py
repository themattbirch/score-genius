# backend/tests/nfl_features/test_drive.py

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import drive

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sample_spine_df() -> pd.DataFrame:
    """A sample games_df (spine) for a season chunk."""
    rows = [
        dict(game_id=1, season=2023, home_team_id="10", away_team_id="20", kickoff_ts="2023-09-10T17:00:00Z"),
        dict(game_id=2, season=2023, home_team_id="10", away_team_id="30", kickoff_ts="2023-09-17T17:00:00Z"),
        dict(game_id=3, season=2024, home_team_id="10", away_team_id="20", kickoff_ts="2024-09-08T17:00:00Z"),
        # Game with team names for fallback testing
        dict(game_id=4, season=2024, home_team="Kansas City Chiefs", away_team="Buffalo Bills", kickoff_ts="2024-09-15T20:00:00Z"),
    ]
    df = pd.DataFrame(rows)
    df["kickoff_ts"] = pd.to_datetime(df["kickoff_ts"])
    return df

@pytest.fixture(scope="module")
def sample_team_metrics_df() -> pd.DataFrame:
    """
    Simulates the output of `compute_drive_metrics` on historical data.
    Includes data points for multiple teams across multiple seasons.
    """
    rows = [
        # Team 10 (2 games in 2023, 1 in 2024)
        dict(game_id=1, team_id="10", season=2023, ts_utc="2023-09-10T17:00:00Z", drive_points_per_drive=2.8, drive_yards_per_play=5.5),
        dict(game_id=2, team_id="10", season=2023, ts_utc="2023-09-17T17:00:00Z", drive_points_per_drive=3.5, drive_yards_per_play=6.0),
        dict(game_id=3, team_id="10", season=2024, ts_utc="2024-09-08T17:00:00Z", drive_points_per_drive=3.0, drive_yards_per_play=5.8),
        # Team 20
        dict(game_id=1, team_id="20", season=2023, ts_utc="2023-09-10T17:00:00Z", drive_points_per_drive=2.1, drive_yards_per_play=4.0),
        dict(game_id=3, team_id="20", season=2024, ts_utc="2024-09-08T17:00:00Z", drive_points_per_drive=1.7, drive_yards_per_play=3.5),
        # Team 30 (only one game)
        dict(game_id=2, team_id="30", season=2023, ts_utc="2023-09-17T17:00:00Z", drive_points_per_drive=1.4, drive_yards_per_play=3.0),
        # Teams for name-based matching
        dict(game_id=4, team_id="KC", team_name="Kansas City Chiefs", season=2024, ts_utc="2024-09-15T20:00:00Z", drive_points_per_drive=4.0, drive_yards_per_play=7.0),
        dict(game_id=4, team_id="BUF", team_name="Buffalo Bills", season=2024, ts_utc="2024-09-15T20:00:00Z", drive_points_per_drive=3.2, drive_yards_per_play=6.2),
    ]
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    return df


# --------------------------------------------------------------------------- #
# Internal Helper Tests                                                       #
# --------------------------------------------------------------------------- #

def test_rolling_pre_game_with_season_reset(sample_team_metrics_df):
    """Verify rolling logic resets at season boundaries."""
    res = drive._rolling_pre_game(
        df_team=sample_team_metrics_df,
        group_keys=["team_id", "season"],
        metrics=["drive_points_per_drive"],
        window=5,
    )
    # Team 10, Game 2 (2nd game of 2023)
    g2 = res.loc[(res["game_id"] == 2) & (res["team_id"] == "10")].iloc[0]
    assert g2["drive_prior_games"] == 1
    assert np.isclose(g2["drive_points_per_drive_avg"], 2.8) # Avg is just Game 1

    # Team 10, Game 3 (1st game of 2024)
    g3 = res.loc[(res["game_id"] == 3) & (res["team_id"] == "10")].iloc[0]
    assert g3["drive_prior_games"] == 0 # Reset for new season
    assert pd.isna(g3["drive_points_per_drive_avg"]) # No prior games in this group

def test_rolling_pre_game_without_season_reset(sample_team_metrics_df):
    """Verify rolling logic continues across seasons when specified."""
    res = drive._rolling_pre_game(
        df_team=sample_team_metrics_df,
        group_keys=["team_id"], # No season key
        metrics=["drive_points_per_drive"],
        window=5,
    )
    # Team 10, Game 3 (3rd game overall)
    g3 = res.loc[(res["game_id"] == 3) & (res["team_id"] == "10")].iloc[0]
    assert g3["drive_prior_games"] == 2 # 3rd game overall for team 10
    expected_avg = (2.8 + 3.5) / 2 # Avg of first two games
    assert np.isclose(g3["drive_points_per_drive_avg"], expected_avg)

def test_aggregate_to_game_level(sample_spine_df):
    """Verify correct pivoting from team-level to game-level rows."""
    # Create a sample input that looks like the output of _rolling_pre_game
    team_level_avg = pd.DataFrame({
        "game_id": [1, 1, 2, 2],
        "team_id": ["10", "20", "10", "30"],
        "drive_points_per_drive_avg": [np.nan, np.nan, 2.8, np.nan],
        "drive_yards_per_play_avg": [np.nan, np.nan, 5.5, np.nan],
        "drive_prior_games_capped": [0, 0, 1, 0],
        "drive_low_sample_any": [1, 1, 1, 1],
    })
    res = drive._aggregate_to_game_level(
        df_team=team_level_avg,
        spine=sample_spine_df,
        metrics_avg=["drive_points_per_drive_avg", "drive_yards_per_play_avg"]
    )
    # Check Game 2, where Team 10 is home and has a prior game
    g2 = res.loc[res["game_id"] == 2].iloc[0]
    assert g2["drive_home_prior_games"] == 1
    assert g2["drive_away_prior_games"] == 0
    assert np.isclose(g2["drive_home_points_per_drive_avg"], 2.8)
    assert pd.isna(g2["drive_away_points_per_drive_avg"])
    # Check diff and total columns
    assert "drive_points_per_drive_avg_diff" in res.columns
    assert "drive_total_points_per_drive_avg" in res.columns
    assert np.isclose(g2["drive_points_per_drive_avg_diff"], 2.8) # 2.8 - 0 (from NaN)
    assert np.isclose(g2["drive_total_points_per_drive_avg"], 2.8) # 2.8 + 0 (from NaN)


# --------------------------------------------------------------------------- #
# Public API: transform() Tests                                               #
# --------------------------------------------------------------------------- #

@mock.patch("backend.nfl_features.drive.compute_drive_metrics")
def test_transform_happy_path(mock_compute_metrics, sample_spine_df, sample_team_metrics_df):
    """Test the end-to-end transform function with season reset enabled."""
    mock_compute_metrics.return_value = sample_team_metrics_df
    
    res = drive.transform(
        games_df=sample_spine_df,
        historical_team_stats_df=pd.DataFrame({"dummy": [1]}), # Needs to be non-empty
        window=5,
        reset_by_season=True,
    )
    assert not res.empty
    assert "game_id" in res.columns
    assert res.game_id.nunique() == res.shape[0]

    # Check Game 3 (game_id=3)
    g3 = res.loc[res["game_id"] == 3].iloc[0]
    # Home team is 10. Avg should be from their 2 games in 2023 (reset=True, but historical data is for all time)
    # The average for the first game of 2024 should be based on the 2 prior games in 2023.
    # Ah, the mock data has a game in 2024. The rolling for game_id=3 should be based on games 1 and 2.
    home_avg_expected = (2.8 + 3.5) / 2
    assert np.isclose(g3["drive_home_points_per_drive_avg"], home_avg_expected)
    # Away team is 20. Avg should be from their 1 game in 2023.
    away_avg_expected = 2.1
    assert np.isclose(g3["drive_away_points_per_drive_avg"], away_avg_expected)
    # Check diff
    assert np.isclose(g3["drive_points_per_drive_avg_diff"], home_avg_expected - away_avg_expected)

def test_transform_empty_inputs():
    """Verify that empty inputs produce an empty DataFrame."""
    res_empty_games = drive.transform(games_df=pd.DataFrame(), historical_team_stats_df=pd.DataFrame({"a":[1]}))
    assert res_empty_games.empty
    
    res_empty_hist = drive.transform(games_df=pd.DataFrame({"a":[1]}), historical_team_stats_df=pd.DataFrame())
    assert res_empty_hist.empty

@mock.patch("backend.nfl_features.drive._rolling_pre_game")
def test_transform_soft_fail(mock_rolling, sample_spine_df):
    """Verify the soft_fail mechanism returns a sentinel DataFrame on error."""
    mock_rolling.side_effect = ValueError("A test error occurred")

    res = drive.transform(
        games_df=sample_spine_df,
        historical_team_stats_df=pd.DataFrame({"game_id": [1,2,3]}),
        soft_fail=True,
    )
    
    expected_cols = {"game_id", "drive_features_unavailable"}
    assert expected_cols.issubset(res.columns)
    assert (res["drive_features_unavailable"] == 1).all()
    assert set(res["game_id"]) == set(sample_spine_df["game_id"])

@mock.patch("backend.nfl_features.drive.compute_drive_metrics")
def test_transform_no_metrics_available(mock_compute_metrics, sample_spine_df, sample_team_metrics_df):
    """Verify it returns empty if no candidate metrics are found."""
    # Return a dataframe with columns that don't match METRIC_CANDIDATES
    bad_metrics_df = sample_team_metrics_df.rename(columns={
        "drive_points_per_drive": "some_other_metric",
        "drive_yards_per_play": "another_metric",
    })
    mock_compute_metrics.return_value = bad_metrics_df

    res = drive.transform(
        games_df=sample_spine_df,
        historical_team_stats_df=pd.DataFrame({"dummy": [1]}),
    )
    assert res.empty