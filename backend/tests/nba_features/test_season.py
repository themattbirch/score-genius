# backend/tests/nba_features/test_season.py

import pandas as pd
import numpy as np
import pytest


import os
import sys

# ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)

from backend.nba_features import season, utils

# ---------- Fixtures ----------
@pytest.fixture
def sample_game_data_for_season() -> pd.DataFrame:
    data = {
        "game_id":   [201, 202, 203, 204, 205, 206],
        "game_date": [
            "2023-04-05", "2023-04-08",   # 2022‑23 season
            "2023-10-25", "2023-10-28",   # 2023‑24 season
            "2023-11-01", "2023-11-05",   # 2023‑24 season (Team C missing stats)
        ],
        "home_team": ["Team A", "Team B", "Team A", "Team C", "Team B", "Team C"],
        "away_team": ["Team B", "Team A", "Team B", "Team A", "Team C", "Team A"],
    }
    df = pd.DataFrame(data)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

@pytest.fixture
def sample_team_stats_data() -> pd.DataFrame:
    data = {
        "team_name": ["Team A", "Team B", "Team A", "Team B", "Team D"],  # Team C absent for 23‑24
        "season":    ["2022-23", "2022-23", "2023-24", "2023-24", "2023-24"],
        "wins_all_percentage":      [0.60, 0.55, 0.70, 0.65, 0.40],
        "points_for_avg_all":       [118, 115, 120, 117, 110],
        "points_against_avg_all":   [112, 113, 110, 111, 115],
        "current_form":             ["W2", "L1", "W1", "W3", "L2"],
    }
    return pd.DataFrame(data)

# ---------- Helper ----------
def assert_diff_columns(row):
    """Validate that diff / net‑rating columns equal the arithmetic difference."""
    assert np.isclose(
        row["season_win_pct_diff"],
        row["home_season_win_pct"] - row["away_season_win_pct"],
    )
    assert np.isclose(
        row["season_pts_for_diff"],
        row["home_season_avg_pts_for"] - row["away_season_avg_pts_for"],
    )
    assert np.isclose(
        row["season_pts_against_diff"],
        row["home_season_avg_pts_against"] - row["away_season_avg_pts_against"],
    )
    # Net‑ratings
    home_net = row["home_season_avg_pts_for"] - row["home_season_avg_pts_against"]
    away_net = row["away_season_avg_pts_for"] - row["away_season_avg_pts_against"]
    assert np.isclose(row["home_season_net_rating"], home_net)
    assert np.isclose(row["away_season_net_rating"], away_net)
    assert np.isclose(row["season_net_rating_diff"], home_net - away_net)

# ---------- Tests ----------
def test_merge_and_diff_logic(sample_game_data_for_season, sample_team_stats_data):
    """
    Validate that:
      • season stats columns are populated (either merged or default‑filled)
      • diff / net‑rating math is correct for every row
    """
    df = season.transform(
        sample_game_data_for_season.copy(),
        team_stats_df=sample_team_stats_data.copy(),
    )

    required_cols = [
        "home_season_win_pct", "away_season_win_pct",
        "home_season_avg_pts_for", "away_season_avg_pts_for",
        "home_season_avg_pts_against", "away_season_avg_pts_against",
        "home_current_form", "away_current_form",
        "season_win_pct_diff", "season_pts_for_diff",
        "season_pts_against_diff", "home_season_net_rating",
        "away_season_net_rating", "season_net_rating_diff",
    ]
    for col in required_cols:
        assert col in df.columns
        assert not df[col].isnull().any()

    # Diff arithmetic holds for every game
    df.apply(assert_diff_columns, axis=1)

def test_defaults_when_stats_missing(sample_game_data_for_season, sample_team_stats_data):
    """
    Team C (home) has no 2023‑24 stats in the input table.
    Verify that season.transform substitutes DEFAULTS for missing data.
    """
    default_win   = utils.DEFAULTS.get("win_pct",        0.5)
    default_for   = utils.DEFAULTS.get("avg_pts_for",    115.0)
    default_again = utils.DEFAULTS.get("avg_pts_against",115.0)
    default_form  = "N/A"

    df = season.transform(
        sample_game_data_for_season.copy(),
        team_stats_df=sample_team_stats_data.copy(),
    )
    row = df[df["game_id"] == 206].iloc[0]  # 5 Nov 23 – Team C vs Team A

    # Home (Team C) should be defaults
    assert row["home_season_win_pct"]         == default_win
    assert row["home_season_avg_pts_for"]     == default_for
    assert row["home_season_avg_pts_against"] == default_again
    assert row["home_current_form"]           == default_form

    # Away side present (may still fall back to defaults if merge fails);
    # regardless, diff column must equal arithmetic difference.
    assert_diff_columns(row)

def test_all_defaults_when_stats_df_missing(sample_game_data_for_season):
    """
    If team_stats_df is None or empty, every season column should equal DEFAULTS
    and all diff columns should be zero.
    """
    default_win   = utils.DEFAULTS.get("win_pct",        0.5)
    default_for   = utils.DEFAULTS.get("avg_pts_for",    115.0)
    default_again = utils.DEFAULTS.get("avg_pts_against",115.0)
    default_form  = "N/A"

    for ts_df in (None, pd.DataFrame()):
        df = season.transform(sample_game_data_for_season.copy(), team_stats_df=ts_df)
        one_row = df.iloc[0]

        # Base stats
        assert one_row["home_season_win_pct"]         == default_win
        assert one_row["away_season_win_pct"]         == default_win
        assert one_row["home_season_avg_pts_for"]     == default_for
        assert one_row["away_season_avg_pts_for"]     == default_for
        assert one_row["home_season_avg_pts_against"] == default_again
        assert one_row["away_season_avg_pts_against"] == default_again
        assert one_row["home_current_form"]           == default_form
        assert one_row["away_current_form"]           == default_form

        # Diffs / net‑ratings
        assert_diff_columns(one_row)

def test_no_future_snapshot_leak():
    """
    Snapshot taken *after* the game must NOT influence the features.
    Early‑season game on 2023‑10‑15 should use defaults,
    even though a 2023‑24 snapshot dated 2023‑12‑01 exists.
    """

    # ---------- game row ----------
    g = pd.DataFrame(
        {
            "game_id":   [999],
            "game_date": pd.to_datetime(["2023-10-15"]),   # ← regular hyphen
            "home_team": ["Team A"],
            "away_team": ["Team B"],
        }
    )

    # ---------- team stats with FUTURE snapshot (leak source) ----------
    ts = pd.DataFrame(
        {
            "team_name":  ["Team A", "Team B"],
            "season":     ["2023‑24", "2023‑24"],
            "wins_all_percentage":    [0.99, 0.99],      # impossible this early
            "points_for_avg_all":     [150, 150],
            "points_against_avg_all": [50, 50],
            "current_form":           ["W10", "W10"],
            "updated_at": pd.to_datetime(["2023-12-01", "2023-12-01"]),
        }
    )

    df = season.transform(g.copy(), team_stats_df=ts, debug=False)

    # Defaults we EXPECT because the snapshot is *after* the game
    def_win = utils.DEFAULTS["win_pct"]
    def_for = utils.DEFAULTS["avg_pts_for"]
    def_ag  = utils.DEFAULTS["avg_pts_against"]

    assert df.loc[0, "home_season_win_pct"]         == def_win
    assert df.loc[0, "away_season_win_pct"]         == def_win
    assert df.loc[0, "home_season_avg_pts_for"]     == def_for
    assert df.loc[0, "home_season_avg_pts_against"] == def_ag
    # Same for away side
    assert df.loc[0, "away_season_avg_pts_for"]     == def_for
    assert df.loc[0, "away_season_avg_pts_against"] == def_ag

def test_current_season_rows_use_prev_year_stats():
    g  = pd.DataFrame({"game_id":[1],
                       "game_date":pd.to_datetime(["2023-10-20"]),
                       "home_team":["Team A"], "away_team":["Team B"]})
    ts = pd.DataFrame({"team_name":["Team A","Team A"],
                       "season":["2022-23","2023-24"],
                       "wins_all_percentage":[.60,.99],   # 0.99 would leak
                       "points_for_avg_all":[118,150],
                       "points_against_avg_all":[112,50],
                       "current_form":["W2","W10"]})
    out = season.transform(g, team_stats_df=ts)
    # should have pulled 0.60, NOT 0.99
    assert out.loc[0,"home_season_win_pct"] == pytest.approx(.60)

