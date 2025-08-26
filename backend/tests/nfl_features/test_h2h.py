# backend/tests/nfl_features/test_h2h.py

from __future__ import annotations
"""
Unit tests for backend.nfl_features.h2h.transform
"""

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import h2h, utils


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                game_id=101,
                game_date="2024-11-17",
                home_team_norm="chiefs",
                away_team_norm="bills",
            ),
            dict(
                game_id=102,
                game_date="2025-11-16",
                home_team_norm="bills",
                away_team_norm="chiefs",
            ),
            dict(
                game_id=103,
                game_date="2024-09-15",
                home_team_norm="panthers",
                away_team_norm="raiders",
            ),
        ]
    )


@pytest.fixture(scope="module")
def sample_hist_df() -> pd.DataFrame:
    rows = [
        # KC vs BUF history (3 meetings)
        dict(
            game_date="2022-10-16",
            home_team_norm="chiefs",
            away_team_norm="bills",
            home_score=20,
            away_score=24,
        ),  # Bills win
        dict(
            game_date="2023-01-21",
            home_team_norm="bills",
            away_team_norm="chiefs",
            home_score=21,
            away_score=27,
        ),  # Chiefs win (away team)
        dict(
            game_date="2023-12-10",
            home_team_norm="chiefs",
            away_team_norm="bills",
            home_score=27,
            away_score=24,
        ),  # Chiefs win
        # Unrelated game
        dict(
            game_date="2023-11-05",
            home_team_norm="packers",
            away_team_norm="rams",
            home_score=20,
            away_score=3,
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_schema_and_values(sample_games_df, sample_hist_df):
    out = h2h.transform(sample_games_df, historical_df=sample_hist_df)

    # Basic schema
    exp_cols = {
        "h2h_games_played",
        "h2h_home_win_pct",
        "h2h_avg_point_diff",
        "h2h_avg_total_points",
    }
    assert exp_cols.issubset(out.columns)
    assert len(out) == len(sample_games_df)

    # Game 101 (Chiefs home vs Bills)
    r = out.loc[out["game_id"] == 101].iloc[0]

    # All 3 prior meetings considered
    assert r["h2h_games_played"] == 3

    # Home‑win flags in prior games: [0 (KC lost), 0 (BUF lost), 1 (KC won)] → mean = 1/3
    assert np.isclose(r["h2h_home_win_pct"], 1 / 3)

    # Point diffs: -4, -6, +3 → avg = -2.333
    assert np.isclose(r["h2h_avg_point_diff"], (-4 - 6 + 3) / 3)


def test_defaults_for_new_matchup(sample_games_df):
    out = h2h.transform(sample_games_df, historical_df=pd.DataFrame())

    assert (out["h2h_games_played"] == 0).all()
    assert (out["h2h_home_win_pct"] == utils.DEFAULTS["matchup_home_win_pct"]).all()
    assert (out["h2h_avg_total_points"] == utils.DEFAULTS["matchup_avg_total_points"]).all()


def test_max_games_window(sample_games_df, sample_hist_df):
    out = h2h.transform(sample_games_df, historical_df=sample_hist_df, max_games=2)
    r = out.loc[out["game_id"] == 101].iloc[0]

    # Still reports total games played
    assert r["h2h_games_played"] == 3

    # Last 2 meetings: home_win flags [0, 1] → mean 0.5
    assert np.isclose(r["h2h_home_win_pct"], 0.5)

    # Point diffs last 2: -6, +3 → avg -1.5
    assert np.isclose(r["h2h_avg_point_diff"], (-6 + 3) / 2)


def test_empty_input(sample_hist_df):
    assert h2h.transform(pd.DataFrame(), historical_df=sample_hist_df).empty
