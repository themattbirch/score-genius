# backend/tests/nfl_features/test_rest.py

from __future__ import annotations
"""
Unit tests for backend.nfl_features.rest.transform
"""

import pandas as pd
import pytest

from backend.nfl_features import rest, utils


# --------------------------------------------------------------------------- #
# Auto‑fixture: clear team‑name cache                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    utils.normalize_team_name.cache_clear()


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Game‑101: Team A vs Team B
            dict(
                game_id=101,
                game_date="2024-10-20",
                home_team_norm="team_a",
                away_team_norm="team_b",
            ),
            # Game‑102: Team C vs Team D
            dict(
                game_id=102,
                game_date="2024-10-20",
                home_team_norm="team_c",
                away_team_norm="team_d",
            ),
        ]
    )

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Mocks the normalize_team_name function within the rest module to prevent
    it from converting our test team names (e.g., 'team_a') to 'unknown_team'.
    """
    # A simple pass-through function is all we need for this test's data
    def mock_normalize(team_name):
        return team_name

    monkeypatch.setattr(rest, "normalize_team_name", mock_normalize)


@pytest.fixture(scope="module")
def sample_hist_df() -> pd.DataFrame:
    rows = [
        # Team A previous Sunday (normal 7‑day gap)
        dict(
            game_date="2024-10-13",
            home_team_norm="team_a",
            away_team_norm="opponent1",
            home_score=1,
            away_score=0,
        ),
        # Team B played Sun + Thu → short week (2 rest days to Sun 20th)
        dict(
            game_date="2024-10-13",
            home_team_norm="opponent2",
            away_team_norm="team_b",
            home_score=0,
            away_score=1,
        ),
        dict(
            game_date="2024-10-17",
            home_team_norm="team_b",
            away_team_norm="opponent3",
            home_score=1,
            away_score=0,
        ),
        # Team C last played two weeks prior → bye
        dict(
            game_date="2024-10-06",
            home_team_norm="team_c",
            away_team_norm="opponent4",
            home_score=1,
            away_score=0,
        ),
        # Team D has no history
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_rest_schema_and_values(sample_games_df, sample_hist_df):
    out = rest.transform(sample_games_df, historical_df=sample_hist_df)

    exp_cols = {
        "home_rest_days",
        "away_rest_days",
        "home_is_on_short_week",
        "away_is_on_short_week",
        "home_is_off_bye",
        "away_is_off_bye",
        "rest_advantage",
    }
    assert exp_cols.issubset(out.columns)
    assert len(out) == len(sample_games_df)

    # Game‑101 checks
    g101 = out.loc[out["game_id"] == 101].iloc[0]
    assert g101["home_rest_days"] == 6  # 20‑Oct minus 13‑Oct minus 1
    assert g101["home_is_on_short_week"] == 0
    assert g101["away_rest_days"] == 2  # 20‑Oct minus 17‑Oct minus 1
    assert g101["away_is_on_short_week"] == 1
    assert g101["rest_advantage"] == 6 - 2


def test_off_bye_flag(sample_games_df, sample_hist_df):
    out = rest.transform(sample_games_df, historical_df=sample_hist_df)
    g102 = out.loc[out["game_id"] == 102].iloc[0]
    # Team C gap: 20‑Oct minus 06‑Oct = 14 → 13 rest days
    assert g102["home_rest_days"] == 13
    assert g102["home_is_off_bye"] == 1
    # Team D (no history) → defaults
    assert g102["away_rest_days"] == utils.DEFAULTS["days_since_last_game"]


def test_defaults_no_history(sample_games_df):
    out = rest.transform(sample_games_df, historical_df=pd.DataFrame())
    
    # FIX: Explicitly check only the rest_days value columns
    rest_days_cols = ["home_rest_days", "away_rest_days"]
    assert (out[rest_days_cols] == utils.DEFAULTS["days_since_last_game"]).all().all()
    
    assert (out["rest_advantage"] == 0).all()


def test_empty_games_returns_empty(sample_hist_df):
    assert rest.transform(pd.DataFrame(), historical_df=sample_hist_df).empty
