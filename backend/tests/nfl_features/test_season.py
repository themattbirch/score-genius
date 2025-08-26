# backend/tests/nfl_features/test_season.py

"""
Unit tests for backend.nfl_features.season
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import season, utils


# --------------------------------------------------------------------------- #
# Utility fixture: clear LRU cache                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    utils.normalize_team_name.cache_clear()


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """Upcoming games to be featurised."""
    return pd.DataFrame(
        [
            dict(
                game_id="g2024_1",
                season=2024,
                home_team_norm="chiefs",
                away_team_norm="ravens",
            ),
            dict(
                game_id="g2024_2",
                season=2024,
                home_team_norm="packers",
                away_team_norm="saints",  # will trigger default imputation
            ),
            dict(
                game_id="g2023_1",
                season=2023,
                home_team_norm="lions",
                away_team_norm="49ers",
            ),
        ]
    )


@pytest.fixture
def sample_historical_team_stats_df() -> pd.DataFrame:
    """Prior‑season aggregates used for look‑ups."""
    rows = [
        # 2023 season stats (feed 2024 games)
        dict(team_norm="chiefs", season=2023, won=11, lost=6, ties=0,
             points_for=371, points_against=294, points_difference=77, srs_lite=5.1),
        dict(team_norm="ravens", season=2023, won=13, lost=4, ties=0,
             points_for=483, points_against=280, points_difference=203, srs_lite=11.9),
        dict(team_norm="packers", season=2023, won=9, lost=8, ties=0,
             points_for=383, points_against=350, points_difference=33, srs_lite=1.8),
        # (saints 2023 intentionally missing)
        # 2022 season stats (feed 2023 games)
        dict(team_norm="lions", season=2022, won=9, lost=8, ties=0,
             points_for=453, points_against=429, points_difference=24, srs_lite=1.4),
        dict(team_norm="49ers", season=2022, won=13, lost=4, ties=0,
             points_for=450, points_against=277, points_difference=173, srs_lite=10.3),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_core_transform_and_schema(sample_games_df, sample_historical_team_stats_df):
    out = season.transform(sample_games_df, historical_team_stats_df=sample_historical_team_stats_df)
    exp_cols = {
        "home_prev_season_win_pct",
        "away_prev_season_win_pct",
        "prev_season_win_pct_diff",
        "home_prev_season_srs_lite",
        "away_prev_season_srs_lite",
        "prev_season_srs_lite_diff",
    }
    assert exp_cols.issubset(out.columns)
    assert len(out) == len(sample_games_df)

    chiefs_wp = 11 / 17
    ravens_wp = 13 / 17
    row = out.loc[out["game_id"] == "g2024_1"].iloc[0]
    assert np.isclose(row["home_prev_season_win_pct"], chiefs_wp)
    assert np.isclose(row["away_prev_season_win_pct"], ravens_wp)
    assert np.isclose(row["prev_season_win_pct_diff"], chiefs_wp - ravens_wp)
    assert np.isclose(row["home_prev_season_srs_lite"], 5.1)
    assert np.isclose(row["away_prev_season_srs_lite"], 11.9)


def test_imputation_flags(sample_games_df, sample_historical_team_stats_df):
    out = season.transform(sample_games_df, historical_team_stats_df=sample_historical_team_stats_df)

    # Saints row should be imputed
    saints_row = out.loc[out["game_id"] == "g2024_2"].iloc[0]
    assert saints_row["away_prev_season_win_pct"] == utils.DEFAULTS["win_pct"]
    assert saints_row["away_prev_season_win_pct_imputed"] == 1
    assert saints_row["away_prev_season_srs_lite_imputed"] == 1

    # Packers should not be imputed
    assert saints_row["home_prev_season_win_pct_imputed"] == 0


def test_all_defaults_when_no_history(sample_games_df):
    out = season.transform(sample_games_df, historical_team_stats_df=pd.DataFrame())
    assert (out["home_prev_season_win_pct"] == utils.DEFAULTS["win_pct"]).all()
    assert (out["prev_season_srs_lite_diff"] == 0.0).all()
    assert (out.filter(like="_imputed") == 1).all().all()


def test_season_n_minus_1_lookup(sample_games_df, sample_historical_team_stats_df):
    out = season.transform(sample_games_df, historical_team_stats_df=sample_historical_team_stats_df)
    lions_row = out.loc[out["game_id"] == "g2023_1"].iloc[0]
    assert np.isclose(lions_row["home_prev_season_win_pct"], 9 / 17)
    assert np.isclose(lions_row["away_prev_season_srs_lite"], 10.3)


def test_flag_imputations_false(sample_games_df, sample_historical_team_stats_df):
    out = season.transform(sample_games_df,
                           historical_team_stats_df=sample_historical_team_stats_df,
                           flag_imputations=False)
    assert not any(col.endswith("_imputed") for col in out.columns)


def test_empty_games():
    assert season.transform(pd.DataFrame(), historical_team_stats_df=pd.DataFrame()).empty
