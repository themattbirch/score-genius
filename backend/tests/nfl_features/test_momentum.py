# backend/tests/nfl_features/test_momentum.py

from __future__ import annotations
"""
Unit tests for backend.nfl_features.momentum.transform
"""

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import momentum, utils


# --------------------------------------------------------------------------- #
# Auto‑fixture: clear team‑name cache after every test                         #
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
            dict(
                game_id=101,
                game_date="2024-10-20",
                home_team_norm="team_a",
                away_team_norm="team_b",
            ),
            dict(
                game_id=102,
                game_date="2024-10-20",
                home_team_norm="team_c",
                away_team_norm="team_a",
            ),
        ]
    )


@pytest.fixture(scope="module")
def sample_hist_df() -> pd.DataFrame:
    rows = [
        # Team A diffs: +10, ‑7, +21
        dict(
            game_date="2024-09-15",
            home_team_norm="team_a",
            away_team_norm="opponent1",
            home_score=30,
            away_score=20,
        ),
        dict(
            game_date="2024-09-22",
            home_team_norm="opponent2",
            away_team_norm="team_a",
            home_score=28,
            away_score=21,
        ),
        dict(
            game_date="2024-09-29",
            home_team_norm="team_a",
            away_team_norm="opponent3",
            home_score=35,
            away_score=14,
        ),
        # Team B diffs: ‑3, ‑14, +7
        dict(
            game_date="2024-09-15",
            home_team_norm="team_b",
            away_team_norm="opponent1",
            home_score=17,
            away_score=20,
        ),
        dict(
            game_date="2024-09-22",
            home_team_norm="opponent2",
            away_team_norm="team_b",
            home_score=31,
            away_score=17,
        ),
        dict(
            game_date="2024-09-29",
            home_team_norm="team_b",
            away_team_norm="opponent3",
            home_score=28,
            away_score=21,
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_momentum_values_and_schema(sample_games_df, sample_hist_df):
    span = 3
    out = momentum.transform(sample_games_df, historical_df=sample_hist_df, span=span)

    exp_cols = {
        f"home_momentum_ewma_{span}",
        f"away_momentum_ewma_{span}",
        f"momentum_ewma_{span}_diff",
    }
    assert exp_cols.issubset(out.columns)
    assert len(out) == len(sample_games_df)

    row = out.loc[out["game_id"] == 101].iloc[0]

    # Manual EWMA calc (adjust=False)
    ewma_a = pd.Series([10, -7, 21]).ewm(span=span, adjust=False).mean().iloc[-1]
    ewma_b = pd.Series([-3, -14, 7]).ewm(span=span, adjust=False).mean().iloc[-1]

    assert np.isclose(row[f"home_momentum_ewma_{span}"], ewma_a)
    assert np.isclose(row[f"away_momentum_ewma_{span}"], ewma_b)
    assert np.isclose(row[f"momentum_ewma_{span}_diff"], ewma_a - ewma_b)


def test_imputation_for_new_team(sample_games_df, sample_hist_df):
    out = momentum.transform(sample_games_df, historical_df=sample_hist_df)

    row = out.loc[out["game_id"] == 102].iloc[0]
    assert row["home_momentum_ewma_5"] == 0.0
    assert row["home_momentum_ewma_5_imputed"] == 1


def test_all_defaults_without_history(sample_games_df):
    out = momentum.transform(sample_games_df, historical_df=pd.DataFrame())

    assert (out.filter(like="momentum_ewma") == 0.0).all().all()


def test_span_parameter_effect(sample_games_df, sample_hist_df):
    span5_val = momentum.transform(sample_games_df, historical_df=sample_hist_df, span=5)\
        .loc[0, "home_momentum_ewma_5"]
    span2_val = momentum.transform(sample_games_df, historical_df=sample_hist_df, span=2)\
        .loc[0, "home_momentum_ewma_2"]

    assert not np.isclose(span5_val, span2_val), "Different spans should give different values"


def test_empty_games_returns_empty(sample_hist_df):
    res = momentum.transform(pd.DataFrame(), historical_df=sample_hist_df)
    assert res.empty and res.columns.size == 0
