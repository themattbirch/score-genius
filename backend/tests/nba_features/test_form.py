# backend/tests/nba_features/test_form.py

import numpy as np
import pandas as pd
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

from backend.nba_features import form, utils

# ---------- Helper ----------
def assert_form_row(row, exp):
    """Compare a DataFrame row’s form metrics to expected values."""
    for k, v in exp.items():
        assert np.isclose(row[k], v)


# ---------- Fixtures ----------
@pytest.fixture
def sample_form_df() -> pd.DataFrame:
    """Three games covering normal, partially‑missing, and blank form strings."""
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "home_current_form": ["WWLWW", "N/A", ""],
            "away_current_form": ["LLWLL", "WW", ""],
        }
    )


# ---------- Unit tests for private helper ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "WWLWW",
            dict(form_win_pct=0.8, current_streak=2, momentum_direction=1.0),
        ),
        (
            "LLWLL",
            dict(form_win_pct=0.2, current_streak=-2, momentum_direction=-1.0),
        ),
        ("WW", dict(form_win_pct=1.0, current_streak=2, momentum_direction=0.0)),
        ("N/A", dict(  # defaults
            form_win_pct=utils.DEFAULTS["form_win_pct"],
            current_streak=utils.DEFAULTS["current_streak"],
            momentum_direction=utils.DEFAULTS["momentum_direction"],
        )),
        (np.nan, dict(
            form_win_pct=utils.DEFAULTS["form_win_pct"],
            current_streak=utils.DEFAULTS["current_streak"],
            momentum_direction=utils.DEFAULTS["momentum_direction"],
        )),
    ],
)
def test_extract_form_metrics_single(raw, expected):
    out = form._extract_form_metrics_single(raw)
    assert_form_row(pd.Series(out), expected)


# ---------- transform() integration ----------
def test_transform_metrics_and_diffs(sample_form_df: pd.DataFrame):
    df = form.transform(sample_form_df.copy())

    required_cols = [
        "home_form_win_pct",
        "away_form_win_pct",
        "home_current_streak",
        "away_current_streak",
        "home_momentum_direction",
        "away_momentum_direction",
        "form_win_pct_diff",
        "streak_advantage",
        "momentum_diff",
    ]
    for col in required_cols:
        assert col in df.columns
        assert not df[col].isnull().any()

    # Row 0 – expected explicit values
    row0 = df.loc[0]
    assert_form_row(
        row0,
        {
            "home_form_win_pct": 0.8,
            "away_form_win_pct": 0.2,
            "home_current_streak": 2,
            "away_current_streak": -2,
            "home_momentum_direction": 1.0,
            "away_momentum_direction": -1.0,
            "form_win_pct_diff": 0.8 - 0.2,
            "streak_advantage": 2 - (-2),
            "momentum_diff": 1.0 - (-1.0),
        },
    )

    # Row 1 – home defaults, away partial string (“WW”)
    defaults = {
        "win_pct": utils.DEFAULTS["form_win_pct"],
        "streak": utils.DEFAULTS["current_streak"],
        "momentum": utils.DEFAULTS["momentum_direction"],
    }
    row1 = df.loc[1]
    assert_form_row(
        row1,
        {
            "home_form_win_pct": defaults["win_pct"],
            "home_current_streak": defaults["streak"],
            "home_momentum_direction": defaults["momentum"],
            "away_form_win_pct": 1.0,
            "away_current_streak": 2,
            "away_momentum_direction": 0.0,
        },
    )
    # diff arithmetic always holds
    assert np.isclose(
        row1["form_win_pct_diff"],
        row1["home_form_win_pct"] - row1["away_form_win_pct"],
    )
    assert np.isclose(
        row1["streak_advantage"],
        row1["home_current_streak"] - row1["away_current_streak"],
    )
    assert np.isclose(
        row1["momentum_diff"],
        row1["home_momentum_direction"] - row1["away_momentum_direction"],
    )


def test_transform_missing_form_columns():
    """If form columns are absent, transform should add columns filled with defaults."""
    df_in = pd.DataFrame({"game_id": [1], "some_other": [123]})
    out = form.transform(df_in.copy())

    # All placeholder columns exist and equal defaults / zero.
    def_win = utils.DEFAULTS["form_win_pct"]
    def_stk = utils.DEFAULTS["current_streak"]
    def_mom = utils.DEFAULTS["momentum_direction"]

    assert out["home_form_win_pct"].iloc[0] == def_win
    assert out["away_form_win_pct"].iloc[0] == def_win
    assert out["home_current_streak"].iloc[0] == def_stk
    assert out["away_current_streak"].iloc[0] == def_stk
    assert out["home_momentum_direction"].iloc[0] == def_mom
    assert out["away_momentum_direction"].iloc[0] == def_mom
    assert out["form_win_pct_diff"].iloc[0] == 0.0
    assert out["streak_advantage"].iloc[0] == 0.0
    assert out["momentum_diff"].iloc[0] == 0.0
