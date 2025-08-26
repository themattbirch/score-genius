# backend/tests/nba_features/test_utils.py

import pandas as pd
import numpy as np
import pytest

import sys
import os

# ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)


from backend.nba_features import utils

# ---------- safe_divide ----------
def test_safe_divide_basic_and_edge_cases():
    num = pd.Series([1, 2, 3, np.nan])
    den = pd.Series([1, 2, 0, 2])

    res = utils.safe_divide(num, den, default_val=0.0)
    expected = pd.Series([1.0, 1.0, 0.0, 0.0])

    pd.testing.assert_series_equal(res, expected)

# ---------- normalize_team_name ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Golden State Warriors", "warriors"),
        ("BOS", "celtics"),         # abbreviation → mapping
        ("Made Up Team", "made up team"),  # unknown → cleaned lower‑case
        (None, "Unknown"),                # None → "Unknown"
        (np.nan, "Unknown"),              # NaN  → "Unknown"
    ],
)
def test_normalize_team_name_variants(raw, expected):
    assert utils.normalize_team_name(raw) == expected

# ---------- determine_season ----------
@pytest.mark.parametrize(
    "date_str,expected",
    [
        ("2023-08-15", "2022-2023"),  # Aug → previous season start
        ("2023-10-01", "2023-2024"),  # Oct → same‑year season start
    ],
)
def test_determine_season_boundaries(date_str, expected):
    assert utils.determine_season(pd.Timestamp(date_str)) == expected

def test_determine_season_missing():
    assert utils.determine_season(pd.NaT) == "Unknown_Season"

# ---------- generate_rolling_column_name ----------
def test_generate_rolling_column_name():
    assert (
        utils.generate_rolling_column_name("home", "pts", "avg", 10)
        == "home_rolling_pts_avg_10"
    )
    # prefix absent
    assert (
        utils.generate_rolling_column_name("", "reb", "sum", 5)
        == "rolling_reb_sum_5"
    )

# ---------- convert_and_fill ----------
def test_convert_and_fill_behavior():
    df = pd.DataFrame({"a": ["1", "2", None]})
    out = utils.convert_and_fill(df.copy(), ["a", "b"], default=0.0)

    # Column a converted to numeric with NaNs filled
    assert out["a"].tolist() == [1.0, 2.0, 0.0]
    # Column b created & filled
    assert (out["b"] == 0.0).all()

# ---------- DEFAULTS sanity ----------
def test_defaults_essential_keys():
    for key in ("win_pct", "avg_pts_for", "avg_pts_against", "rest_days"):
        assert key in utils.DEFAULTS
