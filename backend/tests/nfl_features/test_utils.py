# backend/tests/nfl_features/test_utils.py

"""
Unit‑tests for backend.nfl_features.utils
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import utils

# --------------------------------------------------------------------------- #
# Fixtures & helpers                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear LRU cache on normalize_team_name after each test."""
    yield
    utils.normalize_team_name.cache_clear()


# --------------------------------------------------------------------------- #
# Team‑name normalisation                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, canon",
    [
        ("Green Bay Packers", "packers"),
        ("GB", "packers"),
        ("packers", "packers"),
        ("Oakland Raiders", "raiders"),          # historical
        ("  cleveland browns  ", "browns"),      # whitespace
        ("dAlLaS cOwBoYs", "cowboys"),           # case
        ("Unknown Franchise", "unknown_team"),   # miss
        ("", "unknown_team"),
        (None, "unknown_team"),
        (np.nan, "unknown_team"),
    ],
)
def test_normalize_team_name(raw, canon):
    assert utils.normalize_team_name(raw) == canon


def test_normalize_team_name_id_map(monkeypatch):
    monkeypatch.setattr(utils, "NFL_ID_MAP", {"123": "chiefs"})
    assert utils.normalize_team_name("123") == "chiefs"          # ID wins
    assert utils.normalize_team_name("packers") == "packers"     # falls back


# --------------------------------------------------------------------------- #
# Season determination                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "date_str, exp_season",
    [
        ("2023-09-10", 2023),
        ("2024-01-07", 2023),  # Week 18
        ("2024-02-11", 2023),  # SB LVIII
        ("2024-03-15", 2024),
        ("2024-04-25", 2024),
    ],
)
def test_determine_season(date_str, exp_season):
    ts = pd.Timestamp(date_str)
    assert utils.determine_season(ts) == exp_season


def test_determine_season_nat():
    now = pd.Timestamp.now()
    exp = now.year if now.month >= 3 else now.year - 1
    assert utils.determine_season(pd.NaT) == exp


# --------------------------------------------------------------------------- #
# Safe division / compute_rate                                                #
# --------------------------------------------------------------------------- #
def test_safe_divide_and_compute_rate():
    num = pd.Series([100, 20, 5, np.nan, 50, -99])
    den = pd.Series([10, 4, 0, 10, np.nan, 33])

    # compute_rate (alias w/ fill)
    out_rate = utils.compute_rate(num, den, fill=0.0)
    exp_rate = pd.Series([10.0, 5.0, 0.0, 0.0, 0.0, -3.0])
    pd.testing.assert_series_equal(out_rate, exp_rate, check_exact=True)

    # safe_divide (explicit default_val)
    out_sd = utils.safe_divide(num, den, default_val=-1.0)
    exp_sd = pd.Series([10.0, 5.0, -1.0, -1.0, -1.0, -3.0])
    pd.testing.assert_series_equal(out_sd, exp_sd, check_exact=True)


# --------------------------------------------------------------------------- #
# Column prefixer                                                             #
# --------------------------------------------------------------------------- #
def test_prefix_columns():
    df = pd.DataFrame({"game_id": [1, 2], "a": [10, 20], "b": [100, 200]})
    res = utils.prefix_columns(df, "home", exclude=["game_id"])
    assert list(res.columns) == ["game_id", "home_a", "home_b"]


# --------------------------------------------------------------------------- #
# DEFAULTS sanity                                                             #
# --------------------------------------------------------------------------- #
def test_defaults_keys_present():
    required = {
        "win_pct",
        "points_for_avg",
        "srs_lite",
        "elo",
        "yards_per_play_avg",
        "third_down_pct",
        "matchup_home_win_pct",
        "days_since_last_game",
    }
    missing = required - utils.DEFAULTS.keys()
    assert not missing, f"Missing keys in DEFAULTS: {missing}"
