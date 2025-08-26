"""
Wide-coverage tests for backend.nba_features.rest.transform

Focus
-----
1.  Strict “no-look-ahead” guarantee when multiple games share a date.
2.  Correct rest-day math, B2B flags, 7-/14-day density, and advantages.
3.  Idempotency + NaN-free outputs.
"""

from __future__ import annotations
import os
import sys

# Ensure project root is on the import path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import numpy as np
import pandas as pd
import pytest

from backend.nba_features import rest, utils

# Constants
DEF_REST = utils.DEFAULTS["rest_days"]
REST_COLS = [
    "rest_days_home",
    "rest_days_away",
    "is_back_to_back_home",
    "is_back_to_back_away",
    "rest_advantage",
    "schedule_advantage",
    "games_last_7_days_home",
    "games_last_7_days_away",
    "games_last_14_days_home",
    "games_last_14_days_away",
]

# Fixture: synthetic schedule
@pytest.fixture(scope="module")
def sched() -> pd.DataFrame:
    """
    Build 15 games across 3 teams, including duplicate dates and double-headers.
    """
    rows = [
        (201, "2023-01-01", "A", "C"),
        (202, "2023-01-02", "B", "A"),
        (203, "2023-01-03", "A", "B"),
        (204, "2023-01-03", "C", "A"),
        (205, "2023-01-04", "B", "A"),
        (206, "2023-01-04", "A", "C"),
        (207, "2023-01-05", "C", "B"),
        (208, "2023-01-06", "A", "B"),
        (209, "2023-01-07", "B", "C"),
        (210, "2023-01-08", "A", "B"),
        (211, "2023-01-08", "B", "A"),
        (212, "2023-01-10", "C", "A"),
        (213, "2023-01-12", "B", "C"),
        (214, "2023-01-13", "C", "A"),
        (215, "2023-01-15", "A", "B"),
    ]
    df = pd.DataFrame(rows, columns=["game_id", "game_date", "home_team", "away_team"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

# Helper: ensure columns exist
def _assert_cols(df: pd.DataFrame) -> None:
    for c in REST_COLS:
        assert c in df.columns, f"Missing {c}"

# Tests
def test_columns_and_shape(sched):
    out = rest.transform(sched.copy(), debug=False)
    _assert_cols(out)
    assert len(out) == len(sched)

def test_first_appearance_defaults(sched):
    out = rest.transform(sched.copy())
    # Determine each team's first overall game_id
    flat = pd.concat([
        sched[["game_id","home_team","game_date"]].rename(columns={"home_team":"team"}),
        sched[["game_id","away_team","game_date"]].rename(columns={"away_team":"team"})
    ], ignore_index=True)
    flat = flat.sort_values(by=["team","game_date","game_id"], ignore_index=True)
    first_ids = flat.drop_duplicates(subset="team", keep="first").set_index("team")["game_id"].to_dict()

    is_first_home = out["game_id"] == out["home_team"].map(first_ids)
    is_first_away = out["game_id"] == out["away_team"].map(first_ids)

    assert (out.loc[is_first_home, "rest_days_home"] == DEF_REST).all()
    assert (out.loc[is_first_away, "rest_days_away"] == DEF_REST).all()

    # Verify computed rest on a non-first game
    row_202 = out.query("game_id == 202").iloc[0]
    assert row_202["rest_days_away"] == 1.0  # A: Jan 1 → Jan 2

def test_no_forward_peek_same_day(sched):
    out = rest.transform(sched.copy())
    game_204 = out.query("game_id == 204").iloc[0]
    assert game_204["rest_days_home"] == 2.0
    assert game_204["rest_days_away"] == 1.0

def test_back_to_back_flags(sched):
    out = rest.transform(sched.copy())
    idx_205 = out.query("game_id == 205").index[0]
    assert out.loc[idx_205, "is_back_to_back_home"] == 1
    assert out.loc[idx_205, "rest_days_home"] == 1.0

    idx_210 = out.query("game_id == 210").index[0]
    assert out.loc[idx_210, "is_back_to_back_home"] == 0
    assert out.loc[idx_210, "rest_days_home"] == 2.0

def test_rest_and_schedule_advantage(sched):
    out = rest.transform(sched.copy())
    row_211 = out.query("game_id == 211").iloc[0]
    assert row_211["rest_days_home"] == 1.0
    assert row_211["rest_days_away"] == 2.0
    assert row_211["rest_advantage"] == -1.0
    assert row_211["schedule_advantage"] == (
        row_211["games_last_7_days_away"] - row_211["games_last_7_days_home"]
    )

def test_games_count_window(sched):
    """
    Test Game 215 (A vs B on Jan 15). Count games in prior 7/14 days.
    Window is (date - N days, date), i.e., excluding the game date itself.

    7-Day Window (Jan 8 - Jan 14):
    - Home (A): Played Jan 10 (vs C), Jan 13 (vs C) -> Count = 2
    - Away (B): Played Jan 12 (vs C) -> Count = 1

    14-Day Window (Jan 1 - Jan 14):
    - Home (A): Distinct dates played: Jan 2, 3, 4, 6, 8, 10, 13 -> Count = 7
    - Away (B): Distinct dates played: Jan 2, 3, 4, 5, 6, 7, 8, 12 -> Count = 8
    """
    out = rest.transform(sched.copy())
    row = out.query("game_id == 215").iloc[0]

    # 7-Day Assertions
    assert row["games_last_7_days_home"] == 2
    assert row["games_last_7_days_away"] == 1

    # 14-Day Assertions
    assert row["games_last_14_days_home"] == 7
    assert row["games_last_14_days_away"] == 8

def test_idempotent(sched):
    first = rest.transform(sched.copy())
    second = rest.transform(first.copy())
    pd.testing.assert_frame_equal(first[REST_COLS], second[REST_COLS], check_dtype=False)

def test_no_nans(sched):
    out = rest.transform(sched.copy())
    assert not out[REST_COLS].isna().any().any()
