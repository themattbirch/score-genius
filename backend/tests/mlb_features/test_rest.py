# backend/tests/mlb_features/test_rest.py

import os
import sys
import numpy as np
import pandas as pd
import pytest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.mlb_features import rest # Module to test

# --- Test Constants ---
TEST_MLB_DEF_REST = 3.0 # Define a specific default for testing
MLB_REST_COLS = rest.PLACEHOLDER_COLS

@pytest.fixture(autouse=True)
def mock_mlb_rest_defaults(monkeypatch):
    """Mock the default rest days value used in the rest module."""
    monkeypatch.setattr(rest, "DEF_REST_MLB", TEST_MLB_DEF_REST)
    # If rest.MLB_DEFAULTS had other keys used by _fill_defaults_mlb,
    # they could be mocked here too, but _fill_defaults_mlb primarily uses DEF_REST_MLB
    # or hardcoded 0 for other types.
    monkeypatch.setattr(rest, "MLB_DEFAULTS", {"mlb_rest_days": TEST_MLB_DEF_REST})


# --- Fixtures ---
@pytest.fixture(scope="module")
def sched_mlb() -> pd.DataFrame:
    """
    Synthetic MLB schedule: 15 games, 3 teams (NYY, BOS, TBR), duplicate dates.
    Column names match expected input for mlb_features.rest.
    """
    rows = [
        # Game ID, Date, Home Team, Away Team
        ("g01", "2023-04-01", "NYY", "TBR"), # NYY, TBR first game
        ("g02", "2023-04-02", "BOS", "NYY"), # BOS first game, NYY 2nd game (1 day rest)
        ("g03", "2023-04-03", "NYY", "BOS"), # NYY 3rd game (0 day rest for BOS, 1 day for NYY)
        ("g04", "2023-04-03", "TBR", "NYY"), # TBR 2nd game (1 day rest for TBR), NYY 4th game (NYY plays twice on 04-03)
        ("g05", "2023-04-04", "BOS", "NYY"), # BOS 3rd (0 day rest), NYY 5th (0 day rest from 2nd game on 04-03)
        ("g06", "2023-04-04", "NYY", "TBR"), # NYY 6th (NYY plays twice on 04-04), TBR 3rd (0 day rest)
        ("g07", "2023-04-05", "TBR", "BOS"), # TBR 4th (0 day rest), BOS 4th (0 day rest)
        ("g08", "2023-04-06", "NYY", "BOS"),
        ("g09", "2023-04-07", "BOS", "TBR"),
        ("g10", "2023-04-08", "NYY", "BOS"), # NYY, BOS
        ("g11", "2023-04-08", "BOS", "NYY"), # BOS, NYY (double header for both if same day processing matters for prev_date)
                                            # Script handles this by drop_duplicates on (team, date) before shift
        ("g12", "2023-04-10", "TBR", "NYY"), # 1 day rest for NYY from g11 (04-08), TBR has 2 days rest from g09 (04-07)
        ("g13", "2023-04-12", "BOS", "TBR"),
        ("g14", "2023-04-13", "TBR", "NYY"),
        ("g15", "2023-04-15", "NYY", "BOS"), # Target for window count test
        ("g16", "2023-04-16", "XXX", "YYY"), # Teams with no prior history in this set
        # Game with invalid date
        ("g17", "INVALID_DATE", "NYY", "BOS"),
    ]
    df = pd.DataFrame(rows, columns=["game_id", "game_date_et", "home_team_id", "away_team_id"])
    # game_date_et is converted inside transform, so keep as string here for some rows
    # df["game_date_et"] = pd.to_datetime(df["game_date_et"]) # Done inside transform
    return df.sort_values(["game_date_et", "game_id"]).reset_index(drop=True)

# --- Helper ---
def _assert_cols_mlb(df: pd.DataFrame) -> None:
    for c in MLB_REST_COLS:
        assert c in df.columns, f"Missing MLB rest column: {c}"
        # Check dtypes based on _fill_defaults_mlb
        if c.startswith("is_back_to_back") or c.startswith("games_last_"):
            assert df[c].dtype == int, f"Column {c} dtype is {df[c].dtype}, expected int"
        else: # rest_days_ or advantage cols
            assert df[c].dtype == float, f"Column {c} dtype is {df[c].dtype}, expected float"

# --- Tests ---
def test_columns_and_shape_mlb(sched_mlb):
    # Exclude the row with invalid date which will be dropped
    sched_valid_dates = sched_mlb[sched_mlb["game_date_et"] != "INVALID_DATE"].copy()
    out_df = rest.transform(sched_valid_dates, debug=False)
    _assert_cols_mlb(out_df)
    assert len(out_df) == len(sched_valid_dates)

def test_first_appearance_defaults_mlb(sched_mlb):
    out_df = rest.transform(sched_mlb.copy()) # transform handles invalid dates by dropping

    # Identify each team's actual first game_id in the dataset (after date parsing and potential drops)
    # Need to mimic the date parsing and dropping of invalid dates from transform()
    df_proc = sched_mlb.copy()
    df_proc["game_date"] = pd.to_datetime(df_proc["game_date_et"], errors="coerce").dt.tz_localize(None)
    df_proc = df_proc.dropna(subset=["game_date"])

    home_log = df_proc[["game_id", "game_date", "home_team_id"]].rename(columns={"home_team_id": "team"})
    away_log = df_proc[["game_id", "game_date", "away_team_id"]].rename(columns={"away_team_id": "team"})
    full_log = pd.concat([home_log, away_log], ignore_index=True)
    
    first_game_ids_map = (
        full_log.sort_values(["team", "game_date", "game_id"])
        .drop_duplicates("team", keep="first")
        .set_index("team")["game_id"]
    )
    
    is_first_home_game = out_df["game_id"] == out_df["home_team_id"].map(first_game_ids_map)
    is_first_away_game = out_df["game_id"] == out_df["away_team_id"].map(first_game_ids_map)

    assert (out_df.loc[is_first_home_game, "rest_days_home"] == TEST_MLB_DEF_REST).all()
    assert (out_df.loc[is_first_away_game, "rest_days_away"] == TEST_MLB_DEF_REST).all()
    
    # Test teams XXX, YYY from g16 - they should have default rest
    g16_row = out_df[out_df["game_id"] == "g16"].iloc[0]
    assert g16_row["rest_days_home"] == TEST_MLB_DEF_REST
    assert g16_row["rest_days_away"] == TEST_MLB_DEF_REST


def test_no_forward_peek_same_day_mlb(sched_mlb):
    """NYY plays twice on 2023-04-03 (g03 home vs BOS, g04 away vs TBR).
       NYY plays twice on 2023-04-04 (g05 away vs BOS, g06 home vs TBR).
       The rest for NYY for their second game on these days should be 0.
       Their rest for the *next* day (04-04 or 04-05) should be 1.
    """
    out_df = rest.transform(sched_mlb.copy())

    # Game g04: NYY (away) played g03 (home) earlier on 2023-04-03. Rest for NYY should be 0.
    # NYY's actual prev game date was 04-02 (g02).
    # Script's prev_log uses drop_duplicates(["team", "game_date"], keep="first").
    # So for NYY on 04-03, only one entry for 04-03 is kept in prev_log for calculating shift.
    # prev_game_date for NYY on 04-03 (for g03) is 04-02. Rest = 1.
    # prev_game_date for NYY on 04-03 (for g04) is also 04-02. Rest = 1. This is correct.
    # A team playing two games on the same day has the same rest days for both, based on the prior distinct game date.
    g04_row = out_df[out_df["game_id"] == "g04"].iloc[0] # NYY is away
    assert g04_row["rest_days_away"] == 1.0 # NYY played g02 on 04-02. (04-03 - 04-02 = 1 day)

    # Game g05: NYY (away) on 2023-04-04. NYY played twice on 04-03. Prev distinct game date for NYY is 04-03.
    g05_row = out_df[out_df["game_id"] == "g05"].iloc[0] # NYY is away
    assert g05_row["rest_days_away"] == 1.0 # (04-04 - 04-03 = 1 day)

    # Game g06: NYY (home) on 2023-04-04. Same as g05 for NYY's rest.
    g06_row = out_df[out_df["game_id"] == "g06"].iloc[0] # NYY is home
    assert g06_row["rest_days_home"] == 1.0

def test_back_to_back_flags_mlb(sched_mlb):
    out_df = rest.transform(sched_mlb.copy())
    # Game g05 (04-04): BOS (home) played g03 on 04-03. NYY (away) played on 04-03. Both B2B.
    g05_row = out_df[out_df["game_id"] == "g05"].iloc[0]
    assert g05_row["rest_days_home"] == 1.0 # BOS played 04-03 (g03)
    assert g05_row["is_back_to_back_home"] == 1
    assert g05_row["rest_days_away"] == 1.0 # NYY played 04-03 (g03, g04)
    assert g05_row["is_back_to_back_away"] == 1

    # Game g08 (04-06): NYY (home) played g06 on 04-04. BOS (away) played g07 on 04-05.
    g08_row = out_df[out_df["game_id"] == "g08"].iloc[0]
    assert g08_row["rest_days_home"] == 2.0 # NYY (04-06 vs 04-04)
    assert g08_row["is_back_to_back_home"] == 0
    assert g08_row["rest_days_away"] == 1.0 # BOS (04-06 vs 04-05)
    assert g08_row["is_back_to_back_away"] == 1


def test_rest_and_schedule_advantage_mlb(sched_mlb):
    out_df = rest.transform(sched_mlb.copy())
    # Game g12 (04-10): TBR (home) vs NYY (away)
    # TBR: last game g09 (04-07). Rest for g12 = 3 days.
    # NYY: last distinct game date was g10/g11 on 04-08. Rest for g12 = 2 days.
    g12_row = out_df[out_df["game_id"] == "g12"].iloc[0]
    assert g12_row["rest_days_home"] == 3.0 # TBR
    assert g12_row["rest_days_away"] == 2.0 # NYY
    assert g12_row["rest_advantage"] == (3.0 - 2.0)
    assert g12_row["schedule_advantage"] == (g12_row["games_last_7_days_away"] - g12_row["games_last_7_days_home"])

def test_games_count_window_mlb(sched_mlb):
    """Test Game g15 (NYY vs BOS on 2023-04-15)."""
    out_df = rest.transform(sched_mlb.copy())
    row_g15 = out_df[out_df["game_id"] == "g15"].iloc[0]

    # For NYY (home) on 2023-04-15:
    # 7-day window: (2023-04-08 to 2023-04-14)
    # NYY games: g10/g11 (04-08, counted as 1 date), g12 (04-10), g14 (04-13). Count = 3.
    # 14-day window: (2023-04-01 to 2023-04-14)
    # NYY games: g01(04-01), g02(04-02), g03/g04(04-03), g05/g06(04-04), g08(04-06), g10/g11(04-08), g12(04-10), g14(04-13). Count = 8 distinct dates.
    assert row_g15["games_last_7_days_home"] == 3 # NYY
    assert row_g15["games_last_14_days_home"] == 8 # NYY

    # For BOS (away) on 2023-04-15:
    # 7-day window: (2023-04-08 to 2023-04-14)
    # BOS games: g10/g11 (04-08), g13 (04-12). Count = 2.
    # 14-day window: (2023-04-01 to 2023-04-14)
    # BOS games: g02(04-02), g03(04-03), g05(04-04), g07(04-05), g08(04-06), g09(04-07), g10/g11(04-08), g13(04-12). Count = 8 distinct dates.
    assert row_g15["games_last_7_days_away"] == 2 # BOS
    assert row_g15["games_last_14_days_away"] == 8 # BOS

def test_idempotent_mlb(sched_mlb):
    # Need to use a df with valid dates only for this test, as transform drops invalid ones.
    sched_valid_dates = sched_mlb[sched_mlb["game_date_et"] != "INVALID_DATE"].copy()
    first_run = rest.transform(sched_valid_dates.copy())
    second_run = rest.transform(first_run.copy()) # Pass the already transformed df
    # Compare only the placeholder columns as other columns might be intermediate or from input
    pd.testing.assert_frame_equal(first_run[MLB_REST_COLS], second_run[MLB_REST_COLS], check_dtype=True)

def test_no_nans_mlb(sched_mlb):
    out_df = rest.transform(sched_mlb[sched_mlb["game_date_et"] != "INVALID_DATE"].copy())
    assert not out_df[MLB_REST_COLS].isna().any().any()

def test_empty_input_df_mlb():
    empty_df = pd.DataFrame(columns=["game_id", "game_date_et", "home_team_id", "away_team_id"])
    out_df = rest.transform(empty_df.copy())
    assert len(out_df) == 0 # Should return an empty df with placeholder cols
    _assert_cols_mlb(out_df) # Checks columns exist and have correct dtypes

def test_missing_required_cols_mlb(caplog):
    df_missing = pd.DataFrame({"game_id": ["g1"], "game_date_et": ["2023-01-01"]}) # Missing team_ids
    with caplog.at_level(logging.ERROR):
        out_df = rest.transform(df_missing.copy())
    assert "Missing required columns for rest.py" in caplog.text
    # Should return the input df but with placeholder columns filled with defaults
    _assert_cols_mlb(out_df)
    assert "home_team_id" not in out_df.columns # Original missing col is still missing
    assert len(out_df) == len(df_missing)


def test_invalid_dates_mlb(sched_mlb, caplog):
    with caplog.at_level(logging.WARNING):
        out_df = rest.transform(sched_mlb.copy()) # sched_mlb contains one invalid date
    
    assert "Dropping 1 rows with invalid game_date" in caplog.text
    # Original sched_mlb has 17 rows, 1 is invalid date, so 16 should be processed.
    assert len(out_df) == len(sched_mlb) - 1
    _assert_cols_mlb(out_df)