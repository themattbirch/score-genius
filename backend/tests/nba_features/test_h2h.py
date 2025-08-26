# backend/tests/nba_features/test_h2h.py
"""
Exhaustive tests for backend.nba_features.h2h.transform

Coverage goals
--------------
1.  Validate every derived H2H column on well‑controlled historical data.
2.  Exercise edge‑cases: no history, mixed venues, duplicate dates, NaNs.
3.  Guarantee idempotency (second call unchanged) and NaN‑free outputs.
"""

from __future__ import annotations

import os
import sys

# ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)

import numpy as np
import pandas as pd
import pytest

from backend.nba_features import h2h, utils

# ───────────────────────────────
# CONSTANTS
# ───────────────────────────────
COLS = h2h.H2H_PLACEHOLDER_COLS
DEFAULTS = utils.DEFAULTS


# ───────────────────────────────
# HELPERS
# ───────────────────────────────
def _assert_cols(df: pd.DataFrame) -> None:
    missing = [c for c in COLS if c not in df.columns]
    assert not missing, f"Missing H2H columns: {missing}"


def _manual_stats(hist: pd.DataFrame, home_norm: str, max_games: int = 7) -> dict:
    """
    Produce manual expectations mirroring _get_matchup_history_single()
    for *home_norm* vs. its opponent in *hist* (hist already filtered < game_date).
    """
    if hist.empty:
        return {c: (pd.NaT if c == "matchup_last_date" else DEFAULTS.get(c, 0.0)) for c in COLS}

    recent = hist.sort_values("game_date", ascending=False).head(max_games)
    recent = recent.assign(
        diff=lambda d: np.where(
            d["home_team_norm"] == home_norm,
            d["home_score"] - d["away_score"],
            d["away_score"] - d["home_score"],
        ),
        won=lambda d: np.where(
            d["home_team_norm"] == home_norm,
            d["home_score"] > d["away_score"],
            d["away_score"] > d["home_score"],
        ),
        home_persp=lambda d: np.where(
            d["home_team_norm"] == home_norm, d["home_score"], d["away_score"]
        ),
        away_persp=lambda d: np.where(
            d["home_team_norm"] == home_norm, d["away_score"], d["home_score"]
        ),
    )

    diffs = recent["diff"].to_numpy(float)
    totals = (recent["home_score"] + recent["away_score"]).to_numpy(float)
    hp = recent["home_persp"].to_numpy(float)
    ap = recent["away_persp"].to_numpy(float)
    wins = recent["won"].sum()

    # streak: iterate oldest→newest
    streak = 0
    last = None
    for w in recent.sort_values("game_date")["won"]:
        cur_winner = "home" if w else "away"
        if last is None or cur_winner == last:
            streak = streak + 1 if w else streak - 1 if last is None else streak - 1
        else:
            streak = 1 if w else -1
        last = cur_winner

    return {
        "matchup_num_games": len(recent),
        "matchup_avg_point_diff": float(diffs.mean()),
        "matchup_home_win_pct": float(wins / len(recent)),
        "matchup_avg_total_score": float(totals.mean()),
        "matchup_avg_home_score": float(hp.mean()),
        "matchup_avg_away_score": float(ap.mean()),
        "matchup_last_date": recent["game_date"].max(),
        "matchup_streak": int(streak),
    }


# ───────────────────────────────
# FIXTURES
# ───────────────────────────────
@pytest.fixture(scope="module")
def historical_df() -> pd.DataFrame:
    """
    12 historical games across three teams (A,B,C) with mixed venues + duplicate dates.
    Includes NaNs to ensure transform zero-fills gracefully.
    """
    rows = []
    # use numeric constructor to avoid any hidden Unicode hyphens
    base = pd.Timestamp(year=2023, month=1, day=1)
    gid = 1
    for m in range(12):
        home, away = ("Team A", "Team B") if m % 3 else ("Team B", "Team A")
        if m == 10:  # introduce A-C with NaNs
            home, away = "Team A", "Team C"
        rows.append({
            "game_id": f"h{gid}",
            "game_date": base + pd.Timedelta(days=m // 2),  # duplicates every other row
            "home_team": home,
            "away_team": away,
            "home_score": 100 + m * 2 if m != 10 else np.nan,
            "away_score": 95 + m if m != 10 else np.nan,
        })
        gid += 1

    df = pd.DataFrame(rows)
    # ensure proper datetime dtype
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@pytest.fixture(scope="module")
def upcoming_df() -> pd.DataFrame:
    # likewise, use numeric constructor
    game_dates = [pd.Timestamp(year=2023, month=4, day=1)] * 3
    return pd.DataFrame({
        "game_id": ["u1", "u2", "u3"],
        "game_date": game_dates,
        "home_team": ["Team A", "Team B", "Team A"],
        "away_team": ["Team B", "Team A", "Team C"],  # u3 has no history (A-C with NaNs only)
    })


# ───────────────────────────────
# TESTS
# ───────────────────────────────
def test_columns_and_shape(historical_df, upcoming_df):
    out = h2h.transform(upcoming_df.copy(), historical_df=historical_df.copy(), debug=False)
    _assert_cols(out)
    assert len(out) == len(upcoming_df)


@pytest.mark.parametrize("row_idx", [0, 1])
def test_manual_expectations(historical_df, upcoming_df, row_idx):
    """
    Row 0: A home vs B  ▸ expect  history stats
    Row 1: B home vs A  ▸ expect  same counts but diff perspective
    """
    out = h2h.transform(upcoming_df.copy(), historical_df=historical_df.copy(), debug=False)

    row = out.iloc[row_idx]
    hist_subset = (
        historical_df.assign(
            home_team_norm=lambda d: d["home_team"].map(utils.normalize_team_name),
            away_team_norm=lambda d: d["away_team"].map(utils.normalize_team_name),
        )
        .query("game_date < @row.game_date")
    )

    home_norm = utils.normalize_team_name(row["home_team"])
    opp_norm = utils.normalize_team_name(row["away_team"])
    mask1 = (hist_subset["home_team_norm"] == home_norm) & (hist_subset["away_team_norm"] == opp_norm)
    mask2 = (hist_subset["home_team_norm"] == opp_norm) & (hist_subset["away_team_norm"] == home_norm)
    hist_pair = hist_subset[mask1 | mask2]

    exp = _manual_stats(hist_pair, home_norm)

    for k, v in exp.items():
        if "date" in k:
            assert row[k] == v
        else:
            assert np.isclose(row[k], v)


def test_defaults_no_history(upcoming_df):
    out = h2h.transform(upcoming_df.copy(), historical_df=None)
    _assert_cols(out)
    first = out.iloc[-1]  # game A‑C
    assert first["matchup_num_games"] == 0
    assert first["matchup_last_date"] is pd.NaT or pd.isna(first["matchup_last_date"])
    assert np.isclose(first["matchup_avg_total_score"], DEFAULTS["matchup_avg_total_score"])


def test_no_nans_in_outputs(historical_df, upcoming_df):
    out = h2h.transform(upcoming_df.copy(), historical_df=historical_df.copy())
    assert not out[COLS].isna().any().any()


def test_idempotent(historical_df, upcoming_df):
    first = h2h.transform(upcoming_df.copy(), historical_df=historical_df.copy())
    second = h2h.transform(first.copy(), historical_df=historical_df.copy())
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
