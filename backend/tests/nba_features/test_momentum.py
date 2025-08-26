# backend/tests/nba_features/test_momentum.py
"""
High-coverage tests for backend.nba_features.momentum.transform.

These tests aim to:
1.  Validate every derived column for correctness on realistic, messy data.
2.  Detect quarter-to-quarter leakage, sign errors, or NaN propagation.
3.  Stress EWMA logic for both Q3- and Q4-based momentum proxies.
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
from backend.nba_features import momentum

# ───────────────────────────────
# CONSTANTS
# ───────────────────────────────
Q = range(1, 5)
QUARTER_COLS = [f"{side}_q{i}" for i in Q for side in ("home", "away")]
Q_MARGIN_COLS = [f"q{i}_margin" for i in Q]
# Corrected cumulative difference column names
CUM_DIFF_COLS = ["end_q1_reg_diff", "end_q2_reg_diff", "end_q3_reg_diff", "end_q4_reg_diff"]
MARGIN_CHG_COLS = ["q2_margin_change", "q3_margin_change", "q4_margin_change"]
EWMA_COLS = ["momentum_score_ewma_q4", "momentum_score_ewma_q3"]
# DERIVED list will now use the corrected CUM_DIFF_COLS
DERIVED = Q_MARGIN_COLS + CUM_DIFF_COLS + MARGIN_CHG_COLS + EWMA_COLS + ["std_dev_q_margins"] # Added std_dev_q_margins based on momentum.py


# ───────────────────────────────
# UTILITIES
# ───────────────────────────────
def _assert_all_cols(df: pd.DataFrame) -> None:
    missing = [c for c in DERIVED if c not in df.columns]
    assert not missing, f"Derived columns missing: {missing}"


def _expected_for_row(row: pd.Series) -> dict[str, float]:
    """Manual calculation for one game row (used for spot-checks)."""
    # pull raw quarter scores, then zero-fill NaNs to match convert_and_fill behavior
    h_raw = [row[f"home_q{i}"] for i in Q]
    a_raw = [row[f"away_q{i}"] for i in Q]
    h = [0.0 if pd.isna(x) else x for x in h_raw]
    a = [0.0 if pd.isna(x) else x for x in a_raw]

    margins = [hi - ai for hi, ai in zip(h, a)]
    cumdiff = np.cumsum(margins, dtype=float)
    # Correctly initialize changes for q1 (no change) and calculate for q2-q4
    changes = [np.nan] # q1_margin_change is not calculated
    changes.extend([margins[i] - margins[i - 1] for i in range(1, 4)])


    ewma_q4 = (
        pd.Series(margins, dtype=float)
        .ewm(span=3, adjust=False) # span=3 as per momentum.py for q4
        .mean()
        .iat[-1]
    )
    ewma_q3 = (
        pd.Series(margins[:3], dtype=float)
        .ewm(span=2, adjust=False) # span=2 as per momentum.py for q3
        .mean()
        .iat[-1]
    )
    std_dev_q_margins = pd.Series(margins, dtype=float).std()


    expected_data = {
        **{f"q{i}_margin": margins[i - 1] for i in Q},
        **dict(zip(CUM_DIFF_COLS, cumdiff)), # CUM_DIFF_COLS is now corrected
        # MARGIN_CHG_COLS are 'q2_margin_change', 'q3_margin_change', 'q4_margin_change'
        # The `changes` list is [nan, q2_change, q3_change, q4_change]
        # So, we map from index 1, 2, 3 of `changes`
        **{MARGIN_CHG_COLS[i]: changes[i+1] for i in range(len(MARGIN_CHG_COLS))},
        "momentum_score_ewma_q4": ewma_q4,
        "momentum_score_ewma_q3": ewma_q3,
        "std_dev_q_margins": std_dev_q_margins, # Added
    }
    return expected_data


# ───────────────────────────────
# FIXTURES
# ───────────────────────────────
@pytest.fixture(scope="module")
def games_df() -> pd.DataFrame:
    """
    ~90 games, 4 teams, duplicate dates, sprinkled NaNs, extreme scores.
    Provides broad coverage for transform().
    """
    rng = np.random.default_rng(2025)
    teams = ["ATL", "BOS", "CHI", "DEN"]
    rows = []
    start = pd.Timestamp("2025-01-01")
    gid = 1
    for day in range(30):
        for _ in range(3):
            home, away = rng.choice(teams, 2, replace=False)
            scores = rng.integers(10, 40, 8).astype(float)
            scores[rng.random(8) < 0.05] = np.nan
            if rng.random() < 0.02:
                scores[rng.integers(0, 3)] += 50
            rows.append({
                "game_id": f"g{gid}",
                "game_date": start + pd.Timedelta(days=day),
                "home_team": home,
                "away_team": away,
                **{f"home_q{i}": scores[i - 1]     for i in Q},
                **{f"away_q{i}": scores[i - 1 + 4] for i in Q},
            })
            gid += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=99).reset_index(drop=True)


# ───────────────────────────────
# TESTS
# ───────────────────────────────
def test_shape_and_columns(games_df):
    out = momentum.transform(games_df.copy(), debug=False)
    _assert_all_cols(out)
    assert len(out) == len(games_df)


@pytest.mark.parametrize("idx", [0, 10, 25, 50])
def test_manual_spot_checks(games_df, idx):
    """Validate four disparate rows against manual calculations."""
    transformed = momentum.transform(games_df.copy(), debug=False)
    expected = _expected_for_row(games_df.iloc[idx])
    row_transformed = transformed.iloc[idx]

    for col, exp_value in expected.items():
        # Skip assertion if expected value is NaN (e.g. q1_margin_change which is not calculated)
        if pd.isna(exp_value) and col not in row_transformed: # q1_margin_change won't be in row_transformed
             continue
        if pd.isna(exp_value): # If expected is NaN, actual should also be NaN (or default if applicable)
            assert pd.isna(row_transformed[col]), f"{col} row {idx}: expected NaN, got {row_transformed[col]}"
            continue

        assert col in row_transformed, f"Column {col} missing in transformed output for row {idx}"
        got = row_transformed[col]
        assert np.isclose(got, exp_value, atol=1e-3, equal_nan=True), \
            f"{col} row {idx}: exp {exp_value}, got {got}"


def test_no_nans_in_derived(games_df):
    """convert_and_fill should zero-fill NaNs → no derived NaNs for columns that should not have them."""
    out = momentum.transform(games_df.copy(), debug=False)
    # Some columns like qX_margin_change might be NaN if previous quarter margin was also NaN
    # and convert_and_fill results in 0-0 = 0.
    # The momentum.py script uses fillna for EWMA and std_dev columns.
    # Check specific columns known to be filled.
    cols_to_check_for_nans = EWMA_COLS + ["std_dev_q_margins"]
    assert not out[cols_to_check_for_nans].isna().any().any(), \
        f"NaNs found in EWMA or std_dev columns: {out[cols_to_check_for_nans].isna().sum()}"
    # Margin and cum_diff columns should be fine due to convert_and_fill on raw scores
    assert not out[Q_MARGIN_COLS + CUM_DIFF_COLS].isna().any().any(), \
        f"NaNs found in margin or cum_diff columns: {out[Q_MARGIN_COLS + CUM_DIFF_COLS].isna().sum()}"


def test_idempotent_transform(games_df):
    """Calling transform twice must not change results (no double-count bugs)."""
    first = momentum.transform(games_df.copy(), debug=False)
    second = momentum.transform(first.copy(), debug=False)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_missing_columns_returns_original():
    df = pd.DataFrame({"game_id": [1], "home_q1": [12]}) # Missing away_q1 etc.
    out = momentum.transform(df.copy(), debug=False)
    # momentum.py is designed to return the original df if required raw_cols are missing
    pd.testing.assert_frame_equal(df, out)


def test_extreme_score_sanity():
    df = pd.DataFrame({
        "game_id": [999],
        "home_team": ["TeamExtH"], # Added for completeness if needed by any future logic
        "away_team": ["TeamExtA"], # Added for completeness
        "home_q1": [60.0], "away_q1": [5.0],
        "home_q2": [55.0], "away_q2": [4.0],
        "home_q3": [58.0], "away_q3": [6.0],
        "home_q4": [62.0], "away_q4": [7.0],
    })
    out = momentum.transform(df.copy(), debug=False)
    _assert_all_cols(out) # Checks all DERIVED columns are present
    assert out["momentum_score_ewma_q4"].iloc[0] > 0 # Given home team wins every quarter by a lot
    assert np.isfinite(out["momentum_score_ewma_q4"].iloc[0])
    
    expected_sum = 0
    for i in Q:
        expected_sum += (df[f"home_q{i}"].iloc[0] - df[f"away_q{i}"].iloc[0])
    assert np.isclose(out["end_q4_reg_diff"].iloc[0], expected_sum)