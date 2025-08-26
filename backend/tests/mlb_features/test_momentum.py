# backend/tests/mlb_features/test_momentum.py

import os
import sys
import numpy as np
import pandas as pd
import pytest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.mlb_features import momentum # Module to test

# Define test-specific defaults for EWMA fallbacks
TEST_MLB_DEFAULTS_MOMENTUM = {
    "mlb_momentum_runs_ewma_inn_6": 0.06,
    "mlb_momentum_runs_ewma_inn_9": 0.09, # Assuming default num_innings is 9
    "mlb_momentum_runs_ewma_inn_5": 0.05, # For testing num_innings < 6
    # Defaults for other derived columns will use default_fill (0.0) if not in this map
}

# Test-specific convert_and_fill to ensure consistent behavior
def convert_and_fill_test(df: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default # Should not happen if cols are derived from existing df columns
    return df

@pytest.fixture(autouse=True)
def mock_mlb_momentum_dependencies(monkeypatch):
    monkeypatch.setattr(momentum, "MLB_DEFAULTS", TEST_MLB_DEFAULTS_MOMENTUM)
    monkeypatch.setattr(momentum, "convert_and_fill", convert_and_fill_test)

# Dynamically generate column lists based on num_innings for helpers
def get_mlb_momentum_derived_cols(num_innings=9):
    innings_range = range(1, num_innings + 1)
    margin_cols = [f"inn_{i}_margin" for i in innings_range]
    cum_diff_cols = [f"end_inn_{i}_run_diff" for i in innings_range]
    margin_change_cols = [f"inn_{i}_margin_change" for i in range(2, num_innings + 1)]
    ewma_cols = [f"momentum_runs_ewma_inn_{num_innings}"]
    if num_innings >= 6:
        ewma_cols.append("momentum_runs_ewma_inn_6")
    return margin_cols + cum_diff_cols + margin_change_cols + ewma_cols

def _assert_all_cols_mlb(df: pd.DataFrame, num_innings=9) -> None:
    derived_cols = get_mlb_momentum_derived_cols(num_innings)
    missing = [c for c in derived_cols if c not in df.columns]
    assert not missing, f"Derived MLB momentum columns missing: {missing}"

def _expected_for_row_mlb(row: pd.Series, num_innings=9, span_inn6=3, span_inn9=4,
                           ewma6_default=0.06, ewma_final_default=0.09) -> dict:
    h_raw = [row.get(f"h_inn_{i}") for i in range(1, num_innings + 1)]
    a_raw = [row.get(f"a_inn_{i}") for i in range(1, num_innings + 1)]
    
    # Mimic convert_and_fill_test (0.0 is the default for raw scores if missing/NaN)
    h_scores = [0.0 if pd.isna(x) else float(x) for x in h_raw]
    a_scores = [0.0 if pd.isna(x) else float(x) for x in a_raw]

    margins = [hs - as_ for hs, as_ in zip(h_scores, a_scores)]
    cum_diff = np.cumsum(margins).tolist()
    
    margin_changes = [np.nan] # inn_1_margin_change is not calculated
    for i in range(1, num_innings):
        margin_changes.append(margins[i] - margins[i-1])

    expected = {}
    for i in range(num_innings):
        expected[f"inn_{i+1}_margin"] = margins[i]
        expected[f"end_inn_{i+1}_run_diff"] = cum_diff[i]
    for i in range(1, num_innings): # Corresponds to inn_2_margin_change onwards
        expected[f"inn_{i+1}_margin_change"] = margin_changes[i]

    # EWMA
    if margins:
        series_margins = pd.Series(margins, dtype=float)
        ewma_final_val = series_margins.ewm(span=span_inn9, axis=0, adjust=False, min_periods=1).mean().iloc[-1]
        expected[f"momentum_runs_ewma_inn_{num_innings}"] = ewma_final_val if pd.notna(ewma_final_val) else ewma_final_default

        if num_innings >= 6:
            ewma6_val = series_margins.iloc[:6].ewm(span=span_inn6, axis=0, adjust=False, min_periods=1).mean().iloc[-1]
            expected["momentum_runs_ewma_inn_6"] = ewma6_val if pd.notna(ewma6_val) else ewma6_default
        elif f"momentum_runs_ewma_inn_6" in get_mlb_momentum_derived_cols(num_innings): # Should not happen if num_innings < 6
             expected["momentum_runs_ewma_inn_6"] = ewma6_default


    # Ensure all expected derived columns are present in the 'expected' dict with a value
    all_derived = get_mlb_momentum_derived_cols(num_innings)
    for col in all_derived:
        if col not in expected:
            if col == "momentum_runs_ewma_inn_6" and num_innings <6:
                 expected[col] = TEST_MLB_DEFAULTS_MOMENTUM.get("mlb_momentum_runs_ewma_inn_6", 0.0)
            elif col == f"momentum_runs_ewma_inn_{num_innings}":
                 expected[col] = TEST_MLB_DEFAULTS_MOMENTUM.get(f"mlb_momentum_runs_ewma_inn_{num_innings}",0.0)
            else: # Other derived columns default to 0.0 if not calculable (e.g. margin_change for inn 1)
                 expected[col] = 0.0 # Default fill for numeric cols not EWMA

    return expected

@pytest.fixture(scope="module")
def games_df_mlb_factory():
    def _create_df(num_innings=9, num_games=30):
        rng = np.random.default_rng(2024)
        rows = []
        for game_idx in range(num_games):
            game_data = {"game_id": f"mlb_g{game_idx+1}"}
            raw_scores = rng.integers(0, 7, num_innings * 2).astype(float)
            # Sprinkle some NaNs
            raw_scores[rng.random(num_innings * 2) < 0.1] = np.nan
            for i in range(1, num_innings + 1):
                game_data[f"h_inn_{i}"] = raw_scores[i-1]
                game_data[f"a_inn_{i}"] = raw_scores[i-1 + num_innings]
            rows.append(game_data)
        return pd.DataFrame(rows)
    return _create_df

# Tests
@pytest.mark.parametrize("num_innings_test", [9, 5])
def test_shape_and_columns_mlb(games_df_mlb_factory, num_innings_test):
    games_df = games_df_mlb_factory(num_innings=num_innings_test)
    out_df = momentum.transform(games_df.copy(), num_innings=num_innings_test)
    _assert_all_cols_mlb(out_df, num_innings=num_innings_test)
    assert len(out_df) == len(games_df)
    for col in get_mlb_momentum_derived_cols(num_innings_test):
        assert out_df[col].dtype == float, f"Column {col} is not float type"


@pytest.mark.parametrize("num_innings_test, game_row_idx_to_check", [(9, 0), (9,5), (5,0), (5,2)])
@pytest.mark.parametrize("custom_span_inn6, custom_span_inn9", [(3,4), (2,5)]) # Test default and custom spans
def test_manual_spot_checks_mlb(games_df_mlb_factory, num_innings_test, game_row_idx_to_check,
                                custom_span_inn6, custom_span_inn9):
    games_df = games_df_mlb_factory(num_innings=num_innings_test, num_games=10) # Smaller df for spot checks
    if game_row_idx_to_check >= len(games_df): # skip if index out of bounds for smaller df
        pytest.skip("Game row index out of bounds for this num_games configuration.")

    # Get EWMA defaults based on num_innings for _expected_for_row_mlb
    ewma6_def = TEST_MLB_DEFAULTS_MOMENTUM.get("mlb_momentum_runs_ewma_inn_6", 0.0)
    ewma_final_def_key = f"mlb_momentum_runs_ewma_inn_{num_innings_test}"
    ewma_final_def = TEST_MLB_DEFAULTS_MOMENTUM.get(ewma_final_def_key, 0.0)

    transformed_df = momentum.transform(
        games_df.copy(),
        num_innings=num_innings_test,
        span_inn6=custom_span_inn6,
        span_inn9=custom_span_inn9,
        defaults=TEST_MLB_DEFAULTS_MOMENTUM # Pass test defaults for EWMA fallbacks
    )
    
    expected_metrics = _expected_for_row_mlb(
        games_df.iloc[game_row_idx_to_check],
        num_innings=num_innings_test,
        span_inn6=custom_span_inn6,
        span_inn9=custom_span_inn9,
        ewma6_default=ewma6_def,
        ewma_final_default=ewma_final_def
    )
    actual_row = transformed_df.iloc[game_row_idx_to_check]

    for col, expected_val in expected_metrics.items():
        if col.endswith("margin_change") and col.startswith("inn_1_"): # inn_1_margin_change is not created
            assert col not in actual_row, f"{col} should not exist."
            continue
        
        assert col in actual_row, f"Column {col} missing in transformed output."
        actual_val = actual_row[col]
        
        if pd.isna(expected_val): # Should only happen if _expected_for_row_mlb yields NaN for a change col
            assert pd.isna(actual_val), f"{col} (idx {game_row_idx_to_check}): expected NaN, got {actual_val}"
            continue
        
        assert np.isclose(actual_val, expected_val, atol=1e-4, equal_nan=True), \
            f"{col} (idx {game_row_idx_to_check}, num_inn {num_innings_test}, spans {custom_span_inn6},{custom_span_inn9}): expected {expected_val}, got {actual_val}"


def test_no_nans_in_derived_mlb(games_df_mlb_factory):
    games_df = games_df_mlb_factory(num_innings=9) # Use standard 9 innings
    out_df = momentum.transform(games_df.copy(), num_innings=9)
    # All derived columns should be filled by the end of the transform function
    all_derived = get_mlb_momentum_derived_cols(num_innings=9)
    assert not out_df[all_derived].isna().any().any(), \
        f"NaNs found in derived columns: {out_df[all_derived].isna().sum()[out_df[all_derived].isna().sum() > 0]}"


def test_idempotent_transform_mlb(games_df_mlb_factory):
    games_df = games_df_mlb_factory()
    first_run = momentum.transform(games_df.copy())
    second_run = momentum.transform(first_run.copy()) # Pass the transformed df
    pd.testing.assert_frame_equal(first_run, second_run, check_dtype=True)


def test_missing_inning_columns_returns_original_mlb(caplog):
    # Missing h_inn_3 for example
    df = pd.DataFrame({
        "game_id": ["g1"],
        "h_inn_1": [1], "a_inn_1": [0],
        "h_inn_2": [0], "a_inn_2": [2],
        # "h_inn_3": [0], # Missing
         "a_inn_3": [0],
        "h_inn_4": [1], "a_inn_4": [1], "h_inn_5": [0], "a_inn_5": [0],
        "h_inn_6": [2], "a_inn_6": [0], "h_inn_7": [0], "a_inn_7": [0],
        "h_inn_8": [1], "a_inn_8": [0], "h_inn_9": [0], "a_inn_9": [0],
    })
    with caplog.at_level(logging.WARNING):
        out_df = momentum.transform(df.copy(), num_innings=9)
    assert "Missing inning score cols" in caplog.text
    assert "h_inn_3" in caplog.text
    pd.testing.assert_frame_equal(df, out_df, check_dtype=True) # Should return original

def test_empty_input_df_mlb():
    empty_df = pd.DataFrame()
    out_df = momentum.transform(empty_df.copy())
    assert out_df.empty
    pd.testing.assert_frame_equal(empty_df, out_df, check_dtype=True)


def test_custom_default_fill_and_ewma_defaults_mlb(games_df_mlb_factory):
    games_df = games_df_mlb_factory(num_innings=5, num_games=3)
    # Make one game have all NaNs for inning scores to trigger defaults clearly
    for i in range(1, 6):
        games_df.loc[0, f"h_inn_{i}"] = np.nan
        games_df.loc[0, f"a_inn_{i}"] = np.nan

    custom_fill = -99.0
    custom_ewma_defaults = {
        "mlb_momentum_runs_ewma_inn_5": -5.5,
        # No inn_6 for num_innings=5
    }
    out_df = momentum.transform(
        games_df.copy(),
        num_innings=5,
        default_fill=custom_fill,
        defaults=custom_ewma_defaults
    )
    
    # For the all-NaN row, inning margins should be custom_fill - custom_fill = 0
    # Cumulative diffs should also be 0. Margin changes also 0.
    # EWMA should be the custom EWMA default.
    all_nan_row = out_df.iloc[0]
    assert np.isclose(all_nan_row["inn_1_margin"], 0.0) # custom_fill - custom_fill
    assert np.isclose(all_nan_row["end_inn_5_run_diff"], 0.0)
    assert np.isclose(all_nan_row[f"momentum_runs_ewma_inn_5"], custom_ewma_defaults["mlb_momentum_runs_ewma_inn_5"])
    
    if "momentum_runs_ewma_inn_6" in out_df.columns: # Should not be present for num_innings=5
        # If it were, it should use its value from TEST_MLB_DEFAULTS_MOMENTUM via monkeypatch,
        # or a general default_fill if not in the passed 'defaults' map.
        # However, the logic in transform only creates it if num_innings >= 6
        assert False, "momentum_runs_ewma_inn_6 should not be created for num_innings=5"