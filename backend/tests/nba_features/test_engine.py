# backend/tests/nba_features/test_engine.py
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

from backend.nba_features import engine

# -----------------------------------------------------------------
# Helper factory that builds stub transform functions
# -----------------------------------------------------------------

def _make_stub(name: str, tracker: list):
    def _stub(df: pd.DataFrame, **kwargs):
        # Record the call order and kwargs for later assertions
        tracker.append((name, kwargs))
        new = df.copy()
        new[f"ran_{name}"] = True
        return new
    return _stub

# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------
@pytest.fixture
def base_games_df():
    """Minimal DF with the four essential columns."""
    return pd.DataFrame({
        "game_id": [1],
        "game_date": pd.to_datetime(["2025-01-01"]),
        "home_team": ["Team A"],
        "away_team": ["Team B"],
    })

@pytest.fixture
def hp(monkeypatch):
    """Monkeypatch the TRANSFORM_MAP with stubs and yield the call tracker."""
    tracker = []
    for mod in engine.DEFAULT_EXECUTION_ORDER:
        monkeypatch.setitem(engine.TRANSFORM_MAP, mod, _make_stub(mod, tracker))
    yield tracker  # tests can inspect call order

# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------

def test_execution_order_and_columns(base_games_df, hp):
    tracker = hp
    out = engine.run_feature_pipeline(base_games_df.copy(), debug=False)

    # 1) Every module should have been invoked in order
    invoked = [name for name, _ in tracker]
    assert invoked == engine.DEFAULT_EXECUTION_ORDER

    # 2) Output DF should carry the stub columns
    for name in engine.DEFAULT_EXECUTION_ORDER:
        assert f"ran_{name}" in out.columns


def test_skip_rolling_when_no_windows(base_games_df, hp):
    tracker = hp
    out = engine.run_feature_pipeline(base_games_df.copy(), rolling_windows=[], debug=False)

    invoked = [n for n, _ in tracker]
    assert "rolling" not in invoked
    assert "ran_rolling" not in out.columns


def test_missing_essential_columns_returns_input():
    df = pd.DataFrame({"game_id": [1]})  # missing 3 essentials
    res = engine.run_feature_pipeline(df.copy(), debug=False)
    # Should return the same columns
    assert set(res.columns) == {"game_id"}


def test_pipeline_halts_on_module_error(base_games_df, monkeypatch):
    """Season module raises an error; ensure earlier modules were applied and later skipped."""
    call_order = []

    # --- MODIFIED: Corrected the list of modules that run BEFORE 'season' ---
    # The real order is: 'game_advanced_metrics', 'rest', 'h2h', then 'season'
    modules_before_season = ['game_advanced_metrics', 'rest', 'h2h']
    for mod in modules_before_season:
        # Note: In your test file, engine.TRANSFORM_MAP is used.
        # If that alias was removed, use engine.TRANSFORMS directly.
        monkeypatch.setitem(engine.TRANSFORMS, mod, _make_stub(mod, call_order))

    # Error stub for season (this part of the test is correct)
    def _season_error(df, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setitem(engine.TRANSFORMS, "season", _season_error)

    # Form should never run (this part of the test is correct)
    monkeypatch.setitem(engine.TRANSFORMS, "form", _make_stub("form", call_order))

    # --- MODIFIED: The call to run_feature_pipeline now includes the new DataFrame arguments ---
    out = engine.run_feature_pipeline(
        base_games_df.copy(),
        seasonal_splits_data=pd.DataFrame(), # Pass empty DF for the test
        rolling_features_data=pd.DataFrame(),  # Pass empty DF for the test
        debug=False
    )

    # This assertion is now correct because we fixed the DEFAULT_ORDER
    assert "form" not in [n for n, _ in call_order]
    
    # --- MODIFIED: The final assertion now checks for the correct modules ---
    for mod in modules_before_season:
        assert f"ran_{mod}" in out.columns
        
    # And we can assert that modules that run AFTER the error are not present
    assert "ran_form" not in out.columns
    assert "ran_adv_splits" not in out.columns
    assert "ran_rolling" not in out.columns
