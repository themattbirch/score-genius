# backend/tests/mlb_features/test_engine.py
import pandas as pd
import pytest
import logging

import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.mlb_features import engine

# Helper factory that builds stub transform functions
def _make_stub(name: str, tracker: list, return_empty: bool = False, raise_error: bool = False):
    def _stub(df: pd.DataFrame, **kwargs):
        tracker.append({"name": name, "kwargs": kwargs, "input_df_shape": df.shape})
        if raise_error:
            raise RuntimeError(f"Test error in {name}")
        if return_empty:
            return pd.DataFrame()
        
        new_df = df.copy()
        new_df[f"ran_{name}"] = True
        return new_df
    return _stub

# Fixtures
@pytest.fixture
def base_mlb_games_df():
    """Minimal MLB DF with the essential columns."""
    return pd.DataFrame({
        "game_id": ["MLB001", "MLB002"],
        "game_date_et": pd.to_datetime(["2023-07-15 19:00:00", "2023-07-16 13:00:00"]),
        "home_team_id": ["NYY", "BOS"],
        "away_team_id": ["LAD", "CHC"],
    })

@pytest.fixture
def mock_historical_games_df():
    return pd.DataFrame({"hist_game_id": [1, 2], "game_date_et": pd.to_datetime(["2022-01-01", "2022-01-02"])})

@pytest.fixture
def mock_historical_team_stats_df():
    """ --- MODIFIED: Added required columns for the engine's calculations --- """
    return pd.DataFrame({
        "team_id": ["NYY", "BOS"], 
        "team_norm": ["nyy", "bos"],
        "season": [2022, 2022],
        'wins_all_total': [99, 78], 
        'games_played_all': [162, 162], 
        'runs_for_total_all': [807, 735],
        'runs_against_total_all': [567, 787], 
        'wins_home_total': [57, 43], 
        'games_played_home': [81, 81],
        'wins_away_total': [42, 35], 
        'games_played_away': [81, 81]
    })

@pytest.fixture
def patched_engine_transforms(monkeypatch):
    """Monkeypatches the engine.TRANSFORMS map with stubs."""
    call_tracker = []
    for module_name in engine.DEFAULT_ORDER:
        stub = _make_stub(module_name, call_tracker)
        monkeypatch.setitem(engine.TRANSFORMS, module_name, stub)
    return {"tracker": call_tracker}

# Tests
def test_execution_order(base_mlb_games_df, patched_engine_transforms, mock_historical_games_df, mock_historical_team_stats_df):
    """Tests that modules are called in the default order."""
    tracker = patched_engine_transforms["tracker"]
    
    engine.run_mlb_feature_pipeline(
        df=base_mlb_games_df,
        mlb_historical_games_df=mock_historical_games_df,
        mlb_historical_team_stats_df=mock_historical_team_stats_df,
        seasonal_splits_data=pd.DataFrame(),
        precomputed_rolling_features_df=pd.DataFrame()
    )

    invoked_modules = [call["name"] for call in tracker]
    assert invoked_modules == engine.DEFAULT_ORDER, "Modules were not called in the default order"

def test_missing_essential_columns_returns_input(caplog):
    """Tests pipeline abortion if essential columns are missing."""
    df_missing_home = pd.DataFrame({
        "game_id": ["MLB001"], 
        "game_date_et": [pd.Timestamp("2023-01-01")],
        "away_team_id": ["LAD"]
    })
    
    with caplog.at_level(logging.ERROR):
        res_df = engine.run_mlb_feature_pipeline(df_missing_home.copy())
    
    assert res_df.shape == df_missing_home.shape
    # --- MODIFIED: More robust assertion ---
    assert "missing required columns" in caplog.text.lower()

def test_empty_input_df_returns_empty_df(caplog):
    """Tests behavior with an empty input DataFrame."""
    with caplog.at_level(logging.WARNING):
        empty_df_result = engine.run_mlb_feature_pipeline(pd.DataFrame())
    assert empty_df_result.empty
    assert "Input DataFrame is empty" in caplog.text

def test_pipeline_halts_if_module_returns_empty_df(base_mlb_games_df, monkeypatch, caplog):
    """Tests that the pipeline halts if a module returns an empty DataFrame."""
    call_tracker = []
    monkeypatch.setitem(engine.TRANSFORMS, "rest", _make_stub("rest", call_tracker))
    monkeypatch.setitem(engine.TRANSFORMS, "season", _make_stub("season", call_tracker, return_empty=True))
    monkeypatch.setitem(engine.TRANSFORMS, "rolling", _make_stub("rolling", call_tracker))

    with caplog.at_level(logging.ERROR):
        engine.run_mlb_feature_pipeline(base_mlb_games_df.copy())
    
    invoked_modules = [call["name"] for call in call_tracker]
    assert invoked_modules == ["rest", "season"], "Pipeline did not halt correctly after empty DF"
    assert "DataFrame became empty after module 'season'" in caplog.text