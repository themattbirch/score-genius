# backend/tests/nfl_features/test_engine.py
from __future__ import annotations
import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.nfl_features.engine import NFLFeatureEngine

# Helper factory for creating test stubs
def _make_stub(name: str, tracker: List[Dict[str, Any]], return_empty: bool = False, raise_error: bool = False):
    """
    Creates a more capable stub transform function that can optionally
    return an empty DataFrame or raise an error.
    """
    def _stub(*args, **kwargs):
        tracker.append({"name": name})
        
        # Handle conditional behaviors for testing
        if raise_error:
            raise RuntimeError(f"Test error in {name}")
        if return_empty:
            return pd.DataFrame()
        
        # Default behavior: echo a marker column
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]
            # Return a minimal DataFrame to allow merging
            return pd.DataFrame({"game_id": df["game_id"], f"ran_{name}": True})
        
        # Fallback for rolling module or others without a df arg
        if "game_ids" in kwargs:
            return pd.DataFrame({"game_id": kwargs["game_ids"], f"ran_{name}": True})
            
        return pd.DataFrame()
    return _stub

# Fixtures
@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    # This DataFrame now includes a 'season' column as the engine expects it
    return pd.DataFrame({
        "game_id": [1, 2],
        "game_date": pd.to_datetime(["2024-10-20", "2024-10-21"]),
        "home_team_id": ["team_a", "team_c"],
        "away_team_id": ["team_b", "team_a"],
        "season": ["2024", "2024"] # Add season
    })

@pytest.fixture
def sample_histories() -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_games = pd.DataFrame({
        "game_id": [101, 102],
        "game_date": pd.to_datetime(["2023-09-10", "2023-09-11"]),
        "home_team_id": ["team_a", "team_b"],
        "away_team_id": ["team_b", "team_c"],
    })
    hist_team_stats = pd.DataFrame({"team_id": ["team_a"], "season": [2023]})
    return hist_games, hist_team_stats

# Tests
def test_order_and_kwargs(sample_games_df, sample_histories, monkeypatch):
    """Tests that all modules are called in the correct default order."""
    calls: List[Dict[str, Any]] = []
    
    # Create an engine instance for this test
    eng = NFLFeatureEngine(supabase_url="dummy", supabase_service_key="dummy")

    # Mock the RPC calls to return empty dataframes
    monkeypatch.setattr(eng, '_fetch_season_stats_via_rpc', lambda season: pd.DataFrame(columns=['season']))
    monkeypatch.setattr(eng, '_fetch_recent_form_via_rpc', lambda: pd.DataFrame())

    # Mock all the transform functions
    for mod_name in eng.DEFAULT_ORDER:
        monkeypatch.setitem(eng.TRANSFORMS, mod_name, _make_stub(mod_name, calls))

    eng.build_features(
        sample_games_df,
        historical_games_df=sample_histories[0],
        historical_team_stats_df=sample_histories[1]
    )

    assert [c["name"] for c in calls] == eng.DEFAULT_ORDER

def test_custom_execution_order(sample_games_df, sample_histories, monkeypatch):
    """Tests that a custom execution order is respected."""
    calls: List[Dict[str, Any]] = []
    eng = NFLFeatureEngine(supabase_url="dummy", supabase_service_key="dummy")
    
    monkeypatch.setattr(eng, '_fetch_season_stats_via_rpc', lambda season: pd.DataFrame(columns=['season']))
    monkeypatch.setattr(eng, '_fetch_recent_form_via_rpc', lambda: pd.DataFrame())
    
    custom_order = ["h2h", "form", "rest"]
    eng.execution_order = custom_order

    for mod_name in custom_order:
        monkeypatch.setitem(eng.TRANSFORMS, mod_name, _make_stub(mod_name, calls))

    eng.build_features(
        sample_games_df,
        historical_games_df=sample_histories[0],
        historical_team_stats_df=sample_histories[1],
    )
    assert [c["name"] for c in calls] == custom_order

def test_pipeline_continues_on_empty_module(sample_games_df, sample_histories, monkeypatch):
    """Tests that the pipeline continues if a module returns an empty DataFrame."""
    calls: List[Dict[str, Any]] = []
    eng = NFLFeatureEngine(supabase_url="dummy", supabase_service_key="dummy")
    
    monkeypatch.setattr(eng, '_fetch_season_stats_via_rpc', lambda season: pd.DataFrame(columns=['season']))
    monkeypatch.setattr(eng, '_fetch_recent_form_via_rpc', lambda: pd.DataFrame())
    
    # Mock all modules, but make 'season' return an empty df
    for mod_name in eng.DEFAULT_ORDER:
        is_empty = (mod_name == "season")
        monkeypatch.setitem(eng.TRANSFORMS, mod_name, _make_stub(mod_name, calls, return_empty=is_empty))

    out = eng.build_features(
        sample_games_df,
        historical_games_df=sample_histories[0],
        historical_team_stats_df=sample_histories[1],
    )

    # All modules should still be called
    assert [c["name"] for c in calls] == eng.DEFAULT_ORDER
    # The output should still have columns from other modules
    assert "ran_rest" in out.columns
    assert "ran_form" in out.columns
    # The column from the empty module should NOT be present
    assert "ran_season" not in out.columns

def test_empty_input_df():
    """Tests that an empty input df results in an empty output df."""
    eng = NFLFeatureEngine(supabase_url="dummy", supabase_service_key="dummy")
    out = eng.build_features(
        pd.DataFrame(),
        historical_games_df=pd.DataFrame(),
        historical_team_stats_df=pd.DataFrame(),
    )
    assert out.empty