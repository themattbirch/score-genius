# backend/tests/mlb_features/test_base_windows.py
import pandas as pd
import pytest
import os
import sys

# Ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)

# Import the MLB module to be tested
from backend.mlb_features import base_windows

# ----------------------------------------------------------------------
# Helpers: Replicated from NBA tests - extremely lightweight fakes
# that mimic the supabase-py chain
# ----------------------------------------------------------------------
class _DummyResp:
    def __init__(self, data):
        self.data = data

class _DummyQuery:
    def __init__(self, data_holder, capture):
        self._data_holder = data_holder
        self._capture = capture # To store details of the query for assertion

    # Keep returning self so the call-chain works
    def select(self, *_):               # .select("*")
        return self

    def in_(self, column, ids):         # .in_("game_id", ids)
        self._capture["column"] = column
        self._capture["ids"] = ids
        return self

    def execute(self):                  # .execute()
        # If the data_holder is an Exception, raise it to mimic DB failure
        if isinstance(self._data_holder, Exception):
            raise self._data_holder
        return _DummyResp(self._data_holder)

class _DummyConn:
    """
    Minimal stub of a Supabase client.
    Pass `payload` as either:
      • list[dict]  → normal data return
      • None        → simulate empty query result (response.data would be None)
      • Exception() → simulate database error during execute()
    """
    def __init__(self, payload):
        self._payload = payload
        self.capture: dict[str, any] = {} # Captures table_name, column, ids

    def table(self, table_name):        # .table("mlb_team_rolling_features")
        self.capture["table"] = table_name
        return _DummyQuery(self._payload, self.capture)

# ----------------------------------------------------------------------
# Tests for MLB base_windows
# ----------------------------------------------------------------------

def test_fetch_mlb_rolling_stats_success():
    """Tests successful fetching of MLB rolling stats."""
    mock_data_rows = [
        {"game_id": "MLB001", "team_id": "NYY", "rolling_avg_runs": 5.5, "rolling_era": 3.20},
        {"game_id": "MLB002", "team_id": "LAD", "rolling_avg_runs": 4.8, "rolling_era": 2.95},
    ]
    # Ensure game_ids can be mixed types (int/str) as input to fetch_mlb_rolling_stats
    game_ids_to_fetch = ["MLB001", 12345, "MLB002", 67890] # Assuming DB game_id is text
                                                       # and 12345, 67890 are not in mock_data_rows

    # We only expect MLB001 and MLB002 back based on mock_data_rows
    # The _DummyQuery's execute will only "return" mock_data_rows if the filter matches.
    # For simplicity in the mock, we'll assume the .in_ filter works perfectly if IDs are provided.
    # A more sophisticated mock might filter mock_data_rows based on conn.capture["ids"].
    # However, the current mock focuses on the call chain and data pass-through.
    # The provided mock_data_rows will be what's returned from execute().
    conn = _DummyConn(mock_data_rows)


    df = base_windows.fetch_mlb_rolling_stats(conn, game_ids_to_fetch)

    # DataFrame correctness
    assert not df.empty, "DataFrame should not be empty on success"
    assert len(df) == 2, "DataFrame should contain two rows from mock_data_rows"
    assert set(df["game_id"]) == {"MLB001", "MLB002"}, "DataFrame should contain the correct game_ids"
    assert "rolling_avg_runs" in df.columns
    assert "rolling_era" in df.columns

    # Proper query composition
    assert conn.capture["table"] == base_windows.MLB_ROLLING_STATS_VIEW, \
        f"Should query table '{base_windows.MLB_ROLLING_STATS_VIEW}'"
    assert conn.capture["column"] == "game_id", "Should filter on 'game_id' column"
    # Assert that game_ids were coerced to strings for the Supabase query
    expected_str_ids = [str(gid) for gid in game_ids_to_fetch]
    assert conn.capture["ids"] == expected_str_ids, "Game IDs should be coerced to strings for the query"

def test_fetch_mlb_rolling_stats_none_conn_returns_empty():
    """Tests that a None connection returns an empty DataFrame."""
    out_df = base_windows.fetch_mlb_rolling_stats(None, ["MLB001", "MLB002"])
    assert isinstance(out_df, pd.DataFrame), "Output should be a DataFrame"
    assert out_df.empty, "DataFrame should be empty when connection is None"

def test_fetch_mlb_rolling_stats_empty_ids_returns_empty():
    """Tests that an empty list of game_ids returns an empty DataFrame."""
    # The _DummyConn payload doesn't matter here as the function should short-circuit.
    conn = _DummyConn([{"game_id": "MLB001", "data": 1}])
    out_df_empty_list = base_windows.fetch_mlb_rolling_stats(conn, [])
    assert out_df_empty_list.empty, "DataFrame should be empty for empty list of game_ids"

    out_df_none_list = base_windows.fetch_mlb_rolling_stats(conn, None) # type: ignore
    assert out_df_none_list.empty, "DataFrame should be empty for None game_ids"


def test_fetch_mlb_rolling_stats_db_error_graceful():
    """Tests that a database error during query execution returns an empty DataFrame."""
    conn = _DummyConn(Exception("Simulated Supabase DB Error"))
    out_df = base_windows.fetch_mlb_rolling_stats(conn, ["MLB001"])

    assert isinstance(out_df, pd.DataFrame), "Output should be a DataFrame even on error"
    assert out_df.empty, "DataFrame should be empty on database error"
    # Optionally, you could check logs here if your logger was mockable/capturable

def test_fetch_mlb_rolling_stats_empty_result_from_db():
    """Tests handling of an empty data list from Supabase (response.data = [] or None)."""
    # Simulate Supabase returning an empty list in response.data
    conn_empty_list = _DummyConn([])
    df_empty_list = base_windows.fetch_mlb_rolling_stats(conn_empty_list, ["MLB789", "MLB999"])
    assert isinstance(df_empty_list, pd.DataFrame)
    assert df_empty_list.empty

    # Simulate Supabase returning None in response.data
    conn_none_data = _DummyConn(None)
    df_none_data = base_windows.fetch_mlb_rolling_stats(conn_none_data, ["MLB789", "MLB999"])
    assert isinstance(df_none_data, pd.DataFrame)
    assert df_none_data.empty