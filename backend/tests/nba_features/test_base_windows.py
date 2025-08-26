# backend/tests/nba_features/test_base_windows.py
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

from backend.nba_features import base_windows

# ----------------------------------------------------------------------
# Helpers: extremely lightweight fakes that mimic the supabase‑py chain
# ----------------------------------------------------------------------
class _DummyResp:
    def __init__(self, data):
        self.data = data

class _DummyQuery:
    def __init__(self, data_holder, capture):
        self._data_holder = data_holder
        self._capture = capture

    # keep returning self so the call‑chain works
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
      • None        → simulate empty query
      • Exception() → simulate database error
    """
    def __init__(self, payload):
        self._payload = payload
        self.capture: dict[str, any] = {}

    def table(self, table_name):        # .table("team_rolling_20")
        self.capture["table"] = table_name
        return _DummyQuery(self._payload, self.capture)

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_fetch_rolling_success():
    rows = [
        {"game_id": "1", "metric": 10.0},
        {"game_id": "2", "metric": 20.0},
    ]
    conn = _DummyConn(rows)

    df = base_windows.fetch_rolling(conn, [1, 2])

    # DataFrame correctness
    assert not df.empty and len(df) == 2
    assert set(df["game_id"]) == {"1", "2"}

    # Proper query composition
    assert conn.capture["table"] == base_windows.ROLLING_VIEW
    assert conn.capture["column"] == "game_id"
    assert conn.capture["ids"] == ["1", "2"]  # IDs coerced to str

def test_fetch_rolling_none_conn_returns_empty():
    out = base_windows.fetch_rolling(None, [123])
    assert isinstance(out, pd.DataFrame) and out.empty

def test_fetch_rolling_empty_ids_returns_empty():
    conn = _DummyConn([])  # Data won't be used
    out = base_windows.fetch_rolling(conn, [])
    assert out.empty

def test_fetch_rolling_db_error_graceful():
    conn = _DummyConn(Exception("DB blew up"))
    out = base_windows.fetch_rolling(conn, [1])
    assert out.empty
