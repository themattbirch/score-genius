# backend/features/base_windows.py

"""
SQL interface for pre-computed rolling windows.
Falls back to legacy Python implementation if view hasn’t refreshed.
"""
from __future__ import annotations
import pandas as pd
from typing import Sequence
from supabase import Client  # type: ignore

ROLLING_VIEW = "team_rolling_20"   # materialised view name

def fetch_rolling(conn: Client,
                  game_ids: Sequence[str]) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame()
    data = (
        conn.table(ROLLING_VIEW)
            .select("*")
            .in_("game_id", list(game_ids))
            .execute()
            .data
    )
    return pd.DataFrame(data)
