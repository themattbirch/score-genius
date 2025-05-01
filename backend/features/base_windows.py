# backend/features/base_windows.py

from __future__ import annotations
import logging
from typing import Sequence, Union, Optional

import pandas as pd
from supabase import Client  # type: ignore

logger = logging.getLogger(__name__)

ROLLING_VIEW = "team_rolling_20"   # materialized view name

def fetch_rolling(
    conn: Optional[Client],
    game_ids: Sequence[Union[str, int]]
) -> pd.DataFrame:
    """
    Fetch pre-computed rolling-window stats from the Supabase materialized view.
    Returns an empty DataFrame if no connection, no game_ids, or on error.
    """
    if conn is None:
        logger.warning("fetch_rolling: no database connection provided, returning empty DataFrame.")
        return pd.DataFrame()

    if not game_ids:
        logger.debug("fetch_rolling: received empty game_ids, returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # ensure IDs are strings
        str_ids = [str(g) for g in game_ids]
        resp = (
            conn
            .table(ROLLING_VIEW)
            .select("*")
            .in_("game_id", str_ids)
            .execute()
        )
        data = resp.data or []
        df = pd.DataFrame(data)
        logger.debug(f"fetch_rolling: fetched {len(df)} rows for {len(str_ids)} game_ids.")
        return df

    except Exception as e:
        logger.error(f"fetch_rolling: error querying {ROLLING_VIEW}: {e}", exc_info=True)
        return pd.DataFrame()
