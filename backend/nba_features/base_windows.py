# backend/features/base_windows.py

from __future__ import annotations
import logging
from typing import Sequence, Union, Optional, Any # Added Any for Client type hint

import pandas as pd
# Removed unused numpy, traceback, os, datetime, time, functools, Path, plotting libs
# Removed supabase import from caching, as conn is passed in
from supabase import Client # Assuming Client type hint is desired

# --- Logger Configuration ---
# Configure logging for this module
# Consider moving basicConfig to a higher level (e.g., main script/entry point)
# if multiple modules configure it. If kept here, ensure it doesn't conflict.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Constants --
ROLLING_VIEW = "team_rolling_20"   # Materialized view name in Supabase
# EPSILON = 1e-6 # Likely belongs in utils.py if needed globally

# -- Module Functions --
__all__ = ["fetch_rolling"] # Explicitly export the function

def fetch_rolling(
    conn: Optional[Client], # Use Client type hint if available/desired
    game_ids: Sequence[Union[str, int]]
) -> pd.DataFrame:
    """
    Fetch pre-computed rolling-window stats from the Supabase materialized view.

    Args:
        conn: Active Supabase client connection.
        game_ids: A sequence of game IDs (strings or integers) to fetch data for.

    Returns:
        A pandas DataFrame containing the fetched rolling stats,
        or an empty DataFrame if no connection, no game_ids, or on database error.
    """
    if conn is None:
        logger.warning("fetch_rolling: No Supabase database connection provided. Returning empty DataFrame.")
        return pd.DataFrame()

    # Ensure game_ids is not None and is a non-empty sequence
    if not game_ids: # Handles None and empty list/tuple/etc.
        logger.debug("fetch_rolling: Received empty or None game_ids sequence. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # Ensure all game IDs are strings for the Supabase 'in_' filter
        str_ids = [str(g) for g in game_ids]

        logger.debug(f"Querying view '{ROLLING_VIEW}' for {len(str_ids)} game IDs.")
        # Execute the query against the specified table/view
        resp = (
            conn
            .table(ROLLING_VIEW)
            .select("*") # Select all columns from the view
            .in_("game_id", str_ids) # Filter by the provided game IDs
            .execute()
        )

        # Process the response
        data = resp.data or [] # Use empty list if data is None
        df = pd.DataFrame(data)
        logger.debug(f"fetch_rolling: Fetched {len(df)} rows from '{ROLLING_VIEW}'.")
        return df

    except Exception as e:
        # Log any exception during the database query
        logger.error(f"fetch_rolling: Error querying Supabase view '{ROLLING_VIEW}': {e}", exc_info=True)
        # Return an empty DataFrame on error to prevent downstream issues
        return pd.DataFrame()