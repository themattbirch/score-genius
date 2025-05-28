# backend/mlb_features/base_windows.py

"""
Fetches pre-computed MLB rolling-window statistics from a Supabase materialized view.
This module is intended to be part of the mlb_features package.
"""

from __future__ import annotations
import logging
from typing import Sequence, Union, Optional

import pandas as pd
from supabase import Client # For type hinting the Supabase client

# --- Logger Configuration ---
# Configure logging for this module.
# It's often good practice to configure basicConfig at the application's entry point.
# If configured per module, ensure it's idempotent or uses a shared configuration.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# -- MLB Specific Constants --
# This constant should match the name of your materialized view in Supabase
# that contains the pre-calculated rolling window statistics for MLB teams/games.
# Example: "mlb_team_rolling_20_game_stats" or "vw_mlb_team_rolling_features"
MLB_ROLLING_STATS_VIEW = "mlb_team_rolling_features" # <<< REPLACE WITH YOUR ACTUAL MLB VIEW NAME

__all__ = ["fetch_mlb_rolling_stats"]

def fetch_mlb_rolling_stats(
    conn: Optional[Client],
    game_ids: Sequence[Union[str, int]]
) -> pd.DataFrame:
    """
    Fetch pre-computed MLB rolling-window stats from the Supabase materialized view.

    Args:
        conn: Active Supabase client connection.
        game_ids: A sequence of MLB game IDs (strings or integers) to fetch data for.

    Returns:
        A pandas DataFrame containing the fetched MLB rolling stats,
        or an empty DataFrame if no connection, no game_ids, or on database error.
    """
    if conn is None:
        logger.warning("fetch_mlb_rolling_stats: No Supabase database connection provided. Returning empty DataFrame.")
        return pd.DataFrame()

    if not game_ids: # Handles None, empty list, empty tuple, etc.
        logger.debug("fetch_mlb_rolling_stats: Received empty or None game_ids sequence. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # Ensure all game IDs are strings for the Supabase 'in_' filter, as game_id is often text in DB.
        str_game_ids = [str(gid) for gid in game_ids]

        logger.info(f"Querying MLB rolling stats view '{MLB_ROLLING_STATS_VIEW}' for {len(str_game_ids)} game IDs.")
        
        response = (
            conn
            .table(MLB_ROLLING_STATS_VIEW)
            .select("*") # Select all available columns from the view
            .in_("game_id", str_game_ids) # Filter by the provided game IDs
            .execute()
        )

        # Process the response
        data = response.data or [] # Use empty list if data is None to prevent error with pd.DataFrame
        df = pd.DataFrame(data)
        
        if not df.empty:
            logger.info(f"fetch_mlb_rolling_stats: Fetched {len(df)} rows from '{MLB_ROLLING_STATS_VIEW}'.")
        else:
            logger.info(f"fetch_mlb_rolling_stats: No data returned from '{MLB_ROLLING_STATS_VIEW}' for the given game_ids.")
            if logger.isEnabledFor(logging.DEBUG): # Avoid formatting string if not debugging
                 logger.debug(f"Game IDs queried: {str_game_ids}")

        return df

    except Exception as e:
        # Log any exception during the database query
        logger.error(f"fetch_mlb_rolling_stats: Error querying Supabase view '{MLB_ROLLING_STATS_VIEW}': {e}", exc_info=True)
        # Return an empty DataFrame on error to prevent downstream issues
        return pd.DataFrame()