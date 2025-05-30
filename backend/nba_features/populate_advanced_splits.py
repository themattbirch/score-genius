# backend/scripts/populate_advanced_splits.py

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from typing import Any, Optional, List, Dict, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import Client  # only for type checking
else:
    Client = Any  # fallback at runtime

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Path adjustments for imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in (BACKEND_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- Centralized Config and Supabase Client ---
try:
    import config
    from supabase import create_client
    from caching.supabase_client import supabase as supabase_singleton_instance
    logger.info("Successfully imported config and Supabase singleton.")
    SUPABASE_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(
        f"Failed to import config or supabase modules: {e}. "
        "Ensure your PYTHONPATH is set correctly and dependencies are installed."
    )
    supabase_singleton_instance = None
    SUPABASE_IMPORTS_SUCCESSFUL = False

# --- NBA utilities ---
from nba_features.utils import determine_season
from nba_features.data_processing_utils import (
    pivot_rpc_output_to_seasonal_splits_df,
    upsert_seasonal_splits_to_supabase,
)


def get_supabase_client_from_config() -> Optional["Client"]:
    """
    Returns the singleton Supabase client instance if available,
    otherwise creates a new one using settings from config.py.
    """
    if not SUPABASE_IMPORTS_SUCCESSFUL:
        logger.error("Cannot initialize Supabase client due to import failures.")
        return None

    if supabase_singleton_instance:
        logger.info("Using cached Supabase client instance.")
        return supabase_singleton_instance

    logger.info("Creating a new Supabase client from config.")
    if not getattr(config, 'SUPABASE_URL', None) or not getattr(config, 'SUPABASE_SERVICE_KEY', None):
        logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in config.")
        return None

    try:
        client: "Client" = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully.")
        return client
    except Exception as exc:
        logger.error(f"Error initializing Supabase client: {exc}", exc_info=True)
        return None


def process_season(supabase: "Client", season_year: int) -> bool:
    """Fetch RPC data, pivot, and upsert for a given season_year."""
    logger.info(f'Processing season {season_year} (for {season_year}-{season_year+1}).')
    if not hasattr(supabase, 'rpc'):
        logger.error("Invalid Supabase client: missing 'rpc' method.")
        return False

    try:
        response = supabase.rpc('get_nba_advanced_team_stats_split', {'p_season_year': season_year}).execute()
        if getattr(response, 'error', None):
            logger.error(f"RPC error for season {season_year}: {response.error}")
            return False

        data = getattr(response, 'data', [])
        if not data:
            logger.info(f"No data for season {season_year}; skipping.")
            return True

        rpc_df = pd.DataFrame(data)
        pivoted = pivot_rpc_output_to_seasonal_splits_df(rpc_df, season_year)
        if pivoted.empty and rpc_df.shape[0] > 0:
            logger.error(f"Pivot resulted in empty DataFrame for season {season_year}.")
            return False

        upserted = upsert_seasonal_splits_to_supabase(pivoted, supabase)
        if upserted:
            logger.info(f"Upsert succeeded for season {season_year}.")
        else:
            logger.error(f"Upsert failed for season {season_year}.")
        return upserted

    except Exception as exc:
        logger.error(f"Unexpected error processing season {season_year}: {exc}", exc_info=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate NBA seasonal advanced stats splits.")
    parser.add_argument("--season", type=int, help="Specific season year (e.g. 2022).")
    parser.add_argument("--all-historical", action="store_true", help="Process all seasons back to start-season.")
    parser.add_argument("--start-season", type=int, default=2018, help="Start year for historical processing.")

    args = parser.parse_args()

    supabase_client = get_supabase_client_from_config()
    if not supabase_client:
        logger.critical("No Supabase client available; exiting.")
        return

    try:
        current_year = int(determine_season(datetime.now()).split('-')[0])
    except Exception as exc:
        logger.error(f"Could not determine current season: {exc}")
        return

    seasons: List[int] = []
    if args.all_historical:
        seasons = list(range(args.start_season, current_year + 1))
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [current_year]

    for year in seasons:
        success = process_season(supabase_client, year)
        if not success:
            logger.error(f"Season {year} failed.")
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
