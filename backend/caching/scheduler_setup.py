# In backend/caching/scheduler_setup.py

# --- Standard Imports ---
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING, Optional
import pytz
import time
import logging
import sys

# --- Project Imports ---
try:
    import config
    from caching.supabase_client import supabase

    # --- IMPORTS ---
    from data_pipeline.archive_live_data import archive_live_data
    from data_pipeline.nba_stats_live import scheduled_update_nba_schedule
    from data_pipeline.nba_games_preview import (
        clear_old_games,
        build_game_preview,
        upsert_previews_to_supabase
    )
    from caching.supabase_cache import cache_game_data

    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Error importing project modules in scheduler: {e}", exc_info=True)
    PROJECT_MODULES_IMPORTED = False
    # Define dummy functions if needed for scheduler to load, or exit
    def fetch_live_game_data(*args, **kwargs): logging.error("Dummy fetch_live_game_data called!"); return {}
    def archive_live_data(*args, **kwargs): logging.error("Dummy archive_live_data called!")
    def precompute_features(*args, **kwargs): logging.error("Dummy precompute_features called!")
    def run_model_inference(*args, **kwargs): logging.error("Dummy run_model_inference called!")
    def scheduled_update_nba_schedule(*args, **kwargs): logging.error("Dummy scheduled_update_nba_schedule called!")
    def clear_old_games(*args, **kwargs): logging.error("Dummy clear_old_games called!")
    def build_game_preview(*args, **kwargs): logging.error("Dummy build_game_preview called!"); return []
    def upsert_previews_to_supabase(*args, **kwargs): logging.error("Dummy upsert_previews_to_supabase called!")
    def process_odds_data_main(*args, **kwargs): logging.error("Dummy process_odds_data_main called!")
    def cache_game_data(*args, **kwargs): logging.error("Dummy cache_game_data called!")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variable for Archiving Logic ---
last_archive_date: Optional[datetime.date] = None 

# --- Job Functions ---

def update_cache(league: str = "12", season: str = "2024-2025", date_str: Optional[str] = None):
    """Fetches live data for a given date and caches it."""
    try:
        tz = pytz.timezone('America/Los_Angeles') # Use a timezone
        target_date = datetime.now(tz).strftime("%Y-%m-%d") if date_str is None else date_str
        today_date = datetime.now(tz).strftime("%Y-%m-%d")

        # Avoid fetching for future dates relative to today in PT
        if target_date > today_date:
            logger.info(f"Skipping update_cache for future date: {target_date}")
            return

        logger.info(f"Running update_cache for: league={league}, season={season}, date={target_date}")
        data = fetch_live_game_data(league, season, target_date) # Assumes fetch function exists

        if not data or "response" not in data or not data["response"]:
            logger.info(f"No live game data available from fetcher for {target_date}")
            return

        cached_count = 0
        for game in data.get("response", []):
            game_id = game.get("id")
            if game_id:
                try:
                    cache_game_data(game_id, game) 
                    cached_count += 1
                except Exception as cache_err:
                     logger.error(f"Error caching game ID {game_id}: {cache_err}")
        logger.info(f"Finished update_cache for {target_date}. Attempted to cache {cached_count} games.")

    except Exception as e:
        logger.error(f"Error in update_cache job: {e}", exc_info=True)


def archive_live_team_stats():
    """
    Archives live team data from nba_live_team_stats to nba_historical_team_stats,
    then clears the live table. Returns number of records archived.
    """
    if not supabase:
         logger.error("Supabase client not available for archive_live_team_stats.")
         return 0
    try:
        logger.info("Starting archive_live_team_stats process...")
        response = supabase.table("nba_live_team_stats").select("*").execute()

        if not hasattr(response, 'data'):
             logger.error("Invalid response from Supabase fetching live team stats.")
             return 0
        if not response.data:
            logger.info("No live team stats data to archive.")
            return 0

        live_team_stats = response.data
        logger.info(f"Found {len(live_team_stats)} team stat records to archive.")
        archived_count = 0
        skipped_count = 0
        for record in live_team_stats:
            record.pop('id', None) #
            record.pop('current_form', None) 
            record.pop('current_streak', None)
            record.pop('last_fetched_at', None)
            record['updated_at'] = datetime.now(pytz.utc).isoformat() 

            if not all(k in record for k in ["team_id", "season", "league_id"]):
                 logger.warning(f"Skipping record due to missing keys for conflict resolution: {record.get('team_id')}, {record.get('season')}")
                 skipped_count += 1
                 continue

            try:
                 upsert_response = supabase.table("nba_historical_team_stats").upsert(
                     record, on_conflict="team_id,season,league_id" # Ensure constraint name is correct
                 ).execute()
                 archived_count += 1
            except Exception as upsert_err:
                 logger.error(f"Error upserting team stat record ({record.get('team_id')}): {upsert_err}")

        logger.info(f"Successfully archived/upserted {archived_count} team stat records. Skipped {skipped_count}.")

        try:
            delete_response = supabase.table("nba_live_team_stats").delete().neq('team_id', -1).execute() # Example: delete where team_id is not -1
            logger.info(f"Cleared live team stats table.")
        except Exception as delete_err:
            logger.error(f"Error clearing live team stats table: {delete_err}", exc_info=True)

        return archived_count
    except Exception as e:
        logger.error(f"Error archiving live team stats: {e}", exc_info=True)
        return 0


def attempt_archive_live_data():
    """
    Runs daily archive jobs for game data and team stats.
    Uses a global variable to ensure it only runs once per day.
    """
    global last_archive_date
    tz = pytz.timezone('America/Los_Angeles') 
    today = datetime.now(tz).date()

    if last_archive_date == today:
        logger.info(f"Archive jobs already executed today ({today}). Skipping.")
        return

    logger.info(f"Running archive jobs for {today}...")
    try:
        logger.info("Attempting to archive live game data...")
        archive_live_data()
        logger.info("Live game data archive function called.")

        # Archive team stats
        logger.info("Attempting to archive live team stats...")
        archived_teams = archive_live_team_stats()
        logger.info(f"Live team stats archive function called (archived {archived_teams}).")

        last_archive_date = today
        logger.info(f"Archive jobs completed successfully for {today}.")

    except Exception as e:
        logger.error(f"Error during scheduled archive jobs: {e}", exc_info=True)

def scheduled_update_plus_preview():
    """ Chains the schedule update and game preview generation/upsert. """
    logger.info("Running scheduled_update_plus_preview job...")
    try:
        logger.info("Updating NBA schedule...")
        scheduled_update_nba_schedule() 
        logger.info("Clearing old games from schedule...")
        clear_old_games() 
        logger.info("Building game previews...")
        preview_data = build_game_preview() 
        if preview_data:
            logger.info(f"Upserting {len(preview_data)} game previews...")
            upsert_previews_to_supabase(preview_data) 
            logger.info("Game previews upserted.")
        else:
            logger.info("No new game preview data generated.")
        logger.info("scheduled_update_plus_preview job finished.")
    except Exception as e:
         logger.error(f"Error in scheduled_update_plus_preview job: {e}", exc_info=True)


if __name__ == "__main__":
    if not PROJECT_MODULES_IMPORTED:
         logger.critical("Failed to import necessary project modules. Cannot start scheduler. Exiting.")
         sys.exit(1)
    if not config:
         logger.critical("Config module not loaded. Cannot start scheduler. Exiting.")
         sys.exit(1)
    if not supabase:
         logger.warning("Supabase client not available. Some jobs might fail.")

    scheduler = BackgroundScheduler(timezone=pytz.timezone('America/Los_Angeles')) # Set default timezone

    # --- Schedule Definitions ---

    # Cache live game data frequently 
    scheduler.add_job(
        update_cache,
        'interval',
        minutes=15, 
        args=["12", config.CURRENT_SEASON if hasattr(config, 'CURRENT_SEASON') else "2024-2025", None], # Pass None for date to use current date
        id='update_cache_job',
        next_run_time=datetime.now(pytz.timezone('America/Los_Angeles')) + timedelta(seconds=10) # Start soon after launch
    )

    # Update schedule & generate previews twice daily
    scheduler.add_job(
        scheduled_update_plus_preview,
        'cron',
        hour='6,15', 
        minute=0,
        id='update_plus_preview_job'
    )

    # Process odds data shortly after schedule updates
    scheduler.add_job(
        process_odds_data_main,
        'cron',
        hour='6,15',
        minute=5,
        id='process_odds_data_job'
    )

    # Archive data daily (e.g., early morning before new data fetching)
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=5, 
        minute=0,
        id='archive_job_daily'
    )

    try:
        scheduler.start()
        logger.info("Scheduler started. Running jobs based on schedule. Press Ctrl+C to exit.")
        while True:
            time.sleep(60) 
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down...")
        scheduler.shutdown()
        logger.info("Scheduler shut down gracefully.")