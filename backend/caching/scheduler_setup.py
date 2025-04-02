# In backend/caching/scheduler_setup.py

# --- Standard Imports ---
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING, Optional
import pytz
import time
import logging
import sys

# --- Project Imports (Relative to backend dir added to sys.path) ---
try:
    import config
    from caching.supabase_client import supabase

    # --- CORRECTED IMPORTS using 'src.scripts' ---
    from src.scripts.data_fetcher import fetch_live_game_data
    from src.scripts.archive_live_data import archive_live_data
    from src.scripts.precompute_features import precompute_features
    from src.scripts.model_inference import run_model_inference
    from src.scripts.nba_stats_live import scheduled_update_nba_schedule
    from src.scripts.nba_games_preview import (
        clear_old_games,
        build_game_preview,
        upsert_previews_to_supabase
    )
    from src.scripts.process_odds_data import main as process_odds_data_main
    from caching.supabase_cache import cache_game_data
    # --- END CORRECTED IMPORTS ---

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
    # NBAFeatureEngine not needed if update_expected_features is removed


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variable for Archiving Logic ---
last_archive_date: Optional[datetime.date] = None # Use date type hint

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
                    # Assuming cache_game_data handles potential errors
                    cache_game_data(game_id, game) # Assumes cache function exists
                    # logger.debug(f"Cached data for game ID: {game_id}") # Debug level might be better
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
            # Clean record for upsert into historical table
            record.pop('id', None) # Remove auto-increment ID if exists
            record.pop('current_form', None) # Remove fields not in historical? Verify schema.
            record.pop('current_streak', None)
            record.pop('last_fetched_at', None)
            record['updated_at'] = datetime.now(pytz.utc).isoformat() # Add timestamp

            # Ensure essential keys exist for upsert conflict resolution
            if not all(k in record for k in ["team_id", "season", "league_id"]):
                 logger.warning(f"Skipping record due to missing keys for conflict resolution: {record.get('team_id')}, {record.get('season')}")
                 skipped_count += 1
                 continue

            try:
                 upsert_response = supabase.table("nba_historical_team_stats").upsert(
                     record, on_conflict="team_id,season,league_id" # Ensure constraint name is correct
                 ).execute()
                 # Check upsert_response for errors if needed
                 archived_count += 1
            except Exception as upsert_err:
                 logger.error(f"Error upserting team stat record ({record.get('team_id')}): {upsert_err}")

        logger.info(f"Successfully archived/upserted {archived_count} team stat records. Skipped {skipped_count}.")

        # Clear the live table after successful processing
        try:
            # Use a filter that always evaluates to true to delete all rows
            delete_response = supabase.table("nba_live_team_stats").delete().neq('team_id', -1).execute() # Example: delete where team_id is not -1
            logger.info(f"Cleared live team stats table.")
        except Exception as delete_err:
            logger.error(f"Error clearing live team stats table: {delete_err}", exc_info=True)

        return archived_count
    except Exception as e:
        logger.error(f"Error archiving live team stats: {e}", exc_info=True)
        # Optional: Attempt emergency clear even on error? Risky if upsert failed partially.
        # try:
        #     supabase.table("nba_live_team_stats").delete().neq('team_id', -1).execute()
        #     logger.warning("Attempted emergency clear of live team stats table after error.")
        # except Exception as clear_err: logger.error(f"Failed emergency clear: {clear_err}")
        return 0


def attempt_archive_live_data():
    """
    Runs daily archive jobs for game data and team stats.
    Uses a global variable to ensure it only runs once per day.
    """
    global last_archive_date
    tz = pytz.timezone('America/Los_Angeles') # Define timezone
    today = datetime.now(tz).date()

    if last_archive_date == today:
        logger.info(f"Archive jobs already executed today ({today}). Skipping.")
        return

    logger.info(f"Running archive jobs for {today}...")
    try:
        # Archive game data (assuming archive_live_data function exists and handles its logic)
        logger.info("Attempting to archive live game data...")
        archive_live_data()
        logger.info("Live game data archive function called.")

        # Archive team stats
        logger.info("Attempting to archive live team stats...")
        archived_teams = archive_live_team_stats()
        logger.info(f"Live team stats archive function called (archived {archived_teams}).")

        # Update global variable only after successful completion
        last_archive_date = today
        logger.info(f"Archive jobs completed successfully for {today}.")

    except Exception as e:
        logger.error(f"Error during scheduled archive jobs: {e}", exc_info=True)
        # Do not update last_archive_date if jobs failed


def scheduled_update_plus_preview():
    """ Chains the schedule update and game preview generation/upsert. """
    logger.info("Running scheduled_update_plus_preview job...")
    try:
        logger.info("Updating NBA schedule...")
        scheduled_update_nba_schedule() # Assumes this works
        logger.info("Clearing old games from schedule...")
        clear_old_games() # Assumes this works
        logger.info("Building game previews...")
        preview_data = build_game_preview() # Assumes this works
        if preview_data:
            logger.info(f"Upserting {len(preview_data)} game previews...")
            upsert_previews_to_supabase(preview_data) # Assumes this works
            logger.info("Game previews upserted.")
        else:
            logger.info("No new game preview data generated.")
        logger.info("scheduled_update_plus_preview job finished.")
    except Exception as e:
         logger.error(f"Error in scheduled_update_plus_preview job: {e}", exc_info=True)


# --- Removed update_expected_features function ---
# def update_expected_features(): ...


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

    # Cache live game data frequently (adjust interval as needed)
    # Runs for the current date in Pacific Time
    scheduler.add_job(
        update_cache,
        'interval',
        minutes=15, # Example: Run every 15 minutes
        args=["12", config.CURRENT_SEASON if hasattr(config, 'CURRENT_SEASON') else "2024-2025", None], # Pass None for date to use current date
        id='update_cache_job',
        next_run_time=datetime.now(pytz.timezone('America/Los_Angeles')) + timedelta(seconds=10) # Start soon after launch
    )

    # Update schedule & generate previews twice daily
    scheduler.add_job(
        scheduled_update_plus_preview,
        'cron',
        hour='6,15', # Run at 6 AM and 3 PM PT
        minute=0,
        # timezone='America/Los_Angeles', # Already set on scheduler
        id='update_plus_preview_job'
    )

    # Process odds data shortly after schedule updates
    scheduler.add_job(
        process_odds_data_main,
        'cron',
        hour='6,15',
        minute=5,
        # timezone='America/Los_Angeles',
        id='process_odds_data_job'
    )

    # Data Pipeline: Precompute features daily (e.g., after morning odds update)
    #scheduler.add_job(
       # precompute_features,
        #'cron',
       # hour=6,
       # minute=10,
       # #timezone='America/Los_Angeles',
        #args=[config.DATABASE_URL],
       # id='precompute_features_job'
    #)

    # Run model inference/predictions daily (e.g., later in morning)
    #scheduler.add_job(
        #run_model_inference,
       # 'cron',
        #hour=9,
       # minute=10,
       # # timezone='America/Los_Angeles',
       # id='model_inference_job'
    #)

    # Archive data daily (e.g., early morning before new data fetching)
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=5, # Example: Run at 5 AM PT
        minute=0,
        # timezone='America/Los_Angeles',
        id='archive_job_daily'
    )

    # --- Removed expected features job ---
    # scheduler.add_job(update_expected_features, ...)

    try:
        scheduler.start()
        logger.info("Scheduler started. Running jobs based on schedule. Press Ctrl+C to exit.")
        # Keep the script running
        while True:
            time.sleep(60) # Check every minute or keep alive
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down...")
        scheduler.shutdown()
        logger.info("Scheduler shut down gracefully.")