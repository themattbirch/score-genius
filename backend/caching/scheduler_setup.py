import sys
import os

# Add the backend folder (project root) to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz
import config
import traceback
from caching.supabase_client import supabase

# Import your functions
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

# NEW: Import NBAFeatureEngine from your feature engineering module.
# Adjust the import path as needed.
from models.features import NBAFeatureEngine

# Global variable to track archive execution per day
last_archive_date = None

def update_cache(league, season, date):
    try:
        current_date = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        if current_date > today:
            print(f"Skipping update for future date: {current_date}")
            return
        print(f"Fetching data for: league={league}, season={season}, date={current_date}")
        data = fetch_live_game_data(league, season, current_date)
        if not data or "response" not in data or not data["response"]:
            print(f"No data available for {current_date}")
            return
        for game in data.get("response", []):
            game_id = game.get("id")
            if game_id:
                response = cache_game_data(game_id, game)
                print(f"Cached data for game ID: {game_id}")
    except Exception as e:
        print(f"Error in update_cache: {e}")
        import traceback
        traceback.print_exc()

def archive_live_team_stats():
    """
    Archives live team data from the nba_live_team_stats table to the nba_historical_team_stats table,
    then clears the nba_live_team_stats table.
    """
    try:
        print("Starting team stats archive process...")
        response = supabase.table("nba_live_team_stats").select("*").execute()
        if not response.data:
            print("No live team stats data to archive.")
            return 0
        live_team_stats = response.data
        print(f"Found {len(live_team_stats)} team stat records to archive.")
        archived_count = 0
        for record in live_team_stats:
            if 'id' in record:
                del record['id']
            if 'current_form' in record:
                del record['current_form']
            if 'current_streak' in record:
                del record['current_streak']
            if 'last_fetched_at' in record:
                del record['last_fetched_at']
            record['updated_at'] = datetime.now(pytz.timezone('UTC')).isoformat()
            result = supabase.table("nba_historical_team_stats").upsert(
                record, on_conflict="team_id,season,league_id"
            ).execute()
            archived_count += 1
        print(f"Successfully archived {archived_count} team stat records.")
        try:
            result = supabase.table("nba_live_team_stats").delete().gte('id', 0).execute()
            print(f"Cleared live team stats table. Result: {result}")
        except Exception as e:
            print(f"Error clearing live team stats table: {e}")
            traceback.print_exc()
        return archived_count
    except Exception as e:
        print(f"Error archiving live team stats: {e}")
        traceback.print_exc()
        try:
            result = supabase.table("nba_live_team_stats").delete().gte('id', 0).execute()
            print(f"Attempted emergency clear of live team stats table. Result: {result}")
        except Exception as clear_err:
            print(f"Failed emergency clear of live team stats table: {clear_err}")
        return 0

def attempt_archive_live_data():
    """
    Archives live game data from the nba_live_game_stats table to historical tables,
    then clears the nba_live_game_stats table.
    """
    global last_archive_date
    tz = pytz.timezone('America/Los_Angeles')
    today = datetime.now(tz).date()
    if last_archive_date != today:
        try:
            archive_live_data()
            archive_live_team_stats()
            last_archive_date = today
            print(f"Archive jobs executed for {today}")
        except Exception as e:
            print(f"Error during archive jobs: {e}")

def scheduled_update_plus_preview():
    """
    Chains the schedule update, clears old games, and updates game previews.
    This function:
      1. Updates the NBA schedule.
      2. Clears old (completed) games from nba_game_schedule.
      3. Builds new game preview data and upserts it to Supabase.
    """
    print("Running scheduled update plus preview...")
    scheduled_update_nba_schedule()
    clear_old_games()
    preview_data = build_game_preview()
    if preview_data:
        upsert_previews_to_supabase(preview_data)
        print(f"Upserted {len(preview_data)} game previews to Supabase.")
    else:
        print("No game preview data to upsert.")

# NEW: Define a function to update the expected features.
def update_expected_features():
    try:
        engine = NBAFeatureEngine(debug=True)
        features = engine.get_expected_features(enhanced=True)
        print(f"Expected features updated at {datetime.now()}: {features}")
        # Optionally, upsert or cache these features to Supabase or another store.
    except Exception as e:
        print(f"Error updating expected features: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    
    # Regular cache updates
    scheduler.add_job(
        update_cache,
        'interval',
        minutes=5,
        args=["12", "2024-2025", datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y-%m-%d")],
        id='update_cache_job'
    )
    
    # Chain the schedule update and game preview update into one job.
    # For example, run this job at 6:05 PT and 15:05 PT.
    scheduler.add_job(
        scheduled_update_plus_preview,
        'cron',
        hour=6,
        minute=0,
        timezone='America/Los_Angeles',
        id='update_plus_preview_morning'
    )
    scheduler.add_job(
        scheduled_update_plus_preview,
        'cron',
        hour=15,
        minute=0,
        timezone='America/Los_Angeles',
        id='update_plus_preview_afternoon'
    )
    
    # Process odds data at desired times
    scheduler.add_job(
        process_odds_data_main,
        'cron',
        hour=6,
        minute=5,
        timezone='America/Los_Angeles',
        id='process_odds_data_morning'
    )
    scheduler.add_job(
        process_odds_data_main,
        'cron',
        hour=15,
        minute=5,
        timezone='America/Los_Angeles',
        id='process_odds_data_afternoon'
    )
    
    # Data Pipeline: Precompute features daily at 6:05 PT
    scheduler.add_job(
        precompute_features, 
        'cron', 
        hour=6,
        minute=10,
        timezone='America/Los_Angeles',
        args=[config.DATABASE_URL],
        id='data_pipeline_job'
    )

    # Model Inference: Run daily at 6:10 PT
    scheduler.add_job(
        run_model_inference,
        'cron',
        hour=9,
        minute=10,
        timezone='America/Los_Angeles',
        id='model_inference_job'
    )
    
    # Archive jobs: Run at 6:20 PT and fallback at 16:20 PT
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=6,
        minute=20,
        timezone='America/Los_Angeles',
        id='archive_job_morning'
    )
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=16,
        minute=20,
        timezone='America/Los_Angeles',
        id='archive_job_afternoon'
    )

    # NEW: Add job to update expected features daily at 6:25 PT.
    scheduler.add_job(
        update_expected_features,
        'cron',
        hour=6,
        minute=25,
        timezone='America/Los_Angeles',
        id='update_expected_features_job'
    )
    
    scheduler.start()
    print("Scheduler started. Jobs are scheduled.")
    
    # Keep the script running so the scheduler remains active.
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
