# /backend/caching/scheduler_setup.py

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import config 

from src.scripts.data_fetcher import fetch_live_game_data
from caching.supabase_cache import cache_game_data  # Fixed import path
from src.scripts.archive_live_data import archive_live_data  # Import the archival function


# Import the precompute_features function
from src.scripts.precompute_features import precompute_features

last_archive_date = None  # Track the date of the last successful archival

def update_cache():
    data = fetch_live_game_data()  # Should return a dict with at least {'game_id': ...}
    if data:
        game_id = data.get('game_id')
        if game_id:
            response = cache_game_data(game_id, data)
            print("Cache updated:", response)

def attempt_archive_live_data():
    """
    Checks if archive_live_data has been run today.
    If not, run it now and update last_archive_date.
    """
    global last_archive_date
    tz = pytz.timezone('America/Los_Angeles')
    today = datetime.now(tz).date()

    # Only run archive if we haven't run it yet today
    if last_archive_date != today:
        try:
            archive_live_data()
            last_archive_date = today
            print(f"Archive job executed for {today}")
        except Exception as e:
            # If something fails, you can log it or decide how you want to handle the error
            print(f"Error during archive job: {e}")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    # Run update_cache every minute
    scheduler.add_job(update_cache, 'interval', minutes=1)

    # Primary archival job at 11:00 p.m. Pacific Time
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=23,
        minute=0,
        timezone='America/Los_Angeles',
        id='archive_job_11pm'
    )

    # Fallback archival job at 12:00 p.m. Pacific Time (next day)
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=12,
        minute=0,
        timezone='America/Los_Angeles',
        id='archive_job_noon'
    )

    # Preprocessing Script
    scheduler.add_job(
        precompute_features, 
        'cron', 
        hour=1,  # Run at 1 AM daily
        minute=30,
        args=[config.SUPABASE_URL]  # Or use a dedicated DB connection string from config
    )

    scheduler.start()

    # Keep the script running (for example, in development)
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()