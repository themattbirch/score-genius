from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import config

# Import your functions
from src.scripts.data_fetcher import fetch_live_game_data
from src.scripts.archive_live_data import archive_live_data
from src.scripts.precompute_features import precompute_features
from src.scripts.model_inference import run_model_inference
from caching.supabase_cache import cache_game_data

# Global variable to track archive execution per day
last_archive_date = None

def update_cache():
    data = fetch_live_game_data()  # Should return a dict with at least {'game_id': ...}
    if data:
        game_id = data.get('game_id')
        if game_id:
            response = cache_game_data(game_id, data)
            print("Cache updated:", response)

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
            last_archive_date = today
            print(f"Archive job executed for {today}")
        except Exception as e:
            print(f"Error during archive job: {e}")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    # Regular cache updates (if still needed)
    scheduler.add_job(update_cache, 'interval', minutes=1)
    
    # Archive jobs: Run at 12:00 p.m. PT daily, and fallback at 6:00 p.m. PT
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=12,
        minute=0,
        timezone='America/Los_Angeles',
        id='archive_job_noon'
    )
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=18,  # 6:00 p.m. PT
        minute=0,
        timezone='America/Los_Angeles',
        id='archive_job_6pm'
    )
    
    # Data Pipeline: Precompute features daily at 1:00 p.m. PT, with a fallback at 7:00 p.m.
    scheduler.add_job(
        precompute_features, 
        'cron', 
        hour=13,  # 1:00 p.m.
        minute=0,
        timezone='America/Los_Angeles',
        id='data_pipeline_job'
    )
    scheduler.add_job(
        precompute_features,
        'cron',
        hour=19,  # 7:00 p.m.
        minute=0,
        timezone='America/Los_Angeles',
        id='data_pipeline_fallback_job'
    )
    
    # Model Inference (and optionally retraining): Run daily at 1:05 p.m. PT, fallback at 7:05 p.m.
    scheduler.add_job(
        run_model_inference,
        'cron',
        hour=13,  # 1:05 p.m.
        minute=5,
        timezone='America/Los_Angeles',
        id='model_inference_job'
    )
    scheduler.add_job(
        run_model_inference,
        'cron',
        hour=19,  # 7:05 p.m.
        minute=5,
        timezone='America/Los_Angeles',
        id='model_inference_fallback_job'
    )
    
    scheduler.start()
    print("Scheduler started. Jobs are scheduled.")
    
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
