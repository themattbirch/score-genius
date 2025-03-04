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

def update_cache(league, season, date):
    try:
        data = fetch_live_game_data(league, season, date)
        if data:
            game_id = data.get('game_id')
            if game_id:
                response = cache_game_data(game_id, data)
                print("Cache updated:", response)
    except Exception as e:
        print("Error in update_cache:", e)

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
    
    # Regular cache updates: Supply the required arguments via 'args'
    scheduler.add_job(
        update_cache,
        'interval',
        minutes=1,
        args=["12", "2024-2025", datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y-%m-%d")],
        id='update_cache_job'
    )
    
    # Data Pipeline: Precompute features daily at 13:34 (1:34 p.m. PT)
    scheduler.add_job(
        precompute_features, 
        'cron', 
        hour=20,
        minute=33,
        timezone='America/Los_Angeles',
        args=[config.DATABASE_URL],
        id='data_pipeline_job'
    )
    
    # Model Inference (and retraining): Run daily at 13:36 (1:36 p.m. PT)
    scheduler.add_job(
        run_model_inference,
        'cron',
        hour=20,
        minute=34,
        timezone='America/Los_Angeles',
        id='model_inference_job'
    )
    
    # Archive jobs: Run at 13:39 (1:39 p.m. PT) and fallback at 18:00 (6:00 p.m. PT)
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=20,
        minute=35,
        timezone='America/Los_Angeles',
        id='archive_job_noon'
    )
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=20,  # 6:00 p.m. PT
        minute=36,
        timezone='America/Los_Angeles',
        id='archive_job_6pm'
    )
    
    scheduler.start()
    print("Scheduler started. Jobs are scheduled.")
    
    # Keep the script running so the scheduler remains active.
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
