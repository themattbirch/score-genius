# backend/caching/scheduler_setup.py

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import config

# Import your functions
from src.scripts.data_fetcher import fetch_live_game_data
from src.scripts.archive_live_data import archive_live_data
from src.scripts.precompute_features import precompute_features
from src.scripts.model_inference import run_model_inference
from src.scripts.nba_stats_live import scheduled_update_nba_schedule
from caching.supabase_cache import cache_game_data

# Global variable to track archive execution per day
last_archive_date = None

def update_cache(league, season, date):
    try:
        # Get the current date string in the required format
        current_date = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        
        # Check if date is in the future (beyond API data)
        today = datetime.now().strftime("%Y-%m-%d")
        if current_date > today:
            print(f"Skipping update for future date: {current_date}")
            return
            
        print(f"Fetching data for: league={league}, season={season}, date={current_date}")
        data = fetch_live_game_data(league, season, current_date)
        
        if not data or "response" not in data or not data["response"]:
            print(f"No data available for {current_date}")
            return
            
        # Process each game in the response
        for game in data.get("response", []):
            game_id = game.get("id")
            if game_id:
                response = cache_game_data(game_id, game)
                print(f"Cached data for game ID: {game_id}")
    except Exception as e:
        print(f"Error in update_cache: {e}")
        # Log the stack trace for debugging
        import traceback
        traceback.print_exc()

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
        minutes=5,
        args=["12", "2024-2025", datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y-%m-%d")],
        id='update_cache_job'
    )
    
    # Schedule Update: Run twice daily at 5 AM and 3 PM PT
    scheduler.add_job(
        scheduled_update_nba_schedule,
        'cron', 
        hour=6,
        minute=0,
        timezone='America/Los_Angeles',
        id='schedule_update_morning'
    )
    
    scheduler.add_job(
        scheduled_update_nba_schedule,
        'cron', 
        hour=15,
        minute=0,
        timezone='America/Los_Angeles',
        id='schedule_update_afternoon'
    )
    
    # Data Pipeline: Precompute features daily at 5:26 p.m. PT
    scheduler.add_job(
        precompute_features, 
        'cron', 
        hour=6,
        minute=5,
        timezone='America/Los_Angeles',
        args=[config.DATABASE_URL],
        id='data_pipeline_job'
    )

    # Model Inference: Run daily at 6:10 p.m. PT
    scheduler.add_job(
        run_model_inference,
        'cron',
        hour=6,
        minute=10,
        timezone='America/Los_Angeles',
        id='model_inference_job'
    )
    
    # Archive jobs: Run at 6:15 PM PT and fallback at 6:25 PM PT
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=6,
        minute=15,
        timezone='America/Los_Angeles',
        id='archive_job_noon'
    )
    scheduler.add_job(
        attempt_archive_live_data,
        'cron',
        hour=15,
        minute=25,
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