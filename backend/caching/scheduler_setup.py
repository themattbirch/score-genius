from apscheduler.schedulers.background import BackgroundScheduler
from src.scripts.data_fetcher import fetch_live_game_data
from caching.supabase_cache import cache_game_data  # Fixed import path

def update_cache():
    # Example: fetch live game data for a specific game or all games for a day
    data = fetch_live_game_data()  # This should return a dict with game_id and data
    if data:
        game_id = data.get('game_id')
        if game_id:
            response = cache_game_data(game_id, data)
            print("Cache updated:", response)

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_cache, 'interval', minutes=1)  # Update cache every minute
    scheduler.start()

    # Keep the script running (for example, in development)
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()