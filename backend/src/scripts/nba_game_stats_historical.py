# File: backend/src/scripts/nba_player_stats_historical.py

import json
import requests
import sys, os
import time
from datetime import datetime, timedelta

# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase
# IMPORTANT: Now import the upsert function:
from caching.supabase_stats import upsert_historical_game_stats

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def convert_minutes(time_str):
    """Convert a time string in 'MM:SS' format to a float representing minutes."""
    try:
        if not time_str or ":" not in time_str:
            return 0.0
        m, s = map(int, time_str.split(":"))
        return m + (s / 60.0)
    except:
        return 0.0

def get_games_by_date(league, season, date):
    url = f"{BASE_URL}/games"
    params = {"league": league, "season": season, "date": date}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json().get("response", [])

def get_player_stats(game_id):
    url = f"{BASE_URL}/games/statistics/players"
    params = {"ids": game_id}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    raw_text = resp.text.replace("&amp;apos;", "'").replace("&apos;", "'")
    data = json.loads(raw_text)
    print(f"\n=== DEBUG: Full Player Stats for Game {game_id} ===")
    print(json.dumps(data, indent=2))
    return data.get("response", [])

def process_day(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    # Build a rough "YYYY-YYYY+1" season
    season = f"{date_obj.year}-{date_obj.year+1}" if date_obj.month >= 10 else f"{date_obj.year-1}-{date_obj.year}"
    print(f"\n=== {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    processed_count = 0
    for g in games_list:
        if g.get("status", {}).get("short") != "FT":
            continue  # Only final games

        game_id = g["id"]
        home_team = g["teams"]["home"]["name"]
        away_team = g["teams"]["away"]["name"]
        print(f"\nProcessing game: {away_team} @ {home_team} (Game ID: {game_id})")

        player_stats = get_player_stats(game_id)
        if not player_stats:
            print(f"No player stats found for game ID: {game_id}")
            continue

        players_processed = 0
        for p in player_stats:
            print(f"\n--- DEBUG: Raw Player Data ---")
            print(json.dumps(p, indent=2))

            # Instead of manually building & inserting, we now call the function:
            print(f"\n[INFO] Upserting historical record for player: {p.get('player', {}).get('name','Unknown')}")
            try:
                # This function flips the name AND looks up the correct team_name
                # using dynamic fallback for older seasons.
                res = upsert_historical_game_stats(game_id, p, date_str)
                print(f"[INFO] upsert_historical_game_stats result: {res}")
            except Exception as e:
                print(f"[ERROR] upsert_historical_game_stats failed: {e}")

            players_processed += 1

        print(f"Processed {players_processed} players for game {game_id}")
        processed_count += 1

    print(f"\nProcessed {processed_count} games for {date_str}")

def main():
    start_date = datetime(2021, 10, 19)
    end_date   = datetime(2025, 3, 5)

    print(f"Starting historical data import from {start_date} to {end_date}")
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)
        current += timedelta(days=1)

    print("\nCompleted processing historical player stats")

if __name__ == "__main__":
    main()
