# File: backend/src/scripts/nba_player_stats_historical.py

import json
import requests
import sys
import os
import time
from datetime import datetime, timedelta

# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase
# Import the historical upsert function from supabase_stats:
from caching.supabase_stats import upsert_historical_game_stats

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def convert_minutes(time_str):
    """Convert 'MM:SS' to a float in minutes."""
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
    # Build a rough NBA season like "2023-2024" or "2022-2023"
    season = f"{date_obj.year}-{date_obj.year+1}" if date_obj.month >= 10 else f"{date_obj.year-1}-{date_obj.year}"
    print(f"\n=== {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    processed_count = 0
    for g in games_list:
        if g.get("status", {}).get("short") != "FT":
            # Only handle final games
            continue

        game_id = g["id"]
        home_team_info = g.get("teams", {}).get("home", {})
        away_team_info = g.get("teams", {}).get("away", {})
        home_team_id = home_team_info.get("id")
        home_team_name = home_team_info.get("name", "Unknown H")
        away_team_id = away_team_info.get("id")
        away_team_name = away_team_info.get("name", "Unknown A")
        # === END ADDED BLOCK ===
        print(f"\nProcessing game: {away_team_name} @ {home_team_name} (Game ID: {game_id})")

        player_stats = get_player_stats(game_id)
        if not player_stats:
            print(f"No player stats found for game ID: {game_id}")
            continue

        players_processed = 0
        for p in player_stats:
            print(f"\n--- DEBUG: Raw Player Data ---")
            print(json.dumps(p, indent=2))

                        # === ADD THIS BLOCK to find the player's actual team name ===
            player_team_id = p.get("team", {}).get("id")
            player_actual_team_name = "Unknown" # Default
            if player_team_id == home_team_id:
                player_actual_team_name = home_team_name
            elif player_team_id == away_team_id:
                player_actual_team_name = away_team_name
            # === END ADDED BLOCK ===

        # INFO Print
            print(f"[INFO] Upserting historical record for {p.get('player', {}).get('name', 'Unknown')} (Team: {player_actual_team_name})") # Updated print
            # Actual Upsert Call
            try:
                # Correct call with 4 arguments
                res = upsert_historical_game_stats(game_id, p, date_str, player_actual_team_name)
                print(f"[INFO] upsert_historical_game_stats result: {res}")
            except Exception as e:
                print(f"[ERROR] upsert_historical_game_stats failed: {e}")

            players_processed += 1 # Increment count only ONCE


def main():
    start_date = datetime(2025, 4, 16)
    end_date   = datetime(2025, 4, 21)

    print(f"Starting historical data import from {start_date} to {end_date}")
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)  # Rate-limit to avoid hitting API too hard
        current += timedelta(days=1)

    print("\nCompleted processing historical player stats")

if __name__ == "__main__":
    main()
