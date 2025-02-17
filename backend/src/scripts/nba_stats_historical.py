# backend/src/scripts/nba_stats_historical.py

import json
import requests
import sys, os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pprint import pprint

# Add the backend root to the Python path so we can import from caching
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from caching.supabase_stats import upsert_historical_game_stats
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league: str, season: str, date: str, timezone: str = None) -> dict:
    """
    Fetch historical game data by date.
    Omits timezone by default for historical data.
    """
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date
    }
    if timezone:
        params['timezone'] = timezone
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched game data for {date} (Season {season})")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_player_box_stats(game_id: int) -> dict:
    """
    Fetches player statistics for a given game.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        raw_text = response.text.replace("&amp;apos;", "'").replace("&apos;", "'")
        data = json.loads(raw_text)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player statistics for game ID {game_id}: {e}")
        return {}

def get_season_for_date(date_obj: datetime) -> str:
    """
    Given a date, returns the NBA season string.
    For dates in October–December, season = "{year}-{year+1}".
    For dates in January–September, season = "{year-1}-{year}".
    """
    if date_obj.month >= 10:
        return f"{date_obj.year}-{date_obj.year+1}"
    else:
        return f"{date_obj.year-1}-{date_obj.year}"

def process_games_for_date(current_date: datetime):
    league = '12'
    season = get_season_for_date(current_date)
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n=== Processing data for {date_str} (Season {season}) ===")
    games_data = get_games_by_date(league, season, date_str)
    if not games_data.get('response'):
        print("No game data found for the specified date.")
        return

    for game in games_data['response']:
        game_id = game.get('id')
        # Use the game object's "date" field if available; otherwise use our current date
        game_date = game.get('date', date_str)
        print(f"Processing game ID: {game_id} for {date_str} (Season {season})")
        player_stats_data = get_player_box_stats(game_id)
        if 'response' in player_stats_data:
            for stat in player_stats_data['response']:
                result = upsert_historical_game_stats(game_id, stat, game_date=game_date)
                print(f"Upsert result for {stat['player']['name']}: {result}")
        else:
            print(f"No player stats found for game ID: {game_id}")

def run_historical_games():
    # Define your start and end dates (modify these as needed)
    start_date_str = "2019-02-07"  # Start date (YYYY-MM-DD)
    end_date_str = "2025-02-13"    # End date (YYYY-MM-DD)

    current_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)

    while current_date <= end_date:
        process_games_for_date(current_date)
        # Wait 5 minutes (300 seconds) before processing the next day
        print("Waiting 5 minutes before the next call...\n")
        time.sleep(300)
        current_date += timedelta(days=1)

def main():
    print("Fetching historical NBA game data and upserting stats to Supabase...")
    run_historical_games()

if __name__ == "__main__":
    main()
