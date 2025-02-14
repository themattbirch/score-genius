# backend/src/scripts/nba_stats_historical.py

import json
import requests
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo
import sys, os

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

def run_historical_games():
    league = '12'
    season = '2019-2020'
    date = '2019-11-23'  # Example historical date

    games_data = get_games_by_date(league, season, date)
    if not games_data.get('response'):
        print("No game data found for the specified date.")
        return

    for game in games_data['response']:
        game_id = game.get('id')
        print(f"Processing game ID: {game_id}")
        player_stats_data = get_player_box_stats(game_id)
        if 'response' in player_stats_data:
            for stat in player_stats_data['response']:
                result = upsert_historical_game_stats(game_id, stat)
                print(f"Upsert result for {stat['player']['name']}: {result}")
        else:
            print(f"No player stats found for game ID: {game_id}")

def main():
    print("Fetching historical NBA game data and upserting stats to Supabase...")
    run_historical_games()

if __name__ == "__main__":
    main()
