#!/usr/bin/env python
import json
import requests
import sys, os
from datetime import datetime
from zoneinfo import ZoneInfo
from pprint import pprint

# Ensure the backend root is in the Python path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# --- Local Config & Variables ---
try:
    from config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )
    print("Successfully imported configuration variables from config.py")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate
_missing = [
    name
    for name, val in [
        ("API_SPORTS_KEY",       API_SPORTS_KEY),
        ("ODDS_API_KEY",         ODDS_API_KEY),
        ("SUPABASE_URL",         SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
    ]
    if not val
]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    sys.exit(1)

# API-Sports configuration
API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league: str, season: str, date: str, timezone: str = None) -> dict:
    """
    Fetches game data from API-Basketball for the specified date, league, and season.
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
    Fetches detailed player statistics for a given game.
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
    - For dates in October–December, returns "{year}-{year+1}"
    - For dates in January–September, returns "{year-1}-{year}"
    """
    if date_obj.month >= 10:
        return f"{date_obj.year}-{date_obj.year+1}"
    else:
        return f"{date_obj.year-1}-{date_obj.year}"

def main():
    # Set the test date to November 2, 2017
    test_date_str = "2017-11-02"
    test_date = datetime.fromisoformat(test_date_str)
    season = get_season_for_date(test_date)
    league = "12"  # NBA league code

    print(f"=== Testing NBA game stats for {test_date_str} (Season {season}) for the New York Knicks ===\n")

    # Fetch all game data for the test date
    games_data = get_games_by_date(league, season, test_date_str)
    if not games_data.get('response'):
        print("No game data found for the specified date.")
        return

    # Filter the games to include only those with "New York Knicks" as either home or away team
    knicks_games = [
        game for game in games_data.get('response', [])
        if "New York Knicks" in (
            game.get('teams', {}).get('home', {}).get('name', ""),
            game.get('teams', {}).get('away', {}).get('name', "")
        )
    ]
    if not knicks_games:
        print("No games involving the New York Knicks found for this date.")
        return

    for game in knicks_games:
        game_id = game.get('id')
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print("Game Information:")
        pprint(game)
        print("\nPlayer Box Statistics:")
        player_stats_data = get_player_box_stats(game_id)
        pprint(player_stats_data)
        print("=" * 60, "\n")

if __name__ == "__main__":
    main()
