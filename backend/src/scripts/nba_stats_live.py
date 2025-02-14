# backend/src/scripts/nba_stats_live.py

import json
import requests
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo
import sys, os

# Add the backend root to the Python path so we can import from caching
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from caching.supabase_stats import upsert_live_game_stats
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/Los_Angeles') -> dict:
    """
    Fetch live game data from API-Basketball for the given date, league, season, and timezone.
    """
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date,
        'timezone': timezone
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched game data for {date} (Season {season}, Timezone: {timezone})")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def filter_live_games(games_data: dict) -> list:
    """
    Filters the games_data to only include live games.
    A live game is determined by a status 'short' value not equal to "NS" (Not Started) or "FT" (Finished).
    """
    live_games = []
    for game in games_data.get('response', []):
        status = game.get('status', {})
        status_short = status.get('short')
        if status_short not in ["NS", "FT"]:
            live_games.append(game)
    return live_games

def get_player_box_stats(game_id: int) -> dict:
    """
    Fetches detailed player statistics for a specific game.
    It replaces encoded apostrophes in the raw JSON text before parsing.
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

def print_game_info(game: dict) -> None:
    """
    Prints key game information.
    """
    game_id = game.get('id')
    teams = game.get('teams', {})
    score = game.get('scores', {})
    status = game.get('status', {})
    print(f"Game ID: {game_id}")
    print(f"  Away: {teams.get('away', {}).get('name')}  |  Home: {teams.get('home', {}).get('name')}")
    print(f"  Scores: Away {score.get('away', {}).get('total')} - Home {score.get('home', {}).get('total')}")
    print(f"  Status: {status.get('long')} (Short: {status.get('short')}, Timer: {status.get('timer')})")
    print(f"  Venue: {game.get('venue')}")
    print("-" * 60)

def run_live_games():
    """
    Fetches live games for today (based on Pacific Time) and upserts their player statistics.
    """
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    league = '12'
    season = '2024-2025'
    timezone = 'America/Los_Angeles'
    
    games_data = get_games_by_date(league, season, today_pst, timezone)
    live_games = filter_live_games(games_data)
    
    if not live_games:
        print("No live games found for today.")
        return
    
    print(f"\nFound {len(live_games)} live game(s):\n")
    for game in live_games:
        print_game_info(game)
        game_id = game.get('id')
        player_stats = get_player_box_stats(game_id)
        pprint(player_stats)
        
        if 'response' in player_stats:
            for stat in player_stats['response']:
                result = upsert_live_game_stats(game_id, stat)
                print(f"Upsert result for {stat['player']['name']}: {result}")
        else:
            print(f"No player stats found for game ID: {game_id}")
            
        print("=" * 80)

def main():
    print("Fetching live NBA game data for today (using Pacific Time)...")
    run_live_games()

if __name__ == "__main__":
    main()
