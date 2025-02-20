# backend/src/scripts/nba_stats_final.py

import requests
import json
from pprint import pprint
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from config import API_SPORTS_KEY

# Add the backend root to the Python path so we can import upsert functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from caching.supabase_stats import upsert_2024_25_game_stats

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league: str, season: str, date: str, timezone: str = None) -> dict:
    """
    Fetch final (completed) game data by date, optionally using a specific timezone.
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
        print(f"Fetched game data for {date} (Season {season}, Timezone: {timezone})")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_team_stats(game_id: int) -> dict:
    """
    (Optional) Fetch team-level statistics for a specific final game.
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched team statistics for game ID {game_id}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team statistics for game ID {game_id}: {e}")
        return {}

def get_player_box_stats(game_id: int) -> dict:
    """
    Fetch detailed player statistics for a specific final game.
    Replaces encoded apostrophes before parsing JSON.
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

def post_process_stats(stats_data: dict) -> dict:
    """
    Optional debugging function to print out some key stats.
    """
    for stat in stats_data.get('response', []):
        player_name = stat.get('player', {}).get('name')
        fg_info = stat.get('field_goals', {})
        attempts = fg_info.get('attempts')
        made = fg_info.get('total')
        print(f"Player {player_name}: FG Attempts={attempts}, Made={made}")
    return stats_data

def run_all_games():
    league = '12'         # NBA
    season = '2024-2025'  # Adjust if your final data is from a different season

    # Get today's date in Pacific Time
    pacific_now = datetime.now(ZoneInfo("America/Los_Angeles"))
    todays_date_str = pacific_now.strftime('%Y-%m-%d')

    # Fetch final games for the given date (in LA time)
    games_data = get_games_by_date(league, season, todays_date_str, "America/Los_Angeles")
    if not games_data.get('response'):
        print("No game data found for the specified date.")
        return

    for game in games_data['response']:
        game_id = game.get('id')
        teams = game.get('teams', {})
        scores = game.get('scores', {})
        status = game.get('status', {})

        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Away: {teams.get('away', {}).get('name')} | Home: {teams.get('home', {}).get('name')}")
        print(f"Scores: Away {scores.get('away', {}).get('total')} - Home {scores.get('home', {}).get('total')}")
        print(f"Status: {status.get('long')}")
        
        # (Optional) team-level stats
        team_stats = get_team_stats(game_id)
        pprint(team_stats)

        # player-level stats
        player_stats = get_player_box_stats(game_id)
        post_process_stats(player_stats)

        # Upsert each player's final stats into the 2024_25 table
        for stat in player_stats.get('response', []):
            player_name = stat.get('player', {}).get('name')
            result = upsert_2024_25_game_stats(game_id, stat)
            print(f"Upsert result for {player_name}: {result}")

        print("=" * 60)

def main():
    print("Fetching final NBA game data (using LA time) and upserting stats to Supabase...")
    run_all_games()

if __name__ == "__main__":
    main()
