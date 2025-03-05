# File: backend/src/scripts/nba_stats_live.py

import json
import requests
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pprint import pprint
import time
from caching.supabase_stats import upsert_live_game_stats_team

# Add the backend root to the Python path so we can import from caching
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import Supabase client and config
from caching.supabase_client import supabase  # Adjust if needed
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

###############################################################################
# 1) API Data Retrieval Functions
###############################################################################
def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/Los_Angeles') -> dict:
    """
    Fetch games for a given date from the API.
    Include 'periods' parameter to get quarter scores.
    """
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date,
        'timezone': timezone,
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

def get_team_stats(game_id: int) -> dict:
    """
    Fetch team-level statistics for a specific game.
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team statistics for game ID {game_id}: {e}")
        return {}

def print_game_info(game: dict):
    """
    Print key game information.
    """
    game_id = game.get('id')
    teams = game.get('teams', {})
    scores = game.get('scores', {})
    status = game.get('status', {})
    print(f"Game ID: {game_id}")
    print(f"  Away: {teams.get('away', {}).get('name')}  |  Home: {teams.get('home', {}).get('name')}")
    print(f"  Scores: Away {scores.get('away', {}).get('total')} - Home {scores.get('home', {}).get('total')}")
    print(f"  Status: {status.get('long')}")
    print(f"  Venue: {game.get('venue')}")
    print("-" * 60)

###############################################################################
# 2) Transformation Function
###############################################################################
def transform_team_stats(game: dict, team_stats: dict) -> dict:
    """
    Merge game-level info with home team statistics.
    For quarter scores, try multiple key formats; default to 0 if not provided.
    """
    def get_quarter_score(scores: dict, key1: str, key2: str, key3: str = None) -> int:
        # Try key1; if missing or falsy, try key2; if missing or falsy, try key3; else default to 0
        return scores.get(key1) or scores.get(key2) or scores.get(key3) or 0

    home_scores = game.get('scores', {}).get('home', {})
    away_scores = game.get('scores', {}).get('away', {})
    
    # Debug the scores structure to see what keys are available
    print("DEBUG - Raw home_scores structure:", json.dumps(home_scores, indent=2))
    
    transformed = {
        'game_id': game.get('id'),
        'home_team': game.get('teams', {}).get('home', {}).get('name'),
        'away_team': game.get('teams', {}).get('away', {}).get('name'),
        'home_score': home_scores.get('total'),
        'away_score': away_scores.get('total'),
        'home_q1': get_quarter_score(home_scores, 'q1', 'quarter1', 'quarter_1'),
        'home_q2': get_quarter_score(home_scores, 'q2', 'quarter2', 'quarter_2'),
        'home_q3': get_quarter_score(home_scores, 'q3', 'quarter3', 'quarter_3'),
        'home_q4': get_quarter_score(home_scores, 'q4', 'quarter4', 'quarter_4'),
        'home_ot': home_scores.get('ot', 0) or home_scores.get('over_time', 0) or 0,
        'away_q1': get_quarter_score(away_scores, 'q1', 'quarter1', 'quarter_1'),
        'away_q2': get_quarter_score(away_scores, 'q2', 'quarter2', 'quarter_2'),
        'away_q3': get_quarter_score(away_scores, 'q3', 'quarter3', 'quarter_3'),
        'away_q4': get_quarter_score(away_scores, 'q4', 'quarter4', 'quarter_4'),
        'away_ot': away_scores.get('ot', 0) or away_scores.get('over_time', 0) or 0,
        'game_date': game.get('date'),
    }
    # Merge team stats from the API response (for additional stats)
    transformed.update({
        'home_assists': team_stats.get('assists'),
        'home_steals': team_stats.get('steals'),
        'home_blocks': team_stats.get('blocks'),
        'home_turnovers': team_stats.get('turnovers'),
        'home_fouls': team_stats.get('fouls'),
        'away_assists': team_stats.get('opponent_assists'),
        'away_steals': team_stats.get('opponent_steals'),
        'away_blocks': team_stats.get('opponent_blocks'),
        'away_turnovers': team_stats.get('opponent_turnovers'),
        'away_fouls': team_stats.get('opponent_fouls'),
        'home_off_reb': team_stats.get('off_rebounds'),
        'home_def_reb': team_stats.get('def_rebounds'),
        'home_total_reb': team_stats.get('total_rebounds'),
        'away_off_reb': team_stats.get('opponent_off_rebounds'),
        'away_def_reb': team_stats.get('opponent_def_rebounds'),
        'away_total_reb': team_stats.get('opponent_total_rebounds'),
        'home_3pm': team_stats.get('three_pointers_made'),
        'home_3pa': team_stats.get('three_pointers_attempted'),
        'away_3pm': team_stats.get('opponent_three_pointers_made'),
        'away_3pa': team_stats.get('opponent_three_pointers_attempted'),
    })
    return transformed

###############################################################################
# 3) Main Driver: Fetch Live Games, Transform, and Upsert
###############################################################################
def run_live_games():
    # Use the current date in Pacific Time (live)
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    league = '12'          # NBA league ID
    season = '2024-2025'
    timezone = 'America/Los_Angeles'
    
    games_data = get_games_by_date(league, season, today_pst, timezone)
    if not games_data.get('response'):
        print(f"No game data found for {today_pst}.")
        return

    for game in games_data['response']:
        print_game_info(game)
        game_id = game.get('id')
        
        # Fetch team-level statistics for this game
        team_stats_data = get_team_stats(game_id)
        if not team_stats_data.get('response'):
            print(f"No team statistics found for game ID {game_id}.")
            continue

        # Identify home team statistics by matching the home team's ID
        home_team_id = game.get('teams', {}).get('home', {}).get('id')
        home_stats = None
        for stat in team_stats_data['response']:
            if stat.get('team', {}).get('id') == home_team_id:
                home_stats = stat
                break
        if not home_stats:
            print(f"Home team statistics not found for game ID {game_id}.")
            continue

        # Transform the raw game and team stats into the expected record format
        record = transform_team_stats(game, home_stats)
        print("Transformed Team Record:")
        pprint(record)

        # Upsert the transformed record into the Supabase table 'nba_live_game_stats'
        try:
            result = upsert_live_game_stats_team(record)
            print(f"Upsert result for game ID {game_id}: {result}")
        except Exception as e:
            print(f"Error upserting data for game ID {game_id}: {e}")
        
        print("=" * 60)

def main():
    print("Fetching live NBA game data for today (using Pacific Time) and upserting team stats...")
    run_live_games()

if __name__ == "__main__":
    main()