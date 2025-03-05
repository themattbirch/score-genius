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
from caching.supabase_client import supabase
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

###############################################################################
# Helper Functions
###############################################################################

def get_nested_value(data, *keys, default=None):
    """
    Safely navigate nested dictionaries, returning default if path doesn't exist.
    
    Args:
        data (dict): The dictionary to navigate
        *keys: The sequence of keys to traverse
        default: Value to return if the path doesn't exist
        
    Returns:
        The value at the path or the default value
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default

###############################################################################
# 1) API Data Retrieval Functions
###############################################################################

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/Los_Angeles') -> dict:
    """
    Fetch games for a given date from the API.
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
        data = response.json()
        # Debug the full response structure
        print(f"DEBUG - Team Stats API Response Structure:")
        print(json.dumps(data.get('response', [])[:1], indent=2))  # Print just the first item to avoid huge output
        return data
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

def transform_team_stats(game: dict, team_stats_data: dict) -> dict:
    """
    Transform raw game and team stats data into the expected record format.
    
    Args:
        game: The game object from the /games endpoint
        team_stats_data: The full team stats response from the /games/statistics/teams endpoint
        
    Returns:
        A dictionary containing all relevant stats for database insertion
    """
    def get_quarter_score(scores: dict, key1: str, key2: str, key3: str = None) -> int:
        # Try key1; if missing or falsy, try key2; if missing or falsy, try key3; else default to 0
        return scores.get(key1) or scores.get(key2) or scores.get(key3) or 0

    home_scores = game.get('scores', {}).get('home', {})
    away_scores = game.get('scores', {}).get('away', {})
    
    # Debug the scores structure
    print("DEBUG - Raw home_scores structure:", json.dumps(home_scores, indent=2))
    
    # Get team IDs for matching
    home_team_id = game.get('teams', {}).get('home', {}).get('id')
    away_team_id = game.get('teams', {}).get('away', {}).get('id')
    
    # Find home and away team stats objects from the response array
    home_stats = None
    away_stats = None
    for stat in team_stats_data.get('response', []):
        team_id = stat.get('team', {}).get('id')
        if team_id == home_team_id:
            home_stats = stat
        elif team_id == away_team_id:
            away_stats = stat
    
    print("DEBUG - Home team ID:", home_team_id)
    print("DEBUG - Away team ID:", away_team_id)
    
    # Base record with game and quarter information
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
    
    # Process home team stats if available
    if home_stats:
        # Basic stats directly in the object
        transformed.update({
            'home_assists': home_stats.get('assists', 0),
            'home_steals': home_stats.get('steals', 0),
            'home_blocks': home_stats.get('blocks', 0),
            'home_turnovers': home_stats.get('turnovers', 0),
            'home_fouls': home_stats.get('personal_fouls', 0),
        })
        
        # Rebounds stats may be nested
        rebounds = home_stats.get('rebounds', {})
        transformed.update({
            'home_off_reb': rebounds.get('offence', 0),  # Note British spelling
            'home_def_reb': rebounds.get('defense', 0),
            'home_total_reb': rebounds.get('total', 0),
        })
        
        # 3-point stats come from threepoint_goals
        threepoint = home_stats.get('threepoint_goals', {})
        transformed.update({
            'home_3pm': threepoint.get('total', 0),
            'home_3pa': threepoint.get('attempts', 0),
        })
    
    # Process away team stats if available
    if away_stats:
        # Basic stats directly in the object
        transformed.update({
            'away_assists': away_stats.get('assists', 0),
            'away_steals': away_stats.get('steals', 0),
            'away_blocks': away_stats.get('blocks', 0),
            'away_turnovers': away_stats.get('turnovers', 0),
            'away_fouls': away_stats.get('personal_fouls', 0),
        })
        
        # Rebounds stats may be nested
        rebounds = away_stats.get('rebounds', {})
        transformed.update({
            'away_off_reb': rebounds.get('offence', 0),  # Note British spelling
            'away_def_reb': rebounds.get('defense', 0),
            'away_total_reb': rebounds.get('total', 0),
        })
        
        # 3-point stats come from threepoint_goals
        threepoint = away_stats.get('threepoint_goals', {})
        transformed.update({
            'away_3pm': threepoint.get('total', 0),
            'away_3pa': threepoint.get('attempts', 0),
        })
    
    # Print the extracted stats for debugging
    print("DEBUG - Extracted team stats:")
    pprint(transformed)
    
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

        # Print the full response structure for debugging
        print("FULL TEAM STATS RESPONSE:", json.dumps(team_stats_data, indent=2))
        
        # Transform the raw game and team stats into the expected record format
        record = transform_team_stats(game, team_stats_data)
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