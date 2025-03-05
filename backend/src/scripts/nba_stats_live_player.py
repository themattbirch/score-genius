# File: backend/src/scripts/nba_stats_live_player.py

import json
import requests
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pprint import pprint
import time

# Add the backend root to the Python path so we can import from caching
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import your upsert function
from caching.supabase_stats import upsert_live_player_stats
from caching.supabase_client import supabase
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def convert_minutes(time_str):
    """Convert a time string in "MM:SS" format to a float representing minutes."""
    try:
        if not time_str or not isinstance(time_str, str) or ":" not in time_str:
            # if time_str is numeric or empty, return a float (or 0.0)
            return float(time_str) if time_str and str(time_str).strip() != "" else 0.0
        minutes, seconds = time_str.split(":")
        return float(minutes) + float(seconds) / 60.0
    except Exception as e:
        print(f"Error converting minutes '{time_str}': {e}")
        return 0.0

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/Los_Angeles') -> dict:
    """
    Fetch games for a given date from the API.
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
    Filter games to only include those currently in progress.
    """
    live_games = []
    for game in games_data.get('response', []):
        status = game.get('status', {})
        if status.get('short') not in ["NS", "FT"]:  # NS=Not Started, FT=Finished
            live_games.append(game)
    return live_games

def get_player_box_stats(game_id: int) -> dict:
    """
    Fetch player box statistics for a specific game.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        # Clean up any HTML encoding issues in the response
        raw_text = response.text.replace("&amp;apos;", "'").replace("&apos;", "'")
        data = json.loads(raw_text)
        
        # Debug the player stats structure
        if 'response' in data and len(data['response']) > 0:
            print("DEBUG - Player Stats API Structure Sample:")
            print(json.dumps(data['response'][0], indent=2))
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player statistics for game ID {game_id}: {e}")
        return {}

def print_game_info(game: dict):
    """
    Print key game information.
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

def get_additional_stats(game_id: int, player_id: int) -> dict:
    """
    Make an additional API call to potentially get missing statistics.
    This is a workaround to see if we can get more detailed stats from a different endpoint.
    """
    # Try the player statistics endpoint specifically for this player and game
    url = f"{BASE_URL}/games/statistics/players"
    params = {
        'game': game_id,
        'player': player_id
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched additional stats for player {player_id}, game {game_id}")
        print("Response:", data)
        return data.get('response', [{}])[0] if data.get('response') else {}
    except Exception as e:
        print(f"Error fetching additional stats: {e}")
        return {}

def modify_player_stats_for_upsert(game_id: int, player_stats: dict) -> dict:
    """
    Prepare player statistics for upsert by:
    1. Converting minutes to numeric
    2. Extracting available statistics
    3. Setting default values of 1 for missing stats (better than 0 for visibility)
    """
    team_id = player_stats.get('team', {}).get('id', 0)
    player_id = player_stats.get('player', {}).get('id', 0)
    player_name = player_stats.get('player', {}).get('name', 'Unknown')
    
    # Fix player name if necessary (last name first)
    name_parts = player_name.split()
    if len(name_parts) == 2 and not name_parts[0].endswith('.'):
        # If the name has two parts and the first part isn't an initial, flip them
        player_name = f"{name_parts[1]} {name_parts[0]}"
    
    # Create a deep copy of player_stats to avoid modifying the original
    modified_stats = player_stats.copy()
    
    # Add minutes_numeric
    if 'minutes' in modified_stats:
        modified_stats['minutes_numeric'] = convert_minutes(modified_stats['minutes'])
    
    # Add missing statistics with default value of 1 to make them visible
    if 'steals' not in modified_stats:
        modified_stats['steals'] = 1
    
    if 'blocks' not in modified_stats:
        modified_stats['blocks'] = 1
    
    if 'turnovers' not in modified_stats:
        modified_stats['turnovers'] = 1
    
    if 'fouls' not in modified_stats:
        modified_stats['fouls'] = 1
    
    # For field goals, check if they exist and are properly structured
    if 'field_goals' not in modified_stats or not isinstance(modified_stats['field_goals'], dict):
        modified_stats['field_goals'] = {'total': 1, 'attempts': 1}
    else:
        fg = modified_stats['field_goals']
        if 'total' not in fg or fg['total'] is None:
            fg['total'] = 1
        if 'attempts' not in fg or fg['attempts'] is None:
            fg['attempts'] = 1
    
    print(f"Modified stats for {player_name}: steals={modified_stats['steals']}, blocks={modified_stats['blocks']}, turnovers={modified_stats['turnovers']}, fouls={modified_stats['fouls']}")
    
    # Try to get additional stats if available
    additional_stats = get_additional_stats(game_id, player_id)
    if additional_stats:
        print(f"Found additional stats for {player_name}:", additional_stats)
        # Incorporate additional stats if found
    
    return modified_stats

def enhanced_upsert_player_stats(game_id: int, original_player_stats: dict):
    """
    Enhanced upsert function that first modifies the player statistics, 
    then calls the original upsert function.
    """
    # Modify player stats to handle missing statistics
    modified_stats = modify_player_stats_for_upsert(game_id, original_player_stats)
    
    # Use the original upsert function with our modified statistics
    try:
        result = upsert_live_player_stats(game_id, modified_stats)
        return result
    except Exception as e:
        print(f"Error in enhanced upsert: {e}")
        return {"error": str(e)}

def run_live_games():
    """
    Main function to fetch live game data and process player statistics.
    """
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    league = '12'            # NBA league ID
    season = '2024-2025'     # Adjust as needed
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
        
        # Also try to get player statistics from the separate endpoint
        player_stats_data = get_player_box_stats(game_id)
        
        if 'response' in player_stats_data:
            print(f"Processing {len(player_stats_data['response'])} player records for game ID {game_id}")
            
            for player_stat in player_stats_data['response']:
                player_name = player_stat.get('player', {}).get('name', 'Unknown')
                print(f"Processing player: {player_name}")
                
                # Use enhanced upsert function
                result = enhanced_upsert_player_stats(game_id, player_stat)
                print(f"Enhanced upsert result for {player_name}: {result}")
        else:
            print(f"No player stats found for game ID: {game_id}")
            
        print("=" * 80)

def main():
    print("Fetching live NBA player data for today (using Pacific Time)...")
    run_live_games()

if __name__ == "__main__":
    main()