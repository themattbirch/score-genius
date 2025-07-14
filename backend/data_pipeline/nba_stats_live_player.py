# File: backend/data_pipeline/nba_stats_live_player.py

import json
import requests
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import traceback

# Add the backend root to the Python path so we can import from caching
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from caching.supabase_stats import upsert_live_player_stats
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

def log_with_timestamp(message: str, level: str = "INFO"):
    """Log messages with a timestamp for easier debugging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level}: {message}")

def convert_minutes(time_str):
    """
    Convert a time string in 'MM:SS' format to a float representing total minutes.
    Falls back to float() if the string is numeric, or 0.0 if empty/invalid.
    """
    try:
        if not time_str or not isinstance(time_str, str) or ":" not in time_str:
            return float(time_str) if time_str and str(time_str).strip() != "" else 0.0
        minutes, seconds = time_str.split(":")
        return float(minutes) + float(seconds) / 60.0
    except Exception as e:
        print(f"Error converting minutes '{time_str}': {e}")
        return 0.0

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/Los_Angeles') -> dict:
    """
    Fetch games for a given date from the API, requesting data in the specified timezone.
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
        log_with_timestamp(f"Fetched game data for {date} (Season {season}, Timezone: {timezone})")
        log_with_timestamp(f"Status Code: {response.status_code}")
        log_with_timestamp(f"Request URL: {response.url}")
        return response.json()
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error fetching game data for {date}: {e}", "ERROR")
        traceback.print_exc()
        return {}

def filter_live_games(games_data: dict) -> list:
    """
    Filter games to include only those currently in progress.
    A game is considered live if its status short code (uppercase) is in:
      {"Q1", "Q2", "Q3", "Q4", "OT", "BT", "HT"}
    """
    live_statuses = {"Q1", "Q2", "Q3", "Q4", "OT", "BT", "HT"}
    live_games = []
    for game in games_data.get('response', []):
        status_code = game.get('status', {}).get('short', '').upper()
        if status_code in live_statuses:
            live_games.append(game)
    return live_games

def get_player_box_stats(game_id: int) -> dict:
    """
    Fetch player box statistics for a specific game from the API.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        raw_text = response.text.replace("&amp;apos;", "'").replace("&apos;", "'")
        data = json.loads(raw_text)
        if 'response' in data and len(data['response']) > 0:
            log_with_timestamp("Sample Player Stats API Structure (first record):")
            print(json.dumps(data['response'][0], indent=2))
        return data
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error fetching player statistics for game ID {game_id}: {e}", "ERROR")
        traceback.print_exc()
        return {}

def print_game_info(game: dict):
    """
    Print key info about a single game, for debugging/logging.
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

def modify_player_stats_for_upsert(game_id: int, player_stats: dict) -> dict:
    """
    Prepare player statistics for upsert:
      1) Convert 'minutes' to numeric.
      2) Extract stats, defaulting missing values to 0.
    """
    print("\nDEBUG RAW PLAYER DATA:")
    print(json.dumps(player_stats, indent=2))

    team_id = player_stats.get('team', {}).get('id', 0)
    player_id = player_stats.get('player', {}).get('id', 0)
    player_name = player_stats.get('player', {}).get('name', 'Unknown')

    minutes_raw = player_stats.get('minutes', "0:00")
    minutes_numeric = convert_minutes(minutes_raw)

    points = int(player_stats.get("points", 0))
    rebounds = int(player_stats.get("rebounds", {}).get("total", 0))
    assists = int(player_stats.get("assists", 0))
    steals = int(player_stats.get("steals", 0))
    blocks = int(player_stats.get("blocks", 0))
    turnovers = int(player_stats.get("turnovers", 0))
    fouls = int(player_stats.get("fouls", 0))

    fg = player_stats.get("field_goals", {})
    fg_made = int(fg.get("total", 0))
    fg_attempted = int(fg.get("attempts", 0))

    three = player_stats.get("threepoint_goals", {})
    three_made = int(three.get("total", 0))
    three_attempted = int(three.get("attempts", 0))

    ft_made, ft_attempted = 0, 0
    if 'freethrows_goals' in player_stats and isinstance(player_stats['freethrows_goals'], dict):
        ft = player_stats['freethrows_goals']
        ft_made = int(ft.get('total', 0))
        ft_attempted = int(ft.get('attempts', 0))
    elif 'free_throws' in player_stats and isinstance(player_stats['free_throws'], dict):
        ft = player_stats['free_throws']
        ft_made = int(ft.get('total', 0))
        ft_attempted = int(ft.get('attempts', 0))

    modified_stats = {
        "game_id": game_id,
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "minutes_numeric": minutes_numeric,
        "points": points,
        "rebounds": rebounds,
        "assists": assists,
        "steals": steals,
        "blocks": blocks,
        "turnovers": turnovers,
        "fouls": fouls,
        "fg_made": fg_made,
        "fg_attempted": fg_attempted,
        "three_made": three_made,
        "three_attempted": three_attempted,
        "ft_made": ft_made,
        "ft_attempted": ft_attempted
    }

    print(f"Final stats for {player_name}: FT={ft_made}/{ft_attempted}, steals={steals}, blocks={blocks}, turnovers={turnovers}, fouls={fouls}")
    return modified_stats

def enhanced_upsert_player_stats(game_id: int, original_player_stats: dict, game_status: str = None) -> dict:
    """
    Enhanced upsert that modifies the player stats before calling upsert_live_player_stats().
    Adds the game status to the record; if no status is provided, defaults to "LIVE".
    """
    modified_stats = modify_player_stats_for_upsert(game_id, original_player_stats)
    if not game_status:
        game_status = "LIVE"
    modified_stats["status"] = game_status
    try:
        result = upsert_live_player_stats(game_id, modified_stats)
        return result
    except Exception as e:
        print(f"Error in enhanced upsert: {e}")
        traceback.print_exc()
        return {"error": str(e)}

###############################################################################
# Main driver: run_live_games
###############################################################################

def run_live_games():
    """
    Main function to fetch live game data in Pacific Time, then process player statistics.
    Only games that are currently live (status in {"Q1","Q2","Q3","Q4","OT","BT","HT"}) are included.
    """
    pacific_tz = ZoneInfo("America/Los_Angeles")
    today_pst = datetime.now(pacific_tz).strftime('%Y-%m-%d')
    
    league = '12'            # NBA league ID
    season = '2024-2025'     # Adjust if needed
    timezone = 'America/Los_Angeles'
    
    log_with_timestamp(f"Fetching live NBA player data for {today_pst} (Pacific Time)...")
    
    games_data = get_games_by_date(league, season, today_pst, timezone)
    if not games_data.get('response'):
        print(f"No games found for {today_pst} in Pacific Time.")
        return
    
    # Filter to include only live games based on status (e.g., Q1, Q2, etc.)
    live_games = filter_live_games(games_data)
    if not live_games:
        print("No live (in-progress) games found for today.")
        return
    
    print(f"\nFound {len(live_games)} live game(s) for {today_pst}:\n")
    for game in live_games:
        print_game_info(game)
        game_id = game.get('id')
        # Extract game status from the game data and force uppercase
        game_status = game.get('status', {}).get('short', '').upper()
        if not game_status:
            game_status = "LIVE"
        
        player_stats_data = get_player_box_stats(game_id)
        if not player_stats_data.get('response'):
            print(f"No player stats found for game ID: {game_id}")
            print("=" * 80)
            continue
        
        print(f"Processing {len(player_stats_data['response'])} player records for game ID {game_id}\n")
        for player_stat in player_stats_data['response']:
            player_name = player_stat.get('player', {}).get('name', 'Unknown')
            print(f"Processing player: {player_name}")
            result = enhanced_upsert_player_stats(game_id, player_stat, game_status)
            print(f"Enhanced upsert result for {player_name}: {result}\n")
        print("=" * 80)

def main():
    """
    Entry point if run as a script.
    """
    print("Running nba_stats_live_player.py (Pacific Time logic)...")
    try:
        run_live_games()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
