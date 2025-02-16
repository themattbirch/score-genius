import requests
import json
from datetime import datetime
from .supabase_client import supabase
from config import API_SPORTS_KEY

################################################################################
#                          CONFIG & TEAM NAME CACHE                             #
################################################################################

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

TEAM_NAME_CACHE = {}
TEAM_DEBUG_MODE = True

################################################################################
#                          HELPER: FIX PLAYER NAME                              #
################################################################################

def fix_player_name(name: str) -> str:
    """
    If the name has two parts and the first part is not just an initial,
    flip them. E.g. 'Noah Vonleh' -> 'Vonleh Noah'.
    If the first part ends with '.', leave it unchanged.
    """
    parts = name.split()
    if len(parts) == 2:
        first, last = parts
        if first.endswith('.'):
            return name
        return f"{last} {first}"
    return name

################################################################################
#                          HELPER: GET TEAM NAME                                #
################################################################################

def get_team_name_from_api(team_id: int, league='12', season='2019-2020') -> str:
    """
    Retrieves the team name from the API and caches the result.
    """
    if team_id in TEAM_NAME_CACHE:
        return TEAM_NAME_CACHE[team_id]
    
    url = f"{BASE_URL}/teams"
    params = {'id': team_id, 'league': league, 'season': season}
    if TEAM_DEBUG_MODE:
        print(f"\n** DEBUG: Requesting team name for team_id={team_id}, league={league}, season={season}")
        print(f"** DEBUG: GET {url} with params={params} and headers={HEADERS}")
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        if TEAM_DEBUG_MODE:
            print(f"** DEBUG: /teams response: {data}\n")
        
        teams_list = data.get('response', [])
        if not teams_list:
            TEAM_NAME_CACHE[team_id] = "Unknown"
            return "Unknown"
        
        team_info = teams_list[0]
        name = team_info.get('name', "Unknown")
        TEAM_NAME_CACHE[team_id] = name
        return name
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team name for team_id {team_id}: {e}")
        TEAM_NAME_CACHE[team_id] = "Unknown"
        return "Unknown"

################################################################################
#                           STAT EXTRACTION HELPER                              #
################################################################################

def get_stat_value(player_stats: dict, key: str, subkey: str = None, default=0):
    """
    Safely retrieves nested data from the 'player_stats' dictionary.
    If subkey is provided, attempts to access player_stats[key][subkey],
    otherwise, returns player_stats[key]. Returns default if not found.
    """
    if subkey:
        val_top = player_stats.get(key, {}).get(subkey)
        if val_top is not None:
            return val_top
        return player_stats.get('statistics', {}).get(key, {}).get(subkey, default)
    else:
        val_top = player_stats.get(key)
        if val_top is not None:
            return val_top
        return player_stats.get('statistics', {}).get(key, default)

################################################################################
#                     UPSERT: HISTORICAL GAME STATS                             #
################################################################################

def upsert_historical_game_stats(game_id: int, player_stats: dict, game_date: str) -> dict:
    """
    Upserts historical game statistics into the Supabase table.
    
    Parameters:
      - game_id: The ID of the game.
      - player_stats: The statistics dictionary for a player.
      - game_date: The actual game date (as an ISO‑formatted string) from the API.
    
    Returns the result of the upsert operation.
    """
    team_data = player_stats.get('team', {})
    raw_team_id = team_data.get('id', 0)
    team_id = raw_team_id if raw_team_id is not None else 0
    team_name = get_team_name_from_api(team_id, league='12', season='2019-2020') if team_id else 'Unknown'
    
    # Fix the player's name
    raw_name = player_stats['player']['name']
    player_name_fixed = fix_player_name(raw_name)

    # Convert the 'minutes' value from "MM:SS" to a float (e.g., "27:21" -> 27.35)
    raw_minutes = get_stat_value(player_stats, 'minutes', default=None)
    numeric_minutes = parse_minutes(raw_minutes)  # parse_minutes is defined below

    stats_data = {
        'game_id': game_id,
        'player_id': player_stats['player']['id'],
        'player_name': player_name_fixed,
        'team_id': team_id,
        'team_name': team_name,
        'minutes': numeric_minutes,
        'points': get_stat_value(player_stats, 'points'),
        'rebounds': get_stat_value(player_stats, 'rebounds', 'total'),
        'assists': get_stat_value(player_stats, 'assists'),
        'steals': get_stat_value(player_stats, 'steals'),
        'blocks': get_stat_value(player_stats, 'blocks'),
        'turnovers': get_stat_value(player_stats, 'turnovers'),
        'fouls': get_stat_value(player_stats, 'fouls'),
        'fg_made': get_stat_value(player_stats, 'field_goals', 'total'),
        'fg_attempted': get_stat_value(player_stats, 'field_goals', 'attempts'),
        'three_made': get_stat_value(player_stats, 'threepoint_goals', 'total'),
        'three_attempted': get_stat_value(player_stats, 'threepoint_goals', 'attempts'),
        'ft_made': get_stat_value(player_stats, 'freethrows_goals', 'total'),
        'ft_attempted': get_stat_value(player_stats, 'freethrows_goals', 'attempts'),
        'game_date': game_date,  # Use the actual game date passed in
        'updated_at': datetime.utcnow().isoformat()
    }
    
    result = (
        supabase.table('nba_historical_game_stats')
        .upsert(stats_data, on_conflict='game_id, player_id')
        .execute()
    )
    return result

################################################################################
#                     UPSERT: LIVE GAME STATS                                   #
################################################################################

def upsert_live_game_stats(game_id: int, player_stats: dict) -> dict:
    team_data = player_stats.get('team', {})
    raw_team_id = team_data.get('id', 0)
    team_id = raw_team_id if raw_team_id is not None else 0
    team_name = get_team_name_from_api(team_id, league='12', season='2024-2025') if team_id else 'Unknown'
    
    # Fix the player's name
    raw_name = player_stats['player']['name']
    player_name_fixed = fix_player_name(raw_name)

    raw_minutes = get_stat_value(player_stats, 'minutes', default=None)
    numeric_minutes = parse_minutes(raw_minutes)

    stats_data = {
        'game_id': game_id,
        'player_id': player_stats['player']['id'],
        'player_name': player_name_fixed,
        'team_id': team_id,
        'team_name': team_name,
        'minutes': numeric_minutes,
        'points': get_stat_value(player_stats, 'points'),
        'rebounds': get_stat_value(player_stats, 'rebounds', 'total'),
        'assists': get_stat_value(player_stats, 'assists'),
        'steals': get_stat_value(player_stats, 'steals'),
        'blocks': get_stat_value(player_stats, 'blocks'),
        'turnovers': get_stat_value(player_stats, 'turnovers'),
        'fouls': get_stat_value(player_stats, 'fouls'),
        'fg_made': get_stat_value(player_stats, 'field_goals', 'total'),
        'fg_attempted': get_stat_value(player_stats, 'field_goals', 'attempts'),
        'three_made': get_stat_value(player_stats, 'threepoint_goals', 'total'),
        'three_attempted': get_stat_value(player_stats, 'threepoint_goals', 'attempts'),
        'ft_made': get_stat_value(player_stats, 'freethrows_goals', 'total'),
        'ft_attempted': get_stat_value(player_stats, 'freethrows_goals', 'attempts'),
        'game_date': str(datetime.now().date()),
        'updated_at': datetime.utcnow().isoformat()
    }

    result = (
        supabase.table('nba_live_game_stats')
        .upsert(stats_data, on_conflict='game_id, player_id')
        .execute()
    )
    return result

################################################################################
#                     UPSERT: 2024-25 FINAL GAME STATS                          #
################################################################################

def upsert_2024_25_game_stats(game_id: int, player_stats: dict) -> dict:
    team_data = player_stats.get('team', {})
    raw_team_id = team_data.get('id', 0)
    team_id = raw_team_id if raw_team_id is not None else 0
    team_name = get_team_name_from_api(team_id, league='12', season='2024-2025') if team_id else 'Unknown'
    
    # Fix the player's name
    raw_name = player_stats['player']['name']
    player_name_fixed = fix_player_name(raw_name)

    raw_minutes = get_stat_value(player_stats, 'minutes', default=None)
    numeric_minutes = parse_minutes(raw_minutes)

    stats_data = {
        'game_id': game_id,
        'player_id': player_stats['player']['id'],
        'player_name': player_name_fixed,
        'team_id': team_id,
        'team_name': team_name,
        'minutes': numeric_minutes,
        'points': get_stat_value(player_stats, 'points'),
        'rebounds': get_stat_value(player_stats, 'rebounds', 'total'),
        'assists': get_stat_value(player_stats, 'assists'),
        'steals': get_stat_value(player_stats, 'steals'),
        'blocks': get_stat_value(player_stats, 'blocks'),
        'turnovers': get_stat_value(player_stats, 'turnovers'),
        'fouls': get_stat_value(player_stats, 'fouls'),
        'fg_made': get_stat_value(player_stats, 'field_goals', 'total'),
        'fg_attempted': get_stat_value(player_stats, 'field_goals', 'attempts'),
        'three_made': get_stat_value(player_stats, 'threepoint_goals', 'total'),
        'three_attempted': get_stat_value(player_stats, 'threepoint_goals', 'attempts'),
        'ft_made': get_stat_value(player_stats, 'freethrows_goals', 'total'),
        'ft_attempted': get_stat_value(player_stats, 'freethrows_goals', 'attempts'),
        'game_date': str(datetime.now().date()),
        'updated_at': datetime.utcnow().isoformat()
    }

    result = (
        supabase.table('nba_2024_25_game_stats')
        .upsert(stats_data, on_conflict='game_id, player_id')
        .execute()
    )
    return result

################################################################################
#                           MINUTES PARSER                                      #
################################################################################

def parse_minutes(time_str: str) -> float:
    """
    Convert a 'MM:SS' string to a float representing total minutes.
    E.g. "27:21" -> 27.35. If invalid or empty, returns 0.0.
    """
    if not time_str or ':' not in time_str:
        return 0.0
    try:
        minutes_part, seconds_part = time_str.split(':')
        total_minutes = float(minutes_part) + float(seconds_part) / 60.0
        return round(total_minutes, 2)
    except ValueError:
        return 0.0
