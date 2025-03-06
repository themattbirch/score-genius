import requests
import json
from datetime import datetime
from caching.supabase_client import supabase  # Changed to an absolute import
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
    flip them. E.g., 'Noah Vonleh' -> 'Vonleh Noah'. If the first part ends with '.',
    leave it unchanged.
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
#                           MINUTES PARSER                                      #
################################################################################

def parse_minutes(time_str: str) -> float:
    """
    Convert a 'MM:SS' string to a float representing total minutes.
    E.g., "27:21" -> 27.35. If invalid or empty, returns 0.0.
    """
    if not time_str or ':' not in str(time_str):
        return 0.0
    try:
        minutes_part, seconds_part = str(time_str).split(':')
        total_minutes = float(minutes_part) + float(seconds_part) / 60.0
        return round(total_minutes, 2)
    except ValueError:
        return 0.0

################################################################################
#                     UPSERT: HISTORICAL PLAYER STATS                          #
################################################################################

def upsert_historical_game_stats(game_id: int, player_stats: dict, game_date: str) -> dict:
    """
    Upserts historical player statistics into the Supabase table.
    
    Parameters:
      - game_id: The ID of the game.
      - player_stats: The statistics dictionary for a player.
      - game_date: The actual game date (as an ISO‑formatted string) from the API.
    
    Returns the result of the upsert operation.
    
    **Note:** This now writes to the renamed table: nba_historical_player_stats.
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
        'game_date': game_date,  # Use the actual game date passed in
        'updated_at': datetime.utcnow().isoformat()
    }
    
    result = (
        supabase.table('nba_historical_player_stats')
        .upsert(stats_data, on_conflict='game_id, player_id')
        .execute()
    )
    return result


###############################################################################
#                       LIVE PLAYER STATS UPSERT                               #
###############################################################################

def upsert_live_player_stats(game_id: int, player_stats: dict) -> dict:
    if not isinstance(player_stats, dict):
        return {"error": f"player_stats is not a dictionary. Got {type(player_stats)}."}

    # If your 'modified_stats' only has 'team_id', do this:
    team_id = player_stats.get('team_id', 0)
    team_name = "Unknown"
    if team_id > 0:
        team_name = get_team_name_from_api(team_id, league='12', season='2024-2025')

    raw_name = player_stats.get('player_name', 'Unknown')
    # Or if it's still `player_stats.get('player', {}).get('name')` – but keep consistent
    player_name_fixed = fix_player_name(raw_name)

    minutes_numeric = player_stats.get('minutes_numeric')
    if minutes_numeric is None:
        raw_minutes = player_stats.get('minutes', '0')
        minutes_numeric = parse_minutes(raw_minutes)

    stats_data = {
        'game_id': game_id,
        'player_id': player_stats.get('player_id', 0),
        'player_name': player_name_fixed,
        'team_id': team_id,
        'team_name': team_name,
        'minutes': minutes_numeric,
        'points': player_stats.get('points', 0),
        'rebounds': player_stats.get('rebounds', 0),
        'assists': player_stats.get('assists', 0),
        'steals': player_stats.get('steals', 0),
        'blocks': player_stats.get('blocks', 0),
        'turnovers': player_stats.get('turnovers', 0),
        'fouls': player_stats.get('fouls', 0),
        'fg_made': player_stats.get('fg_made', 0),
        'fg_attempted': player_stats.get('fg_attempted', 0),
        'three_made': player_stats.get('three_made', 0),
        'three_attempted': player_stats.get('three_attempted', 0),
        'ft_made': player_stats.get('ft_made', 0),
        'ft_attempted': player_stats.get('ft_attempted', 0),
        'game_date': str(datetime.now().date()),
        'updated_at': datetime.utcnow().isoformat()
    }

    print(f"\n[DEBUG] upsert_live_player_stats - about to upsert record:")
    print(stats_data)

    try:
        response = supabase.table('nba_live_player_stats')\
            .upsert(stats_data, on_conflict='game_id,player_id')\
            .execute()
        return response
    except Exception as e:
        print(f"[ERROR] upsert_live_player_stats: {e}")
        return {"error": str(e)}


################################################################################
#                     UPSERT: LIVE GAME STATS                                   #
################################################################################


def upsert_live_game_stats_team(record: dict) -> dict:
    """
    Upserts a single record of TEAM stats into the 'nba_live_game_stats' table.
    The `record` should contain columns like:
      - game_id, home_team, away_team, home_score, away_score,
      - home_q1, home_q2, home_q3, home_q4, home_ot,
      - away_q1, away_q2, away_q3, away_q4, away_ot,
      - home_assists, home_steals, etc.
    """
    response = (
        supabase.table('nba_live_game_stats')
        .upsert(record, on_conflict='game_id')  
        # or on_conflict='game_id,home_team' if you like
        .execute()
    )
    return response


################################################################################
#                     UPSERT: 2024-25 FINAL GAME STATS                          #
################################################################################

def upsert_nba_recent_player_stats(game_id: int, player_stats: dict) -> dict:
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
        supabase.table('nba_historical_player_stats')
        .upsert(stats_data, on_conflict='game_id, player_id')
        .execute()
    )
    return result