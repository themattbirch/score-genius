# File: caching/supabase_stats.py

import requests
import json
from datetime import datetime
from caching.supabase_client import supabase
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

TEAM_NAME_CACHE = {}
TEAM_DEBUG_MODE = True

# Add a hardcoded team ID to name mapping as fallback
TEAM_ID_TO_NAME = {
    134: "Memphis Grizzlies",  # Based on the screenshot, this is likely Memphis
    140: "Detroit Pistons",
    161: "Washington Wizards",
    # Add more teams as needed
}

###############################################################################
#                    HELPER: FLIP 'Last First' TO 'First Last'                #
###############################################################################

def fix_player_name(name: str) -> str:
    parts = name.split()
    if len(parts) == 2:
        last, first = parts
        # If last ends with '.', treat as an initial, skip flipping
        if last.endswith('.'):
            return name
        return f"{first} {last}"
    return name

###############################################################################
#                 DETERMINE SEASON FROM 'YYYY-MM-DD' DATE                     #
###############################################################################

def parse_season_from_date(game_date: str) -> str:
    from datetime import datetime
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        if dt.month >= 10:
            return f"{dt.year}-{dt.year + 1}"
        else:
            return f"{dt.year - 1}-{dt.year}"
    except:
        return "2024-2025"  # Default to current season

###############################################################################
#            LOOKUP TEAM NAME, FALLBACK 'YYYY-YYYY+1' -> 'YYYY'               #
###############################################################################

def get_team_name_from_api(team_id: int, league='12', season='2024-2025') -> str:
    """
    Enhanced team name resolution with multiple fallbacks:
    1. Check cache
    2. Try current season
    3. Try fallback to year-only season
    4. Try previous season
    5. Use hardcoded mapping
    """
    cache_key = (team_id, season)
    if cache_key in TEAM_NAME_CACHE:
        return TEAM_NAME_CACHE[cache_key]

    # Try current season format (e.g., "2024-2025")
    name_result = _fetch_team_name(team_id, league, season)
    if name_result != "Unknown":
        TEAM_NAME_CACHE[cache_key] = name_result
        return name_result

    # Try fallback to year-only format if season has a dash
    if "-" in season:
        just_year = season.split("-")[0]
        fallback_key = (team_id, just_year)
        if fallback_key in TEAM_NAME_CACHE:
            fallback_name = TEAM_NAME_CACHE[fallback_key]
        else:
            fallback_name = _fetch_team_name(team_id, league, just_year)
            TEAM_NAME_CACHE[fallback_key] = fallback_name
        
        if fallback_name != "Unknown":
            TEAM_NAME_CACHE[cache_key] = fallback_name
            return fallback_name
        
        # Try previous season as another fallback
        prev_year = int(just_year) - 1
        prev_season = f"{prev_year}-{just_year}"
        prev_season_key = (team_id, prev_season)
        
        if prev_season_key in TEAM_NAME_CACHE:
            prev_name = TEAM_NAME_CACHE[prev_season_key]
        else:
            prev_name = _fetch_team_name(team_id, league, prev_season)
            TEAM_NAME_CACHE[prev_season_key] = prev_name
        
        if prev_name != "Unknown":
            TEAM_NAME_CACHE[cache_key] = prev_name
            return prev_name
    
    # Last resort: check hardcoded mapping
    if team_id in TEAM_ID_TO_NAME:
        team_name = TEAM_ID_TO_NAME[team_id]
        TEAM_NAME_CACHE[cache_key] = team_name
        if TEAM_DEBUG_MODE:
            print(f"[DEBUG] Using hardcoded team name '{team_name}' for team_id={team_id}")
        return team_name

    # If all else fails, log the issue and return Unknown
    if TEAM_DEBUG_MODE:
        print(f"[WARNING] Failed to resolve team name for team_id={team_id} after all fallbacks")
    
    TEAM_NAME_CACHE[cache_key] = "Unknown"
    return "Unknown"

def _fetch_team_name(team_id: int, league: str, season: str) -> str:
    """
    Improved fetch function with better error handling and logging
    """
    if TEAM_DEBUG_MODE:
        print(f"[DEBUG] _fetch_team_name: team_id={team_id}, league={league}, season={season}")
    
    if not team_id or team_id <= 0:
        return "Unknown"
        
    url = f"{BASE_URL}/teams"
    params = {"id": team_id, "league": league, "season": season}
    
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        data = r.json()
        teams_list = data.get("response", [])
        
        if not teams_list:
            if TEAM_DEBUG_MODE:
                print(f"[DEBUG] No teams found for team_id={team_id}, league={league}, season={season}")
            return "Unknown"
            
        team_name = teams_list[0].get("name", "Unknown")
        if TEAM_DEBUG_MODE:
            print(f"[DEBUG] Found team name '{team_name}' for team_id={team_id}, season={season}")
        return team_name
        
    except requests.exceptions.RequestException as e:
        if TEAM_DEBUG_MODE:
            print(f"[ERROR] API request failed for team_id={team_id}: {str(e)}")
        return "Unknown"
    except Exception as e:
        if TEAM_DEBUG_MODE:
            print(f"[ERROR] Unexpected error for team_id={team_id}: {str(e)}")
        return "Unknown"

################################################################################
#                           HISTORICAL PLAYER STATS                             #
################################################################################

def upsert_historical_game_stats(game_id: int, player_stats: dict, game_date: str) -> dict:
    season_str = parse_season_from_date(game_date)
    team_data = player_stats.get('team', {})
    tid = team_data.get('id', 0)
    team_name = "Unknown"
    if tid > 0:
        team_name = get_team_name_from_api(tid, '12', season_str)

    raw_name = player_stats.get('player', {}).get('name', "Unknown")
    player_name_fixed = fix_player_name(raw_name)

    # Convert "MM:SS" -> float
    raw_minutes = player_stats.get('minutes', "0:00")
    minutes_val = 0.0
    try:
        mins, secs = raw_minutes.split(":")
        minutes_val = round(float(mins) + float(secs)/60.0, 2)
    except:
        pass

    stats_data = {
        "game_id": game_id,
        "player_id": player_stats.get('player', {}).get('id', 0),
        "player_name": player_name_fixed,
        "team_id": tid,
        "team_name": team_name,
        "minutes": minutes_val,
        "points": player_stats.get("points", 0),
        "rebounds": player_stats.get("rebounds", {}).get("total", 0),
        "assists": player_stats.get("assists", 0),
        "steals": player_stats.get("steals", 0),
        "blocks": player_stats.get("blocks", 0),
        "turnovers": player_stats.get("turnovers", 0),
        "fouls": player_stats.get("fouls", 0),
        "fg_made": player_stats.get("field_goals", {}).get("total", 0),
        "fg_attempted": player_stats.get("field_goals", {}).get("attempts", 0),
        "three_made": player_stats.get("threepoint_goals", {}).get("total", 0),
        "three_attempted": player_stats.get("threepoint_goals", {}).get("attempts", 0),
        "ft_made": player_stats.get("freethrows_goals", {}).get("total", 0),
        "ft_attempted": player_stats.get("freethrows_goals", {}).get("attempts", 0),
        "game_date": game_date,
        "updated_at": datetime.utcnow().isoformat()
    }

    try:
        result = (
            supabase.table("nba_historical_player_stats")
            .upsert(stats_data, on_conflict="game_id, player_id")
            .execute()
        )
        return result
    except Exception as e:
        return {"error": str(e)}

################################################################################
#                       LIVE PLAYER STATS (REFERENCE)                          #
################################################################################

def upsert_live_player_stats(game_id: int, player_stats: dict) -> dict:
    if not isinstance(player_stats, dict):
        return {"error": f"player_stats is not a dictionary: {type(player_stats)}"}

    # Get current season dynamically instead of hardcoding
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_season = parse_season_from_date(current_date)
    
    team_id = player_stats.get('team_id', 0)
    team_name = "Unknown"
    if team_id > 0:
        # Use dynamic season determination
        team_name = get_team_name_from_api(team_id, '12', current_season)

    raw_name = player_stats.get('player_name', 'Unknown')
    fixed_name = fix_player_name(raw_name)

    # Convert minutes
    raw_minutes = player_stats.get('minutes_numeric')
    if raw_minutes is not None:
        minutes_float = round(raw_minutes, 2)
    else:
        minutes_str = player_stats.get('minutes', "0:00")
        try:
            mm, ss = minutes_str.split(":")
            minutes_float = round(float(mm) + float(ss)/60.0, 2)
        except:
            minutes_float = 0.0

    stats_data = {
        "game_id": game_id,
        "player_id": player_stats.get('player_id', 0),
        "player_name": fixed_name,
        "team_id": team_id,
        "team_name": team_name,
        "minutes": minutes_float,
        "points": player_stats.get("points", 0),
        "rebounds": player_stats.get("rebounds", 0),
        "assists": player_stats.get("assists", 0),
        "steals": player_stats.get("steals", 0),
        "blocks": player_stats.get("blocks", 0),
        "turnovers": player_stats.get("turnovers", 0),
        "fouls": player_stats.get("fouls", 0),
        "fg_made": player_stats.get("fg_made", 0),
        "fg_attempted": player_stats.get("fg_attempted", 0),
        "three_made": player_stats.get("three_made", 0),
        "three_attempted": player_stats.get("three_attempted", 0),
        "ft_made": player_stats.get("ft_made", 0),
        "ft_attempted": player_stats.get("ft_attempted", 0),
        "game_date": str(datetime.now().date()),
        "updated_at": datetime.utcnow().isoformat()
    }

    try:
        response = (
            supabase.table("nba_live_player_stats")
            .upsert(stats_data, on_conflict="game_id,player_id")
            .execute()
        )
        return response
    except Exception as e:
        return {"error": str(e)}

################################################################################
#                   LIVE GAME STATS (TEAM-LEVEL)                               #
################################################################################

def upsert_live_game_stats_team(record: dict) -> dict:
    """
    Upserts a single record of TEAM-level stats into 'nba_live_game_stats' table.
    The `record` should have fields like game_id, home_team, away_team, home_score, away_score, etc.
    """
    try:
        response = (
            supabase.table('nba_live_game_stats')
            .upsert(record, on_conflict='game_id')
            .execute()
        )
        return response
    except Exception as e:
        return {"error": str(e)}

################################################################################
#                  HISTORICAL GAME STATS (TEAM-LEVEL) - RESTORED               #
################################################################################

def upsert_historical_game_stats_team(record: dict) -> dict:
    """
    Upserts a single record of TEAM-level stats into 'nba_historical_game_stats' table.
    The `record` should have fields such as:
      - game_id, home_team, away_team,
      - home_score, away_score,
      - home_q1, home_q2, home_q3, home_q4, home_ot,
      - away_q1, away_q2, away_q3, away_q4, away_ot,
      - home_assists, home_steals, home_blocks, ...
      - away_assists, away_steals, away_blocks, ...
      - plus any other relevant fields you'd like to store
    """
    try:
        response = (
            supabase.table('nba_historical_game_stats')
            .upsert(record, on_conflict='game_id')
            .execute()
        )
        return response
    except Exception as e:
        return {"error": str(e)}