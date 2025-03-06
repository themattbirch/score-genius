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
        return "2019-2020"

###############################################################################
#            LOOKUP TEAM NAME, FALLBACK 'YYYY-YYYY+1' -> 'YYYY'               #
###############################################################################

def get_team_name_from_api(team_id: int, league='12', season='2024-2025') -> str:
    """
    If the direct (team_id, season) fails, try a single-year fallback.
    """
    cache_key = (team_id, season)
    if cache_key in TEAM_NAME_CACHE:
        return TEAM_NAME_CACHE[cache_key]

    name_result = _fetch_team_name(team_id, league, season)
    if name_result != "Unknown":
        TEAM_NAME_CACHE[cache_key] = name_result
        return name_result

    # If there's a dash, attempt fallback
    if "-" in season:
        just_year = season.split("-")[0]
        fallback_key = (team_id, just_year)
        if fallback_key in TEAM_NAME_CACHE:
            fallback_name = TEAM_NAME_CACHE[fallback_key]
        else:
            fallback_name = _fetch_team_name(team_id, league, just_year)
            TEAM_NAME_CACHE[fallback_key] = fallback_name
        TEAM_NAME_CACHE[cache_key] = fallback_name
        return fallback_name

    TEAM_NAME_CACHE[cache_key] = "Unknown"
    return "Unknown"

def _fetch_team_name(team_id: int, league: str, season: str) -> str:
    if TEAM_DEBUG_MODE:
        print(f"[DEBUG] _fetch_team_name: team_id={team_id}, league={league}, season={season}")
    url = f"{BASE_URL}/teams"
    params = {"id": team_id, "league": league, "season": season}
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        data = r.json()
        teams_list = data.get("response", [])
        if not teams_list:
            return "Unknown"
        return teams_list[0].get("name", "Unknown")
    except:
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

    team_id = player_stats.get('team_id', 0)
    team_name = "Unknown"
    if team_id > 0:
        team_name = get_team_name_from_api(team_id, '12', '2024-2025')

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
