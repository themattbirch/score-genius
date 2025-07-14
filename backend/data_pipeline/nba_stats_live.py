# File: backend/data_pipeline/nba_stats_live.py

import json
import requests
import sys
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pprint import pprint
import time
import traceback
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
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default

def normalize_team_name(name):
    """
    Normalize team name for consistent matching.
    """
    if not name:
        return ""
    
    # Common name variations and abbreviations
    name_map = {
        "sixers": "philadelphia 76ers",
        "76ers": "philadelphia 76ers",
        "blazers": "portland trail blazers",
        "trailblazers": "portland trail blazers",
        "trail blazers": "portland trail blazers",
        "cavs": "cleveland cavaliers",
        "mavs": "dallas mavericks",
        "knicks": "new york knicks", 
        "nets": "brooklyn nets",
        "lakers": "los angeles lakers",
        "clippers": "los angeles clippers",
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "warriors": "golden state warriors",
        "t-wolves": "minnesota timberwolves",
        "timberwolves": "minnesota timberwolves",
        "nuggets": "denver nuggets",
        "heat": "miami heat",
        "bulls": "chicago bulls",
        "celtics": "boston celtics",
        "bucks": "milwaukee bucks",
        "suns": "phoenix suns",
        "spurs": "san antonio spurs",
        "raptors": "toronto raptors",
        "wizards": "washington wizards",
        "magic": "orlando magic",
        "hawks": "atlanta hawks",
        "hornets": "charlotte hornets",
        "pistons": "detroit pistons",
        "pacers": "indiana pacers",
        "grizzlies": "memphis grizzlies",
        "pelicans": "new orleans pelicans",
        "thunder": "oklahoma city thunder",
        "jazz": "utah jazz",
        "kings": "sacramento kings",
        "rockets": "houston rockets"
    }
    
    name_lower = name.lower()
    for key, value in name_map.items():
        if key == name_lower:
            return value
    
    # If no exact match on alias, return cleaned original
    cleaned = ' '.join(name.split()).title()
    return cleaned

def get_official_schedule():
    """
    Get the official schedule from the nba_game_schedule table.
    Returns a dictionary mapping game keys (home_team-away_team) to game IDs.
    """
    try:
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
        
        response = supabase.table("nba_game_schedule").select("*").in_(
            "game_date", [yesterday, today, tomorrow]
        ).execute()
        
        if not response.data:
            update_nba_schedule()
            response = supabase.table("nba_game_schedule").select("*").in_(
                "game_date", [yesterday, today, tomorrow]
            ).execute()
        
        game_map = {}
        for game in response.data:
            home_team = normalize_team_name(game.get('home_team', '')).lower()
            away_team = normalize_team_name(game.get('away_team', '')).lower()
            game_id = game.get('game_id')
            if home_team and away_team and game_id:
                game_map[f"{home_team}-{away_team}"] = game_id
                game_map[f"{away_team}-{home_team}"] = game_id
        return game_map
    except Exception as e:
        print(f"Error getting official schedule: {e}")
        traceback.print_exc()
        return {}

def get_current_season():
    """
    Determine the current NBA season based on today's date.
    """
    today = datetime.now()
    if today.month >= 7 and today.month <= 9:
        return f"{today.year}-{today.year + 1}"
    elif today.month >= 10:
        return f"{today.year}-{today.year + 1}"
    else:
        return f"{today.year - 1}-{today.year}"

def update_nba_schedule():
    """
    Update the NBA schedule for the current date range.
    """
    season = get_current_season()
    league = '12'  # NBA league ID
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    for date_obj in [yesterday, today, tomorrow]:
        date_str = date_obj.isoformat()
        print(f"Updating schedule for {date_str}...")
        
        games_data = get_games_by_date(league, season, date_str)
        if not games_data.get('response'):
            print(f"No games found for {date_str}")
            continue
        
        print(f"Found {len(games_data.get('response', []))} games for {date_str}")
        
        for game in games_data.get('response', []):
            try:
                game_id = game.get('id')
                if not game_id:
                    continue
                
                home_team = get_nested_value(game, 'teams', 'home', 'name')
                away_team = get_nested_value(game, 'teams', 'away', 'name')
                venue_value = game.get('venue', None)
                if isinstance(venue_value, dict):
                    venue = venue_value.get('name', '')
                elif isinstance(venue_value, str):
                    venue = venue_value
                else:
                    venue = ''
                print(f"DEBUG - Extracted venue: {venue}")
                
                status = get_nested_value(game, 'status', 'short')
                game_date_str = game.get('date')
                try:
                    game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    pacific_tz = ZoneInfo("America/Los_Angeles")
                    if game_datetime.tzinfo is not None:
                        game_datetime = game_datetime.astimezone(pacific_tz)
                    else:
                        utc_tz = ZoneInfo("UTC")
                        game_datetime = game_datetime.replace(tzinfo=utc_tz).astimezone(pacific_tz)
                    
                    game_date = game_datetime.date().isoformat()
                    naive_pt_time = game_datetime.replace(tzinfo=None)
                    scheduled_time = naive_pt_time.isoformat() + "+00:00"
                    print(f"Converted game time to Pacific: {game_datetime}")
                    print(f"Storing as PT with fake UTC indicator: {scheduled_time}")
                except Exception as e:
                    print(f"Error parsing date '{game_date_str}': {e}")
                    game_date = date_str
                    scheduled_time = f"{date_str}T19:00:00+00:00"
                
                home_team = normalize_team_name(home_team)
                away_team = normalize_team_name(away_team)
                
                schedule_record = {
                    'game_id': game_id,
                    'game_date': game_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'scheduled_time': scheduled_time,
                    'venue': venue,
                    'status': status or 'scheduled'
                }
                
                supabase.table("nba_game_schedule").upsert(
                    schedule_record, on_conflict='game_id'
                ).execute()
                print(f"Updated schedule: {home_team} vs {away_team}")
                
            except Exception as e:
                print(f"Error processing game {game.get('id')}: {e}")
                continue
    return True

def scheduled_update_nba_schedule():
    """
    Standalone function to update the NBA schedule.
    """
    try:
        print(f"[{datetime.now()}] Running scheduled NBA schedule update...")
        result = update_nba_schedule()
        print(f"[{datetime.now()}] Schedule update completed. Result: {result}")
        return result
    except Exception as e:
        print(f"[{datetime.now()}] Error in scheduled NBA schedule update: {e}")
        traceback.print_exc()
        return False

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
        data = response.json()
        
        if 'response' in data and data['response']:
            for game in data['response']:
                print(f"DEBUG - Game ID {game.get('id')} Venue Data:", json.dumps(game.get('venue', {}), indent=2))
                game_date = game.get('date')
                print(f"DEBUG - Game ID {game.get('id')} Date: {game_date} (Requested timezone: {timezone})")
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_team_stats(game_id: int, league: str = '12', season: str = None) -> dict:
    """
    Fetch team-level statistics for a specific game.
    """
    if season is None:
        season = get_current_season()
    
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    
    try:
        print(f"DEBUG - Requesting team stats with params: {params}")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"DEBUG - Team Stats API Response Status: {response.status_code}")
        if 'response' in data and data['response']:
            print(f"DEBUG - Found {len(data['response'])} team stats entries")
        else:
            print("DEBUG - Empty or missing response array in team stats")
        
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
    print(f"  Date: {game.get('date')}")
    print("-" * 60)

###############################################################################
# 2) Transformation Functions
###############################################################################

def determine_current_quarter(game):
    """
    Determine the current quarter based on game status and scores.
    """
    status = game.get('status', {}).get('short', '')
    if status == 'NS':
        return 0
    elif status == 'FT':
        return 4
    
    status_map = {
        'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4,
        'OT': 4, 'BT': 1, 'HT': 2
    }
    if status in status_map:
        return status_map[status]
    
    home_scores = game.get('scores', {}).get('home', {})
    for q in ['q4', 'quarter4', 'quarter_4']:
        if home_scores.get(q):
            return 4
    for q in ['q3', 'quarter3', 'quarter_3']:
        if home_scores.get(q):
            return 3
    for q in ['q2', 'quarter2', 'quarter_2']:
        if home_scores.get(q):
            return 2
    for q in ['q1', 'quarter1', 'quarter_1']:
        if home_scores.get(q):
            return 1
    return 0

def transform_team_stats(game: dict, team_stats_data: dict, official_schedule: dict = None) -> dict:
    """
    Transform raw game and team stats data into the expected record format.
    """
    def get_quarter_score(scores: dict, key1: str, key2: str, key3: str = None) -> int:
        for key in [key1, key2, key3, f"q_{key1[-1]}", f"quarter{key1[-1]}", 
                    f"quarter_{key1[-1]}", f"Q{key1[-1]}"]:
            if key and key in scores and scores[key] is not None:
                try:
                    return int(scores[key])
                except:
                    continue
        for k, value in scores.items():
            if k and value is not None and key1[-1] in k:
                try:
                    return int(value)
                except:
                    continue
        return 0

    home_scores = game.get('scores', {}).get('home', {}) or {}
    away_scores = game.get('scores', {}).get('away', {}) or {}

    print(f"DEBUG - Home scores keys available: {list(home_scores.keys())}")
    print(f"DEBUG - Away scores keys available: {list(away_scores.keys())}")
    print("DEBUG - Raw home_scores structure:", json.dumps(home_scores, indent=2))

    home_team_id = game.get('teams', {}).get('home', {}).get('id')
    away_team_id = game.get('teams', {}).get('away', {}).get('id')
    
    home_stats = None
    away_stats = None
    for stat in team_stats_data.get('response', []):
        tid = stat.get('team', {}).get('id')
        if tid == home_team_id:
            home_stats = stat
        elif tid == away_team_id:
            away_stats = stat
    
    raw_home_team = game.get('teams', {}).get('home', {}).get('name')
    raw_away_team = game.get('teams', {}).get('away', {}).get('name')
    normalized_home = normalize_team_name(raw_home_team)
    normalized_away = normalize_team_name(raw_away_team)
    
    game_id = game.get('id')
    if official_schedule:
        key1 = f"{normalized_home.lower()}-{normalized_away.lower()}"
        key2 = f"{normalized_away.lower()}-{normalized_home.lower()}"
        if key1 in official_schedule:
            game_id = official_schedule[key1]
        elif key2 in official_schedule:
            game_id = official_schedule[key2]
    
    game_date_str = game.get('date')
    try:
        game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        pacific_tz = ZoneInfo("America/Los_Angeles")
        if game_datetime.tzinfo is not None and str(game_datetime.tzinfo) != str(pacific_tz):
            game_datetime = game_datetime.astimezone(pacific_tz)
        elif game_datetime.tzinfo is None:
            utc_tz = ZoneInfo("UTC")
            game_datetime = game_datetime.replace(tzinfo=utc_tz).astimezone(pacific_tz)
        naive_pt_time = game_datetime.replace(tzinfo=None)
        formatted_game_date = naive_pt_time.isoformat() + "+00:00"
    except:
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        naive_pt_time = now_pt.replace(tzinfo=None)
        formatted_game_date = naive_pt_time.isoformat() + "+00:00"

    game_status = game.get('status', {}).get('short', '')
    
    transformed = {
        'game_id': game_id,
        'home_team': normalized_home,
        'away_team': normalized_away,
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
        'game_date': formatted_game_date,
        'current_quarter': determine_current_quarter(game),
        'status': game_status  # <-- store short status in Supabase
    }

    if home_stats:
        rebounds = home_stats.get('rebounds', {})
        threepoint = home_stats.get('threepoint_goals', {}) or home_stats.get('threePoint', {}) or {}
        # Add field goals and free throws for home team
        field_goals = home_stats.get('field_goals', {}) or home_stats.get('fieldGoals', {}) or {}
        free_throws = home_stats.get('freethrows_goals', {}) or home_stats.get('freeThrows', {}) or {}
        
        transformed.update({
            'home_assists': home_stats.get('assists', 0),
            'home_steals': home_stats.get('steals', 0),
            'home_blocks': home_stats.get('blocks', 0),
            'home_turnovers': home_stats.get('turnovers', 0),
            'home_fouls': home_stats.get('personal_fouls', 0) or home_stats.get('fouls', 0),
            'home_off_reb': rebounds.get('offence', 0) or rebounds.get('offensive', 0),
            'home_def_reb': rebounds.get('defense', 0) or rebounds.get('defensive', 0),
            'home_total_reb': rebounds.get('total', 0),
            'home_3pm': threepoint.get('total', 0) or threepoint.get('made', 0),
            'home_3pa': threepoint.get('attempts', 0) or threepoint.get('attempted', 0),
            # Field Goals - NEW
            'home_fg_made': field_goals.get('total', 0) or field_goals.get('made', 0),
            'home_fg_attempted': field_goals.get('attempts', 0) or field_goals.get('attempted', 0),
            # Free Throws - NEW
            'home_ft_made': free_throws.get('total', 0) or free_throws.get('made', 0),
            'home_ft_attempted': free_throws.get('attempts', 0) or free_throws.get('attempted', 0),
        })
    if away_stats:
        rebounds = away_stats.get('rebounds', {})
        threepoint = away_stats.get('threepoint_goals', {}) or away_stats.get('threePoint', {}) or {}
        # Add field goals and free throws for away team
        field_goals = away_stats.get('field_goals', {}) or away_stats.get('fieldGoals', {}) or {}
        free_throws = away_stats.get('freethrows_goals', {}) or away_stats.get('freeThrows', {}) or {}
        
        transformed.update({
            'away_assists': away_stats.get('assists', 0),
            'away_steals': away_stats.get('steals', 0),
            'away_blocks': away_stats.get('blocks', 0),
            'away_turnovers': away_stats.get('turnovers', 0),
            'away_fouls': away_stats.get('personal_fouls', 0) or away_stats.get('fouls', 0),
            'away_off_reb': rebounds.get('offence', 0) or rebounds.get('offensive', 0),
            'away_def_reb': rebounds.get('defense', 0) or rebounds.get('defensive', 0),
            'away_total_reb': rebounds.get('total', 0),
            'away_3pm': threepoint.get('total', 0) or threepoint.get('made', 0),
            'away_3pa': threepoint.get('attempts', 0) or threepoint.get('attempted', 0),
            # Field Goals - NEW
            'away_fg_made': field_goals.get('total', 0) or field_goals.get('made', 0),
            'away_fg_attempted': field_goals.get('attempts', 0) or field_goals.get('attempted', 0),
            # Free Throws - NEW
            'away_ft_made': free_throws.get('total', 0) or free_throws.get('made', 0),
            'away_ft_attempted': free_throws.get('attempts', 0) or free_throws.get('attempted', 0),
        })

    print("DEBUG - Extracted team stats:")
    pprint(transformed)
    return transformed

def create_base_game_record(game: dict, official_schedule: dict = None) -> dict:
    """
    Create a basic game record from just the game object, without team stats.
    """
    def get_quarter_score(scores: dict, key1: str, key2: str, key3: str = None) -> int:
        for key in [key1, key2, key3, f"q_{key1[-1]}", f"quarter{key1[-1]}", 
                    f"quarter_{key1[-1]}", f"Q{key1[-1]}"]:
            if key and key in scores and scores[key] is not None:
                try:
                    return int(scores[key])
                except:
                    continue
        for k, value in scores.items():
            if k and value is not None and key1[-1] in k:
                try:
                    return int(value)
                except:
                    continue
        return 0
    
    home_scores = game.get('scores', {}).get('home', {}) or {}
    away_scores = game.get('scores', {}).get('away', {}) or {}

    raw_home_team = get_nested_value(game, 'teams', 'home', 'name', default="Unknown")
    raw_away_team = get_nested_value(game, 'teams', 'away', 'name', default="Unknown")
    normalized_home = normalize_team_name(raw_home_team)
    normalized_away = normalize_team_name(raw_away_team)
    
    game_id = game.get('id')
    if official_schedule:
        key1 = f"{normalized_home.lower()}-{normalized_away.lower()}"
        key2 = f"{normalized_away.lower()}-{normalized_home.lower()}"
        if key1 in official_schedule:
            game_id = official_schedule[key1]
        elif key2 in official_schedule:
            game_id = official_schedule[key2]
    
    game_date_str = game.get('date')
    try:
        game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        pacific_tz = ZoneInfo("America/Los_Angeles")
        if game_datetime.tzinfo is not None:
            game_datetime = game_datetime.astimezone(pacific_tz)
        else:
            utc_tz = ZoneInfo("UTC")
            game_datetime = game_datetime.replace(tzinfo=utc_tz).astimezone(pacific_tz)
        naive_pt_time = game_datetime.replace(tzinfo=None)
        formatted_game_date = naive_pt_time.isoformat() + "+00:00"
    except:
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        naive_pt_time = now_pt.replace(tzinfo=None)
        formatted_game_date = naive_pt_time.isoformat() + "+00:00"

    game_status = game.get('status', {}).get('short', '')
    
    record = {
        'game_id': game_id,
        'home_team': normalized_home,
        'away_team': normalized_away,
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
        'game_date': formatted_game_date,
        'current_quarter': determine_current_quarter(game),
        'status': game_status,  # <-- store short status
        
        # Default values for field goal and free throw stats
        'home_fg_made': 0,
        'home_fg_attempted': 0,
        'away_fg_made': 0,
        'away_fg_attempted': 0,
        'home_ft_made': 0,
        'home_ft_attempted': 0,
        'away_ft_made': 0,
        'away_ft_attempted': 0
    }
    return record

###############################################################################
# Team Season Stats Functions
###############################################################################

def get_teams_by_league_season(league: str, season: str) -> list:
    """
    Fetches a list of teams for the given league and season.
    Returns the "response" list from the JSON.
    """
    url = f"{BASE_URL}/teams"
    params = {"league": league, "season": season}
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        print(f"Fetched {len(data.get('response', []))} teams for league {league}, season {season}")
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching teams for league {league}, season {season}: {e}")
        return []

def get_team_season_stats(team_id, league_id, season):
    """
    Fetch team season statistics from the /statistics endpoint.
    """
    url = f"{BASE_URL}/statistics"
    params = {
        "team": team_id,
        "league": league_id,
        "season": season
    }
    
    try:
        print(f"DEBUG - Requesting team season stats for team_id={team_id}, season={season}")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"DEBUG - Team Season Stats API Response Status: {response.status_code}")
        if 'response' in data and data['response']:
            print(f"DEBUG - Successfully fetched season stats for team_id={team_id}")
            return data.get('response', {})
        else:
            print(f"DEBUG - Empty or missing response for team_id={team_id} season stats")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team season statistics for team_id={team_id}: {e}")
        return {}

def transform_team_season_stats(team_id, team_name, stats, league_id, season):
    """
    Transform raw team season stats data into the expected record format.
    """
    # Extract games statistics
    games = stats.get("games", {})
    games_played_home = get_nested_value(games, "played", "home", default=0)
    games_played_away = get_nested_value(games, "played", "away", default=0)
    games_played_all = get_nested_value(games, "played", "all", default=0)
    
    # Extract wins statistics
    wins_home_total = get_nested_value(games, "wins", "home", "total", default=0)
    wins_home_percentage = get_nested_value(games, "wins", "home", "percentage", default=0)
    wins_away_total = get_nested_value(games, "wins", "away", "total", default=0)
    wins_away_percentage = get_nested_value(games, "wins", "away", "percentage", default=0)
    wins_all_total = get_nested_value(games, "wins", "all", "total", default=0)
    wins_all_percentage = get_nested_value(games, "wins", "all", "percentage", default=0)
    
    # Extract losses statistics - API might use either "loses" or "losses" as the key
    losses_home_total = get_nested_value(games, "loses", "home", "total", default=0) or get_nested_value(games, "losses", "home", "total", default=0)
    losses_home_percentage = get_nested_value(games, "loses", "home", "percentage", default=0) or get_nested_value(games, "losses", "home", "percentage", default=0)
    losses_away_total = get_nested_value(games, "loses", "away", "total", default=0) or get_nested_value(games, "losses", "away", "total", default=0)
    losses_away_percentage = get_nested_value(games, "loses", "away", "percentage", default=0) or get_nested_value(games, "losses", "away", "percentage", default=0)
    losses_all_total = get_nested_value(games, "loses", "all", "total", default=0) or get_nested_value(games, "losses", "all", "total", default=0)
    losses_all_percentage = get_nested_value(games, "loses", "all", "percentage", default=0) or get_nested_value(games, "losses", "all", "percentage", default=0)
    
    # Extract points for statistics
    points = stats.get("points", {})
    points_for_total_home = get_nested_value(points, "for", "total", "home", default=0)
    points_for_total_away = get_nested_value(points, "for", "total", "away", default=0)
    points_for_total_all = get_nested_value(points, "for", "total", "all", default=0)
    points_for_avg_home = get_nested_value(points, "for", "average", "home", default=0)
    points_for_avg_away = get_nested_value(points, "for", "average", "away", default=0)
    points_for_avg_all = get_nested_value(points, "for", "average", "all", default=0)
    
    # Extract points against statistics
    points_against_total_home = get_nested_value(points, "against", "total", "home", default=0)
    points_against_total_away = get_nested_value(points, "against", "total", "away", default=0)
    points_against_total_all = get_nested_value(points, "against", "total", "all", default=0)
    points_against_avg_home = get_nested_value(points, "against", "average", "home", default=0)
    points_against_avg_away = get_nested_value(points, "against", "average", "away", default=0)
    points_against_avg_all = get_nested_value(points, "against", "average", "all", default=0)
    
    # Extract form/streak if available
    form = get_nested_value(stats, "form", default="")
    streak = 0
    if form:
        # Count consecutive wins or losses at the end of the form string
        # Example: "WWLWW" would give a streak of 2 (wins)
        streak_char = form[-1] if form else ""
        for i in range(len(form)-1, -1, -1):
            if form[i] != streak_char:
                break
            streak += 1
        # Make streak negative for losses
        if streak_char.upper() == "L":
            streak = -streak
    
    record = {
        "team_id": team_id,
        "team_name": team_name,
        "season": season,
        "league_id": league_id,
        
        # Games played statistics
        "games_played_home": games_played_home,
        "games_played_away": games_played_away,
        "games_played_all": games_played_all,
        
        # Wins statistics
        "wins_home_total": wins_home_total,
        "wins_home_percentage": wins_home_percentage,
        "wins_away_total": wins_away_total,
        "wins_away_percentage": wins_away_percentage,
        "wins_all_total": wins_all_total,
        "wins_all_percentage": wins_all_percentage,
        
        # Losses statistics
        "losses_home_total": losses_home_total,
        "losses_home_percentage": losses_home_percentage,
        "losses_away_total": losses_away_total,
        "losses_away_percentage": losses_away_percentage,
        "losses_all_total": losses_all_total,
        "losses_all_percentage": losses_all_percentage,
        
        # Points for statistics
        "points_for_total_home": points_for_total_home,
        "points_for_total_away": points_for_total_away,
        "points_for_total_all": points_for_total_all,
        "points_for_avg_home": points_for_avg_home,
        "points_for_avg_away": points_for_avg_away,
        "points_for_avg_all": points_for_avg_all,
        
        # Points against statistics
        "points_against_total_home": points_against_total_home,
        "points_against_total_away": points_against_total_away,
        "points_against_total_all": points_against_total_all,
        "points_against_avg_home": points_against_avg_home,
        "points_against_avg_away": points_against_avg_away,
        "points_against_avg_all": points_against_avg_all,
        
        # Live specific fields
        "last_fetched_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "updated_at": datetime.now(ZoneInfo("UTC")).isoformat()
    }
    
    return record

def upsert_live_team_season_stats(record):
    """
    Upsert team season statistics to the nba_live_team_stats table.
    """
    try:
        result = supabase.table("nba_live_team_stats").upsert(
            record, on_conflict="team_id,season,league_id"
        ).execute()
        return result
    except Exception as e:
        print(f"Error upserting team season stats: {e}")
        traceback.print_exc()
        return None

def fetch_and_store_team_season_stats():
    """
    Fetch and store team season statistics for all NBA teams.
    """
    league_id = "12"  # NBA league ID
    season = get_current_season()
    
    print(f"\n=== Fetching team season stats for NBA (league_id={league_id}) season {season} ===")
    
    # First, get all teams for this league and season
    teams_list = get_teams_by_league_season(league_id, season)
    if not teams_list:
        print("No teams found.")
        return 0
    
    # Filter out All-Star and special event teams
    regular_teams = []
    for team in teams_list:
        team_name = team.get("name", "")
        if "all-star" not in team_name.lower() and "team " not in team_name.lower():
            if not any(x in team_name.lower() for x in ["rising stars", "global stars", "young stars", "ogs"]):
                regular_teams.append(team)
            else:
                print(f"Filtering out special event team: {team_name}")
        else:
            print(f"Filtering out All-Star team: {team_name}")
    
    print(f"Found {len(regular_teams)} regular teams after filtering.")
    
    processed_count = 0
    for team in regular_teams:
        team_id = team.get("id")
        team_name = team.get("name")
        print(f"Processing season stats for team: {team_name} (ID: {team_id})")
        
        # Fetch team season statistics
        team_stats = get_team_season_stats(team_id, league_id, season)
        if not team_stats:
            print(f"No statistics found for team_id={team_id}.")
            continue
        
        # Transform the data to our required format
        record = transform_team_season_stats(team_id, team_name, team_stats, league_id, season)
        
        print(f"[INFO] Upserting team season stats => Team ID: {record['team_id']}, Team Name: {record['team_name']}")
        try:
            res = upsert_live_team_season_stats(record)
            print(f"[INFO] Upsert result: {res}")
        except Exception as e:
            print(f"[ERROR] Could not upsert team_id={team_id}: {e}")
        
        processed_count += 1
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    print(f"Processed season stats for {processed_count} teams")
    return processed_count

###############################################################################
# 3) Main Driver: Fetch Live Games, Transform, and Upsert
###############################################################################

def run_live_games():
    pacific_tz = ZoneInfo("America/Los_Angeles")
    today_pt = datetime.now(pacific_tz)
    today_pt_str = today_pt.strftime('%Y-%m-%d')
    
    print(f"Fetching games for {today_pt_str} in Pacific Time...")
    
    league = '12'
    season = get_current_season()
    timezone = 'America/Los_Angeles'
    
    official_schedule = get_official_schedule()
    if not official_schedule:
        print("Warning: Could not load official schedule. Using API game IDs only.")
    
    games_data = get_games_by_date(league, season, today_pt_str, timezone)
    if not games_data.get('response'):
        print(f"No game data found for {today_pt_str}.")
        return

    for game in games_data['response']:
        raw_date = game.get('date')
        print(f"API returned game date: {raw_date}")
        print_game_info(game)
        game_id = game.get('id')
        
        game_status = game.get('status', {}).get('short', '')
        team_stats_data = get_team_stats(game_id)
        
        home_scores_raw = game.get('scores', {}).get('home', {})
        away_scores_raw = game.get('scores', {}).get('away', {})
        print(f"DEBUG - RAW HOME SCORES FROM GAME API: {json.dumps(home_scores_raw, indent=2)}")
        print(f"DEBUG - RAW AWAY SCORES FROM GAME API: {json.dumps(away_scores_raw, indent=2)}")
        
        has_stats = team_stats_data.get('response') and len(team_stats_data.get('response', [])) > 0
        
        if game_status == 'NS':
            print(f"Game ID {game_id} not started. Creating basic record.")
            record = create_base_game_record(game, official_schedule)
        elif not has_stats:
            print(f"No team stats found for game ID {game_id}. Creating basic record.")
            record = create_base_game_record(game, official_schedule)
        else:
            print(f"Found team stats for game ID {game_id}. Creating full record.")
            record = transform_team_stats(game, team_stats_data, official_schedule)
        
        print("Upserting record for game ID:", game_id)
        pprint(record)
        try:
            result = upsert_live_game_stats_team(record)
            print(f"Upsert result for game ID {game_id}: {result}")
        except Exception as e:
            print(f"Error upserting data for game ID {game_id}: {e}")
            traceback.print_exc()
        
        print("=" * 60)

def main():
    print("Fetching live NBA game data for today (using Pacific Time) and upserting team stats...")
    try:
        update_nba_schedule()
        run_live_games()
        fetch_and_store_team_season_stats()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()


###############################################################################
# 4) Optional Helper: Fetch ONLY Live Games (Based on Status)
###############################################################################

import pandas as pd
import pytz

def fetch_live_games_in_pacific_time():
    """
    Pull from 'nba_live_game_stats' and return only today's in-progress games,
    using status in [Q1, Q2, Q3, Q4, OT, BT, HT].
    """
    live_codes = {"Q1", "Q2", "Q3", "Q4", "OT", "BT", "HT"}
    pacific_tz = pytz.timezone("America/Los_Angeles")
    now_pt = datetime.now(pacific_tz)
    today_pt = now_pt.date()
    
    response = supabase.table("nba_live_game_stats").select("*").execute()
    if not response.data:
        print("No data found in 'nba_live_game_stats' table.")
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    
    # Ensure we have game_date
    if "game_date" not in df.columns:
        print("Column 'game_date' not found!")
        return pd.DataFrame()
    
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    df["game_date_pt"] = df["game_date"].dt.tz_convert(pacific_tz)
    df["date_only_pt"] = df["game_date_pt"].dt.date
    
    # Filter for today's games
    df_today = df[df["date_only_pt"] == today_pt].copy()
    if df_today.empty:
        print("No games match today's date in Pacific Time.")
        return pd.DataFrame()
    
    # Convert status to uppercase to match known codes
    df_today["status"] = df_today["status"].astype(str).str.upper()
    active_games = df_today[df_today["status"].isin(live_codes)].copy()
    
    print(f"ACTIVELY LIVE GAMES: {len(active_games)}")
    if not active_games.empty:
        for _, row in active_games.iterrows():
            print(
                f"  - {row.get('home_team','Unknown')} vs {row.get('away_team','Unknown')}: "
                f"{row.get('home_score','N/A')}-{row.get('away_score','N/A')}, "
                f"Status={row.get('status')}"
            )
    else:
        print("  No active games found!")
    
    return active_games