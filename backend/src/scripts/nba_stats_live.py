# File: backend/src/scripts/nba_stats_live.py

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

def normalize_team_name(name):
    """
    Normalize team name for consistent matching.
    
    Args:
        name: Team name to normalize
        
    Returns:
        Normalized team name
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
    
    # Convert to lowercase for case-insensitive matching
    name_lower = name.lower()
    
    # Direct mapping for known variations
    for key, value in name_map.items():
        if key == name_lower:
            return value
    
    # If no exact match on alias, return cleaned original
    # Remove extra spaces and standardize to title case
    cleaned = ' '.join(name.split()).title()
    return cleaned

def get_official_schedule():
    """
    Get the official schedule from the nba_game_schedule table.
    Returns a dictionary mapping game keys (home_team-away_team) to game IDs.
    """
    try:
        # Get games from recent dates
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
        
        # Try to query the table
        response = supabase.table("nba_game_schedule").select("*").in_("game_date", [yesterday, today, tomorrow]).execute()
        
        if not response.data:
            # No data found, try updating the schedule
            update_nba_schedule()
            # Try querying again
            response = supabase.table("nba_game_schedule").select("*").in_("game_date", [yesterday, today, tomorrow]).execute()
        
        # Build game map for easy lookup
        game_map = {}
        for game in response.data:
            home_team = normalize_team_name(game.get('home_team', '')).lower()
            away_team = normalize_team_name(game.get('away_team', '')).lower()
            game_id = game.get('game_id')
            
            if home_team and away_team and game_id:
                # Create keys for both directions
                game_map[f"{home_team}-{away_team}"] = game_id
                game_map[f"{away_team}-{home_team}"] = game_id  # For reverse lookup
        
        return game_map
    
    except Exception as e:
        print(f"Error getting official schedule: {e}")
        traceback.print_exc()
        return {}

def get_current_season():
    """
    Determine the current NBA season based on today's date.
    NBA season generally runs from October to June.
    """
    today = datetime.now()
    if today.month >= 7 and today.month <= 9:
        # July-September is off-season, return the upcoming season
        return f"{today.year}-{today.year + 1}"
    elif today.month >= 10:
        # October to December is the start of a new season
        return f"{today.year}-{today.year + 1}"
    else:
        # January to June is the latter part of the current season
        return f"{today.year - 1}-{today.year}"

def update_nba_schedule():
    """
    Update the NBA schedule for the current date range.
    """
    season = get_current_season()
    league = '12'  # NBA league ID
    
    # Get dates for yesterday, today, and tomorrow
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    # Fetch and process each date
    for date_obj in [yesterday, today, tomorrow]:
        date_str = date_obj.isoformat()
        print(f"Updating schedule for {date_str}...")
        
        games_data = get_games_by_date(league, season, date_str)
        if not games_data.get('response'):
            print(f"No games found for {date_str}")
            continue
        
        print(f"Found {len(games_data.get('response', []))} games for {date_str}")
        
        # Process each game
        for game in games_data.get('response', []):
            try:
                game_id = game.get('id')
                if not game_id:
                    continue
                
                # Extract game details
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
                
                # Convert date string to proper format
                game_date_str = game.get('date')
                try:
                    game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    game_date = game_datetime.date().isoformat()
                    scheduled_time = game_datetime.isoformat()
                except:
                    # If parse fails, use the date we're querying
                    game_date = date_str
                    scheduled_time = f"{date_str}T19:00:00.000Z"  # Default to 7pm
                
                # Normalize team names
                home_team = normalize_team_name(home_team)
                away_team = normalize_team_name(away_team)
                
                # Prepare record for insertion
                schedule_record = {
                    'game_id': game_id,
                    'game_date': game_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'scheduled_time': scheduled_time,
                    'venue': venue,
                    'status': status or 'scheduled',
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                # Upsert to schedule table
                result = supabase.table("nba_game_schedule").upsert(
                    schedule_record, 
                    on_conflict='game_id'
                ).execute()
                
                print(f"Updated schedule: {home_team} vs {away_team}")
                
            except Exception as e:
                print(f"Error processing game {game.get('id')}: {e}")
                continue
    
    return True

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
        
        # Debug venue data
        if 'response' in data and data['response']:
            for game in data['response']:
                print(f"DEBUG - Game ID {game.get('id')} Venue Data:", json.dumps(game.get('venue', {}), indent=2))
        
        return data
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

def transform_team_stats(game: dict, team_stats_data: dict, official_schedule: dict = None) -> dict:
    """
    Transform raw game and team stats data into the expected record format.
    
    Args:
        game: The game object from the /games endpoint
        team_stats_data: The full team stats response from the /games/statistics/teams endpoint
        official_schedule: Dictionary mapping normalized team pairs to game IDs
        
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
    
    # Get team names and normalize them
    raw_home_team = game.get('teams', {}).get('home', {}).get('name')
    raw_away_team = game.get('teams', {}).get('away', {}).get('name')
    
    normalized_home = normalize_team_name(raw_home_team)
    normalized_away = normalize_team_name(raw_away_team)
    
    print(f"Normalized team names: {normalized_home} vs {normalized_away}")
    
    # Get game_id from API first
    game_id = game.get('id')
    
    # If official schedule is provided, try to match to get consistent game_id
    if official_schedule:
        # Create key for lookup (both directions)
        key1 = f"{normalized_home.lower()}-{normalized_away.lower()}"
        key2 = f"{normalized_away.lower()}-{normalized_home.lower()}"
        
        # Check if this game exists in official schedule
        if key1 in official_schedule:
            official_game_id = official_schedule[key1]
            print(f"Found game in official schedule: {key1} -> {official_game_id}")
            game_id = official_game_id
        elif key2 in official_schedule:
            official_game_id = official_schedule[key2]
            print(f"Found game in official schedule (reversed): {key2} -> {official_game_id}")
            game_id = official_game_id
    
    # Base record with game and quarter information
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
        'game_date': game.get('date'),
        'current_quarter': determine_current_quarter(game),
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
    
    if 'current_quarter' in transformed:
        print(f"Removing current_quarter field (value: {transformed['current_quarter']}) from data")
        del transformed['current_quarter']
    
    return transformed

def determine_current_quarter(game):
    """
    Determine the current quarter based on game status and scores.
    
    Args:
        game: Game data dictionary
        
    Returns:
        Integer representing the current quarter (0-4, 0 = not started)
    """
    status = game.get('status', {}).get('short', '')
    
    # Check status first
    if status == 'NS':  # Not Started
        return 0
    elif status == 'FT':  # Finished
        return 4  # Assuming a standard game
    
    # For in-progress games, check the status directly
    status_map = {
        'Q1': 1,
        'Q2': 2,
        'Q3': 3,
        'Q4': 4,
        'OT': 4,  # Overtime counts as 4th quarter for our purposes
        'BT': 0,  # Break Time
        'HT': 2   # Halftime (after Q2)
    }
    
    if status in status_map:
        return status_map[status]
    
    # If status doesn't clearly indicate, check scores
    home_scores = game.get('scores', {}).get('home', {})
    
    # Check quarters in reverse order
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
    
    # Default if we can't determine
    return 0

###############################################################################
# 3) Main Driver: Fetch Live Games, Transform, and Upsert
###############################################################################

def run_live_games():
    # Use the current date in Pacific Time (live)
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    league = '12'          # NBA league ID
    season = get_current_season()
    timezone = 'America/Los_Angeles'
    
    # Get official schedule for verification
    official_schedule = get_official_schedule()
    if not official_schedule:
        print("Warning: Could not load official schedule. Using API game IDs only.")
    
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
        
        # Transform the raw game and team stats into the expected record format
        # Pass the official schedule for consistent game IDs
        record = transform_team_stats(game, team_stats_data, official_schedule)
        print("Transformed Team Record:")
        pprint(record)

        # Upsert the transformed record into the Supabase table 'nba_live_game_stats'
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
        # Update the NBA schedule first
        update_nba_schedule()
        
        # Then run live games data collection
        run_live_games()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()