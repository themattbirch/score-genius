# backend/data_pipeline/nba_team_stats_historical.py

import json
import requests
import sys
import os
import time
import traceback
from datetime import datetime, timedelta

# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# --- Local Config & Variables ---
try:
    from config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )
    print("Successfully imported configuration variables from config.py")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate
_missing = [
    name
    for name, val in [
        ("API_SPORTS_KEY",       API_SPORTS_KEY),
        ("ODDS_API_KEY",         ODDS_API_KEY),
        ("SUPABASE_URL",         SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
    ]
    if not val
]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    sys.exit(1)


HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from caching.supabase_client import supabase

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

##############################################################################
# Helper Functions
##############################################################################

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

def is_regular_team(team_name):
    """
    Determine if a team is a regular NBA team or a special event team (All-Star, Rising Stars, etc.).
    Returns True for regular teams, False for special event teams.
    """
    if not team_name:
        return False
        
    # List of keywords that indicate special event teams
    special_event_keywords = [
        "rising stars", 
        "global stars", 
        "young stars", 
        "ogs",
        "all-stars",
        "all stars",
        "rookies",
        "sophomores",
        "world",
        "usa"
    ]
    
    # Check if team name contains any of the special event keywords
    team_name_lower = team_name.lower()
    for keyword in special_event_keywords:
        if keyword in team_name_lower:
            return False
    
    # Check if "team" is in the name (All-Star teams often have this pattern)
    if "team" in team_name_lower:
        return False
        
    # Specific exclusions (add more as needed)
    specific_exclusions = [
        "candace's",
        "chuck's",
        "kenny's",
        "shaq's"
    ]
    
    for exclusion in specific_exclusions:
        if exclusion in team_name_lower:
            return False
    
    # If we've made it here, it's likely a regular team
    return True

##############################################################################
# 1) Fetch from the API
##############################################################################

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

import json  # Make sure to add this import at the top

def get_team_stats(team_id: int, league: str, season: str) -> dict:
    """
    Fetches team statistics from /statistics for a given team, league, and season.
    """
    url = f"{BASE_URL}/statistics"
    params = {
        "team": team_id,
        "league": league,
        "season": season
    }
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # Add these debug lines:
        print(f"DEBUG - Full API response for team {team_id}:")
        print(json.dumps(data, indent=2))
        
        print(f"Fetched stats for team_id={team_id}, league={league}, season={season}")
        # Return directly the response object, not the full dictionary
        # The statistics endpoint returns a single object, not a list
        return data.get("response", {})
    except Exception as e:
        print(f"Error fetching team stats for team_id={team_id}, league={league}, season={season}: {e}")
        return {}

##############################################################################
# 2) Transform Team Stats
##############################################################################

def transform_team_stats(team: dict, stats: dict, league_id: str, season: str) -> dict:
    """
    Transform raw team stats data into the expected record format for our database.
    """
    # Extract basic team info
    team_id = team.get("id")
    team_name = team.get("name")
    
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
    
    # Extract form/streak data (NEW)
    form = get_nested_value(stats, "form", default="")
    
    # Calculate streak based on form (NEW)
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
        
        # NEW FIELDS
        "current_form": form,
        
        # Metadata
        "updated_at": datetime.now().isoformat()
    }
    
    return record

##############################################################################
# 3) Upsert to Supabase
##############################################################################

def upsert_historical_team_stats(record):
    """
    Upsert team statistics to the nba_historical_team_stats table.
    """
    try:
        result = supabase.table("nba_historical_team_stats").upsert(
            record, on_conflict="team_id,season,league_id"
        ).execute()
        return result
    except Exception as e:
        print(f"Error upserting team stats: {e}")
        return None

##############################################################################
# 4) Process Teams for Season
##############################################################################

def calculate_team_form(team_id, team_name, num_games=5):
    """
    Calculate a team's form based on their most recent games.
    
    Args:
        team_id (int): The ID of the team
        team_name (str): The name of the team (for matching in game records)
        num_games (int): Number of recent games to include in form (default: 5)
        
    Returns:
        str: A string representation of recent form (e.g., "WLWWL")
    """
    try:
        # Query recent games involving this team
        # Use the correct order syntax for your Supabase client
        response = supabase.table("nba_historical_game_stats").select("*").or_(
            f"home_team.eq.{team_name},away_team.eq.{team_name}"
        ).order('game_date.desc').limit(num_games).execute()
        
        games = response.data
        
        # If no games found, return empty form
        if not games:
            print(f"No recent games found for team: {team_name}")
            return ""
        
        form = ""
        for game in games:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            is_home_team = home_team == team_name
            
            # Get relevant scores
            if is_home_team:
                team_score = game.get('home_score', 0)
                opponent_score = game.get('away_score', 0)
            else:
                team_score = game.get('away_score', 0)
                opponent_score = game.get('home_score', 0)
            
            # Determine win or loss
            if team_score > opponent_score:
                form += "W"  # Add win at the end
            else:
                form += "L"  # Add loss at the end
        
        return form
    
    except Exception as e:
        print(f"Error calculating form for team {team_name}: {e}")
        traceback.print_exc()  # Added stacktrace to see more details
        return ""

def process_teams_for_season(league_id: str, season: str):
    """
    Process all teams for a given league and season.
    """
    print(f"\n=== Processing teams for league {league_id}, season {season} ===")
    
    # Fetch teams for this league and season
    teams_list = get_teams_by_league_season(league_id, season)
    if not teams_list:
        print("No teams found.")
        return 0
    
    # Filter out All-Star and special event teams
    regular_teams = []
    filtered_count = 0
    
    for team in teams_list:
        team_name = team.get("name", "")
        if is_regular_team(team_name):
            regular_teams.append(team)
        else:
            filtered_count += 1
            print(f"Filtering out special event team: {team_name}")
    
    print(f"Filtered out {filtered_count} special event teams. Processing {len(regular_teams)} regular teams.")
    
    processed_count = 0
    for team in regular_teams:
        team_id = team.get("id")
        team_name = team.get("name")
        print(f"Processing team: {team_name} (ID: {team_id})")
        
        # Fetch team statistics
        team_stats = get_team_stats(team_id, league_id, season)
        if not team_stats:
            print(f"No statistics found for team_id={team_id}.")
            continue
        
        # Transform the data to our required format
        record = transform_team_stats(team, team_stats, league_id, season)
        
        # Calculate and add custom form data
        current_form = calculate_team_form(team_id, team_name)
        if current_form:
            print(f"Calculated form for {team_name}: {current_form}")
            record['current_form'] = current_form
        
        print(f"[INFO] Upserting team stats => Team ID: {record['team_id']}, Team Name: {record['team_name']}")
        try:
            res = upsert_historical_team_stats(record)
            print(f"[INFO] Upsert result: {res}")
        except Exception as e:
            print(f"[ERROR] Could not upsert team_id={team_id}: {e}")
        
        processed_count += 1
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    print(f"Processed {processed_count} teams for league {league_id}, season {season}")
    return processed_count



##############################################################################
# 5) Main Runner
##############################################################################

def main():
    # Set your date range here
    start_date = datetime(2025, 3, 10)
    end_date = datetime(2025, 3, 17)
    
    # NBA League ID19
    league_id = "12"
    
    # Get unique seasons between start_date and end_date
    seasons = set()
    current = start_date
    while current <= end_date:
        year = current.year
        month = current.month
        
        # Determine the season based on the date
        if month >= 10:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"
        
        seasons.add(season)
        current += timedelta(days=1)
    
    # Convert to sorted list
    seasons_list = sorted(list(seasons))
    
    print(f"Starting historical team stats import from {start_date} to {end_date}")
    print(f"Seasons to process: {', '.join(seasons_list)}")
    
    total_processed = 0
    for season in seasons_list:
        count = process_teams_for_season(league_id, season)
        total_processed += count
        # Sleep between seasons to avoid rate limiting
        time.sleep(5)
    
    print(f"\nCompleted processing historical TEAM-level stats for {total_processed} teams.")

if __name__ == "__main__":
    main()