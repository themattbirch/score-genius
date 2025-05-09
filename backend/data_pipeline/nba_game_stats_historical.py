# backend/data_pipeline/nba_game_stats_historical.py

import json
import requests
import sys
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# --- Local Config & Variables (same pattern as MLB script) ---
try:
    from config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )
    print("Using config.py for credentials")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate
_missing = [ name for name,val in [
    ("API_SPORTS_KEY",       API_SPORTS_KEY),
    ("ODDS_API_KEY",         ODDS_API_KEY),
    ("SUPABASE_URL",         SUPABASE_URL),
    ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
] if not val ]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    exit(1)

HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

    
from caching.supabase_client import supabase
from caching.supabase_stats import upsert_historical_game_stats_team

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

def get_games_by_date(league: str, season: str, date_str: str) -> list:
    """
    Fetches a list of final/historical games from /games for the given league, season, and date.
    Returns the "response" list from the JSON.
    """
    url = f"{BASE_URL}/games"
    params = {"league": league, "season": season, "date": date_str}
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        print(f"Fetched {len(data.get('response', []))} games for {date_str}")
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
        return []

def get_team_stats(game_id: int) -> list:
    """
    Fetches advanced team-level stats from /games/statistics/teams for a given game_id.
    Typically returns a list with two entries (home/away).
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {"id": game_id}
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # Debug: Print the raw response structure
        print(f"DEBUG: Response status: {resp.status_code}")
        print(f"DEBUG: Response keys: {list(data.keys())}")
        print(f"DEBUG: Response has {len(data.get('response', []))} items")
        
        # Print sample of first response item
        if data.get("response") and len(data.get("response")) > 0:
            first_item = data.get("response")[0]
            print(f"DEBUG: First item keys: {list(first_item.keys())}")
            # Look for statistics fields
            for key in list(first_item.keys())[:5]:
                if key != 'team' and key != 'game':
                    print(f"DEBUG: Sample stat {key}: {first_item.get(key)}")
        
        print(f"Fetched team stats for game_id={game_id}")
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching team stats for game_id={game_id}: {e}")
        if 'resp' in locals():
            print(f"DEBUG: Response content (if available): {resp.text}")
        return []

def parse_season_for_date(d: datetime) -> str:
    """
    For an NBA-like approach:
      - If month >= 10, we say "YYYY-YYYY+1"
      - Else "YYYY-1-YYYY"
    """
    if d.month >= 10:
        return f"{d.year}-{d.year+1}"
    else:
        return f"{d.year-1}-{d.year}"

def _find_statistics_for_team(team_stats_list, team_id):
    """
    Given the full list of stats for both teams, return all statistics
    for the team matching team_id, or {} if not found.
    """
    for item in team_stats_list:
        if item.get("team", {}).get("id") == team_id:
            # Return the item without the team and game keys
            stats = {k: v for k, v in item.items() if k not in ['team', 'game']}
            return stats
    return {}

##############################################################################
# Transform Game Stats for nba_historical_game_stats
##############################################################################

def transform_game_stats(game: dict, team_stats_list: list, date_str: str) -> dict:
    """
    Build a single record with home/away columns for the nba_historical_game_stats table.
    """
    # Basic game info
    game_id = game["id"]
    home_team_name = get_nested_value(game, "teams", "home", "name", default="Unknown")
    away_team_name = get_nested_value(game, "teams", "away", "name", default="Unknown")
    
    home_score = get_nested_value(game, "scores", "home", "total", default=0)
    away_score = get_nested_value(game, "scores", "away", "total", default=0)
    
    # Access quarter scores directly by field name instead of linescore array
    home_q1 = get_nested_value(game, "scores", "home", "quarter_1", default=0)
    home_q2 = get_nested_value(game, "scores", "home", "quarter_2", default=0)
    home_q3 = get_nested_value(game, "scores", "home", "quarter_3", default=0)
    home_q4 = get_nested_value(game, "scores", "home", "quarter_4", default=0)
    # Handle overtime if it exists (it's null when there's no OT)
    home_ot_value = get_nested_value(game, "scores", "home", "over_time")
    home_ot = home_ot_value if home_ot_value is not None else 0
    
    away_q1 = get_nested_value(game, "scores", "away", "quarter_1", default=0)
    away_q2 = get_nested_value(game, "scores", "away", "quarter_2", default=0)
    away_q3 = get_nested_value(game, "scores", "away", "quarter_3", default=0)
    away_q4 = get_nested_value(game, "scores", "away", "quarter_4", default=0)
    away_ot_value = get_nested_value(game, "scores", "away", "over_time")
    away_ot = away_ot_value if away_ot_value is not None else 0

    # Identify home/away team IDs to pull advanced stats
    home_team_id = get_nested_value(game, "teams", "home", "id", default=0)
    away_team_id = get_nested_value(game, "teams", "away", "id", default=0)
    
    # Advanced stats
    home_stats = _find_statistics_for_team(team_stats_list, home_team_id)
    away_stats = _find_statistics_for_team(team_stats_list, away_team_id)
    
    # Get the structured objects
    field_goals_home = home_stats.get("field_goals", {})
    field_goals_away = away_stats.get("field_goals", {})
    three_points_home = home_stats.get("threepoint_goals", {})
    three_points_away = away_stats.get("threepoint_goals", {})
    free_throws_home = home_stats.get("freethrows_goals", {})
    free_throws_away = away_stats.get("freethrows_goals", {})
    rebounds_home = home_stats.get("rebounds", {})
    rebounds_away = away_stats.get("rebounds", {})

    record = {
        "game_id": game_id,
        "home_team": home_team_name,
        "away_team": away_team_name,
        "home_score": home_score,
        "away_score": away_score,
        "home_q1": home_q1,
        "home_q2": home_q2,
        "home_q3": home_q3,
        "home_q4": home_q4,
        "home_ot": home_ot,
        "away_q1": away_q1,
        "away_q2": away_q2,
        "away_q3": away_q3,
        "away_q4": away_q4,
        "away_ot": away_ot,
        "game_date": date_str,

        # Main stats (these are working correctly already)
        "home_assists": home_stats.get("assists", 0),
        "home_steals": home_stats.get("steals", 0),
        "home_blocks": home_stats.get("blocks", 0),
        "home_turnovers": home_stats.get("turnovers", 0),
        "home_fouls": home_stats.get("personal_fouls", 0),
        
        # Nested stats (these are working correctly now)
        "home_off_reb": rebounds_home.get("offence", 0),
        "home_def_reb": rebounds_home.get("defense", 0),
        "home_total_reb": rebounds_home.get("total", 0),
        "home_3pm": three_points_home.get("total", 0),
        "home_3pa": three_points_home.get("attempts", 0),
        "home_fg_made": field_goals_home.get("total", 0),
        "home_fg_attempted": field_goals_home.get("attempts", 0),
        "home_ft_made": free_throws_home.get("total", 0),
        "home_ft_attempted": free_throws_home.get("attempts", 0),

        # Away team stats (same structure as home)
        "away_assists": away_stats.get("assists", 0),
        "away_steals": away_stats.get("steals", 0),
        "away_blocks": away_stats.get("blocks", 0),
        "away_turnovers": away_stats.get("turnovers", 0),
        "away_fouls": away_stats.get("personal_fouls", 0),
        "away_off_reb": rebounds_away.get("offence", 0),
        "away_def_reb": rebounds_away.get("defense", 0),
        "away_total_reb": rebounds_away.get("total", 0),
        "away_3pm": three_points_away.get("total", 0),
        "away_3pa": three_points_away.get("attempts", 0),
        "away_fg_made": field_goals_away.get("total", 0),
        "away_fg_attempted": field_goals_away.get("attempts", 0),
        "away_ft_made": free_throws_away.get("total", 0),
        "away_ft_attempted": free_throws_away.get("attempts", 0),
    }

    return record

##############################################################################
# Process Each Day
##############################################################################

def process_day(date_obj: datetime):
    date_str = date_obj.strftime("%Y-%m-%d")
    season = parse_season_for_date(date_obj)
    print(f"\n=== {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    processed_count = 0
    for g in games_list:
        # Only handle final games
        if g.get("status", {}).get("short") != "FT":
            continue

        game_id = g["id"]
        print(f"Processing game ID: {game_id}")

        # Fetch advanced box stats
        team_stats_list = get_team_stats(game_id)
        if len(team_stats_list) < 2:
            print(f"No advanced team stats for game_id={game_id}.")
            continue

        # Transform the data to our required format
        record = transform_game_stats(g, team_stats_list, date_str)
        if not record:
            print(f"Could not build record for game_id={game_id}.")
            continue

        print(
            f"[INFO] Upserting game stats => "
            f"Game ID: {record['game_id']}, "
            f"{record['away_team']} @ {record['home_team']}"
        )
        try:
            res = upsert_historical_game_stats_team(record)
            print(f"[INFO] Upsert result: {res}")
        except Exception as e:
            print(f"[ERROR] Could not upsert game_id={game_id}: {e}")

        processed_count += 1

    print(f"Processed {processed_count} final games for {date_str}")

##############################################################################
# Main Runner
##############################################################################

def main():
    # Determine “yesterday” in Eastern Time
    et = ZoneInfo("America/New_York")
    yesterday_et = (datetime.now(et) - timedelta(days=1)).date()

    print(f"Starting historical data import for {yesterday_et}")
    # process_day expects a datetime; zero‐out the time but keep tzinfo
    process_day(datetime(
        yesterday_et.year,
        yesterday_et.month,
        yesterday_et.day,
        tzinfo=et
    ))
    time.sleep(5)  # avoid rate limits

    print("\nCompleted processing historical GAME-level stats.")

if __name__ == "__main__":
    main()