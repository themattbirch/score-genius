import json
import requests
import sys
import os
import time
from datetime import datetime, timedelta

# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase
# IMPORTANT: Must upsert columns matching home_assists, home_steals, home_off_reb, etc.
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

##############################################################################
# 1) Fetch from the API
##############################################################################

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
        print(f"Fetched team stats for game_id={game_id}")
        # Return directly the response list, not the full dictionary
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching team stats for game_id={game_id}: {e}")
        return []

##############################################################################
# 2) Determine the NBA season
##############################################################################

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

##############################################################################
# 3) Transform Team Stats
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
    
    # Removed extraction of form/streak data
    
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
        
        # Keeping current_form (you specified to keep this one)
        "current_form": get_nested_value(stats, "form", default=""),
        
        # Metadata
        "updated_at": datetime.now().isoformat()
    }
    
    return record

##############################################################################
# 4) Process Day
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
        record = transform_team_stats(g, team_stats_list)
        
        print(f"[INFO] Upserting game stats => Game ID: {record['game_id']}, {record['away_team']} @ {record['home_team']}")
        try:
            res = upsert_historical_game_stats_team(record)
            print(f"[INFO] Upsert result: {res}")
        except Exception as e:
            print(f"[ERROR] Could not upsert game_id={game_id}: {e}")

        processed_count += 1

    print(f"Processed {processed_count} final games for {date_str}")

##############################################################################
# 5) Main Runner
##############################################################################

def main():
    start_date = datetime(2025, 3, 16)
    end_date = datetime(2025, 3, 17)

    print(f"Starting historical data import from {start_date} to {end_date}")
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)  # to avoid potential rate-limiting
        current += timedelta(days=1)

    print("\nCompleted processing historical GAME-level stats.")

if __name__ == "__main__":
    main()