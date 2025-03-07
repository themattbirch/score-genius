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

def transform_team_stats(game: dict, team_stats_list: list) -> dict:
    """
    Transform raw game and team stats data into the expected record format.
    """
    def get_quarter_score(scores: dict, key: str, fallback_key: str = None, default: int = 0) -> int:
        # Handle different API formats for quarter scores
        value = scores.get(key, None)
        if value is not None:
            return value
        if fallback_key:
            return scores.get(fallback_key, default)
        return default

    # Extract basic game info
    game_id = game.get("id")
    
    home_id = game.get("teams", {}).get("home", {}).get("id")
    away_id = game.get("teams", {}).get("away", {}).get("id")
    home_name = game.get("teams", {}).get("home", {}).get("name")
    away_name = game.get("teams", {}).get("away", {}).get("name")

    # Get quarter-by-quarter from g["scores"]
    home_scores = game.get("scores", {}).get("home", {})
    away_scores = game.get("scores", {}).get("away", {})

    home_q1 = get_quarter_score(home_scores, "quarter_1", "q1", 0) or 0
    home_q2 = get_quarter_score(home_scores, "quarter_2", "q2", 0) or 0
    home_q3 = get_quarter_score(home_scores, "quarter_3", "q3", 0) or 0
    home_q4 = get_quarter_score(home_scores, "quarter_4", "q4", 0) or 0
    home_ot = home_scores.get("over_time", 0) or home_scores.get("ot", 0) or 0
    home_score = home_scores.get("total", 0) or 0

    away_q1 = get_quarter_score(away_scores, "quarter_1", "q1", 0) or 0
    away_q2 = get_quarter_score(away_scores, "quarter_2", "q2", 0) or 0
    away_q3 = get_quarter_score(away_scores, "quarter_3", "q3", 0) or 0
    away_q4 = get_quarter_score(away_scores, "quarter_4", "q4", 0) or 0
    away_ot = away_scores.get("over_time", 0) or away_scores.get("ot", 0) or 0
    away_score = away_scores.get("total", 0) or 0

    # Find home and away team stats in the team_stats_list (already the response array)
    home_data = None
    away_data = None
    
    for team_obj in team_stats_list:
        team_id = team_obj.get("team", {}).get("id")
        if team_id == home_id:
            home_data = team_obj
        elif team_id == away_id:
            away_data = team_obj
    
    # Debug output
    print(f"Home team ID: {home_id}, Found home data: {home_data is not None}")
    print(f"Away team ID: {away_id}, Found away data: {away_data is not None}")
    
    # CRITICAL FIX: For historical games, the statistics aren't nested in a 'statistics'
    # field - they are directly in the team_obj itself!
    
    # Get home statistics
    if home_data:
        home_stats = home_data  # Use the whole object, not just a nested 'statistics' field
        print(f"Home stats direct access - assists: {home_stats.get('assists', 0)}")
    else:
        home_stats = {}
        
    # Get away statistics
    if away_data:
        away_stats = away_data  # Use the whole object, not just a nested 'statistics' field
        print(f"Away stats direct access - assists: {away_stats.get('assists', 0)}")
    else:
        away_stats = {}

    # Rebounds
    home_rebs = home_stats.get("rebounds", {})
    away_rebs = away_stats.get("rebounds", {})

    # Three-pointers
    home_3pt = home_stats.get("threepoint_goals", {}) or home_stats.get("threePoints", {})
    away_3pt = away_stats.get("threepoint_goals", {}) or away_stats.get("threePoints", {})

    record = {
        "game_id": game_id,

        # Basic identity
        "home_team": home_name,
        "away_team": away_name,

        # Quarter-based scoring
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

        # Advanced Stats - direct access from team object
        "home_assists": home_stats.get("assists", 0),
        "home_steals": home_stats.get("steals", 0),
        "home_blocks": home_stats.get("blocks", 0),
        "home_turnovers": home_stats.get("turnovers", 0),
        "home_fouls": home_stats.get("fouls", 0) or home_stats.get("personal_fouls", 0),

        "away_assists": away_stats.get("assists", 0),
        "away_steals": away_stats.get("steals", 0),
        "away_blocks": away_stats.get("blocks", 0),
        "away_turnovers": away_stats.get("turnovers", 0),
        "away_fouls": away_stats.get("fouls", 0) or away_stats.get("personal_fouls", 0),

        # Rebounds
        "home_off_reb": home_rebs.get("offence", 0) or home_rebs.get("offensive", 0),
        "home_def_reb": home_rebs.get("defense", 0) or home_rebs.get("defensive", 0),
        "home_total_reb": home_rebs.get("total", 0),

        "away_off_reb": away_rebs.get("offence", 0) or away_rebs.get("offensive", 0),
        "away_def_reb": away_rebs.get("defense", 0) or away_rebs.get("defensive", 0),
        "away_total_reb": away_rebs.get("total", 0),

        # 3-Point Goals
        "home_3pm": home_3pt.get("total", 0) or home_3pt.get("made", 0),
        "home_3pa": home_3pt.get("attempts", 0) or home_3pt.get("attempted", 0),
        "away_3pm": away_3pt.get("total", 0) or away_3pt.get("made", 0),
        "away_3pa": away_3pt.get("attempts", 0) or away_3pt.get("attempted", 0),

        # A date field if you store it
        "game_date": game.get("date", "").split("T")[0],  # Extract just the date part before the 'T'
        "updated_at": datetime.utcnow().isoformat()
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
    start_date = datetime(2025, 2, 21)
    end_date = datetime(2025, 3, 7)

    print(f"Starting historical data import from {start_date} to {end_date}")
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)  # to avoid potential rate-limiting
        current += timedelta(days=1)

    print("\nCompleted processing historical GAME-level stats.")

if __name__ == "__main__":
    main()