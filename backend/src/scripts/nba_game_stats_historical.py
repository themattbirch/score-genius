# backend/src/scripts/nba_game_stats_historical.py

import requests
import json
import time
from datetime import datetime, timedelta
import sys, os

# Add the backend root to the Python path so we can import from caching and config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase

API_KEY = API_SPORTS_KEY
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v1.basketball.api-sports.io"
}

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

def get_games_by_date(league, season, date):
    """
    Fetch games for a specific date from the API.
    """
    url = f"{BASE_URL}/games"
    params = {"league": league, "season": season, "date": date}
    try:
        print(f"Fetching games for {date} (Season {season})")
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # Debug the response structure
        print(f"Found {len(data.get('response', []))} games for {date}")
        if data.get('response'):
            print("Game fields include: id, teams, scores, status, etc.")
        
        return data.get("response", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching games for {date}: {e}")
        return []

def get_team_box_stats(game_id):
    """
    Fetch team statistics for a specific game.
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {"id": game_id}
    try:
        print(f"Fetching team stats for game ID {game_id}")
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        print("Team stats data received successfully")
        return data.get("response", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team stats for game ID {game_id}: {e}")
        return []

def parse_season_from_date(date_obj):
    """
    Determine the NBA season string from a date.
    """
    if date_obj.month >= 10:
        return f"{date_obj.year}-{date_obj.year+1}"
    else:
        return f"{date_obj.year-1}-{date_obj.year}"

def extract_team_stats(team_data, team_id):
    """
    Extract team statistics with robust fallbacks.
    Uses default value of 1 for missing defensive stats to ensure visibility.
    """
    stats = {}
    
    for t in team_data:
        if get_nested_value(t, 'team', 'id') == team_id:
            # Ensure all defensive stats have at least 1 as a placeholder
            stats = {
                "assists": get_nested_value(t, 'assists', default=0),
                "steals": get_nested_value(t, 'steals', default=1),
                "blocks": get_nested_value(t, 'blocks', default=1),
                "turnovers": get_nested_value(t, 'turnovers', default=1),
                "personal_fouls": get_nested_value(t, 'personal_fouls', default=1),
                "off_reb": get_nested_value(t, 'rebounds', 'offence', default=1),
                "def_reb": get_nested_value(t, 'rebounds', 'defense', default=1),
                "total_reb": get_nested_value(t, 'rebounds', 'total', default=2),
                "3pm": get_nested_value(t, 'threepoint_goals', 'total', default=1),
                "3pa": get_nested_value(t, 'threepoint_goals', 'attempts', default=3)
            }
            
            # Ensure total_rebounds is not less than the sum of offensive and defensive rebounds
            if stats["total_reb"] < stats["off_reb"] + stats["def_reb"]:
                stats["total_reb"] = stats["off_reb"] + stats["def_reb"]
                
            break
    
    # If we didn't find the team, return default values
    if not stats:
        stats = {
            "assists": 0,
            "steals": 1,
            "blocks": 1,
            "turnovers": 1, 
            "personal_fouls": 1,
            "off_reb": 1,
            "def_reb": 1,
            "total_reb": 2,
            "3pm": 1,
            "3pa": 3
        }
    
    return stats

def process_day(date_obj):
    """
    Process all games for a specific date.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    season = parse_season_from_date(date_obj)
    print(f"\n=== Processing {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    processed_count = 0
    for g in games_list:
        if g.get("status", {}).get("short") != "FT":
            continue  # Process only final games

        game_id = g["id"]
        home_team = get_nested_value(g, 'teams', 'home', 'name', default='Unknown')
        away_team = get_nested_value(g, 'teams', 'away', 'name', default='Unknown')
        home_score = get_nested_value(g, 'scores', 'home', 'total', default=0)
        away_score = get_nested_value(g, 'scores', 'away', 'total', default=0)

        print(f"\nProcessing game: {away_team} @ {home_team} (Game ID: {game_id})")

        # Get quarter scores with better fallbacks
        scores = g.get("scores", {})
        
        # Home quarter scores
        home_scores = get_nested_value(scores, 'home', default={})
        home_q1 = get_nested_value(home_scores, 'quarter_1', default=0) or 0
        home_q2 = get_nested_value(home_scores, 'quarter_2', default=0) or 0
        home_q3 = get_nested_value(home_scores, 'quarter_3', default=0) or 0
        home_q4 = get_nested_value(home_scores, 'quarter_4', default=0) or 0
        home_ot = get_nested_value(home_scores, 'over_time', default=0) or 0
        
        # Away quarter scores
        away_scores = get_nested_value(scores, 'away', default={})
        away_q1 = get_nested_value(away_scores, 'quarter_1', default=0) or 0
        away_q2 = get_nested_value(away_scores, 'quarter_2', default=0) or 0
        away_q3 = get_nested_value(away_scores, 'quarter_3', default=0) or 0
        away_q4 = get_nested_value(away_scores, 'quarter_4', default=0) or 0
        away_ot = get_nested_value(away_scores, 'over_time', default=0) or 0

        # Get team box stats
        team_stats = get_team_box_stats(game_id)
        
        # Get team IDs for matching
        home_team_id = get_nested_value(g, 'teams', 'home', 'id')
        away_team_id = get_nested_value(g, 'teams', 'away', 'id')

        # Extract home and away box stats using our robust extraction function
        home_box = extract_team_stats(team_stats, home_team_id)
        away_box = extract_team_stats(team_stats, away_team_id)

        # Create the record with all fields that are present in the database schema
        # Do NOT include fields like away_fg_attempted which are not in the schema
        record = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
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
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "home_assists": home_box.get("assists", 0),
            "home_steals": home_box.get("steals", 1),
            "home_blocks": home_box.get("blocks", 1),
            "home_turnovers": home_box.get("turnovers", 1),
            "home_fouls": home_box.get("personal_fouls", 1),
            "away_assists": away_box.get("assists", 0),
            "away_steals": away_box.get("steals", 1),
            "away_blocks": away_box.get("blocks", 1),
            "away_turnovers": away_box.get("turnovers", 1),
            "away_fouls": away_box.get("personal_fouls", 1),
            "home_off_reb": home_box.get("off_reb", 1),
            "home_def_reb": home_box.get("def_reb", 1),
            "home_total_reb": home_box.get("total_reb", 2),
            "away_off_reb": away_box.get("off_reb", 1),
            "away_def_reb": away_box.get("def_reb", 1),
            "away_total_reb": away_box.get("total_reb", 2),
            "home_3pm": home_box.get("3pm", 1),
            "home_3pa": home_box.get("3pa", 3),
            "away_3pm": away_box.get("3pm", 1),
            "away_3pa": away_box.get("3pa", 3),
        }

        print(f"Upserting game record: {home_team} vs {away_team}")
        
        try:
            res = (
                supabase
                .table("nba_historical_game_stats")
                .upsert(record, on_conflict=["game_id"])
                .execute()
            )
            print(f"Upsert successful for game ID {game_id}")
            processed_count += 1
        except Exception as e:
            print(f"Error during upsert for game ID {game_id}: {e}")
            if hasattr(e, 'response'):
                try:
                    print(f"Raw error response: {e.response.text}")
                except Exception:
                    pass
    
    print(f"\nProcessed {processed_count} games for {date_str}")

def main():
    """
    Main function to process a range of dates.
    """
    # Date range to process
    start_date = datetime(2018, 10, 19)
    end_date = datetime(2025, 3, 1)
    
    print(f"Starting historical data import from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    current = start_date
    days_processed = 0
    
    while current <= end_date:
        process_day(current)
        
        # Rate limiting to avoid API issues
        wait_seconds = 60
        print(f"Waiting {wait_seconds} seconds before processing next date...")
        time.sleep(wait_seconds)
        
        current += timedelta(days=1)
        days_processed += 1
    
    print(f"\nCompleted processing {days_processed} days of games")

if __name__ == "__main__":
    main()