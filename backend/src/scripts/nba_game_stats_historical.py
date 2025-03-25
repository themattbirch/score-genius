# File: nba_game_stats_historical.py

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
# IMPORTANT: We only need this upsert function for game-level stats
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
        print(f"Fetched team stats for game_id={game_id}")
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching team stats for game_id={game_id}: {e}")
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
    Given the full list of stats for both teams, return the 'statistics' object
    for the team matching team_id, or {} if not found.
    """
    for item in team_stats_list:
        if item.get("team", {}).get("id") == team_id:
            return item.get("statistics", {})
    return {}

##############################################################################
# NEW: Transform Game Stats for nba_historical_game_stats
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
    
    # linescore is typically a list [Q1, Q2, Q3, Q4, (OT if any), ...]
    home_linescore = get_nested_value(game, "scores", "home", "linescore", default=[])
    away_linescore = get_nested_value(game, "scores", "away", "linescore", default=[])

    # Quarter-by-quarter; handle if the API returns fewer or more than 4
    home_q1 = home_linescore[0] if len(home_linescore) > 0 else 0
    home_q2 = home_linescore[1] if len(home_linescore) > 1 else 0
    home_q3 = home_linescore[2] if len(home_linescore) > 2 else 0
    home_q4 = home_linescore[3] if len(home_linescore) > 3 else 0
    # If there's more than 4 entries, that means OT (or multiple OTs). Summation in home_ot:
    home_ot = sum(home_linescore[4:]) if len(home_linescore) > 4 else 0

    away_q1 = away_linescore[0] if len(away_linescore) > 0 else 0
    away_q2 = away_linescore[1] if len(away_linescore) > 1 else 0
    away_q3 = away_linescore[2] if len(away_linescore) > 2 else 0
    away_q4 = away_linescore[3] if len(away_linescore) > 3 else 0
    away_ot = sum(away_linescore[4:]) if len(away_linescore) > 4 else 0

    # Identify home/away team IDs to pull advanced stats
    home_team_id = get_nested_value(game, "teams", "home", "id", default=0)
    away_team_id = get_nested_value(game, "teams", "away", "id", default=0)

    # Advanced stats
    home_stats = _find_statistics_for_team(team_stats_list, home_team_id)
    away_stats = _find_statistics_for_team(team_stats_list, away_team_id)

    # For each side, we pick out the relevant fields (assists, steals, blocks, etc.)
    # The /games/statistics/teams response typically includes something like:
    #   {
    #       "points": 100,
    #       "assists": 25,
    #       "steals": 8,
    #       "blocks": 4,
    #       "turnovers": 12,
    #       "fouls": 18,
    #       "rebounds": {"offensive": 10, "defensive": 30, "total": 40},
    #       "field_goals": {"made": 40, "attempts": 90},
    #       "three_points": {"made": 10, "attempts": 25},
    #       "free_throws": {"made": 10, "attempts": 14},
    #       ...
    #   }
    #
    # If your actual API data is named slightly differently, adjust accordingly.
    
    def_off_reb_home = home_stats.get("rebounds", {})
    def_off_reb_away = away_stats.get("rebounds", {})
    field_goals_home = home_stats.get("field_goals", {})
    field_goals_away = away_stats.get("field_goals", {})
    three_points_home = home_stats.get("three_points", {})
    three_points_away = away_stats.get("three_points", {})
    free_throws_home = home_stats.get("free_throws", {})
    free_throws_away = away_stats.get("free_throws", {})

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

        "home_assists": home_stats.get("assists", 0),
        "home_steals": home_stats.get("steals", 0),
        "home_blocks": home_stats.get("blocks", 0),
        "home_turnovers": home_stats.get("turnovers", 0),
        "home_fouls": home_stats.get("fouls", 0),
        "home_off_reb": def_off_reb_home.get("offensive", 0),
        "home_def_reb": def_off_reb_home.get("defensive", 0),
        "home_total_reb": def_off_reb_home.get("total", 0),
        "home_3pm": three_points_home.get("made", 0),
        "home_3pa": three_points_home.get("attempts", 0),
        "home_fg_made": field_goals_home.get("made", 0),
        "home_fg_attempted": field_goals_home.get("attempts", 0),
        "home_ft_made": free_throws_home.get("made", 0),
        "home_ft_attempted": free_throws_home.get("attempts", 0),

        "away_assists": away_stats.get("assists", 0),
        "away_steals": away_stats.get("steals", 0),
        "away_blocks": away_stats.get("blocks", 0),
        "away_turnovers": away_stats.get("turnovers", 0),
        "away_fouls": away_stats.get("fouls", 0),
        "away_off_reb": def_off_reb_away.get("offensive", 0),
        "away_def_reb": def_off_reb_away.get("defensive", 0),
        "away_total_reb": def_off_reb_away.get("total", 0),
        "away_3pm": three_points_away.get("made", 0),
        "away_3pa": three_points_away.get("attempts", 0),
        "away_fg_made": field_goals_away.get("made", 0),
        "away_fg_attempted": field_goals_away.get("attempts", 0),
        "away_ft_made": free_throws_away.get("made", 0),
        "away_ft_attempted": free_throws_away.get("attempts", 0),
    }

    return record

##############################################################################
# 4) Process Each Day
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
# 5) Main Runner
##############################################################################

def main():
    # Example: fetch data from 2025-03-16 to 2025-03-17
    start_date = datetime(2025, 3, 10)
    end_date = datetime(2025, 3, 24)

    print(f"Starting historical data import from {start_date} to {end_date}")
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)  # to avoid potential rate-limiting
        current += timedelta(days=1)

    print("\nCompleted processing historical GAME-level stats.")

if __name__ == "__main__":
    main()
