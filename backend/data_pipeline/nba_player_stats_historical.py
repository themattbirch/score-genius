# backend/data_pipeline/nba_player_stats_historical.py

"""
Fetch and upsert historical NBA player stats from API-Sports into Supabase.
"""

import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests

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

from caching.supabase_client import supabase
from caching.supabase_stats import upsert_historical_game_stats

# --- Constants ---
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.basketball.api-sports.io",
}
LEAGUE_ID = "12"  # NBA
RATE_LIMIT_SLEEP = 60  # seconds between days to avoid rate limits


def convert_minutes(time_str: str) -> float:
    """Convert 'MM:SS' to minutes as float."""
    try:
        m, s = map(int, time_str.split(":"))
        return m + s / 60.0
    except Exception:
        return 0.0


def get_games_by_date(league: str, season: str, date_str: str) -> List[Dict[str, Any]]:
    """Fetch all games for a given league, season, and date."""
    url = f"{BASE_URL}/games"
    resp = requests.get(url, headers=HEADERS, params={
        "league": league,
        "season": season,
        "date": date_str,
    })
    resp.raise_for_status()
    return resp.json().get("response", [])


def get_player_stats(game_id: int) -> List[Dict[str, Any]]:
    """Fetch per-player statistics for a single game."""
    url = f"{BASE_URL}/games/statistics/players"
    resp = requests.get(url, headers=HEADERS, params={"ids": game_id})
    resp.raise_for_status()
    # fix apostrophe encoding
    data = json.loads(
        resp.text.replace("&amp;apos;", "'").replace("&apos;", "'")
    )
    return data.get("response", [])


def process_day(date_obj: datetime) -> None:
    """Process all final games on a given date."""
    date_str = date_obj.strftime("%Y-%m-%d")
    # NBA season string e.g. "2024-2025"
    season = (
        f"{date_obj.year-1}-{date_obj.year}"
        if date_obj.month < 10
        else f"{date_obj.year}-{date_obj.year+1}"
    )
    print(f"\n=== {date_str} (Season {season}) ===")

    games = get_games_by_date(LEAGUE_ID, season, date_str)
    if not games:
        print("No games found.")
        return

    for game in games:
        if game.get("status", {}).get("short") != "FT":
            continue  # only final games

        game_id = game["id"]
        home = game["teams"]["home"]
        away = game["teams"]["away"]
        home_id, home_name = home.get("id"), home.get("name")
        away_id, away_name = away.get("id"), away.get("name")

        print(f"\nProcessing {away_name} @ {home_name} (Game ID: {game_id})")
        player_stats = get_player_stats(game_id)
        if not player_stats:
            print("  No player stats.")
            continue

        for p in player_stats:
            # determine player's actual team
            pt_id = p.get("team", {}).get("id")
            team_name = (
                home_name if pt_id == home_id
                else away_name if pt_id == away_id
                else "Unknown"
            )

            print(f"  Upserting {p.get('player', {}).get('name', 'Unknown')} ({team_name})")
            try:
                res = upsert_historical_game_stats(
                    game_id,
                    p,
                    date_str,
                    team_name
                )
                print(f"    → Result: {res}")
            except Exception as e:
                print(f"    ✖️ Error upserting: {e}")


def main() -> None:
    start = datetime(2025, 4, 26)
    end = datetime(2025, 4, 29)
    print(f"Starting import: {start.date()} → {end.date()}")

    current = start
    while current <= end:
        process_day(current)
        time.sleep(RATE_LIMIT_SLEEP)
        current += timedelta(days=1)

    print("Completed historical player stats import.")


if __name__ == "__main__":
    main()
