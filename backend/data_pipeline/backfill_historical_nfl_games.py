# backend/data_pipeline/backfill_historical_nfl_games.py
"""
backfill_historical_nfl_games.py - One‑off script to backfill historical NFL games
Dates: December 11–15, 2024

Fetch real game data via API‑SPORTS and upsert into `nfl_game_schedule` Supabase table.
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import sys
import time
from datetime import date, timedelta, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from caching.supabase_client import supabase

# ── Configuration ────────────────────────────────────────────────────────────
try:
    from config import API_SPORTS_KEY
    print("Using config.py for API key")
except ImportError:
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")

if not API_SPORTS_KEY:
    print("FATAL: Missing API_SPORTS_KEY")
    sys.exit(1)

BASE_URL       = "https://v1.american-football.api-sports.io"
HEADERS        = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID  = 1
SUPABASE_TABLE = "nfl_game_schedule"
ET_ZONE        = ZoneInfo("America/New_York")

# ── Date Range (inclusive) ──────────────────────────────────────────────────
START_DATE = date(2024, 12, 11)
END_DATE   = date(2024, 12, 15)

def date_range(start: date, end: date) -> List[str]:
    """Generate ISO dates from start to end inclusive."""
    days = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(days + 1)]

def api_get(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"API error {url} {params}: {e}")
        return None

def fetch_games_for_date(date_iso: str) -> List[Dict[str, Any]]:
    """Fetch games for a given date (all statuses)."""
    resp = api_get("games", {
        "league": NFL_LEAGUE_ID,
        "date": date_iso,
        "timezone": "America/New_York"
    })
    return resp.get("response", []) if resp and isinstance(resp.get("response"), list) else []

def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map API record to Supabase row schema."""
    g = rec.get("game", {})
    teams = rec.get("teams", {})
    gid = g.get("id")
    date_info = g.get("date", {})
    game_date = date_info.get("date")
    ts = date_info.get("timestamp")
    scheduled_iso = (
        datetime.fromtimestamp(int(ts), tz=ET_ZONE).isoformat()
        if ts else None
    )
    venue = g.get("venue", {}).get("name")
    status = g.get("status", {}).get("short")

    row = {
        "game_id":       gid,
        "game_date":     game_date,
        "home_team":     teams.get("home", {}).get("name"),
        "away_team":     teams.get("away", {}).get("name"),
        "home_team_id":  teams.get("home", {}).get("id"),
        "away_team_id":  teams.get("away", {}).get("id"),
        "scheduled_time": scheduled_iso,
        "venue":         venue,
        "status":        status,
    }
    # Ensure required fields are present
    if None in (gid, game_date, row["home_team"], row["away_team"], row["home_team_id"], row["away_team_id"], scheduled_iso, venue, status):
        return None
    return row

def run_backfill() -> None:
    records: List[Dict[str, Any]] = []
    for d in date_range(START_DATE, END_DATE):
        for rec in fetch_games_for_date(d):
            row = transform_game(rec)
            if row:
                records.append(row)
        time.sleep(1)  # throttle between days

    if not records:
        print("No games found for the given date range.")
        return

    print(f"Upserting {len(records)} games into `{SUPABASE_TABLE}`…")
    try:
        supabase.table(SUPABASE_TABLE).upsert(records, on_conflict="game_id").execute()
        print("✔ Upsert complete")
    except Exception as e:
        print(f"Supabase upsert error: {e}")

if __name__ == "__main__":
    run_backfill()
