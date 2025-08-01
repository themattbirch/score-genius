"""
Fetch and upsert **upcoming** NFL preseason games (next 7 days) into `nfl_preseason_schedule`, enriching venue info via local stadium_data.json lookup.

• For each of the next 7 calendar dates (ET) calls `/games?league=1&date=YYYY-MM-DD&timezone=America/New_York`.
• Keeps whatever games it returns, filtering only for valid IDs and team names.
• Loads local `stadium_data.json`, reads `st_data = data["NFL"]` for stadium info.
• Maps API fields into the Supabase schema:
    game_id, game_date, home_team, away_team, scheduled_time (ET ISO), stadium, city, latitude, longitude, is_indoor.
• Betting/prediction columns are left NULL.
• Upserts on `game_id` with a 5-second delay between date requests.
"""

from __future__ import annotations

import os
import sys
import time
import json
from datetime import date, timedelta, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import requests


# -----------------------------------------------------------------
# Add project root to Python path to allow for absolute imports
# -----------------------------------------------------------------
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir)) # Adds the main project folder

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Adds the project root to the path
# -----------------------------------------------------------------

from backend.nfl_features.make_nfl_preseason_snapshots import make_nfl_preseason_snapshot

# ── Credentials & Supabase Client Initialization ───────────────────────────
try:
    from config import API_SPORTS_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
    print("Using config.py for credentials")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

missing = [n for n, v in [
    ("API_SPORTS_KEY", API_SPORTS_KEY),
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
] if not v]
if missing:
    print(f"FATAL ERROR: Missing env vars: {', '.join(missing)}")
    sys.exit(1)

from caching.supabase_client import supabase

# ── Constants ──────────────────────────────────────────────
BASE_URL          = "https://v1.american-football.api-sports.io"
HEADERS           = {"x-apisports-key": API_SPORTS_KEY}
NFL_LEAGUE_ID     = 1
SUPABASE_TABLE    = "nfl_preseason_schedule"
DAYS_AHEAD        = 7
REQUEST_DELAY_SEC = 5
ET_ZONE           = ZoneInfo("America/New_York")

# ── Load Local Stadium Data ─────────────────────────────────
# Adjust path to where stadium_data.json lives relative to this script
BASE_DIR = os.path.dirname(__file__)
STADIUM_JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "stadium_data.json"))
try:
    with open(STADIUM_JSON_PATH) as f:
        STADIUM_DATA = json.load(f).get("NFL", {})
except Exception as e:
    print(f"Error loading stadium_data.json from {STADIUM_JSON_PATH}: {e}")
    STADIUM_DATA = {}

# ── Helpers ─────────────────────────────────────────────────

def api_get(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"API error {url} {params}: {e}")
        return None


def date_range_iso(days: int) -> List[str]:
    # Get the current time in the Eastern Time Zone
    today_et = datetime.now(ET_ZONE).date()
    print(f"Correctly determined today's date in Eastern Time as: {today_et.isoformat()}")
    return [(today_et + timedelta(days=i)).isoformat() for i in range(days)]


def fetch_games(date_iso: str) -> List[Dict[str, Any]]:
    payload = {"league": NFL_LEAGUE_ID, "date": date_iso, "timezone": "America/New_York"}
    data = api_get("games", payload)
    return data.get("response", []) if data and data.get("results") else []


def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    g = rec.get("game", {})
    gid = g.get("id")
    date_info = g.get("date", {})
    game_date = date_info.get("date")  # e.g., "2025-07-31"

    # scheduled_time
    ts = date_info.get("timestamp")
    if ts is not None:
        scheduled_iso = datetime.fromtimestamp(int(ts), tz=ET_ZONE).isoformat()
    else:
        t_str = date_info.get("time")
        offset = ET_ZONE.utcoffset(None)
        hours = int(offset.total_seconds() // 3600)
        scheduled_iso = f"{game_date}T{t_str}:00{hours:+03d}:00" if (game_date and t_str) else None

    teams = rec.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})

    home_name = home.get("name")
    away_name = away.get("name")
    home_id = home.get("id")
    away_id = away.get("id")

    # stadium lookup via home_team name
    stadium_info = STADIUM_DATA.get(home_name or "", {})
    stadium = stadium_info.get("stadium")
    city = stadium_info.get("city")
    latitude = stadium_info.get("latitude")
    longitude = stadium_info.get("longitude")
    is_indoor = stadium_info.get("is_indoor", False)

    # season derivation (preseason uses calendar year)
    season = None
    if game_date:
        try:
            season = int(game_date.split("-")[0])
        except ValueError:
            season = None

    # attempt to get stage if available
    stage = g.get("stage")  # might be None if API omits it

    row = {
        "game_id":         gid,
        "game_date":       game_date,
        "home_team":       home_name,
        "away_team":       away_name,
        "home_team_id":    home_id,
        "away_team_id":    away_id,
        "scheduled_time":  scheduled_iso,
        "stadium":         stadium,
        "city":            city,
        "latitude":        latitude,
        "longitude":       longitude,
        "is_indoor":       is_indoor,
        "status":          g.get("status", {}).get("short", ""),
        "season":          season,
        "stage":           stage,
    }

    if not (gid and home_name and away_name):
        return None

    return row

# ── Runner ─────────────────────────────────────────────────

def run_once() -> None:
    records: List[Dict[str, Any]] = []
    for d in date_range_iso(DAYS_AHEAD):
        print(f"Fetching games for {d}...")
        for rec in fetch_games(d):
            row = transform_game(rec)
            if row:
                records.append(row)
        time.sleep(REQUEST_DELAY_SEC)

    if not records:
        print(f"No upcoming NFL preseason games found in next {DAYS_AHEAD} days.")
        return

    # --- Step 1: Upsert the schedule data (your existing logic) ---
    print(f"Upserting {len(records)} preseason games → Supabase…")
    try:
        supabase.table(SUPABASE_TABLE).upsert(records, on_conflict="game_id").execute()
        print("✔ Schedule upsert complete.")
    except Exception as e:
        print(f"Supabase upsert error: {e}")
        # We still continue to try and generate snapshots for any existing data
    
    # --- Step 2: Generate a snapshot for each game found ---
    print(f"\nGenerating snapshots for {len(records)} games...")
    success_count = 0
    failure_count = 0
    for record in records:
        game_id = record.get("game_id")
        if not game_id:
            continue
        
        try:
            make_nfl_preseason_snapshot(str(game_id))
            success_count += 1
        except Exception as e:
            print(f"❌ Error generating snapshot for NFL game {game_id}: {e}")
            failure_count += 1
            
    print("\n--- NFL Preseason Pipeline Finished ---")
    print(f"Successfully generated snapshots: {success_count}")
    print(f"Failed to generate snapshots: {failure_count}")


if __name__ == "__main__":
    run_once()
