"""
Fetch and upsert **upcoming** NFL games (next 8 days) into `nfl_game_schedule`.

• For each of the next 8 calendar dates (ET) calls `/games?league=1&date=YYYY‑MM‑DD&timezone=America/New_York`.
• Keeps only games whose `status.short == "NS"` (Not Started).
• Maps API fields into the Supabase schema:
    game_id, game_date, home_team, away_team, scheduled_time (ET ISO), venue, status.
• Betting / prediction columns are left NULL.
• Upserts on `game_id` with a 5‑second delay between date requests.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta, datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import requests

# -----------------------------------------------------------------
# Add project root to Python path to allow for absolute imports
# -----------------------------------------------------------------
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------------

# 👇 NEW: Import the regular season snapshot function
from backend.nfl_features.make_nfl_snapshots import make_nfl_snapshot


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
HEADERS           = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID     = 1
SUPABASE_TABLE    = "nfl_game_schedule"
DAYS_AHEAD        = 8
REQUEST_DELAY_SEC = 5
ET_ZONE           = ZoneInfo("America/New_York")

# ── Helpers ────────────────────────────────────────────────

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
    today = date.today()
    return [(today + timedelta(days=i)).isoformat() for i in range(days)]


def fetch_games(date_iso: str) -> List[Dict[str, Any]]:
    payload = {"league": NFL_LEAGUE_ID, "date": date_iso, "timezone": "America/New_York"}
    data = api_get("games", payload)
    return data.get("response", []) if data and data.get("results") else []


def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    g = rec.get("game", {})
    if g.get("status", {}).get("short") != "NS":
        return None  # Only upcoming

    gid = g.get("id")
    date_info = g.get("date", {})
    game_date = date_info.get("date")

    ts = date_info.get("timestamp")
    scheduled_iso = None
    if ts is not None:
        scheduled_iso = datetime.fromtimestamp(int(ts), tz=ET_ZONE).isoformat()
    else:
        t_str = date_info.get("time")
        if game_date and t_str:
            scheduled_iso = f"{game_date}T{t_str}:00{ET_ZONE.utcoffset(None).total_seconds()//3600:+03.0f}:00"

    teams = rec.get("teams", {})
    row = {
        "game_id":        gid,
        "game_date":      game_date,
        "home_team":      teams.get("home", {}).get("name"),
        "away_team":      teams.get("away", {}).get("name"),
        "scheduled_time": scheduled_iso,
        "venue":          g.get("venue", {}).get("name"),
        "status":         "NS",
    }
    return row if all(row.values()) else None

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
        print(f"No upcoming NFL games found in next {DAYS_AHEAD} days.")
        return

    # --- Step 1: Upsert schedule data ---
    print(f"Upserting {len(records)} games → Supabase…")
    try:
        supabase.table(SUPABASE_TABLE).upsert(records, on_conflict="game_id").execute()
        print("✔ Schedule upsert complete.")
    except Exception as e:
        print(f"Supabase upsert error: {e}")
        # Continue anyway to attempt snapshot generation
    
    # --- Step 2: Generate a snapshot for each game ---
    print(f"\nGenerating snapshots for {len(records)} games...")
    success_count = 0
    failure_count = 0
    for record in records:
        game_id = record.get("game_id")
        if not game_id:
            continue
        
        try:
            make_nfl_snapshot(str(game_id))
            success_count += 1
        except Exception as e:
            print(f"❌ Error generating snapshot for NFL game {game_id}: {e}")
            failure_count += 1
            
    print("\n--- NFL Regular Season Pipeline Finished ---")
    print(f"Successfully generated snapshots: {success_count}")
    print(f"Failed to generate snapshots: {failure_count}")

if __name__ == "__main__":
    run_once()
