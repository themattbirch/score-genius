"""
Fetch and upsert **upcoming** NFL preseason games (next 7 days) into `nfl_preseason_schedule`.

• For each of the next 7 calendar dates (ET) calls `/games?league=1&date=YYYY-MM-DD&timezone=America/New_York`.
• Keeps only games whose `status.short == "NS"` (Not Started).
• Maps API fields into the Supabase schema:
    game_id, game_date, home_team, away_team, scheduled_time (ET ISO), venue, status.
• Betting/prediction columns are left NULL.
• Upserts on `game_id` with a 5‑second delay between date requests.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

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
SUPABASE_TABLE    = "nfl_preseason_schedule"
DAYS_AHEAD        = 7
REQUEST_DELAY_SEC = 5
ET_ZONE           = ZoneInfo("America/New_York")

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
    today = date.today()
    return [(today + timedelta(days=i)).isoformat() for i in range(days)]


def fetch_games(date_iso: str) -> List[Dict[str, Any]]:
    payload = {"league": NFL_LEAGUE_ID, "date": date_iso, "timezone": "America/New_York"}
    data = api_get("games", payload)
    return data.get("response", []) if data and data.get("results") else []


def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    g = rec.get("game", {})
    gid = g.get("id")

    date_info = g.get("date", {})
    game_date = date_info.get("date")

    ts = date_info.get("timestamp")
    if ts is not None:
        scheduled_iso = datetime.fromtimestamp(int(ts), tz=ET_ZONE).isoformat()
    else:
        t_str = date_info.get("time")
        offset = ET_ZONE.utcoffset(None)
        hours = int(offset.total_seconds() // 3600)
        scheduled_iso = f"{game_date}T{t_str}:00{hours:+03d}:00" if (game_date and t_str) else None

    teams = rec.get("teams", {})
    row = {
        "game_id":        gid,
        "game_date":      game_date,
        "home_team":      teams.get("home", {}).get("name"),
        "away_team":      teams.get("away", {}).get("name"),
        "scheduled_time": scheduled_iso,
        "venue":          g.get("venue", {}).get("name"),
        "status":         g.get("status", {}).get("short", "NS"),
    }

    return row if gid and row["home_team"] and row["away_team"] else None

# ── Runner ─────────────────────────────────────────────────

def run_once() -> None:
    records: List[Dict[str, Any]] = []
    for d in date_range_iso(DAYS_AHEAD):
        games = fetch_games(d)
        for rec in games:
            row = transform_game(rec)
            if row:
                records.append(row)
        time.sleep(REQUEST_DELAY_SEC)

    if not records:
        print(f"No upcoming NFL preseason games found in next {DAYS_AHEAD} days.")
        return

    print(f"Upserting {len(records)} preseason games → Supabase…")
    try:
        supabase.table(SUPABASE_TABLE).upsert(records, on_conflict="game_id").execute()
        print("✔ Upsert complete.")
    except Exception as e:
        print(f"Supabase upsert error: {e}")


if __name__ == "__main__":
    run_once()
