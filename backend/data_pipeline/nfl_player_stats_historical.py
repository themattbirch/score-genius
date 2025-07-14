# backend/data_pipeline/nfl_player_stats_historical.py

import os
import sys
import time
import json
import logging
import requests

from zoneinfo import ZoneInfo
from datetime import datetime

# Add backend root to path
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import API_SPORTS_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
from caching.supabase_client import supabase

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_player_stats")

# ── Constants ─────────────────────────────────────────────
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS  = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
SEASON   = 2024
LEAGUE   = 1  # NFL
RATE_LIMIT = 0.5  # seconds between requests

def fetch(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    log.info(f"GET {endpoint} params={params}")
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def upsert_player_stats(player_id: int, stats: dict):
    record = {
        "player_id": player_id,
        "player_name": stats["player"]["name"],
        "team_id": stats["teams"][0]["team"]["id"],
        "season": SEASON,
        "raw_api_response": stats,
        # map each stat group into flattened columns...
    }
    # upsert into your table
    supabase.table("nfl_historical_player_stats").upsert(
        record, on_conflict="player_id,season"
    ).execute()

def main():
    # 1) Get all teams from standings
    standings = fetch("standings", {"season": SEASON, "league": LEAGUE})
    team_ids = [e["team"]["id"] for e in standings.get("response", [])]

    for tid in team_ids:
        # 2) Get roster for team
        roster = fetch("players", {"team": tid, "season": SEASON}).get("response", [])
        for p in roster:
            pid = p.get("player", {}).get("id") or p.get("id")
            if not pid:
                continue

            # 3) Fetch season-long stats
            stats = fetch("players/statistics", {"id": pid, "season": SEASON})
            if stats.get("results", 0) > 0:
                upsert_player_stats(pid, stats["response"][0])
            else:
                log.warning(f"No stats for player {pid}")

            time.sleep(RATE_LIMIT)

if __name__ == "__main__":
    main()
