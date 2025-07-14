"""
NFL Player Discovery – dump rosters and season-long stats for 2024 and upsert into Supabase.
"""
import json
import logging
import re
import time
from pathlib import Path

import requests
from backend.config import API_SPORTS_KEY
from caching.supabase_client import supabase

# ── Configuration ────────────────────────────────────────────
BASE_URL   = "https://v1.american-football.api-sports.io"
HEADERS    = {
    "x-apisports-key":  API_SPORTS_KEY,
    "x-rapidapi-host":  "v1.american-football.api-sports.io",
}
SEASON     = 2024
LEAGUE     = 1   # NFL
RATE_LIMIT = 0.5 # seconds between requests

# ── Output Directory ─────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "raw_json"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("nfl_player_discovery")


def fetch(endpoint: str, params: dict) -> dict:
    """GET from API and return JSON"""
    url = f"{BASE_URL}/{endpoint}"
    log.info(f"GET {endpoint} params={params}")
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def parse_value(val):
    """Convert numeric strings (including those with commas) to int/float, else return original."""
    if isinstance(val, str):
        # Remove thousands separators
        clean = val.replace(",", "")
        try:
            f = float(clean)
            return int(f) if f.is_integer() else f
        except ValueError:
            return val
    return val


def slugify(name: str) -> str:
    """Turn arbitrary stat names into safe column identifiers."""
    name = name.lower()
    # fix common API typos
    name = name.replace("touchdows", "touchdowns")
    # remove anything not alphanumeric or whitespace
    name = re.sub(r"[^\w\s]", "", name)
    # collapse whitespace and replace with underscore
    return re.sub(r"\s+", "_", name).strip("_")


def upsert_player_stats(stats_json: dict):
    """Flatten stats JSON and upsert into Supabase"""
    for rec in stats_json.get("response", []):
        player = rec.get("player", {})
        teams  = rec.get("teams", [])
        if not teams:
            continue
        team_entry = teams[0]
        team = team_entry.get("team", {})

        # Prepare record
        record = {
            "player_id":        player.get("id"),
            "player_name":      player.get("name"),
            "team_id":          team.get("id"),
            "team_name":        team.get("name"),
            "season":           SEASON,
            "raw_api_response": rec,
        }

        # Flatten each group of stats
        for group in team_entry.get("groups", []):
            for stat in group.get("statistics", []):
              raw = stat.get("value", "")
              # skip any "made-attempted" stats
              if isinstance(raw, str) and "-" in raw:
                  continue

              col = stat["name"].lower().replace(" ", "_")
              col = col.replace("touchdows", "touchdowns")
              record[col] = parse_value(raw)

        try:
            supabase.table("nfl_historical_player_stats") \
                     .upsert(record, on_conflict="player_id,season") \
                     .execute()
            log.info(f"Upserted player stats: {player.get('id')}")
        except Exception as e:
            log.error(f"Failed upsert for player {player.get('id')}: {e}")


def main():
    # 1) Fetch teams for season
    teams_data = fetch("teams", {"league": LEAGUE, "season": SEASON})
    teams = teams_data.get("response", [])

    # 2) For each team: dump roster, then fetch+upsert player stats
    for t in teams:
        tid = t.get("team", {}).get("id") or t.get("id")
        if not tid:
            continue

        # 2a) Roster
        roster = fetch("players", {"team": tid, "season": SEASON})

        # 2b) Season stats per player
        for p in roster.get("response", []):
            pid = p.get("player", {}).get("id") or p.get("id")
            if not pid:
                continue

            stats = fetch("players/statistics", {"id": pid, "season": SEASON})
            if stats.get("results", 0) > 0:
                upsert_player_stats(stats)
            time.sleep(RATE_LIMIT)

    log.info("✓ NFL player discovery + upsert complete.")


if __name__ == "__main__":
    main()
