"""
NFL discovery – now fetching season‐long team & player stats via /statistics.
Dumps JSON into backend/data_pipeline/raw_json/.
"""

import json
import logging
from pathlib import Path

import requests
from backend.config import API_SPORTS_KEY  # your RapidAPI key

# ── Configuration ────────────────────────────────────────────
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS  = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}

SEASON = 2024
LEAGUE = 1  # NFL

# ── Output Directory ─────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "raw_json"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nfl_discovery")


def fetch(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    log.info(f"GET {endpoint}  params={params}")
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def dump(obj: dict, name: str):
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    log.info(f"Saved → {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    # 1) roster of teams
    teams = fetch("teams", {"league": LEAGUE, "season": SEASON}).get("response", [])
    dump({"response": teams}, f"raw_nfl_teams_{SEASON}.json")

    # 2) season‐long team stats
    for t in teams:
        tid = t.get("team", {}).get("id") or t.get("id")
        if not tid:
            continue

        stats = fetch(
            "statistics",
            {"league": LEAGUE, "season": SEASON, "team": tid}
        )
        dump(stats, f"raw_nfl_team_stats_{tid}_{SEASON}.json")

    # 3) roster + player season stats per team
    for t in teams:
        tid = t.get("team", {}).get("id") or t.get("id")
        if not tid:
            continue

        roster = fetch("players", {"team": tid, "season": SEASON})
        dump(roster, f"raw_nfl_players_team{tid}_{SEASON}.json")

        for p in roster.get("response", []):
            pid = p.get("player", {}).get("id") or p.get("id")
            if not pid:
                continue

            stats = fetch(
                "statistics",
                {"league": LEAGUE, "season": SEASON, "player": pid}
            )
            dump(stats, f"raw_nfl_player_stats_{pid}_{SEASON}.json")

    log.info("✓ Discovery complete")
