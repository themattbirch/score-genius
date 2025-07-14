"""
Fetch NFL season-long team standings (wins, losses, PF/PA, form, etc.)
and dump to backend/data_pipeline/raw_json/.
"""

import json
import logging
from pathlib import Path
import requests
from backend.config import API_SPORTS_KEY   # RapidAPI key you already use

BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS  = {
    "x-rapidapi-key":  API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}

SEASON  = 2024
LEAGUE  = 1   # NFL

OUT_DIR = Path(__file__).parent / "raw_json"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nfl_team_standings")

def fetch(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    log.info(f"GET {url} {params}")
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def dump(js: dict, fname: str):
    p = OUT_DIR / fname
    p.write_text(json.dumps(js, indent=2))
    log.info(f"Saved → {p.relative_to(Path.cwd())}")

if __name__ == "__main__":
    standings = fetch("standings", {"season": SEASON, "league": LEAGUE})
    dump(standings, f"raw_nfl_standings_{SEASON}.json")
    log.info("✓ NFL team-standings discovery complete")
