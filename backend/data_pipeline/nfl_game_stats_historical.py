# backend/data_pipeline/nfl_game_stats_historical.py

"""
Fetch and upsert historical NFL game stats from API‑Sports to Supabase.
Crawls by calendar date (one `/games?date=YYYY‑MM‑DD` request per day) for
seasons 2021‑2024 and inserts/upserts into `nfl_historical_game_stats`.

**Important**: Omits any JSON dump fields to match current schema.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

import requests

# ── Credentials ──────────────────────────────────────────────
try:
    from config import API_SPORTS_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
    print("Using config.py for credentials")
except ImportError:
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

missing = [n for n, v in [
    ("API_SPORTS_KEY", API_SPORTS_KEY),
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
] if not v]
if missing:
    print("FATAL: missing env vars → " + ", ".join(missing))
    sys.exit(1)

from caching.supabase_client import supabase

# ── Constants ────────────────────────────────────────────────
BASE_URL          = "https://v1.american-football.api-sports.io"
HEADERS           = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID     = 1
SUPABASE_TABLE    = "nfl_historical_game_stats"
REQUEST_DELAY_SEC = 5   # reduced to 5 seconds per request

# Season → (start_date, end_date)
SEASON_RANGE: Dict[int, tuple[str, str]] = {
    2021: ("2021-09-01", "2022-02-28"),
    2022: ("2022-09-01", "2023-02-28"),
    2023: ("2023-09-01", "2024-02-28"),
    2024: ("2024-09-01", "2025-02-28"),  # through Feb 2025
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_hist_fetch")

# ── Helpers – API ────────────────────────────────────────────

def api_get(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning(f"API error {url} params={params}: {e}")
        if e.response is not None:
            log.debug(e.response.text[:300])
        return None


def generate_dates(start: str, end: str) -> Generator[str, None, None]:
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    cur     = start_d
    while cur <= end_d:
        yield cur.isoformat()
        cur += timedelta(days=1)

# ── Core fetch/transform ────────────────────────────────────

def fetch_games_by_date(d: str) -> List[Dict[str, Any]]:
    data = api_get("games", {"date": d, "league": NFL_LEAGUE_ID})
    if not data or data.get("results", 0) == 0:
        return []
    return data.get("response", [])


def transform(g: Dict[str, Any]) -> Dict[str, Any]:
    core = g.get("game") or g.get("fixture") or g
    league = g.get("league", {})
    season_val = league.get("season") or league.get("year")
    venue = core.get("venue") or g.get("venue", {})
    status = core.get("status") or g.get("status", {})
    teams  = g.get("teams", {})
    scores = g.get("scores", {})

    # Parse date/time
    raw_date = core.get("date")
    game_date = game_time = None
    epoch_ts = None
    if isinstance(raw_date, dict):
        game_date = raw_date.get("date")
        game_time = raw_date.get("time")
        epoch_ts  = raw_date.get("timestamp")
    elif isinstance(raw_date, str):
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            game_date = dt.date().isoformat()
            game_time = dt.time().strftime("%H:%M")
            epoch_ts  = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass

    return {
        "game_id":        core.get("id"),
        "season":         season_val,
        "stage":          core.get("stage") or core.get("round", {}).get("name"),
        "week":           core.get("week"),
        "game_date":      game_date,
        "game_time":      game_time,
        "game_timestamp": epoch_ts,
        "venue_name":     venue.get("name"),
        "venue_city":     venue.get("city"),
        # new team names
        "home_team_id":   teams.get("home", {}).get("id"),
        "home_team_name": teams.get("home", {}).get("name"),
        "away_team_id":   teams.get("away", {}).get("id"),
        "away_team_name": teams.get("away", {}).get("name"),
        # scores by quarter + OT
        "home_q1":        scores.get("home", {}).get("quarter_1"),
        "home_q2":        scores.get("home", {}).get("quarter_2"),
        "home_q3":        scores.get("home", {}).get("quarter_3"),
        "home_q4":        scores.get("home", {}).get("quarter_4"),
        "home_ot":        scores.get("home", {}).get("overtime"),
        "away_q1":        scores.get("away", {}).get("quarter_1"),
        "away_q2":        scores.get("away", {}).get("quarter_2"),
        "away_q3":        scores.get("away", {}).get("quarter_3"),
        "away_q4":        scores.get("away", {}).get("quarter_4"),
        "away_ot":        scores.get("away", {}).get("overtime"),
        # final scores
        "home_score":     scores.get("home", {}).get("total"),
        "away_score":     scores.get("away", {}).get("total"),
    }

# ── Supabase upsert ─────────────────────────────────────────

def upsert(rec: Dict[str, Any]):
    try:
        res = supabase.table(SUPABASE_TABLE).upsert(rec, on_conflict="game_id").execute()
        err = res.get("error") if isinstance(res, dict) else getattr(res, "error", None)
        if err:
            log.error(f"DB error {err}")
        else:
            log.info(f"✔ Upserted {rec['game_id']}")
    except Exception as e:
        log.error(f"Supabase exception {e}")

# ── Driver ──────────────────────────────────────────────────

def run_once():
    for season, (start, end) in SEASON_RANGE.items():
        log.info(f"Season {season}: {start} → {end}")
        for d in generate_dates(start, end):
            games = fetch_games_by_date(d)
            if not games:
                continue
            log.info(f"{d}: {len(games)} games")
            for g in games:
                rec = transform(g)
                upsert(rec)
                time.sleep(REQUEST_DELAY_SEC)

if __name__ == "__main__":
    while True:
        run_once()
