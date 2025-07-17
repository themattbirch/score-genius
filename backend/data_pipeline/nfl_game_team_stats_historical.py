# backend/data_pipeline/nfl_game_team_stats_historical.py

"""
Fetch and upsert historical NFL **team** statistics for every game from
API‑Sports into the `nfl_historical_game_team_stats` Supabase table.

• Crawls by calendar date (same season windows as game‑level script)
• For each game ID found, hits `/games/statistics/teams?id=<game_id>`
• Flattens and transforms the JSON using the FIELD_MAPPING defined below
• Splits hyphen‑encoded metrics into attempts/made/pct pairs
• Preserves hyphenated metrics `passing_comp_att` and `passing_sacks_yards_lost`
• Records the full `raw_api_response` JSON for debugging
• Upserts one row per team (`on_conflict = game_id, team_id`)
• 5‑second delay between API requests for rate‑limit safety
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

import requests

# ── Credentials & Supabase Client Initialization ───────────────────────────────────
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
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(missing)}")
    sys.exit(1)

from caching.supabase_client import supabase

# ── Constants ──────────────────────────────────────────────
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

# ── Constants ────────────────────────────────────────────────
BASE_URL          = "https://v1.american-football.api-sports.io"
HEADERS           = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID     = 1
SUPABASE_TABLE    = "nfl_historical_game_team_stats"
REQUEST_DELAY_SEC = 2

SEASON_RANGE: Dict[int, tuple[str, str]] = {
    2021: ("2021-09-01", "2022-02-28"),
    2022: ("2022-09-01", "2023-02-28"),
    2023: ("2023-09-01", "2024-02-28"),
    2024: ("2024-09-01", "2025-02-28"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_team_stats_hist")

# ── Mapping configuration ───────────────────────────────────
FIELD_MAPPING = {
    "first_downs.total":                 "first_downs_total",
    "first_downs.passing":               "first_downs_passing",
    "first_downs.rushing":               "first_downs_rushing",
    "first_downs.from_penalties":        "first_downs_penalty",
    "first_downs.third_down_efficiency": "third_down_efficiency",
    "first_downs.fourth_down_efficiency":"fourth_down_efficiency",
    "plays.total":                       "plays_total",
    "yards.total":                       "yards_total",
    "yards.yards_per_play":              "yards_per_play",
    "yards.total_drives":                "total_drives",
    "passing.total":                     "passing_total",
    "passing.comp_att":                  "passing_comp_att",
    "passing.yards_per_pass":            "passing_yards_per_pass",
    "passing.interceptions_thrown":      "passing_interceptions",
    "passing.sacks_yards_lost":          "passing_sacks_yards_lost",
    "rushings.total":                    "rushings_total",
    "rushings.attempts":                 "rushings_attempts",
    "rushings.yards_per_rush":           "rushings_yards_per_rush",
    "red_zone.made_att":                 "red_zone_made_att",
    "penalties.total":                   "penalties_yards",
    "turnovers.total":                   "turnovers_total",
    "turnovers.lost_fumbles":            "turnovers_lost_fumbles",
    "turnovers.interceptions":           "turnovers_interceptions",
    "interceptions.total":               "interceptions_total",
    "fumbles_recovered.total":           "fumbles_recovered",
    "sacks.total":                       "sacks_total",
    "safeties.total":                    "safeties_total",
    "int_touchdowns.total":              "int_touchdowns_total",
    "points_against.total":              "points_against_total",
    "posession.total":                   "possession_time",  # API typo
}

TEXT_COLUMNS = {"red_zone_made_att", "possession_time"}
HYPHEN_COLUMNS = {"passing_comp_att", "passing_sacks_yards_lost"}

# ── Helpers ─────────────────────────────────────────────────
def api_get(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"API error {endpoint} {params}: {e}")
        return None


def generate_dates(start: str, end: str) -> Generator[str, None, None]:
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= end_d:
        yield cur.isoformat()
        cur += timedelta(days=1)


def get_nested(d: dict, path: str, default=None):
    for key in path.split('.'):
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def parse_value(val: Any, col_name: str) -> Any:
    if isinstance(val, str):
        if col_name in HYPHEN_COLUMNS and '-' in val:
            return val
        try:
            f = float(val)
            return int(f) if f.is_integer() else f
        except ValueError:
            return val
    return val


def fetch_games_by_date(d: str) -> List[Dict[str, Any]]:
    data = api_get("games", {"date": d, "league": NFL_LEAGUE_ID})
    if not data or data.get("results", 0) == 0:
        return []
    return data.get("response", [])


def fetch_team_stats(game_id: int) -> Optional[Dict[str, Any]]:
    return api_get("games/statistics/teams", {"id": game_id})


def split_pair(pair: str) -> tuple[int, int]:
    a, b = pair.split('-', 1)
    return int(a), int(b)


def upsert_game_team_stats(stats_json: dict, game_id: int, season: int, teams: dict) -> None:
    for rec_json in stats_json.get("response", []):
        team = rec_json.get("team", {})
        stats = rec_json.get("statistics", {})

        record: Dict[str, Any] = {
            "game_id":          game_id,
            "team_id":          team.get("id"),
            "season":           season,
            "raw_api_response": rec_json,
            "home_team_id":     teams.get("home", {}).get("id"),
            "home_team_name":   teams.get("home", {}).get("name"),
            "away_team_id":     teams.get("away", {}).get("id"),
            "away_team_name":   teams.get("away", {}).get("name"),
        }

        for path, col in FIELD_MAPPING.items():
            raw = get_nested(stats, path)
            if raw is None:
                raw = "" if col in TEXT_COLUMNS else 0
            record[col] = parse_value(raw, col)

        splits = [
            ("third_down_efficiency", "third_down_made", "third_down_attempts", "third_down_pct"),
            ("fourth_down_efficiency", "fourth_down_made", "fourth_down_attempts", "fourth_down_pct"),
            ("penalties_yards", "penalties", "penalty_yards", None),
            ("red_zone_made_att", "red_zone_made", "red_zone_att", "red_zone_pct"),
        ]
        for raw_key, made_col, att_col, pct_col in splits:
            raw_val = record.pop(raw_key, None)
            if isinstance(raw_val, str) and '-' in raw_val:
                made, att = split_pair(raw_val)
            else:
                made = int(raw_val) if isinstance(raw_val, (int, str)) and str(raw_val).isdigit() else 0
                att = 0
            record[made_col] = made
            record[att_col] = att
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            if pct_col:
                record[pct_col] = round(made/att, 3) if att else None

        for k, v in list(record.items()):
            if isinstance(v, str) and '-' in v and k not in HYPHEN_COLUMNS:
                del record[k]

        try:
            supabase.table(SUPABASE_TABLE).upsert(record, on_conflict="game_id,team_id").execute()            
            log.info(f"✔ Upserted game {game_id}, team {team.get('id')}" )
        except Exception as e:
            log.error(f"DB upsert error game {game_id}, team {team.get('id')}: {e}")


def run_once() -> None:
    for season, (start, end) in SEASON_RANGE.items():
        log.info(f"Season {season}: {start} → {end}")
        for d in generate_dates(start, end):
            games = fetch_games_by_date(d)
            if not games:
                continue
            log.info(f"{d}: {len(games)} games")
            teams_map = {g.get("game", {}).get("id"): g.get("teams", {}) for g in games}
            for g in games:
                gid = g.get("game", {}).get("id")
                if not gid:
                    continue
                stats = fetch_team_stats(gid)
                if not stats or not stats.get("results"): 
                    continue
                upsert_game_team_stats(stats, gid, season, teams_map.get(gid, {}))
                time.sleep(REQUEST_DELAY_SEC)

if __name__ == "__main__":
    while True:
        run_once()
