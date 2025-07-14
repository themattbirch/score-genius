import json
import sys
import logging
import time
from pathlib import Path

import requests
from backend.config import API_SPORTS_KEY
from caching.supabase_client import supabase

# ── Configuration ────────────────────────────────────────────
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
SEASON = 2024
LEAGUE = 1           # NFL league id
RATE_LIMIT = 0.5     # seconds between requests

# ── Paths ───────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RAW_JSON_DIR = HERE / "raw_json"
RAW_JSON_DIR.mkdir(exist_ok=True)
GAMES_FILE = RAW_JSON_DIR / f"raw_nfl_games_{SEASON}.json"

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("nfl_game_team_stats_discovery")
log.setLevel(logging.DEBUG)

# ── JSON → COLUMN mapping ────────────────────────────────────
# Keys are the JSON paths under "statistics"; values are your DB columns.
FIELD_MAPPING = {
    # First Downs
    "first_downs.total":                 "first_downs_total",
    "first_downs.passing":               "first_downs_passing",
    "first_downs.rushing":               "first_downs_rushing",
    "first_downs.from_penalties":        "first_downs_penalty",
    # note: these two still split later
    "first_downs.third_down_efficiency": "third_down_efficiency",
    "first_downs.fourth_down_efficiency":"fourth_down_efficiency",

    # Plays & Yards
    "plays.total":                       "plays_total",
    "yards.total":                       "yards_total",
    "yards.yards_per_play":              "yards_per_play",
    "yards.total_drives":                "total_drives",

    # Passing
    "passing.total":                     "passing_total",
    "passing.comp_att":                  "passing_comp_att",
    "passing.yards_per_pass":            "passing_yards_per_pass",
    "passing.interceptions_thrown":      "passing_interceptions",
    "passing.sacks_yards_lost":          "passing_sacks_yards_lost",

    # Rushing
    "rushings.total":                    "rushings_total",
    "rushings.attempts":                 "rushings_attempts",
    "rushings.yards_per_rush":           "rushings_yards_per_rush",

    # Red Zone & Penalties
    "red_zone.made_att":                 "red_zone_made_att",
    "penalties.total":                   "penalties_yards",   # raw "3-15"

    # Turnovers & Scoring
    "turnovers.total":                   "turnovers_total",
    "turnovers.lost_fumbles":            "turnovers_lost_fumbles",
    "turnovers.interceptions":           "turnovers_interceptions",
    "interceptions.total":               "interceptions_total",
    "fumbles_recovered.total":           "fumbles_recovered",

    # Sacks, Safeties, Pick-sixes
    "sacks.total":                       "sacks_total",
    "safeties.total":                    "safeties_total",
    "int_touchdowns.total":              "int_touchdowns_total",

    # Points & Possession
    "points_against.total":              "points_against_total",
    "posession.total":                   "possession_time",   # note typo in API JSON
}

# Columns that must remain TEXT even if they look numeric
TEXT_COLUMNS = {
    "passing_comp_att",
    "passing_sacks_yards_lost",
    "yards_per_play",
    "red_zone_made_att",
    "possession_time",
}

# Hyphenated metrics should always be kept as text
HYPHEN_COLUMNS = {
    "passing_comp_att",
    "passing_sacks_yards_lost",
}

def fetch(endpoint: str, params: dict) -> dict:
    """GET from API-Sports and return parsed JSON."""
    url = f"{BASE_URL}/{endpoint}"
    log.info(f"GET {url} params={params}")
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def dump(data: dict, filename: str):
    path = RAW_JSON_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    log.info(f"Saved → {path.name}")

def get_nested(d: dict, path: str, default=None):
    """Traverse a nested dict by dot-separated path."""
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d

def parse_value(val, col_name):
    """
    Convert numeric strings to int/float, else return original.
    But for hyphenated metrics in HYPHEN_COLUMNS, keep as-is.
    """
    # supply defaults upstream, so val here is never None
    if isinstance(val, str):
        if col_name in HYPHEN_COLUMNS and "-" in val:
            return val
        try:
            f = float(val)
            return int(f) if f.is_integer() else f
        except ValueError:
            # non-numeric string
            return val
    return val  # already numeric

def upsert_stats(stats_json: dict, game_id: int):
    """Flatten, split hyphens, and upsert into Supabase."""
    for rec in stats_json.get("response", []):
        team  = rec.get("team", {})
        stats = rec.get("statistics", {})

        # 1) Base record
        record = {
            "game_id":          game_id,
            "team_id":          team.get("id"),
            "season":           SEASON,
            "raw_api_response": rec,
        }

        # 2) Map all the simple fields
        for json_path, col in FIELD_MAPPING.items():
            raw = get_nested(stats, json_path, None)
            if raw is None:
                raw = "" if col in TEXT_COLUMNS else 0
            record[col] = parse_value(raw, col)

        # 3) Split every remaining hyphen-pair into two ints + optional pct
        def split_pair(s: str):
            a, b = (int(x) for x in s.split("-", 1))
            return a, b

        # fields to split: (raw_key, col_a, col_b, pct_col_or_None)
        to_split = [
            ("third_down_efficiency",  "third_down_made",    "third_down_attempts",  "third_down_pct"),
            ("fourth_down_efficiency", "fourth_down_made",   "fourth_down_attempts", "fourth_down_pct"),
            ("penalties_yards",        "penalties",          "penalty_yards",       None),
            ("red_zone_made_att",      "red_zone_made",      "red_zone_att",        "red_zone_pct"),
        ]
        for raw_key, col_a, col_b, pct_col in to_split:
            raw_val = record.pop(raw_key, None)
            if isinstance(raw_val, str) and "-" in raw_val:
                a, b = split_pair(raw_val)
            else:
                a = int(raw_val) if isinstance(raw_val, (int, str)) and str(raw_val).isdigit() else 0
                b = 0
            record[col_a] = a
            record[col_b] = b
            if pct_col:
                record[pct_col] = round(a / b, 3) if b else None

        # 4) Final safety + debug
        # Remove anything still with a dash (just in case)
        for k, v in list(record.items()):
            if isinstance(v, str) and "-" in v:
                record.pop(k)

        log.debug("🧩 Final record keys: %s", sorted(record.keys()))

        # 5) Upsert
        try:
            supabase.table("nfl_historical_game_team_stats") \
                    .upsert(record, on_conflict="game_id,team_id") \
                    .execute()
            log.info(f"✔ Upserted game {game_id}, team {team.get('id')}")
        except Exception as e:
            log.error(f"✖ Failed upsert for game {game_id}, team {team.get('id')}: {e}")

def fetch_game_team_stats(game_id: int) -> dict:
    return fetch("games/statistics/teams", {"id": game_id})

def main():
    # fetch or load games list
    if not GAMES_FILE.exists():
        log.info(f"No games file; fetching season {SEASON} list…")
        data = fetch("games", {"season": SEASON, "league": LEAGUE})
        dump(data, GAMES_FILE.name)
    else:
        data = json.loads(GAMES_FILE.read_text())

    games = data.get("response", [])
    log.info(f"→ {len(games)} games found for season {SEASON}")

    for game in games:              # fetch only the first game for now
        gid = game.get("game", {}).get("id")
        if not gid:
            continue

        stats = fetch_game_team_stats(gid)
        dump(stats, f"raw_nfl_game_team_stats_{gid}_{SEASON}.json")

        if stats.get("results", 0) > 0:
            upsert_stats(stats, gid)
        time.sleep(RATE_LIMIT)

    log.info("✓ NFL game team stats discovery + upsert complete.")

if __name__ == "__main__":
    main()
