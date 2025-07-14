import os
import re
import logging
import requests
from collections import defaultdict

from config import RAPIDAPI_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
from caching.supabase_client import supabase

# ── Config ────────────────────────────────────────────────────────
BASE_URL    = "https://sports-information.p.rapidapi.com"
HEADERS     = {
    "x-rapidapi-host": "sports-information.p.rapidapi.com",
    "x-rapidapi-key":  RAPIDAPI_KEY,
}
SEASON      = 2024
COWBOYS_ID  = 6
DAK_ID      = 2577417
TOTAL_WEEKS = 3    # <<< only weeks 1–3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_season_aggregator")


def fetch(path: str, params: dict = None):
    url = f"{BASE_URL}/{path}"
    log.debug(f"GET {url} params={params or {}}")
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def slugify(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^\w]", "_", s).lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_value(v):
    if isinstance(v, str):
        clean = v.replace(",", "")
        try:
            f = float(clean)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v
    return v


def collect_season_stats():
    agg = defaultdict(float)

    for week in range(1, 4):  # only weeks 1–3
        log.info(f"Fetching week {week} weekly-schedule…")
        try:
            week_games = fetch("nfl/weekly-schedule", {
                "year": SEASON,
                "week": week,
                "seasonType": 2,
            })
        except requests.HTTPError as e:
            log.warning(f"  → could not load week {week}: {e}")
            continue

        # week_games might be a list of strings or dicts
        for game in week_games:
            if isinstance(game, str):
                gid = game
            elif isinstance(game, dict):
                gid = game.get("id")
            else:
                continue

            if not gid:
                continue

            log.info(f"  → week {week}, game {gid}, fetching box-score…")
            try:
                box = fetch(f"nfl/box-score/{gid}")
            except requests.HTTPError as e:
                log.warning(f"    box-score {gid} failed: {e}")
                continue

            comps = box.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            # find the Cowboys side
            side = next(
                (t for t in comp.get("competitors", [])
                 if t["team"]["id"] == COWBOYS_ID),
                None
            )
            if not side:
                continue

            # accumulate Dak’s stats from each stat group
            for group in side.get("players", []):
                for stat_group in group.get("statistics", []):
                    keys = stat_group.get("keys", [])
                    for athlete in stat_group.get("athletes", []):
                        if athlete["athlete"]["id"] != DAK_ID:
                            continue
                        for key, raw in zip(keys, athlete.get("stats", [])):
                            col = slugify(key)
                            agg[col] += parse_value(raw)

    return agg

def upsert_season(agg):
    record = {
        "player_id":   DAK_ID,
        "season":      SEASON,
        "player_name": "Dak Prescott",
        "team_id":     COWBOYS_ID,
        "team_name":   "Dallas Cowboys",
    }
    record.update({
        k: int(v) if isinstance(v, float) and v.is_integer() else v
        for k, v in agg.items()
    })
    log.info("Upserting season aggregate → %s", record)
    supabase.table("nfl_historical_player_stats") \
             .upsert(record, on_conflict="player_id,season") \
             .execute()
    log.info("Done.")


if __name__ == "__main__":
    stats = collect_season_stats()
    upsert_season(stats)
