import json
import logging
from datetime import datetime, timezone
import requests
from backend.config import API_SPORTS_KEY
from caching.supabase_client import supabase

# ── Configuration ────────────────────────────────────────────
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_game_stats_upsert")

# ─────────────────────────────────────────────────────────────
# Helper ─ fetch any API‑Sports endpoint
# ─────────────────────────────────────────────────────────────

def fetch(endpoint: str, params: dict) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    log.debug(f"GET {url} params={params}")
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ─────────────────────────────────────────────────────────────
# Core ─ upsert the game into Supabase
# ─────────────────────────────────────────────────────────────

def upsert_game(g: dict):
    """Transform API response → record → upsert."""
    # API returns either a nested `game` dict or top‑level fields
    game = g.get("game") or g

    # ── Date / time / epoch ─────────────────────────────────
    game_date = None      # YYYY‑MM‑DD
    game_time = None      # HH:MM
    epoch_ts  = None      # bigint column in DB

    raw_date = game.get("date")

    if isinstance(raw_date, dict):
        # Modern structure
        game_date = raw_date.get("date")            # '2024-08-02'
        game_time = raw_date.get("time")            # '00:00'
        epoch_ts  = raw_date.get("timestamp")       # 1722556800
    else:
        # Legacy ISO string or similar
        iso_str = None
        for k in ("date", "date_start", "scheduled", "timestamp"):
            val = game.get(k)
            if isinstance(val, str):
                iso_str = val
                break
        if iso_str:
            try:
                dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                game_date = dt.date().isoformat()
                game_time = dt.time().strftime("%H:%M")
                epoch_ts  = int(dt.replace(tzinfo=timezone.utc).timestamp())
            except Exception as e:
                log.debug(f"ISO parse failed: {e}")

    # Final fallback: derive epoch from separate date+time if available
    if epoch_ts is None and game_date and game_time:
        try:
            dt = datetime.fromisoformat(f"{game_date}T{game_time}:00+00:00")
            epoch_ts = int(dt.timestamp())
        except Exception as e:
            log.debug(f"Epoch derivation failed: {e}")

    log.debug(f"Parsed → game_date={game_date}, game_time={game_time}, epoch_ts={epoch_ts}")

    teams   = g.get("teams", {})
    scores  = g.get("scores", {})
    venue   = game.get("venue", {})
    status  = game.get("status", {})

    record = {
        "game_id":        game.get("id"),
        "stage":          game.get("stage") or game.get("round", {}).get("name"),
        "week":           game.get("week"),
        "game_date":      game_date,
        "game_time":      game_time,
        "game_timestamp": epoch_ts,   # bigint expected by DB
        "venue_name":     venue.get("name"),
        "venue_city":     venue.get("city"),
        "status_short":   status.get("short"),
        "status_long":    status.get("long"),
        "home_team_id":   teams.get("home", {}).get("id"),
        "away_team_id":   teams.get("away", {}).get("id"),
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
        "home_score":     scores.get("home", {}).get("total"),
        "away_score":     scores.get("away", {}).get("total"),
    }

    log.debug(f"Record → {json.dumps(record, indent=2)}")

    res = (
        supabase.table("nfl_historical_game_stats")
        .upsert(record, on_conflict="game_id")
        .execute()
    )

    # Log the response for visibility
    log.info(f"Supabase response object: {res}")

    # Gracefully handle either dict‑like or attr‑based response
    res_err = None
    if isinstance(res, dict):
        res_err = res.get("error")
    else:
        res_err = getattr(res, "error", None)

    if res_err:
        log.error(f"Upsert error: {res_err}")
    else:
        log.info(f"✔ Upserted game {record['game_id']}")

# ─────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────

def main():
    GAME_ID = 13146
    data = fetch("games", {"id": GAME_ID})
    resp = data.get("response", [])
    if not resp:
        log.error("No game found 🙁")
        return
    upsert_game(resp[0])

if __name__ == "__main__":
    main()
