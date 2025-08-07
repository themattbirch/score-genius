# backend/data_pipeline/nfl_preseason_preview.py

"""
Fetch and upsert **upcoming** NFL preseason games (next 7 days) into `nfl_preseason_schedule`,
including betting odds from The Odds API, enriching venue info via local stadium_data.json lookup,
then generating preseason snapshots.
"""

from __future__ import annotations

import os
import sys
import time
import json
import requests
import re
from datetime import date, timedelta, datetime as dt_datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------
# Path setup for absolute imports
# -----------------------------------------------------------------
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------------

from backend.nfl_features.make_nfl_preseason_snapshots import make_nfl_preseason_snapshot

# --- Config / Credentials ---
try:
    from config import API_SPORTS_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, ODDS_API_KEY
    print("Using config.py for credentials")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")

missing = [
    name
    for name, val in [
        ("API_SPORTS_KEY", API_SPORTS_KEY),
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
        ("ODDS_API_KEY", ODDS_API_KEY),
    ]
    if not val
]
if missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(missing)}")
    sys.exit(1)

from caching.supabase_client import supabase

# --- Constants ---
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID = 1
SUPABASE_TABLE = "nfl_preseason_schedule"
DAYS_AHEAD = 7
REQUEST_DELAY_SEC = 5
ET_ZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Odds API
PREFERRED_BOOKMAKER_KEY = "draftkings"
ODDS_API_BASE = ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl_preseason/odds"

# --- Load Local Stadium Data ---
BASE_DIR = os.path.dirname(__file__)
STADIUM_JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "stadium_data.json"))
try:
    with open(STADIUM_JSON_PATH) as f:
        STADIUM_DATA = json.load(f).get("NFL", {})
except Exception as e:
    print(f"Error loading stadium_data.json from {STADIUM_JSON_PATH}: {e}")
    STADIUM_DATA = {}

# --- Helpers ---

def normalize_team_name_nfl(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    temp = name.strip().lower()
    mapping = {
        "nyg": "New York Giants",
        "nyj": "New York Jets",
        "ne": "New England Patriots",
        "gb": "Green Bay Packers",
        "kc": "Kansas City Chiefs",
        "sf": "San Francisco 49ers",
        "la": "Los Angeles Rams",
        "lac": "Los Angeles Chargers",
        "dal": "Dallas Cowboys",
        "phi": "Philadelphia Eagles",
        "sea": "Seattle Seahawks",
        "pit": "Pittsburgh Steelers",
        "den": "Denver Broncos",
        "lv": "Las Vegas Raiders",
        "ari": "Arizona Cardinals",
        "min": "Minnesota Vikings",
        "ten": "Tennessee Titans",
        "cin": "Cincinnati Bengals",
        "bal": "Baltimore Ravens",
        "hou": "Houston Texans",
        "jax": "Jacksonville Jaguars",
        "atl": "Atlanta Falcons",
        "car": "Carolina Panthers",
        "tb": "Tampa Bay Buccaneers",
        "no": "New Orleans Saints",
        "chi": "Chicago Bears",
        "det": "Detroit Lions",
        "ind": "Indianapolis Colts",
        "oak": "Las Vegas Raiders",
        "wash": "Washington Commanders",
    }
    if temp in mapping:
        return mapping[temp]
    return temp.title()

def title_case(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split())

def safe_float_conversion(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_american_odds(numeric_odds: Optional[float]) -> Optional[str]:
    if numeric_odds is None:
        return None
    odds_int = int(round(numeric_odds))
    return f"+{odds_int}" if odds_int > 0 else f"{odds_int}"

# NFL schedule API
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
    today_et = dt_datetime.now(ET_ZONE).date()
    return [(today_et + timedelta(days=i)).isoformat() for i in range(days)]

def fetch_preseason_games_for_date(date_iso: str) -> List[Dict[str, Any]]:
    payload = {"league": NFL_LEAGUE_ID, "date": date_iso, "timezone": "America/New_York"}
    data = api_get("games", payload)
    if not data:
        return []
    return data.get("response", []) if data.get("response") else []

def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    g = rec.get("game", {})
    gid = g.get("id")
    date_info = g.get("date", {}) or {}
    game_date = date_info.get("date")
    ts = date_info.get("timestamp")
    scheduled_iso = None
    if ts is not None:
        scheduled_iso = dt_datetime.fromtimestamp(int(ts), tz=ET_ZONE).isoformat()
    else:
        t_str = date_info.get("time")
        if game_date and t_str:
            offset = ET_ZONE.utcoffset(None)
            hours = int(offset.total_seconds() // 3600)
            scheduled_iso = f"{game_date}T{t_str}:00{hours:+03d}:00"
    teams = rec.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_name = home.get("name")
    away_name = away.get("name")
    home_id = home.get("id")
    away_id = away.get("id")

    # stadium lookup via home_team name
    stadium_info = STADIUM_DATA.get(home_name or "", {})
    stadium = stadium_info.get("stadium")
    city = stadium_info.get("city")
    latitude = stadium_info.get("latitude")
    longitude = stadium_info.get("longitude")
    is_indoor = stadium_info.get("is_indoor", False)

    season = None
    if game_date:
        try:
            season = int(game_date.split("-")[0])
        except ValueError:
            season = None

    stage = g.get("stage")

    row = {
        "game_id":         gid,
        "game_date":       game_date,
        "home_team":       home_name,
        "away_team":       away_name,
        "home_team_id":    home_id,
        "away_team_id":    away_id,
        "scheduled_time":  scheduled_iso,
        "venue_name":      stadium,
        "venue_city":      city,
        "stadium":         stadium,
        "city":            city,
        "latitude":        latitude,
        "longitude":       longitude,
        "is_indoor":       is_indoor,
        "status":          g.get("status", {}).get("short", ""),
        "season":          season,
        "stage":           stage,
    }

    if not (gid and home_name and away_name):
        return None
    return row

# Odds logic
def build_odds_time_window(days: int) -> tuple[str, str]:
    now_et = dt_datetime.now(ET_ZONE)
    start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et_date = (now_et.date() + timedelta(days=days - 1))
    end_et_dt = dt_datetime.combine(end_et_date, dt_datetime.max.time()).replace(tzinfo=ET_ZONE)
    start_utc = start_et.astimezone(UTC)
    end_utc = end_et_dt.astimezone(UTC)
    commence_time_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return commence_time_from, commence_time_to

def fetch_betting_odds(days_window: int) -> List[Dict[str, Any]]:
    if not ODDS_API_KEY:
        print("ODDS_API_KEY missing")
        return []
    c_from, c_to = build_odds_time_window(days_window)
    params = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": c_from,
        "commenceTimeTo": c_to,
        "apiKey": ODDS_API_KEY,
    }
    print(f"Fetching NFL preseason odds from The Odds API between {c_from} and {c_to}")
    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        print(f"Fetched {len(data)} odds events.")
        return data
    except requests.RequestException as e:
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        print(f"Error fetching odds: {e} (status: {status})")
        return []

def match_odds_for_game(game: Dict[str, Any], odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not odds_events or not game:
        return None
    home_norm = normalize_team_name_nfl(game.get("home_team", "") or "")
    away_norm = normalize_team_name_nfl(game.get("away_team", "") or "")

    scheduled = game.get("scheduled_time")
    try:
        game_dt = dt_datetime.fromisoformat(scheduled)
    except Exception:
        game_dt = None

    for odds_event in odds_events:
        odds_home = normalize_team_name_nfl(odds_event.get("home_team", ""))
        odds_away = normalize_team_name_nfl(odds_event.get("away_team", ""))
        if home_norm != odds_home or away_norm != odds_away:
            continue
        commence_time_str = odds_event.get("commence_time")
        if not commence_time_str:
            continue
        try:
            event_dt = dt_datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
            if game_dt:
                if event_dt.astimezone(ET_ZONE).date() != game_dt.astimezone(ET_ZONE).date():
                    continue
            print(f"Matched odds for {away_norm} @ {home_norm} on {event_dt.date()}")
            return odds_event
        except Exception:
            continue
    return None

def extract_odds_data(odds_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw_odds = {"moneyline": {}, "spread": {}, "total": {}}
    clean_odds = {
        "moneyline_home_clean": None,
        "moneyline_away_clean": None,
        "spread_home_line_clean": None,
        "spread_home_price_clean": None,
        "spread_away_price_clean": None,
        "total_line_clean": None,
        "total_over_price_clean": None,
        "total_under_price_clean": None,
    }
    if not odds_event:
        return {"raw": raw_odds, "clean": clean_odds}
    bookmakers = odds_event.get("bookmakers", [])
    if not bookmakers:
        return {"raw": raw_odds, "clean": clean_odds}

    target_bookmaker = next(
        (bm for bm in bookmakers if bm.get("key") == PREFERRED_BOOKMAKER_KEY), bookmakers[0]
    )
    if not target_bookmaker:
        return {"raw": raw_odds, "clean": clean_odds}

    raw_odds["bookmaker_key"] = target_bookmaker.get("key")
    home_team_title = title_case(normalize_team_name_nfl(odds_event.get("home_team", "")))
    away_team_title = title_case(normalize_team_name_nfl(odds_event.get("away_team", "")))

    markets = target_bookmaker.get("markets", [])
    for market in markets:
        mkey = market.get("key")
        outcomes = market.get("outcomes", []) or []
        if mkey == "h2h":
            raw_odds["moneyline"] = {
                title_case(o.get("name", "")): o.get("price")
                for o in outcomes
                if o.get("name") and o.get("price") is not None
            }
            for o in outcomes:
                team = title_case(o.get("name", ""))
                price = safe_float_conversion(o.get("price"))
                fmt = format_american_odds(price)
                if team == home_team_title:
                    clean_odds["moneyline_home_clean"] = fmt
                elif team == away_team_title:
                    clean_odds["moneyline_away_clean"] = fmt
        elif mkey == "spreads":
            raw_odds["spread"] = {
                title_case(o.get("name", "")): {"price": o.get("price"), "point": o.get("point")}
                for o in outcomes
                if o.get("name") and o.get("price") is not None and o.get("point") is not None
            }
            for o in outcomes:
                team = title_case(o.get("name", ""))
                price = safe_float_conversion(o.get("price"))
                point = safe_float_conversion(o.get("point"))
                fmt = format_american_odds(price)
                if team == home_team_title:
                    clean_odds["spread_home_line_clean"] = point
                    clean_odds["spread_home_price_clean"] = fmt
                elif team == away_team_title:
                    clean_odds["spread_away_price_clean"] = fmt
        elif mkey == "totals":
            raw_odds["total"] = {
                o.get("name", ""): {"price": o.get("price"), "point": o.get("point")}
                for o in outcomes
                if o.get("name") in ["Over", "Under"] and o.get("price") is not None and o.get("point") is not None
            }
            num_total_line = None
            for o in outcomes:
                num_total_line = safe_float_conversion(o.get("point"))
                break
            clean_odds["total_line_clean"] = num_total_line
            for o in outcomes:
                pos = o.get("name")
                price = safe_float_conversion(o.get("price"))
                fmt = format_american_odds(price)
                if pos == "Over":
                    clean_odds["total_over_price_clean"] = fmt
                elif pos == "Under":
                    clean_odds["total_under_price_clean"] = fmt
    return {"raw": raw_odds, "clean": clean_odds}

# --- Main runner ---
def build_and_upsert_preseason_nfl_previews() -> int:
    print("\n--- Running NFL Preseason Game Preview Script (Schedule + Odds) ---")
    all_dates = date_range_iso(DAYS_AHEAD)
    records: List[Dict[str, Any]] = []

    for d in all_dates:
        print(f"Fetching preseason games for {d}...")
        games = fetch_preseason_games_for_date(d)
        for rec in games:
            transformed = transform_game(rec)
            if transformed:
                records.append(transformed)
        time.sleep(REQUEST_DELAY_SEC)

    if not records:
        print(f"No upcoming NFL preseason games found in next {DAYS_AHEAD} days.")
        return 0

    # fetch odds
    odds_events = fetch_betting_odds(DAYS_AHEAD)

    enriched: List[Dict[str, Any]] = []
    for row in records:
        odds_event = match_odds_for_game(row, odds_events)
        extracted = extract_odds_data(odds_event)

        # consolidate clean structures
        moneyline_clean = {
            "home": extracted["clean"].get("moneyline_home_clean"),
            "away": extracted["clean"].get("moneyline_away_clean"),
        }
        spread_clean = {
            "home": {
                "line": extracted["clean"].get("spread_home_line_clean"),
                "price": extracted["clean"].get("spread_home_price_clean"),
            },
            "away": {
                "line": None,
                "price": extracted["clean"].get("spread_away_price_clean"),
            },
        }
        total_clean = {
            "line": extracted["clean"].get("total_line_clean"),
            "over_price": extracted["clean"].get("total_over_price_clean"),
            "under_price": extracted["clean"].get("total_under_price_clean"),
        }

        merged = {
            **row,
            "moneyline": extracted["raw"].get("moneyline"),
            "spread": extracted["raw"].get("spread"),
            "total": extracted["raw"].get("total"),
            "moneyline_clean": moneyline_clean,
            "spread_clean": spread_clean,
            "total_clean": total_clean,
            "updated_at": dt_datetime.now(UTC).isoformat(),
        }
        enriched.append(merged)

    # upsert to Supabase
    print(f"\nStep: Upserting {len(enriched)} NFL preseason preview rows to Supabase...")
    try:
        supabase.table(SUPABASE_TABLE).upsert(enriched, on_conflict="game_id").execute()
        print("✔ Supabase upsert complete.")
    except Exception as e:
        print(f"Supabase upsert error: {e}")

    # snapshots
    print(f"\nGenerating preseason snapshots for {len(enriched)} games...")
    success = 0
    fail = 0
    for rec in enriched:
        gid = rec.get("game_id")
        if not gid:
            continue
        try:
            make_nfl_preseason_snapshot(str(gid))
            success += 1
        except Exception as e:
            print(f"❌ Snapshot error for NFL preseason game {gid}: {e}")
            fail += 1

    print(f"\n--- Completed NFL Preseason Preview: {len(enriched)} games, snapshots success={success}, fail={fail} ---")
    return len(enriched)

def run_once():
    build_and_upsert_preseason_nfl_previews()

if __name__ == "__main__":
    run_once()
