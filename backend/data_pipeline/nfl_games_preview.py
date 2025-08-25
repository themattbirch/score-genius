# backend/data_pipeline/nfl_games_preview.py

"""
Fetch and upsert **upcoming** NFL games (next 8 days) into `nfl_game_schedule`,
including betting odds from The Odds API, then generate snapshots.
"""

import os
import sys
import time
import json
import requests
import re
from datetime import date, timedelta, datetime as dt_datetime, timezone
from datetime import datetime as dt 
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

from backend.nfl_features.make_nfl_snapshots import make_nfl_snapshot

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

# existing Supabase wrapper used elsewhere
from caching.supabase_client import supabase

# --- Constants ---
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.american-football.api-sports.io",
}
NFL_LEAGUE_ID = 1
SUPABASE_TABLE = "nfl_game_schedule"
DAYS_AHEAD = 8
REQUEST_DELAY_SEC = 5
ET_ZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PREFERRED_BOOKMAKER_KEY = "draftkings"
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# --- Helpers: team name normalization for NFL ---
def normalize_team_name_nfl(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    temp = name.strip().lower()
    # common abbreviations to full names (expand as needed)
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
        "chi": "Chicago Bears",
        "oak": "Las Vegas Raiders",
        "wash": "Washington Commanders",
        # fallback: title case
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

# --- NFL API schedule helpers ---
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

def fetch_nfl_games_for_date(date_iso: str) -> List[Dict[str, Any]]:
    payload = {"league": NFL_LEAGUE_ID, "date": date_iso, "timezone": "America/New_York"}
    data = api_get("games", payload)
    if not data:
        return []
    # some endpoints use 'response' key; defensive check
    return data.get("response", []) if data.get("response") else []

def transform_game(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Expect similar structure to MLB version: teams under rec["teams"], game under rec["game"]
    g = rec.get("game", {})
    if g.get("status", {}).get("short") != "NS":
        return None  # only upcoming
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
            # fallback naive compose; offset from ET zone
            offset = ET_ZONE.utcoffset(None)
            hours = int(offset.total_seconds() // 3600)
            scheduled_iso = f"{game_date}T{t_str}:00{hours:+03d}:00"
    teams = rec.get("teams", {})
    home_team_raw = teams.get("home", {}).get("name")
    away_team_raw = teams.get("away", {}).get("name")
    row = {
        "game_id": gid,
        "game_date": game_date,
        "home_team": home_team_raw,
        "away_team": away_team_raw,
        "scheduled_time": scheduled_iso,
        "venue": g.get("venue", {}).get("name"),
        "status": "NS",
    }
    return row if all([row.get("game_id"), row.get("home_team"), row.get("away_team"), row.get("scheduled_time")]) else None

# --- Odds fetching and matching ---
def build_odds_time_window(days: int) -> tuple[str, str]:
    # Cover from start of today ET to end of last day ET, converted to UTC ISO
    now_et = dt_datetime.now(ET_ZONE)
    start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et = (now_et.date() + timedelta(days=days - 1))
    end_et_dt = dt_datetime.combine(end_et, dt_datetime.max.time()).replace(tzinfo=ET_ZONE)
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
    print(f"Fetching NFL odds from The Odds API between {c_from} and {c_to}")
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

    # parse game date/time in ET to compare to commence_time
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
        # optional: verify date proximity (same day)
        commence_time_str = odds_event.get("commence_time")
        if not commence_time_str:
            continue
        try:
            event_dt = dt_datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
            if game_dt:
                # compare dates in ET to avoid timezone drift
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
def build_and_upsert_nfl_previews() -> int:
    print("\n--- Running NFL Game Preview Script (Schedule + Odds) ---")
    all_dates = date_range_iso(DAYS_AHEAD)
    records: List[Dict[str, Any]] = []

    # fetch schedule
    for d in all_dates:
        print(f"Fetching games for {d}...")
        games = fetch_nfl_games_for_date(d)
        for rec in games:
            transformed = transform_game(rec)
            if transformed:
                records.append(transformed)
        time.sleep(REQUEST_DELAY_SEC)

    if not records:
        print("No upcoming NFL games found; exiting.")
        return 0

    # fetch odds covering the same window
    odds_events = fetch_betting_odds(DAYS_AHEAD)

    # merge odds into preview rows
    enriched: List[Dict[str, Any]] = []
    for row in records:
        odds_event = match_odds_for_game(row, odds_events)
        extracted = extract_odds_data(odds_event)

        # build consolidated clean fields
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
                "line": None,  # original script didn't extract away line separately
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
            # omit raw_api_response since it's not in schema
        }
        enriched.append(merged)

    # upsert to Supabase
    print(f"\nStep: Upserting {len(enriched)} NFL preview rows to Supabase...")
    try:
        supabase.table(SUPABASE_TABLE).upsert(enriched, on_conflict="game_id").execute()
        print("✔ Supabase upsert complete.")
        # run predictions (generate and upsert predicted_home_score / predicted_away_score)
        print("\nStep: Generating NFL predictions (score projections) ...")
        pred_ok = False
        try:
            # Preferred: direct import of function + explicit upsert
            from backend.nfl_score_prediction.prediction import (
                generate_predictions as _nfl_generate_predictions,
                upsert_score_predictions as _nfl_upsert_predictions,
                get_supabase_client as _nfl_get_sb_client,
            )
            preds = _nfl_generate_predictions(days_window=DAYS_AHEAD)
            if preds:
                sb_cli = _nfl_get_sb_client()
                if sb_cli:
                    _nfl_upsert_predictions(preds, sb_cli, debug=False)
                    pred_ok = True
                    print(f"✔ NFL predictions generated and upserted for {len(preds)} games.")
                else:
                    print("⚠ Could not init Supabase client for prediction upsert.")
            else:
                print("⚠ Prediction function returned no payload.")
        except Exception as e:
            print(f"Prediction function path failed: {e}")

        # Final fallback: invoke CLI entrypoint which handles upsert internally
        if not pred_ok:
            try:
                from backend.nfl_score_prediction import prediction as _nfl_pred_mod
                if hasattr(_nfl_pred_mod, "main"):
                    _nfl_pred_mod.main()  # this will upsert
                    pred_ok = True
                    print("✔ NFL predictions generated via main() (CLI path).")
            except Exception as e:
                print(f"Prediction CLI fallback failed: {e}")

        if not pred_ok:
            print("⚠ NFL predictions did not run; check prediction.py entrypoints.")


    except Exception as e:
        print(f"Supabase upsert error: {e}")

    # snapshots
    print(f"\nGenerating snapshots for {len(enriched)} games...")
    success = 0
    fail = 0
    for rec in enriched:
        gid = rec.get("game_id")
        if not gid:
            continue
        try:
            make_nfl_snapshot(str(gid))
            success += 1
        except Exception as e:
            print(f"❌ Snapshot error for NFL game {gid}: {e}")
            fail += 1

    print(f"\n--- Completed NFL Preview: {len(enriched)} games, snapshots success={success}, fail={fail} ---")
    return len(enriched)

def run_once():
    build_and_upsert_nfl_previews()

if __name__ == "__main__":
    run_once()