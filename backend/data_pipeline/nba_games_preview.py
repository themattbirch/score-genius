# backend/data_pipeline/nba_games_preview.py

"""
Fetch upcoming NBA games, enrich with betting odds, store in Supabase,
and generate score predictions.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys

import difflib
import requests
from zoneinfo import ZoneInfo

# --- Local Config & Variables (same pattern as MLB script) ---
try:
    from config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )
    print("Using config.py for credentials")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate
_missing = [ name for name,val in [
    ("API_SPORTS_KEY",       API_SPORTS_KEY),
    ("ODDS_API_KEY",         ODDS_API_KEY),
    ("SUPABASE_URL",         SUPABASE_URL),
    ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
] if not val ]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    exit(1)

HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from caching.supabase_client import supabase
from nba_score_prediction.prediction import (
    DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
    DEFAULT_UPCOMING_DAYS_WINDOW,
    generate_predictions,
    upsert_score_predictions,
)

# --- Constants ---
MODELS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "models" / "saved"
)
NBA_SPORTS_API = "https://v1.basketball.api-sports.io"
HEADERS_SPORTS = {
    "x-rapidapi-key": API_SPORTS_KEY,
    "x-rapidapi-host": "v1.basketball.api-sports.io",
}
NBA_ODDS_API = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ET_ZONE = ZoneInfo("America/New_York")


def normalize_team_name(name: str) -> str:
    return " ".join(name.split()).lower()


def title_case_team_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split())


def make_api_request(
    url: str, headers: Dict[str, str], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """GET and return JSON or None on error."""
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"API request error for {url} {params}: {e}")
        return None


def get_games_by_date(
    league: str, season: str, date: str, timezone: str = "America/New_York"
) -> Dict[str, Any]:
    """Fetch games for a specific date."""
    url = f"{NBA_SPORTS_API}/games"
    params = dict(league=league, season=season, date=date, timezone=timezone)
    return make_api_request(url, HEADERS_SPORTS, params) or {}


def get_betting_odds(et_date: datetime) -> List[Dict[str, Any]]:
    """Fetch odds for a given ET date from The Odds API."""
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not configured.")
        return []

    start_et = et_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et = et_date.replace(hour=23, minute=59, second=59, microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": start_et.astimezone(ZoneInfo("UTC")).strftime(fmt),
        "commenceTimeTo": end_et.astimezone(ZoneInfo("UTC")).strftime(fmt),
        "apiKey": ODDS_API_KEY,
    }

    try:
        resp = requests.get(NBA_ODDS_API, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as he:
        print(f"HTTP error fetching odds: {he} (Status {resp.status_code})")
    except requests.RequestException as re:
        print(f"Request error fetching odds: {re}")
    return []


def match_odds_for_game(
    game: Dict[str, Any], odds_events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Match one odds event to a game by team names and ET date."""
    if not odds_events:
        return None

    # Normalize game teams + date
    home = normalize_team_name(game["teams"]["home"]["name"])
    away = normalize_team_name(game["teams"]["away"]["name"])
    try:
        gdt = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
        gdate = gdt.astimezone(ET_ZONE).date()
    except Exception:
        return None

    # Exact match first
    for ev in odds_events:
        oh = normalize_team_name(ev.get("home_team", ""))
        oa = normalize_team_name(ev.get("away_team", ""))
        if home == oh and away == oa:
            return ev if _same_et_date(ev, gdate) else None

    # Fuzzy fallback
    for ev in odds_events:
        oh = normalize_team_name(ev.get("home_team", ""))
        oa = normalize_team_name(ev.get("away_team", ""))
        if (
            difflib.SequenceMatcher(None, home, oh).ratio() > 0.8
            and difflib.SequenceMatcher(None, away, oa).ratio() > 0.8
            and _same_et_date(ev, gdate)
        ):
            return ev
    return None


def _same_et_date(event: Dict[str, Any], gdate: Any) -> bool:
    """Helper: check if odds event matches a given ET date."""
    try:
        edt = datetime.fromisoformat(
            event["commence_time"].replace("Z", "+00:00")
        )
        return edt.astimezone(ET_ZONE).date() == gdate
    except Exception:
        return False


def extract_odds_by_market(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull moneyline, spread, and totals from a matched event (preferring DraftKings)."""
    result = {"moneyline": {}, "spread": {}, "total": {}}
    if not event:
        return result

    bms = event.get("bookmakers", [])
    bm = next((b for b in bms if b.get("key") == "draftkings"), bms[0]) if bms else {}
    for m in bm.get("markets", []):
        out = m.get("outcomes", [])
        if m.get("key") == "h2h":
            for o in out:
                result["moneyline"][title_case_team_name(o["name"])] = o.get("price")
        elif m.get("key") == "spreads":
            for o in out:
                name = title_case_team_name(o["name"])
                result["spread"][name] = {"point": o.get("point"), "price": o.get("price")}
        elif m.get("key") == "totals":
            for o in out:
                result["total"][o.get("name")] = {"point": o.get("point"), "price": o.get("price")}
    return result


def clear_old_games() -> None:
    """Delete any nba_game_schedule entries with ET date before today."""
    today = datetime.now(ET_ZONE).date()
    rows = supabase.table("nba_game_schedule").select("game_id,scheduled_time").execute().data or []
    to_delete = []

    for r in rows:
        try:
            sched = datetime.fromisoformat(r["scheduled_time"])
            if sched.astimezone(ET_ZONE).date() < today:
                to_delete.append(r["game_id"])
        except Exception:
            continue

    if to_delete:
        supabase.table("nba_game_schedule").delete().in_("game_id", to_delete).execute()
        print(f"Deleted {len(to_delete)} old games")


def build_game_preview(window_days: int = 2) -> List[Dict[str, Any]]:
    """Fetch pregame NBA schedule + odds and return a list of preview dicts."""
    now_et = datetime.now(ET_ZONE)
    all_games = []

    # 1) Get all NS/scheduled games in window
    for offset in range(window_days):
        dt = (now_et + timedelta(days=offset)).date().isoformat()
        data = get_games_by_date("12", "2024-2025", dt)
        games = data.get("response", [])
        pre = [
            g
            for g in games
            if g.get("status", {}).get("short") in ("NS",)
            or "scheduled" in (g["status"].get("long", "")).lower()
        ]
        all_games.extend(pre)

    if not all_games:
        return []

    # 2) Fetch odds once per day
    odds_list = []
    for offset in range(window_days):
        odds_list.extend(get_betting_odds(now_et + timedelta(days=offset)))

    # 3) Match & build previews
    previews = []
    for g in all_games:
        ev = match_odds_for_game(g, odds_list)
        odds = extract_odds_by_market(ev)
        venue = g.get("venue", {}).get("name") if isinstance(g.get("venue"), dict) else g.get("venue", "N/A")
        gd = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).astimezone(ET_ZONE).date().isoformat()

        previews.append(
            {
                "game_id": g["id"],
                "scheduled_time": g["date"],
                "game_date": gd,
                "venue": venue,
                "away_team": g["teams"]["away"]["name"],
                "home_team": g["teams"]["home"]["name"],
                **odds,
            }
        )

    return previews


def upsert_previews(previews: List[Dict[str, Any]]) -> None:
    """Upsert preview records into Supabase by game_id."""
    if not previews:
        return
    supabase.table("nba_game_schedule").upsert(previews, on_conflict="game_id").execute()
    print(f"Upserted {len(previews)} game previews")


def main() -> None:
    print("\n--- NBA Game Preview Pipeline ---")
    clear_old_games()

    previews = build_game_preview(DEFAULT_UPCOMING_DAYS_WINDOW)
    if previews:
        upsert_previews(previews)

        preds, _ = generate_predictions(
            days_window=DEFAULT_UPCOMING_DAYS_WINDOW,
            model_dir=MODELS_DIR,
            calibrate_with_odds=True,
            blend_factor=0.3,
            historical_lookback=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
        )
        if preds:
            upsert_score_predictions(preds)
            print(f"Upserted {len(preds)} score predictions")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
