import os, logging, threading, sys, time

if os.getenv("CI") or os.getenv("LOG_LEVEL_OVERRIDE", "").upper() == "ERROR":
    logging.disable(logging.ERROR)
    def _tick():
        while True:
            time.sleep(30)
            print('.', flush=True)
    threading.Thread(target=_tick, daemon=True).start()

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback
import re
import difflib
import requests
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# allow from config import … to find /backend/config.py
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from zoneinfo import ZoneInfo

# make project root importable
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
        RAPIDAPI_KEY,
        RAPIDAPI_HOST,
    )
    print("Loaded configuration from config.py")
except ImportError as e:
    print(f"config.py not found ({e}), falling back to environment variables")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    RAPIDAPI_KEY         = os.getenv("RAPIDAPI_KEY")
    RAPIDAPI_HOST        = os.getenv("RAPIDAPI_HOST")

# shared constants from prediction.py
from backend.config import MAIN_MODELS_DIR as MODELS_DIR
from caching.supabase_client import supabase
from backend.nba_features.make_nba_snapshots import make_nba_snapshot
from nba_score_prediction.prediction import (
    DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
    generate_predictions,
    upsert_score_predictions,
)

# ─── Fix: Use x-apisports-key (not x-rapidapi-key) ─────────────────────────────
NBA_SPORTS_API = "https://v1.basketball.api-sports.io"
HEADERS_SPORTS = {
    "x-apisports-key": API_SPORTS_KEY
}
# ────────────────────────────────────────────────────────────────────────────────

# RapidAPI (for Injuries)
HEADERS_INJURIES = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
}

NBA_ODDS_API = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ET_ZONE = ZoneInfo("America/New_York")
NBA_INJURIES_TABLE = "nba_injuries"
NBA_SCHEDULE_TABLE = "nba_game_schedule"

# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    return " ".join(name.split()).lower()

def title_case_team_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split())

def make_api_request(
    url: str, headers: Dict[str, str], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """GET and return JSON or None on error."""
    print(f"Making API request to: {url} with params: {params}")
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        print(f"API Request successful (Status {resp.status_code}): {url}")
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"API request error for {url} {params}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}...")
        return None

# ESPN fallback helper
def espn_scoreboard_fallback(et_date_iso: str) -> List[Dict[str, Any]]:
    """
    Query ESPN’s public scoreboard for a given ET date (YYYY-MM-DD), 
    parse events into the same shape as API-Sports `/games`.
    """
    # Convert ET date string "YYYY-MM-DD" to ESPN format "YYYYMMDD"
    y, m, d = et_date_iso.split("-")
    espn_date = f"{y}{m}{d}"  # e.g. "20250530"

    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {"dates": espn_date}

    print(f"📺 Falling back to ESPN scoreboard for ET date {et_date_iso} -> {espn_date}")
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching ESPN scoreboard for {et_date_iso}: {e}")
        return []

    events = data.get("events", [])
    fallback_games: List[Dict[str, Any]] = []
    for ev in events:
        # Unique game ID as int (ESPN’s ID is a string like "401567891"; cast to int)
        game_id_str = ev.get("id")
        try:
            game_id = int(game_id_str)
        except:
            # If parsing fails, skip
            continue

        # ESPN’s event date is in UTC (e.g. "2025-05-31T00:00Z"), 
        # but they also provide an offset in "date" itself if you look at their mobile JSON.
        # Here, we trust the "date" field—later we convert to ET if needed.
        date_utc = ev.get("date")  # e.g. "2025-05-31T00:00Z"

        # Venue
        venue_info = ev.get("competitions", [])[0].get("venue", {})
        venue_name = venue_info.get("fullName", "Unknown Venue")
        venue_city = venue_info.get("address", {}).get("city", "")

        # Teams (home/away)
        competitors = ev.get("competitions", [])[0].get("competitors", [])
        home_team = {}
        away_team = {}
        for comp in competitors:
            team_obj = comp.get("team", {})
            tid = team_obj.get("id")
            tname = team_obj.get("displayName")
            home_away_flag = comp.get("homeAway")  # "home" or "away"
            if home_away_flag == "home":
                home_team = {"id": tid, "name": tname}
            else:
                away_team = {"id": tid, "name": tname}

        # Build our pseudo–/games object
        fallback_games.append({
            "id": game_id,
            "date": date_utc,  # We’ll re-interpret as ET downstream
            "timezone": "UTC",  # Mark as UTC; downstream will convert to ET
            "status": {"long": ev.get("status", {}).get("type", {}).get("description", ""), 
                       "short": ev.get("status", {}).get("type", {}).get("state", "")},
            "league": {
                "id": 12,
                "name": "NBA",
                "country": "USA",
                "logo": "",  # ESPN doesn’t provide API-Sports logo
                "season": 2024,
                "type": "League",
            },
            "season": 2024,  # 2024–2025 season
            "venue": {"name": venue_name, "city": venue_city},
            "teams": {
                "home": {"id": home_team.get("id"), "name": home_team.get("name")},
                "away": {"id": away_team.get("id"), "name": away_team.get("name")},
            },
            "scores": {
                "home": {"full": None},
                "away": {"full": None},
            },
            # You can add "broadcasts": […] if downstream code expects it, but often preview only cares about teams/date/venue
        })

    print(f"🌐 ESPN fallback returned {len(fallback_games)} events for {et_date_iso}")
    return fallback_games

def get_games_by_date(
    league: str, season: str, date: str, timezone: str = "America/New_York"
) -> Dict[str, Any]:
    """Fetch games for a specific date using API-Sports."""
    url = f"{NBA_SPORTS_API}/games"
    params = dict(league=league, season=season, date=date, timezone=timezone)
    return make_api_request(url, HEADERS_SPORTS, params) or {}

def get_betting_odds(et_date: datetime) -> List[Dict[str, Any]]:
    """Fetch odds for a given ET date from The Odds API, with head-to-head fallback."""
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not configured.")
        return []

    start_et = et_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et   = et_date.replace(hour=23, minute=59, second=59, microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": start_et.astimezone(ZoneInfo("UTC")).strftime(fmt),
        "commenceTimeTo":   end_et.astimezone(ZoneInfo("UTC")).strftime(fmt),
    }

    print(f"Requesting odds (all markets) with params: {params}")
    resp = requests.get(NBA_ODDS_API, params=params)

    # success path
    if resp.status_code == 200:
        return resp.json()

    # fallback on unauthorized
    if resp.status_code == 401 and "," in params["markets"]:
        print("Received 401 for full markets → retrying head-to-head only")
        params["markets"] = "h2h"
        print(f"Requesting odds (h2h-only) with params: {params}")
        resp = requests.get(NBA_ODDS_API, params=params)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Fallback also failed: {resp.status_code} {resp.text[:200]}")

    print(f"Error fetching odds: HTTP {resp.status_code} {resp.text[:200]}")
    return []

def match_odds_for_game(
    game: Dict[str, Any], odds_events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Match one odds event to a game by team names and ET date."""
    if not odds_events:
        return None

    home = normalize_team_name(game["teams"]["home"]["name"])
    away = normalize_team_name(game["teams"]["away"]["name"])
    try:
        gdt = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
        gdate = gdt.astimezone(ET_ZONE).date()
    except Exception:
        return None

    for ev in odds_events:
        oh = normalize_team_name(ev.get("home_team", ""))
        oa = normalize_team_name(ev.get("away_team", ""))
        if home == oh and away == oa and _same_et_date(ev, gdate):
            return ev

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

# --- Schedule Clearing Function ---
def clear_past_schedule_data() -> None:
    """Delete nba_game_schedule entries with ET game_date < today."""
    today_et_iso = datetime.now(ET_ZONE).date().isoformat()
    print(f"Clearing past games before {today_et_iso}")
    resp = supabase.table(NBA_SCHEDULE_TABLE).delete().lt("game_date", today_et_iso).execute()
    if getattr(resp, "error", None):
        print("Error clearing past games:", resp.error)
    else:
        print("Successfully cleared past games.")

def _fetch_games_for_date(date_iso: str) -> List[Dict[str, Any]]:
    """
    Fetch games for a given ET date by first trying:
      1) API-Sports “/games?league=12&season=2024-2025&date=<date_iso>”
      2) If that returns nothing, API-Sports “/games?league=12&next=10&timezone=America/New_York”
         filtered down to games whose ET date == date_iso
      3) If *still* nothing, fall back to ESPN’s scoreboard for that ET date.

    At each stage, we key games by their `id`, so if API-Sports ever returns duplicates
    (two different JSON objects with different `"id"` values but the same home/way matchup),
    they will be kept separately—but in practice API-Sports rarely duplicates IDs in stage (1).
    The merging logic ensures we never get more than one “primary” or “fallback” entry per ID.
    ESPN is only consulted if the union of primary+fallback is empty.
    """

    # 1) PRIMARY: “/games?league=12&season=2024-2025&date=<date_iso>”
    primary_resp = get_games_by_date("12", "2024-2025", date_iso)
    primary = primary_resp.get("response", []) or []

    # 2) FALLBACK: “/games?league=12&next=10&timezone=America/New_York” → keep only those whose ET date == date_iso
    raw = make_api_request(
        f"{NBA_SPORTS_API}/games",
        HEADERS_SPORTS,
        {"league": "12", "next": 10, "timezone": "America/New_York"},
    ) or {}
    raw_list = raw.get("response", []) or []

    def _is_et_date(g: Dict[str, Any]) -> bool:
        """
        Convert g["date"] (ISO UTC, e.g. "2025-05-31T19:00:00Z") → ET timezone,
        then compare that .date().isoformat() to date_iso.
        """
        try:
            dt_utc = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
            dt_et = dt_utc.astimezone(ET_ZONE)
            return dt_et.date().isoformat() == date_iso
        except Exception:
            return False

    fallback = [g for g in raw_list if _is_et_date(g)]

    # 3) MERGE primary + fallback BY game ID (so no duplicate IDs in the union)
    #    In the old script, this was exactly:
    #      games_by_id = {g["id"]: g for g in primary}
    #      for g in fallback: games_by_id.setdefault(g["id"], g)
    #      return list(games_by_id.values())
    #
    #    We’ll do that, then check if it’s empty. If it is empty, call ESPN fallback:
    games_by_id: Dict[int, Dict[str, Any]] = {}
    for g in primary:
        # Ensure g["id"] is an integer:
        try:
            gid = int(g.get("id"))
        except Exception:
            continue
        games_by_id[gid] = g

    for g in fallback:
        try:
            gid = int(g.get("id"))
        except Exception:
            continue
        # Only insert if we did not already see that ID in `primary`
        games_by_id.setdefault(gid, g)

    # If after merging primary+fallback we already have ≥1 game, return them now:
    if games_by_id:
        return list(games_by_id.values())

    # 4) ESPN fallback (only if both primary and fallback were empty)
    espn_games = espn_scoreboard_fallback(date_iso)
    if not espn_games:
        # Nothing from any source
        return []

    # Merge ESPN by ID to avoid duplicates if ESPN itself somehow returned two events with same ID
    espn_by_id: Dict[int, Dict[str, Any]] = {}
    for g in espn_games:
        try:
            gid = int(g.get("id"))
        except Exception:
            continue
        if gid not in espn_by_id:
            espn_by_id[gid] = g

    return list(espn_by_id.values())

# --- Game Preview Building/Upserting Functions ---
def build_game_preview(window_days: int = 3) -> List[Dict[str, Any]]:
    """Fetch pregame NBA schedule + odds and return preview dicts for today + next window_days-1 days."""
    all_games: List[Dict[str, Any]] = []
    today_et = datetime.now(ET_ZONE).date()

    for offset in range(window_days):
        fetch_date = (today_et + timedelta(days=offset)).isoformat()
        print(f"→ Fetching schedule for ET date {fetch_date}")

        games = _fetch_games_for_date(fetch_date)

        scheduled = [
            g for g in games
            if (
                g.get("status", {}).get("short") in {"NS", "TBD"}
                or any(
                    phrase in g.get("status", {}).get("long", "").lower()
                    for phrase in (
                        "scheduled",
                        "time to be determined",
                        "to be announced",
                    )
                )
            )
        ]
        print(f"   found {len(scheduled)} games")
        all_games.extend(scheduled)

    if not all_games:
        return []

    # 3) Odds fetch at midnight ET for each day
    odds_list: List[Dict[str, Any]] = []
    for offset in range(window_days):
        d = today_et + timedelta(days=offset)
        midnight_et = datetime(d.year, d.month, d.day, tzinfo=ET_ZONE)
        odds_list.extend(get_betting_odds(midnight_et))

    # 4) Build previews
    previews: List[Dict[str, Any]] = []
    for g in all_games:
        ev = match_odds_for_game(g, odds_list)
        odds = extract_odds_by_market(ev)
        venue = (
            g.get("venue", {}).get("name")
            if isinstance(g.get("venue"), dict)
            else g.get("venue", "N/A")
        )
        gd = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).astimezone(ET_ZONE).date().isoformat()
        sched_dt = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).astimezone(ET_ZONE)
        previews.append({
            "game_id":         str(g["id"]),
            "scheduled_time":  sched_dt.isoformat(),
            "game_date":       sched_dt.date().isoformat(),
            "venue":           venue,
            "away_team":       g["teams"]["away"]["name"],
            "home_team":       g["teams"]["home"]["name"],
            **odds,
        })

    return previews

def upsert_previews(previews: List[Dict[str, Any]]) -> None:
    """Upsert preview records into Supabase by game_id."""
    if not previews:
        return
    supabase.table(NBA_SCHEDULE_TABLE).upsert(previews, on_conflict="game_id").execute()
    print(f"Upserted {len(previews)} game previews")

# --- Injury Functions (unchanged from your existing code) ---
def _extract_player_id(athlete: Dict[str, Any]) -> Optional[str]:
    player_id = athlete.get("id")
    if player_id: return str(player_id)
    links = athlete.get("links", [])
    for link in links:
        if "playercard" in link.get("rel", []):
            match = re.search(r"/id/(\d+)", link.get("href", ""))
            if match: return match.group(1)
    uid = athlete.get("uid", "")
    match_uid = re.search(r"a:(\d+)", uid)
    if match_uid: return match_uid.group(1)
    return None

def fetch_rapidapi_injuries() -> List[Dict[str, Any]]:
    print("\n--- Fetching Injury Data (RapidAPI) ---")
    endpoint_path = "/nba/injuries"
    url = f"https://{RAPIDAPI_HOST}{endpoint_path}"
    response_json = make_api_request(url, HEADERS_INJURIES, {})
    if response_json and "injuries" in response_json:
        team_injury_list = response_json["injuries"]
        if isinstance(team_injury_list, list):
            print(f"Fetched injury data for {len(team_injury_list)} teams from RapidAPI.")
            return team_injury_list
        else:
            print(f"Warn: Expected 'injuries' field to be a list, but got {type(team_injury_list)}.")
            return []
    else:
        print("Failed to fetch injuries or response format unexpected from RapidAPI.")
        return []

def _transform_single_injury(rec: Dict[str, Any], team_id_added: Optional[str], team_displayName_added: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        injury_id = rec.get("id")
        report_date_utc = rec.get("date")
        athlete = rec.get("athlete", {}); player_id = _extract_player_id(athlete) if isinstance(athlete, dict) else None
        team_id = team_id_added; team_display_name = team_displayName_added

        injury_id_str = str(injury_id) if injury_id is not None else None
        player_id_str = str(player_id) if player_id is not None else None
        team_id_str = str(team_id) if team_id is not None else None

        if not all([injury_id_str, player_id_str, team_id_str, report_date_utc]):
            print(f"Warn: Skipping injury {injury_id_str}. Missing required ID/Date fields.")
            return None

        player_display_name = athlete.get("displayName") if isinstance(athlete, dict) else None
        type_info = rec.get("type", {}); injury_status = type_info.get("description"); injury_status_abbr = type_info.get("abbreviation")
        details = rec.get("details", {}); injury_type = details.get("type"); injury_location = details.get("location"); injury_detail = details.get("detail"); injury_side = details.get("side"); return_date_est = details.get("returnDate")
        short_comment = rec.get("shortComment"); long_comment = rec.get("longComment")

        transformed = {
            "injury_id":         injury_id_str,
            "player_id":         player_id_str,
            "player_display_name": player_display_name,
            "team_id":           team_id_str,
            "team_display_name": team_display_name,
            "report_date_utc":   report_date_utc,
            "injury_status":     injury_status,
            "injury_status_abbr": injury_status_abbr,
            "injury_type":       injury_type,
            "injury_location":   injury_location,
            "injury_detail":     injury_detail,
            "injury_side":       injury_side,
            "return_date_est":   return_date_est if return_date_est else None,
            "short_comment":     short_comment,
            "long_comment":      long_comment,
            "last_api_update_time": report_date_utc,
            "raw_api_response":  json.dumps(rec)
        }
        return transformed
    except Exception as e:
        print(f"Error transforming nested injury record id {rec.get('id', 'UNKNOWN')}: {e}")
        return None

def process_and_normalize_rapidapi_injuries() -> List[Dict[str, Any]]:
    team_injury_list = fetch_rapidapi_injuries()
    all_normalized_records: List[Dict[str, Any]] = []
    if not team_injury_list:
        return all_normalized_records
    print(f"Processing {len(team_injury_list)} team injury records...")
    for team_record in team_injury_list:
        team_id = team_record.get("id"); team_display_name = team_record.get("displayName")
        nested_injuries = team_record.get("injuries", [])
        if not isinstance(nested_injuries, list):
            print(f"Warn: Expected list for nested 'injuries' in team {team_id}. Skipping.")
            continue
        for injury_rec in nested_injuries:
            if isinstance(injury_rec, dict):
                transformed = _transform_single_injury(injury_rec, str(team_id) if team_id else None, team_display_name)
                if transformed: all_normalized_records.append(transformed)
            else:
                print(f"Warn: Found non-dict item in nested injuries list for team {team_id}.")
    print(f"Finished processing. Total normalized injury records: {len(all_normalized_records)}")
    return all_normalized_records

def update_injuries_table_clear_insert(injuries: List[Dict[str, Any]]) -> None:
    if not injuries:
        print("No normalized injuries to update in the table.")
        return
    print(f"\n--- Updating '{NBA_INJURIES_TABLE}' Table (Clear & Insert) ---")
    try:
        delete_response = supabase.table(NBA_INJURIES_TABLE).delete().neq('player_id', '-99999').execute()
        if hasattr(delete_response, 'error') and delete_response.error:
            print(f"Error clearing '{NBA_INJURIES_TABLE}': {delete_response.error}")
            print("Aborting injury update due to clearing error.")
            return
        else:
            print(f"Successfully cleared '{NBA_INJURIES_TABLE}'.")
        print(f"Inserting {len(injuries)} new injury records into '{NBA_INJURIES_TABLE}'...")
        insert_response = supabase.table(NBA_INJURIES_TABLE).insert(injuries).execute()
        if hasattr(insert_response, 'error') and insert_response.error:
            print(f"Error inserting injury data: {insert_response.error}")
            if hasattr(insert_response.error, 'details'): print(f"Details: {insert_response.error.details}")
            if hasattr(insert_response.error, 'hint'): print(f"Hint: {insert_response.error.hint}")
            if hasattr(insert_response.error, 'message'): print(f"Message: {insert_response.error.message}")
        else:
            inserted_count = len(insert_response.data) if hasattr(insert_response, 'data') else 'N/A'
            print(f"Successfully executed insert request. Response indicates {inserted_count} rows processed.")
    except Exception as e:
        print(f"An exception occurred during updating '{NBA_INJURIES_TABLE}': {e}")

def main():
    start = time.time()
    print("\n--- NBA Daily Pipeline Start ---")
    try:
        # 1) Clear past games
        clear_past_schedule_data()

        # 2) Injuries logic
        normalized = process_and_normalize_rapidapi_injuries()
        update_injuries_table_clear_insert(normalized)

        # 3) Previews + odds for today & tomorrow
        previews = build_game_preview(window_days=3)
        if previews:
            # ─── REMOVE ANY EXISTING schedule rows for these preview dates ─────────────────
            # We extract the distinct game_date values from the new `previews` list,
            # then delete old rows for exactly those dates. This prevents stale IDs
            # (like 449706) from lingering in Supabase.
            dates_to_delete = sorted({p["game_date"] for p in previews})
            # Perform a delete WHERE game_date IN dates_to_delete
            supabase.table(NBA_SCHEDULE_TABLE).delete().in_("game_date", dates_to_delete).execute()

            # Now upsert only the fresh preview rows
            upsert_previews(previews)
            from nba_score_prediction.prediction import fetch_and_parse_betting_odds
            game_ids = [p["game_id"] for p in previews]
            fetch_and_parse_betting_odds(supabase, game_ids)
            print(f"Upserted {len(previews)} game previews")

            # ─── Generate + upsert NBA snapshots for each previewed game ─────────
            print(f"Generating NBA snapshots for {len(game_ids)} games…")
            for gid in game_ids:
                try:
                    make_nba_snapshot(gid)
                    print(f"✅ NBA snapshot generated for game {gid}")
                except Exception as e:
                    print(f"❌ NBA snapshot failed for game {gid}: {e}")

            # 4) Generate & upsert predictions
            print("\n--- Generating Predictions ---")
            preds, _ = generate_predictions(
                days_window=3,
                model_dir=MODELS_DIR,
                historical_lookback=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
                debug_mode=False
            )
            if preds:
                upsert_score_predictions(preds)
                print(f"Upserted {len(preds)} predictions")
            else:
                print("No predictions generated")
        else:
            print("Skipped previews & predictions; no games found")

    except Exception:
        logger.error("Pipeline failed:", exc_info=True)
        sys.exit(1)
    finally:
        elapsed = time.time() - start
        logger.info(f"Pipeline finished in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
