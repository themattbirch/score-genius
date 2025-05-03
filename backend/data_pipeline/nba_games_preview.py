# backend/data_pipeline/nba_games_preview.py

"""
Fetch upcoming NBA games, enrich with betting odds, fetch injuries, store in Supabase,
and generate score predictions. 
"""

# ‑‑‑‑‑ ABSOLUTE SILENCE IN CI ‑‑‑‑‑
import os, sys, logging, types

if os.getenv("CI") or os.getenv("LOG_LEVEL_OVERRIDE", "").upper() == "ERROR":
    # 1) Disable logging calls WARNING and below
    logging.disable(logging.ERROR)          # CRITICAL still shows
    # 2) Replace stdout/stderr with a dummy stream
    class _Null(types.SimpleNamespace):
        write = lambda *a, **k: None
        flush = lambda *a, **k: None
    sys.stdout = sys.stderr = _Null()
# ‑‑‑‑‑ END SILENCE BLOCK ‑‑‑‑‑

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback
import re # Added re
import difflib
import requests
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# allow from config import … to find /backend/config.py
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# now config.py can be imported, and load_dotenv will run
try:
    from config import (
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

# Validate
_missing = [
    name for name, val in [
        ("API_SPORTS_KEY",       API_SPORTS_KEY),
        ("ODDS_API_KEY",         ODDS_API_KEY),
        ("SUPABASE_URL",         SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
        ("RAPIDAPI_KEY",         RAPIDAPI_KEY),      # Added
        ("RAPIDAPI_HOST",        RAPIDAPI_HOST),     # Added
    ] if not val
]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    sys.exit(1)


from caching.supabase_client import supabase # Assumes this initializes Supabase client correctly
from nba_score_prediction.prediction import (
    DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
    generate_predictions,
    upsert_score_predictions,
)


# --- Constants ---
MODELS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "models" / "saved"
)
# API-Sports (for Games/Odds)
NBA_SPORTS_API = "https://v1.basketball.api-sports.io"
HEADERS_SPORTS = {
    "x-rapidapi-key": API_SPORTS_KEY, # Note: Key name might be confusing, but using your var
    "x-rapidapi-host": "v1.basketball.api-sports.io",
}
# RapidAPI (for Injuries) - Using different host/key from config
HEADERS_INJURIES = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
}

NBA_ODDS_API = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ET_ZONE = ZoneInfo("America/New_York")
NBA_INJURIES_TABLE = "nba_injuries" # Confirmed table name
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
    print(f"Making API request to: {url} with params: {params}") # Log request
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

# --- Game/Odds Functions ---
def get_games_by_date(
    league: str, season: str, date: str, timezone: str = "America/New_York"
) -> Dict[str, Any]:
    """Fetch games for a specific date using API-Sports."""
    url = f"{NBA_SPORTS_API}/games"
    params = dict(league=league, season=season, date=date, timezone=timezone)
    # Using HEADERS_SPORTS for this request
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

# --- Schedule Clearing Function ---
# (Keep clear_upcoming_schedule_data)
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
    """Try strict date lookup first; if empty, fall back to `next`."""
    # 1) primary query
    primary = get_games_by_date("12", "2024-2025", date_iso).get("response", [])
    if primary:
        return primary

    # 2) fallback – grab next 10 fixtures, filter locally
    fallback = make_api_request(
        f"{NBA_SPORTS_API}/games",
        HEADERS_SPORTS,
        {"league": "12", "next": 10, "timezone": "America/New_York"},
    ).get("response", [])

    return [
        g for g in fallback
        if datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
              .astimezone(ET_ZONE)
              .date()
              .isoformat() == date_iso
    ]

# --- Game Preview Building/Upserting Functions ---
def build_game_preview(window_days: int = 2) -> List[Dict[str, Any]]:
    """Fetch pregame NBA schedule + odds and return a list of preview dicts for today + next window_days-1 days."""

    all_games: List[Dict[str, Any]] = []

    today_et = datetime.now(ET_ZONE).date()

    # 2) Gather all scheduled (NS) games for today, tomorrow, ...
    for offset in range(window_days):
        fetch_date = (today_et + timedelta(days=offset)).isoformat()
        print(f"→ Fetching schedule for ET date {fetch_date}")

        # OLD: data = get_games_by_date("12", "2024-2025", fetch_date)
        #       games = data.get("response", [])
        # NEW: call the robust helper instead
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
        gd = (
            datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
            .astimezone(ET_ZONE)
            .date()
            .isoformat()
        )
        previews.append(
            {
                "game_id": str(g["id"]),
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
    supabase.table(NBA_SCHEDULE_TABLE).upsert(previews, on_conflict="game_id").execute()
    print(f"Upserted {len(previews)} game previews")


# --- <<< Injury Functions (Integrated from nba_injuries.py) >>> ---

def _extract_player_id(athlete: Dict[str, Any]) -> Optional[str]:
    """ Robustly extracts player ID from RapidAPI athlete object. """
    # (Copied directly from nba_injuries.py)
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
    """ Retrieves NBA injuries data from RapidAPI (espn source). """
    # Based on get_nba_injuries from nba_injuries.py
    print("\n--- Fetching Injury Data (RapidAPI) ---")
    endpoint_path = "/nba/injuries" # Check if this is the correct endpoint
    url = f"https://{RAPIDAPI_HOST}{endpoint_path}"
    # Using HEADERS_INJURIES for this request
    response_json = make_api_request(url, HEADERS_INJURIES, {}) # No params needed?

    if response_json and "injuries" in response_json:
        team_injury_list = response_json["injuries"]
        if isinstance(team_injury_list, list):
             print(f"Fetched injury data for {len(team_injury_list)} teams from RapidAPI.")
             return team_injury_list
        else:
             print(f"Warn: Expected 'injuries' field to be a list, but got {type(team_injury_list)}. Data: {str(response_json)[:500]}")
             return []
    else:
        print("Failed to fetch injuries or response format unexpected from RapidAPI.")
        return []


def _transform_single_injury(rec: Dict[str, Any], team_id_added: Optional[str], team_displayName_added: Optional[str]) -> Optional[Dict[str, Any]]:
    """ Transforms a single nested injury record from RapidAPI. """
    # (Copied directly from nba_injuries.py, ensure keys match YOUR table)
    try:
        injury_id = rec.get("id") # API's ID for the injury report event
        report_date_utc = rec.get("date") # Expect ISO string (TIMESTAMPTZ)
        athlete = rec.get("athlete", {}); player_id = _extract_player_id(athlete) if isinstance(athlete, dict) else None
        team_id = team_id_added; team_display_name = team_displayName_added

        # Convert IDs to TEXT (or keep as INT if your DB uses INT)
        injury_id_str = str(injury_id) if injury_id is not None else None
        player_id_str = str(player_id) if player_id is not None else None
        team_id_str = str(team_id) if team_id is not None else None

        # Validation Check (Essential IDs + Date)
        if not all([injury_id_str, player_id_str, team_id_str, report_date_utc]):
            print(f"Warn: Skipping injury {injury_id_str}. Missing required ID/Date fields.")
            return None

        # Extract other fields
        player_display_name = athlete.get("displayName") if isinstance(athlete, dict) else None
        type_info = rec.get("type", {}); injury_status = type_info.get("description"); injury_status_abbr = type_info.get("abbreviation")
        details = rec.get("details", {}); injury_type = details.get("type"); injury_location = details.get("location"); injury_detail = details.get("detail"); injury_side = details.get("side"); return_date_est = details.get("returnDate")
        short_comment = rec.get("shortComment"); long_comment = rec.get("longComment")

        # Prepare final dictionary - *** ADJUST KEYS TO MATCH YOUR TABLE ***
        transformed = {
            "injury_id": injury_id_str, # This is the API's report ID
            "player_id": player_id_str,
            "player_display_name": player_display_name,
            "team_id": team_id_str,
            "team_display_name": team_display_name,
            "report_date_utc": report_date_utc, # Store as TIMESTAMPTZ compatible string
            "injury_status": injury_status,
            "injury_status_abbr": injury_status_abbr,
            "injury_type": injury_type,
            "injury_location": injury_location,
            "injury_detail": injury_detail,
            "injury_side": injury_side,
            "return_date_est": return_date_est if return_date_est else None,
            "short_comment": short_comment,
            "long_comment": long_comment,
            "last_api_update_time": report_date_utc, # Use API report date
            "raw_api_response": json.dumps(rec) # Optional: Store raw data
            # Supabase handles 'created_at'/'updated_at' automatically if configured
        }
        return transformed
    except Exception as e: print(f"Error transforming nested injury record id {rec.get('id', 'UNKNOWN')}: {e}"); return None

def process_and_normalize_rapidapi_injuries() -> List[Dict[str, Any]]:
    """ Fetches from RapidAPI, processes team records, and returns a flat list of normalized injuries. """
    team_injury_list = fetch_rapidapi_injuries()
    all_normalized_records: List[Dict[str, Any]] = []

    if not team_injury_list:
        return all_normalized_records # Return empty list if fetch failed

    print(f"Processing {len(team_injury_list)} team injury records...")
    for team_record in team_injury_list:
         # Based on transform_team_injury_record from nba_injuries.py
        team_id = team_record.get("id"); team_display_name = team_record.get("displayName")
        nested_injuries = team_record.get("injuries", [])

        if not isinstance(nested_injuries, list):
            print(f"Warn: Expected list for nested 'injuries' in team {team_id}. Skipping.")
            continue # Skip this team record

        for injury_rec in nested_injuries:
            if isinstance(injury_rec, dict):
                transformed = _transform_single_injury(injury_rec, str(team_id) if team_id else None, team_display_name)
                if transformed: all_normalized_records.append(transformed)
            else:
                print(f"Warn: Found non-dict item in nested injuries list for team {team_id}.")

    print(f"Finished processing. Total normalized injury records: {len(all_normalized_records)}")
    # Removed the filter for "today's" injuries - we want the full current report
    return all_normalized_records


def update_injuries_table_clear_insert(injuries: List[Dict[str, Any]]) -> None:
    """ Clears the nba_injuries table and inserts the latest fetched injuries. """
    if not injuries:
        print("No normalized injuries to update in the table.")
        return

    print(f"\n--- Updating '{NBA_INJURIES_TABLE}' Table (Clear & Insert) ---")
    try:
        # 1. Clear the entire table
        print(f"Clearing ALL data from '{NBA_INJURIES_TABLE}'...")
        # Use a condition that will always be true to delete all rows.
        # Adjust column name 'player_id' if needed.
        delete_response = supabase.table(NBA_INJURIES_TABLE).delete().neq('player_id', '-99999').execute()

        if hasattr(delete_response, 'error') and delete_response.error:
            print(f"Error clearing '{NBA_INJURIES_TABLE}': {delete_response.error}")
            print("Aborting injury update due to clearing error.")
            return # Exit if clearing failed
        else:
            print(f"Successfully cleared '{NBA_INJURIES_TABLE}'.")

        # 2. Insert the new batch
        print(f"Inserting {len(injuries)} new injury records into '{NBA_INJURIES_TABLE}'...")
        # Increase chunk size if needed and supported by your Supabase plan/client library version
        # chunk_size = 500
        # for i in range(0, len(injuries), chunk_size):
        #    chunk = injuries[i:i + chunk_size]
        #    insert_response = supabase.table(NBA_INJURIES_TABLE).insert(chunk).execute()
        # Using single insert for simplicity now:
        insert_response = supabase.table(NBA_INJURIES_TABLE).insert(injuries).execute()


        if hasattr(insert_response, 'error') and insert_response.error:
             print(f"Error inserting injury data: {insert_response.error}")
             if hasattr(insert_response.error, 'details'): print(f"Details: {insert_response.error.details}")
             if hasattr(insert_response.error, 'hint'): print(f"Hint: {insert_response.error.hint}")
             if hasattr(insert_response.error, 'message'): print(f"Message: {insert_response.error.message}")
        else:
            inserted_count = len(insert_response.data) if hasattr(insert_response, 'data') else 'N/A' # Varies by client version
            print(f"Successfully executed insert request. Response indicates {inserted_count} rows processed (count may vary based on client library).")

    except Exception as e:
        print(f"An exception occurred during updating '{NBA_INJURIES_TABLE}': {e}")

# --- <<< END: Injury Functions >>> ---


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
        previews = build_game_preview(window_days=2)
        if previews:
            upsert_previews(previews)
            print(f"Upserted {len(previews)} game previews")

            # 4) Generate & upsert predictions
            print("\n--- Generating Predictions ---")
            preds, _ = generate_predictions(
                days_window=2,
                model_dir=MODELS_DIR,
                calibrate_with_odds=True,
                blend_factor=0.3,
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
