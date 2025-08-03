# backend/data_pipeline/nfl_injuries.py

import os
import sys
import json
import re
import requests
import time

from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser
from typing import Dict, Optional, Tuple, List, Any

from supabase import create_client, Client

# --- Local Config & Variables ---
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

_missing = [
    name for name, val in [
        ("API_SPORTS_KEY",       API_SPORTS_KEY),
        ("ODDS_API_KEY",         ODDS_API_KEY),
        ("SUPABASE_URL",         SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
        ("RAPIDAPI_KEY",         RAPIDAPI_KEY),
        ("RAPIDAPI_HOST",        RAPIDAPI_HOST),
    ] if not val
]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    sys.exit(1)

# Allow imports from backend/
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# from caching.supabase_client import supabase # Assuming you have a shared client wrapper

# --- Config ---
SUPABASE_TABLE_NAME = "nfl_injuries"
ET_ZONE = ZoneInfo("America/New_York")
# --- End Configuration ---

# --- Helper Functions ---
def _extract_player_id(athlete: Dict[str, Any]) -> Optional[str]:
    """ Robustly extracts player ID from athlete object. """
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

# --- API Fetching Function ---
def get_nfl_injuries(rapidapi_key: str, rapidapi_host: str) -> Optional[Dict[str, Any]]:
    """ Retrieves NFL injuries data from RapidAPI. """
    endpoint_path = "/nfl/injuries"
    url = f"https://{rapidapi_host}{endpoint_path}"
    headers = {"x-rapidapi-key": rapidapi_key, "x-rapidapi-host": rapidapi_host}
    print(f"Fetching NFL injuries data from {url}...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(f"Request successful. Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.HTTPError as http_err: print(f"HTTP error: {http_err} - {http_err.response.text[:200]}..."); return None
    except Exception as err: print(f"Other error occurred fetching injuries: {err}"); return None

# --- Transformation Functions ---
def _transform_single_injury(rec: Dict[str, Any], team_id_added: Optional[str], team_displayName_added: Optional[str]) -> Optional[Dict[str, Any]]:
    """ Transforms a single nested injury record. """
    try:
        injury_id = rec.get("id")
        report_date_utc = rec.get("date") # Expect ISO string
        athlete = rec.get("athlete", {}); player_id = _extract_player_id(athlete) if isinstance(athlete, dict) else None
        team_id = team_id_added; team_display_name = team_displayName_added

        # Convert IDs to TEXT
        injury_id_str = str(injury_id) if injury_id is not None else None
        player_id_str = str(player_id) if player_id is not None else None
        team_id_str = str(team_id) if team_id is not None else None

        # Validation Check
        if not all([injury_id_str, player_id_str, team_id_str, report_date_utc]):
            print(f"Warn: Skipping injury {injury_id_str}. Missing required ID/Date fields.")
            return None

        # Extract other fields
        player_display_name = athlete.get("displayName") if isinstance(athlete, dict) else None
        type_info = rec.get("type", {}); injury_status = type_info.get("description"); injury_status_abbr = type_info.get("abbreviation")
        details = rec.get("details", {}); injury_type = details.get("type"); injury_location = details.get("location"); injury_detail = details.get("detail"); injury_side = details.get("side"); return_date_est = details.get("returnDate")
        short_comment = rec.get("shortComment"); long_comment = rec.get("longComment")

        # Prepare final dictionary for Supabase
        transformed = {
            "injury_id": injury_id_str, "player_id": player_id_str, "player_display_name": player_display_name,
            "team_id": team_id_str, "team_display_name": team_display_name,
            "report_date_utc": report_date_utc,
            "injury_status": injury_status, "injury_status_abbr": injury_status_abbr,
            "injury_type": injury_type, "injury_location": injury_location,
            "injury_detail": injury_detail, "injury_side": injury_side,
            "return_date_est": return_date_est if return_date_est else None,
            "short_comment": short_comment, "long_comment": long_comment,
            "last_api_update_time": report_date_utc,
            "raw_api_response": json.dumps(rec)
        }
        return transformed
    except Exception as e: print(f"Error transforming nested injury record id {rec.get('id', 'UNKNOWN')}: {e}"); return None

def transform_team_injury_record(team_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Processes a team record, extracts nested injuries, adds team context, and transforms them. """
    transformed_records = []
    team_id = team_record.get("id"); team_display_name = team_record.get("displayName")
    nested_injuries = team_record.get("injuries", [])
    if not isinstance(nested_injuries, list): print(f"Warn: Expected list for nested 'injuries' in team {team_id}. Skipping."); return []

    for injury_rec in nested_injuries:
        if isinstance(injury_rec, dict):
            transformed = _transform_single_injury(injury_rec, str(team_id) if team_id else None, team_display_name)
            if transformed: transformed_records.append(transformed)
        else: print(f"Warn: Found non-dict item in nested injuries list for team {team_id}.")
    return transformed_records

# --- Supabase Upsert Function ---
def upsert_nfl_injuries(supabase_client: Client, injuries_payload: List[Dict[str, Any]]) -> int:
    """ Upserts transformed NFL injury records using the provided client. """
    if not injuries_payload: print("No valid injury records to upsert."); return 0
    print(f"Attempting to upsert {len(injuries_payload)} valid injury record(s) to '{SUPABASE_TABLE_NAME}'...")
    try:
        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(injuries_payload, on_conflict="injury_id").execute()
        updated_count = 0
        if hasattr(response,"data") and response.data: updated_count = len(response.data); print(f"Upsert success: {updated_count} rows affected.")
        elif hasattr(response,'error') and response.error: print(f"Supabase Upsert Error: {response.error}")
        else: print("Upsert complete (may have had no changes).")
        return updated_count
    except Exception as e: print(f"Error during Supabase upsert: {e}"); return 0

def main():
    print("Starting NFL Injuries Update Script…")
    start = time.time()

    # init supabase client
    supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # fetch & flatten
    injuries_raw = get_nfl_injuries(RAPIDAPI_KEY, RAPIDAPI_HOST)
    if not injuries_raw:
        print("Failed to fetch NFL injuries. Exiting.")
        return

    team_list = injuries_raw
    all_recs: List[Dict[str, Any]] = []
    for team in team_list:
        all_recs.extend(transform_team_injury_record(team))

    upsert_nfl_injuries(supa, all_recs)

    # To upsert all fetched injuries (not just today's), uncomment the next line
    # upsert_nfl_injuries(supa, all_recs)

    print(f"Done in {time.time() - start:.1f}s.")

if __name__ == "__main__":
    main()