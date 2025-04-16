# backend/data_pipeline/nba_injuries.py

import os
import json
import re # Keep re for player ID extraction
import time # Keep time? Only used for sleep if API helper added delays? Remove for now.
import requests # Keep requests as get_nba_injuries uses it directly
import datetime
from datetime import date, time as dt_time, timedelta, datetime as dt_datetime
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser # Keep if parsing dates
from typing import Dict, Optional, Tuple, List, Any

# --- Supabase and Config Imports ---
try:
    from supabase import create_client, Client
    # Import needed config vars
    from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, RAPIDAPI_KEY, RAPIDAPI_HOST # Use SERVICE_KEY
    print("Successfully imported config variables for NBA injuries.")
except ImportError as e: print(f"FATAL ERROR: Import failed: {e}"); exit(1)
except Exception as e: print(f"FATAL ERROR: Unexpected import error: {e}"); exit(1)

# --- Configuration ---
SUPABASE_TABLE_NAME = "nba_injuries"
ET_ZONE = ZoneInfo("America/New_York")
# No scraping config needed
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
def get_nba_injuries(rapidapi_key: str, rapidapi_host: str) -> Optional[Dict[str, Any]]:
    """ Retrieves NBA injuries data from RapidAPI. """
    endpoint_path = "/nba/injuries"
    url = f"https://{rapidapi_host}{endpoint_path}"
    headers = {"x-rapidapi-key": rapidapi_key, "x-rapidapi-host": rapidapi_host}
    print(f"Fetching NBA injuries data from {url}...")
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
        team_id = team_id_added # Use value passed from parent team object
        team_display_name = team_displayName_added

        # Convert IDs to TEXT right away
        injury_id_str = str(injury_id) if injury_id is not None else None
        player_id_str = str(player_id) if player_id is not None else None
        team_id_str = str(team_id) if team_id is not None else None

        # --- Validation ---
        if not all([injury_id_str, player_id_str, team_id_str, report_date_utc]):
            print(f"Warn: Skipping injury {injury_id_str}. Missing injury_id ('{injury_id_str}'), player_id ('{player_id_str}'), team_id ('{team_id_str}'), or report_date ('{report_date_utc}').")
            return None
        # --- End Validation ---

        player_display_name = athlete.get("displayName") if isinstance(athlete, dict) else None
        type_info = rec.get("type", {}); injury_status = type_info.get("description"); injury_status_abbr = type_info.get("abbreviation")
        details = rec.get("details", {}); injury_type = details.get("type"); injury_location = details.get("location"); injury_detail = details.get("detail"); injury_side = details.get("side"); return_date_est = details.get("returnDate")
        short_comment = rec.get("shortComment"); long_comment = rec.get("longComment")

        # Prepare final dictionary for Supabase
        transformed = {
            "injury_id": injury_id_str, "player_id": player_id_str, "player_display_name": player_display_name,
            "team_id": team_id_str, "team_display_name": team_display_name,
            "report_date_utc": report_date_utc, # Store as TIMESTAMPTZ compatible string
            "injury_status": injury_status, "injury_status_abbr": injury_status_abbr,
            "injury_type": injury_type, "injury_location": injury_location,
            "injury_detail": injury_detail, "injury_side": injury_side,
            "return_date_est": return_date_est if return_date_est else None, # Handle potential null/empty string
            "short_comment": short_comment, "long_comment": long_comment,
            "last_api_update_time": report_date_utc, # Correctly use API report date
            "raw_api_response": json.dumps(rec) # Store original nested record
            # created_at/updated_at handled by Supabase default/trigger or set in upsert
        }
        return transformed
    except Exception as e: print(f"Error transforming nested injury record id {rec.get('id', 'UNKNOWN')}: {e}"); return None

def transform_team_injury_record(team_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Processes a team record, extracts nested injuries, adds team context, and transforms them. """
    transformed_records = []
    team_id = team_record.get("id")
    team_display_name = team_record.get("displayName")
    nested_injuries = team_record.get("injuries", [])

    if not isinstance(nested_injuries, list):
        print(f"Warn: Expected list for nested 'injuries' in team {team_id}, found {type(nested_injuries)}. Skipping."); return []

    # print(f"Processing {len(nested_injuries)} injury reports for team: {team_display_name} (ID: {team_id})")
    for injury_rec in nested_injuries:
        if isinstance(injury_rec, dict):
            # Pass team info explicitly to the single transform function
            transformed = _transform_single_injury(injury_rec, str(team_id) if team_id else None, team_display_name)
            if transformed:
                transformed_records.append(transformed)
        else: print(f"Warn: Found non-dict item in nested injuries list for team {team_id}.")
    return transformed_records

# --- Supabase Upsert Function ---
def upsert_nba_injuries(supabase_client: Client, injuries_payload: List[Dict[str, Any]]) -> int:
    """ Upserts transformed NBA injury records. Expects already transformed data. """
    if not injuries_payload: print("No valid injury records to upsert."); return 0

    print(f"Attempting to upsert {len(injuries_payload)} valid injury record(s)...")
    try:
        # Use the passed-in Supabase client (should be initialized with service key)
        response = supabase_client.table("nba_injuries").upsert(injuries_payload, on_conflict="injury_id").execute()
        updated_count = 0
        if hasattr(response,"data") and response.data: updated_count = len(response.data); print(f"Upsert success: {updated_count} rows affected.")
        elif hasattr(response,'error') and response.error: print(f"Supabase Upsert Error: {response.error}")
        else: print("Upsert complete (may have had no changes).")
        return updated_count
    except Exception as e: print(f"Error during Supabase upsert: {e}"); return 0

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting NBA Injuries Update Script...")
    script_start_time = time.time() # Keep time

    # Load Environment Variables - Use values directly from config import
    # No need for load_environment function if importing directly from config

    # Initialize Supabase client ONCE using the Service Key
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("FATAL ERROR: Supabase URL or Service Key missing. Check config import/env.")
        exit(1)
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Supabase client: {e}"); exit(1)

    # Fetch Raw Data from API
    injuries_raw = get_nba_injuries(RAPIDAPI_KEY, RAPIDAPI_HOST) # Use imported keys
    if not injuries_raw or not isinstance(injuries_raw, dict):
        print("Failed to retrieve valid data structure from NBA injuries API."); exit(1)

    # Extract list of team injury records
    team_injury_list = injuries_raw.get("injuries") # Key confirmed from logs
    if not isinstance(team_injury_list, list):
        print(f"Error: Expected list under 'injuries' key, found {type(team_injury_list)}."); exit(1)
    print(f"Retrieved injury data for {len(team_injury_list)} teams from API.")

    # Process and transform nested structure
    all_transformed_injuries = []
    print(f"Transforming {len(team_injury_list)} raw team records...")
    for team_rec in team_injury_list:
        transformed_list = transform_team_injury_record(team_rec) # Renamed outer transform function
        if transformed_list:
            all_transformed_injuries.extend(transformed_list) # Flatten the list

    # Upsert the flattened list of transformed records
    updated_count = upsert_nba_injuries(supabase, all_transformed_injuries) # Pass client and flattened list

    script_end_time = time.time() # Keep time
    print(f"\nFinished upserting injuries. Updated/Inserted {updated_count} records.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")