import os
import json
import re
import requests
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

def load_environment() -> Tuple[str, str, str, str]:
    """
    Load environment variables for RapidAPI and Supabase.
    Returns a tuple (RAPIDAPI_KEY, RAPIDAPI_HOST, SUPABASE_URL, SUPABASE_ANON_KEY)
    """
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = os.getenv("RAPIDAPI_HOST")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not all([rapidapi_key, rapidapi_host, supabase_url, supabase_anon_key]):
        print("Error: Missing one or more required environment variables.")
        exit(1)
    return rapidapi_key, rapidapi_host, supabase_url, supabase_anon_key

def get_nba_injuries(rapidapi_key: str, rapidapi_host: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the latest NBA injuries data from the RapidAPI endpoint.
    Endpoint: /nba/injuries
    Method: GET
    """
    endpoint_path = "/nba/injuries"
    url = f"https://{rapidapi_host}{endpoint_path}"
    
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": rapidapi_host
    }
    
    print("Fetching NBA injuries data...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(f"Request successful. Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response: {http_err.response.text[:500]}...")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

def _extract_player_id(athlete: Dict[str, Any]) -> Optional[str]:
    """
    Attempts to extract the player's ID from the athlete dictionary.
    Priority:
      1) Directly from athlete["id"].
      2) From athlete["links"] if we find a 'playercard' link with /id/(\d+).
      3) From athlete["uid"], if it contains something like a:12345
    """
    # 1) Direct check
    player_id = athlete.get("id")
    if player_id:
        return str(player_id)
    
    # 2) Check athlete["links"]
    links = athlete.get("links", [])
    for link in links:
        rels = link.get("rel", [])
        if "playercard" in rels:
            href = link.get("href", "")
            match = re.search(r"/id/(\d+)", href)
            if match:
                return match.group(1)
    
    # 3) Parse from athlete["uid"], which sometimes looks like "s:40~l:46~a:4395628"
    uid = athlete.get("uid", "")
    match_uid = re.search(r"a:(\d+)", uid)
    if match_uid:
        return match_uid.group(1)
    
    # If all else fails, return None
    return None

def _transform_single_injury(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transforms a single **INDIVIDUAL** injury record (including added team info)
    into the Supabase table format. Corrected exception handler.
    """
    # Use the ID from the nested record if available, primarily for logging
    injury_id_for_logging = rec.get("id", "UNKNOWN_INJURY_ID")
    try:
        # --- Extract fields from the individual injury record ---
        injury_id = rec.get("id")
        report_date_utc = rec.get("date")

        # Player Info
        athlete = rec.get("athlete", {});
        player_id = _extract_player_id(athlete) if isinstance(athlete, dict) else None # Use helper

        # Team Info (Passed in from the outer loop)
        team_id = rec.get("team_id_added") # Use the key added in the main loop
        team_display_name = rec.get("team_displayName_added")

        # Fallback for team info if not added (shouldn't be needed with current main loop)
        if team_id is None:
             team_obj_in_athlete = athlete.get("team", {}) if isinstance(athlete, dict) else {}
             team_id = team_obj_in_athlete.get("id") if isinstance(team_obj_in_athlete, dict) else None
        if team_display_name is None:
             team_obj_in_athlete = athlete.get("team", {}) if isinstance(athlete, dict) else {}
             team_display_name = team_obj_in_athlete.get("displayName") if isinstance(team_obj_in_athlete, dict) else None

        # Player Name (after getting athlete obj)
        player_display_name = athlete.get("displayName") if isinstance(athlete, dict) else None

        # Status Info
        type_info = rec.get("type", {});
        injury_status = type_info.get("description"); injury_status_abbr = type_info.get("abbreviation")

        # Injury Details
        details = rec.get("details", {});
        injury_type = details.get("type"); injury_location = details.get("location");
        injury_detail = details.get("detail"); injury_side = details.get("side");
        return_date_est = details.get("returnDate")

        # Comments
        short_comment = rec.get("shortComment"); long_comment = rec.get("longComment")

        # Convert IDs to TEXT right away
        injury_id_str = str(injury_id) if injury_id is not None else None
        player_id_str = str(player_id) if player_id is not None else None
        team_id_str = str(team_id) if team_id is not None else None

        # --- Validation ---
        validation_check = all([player_id_str, team_id_str, report_date_utc])
        if not validation_check:
            print(f"Warn: Skipping injury {injury_id_str}. Missing player_id ('{player_id_str}'), team_id ('{team_id_str}'), or report_date ('{report_date_utc}').")
            return None
        # --- End Validation ---

        transformed = {
            "injury_id": injury_id_str, "player_id": player_id_str, "player_display_name": player_display_name,
            "team_id": team_id_str, "team_display_name": team_display_name,
            "report_date_utc": report_date_utc,
            "injury_status": injury_status, "injury_status_abbr": injury_status_abbr, "injury_type": injury_type,
            "injury_location": injury_location, "injury_detail": injury_detail, "injury_side": injury_side,
            "return_date_est": return_date_est if return_date_est else None, "short_comment": short_comment,
            "long_comment": long_comment, "last_api_update_time": report_date_utc,
            "raw_api_response": json.dumps(rec) # Store original record + added keys
        }
        return transformed
    except Exception as e:
        # --- FIXED: Use 'rec' here instead of 'record' ---
        print(f"Error transforming individual injury record id {rec.get('id', 'UNKNOWN')}: {e}")
        # --- End FIX ---
        return None

def transform_injury_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms a top-level TEAM injury record by processing its nested "injuries" list.
    Returns a list of transformed individual injury records.
    """
    transformed_records = []
    # Get team info from the top-level record
    team_id = record.get("id")
    team_display_name = record.get("displayName")

    if "injuries" in record and isinstance(record["injuries"], list):
        nested_injuries = record["injuries"]
        # print(f"Record ID {team_id} contains nested 'injuries' list with {len(nested_injuries)} item(s).") # Optional log

        for nested_rec in nested_injuries:
            if isinstance(nested_rec, dict):
                 # --- REMOVED Athlete Injection Logic ---
                 # Add team info directly for the inner transform function to use
                 nested_rec['team_id_added'] = team_id
                 nested_rec['team_displayName_added'] = team_display_name
                 # --- End Modification ---

                 # Call the function that transforms the *individual* injury record
                 transformed = _transform_single_injury(nested_rec)
                 if transformed:
                     transformed_records.append(transformed)
            else:
                 print(f"Warning: Found non-dictionary item in nested injuries list for team {team_id}.")

    # This handles cases where the top-level record might BE an injury record itself
    # (less likely based on logs, but safe fallback)
    # Check if the record has keys typical of an individual injury report AND not the team structure keys
    elif 'athlete' in record and 'details' in record and 'type' in record and 'date' in record:
         print(f"Record ID {record.get('id')} looks like an individual record. Processing directly.")
         # Need to somehow get team context if possible, or leave as None?
         # This path is less likely given the API structure observed.
         # For safety, let's assume team info might be missing here.
         record['team_id_added'] = record.get('athlete',{}).get('team',{}).get('id')
         record['team_displayName_added'] = record.get('athlete',{}).get('team',{}).get('displayName')
         transformed = _transform_single_injury(record)
         if transformed:
             transformed_records.append(transformed)
    # else: # Optional log if record is neither structure
         # print(f"Record ID {record.get('id')} does not match expected team or individual injury structure.")

    return transformed_records

def upsert_nba_injuries(supabase_url: str, supabase_anon_key: str, injuries: List[Dict[str, Any]]) -> int:
    """
    Upserts NBA injury records into the Supabase table 'nba_injuries'.
    Returns the number of records updated/inserted.
    """
    if not injuries:
        print("No injury records received.")
        return 0
    supabase: Client = create_client(supabase_url, supabase_anon_key)
    upsert_payload = []
    print(f"Transforming {len(injuries)} raw top-level record(s)...")
    for rec in injuries:
        transformed_list = transform_injury_record(rec)
        if transformed_list:
            upsert_payload.extend(transformed_list)
    if not upsert_payload:
        print("No valid records after transformation.")
        return 0
    print(f"Attempting to upsert {len(upsert_payload)} valid injury record(s)...")
    try:
        response = supabase.table("nba_injuries").upsert(upsert_payload, on_conflict="injury_id").execute()
        updated_count = 0
        if hasattr(response, "data") and response.data:
            updated_count = len(response.data)
            print(f"Upsert success: {updated_count} rows affected.")
        elif hasattr(response, 'error') and response.error:
            print(f"Supabase Upsert Error: {response.error}")
        else:
            print("Upsert complete (may have had no changes).")
        return updated_count
    except Exception as e:
        print(f"Error during Supabase upsert: {e}")
        return 0

if __name__ == "__main__":
    rapidapi_key, rapidapi_host, supabase_url, supabase_anon_key = load_environment()
    
    injuries_raw = get_nba_injuries(rapidapi_key, rapidapi_host)
    if not injuries_raw or not isinstance(injuries_raw, dict):
        print("Failed to retrieve valid NBA injuries data.")
        exit(1)
    
    # Determine which top-level key holds the injury records.
    injuries_list = None
    possible_keys = ["injuries", "response", "data", "items"]
    actual_keys = list(injuries_raw.keys())
    print(f"Debug: Top-level keys in API response: {actual_keys}")
    
    for key in possible_keys:
        potential_list = injuries_raw.get(key)
        if isinstance(potential_list, list):
            injuries_list = potential_list
            print(f"Found injury list under key: '{key}'")
            break
    if injuries_list is None and isinstance(injuries_raw, list):
        injuries_list = injuries_raw
        print("Found injury list directly at the root of the response.")
    
    if injuries_list is None:
        print("Error: Could not locate the list of injury records in the API response.")
        print(f"Response sample: {str(injuries_raw)[:500]}...")
        exit(1)
    
    print(f"\nRetrieved {len(injuries_list)} top-level record(s) from API.")
    if injuries_list:
        print("\n--- Sample Raw API Record (First 1) ---")
        print(json.dumps(injuries_list[0], indent=2))
        print("--- End Sample ---")
    else:
        print("The injury list is empty. Nothing to process.")
        exit(0)
    
    updated_count = upsert_nba_injuries(supabase_url, supabase_anon_key, injuries_list)
    print(f"Finished upserting injuries. Updated/Inserted {updated_count} records in Supabase table 'nba_injuries'.")
