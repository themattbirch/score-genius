# backend/data_pipeline/mlb_games_preview.py

import requests
import os
import json
import time
from supabase import create_client, Client
import datetime
from datetime import date, time as dt_time, timedelta # Use alias for time class
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional, Any

# --- Import variables directly from config.py ---
# This script assumes config.py is in the parent directory (../)
# and that config.py correctly loads .env from the project root.
# NO try...except fallback here - let it fail if import doesn't work.
try:
    from config import (
        RAPIDAPI_KEY,
        RAPIDAPI_HOST,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_ANON_KEY
    )
    print("Successfully imported variables from config.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import variables from config.py: {e}")
    print("Ensure config.py exists in the backend directory and there are no syntax errors.")
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during config import: {e}")
    exit()


# --- Configuration ---
# Constants using imported variables
MLB_ODDS_URL = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds' # API Key added in params
SUPABASE_TABLE_NAME = "mlb_game_schedule" # Target table for MLB previews
ET_ZONE = ZoneInfo("America/New_York")
PREFERRED_BOOKMAKER_KEY = 'draftkings'
REQUEST_DELAY_SECONDS = 2 # Delay between RapidAPI calls
# BATCH_SIZE = 50 # Batching not strictly necessary for daily preview, upsert all at once

# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    """Normalizes a team name."""
    if not name: return ""
    name = name.replace("St.", "St")
    return ' '.join(name.split()).lower()

def title_case_team_name(name: str) -> str:
    """Converts a normalized team name back to title case."""
    return ' '.join(word.capitalize() for word in name.split())

def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely converts a value to float, returning None on failure."""
    if value is None: return None
    try:
        if isinstance(value, str): value = value.replace('%', '')
        return float(value)
    except (ValueError, TypeError): return None

# --- API Fetching Functions ---
def make_rapidapi_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request to the RapidAPI endpoint."""
    print(f"Attempting RapidAPI fetch from: {url}")
    print(f"Using parameters: {params}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        print(f"RapidAPI Request successful.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching from RapidAPI: {http_err}")
        print(f"Response Status Code: {http_err.response.status_code}")
        print(f"Response Body: {http_err.response.text[:500]}...")
    except requests.exceptions.RequestException as e:
        print(f"Network or Request Error fetching from RapidAPI: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from RapidAPI URL {url}.")
    return None

def get_schedule_and_pitchers(target_date: date) -> List[Dict[str, Any]]:
    """Fetches MLB schedule and probable pitchers from RapidAPI Sports for a specific date."""
    if not RAPIDAPI_KEY or not RAPIDAPI_HOST: # Use imported variables
        print("Error: RapidAPI Key or Host not available from config.")
        return []

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    endpoint_path = "/mlb/schedule"
    url = f"https://{RAPIDAPI_HOST}{endpoint_path}"
    params = {
        "year": target_date.year,
        "month": f"{target_date.month:02d}",
        "day": f"{target_date.day:02d}"
    }
    data = make_rapidapi_request(url, headers, params)

    if data and isinstance(data.get("events"), list):
        print(f"Found {len(data['events'])} events from RapidAPI schedule.")
        return data["events"]
    elif isinstance(data, list):
        print(f"Found {len(data)} events directly in RapidAPI response list.")
        return data
    else:
        print(f"Could not find 'events' list in RapidAPI response for {target_date}. Response: {str(data)[:200]}")
        return []


def get_betting_odds(target_date: datetime.datetime) -> List[Dict[str, Any]]:
    """Fetches betting odds for MLB events from The Odds API for a specific date (UTC)."""
    if not ODDS_API_KEY: # Use imported variable
        print("Error: ODDS_API_KEY not available from config.")
        return []

    utc_zone = ZoneInfo("UTC")
    start_of_day_utc = datetime.datetime.combine(target_date.date(), dt_time.min).replace(tzinfo=utc_zone)
    end_of_day_utc = datetime.datetime.combine(target_date.date(), dt_time.max).replace(tzinfo=utc_zone)

    commence_time_from = start_of_day_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = end_of_day_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apiKey": ODDS_API_KEY, # Use imported variable
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": commence_time_from,
        "commenceTimeTo": commence_time_to,
    }
    mlb_url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds'
    print(f"Fetching odds from: {mlb_url}")
    try:
        response = requests.get(mlb_url, params=params)
        response.raise_for_status()
        odds_data = response.json()
        print(f"Fetched {len(odds_data)} odds events.")
        return odds_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching betting odds: {http_err} - Status Code: {response.status_code}")
        if response.status_code == 422: print(f"Odds API response body: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Network or Request Error fetching betting odds: {e}")
    except Exception as e:
        print(f"Unexpected error in get_betting_odds: {e}")
    return []

# --- Matching and Extraction ---
# (Keep match_odds_for_game and extract_odds_data functions as defined previously)
def match_odds_for_game(rapidapi_game_event: Dict[str, Any], odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Attempts to match an Odds API event to a RapidAPI game event by team names and date (ET)."""
    # ... (implementation from previous response) ...
    if not odds_events or not rapidapi_game_event: return None
    competitions = rapidapi_game_event.get('competitions', [])
    if not competitions: return None
    competitors = competitions[0].get('competitors', [])
    if len(competitors) != 2: return None
    home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), None)
    away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), None)
    if not home_comp or not away_comp: return None
    game_home_name = normalize_team_name(home_comp.get('team', {}).get('displayName', ''))
    game_away_name = normalize_team_name(away_comp.get('team', {}).get('displayName', ''))
    game_date_str_utc = rapidapi_game_event.get('date')
    game_local_date_et = None
    try:
        game_dt_utc = dateutil_parser.isoparse(game_date_str_utc)
        game_local_date_et = game_dt_utc.astimezone(ET_ZONE).date()
    except: return None
    if not game_home_name or not game_away_name: return None
    for odds_event in odds_events:
        odds_home = normalize_team_name(odds_event.get('home_team', ''))
        odds_away = normalize_team_name(odds_event.get('away_team', ''))
        if game_home_name == odds_home and game_away_name == odds_away:
            commence_time_str = odds_event.get('commence_time', '')
            try:
                event_dt_utc = dateutil_parser.isoparse(commence_time_str)
                event_local_date_et = event_dt_utc.astimezone(ET_ZONE).date()
                if game_local_date_et == event_local_date_et:
                    print(f"Matched Odds: {title_case_team_name(game_away_name)} @ {title_case_team_name(game_home_name)} on {game_local_date_et}")
                    return odds_event
            except: continue
    return None

def extract_odds_data(odds_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts raw odds (for JSONB) and clean numeric odds from a matched odds event."""
    # ... (implementation from previous response) ...
    raw_odds = {"moneyline": {}, "spread": {}, "total": {}}
    clean_odds = {"moneyline_home_clean": None, "moneyline_away_clean": None,"spread_home_line_clean": None, "spread_home_price_clean": None, "spread_away_price_clean": None,"total_line_clean": None, "total_over_price_clean": None, "total_under_price_clean": None}
    if not odds_event: return {"raw": raw_odds, "clean": clean_odds}
    bookmakers = odds_event.get('bookmakers', [])
    if not bookmakers: return {"raw": raw_odds, "clean": clean_odds}
    target_bookmaker = next((bm for bm in bookmakers if bm.get('key') == PREFERRED_BOOKMAKER_KEY), bookmakers[0])
    if not target_bookmaker: return {"raw": raw_odds, "clean": clean_odds}
    raw_odds["bookmaker_key"] = target_bookmaker.get("key")
    odds_home_team = title_case_team_name(odds_event.get('home_team', ''))
    odds_away_team = title_case_team_name(odds_event.get('away_team', ''))
    markets = target_bookmaker.get('markets', [])
    for market in markets:
        market_key = market.get('key')
        outcomes = market.get('outcomes', [])
        if not isinstance(outcomes, list): continue
        if market_key == 'h2h':
            raw_odds['moneyline'] = {title_case_team_name(o.get('name','')): o.get('price') for o in outcomes if o.get('name') and o.get('price') is not None}
            for o in outcomes:
                 team = title_case_team_name(o.get('name',''))
                 price = safe_float_conversion(o.get('price'))
                 if team == odds_home_team: clean_odds['moneyline_home_clean'] = price
                 elif team == odds_away_team: clean_odds['moneyline_away_clean'] = price
        elif market_key == 'spreads':
             raw_odds['spread'] = {title_case_team_name(o.get('name','')): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') and o.get('price') is not None and o.get('point') is not None}
             for o in outcomes:
                  team = title_case_team_name(o.get('name',''))
                  price = safe_float_conversion(o.get('price'))
                  point = safe_float_conversion(o.get('point'))
                  if team == odds_home_team:
                      clean_odds['spread_home_line_clean'] = point
                      clean_odds['spread_home_price_clean'] = price
                  elif team == odds_away_team:
                      clean_odds['spread_away_price_clean'] = price
        elif market_key == 'totals':
             raw_odds['total'] = {o.get('name',''): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') in ['Over', 'Under'] and o.get('price') is not None and o.get('point') is not None}
             for o in outcomes:
                 pos = o.get('name'); price = safe_float_conversion(o.get('price')); point = safe_float_conversion(o.get('point'))
                 clean_odds['total_line_clean'] = point
                 if pos == 'Over': clean_odds['total_over_price_clean'] = price
                 elif pos == 'Under': clean_odds['total_under_price_clean'] = price
    return {"raw": raw_odds, "clean": clean_odds}

# --- Main Build & Upsert ---
def build_and_upsert_mlb_previews():
    """Builds and upserts MLB game previews."""
    print("\n--- Running MLB Game Preview Script ---")
    script_start_time = time.time()

    # Check if essential config variables were loaded
    if not all([RAPIDAPI_KEY, RAPIDAPI_HOST, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY]):
         print("FATAL ERROR: One or more critical config variables not loaded. Exiting.")
         exit()

    eastern_now = datetime.datetime.now(ET_ZONE)
    target_date_et = eastern_now.date()
    print(f"\nStep 1: Fetching schedule & pitchers for date (ET): {target_date_et}")
    rapidapi_game_events = get_schedule_and_pitchers(target_date_et)

    if not rapidapi_game_events:
        print("No game events found from RapidAPI schedule. Exiting.")
        return 0

    print(f"\nStep 2: Fetching betting odds for date (ET): {target_date_et}")
    odds_events = get_betting_odds(eastern_now)

    print("\nStep 3: Processing games and matching odds...")
    previews_to_upsert = []
    processed_game_ids = set()

    for event in rapidapi_game_events:
        # (Keep the extraction logic from the previous version)
        try:
            game_id_str = event.get("id")
            if not game_id_str: continue
            try: game_id = int(game_id_str)
            except ValueError: continue
            if game_id in processed_game_ids: continue
            processed_game_ids.add(game_id)
            status_state = event.get('status', {}).get('type', {}).get('state', '').lower()
            if status_state != 'pre': continue

            competitions = event.get('competitions', [])
            if not competitions: continue
            competition = competitions[0]
            competitors = competition.get('competitors', [])
            if len(competitors) != 2: continue
            home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            home_team = home_comp.get('team', {})
            away_team = away_comp.get('team', {})
            venue = competition.get('venue', {})
            address = venue.get('address', {})
            status_type = event.get('status', {}).get('type', {})
            home_probable = home_comp.get('probables', [{}])[0]
            away_probable = away_comp.get('probables', [{}])[0]
            home_pitcher = home_probable.get('athlete', {})
            away_pitcher = away_probable.get('athlete', {})
            game_date_et_str = None
            scheduled_time_utc_str = event.get('date')
            if scheduled_time_utc_str:
                 try:
                      game_dt_utc = dateutil_parser.isoparse(scheduled_time_utc_str)
                      game_date_et_str = game_dt_utc.astimezone(ET_ZONE).date().isoformat()
                 except Exception: pass
            matched_odds_event = match_odds_for_game(event, odds_events)
            extracted_odds = extract_odds_data(matched_odds_event)
            preview_data = {
                'game_id': game_id,
                'game_uid': event.get('uid'),
                'scheduled_time_utc': scheduled_time_utc_str,
                'game_date_et': game_date_et_str,
                'status_detail': status_type.get('shortDetail') or status_type.get('detail'),
                'status_state': status_type.get('state'),
                'home_team_id': home_team.get('id'),
                'home_team_name': home_team.get('displayName'),
                'home_team_abbr': home_team.get('abbreviation'),
                'away_team_id': away_team.get('id'),
                'away_team_name': away_team.get('displayName'),
                'away_team_abbr': away_team.get('abbreviation'),
                'venue_name': venue.get('fullName'),
                'venue_city': address.get('city'),
                'venue_state': address.get('state'),
                'venue_is_indoor': venue.get('indoor'),
                'home_probable_pitcher_id': home_pitcher.get('id'),
                'home_probable_pitcher_name': home_pitcher.get('fullName'),
                'home_probable_pitcher_record': home_probable.get('record'),
                'away_probable_pitcher_id': away_pitcher.get('id'),
                'away_probable_pitcher_name': away_pitcher.get('fullName'),
                'away_probable_pitcher_record': away_probable.get('record'),
                'moneyline': extracted_odds['raw'].get('moneyline'),
                'spread': extracted_odds['raw'].get('spread'),
                'total': extracted_odds['raw'].get('total'),
                'moneyline_home_clean': extracted_odds['clean'].get('moneyline_home_clean'),
                'moneyline_away_clean': extracted_odds['clean'].get('moneyline_away_clean'),
                'spread_home_line_clean': extracted_odds['clean'].get('spread_home_line_clean'),
                'spread_home_price_clean': extracted_odds['clean'].get('spread_home_price_clean'),
                'spread_away_price_clean': extracted_odds['clean'].get('spread_away_price_clean'),
                'total_line_clean': extracted_odds['clean'].get('total_line_clean'),
                'total_over_price_clean': extracted_odds['clean'].get('total_over_price_clean'),
                'total_under_price_clean': extracted_odds['clean'].get('total_under_price_clean'),
                'raw_api_response': json.dumps(event)
            }
            previews_to_upsert.append(preview_data)
        except Exception as e:
            print(f"Error processing RapidAPI event {event.get('id', 'UNKNOWN')}: {e}")


    # Upsert the processed previews
    if previews_to_upsert:
        print(f"\nStep 4: Upserting {len(previews_to_upsert)} processed previews to Supabase...")
        try:
            # Initialize Supabase client *inside* the block that uses it
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(
                previews_to_upsert,
                on_conflict='game_id'
            ).execute()
            print(f"Supabase upsert attempt finished. Response snippet: {str(upsert_response)[:200]}")
        except Exception as e:
            print(f"Error during Supabase upsert call: {e}")
            if hasattr(e, 'message'): print(f"Supabase error message: {e.message}")
    else:
        print("\nStep 4: No valid game previews generated to upsert.")

    script_end_time = time.time()
    print(f"\n--- MLB Preview Script finished. Processed {len(processed_game_ids)} unique RapidAPI events. ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    return len(previews_to_upsert)

# --- Cleanup Function ---
def clear_old_mlb_games(): # Keep implementation separate
    """Clears games from mlb_game_schedule older than today (ET)."""
    print("\nAttempting to clear old games from mlb_game_schedule...")
    # Initialize Supabase client inside function
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    # (Keep implementation from previous version)
    try:
        response = supabase.table(SUPABASE_TABLE_NAME).select("game_id, scheduled_time_utc").execute()
        if not hasattr(response, 'data') or response.data is None: print(f"Clearing check: No games found or error. Response: {response}"); return
        rows = response.data
        if not rows: print("Clearing check: No games found in table."); return
        today_et_date = datetime.datetime.now(ET_ZONE).date()
        ids_to_delete = []
        for row in rows:
            game_id = row.get("game_id"); scheduled_time_str = row.get("scheduled_time_utc")
            if not scheduled_time_str or not game_id: continue
            try:
                dt_scheduled_utc = dateutil_parser.isoparse(scheduled_time_str)
                if dt_scheduled_utc.astimezone(ET_ZONE).date() < today_et_date: ids_to_delete.append(game_id)
            except Exception as e: print(f"Error parsing date '{scheduled_time_str}' for game {game_id} during cleanup: {e}")
        if ids_to_delete:
            print(f"Attempting to batch delete {len(ids_to_delete)} old MLB games...")
            delete_response = supabase.table(SUPABASE_TABLE_NAME).delete().in_("game_id", ids_to_delete).execute()
            print(f"Batch delete attempted. Response snippet: {str(delete_response)[:200]}")
        else: print("No old MLB games found to delete.")
    except Exception as e: print(f"Error during Supabase query/delete in clear_old_mlb_games: {e}")


if __name__ == "__main__":
    # Decide whether to run cleanup first
    run_cleanup = True # Set to False to skip cleanup
    if run_cleanup:
        clear_old_mlb_games()

    # Run the main preview builder and upsert process
    build_and_upsert_mlb_previews()

    print("\nDirect execution finished.")