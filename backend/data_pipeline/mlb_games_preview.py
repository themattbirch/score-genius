# backend/data_pipeline/mlb_games_preview.py

import requests
import os
import json
import time
from supabase import create_client, Client
import datetime
from datetime import date, time as dt_time, timedelta, datetime as dt_datetime
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional, Any

# --- Import variables directly from config.py ---
# Only need keys for api-baseball and The Odds API now
try:
    from config import (
        API_SPORTS_KEY,  # For api-baseball schedule
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
        RAPIDAPI_KEY,
        RAPIDAPI_HOST
    )
    print("Successfully imported variables from config.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import variables from config.py: {e}")
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during config import: {e}")
    exit()


# --- Configuration ---
API_BASEBALL_URL = "https://v1.baseball.api-sports.io"
MLB_ODDS_URL = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds'
SUPABASE_TABLE_NAME = "mlb_game_schedule"
ET_ZONE = ZoneInfo("America/New_York")
PREFERRED_BOOKMAKER_KEY = 'draftkings'
REQUEST_DELAY_SECONDS = 1 # Delay between api-baseball calls
MLB_LEAGUE_ID = 1
# --- End Configuration ---


# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    if not name: return ""
    name = name.replace("St.", "St"); return ' '.join(name.split()).lower()
def title_case_team_name(name: str) -> str: return ' '.join(word.capitalize() for word in name.split())
def safe_float_conversion(value: Any) -> Optional[float]:
    if value is None: return None
    try:
        if isinstance(value, str): value = value.replace('%', '')
        return float(value)
    except: return None
def format_american_odds(numeric_odds: Optional[float]) -> Optional[str]:
    """Formats numeric odds into American odds string (e.g., +150, -110)."""
    if numeric_odds is None: return None
    odds_int = int(round(numeric_odds))
    if odds_int > 0: return f"+{odds_int}"
    else: return f"{odds_int}"

# --- API Fetching Functions ---
def make_api_request(url: str, headers: dict, params: dict):
    """ Generic API request helper """
    print(f"Querying {url.split('?')[0]} with params {params}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status(); print("Request successful.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error: {http_err} - Status: {http_err.response.status_code}")
        print(f"Body: {http_err.response.text[:500]}...")
    except Exception as e: print(f"Error in make_api_request: {e}")
    return None

def make_rapidapi_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request to the RapidAPI endpoint."""
    print(f"Attempting RapidAPI fetch from: {url}")
    print(f"Using parameters: {params}")
    # Optional: Mask key in log if desired
    # print(f"Using Host: {headers.get('x-rapidapi-host')}")
    # print(f"Using Key starting: {headers.get('x-rapidapi-key', '')[:5]}...")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Check for HTTP errors
        # RapidAPI typically includes usage in headers
        remaining = response.headers.get('X-RateLimit-Requests-Remaining')
        limit = response.headers.get('X-RateLimit-Requests-Limit')
        print(f"RapidAPI Request successful. Usage: Remaining={remaining}, Limit={limit}")
        # Assume JSON response
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching from RapidAPI: {http_err} - Status Code: {http_err.response.status_code}")
        print(f"Response Body: {http_err.response.text[:500]}...") # Log error body
        if http_err.response.status_code in [401, 403]:
             print("NOTE: Received 401/403 Unauthorized/Forbidden. Check RAPIDAPI_KEY, RAPIDAPI_HOST, and your subscription.")
        elif http_err.response.status_code == 404:
             print("NOTE: Received 404 Not Found. The endpoint path might be incorrect.")
    except requests.exceptions.RequestException as e:
        print(f"Network or Request Error fetching from RapidAPI: {e}")
    except json.JSONDecodeError:
        # Log raw response if JSON decoding fails
        print(f"Error decoding JSON response from RapidAPI URL {url}. Raw response text:")
        # Ensure 'response' exists before trying to access .text
        if 'response' in locals() and hasattr(response, 'text'):
             print(response.text[:1000] + "...")
    except Exception as e:
        print(f"An unexpected error occurred in make_rapidapi_request: {e}")
    return None


def get_games_from_apibaseball(target_date: date) -> List[Dict[str, Any]]:
    """Fetches MLB schedule from api-baseball for a specific date."""
    if not API_SPORTS_KEY: print("Error: API_SPORTS_KEY not configured."); return []
    headers = {'x-apisports-key': API_SPORTS_KEY}
    url = f"{API_BASEBALL_URL}/games"
    params = {"league": MLB_LEAGUE_ID, "date": target_date.strftime("%Y-%m-%d"), "season": target_date.year}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list):
        print(f"Found {len(data['response'])} games from api-baseball for {target_date}.")
        return data["response"]
    else:
        print(f"No games found or error fetching from api-baseball for {target_date}.")
        return []
    
  # Replace the existing get_schedule_and_pitchers function with this one

def get_schedule_and_pitchers(actual_target_date: date) -> List[Dict[str, Any]]:
    """
    Fetches MLB schedule from RapidAPI Sports by QUERYING FOR target_date - 2 days,
    attempting to get pitcher info for the actual_target_date.
    Returns the 'events' list if available and detailed, otherwise empty list.
    """
    # Calculate the date to use in the API query parameters based on the offset
    query_date = actual_target_date - timedelta(days=2)
    print(f"Attempting to fetch pitcher data for actual date {actual_target_date} by querying RapidAPI for offset date {query_date}...")

    # Ensure config variables are available (should be imported at top level)
    if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
        print("Error: RapidAPI Key/Host missing in get_schedule_and_pitchers.")
        return []

    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    endpoint_path = "/mlb/schedule"
    url = f"https://{RAPIDAPI_HOST}{endpoint_path}"
    params = {
        "year": query_date.year,
        "month": f"{query_date.month:02d}",
        "day": f"{query_date.day:02d}"
    }

    # Assumes make_rapidapi_request function exists and handles basic errors
    data = make_rapidapi_request(url, headers, params)

    # Check specifically for the 'events' list which contains detailed data
    if data and isinstance(data.get("events"), list):
        events = data["events"]
        print(f"Found {len(events)} detailed events in RapidAPI response when querying for {query_date} (intended for {actual_target_date}).")
        # Optional: Filter events here if API returns data for multiple dates from offset query
        return events
    elif data and isinstance(data.get(query_date.strftime('%Y%m%d')), dict):
        print(f"Found basic calendar structure, but no detailed 'events' when querying for {query_date}.")
        return [] # Return empty list as detailed data is missing
    else:
        print(f"Could not find 'events' list or expected structure in RapidAPI response when querying for {query_date}.")
        return [] # Return empty list for unexpected structures

def get_betting_odds(target_date_dt: dt_datetime) -> List[Dict[str, Any]]:
    """Fetches betting odds for MLB events from The Odds API covering today/tomorrow ET."""
    if not ODDS_API_KEY: print("Error: ODDS_API_KEY not available from config."); return []
    utc_zone = ZoneInfo("UTC")
    start_utc = dt_datetime.combine(target_date_dt.date(), dt_time.min).replace(tzinfo=utc_zone)
    end_utc = dt_datetime.combine(target_date_dt.date() + timedelta(days=1), dt_time.max).replace(tzinfo=utc_zone)
    commence_time_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "commenceTimeFrom": commence_time_from, "commenceTimeTo": commence_time_to}
    print(f"Fetching odds from: {MLB_ODDS_URL} between {commence_time_from} and {commence_time_to}")
    try:
        response = requests.get(MLB_ODDS_URL, params=params)
        response.raise_for_status(); odds_data = response.json()
        print(f"Fetched {len(odds_data)} odds events.")
        return odds_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching odds: {http_err} - Status: {response.status_code}")
        if response.status_code == 422: print(f"Odds API body: {response.text}")
    except Exception as e: print(f"Error in get_betting_odds: {e}")
    return []

# --- Matching and Extraction ---
def match_odds_for_apibaseball_game(apibaseball_game: Dict[str, Any], odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Matches Odds API event to api-baseball game event by team names and date (ET)."""
    if not odds_events or not apibaseball_game: return None
    teams_info=apibaseball_game.get('teams',{}); home_team_info=teams_info.get('home',{}); away_team_info=teams_info.get('away',{})
    game_home_name=normalize_team_name(home_team_info.get('name','')); game_away_name=normalize_team_name(away_team_info.get('name',''))
    game_date_str_utc=apibaseball_game.get('date'); game_local_date_et=None
    try: game_dt_utc=dateutil_parser.isoparse(game_date_str_utc); game_local_date_et=game_dt_utc.astimezone(ET_ZONE).date()
    except: print(f"Warn: Could not parse api-baseball date {game_date_str_utc}"); return None
    if not game_home_name or not game_away_name: return None
    for odds_event in odds_events:
        odds_home=normalize_team_name(odds_event.get('home_team','')); odds_away=normalize_team_name(odds_event.get('away_team',''))
        if game_home_name == odds_home and game_away_name == odds_away:
            commence_time_str=odds_event.get('commence_time','')
            try:
                event_dt_utc=dateutil_parser.isoparse(commence_time_str); event_local_date_et=event_dt_utc.astimezone(ET_ZONE).date()
                if game_local_date_et == event_local_date_et:
                    print(f"Matched Odds: {title_case_team_name(game_away_name)} @ {title_case_team_name(game_home_name)} on {game_local_date_et}")
                    return odds_event
            except: continue
    return None

def extract_odds_data(odds_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts raw odds (JSONB) and clean formatted odds strings."""
    # (Implementation remains the same, including format_american_odds call)
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
        mkey=market.get('key'); outcomes=market.get('outcomes',[])
        if not isinstance(outcomes,list): continue
        if mkey == 'h2h':
            raw_odds['moneyline'] = {title_case_team_name(o.get('name','')): o.get('price') for o in outcomes if o.get('name') and o.get('price') is not None}
            for o in outcomes:
                 team=title_case_team_name(o.get('name','')); price=safe_float_conversion(o.get('price'))
                 fmt_price = format_american_odds(price) # Format here
                 if team == odds_home_team: clean_odds['moneyline_home_clean'] = fmt_price
                 elif team == odds_away_team: clean_odds['moneyline_away_clean'] = fmt_price
        elif mkey == 'spreads':
             raw_odds['spread'] = {title_case_team_name(o.get('name','')): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') and o.get('price') is not None and o.get('point') is not None}
             for o in outcomes:
                  team=title_case_team_name(o.get('name','')); price=safe_float_conversion(o.get('price')); point=safe_float_conversion(o.get('point'))
                  fmt_price = format_american_odds(price) # Format here
                  if team == odds_home_team: clean_odds['spread_home_line_clean']=point; clean_odds['spread_home_price_clean']=fmt_price
                  elif team == odds_away_team: clean_odds['spread_away_price_clean']=fmt_price
        elif mkey == 'totals':
             raw_odds['total'] = {o.get('name',''): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') in ['Over', 'Under'] and o.get('price') is not None and o.get('point') is not None}
             num_total_line = None # Find line first
             for o in outcomes: num_total_line=safe_float_conversion(o.get('point')); break
             clean_odds['total_line_clean']=num_total_line
             for o in outcomes: # Format prices
                 pos=o.get('name'); price=safe_float_conversion(o.get('price')); fmt_price = format_american_odds(price)
                 if pos=='Over': clean_odds['total_over_price_clean']=fmt_price
                 elif pos=='Under': clean_odds['total_under_price_clean']=fmt_price
    return {"raw": raw_odds, "clean": clean_odds}

# Replace the existing build_and_upsert_mlb_previews function with this correct version

def build_and_upsert_mlb_previews():
    """
    Builds and upserts MLB game previews using api-baseball for schedule
    and The Odds API for odds. Pitcher info is set to NULL.
    """
    print("\n--- Running MLB Game Preview Script (api-baseball schedule + Odds API) ---")
    script_start_time = time.time()
    # Ensure necessary keys for THIS version are loaded
    if not all([API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY]):
         print("FATAL ERROR: Config variables missing (API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY). Exiting."); return 0

    eastern_now = dt_datetime.now(ET_ZONE)
    today_et = eastern_now.date()
    tomorrow_et = today_et + timedelta(days=1)

    # Step 1: Fetch base schedule from api-baseball
    print(f"\nStep 1a: Fetching schedule from api-baseball for date (ET): {today_et}")
    apibaseball_games_today = get_games_from_apibaseball(today_et)
    # Apply delay even if only one API source for schedule now
    time.sleep(REQUEST_DELAY_SECONDS)
    print(f"\nStep 1b: Fetching schedule from api-baseball for date (ET): {tomorrow_et}")
    apibaseball_games_tomorrow = get_games_from_apibaseball(tomorrow_et)
    apibaseball_games = apibaseball_games_today + apibaseball_games_tomorrow

    if not apibaseball_games:
        print(f"No game events found from api-baseball schedule for {today_et} or {tomorrow_et}. Exiting.")
        return 0

    # Step 2: Fetch odds from The Odds API
    print(f"\nStep 2: Fetching betting odds (covering {today_et} & {tomorrow_et} ET)")
    odds_events = get_betting_odds(eastern_now)

    # Step 3: Process games and match odds (No pitcher lookup needed in this version)
    print("\nStep 3: Processing games and matching odds...")
    previews_to_upsert = []
    processed_game_ids = set()

    for game in apibaseball_games: # Loop through api-baseball results
        try:
            game_id = game.get("id")
            if not game_id or game_id in processed_game_ids: continue
            processed_game_ids.add(game_id)
            status_info = game.get('status', {}); status_short = status_info.get('short', '').upper()
            if status_short != 'NS': continue # Only scheduled games

            teams_info=game.get('teams',{}); home_team_info=teams_info.get('home',{}); away_team_info=teams_info.get('away',{})
            home_team_name_raw=home_team_info.get('name'); away_team_name_raw=away_team_info.get('name')
            venue_info = game.get('venue', {}); venue_name = venue_info.get('name') if isinstance(venue_info, dict) else None
            venue_city = venue_info.get('city') if isinstance(venue_info, dict) else None
            scheduled_time_utc_str=game.get('date'); game_date_et_str=None
            if scheduled_time_utc_str:
                 try: game_dt_utc=dateutil_parser.isoparse(scheduled_time_utc_str); game_date_et_str=game_dt_utc.astimezone(ET_ZONE).date().isoformat()
                 except Exception as e: print(f"Warn: Could not parse date {scheduled_time_utc_str} for game {game_id}: {e}")

            # Match odds using api-baseball game info
            # Ensure the function name here matches the one defined in your script
            matched_odds_event=match_odds_for_apibaseball_game(game, odds_events)
            # Extract odds (raw and clean formatted)
            extracted_odds=extract_odds_data(matched_odds_event)

            preview_data = {
                'game_id': game_id, 'game_uid': None,
                'scheduled_time_utc': scheduled_time_utc_str, 'game_date_et': game_date_et_str,
                'status_detail': status_info.get('long'), 'status_state': 'pre' if status_short == 'NS' else status_short,
                'home_team_id': home_team_info.get('id'), 'home_team_name': home_team_name_raw, 'home_team_abbr': None,
                'away_team_id': away_team_info.get('id'), 'away_team_name': away_team_name_raw, 'away_team_abbr': None,
                'venue_name': venue_name, 'venue_city': venue_city, 'venue_state': None, 'venue_is_indoor': None,
                # --- Set Pitcher info to NULL ---
                'home_probable_pitcher_id': None, 'home_probable_pitcher_name': None, 'home_probable_pitcher_record': None,
                'away_probable_pitcher_id': None, 'away_probable_pitcher_name': None, 'away_probable_pitcher_record': None,
                # --- Odds Data ---
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
                'raw_api_response': json.dumps(game) # Store api-baseball game object
            }
            if preview_data['game_id'] and preview_data['home_team_name'] and preview_data['away_team_name']:
                 previews_to_upsert.append(preview_data)
            else: print(f"Warning: Skipping game due to missing essential fields. ID: {preview_data.get('game_id')}")
        except Exception as e: print(f"Error processing api-baseball game {game.get('id', 'UNKNOWN')}: {e}")

    # --- Upsert Logic ---
    if previews_to_upsert:
        # Changed step number to 4 since pitcher fetch is removed
        print(f"\nStep 4: Upserting {len(previews_to_upsert)} processed previews to Supabase...")
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            # Upsert using api-baseball game_id as the conflict target
            upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(previews_to_upsert, on_conflict='game_id').execute()
            print(f"Supabase upsert attempt finished.")
        except Exception as e: print(f"Error during Supabase upsert call: {e}"); print(f"Supabase error message: {e.message}" if hasattr(e, 'message') else "")
    else: print("\nStep 4: No valid game previews generated to upsert.") # Changed step number

    script_end_time = time.time()
    # Update final log message
    print(f"\n--- MLB Preview Script finished. Processed {len(processed_game_ids)} unique api-baseball games from {today_et} and {tomorrow_et}. Pitcher data skipped. ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    return len(previews_to_upsert)

# --- Cleanup Function ---
def clear_old_mlb_games():
    # (Implementation remains the same)
    print("\nAttempting to clear old games from mlb_game_schedule...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    try:
        response = supabase.table(SUPABASE_TABLE_NAME).select("game_id, scheduled_time_utc").execute()
        if not hasattr(response, 'data') or response.data is None: print(f"Clearing check: No games/error. Resp: {response}"); return
        rows = response.data; today_et_date = dt_datetime.now(ET_ZONE).date(); ids_to_delete = []
        if not rows: print("Clearing check: No games found."); return
        for row in rows:
            game_id=row.get("game_id"); scheduled_time_str=row.get("scheduled_time_utc")
            if not scheduled_time_str or not game_id: continue
            try:
                if dateutil_parser.isoparse(scheduled_time_str).astimezone(ET_ZONE).date() < today_et_date: ids_to_delete.append(game_id)
            except Exception as e: print(f"Error parsing date '{scheduled_time_str}' for game {game_id} during cleanup: {e}")
        if ids_to_delete:
            print(f"Attempting batch delete {len(ids_to_delete)} old MLB games..."); delete_response = supabase.table(SUPABASE_TABLE_NAME).delete().in_("game_id", ids_to_delete).execute()
            print(f"Batch delete attempted.")
        else: print("No old MLB games found to delete.")
    except Exception as e: print(f"Error during Supabase query/delete in clear_old_mlb_games: {e}")

# --- __main__ block ---
if __name__ == "__main__":
    run_cleanup = True
    if run_cleanup:
        clear_old_mlb_games()
    build_and_upsert_mlb_previews()
    print("\nDirect execution finished.")