# backend/data_pipeline/mlb_games_preview.py
# --- Final Simplified Version ---
# Uses api-baseball for schedule, The Odds API for odds. Pitchers set to NULL.
# Removed unused RapidAPI functions and imports.

import requests
import os
import json
import time
from supabase import create_client, Client
# Keep main datetime import, but others are within functions
from datetime import date, time as dt_time, timedelta, datetime as dt_datetime
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional, Any

# --- Import variables directly from config.py ---
try:
    # Only import keys needed for this version
    from config import (
        API_SPORTS_KEY,  # For api-baseball schedule
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_ANON_KEY
    )
    print("Successfully imported variables from config.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import variables from config.py: {e}"); exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during config import: {e}"); exit()

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
# Replace the existing normalize_team_name function with this one

def normalize_team_name(name: str) -> str:
    """Normalizes team name: handles St./St Louis, Athletics, lowercase, std spacing."""
    if not name or not isinstance(name, str):
        return ""

    # --- Updated Handling ---
    # Specific replacements BEFORE lowercasing to handle variations like "St.Louis"
    name = name.replace("St.Louis", "St Louis") # Handle specific case without space after period
    name = name.replace("St.", "St") # Handle general case "St. " (like St. Paul Saints if needed later)

    name = name.lower() # NOW lowercase the result

    # Handle Athletics variations -> standardize to 'oakland athletics'
    if name == "athletics":
         name = "oakland athletics"
    # Add other team-specific fixes here if needed (e.g., "la angels" vs "los angeles angels")

    # Standardize spacing (removes leading/trailing/multiple spaces)
    result = " ".join(name.split())
    # --- End Updated Handling ---

    # Optional Debug Print:
    # print(f"Debug Norm: Input='{original_name_for_debug}', Output='{result}'")
    return result
def title_case_team_name(name: str) -> str:
    """Converts a normalized name back to Title Case."""
    return ' '.join(word.capitalize() for word in name.split())
def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely converts a value to float, handling None and errors."""
    if value is None: return None
    try:
        if isinstance(value, str): value = value.replace('%', '') # Handle percentage strings if needed
        return float(value)
    except (ValueError, TypeError): return None
def format_american_odds(numeric_odds: Optional[float]) -> Optional[str]:
    """Formats numeric odds into American odds string (e.g., +150, -110)."""
    if numeric_odds is None: return None
    try:
        odds_int = int(round(numeric_odds))
        return f"+{odds_int}" if odds_int > 0 else f"{odds_int}"
    except (ValueError, TypeError): return None # Handle potential round errors

# --- API Fetching Functions ---
def make_api_request(url: str, headers: dict, params: dict):
    """Generic API request helper (Used for api-baseball)."""
    print(f"Querying {url.split('?')[0]} with params {params}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status(); print("Request successful.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error: {http_err} - Status: {http_err.response.status_code}"); print(f"Body: {http_err.response.text[:500]}...")
    except Exception as e: print(f"Error in make_api_request: {e}")
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
    else: print(f"No games found or error fetching from api-baseball for {target_date}."); return []

def get_betting_odds(target_date_dt: dt_datetime) -> List[Dict[str, Any]]:
    """Fetches betting odds for MLB events from The Odds API covering today/tomorrow ET."""
    if not ODDS_API_KEY: print("Error: ODDS_API_KEY not available."); return []
    utc_zone = ZoneInfo("UTC")
    # Calculate UTC start/end times based on the target ET date
    start_utc = dt_datetime.combine(target_date_dt.date(), dt_time.min).astimezone(utc_zone)
    end_utc = dt_datetime.combine(target_date_dt.date() + timedelta(days=1), dt_time.max).astimezone(utc_zone)
    commence_time_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"); commence_time_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "commenceTimeFrom": commence_time_from, "commenceTimeTo": commence_time_to}
    print(f"Fetching odds from: {MLB_ODDS_URL} between {commence_time_from} and {commence_time_to}")
    try:
        response = requests.get(MLB_ODDS_URL, params=params)
        response.raise_for_status(); odds_data = response.json()
        print(f"Fetched {len(odds_data)} odds events."); return odds_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching odds: {http_err} - Status: {response.status_code}")
        if response.status_code == 422: print(f"Odds API body: {response.text}")
    except Exception as e: print(f"Error in get_betting_odds: {e}")
    return []

# --- Matching and Extraction ---
# Replace the existing match_odds_for_apibaseball_game function with this one

def match_odds_for_apibaseball_game(apibaseball_game: Dict[str, Any], odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Matches Odds API event to api-baseball game event. Includes careful name checks."""
    if not odds_events or not apibaseball_game: return None

    game_id = apibaseball_game.get("id", "NO_ID_API_B")
    teams_info = apibaseball_game.get('teams', {})
    home_t = teams_info.get('home', {})
    away_t = teams_info.get('away', {})

    # Step 1: Extract Raw Names
    game_home_raw = home_t.get('name')
    game_away_raw = away_t.get('name')
    print(f"Debug Match Game {game_id}: Raw names = Away: '{game_away_raw}', Home: '{game_home_raw}'")

    # Step 2: Validate Raw Names
    if not game_home_raw or not game_away_raw:
        print(f"Warn: Missing raw team name from api-baseball for game {game_id}. Skipping match.")
        return None

    # Step 3: Normalize Names (using the original function)
    game_home_norm = normalize_team_name(game_home_raw)
    game_away_norm = normalize_team_name(game_away_raw) # Use correct variable
    print(f"Debug Match Game {game_id}: Normalized names = Away: '{game_away_norm}', Home: '{game_home_norm}'")

    # Step 4: Validate Normalized Names
    if not game_home_norm or not game_away_norm:
        # This *shouldn't* happen now if normalize_team_name works and raw names were valid
        print(f"Warn: Normalization resulted in empty string for game {game_id}. Skipping match.")
        return None

    # Step 5: Parse Game Date
    game_date_str = apibaseball_game.get('date')
    game_local_date = None
    try:
        game_dt_utc = dateutil_parser.isoparse(game_date_str)
        game_local_date = game_dt_utc.astimezone(ET_ZONE).date()
    except Exception as e:
        print(f"Warn: Could not parse api-baseball date {game_date_str} for game {game_id}: {e}")
        return None # Cannot match without a valid game date

    # Step 6: Loop through Odds Events
    for i, event in enumerate(odds_events):
        odds_home_raw = event.get('home_team','')
        odds_away_raw = event.get('away_team','')
        odds_home_norm = normalize_team_name(odds_home_raw)
        odds_away_norm = normalize_team_name(odds_away_raw)

        # Compare Names
        if game_home_norm == odds_home_norm and game_away_norm == odds_away_norm:
            # Compare Dates
            commence_str = event.get('commence_time','')
            try:
                event_dt = dateutil_parser.isoparse(commence_str)
                event_local_date = event_dt.astimezone(ET_ZONE).date()
                if game_local_date == event_local_date:
                    print(f"  ✅ Matched Odds: {title_case_team_name(game_away_norm)} @ {title_case_team_name(game_home_norm)} on {game_local_date}")
                    return event # Match found!
            except Exception:
                continue # Skip if odds date is invalid

    return None # No match found

def extract_odds_data(odds_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts raw odds (JSONB) and clean formatted odds strings from matched Odds API event."""
    # Default structures
    raw_odds = {"bookmaker_key": None, "moneyline": {}, "spread": {}, "total": {}}
    clean_odds = {"moneyline_home_clean": None, "moneyline_away_clean": None,"spread_home_line_clean": None, "spread_home_price_clean": None, "spread_away_price_clean": None,"total_line_clean": None, "total_over_price_clean": None, "total_under_price_clean": None}
    if not odds_event: return {"raw": raw_odds, "clean": clean_odds}

    bookmakers = odds_event.get('bookmakers', [])
    # Find preferred bookmaker or default to the first one
    target_bookmaker = next((bm for bm in bookmakers if bm.get('key') == PREFERRED_BOOKMAKER_KEY), bookmakers[0] if bookmakers else None)
    if not target_bookmaker: return {"raw": raw_odds, "clean": clean_odds}

    raw_odds["bookmaker_key"] = target_bookmaker.get("key")
    # Get title-cased team names from the odds event for comparison inside market loops
    odds_home=title_case_team_name(odds_event.get('home_team','')); odds_away=title_case_team_name(odds_event.get('away_team',''))
    markets = target_bookmaker.get('markets', [])

    for market in markets:
        mkey=market.get('key'); outcomes=market.get('outcomes',[]);
        if not isinstance(outcomes,list): continue # Ensure outcomes is a list

        if mkey == 'h2h': # Moneyline
            # Store raw prices
            raw_odds['moneyline'] = {title_case_team_name(o.get('name','')): o.get('price') for o in outcomes if o.get('name') and o.get('price') is not None}
            # Extract and format clean prices
            for o in outcomes:
                 team=title_case_team_name(o.get('name','')); price=safe_float_conversion(o.get('price')); fmt=format_american_odds(price)
                 if team == odds_home: clean_odds['moneyline_home_clean'] = fmt
                 elif team == odds_away: clean_odds['moneyline_away_clean'] = fmt

        elif mkey == 'spreads': # Spreads
             # Store raw prices/points
             raw_odds['spread'] = {title_case_team_name(o.get('name','')): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') and o.get('price') is not None and o.get('point') is not None}
             # Extract clean line/prices
             for o in outcomes:
                 team=title_case_team_name(o.get('name','')); price=safe_float_conversion(o.get('price')); point=safe_float_conversion(o.get('point')); fmt=format_american_odds(price)
                 if team == odds_home: clean_odds['spread_home_line_clean']=point; clean_odds['spread_home_price_clean']=fmt
                 elif team == odds_away: clean_odds['spread_away_price_clean']=fmt # Only need price for away spread

        elif mkey == 'totals': # Totals (Over/Under)
             # Store raw prices/points
             raw_odds['total'] = {o.get('name',''): {'price': o.get('price'), 'point': o.get('point')} for o in outcomes if o.get('name') in ['Over', 'Under'] and o.get('price') is not None and o.get('point') is not None}
             # Extract the numeric line from the first outcome (Over or Under)
             for o in outcomes: point=safe_float_conversion(o.get('point')); clean_odds['total_line_clean']=point; break
             # Extract and format prices for Over and Under
             for o in outcomes:
                 pos=o.get('name'); price=safe_float_conversion(o.get('price')); fmt=format_american_odds(price)
                 if pos=='Over': clean_odds['total_over_price_clean']=fmt
                 elif pos=='Under': clean_odds['total_under_price_clean']=fmt

    return {"raw": raw_odds, "clean": clean_odds}

# --- Main Build & Upsert ---
def build_and_upsert_mlb_previews():
    """Builds and upserts MLB game previews using api-baseball and The Odds API."""
    print("\n--- Running MLB Game Preview Script (api-baseball schedule + Odds API) ---")
    script_start_time = time.time()
    if not all([API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY]):
        print("FATAL ERROR: Config variables missing. Exiting."); return 0

    eastern_now=dt_datetime.now(ET_ZONE); today_et=eastern_now.date(); tomorrow_et=today_et+timedelta(days=1)

    print(f"\nStep 1: Fetching api-baseball schedule for {today_et} and {tomorrow_et}")
    apibaseball_games_today = get_games_from_apibaseball(today_et)
    time.sleep(REQUEST_DELAY_SECONDS) # Respect potential rate limits
    apibaseball_games_tomorrow = get_games_from_apibaseball(tomorrow_et)
    apibaseball_games = apibaseball_games_today + apibaseball_games_tomorrow

    if not apibaseball_games:
        print(f"No games found from api-baseball. Exiting."); return 0

    print(f"\nStep 2: Fetching betting odds covering {today_et} & {tomorrow_et}")
    odds_events=get_betting_odds(eastern_now)

    print("\nStep 3: Processing games and matching odds...")
    previews_to_upsert=[]; processed_game_ids=set()
    for game in apibaseball_games:
        try:
            game_id=game.get("id"); status=game.get('status',{}); status_s=status.get('short','').upper()
            # Skip if no game_id or already processed
            if not game_id or game_id in processed_game_ids: continue
            processed_game_ids.add(game_id);
            # Only process Not Started games
            if status_s != 'NS': continue

            # Extract core game info
            teams=game.get('teams',{}); home_t=teams.get('home',{}); away_t=teams.get('away',{})
            home_n=home_t.get('name'); away_n=away_t.get('name'); venue_info=game.get('venue',{})
            venue_n=venue_info.get('name') if isinstance(venue_info,dict) else None
            venue_c=venue_info.get('city') if isinstance(venue_info,dict) else None
            sch_utc=game.get('date'); date_et=None
            # Safely parse date
            if sch_utc: 
                try: dt_utc=dateutil_parser.isoparse(sch_utc); date_et=dt_utc.astimezone(ET_ZONE).date().isoformat() 
                except Exception as e: print(f"Warn: Could not parse date {sch_utc} for game {game_id}: {e}")

            # Match and extract odds
            odds_event=match_odds_for_apibaseball_game(game,odds_events); odds_data=extract_odds_data(odds_event)

            # Prepare data for Supabase - explicitly set pitcher fields to None
            preview_data = {
                'game_id':game_id, 'game_uid':None, # game_uid can be populated if needed from another source
                'scheduled_time_utc':sch_utc, 'game_date_et':date_et,
                'status_detail':status.get('long'), 'status_state':'pre' if status_s=='NS' else status_s,
                'home_team_id':home_t.get('id'), 'home_team_name':home_n, 'home_team_abbr':None, # abbr can be added if available
                'away_team_id':away_t.get('id'), 'away_team_name':away_n, 'away_team_abbr':None,
                'venue_name':venue_n, 'venue_city':venue_c, 'venue_state':None, 'venue_is_indoor':None, # More venue details can be added
                'home_probable_pitcher_id':None, 'home_probable_pitcher_name':None, 'home_probable_pitcher_record':None,
                'away_probable_pitcher_id':None, 'away_probable_pitcher_name':None, 'away_probable_pitcher_record':None,
                'moneyline':odds_data['raw'].get('moneyline'), # Store raw odds JSON
                'spread':odds_data['raw'].get('spread'),
                'total':odds_data['raw'].get('total'),
                'moneyline_home_clean':odds_data['clean'].get('moneyline_home_clean'), # Formatted string or None
                'moneyline_away_clean':odds_data['clean'].get('moneyline_away_clean'),
                'spread_home_line_clean':odds_data['clean'].get('spread_home_line_clean'), # Numeric line or None
                'spread_home_price_clean':odds_data['clean'].get('spread_home_price_clean'),# Formatted string or None
                'spread_away_price_clean':odds_data['clean'].get('spread_away_price_clean'),
                'total_line_clean':odds_data['clean'].get('total_line_clean'), # Numeric line or None
                'total_over_price_clean':odds_data['clean'].get('total_over_price_clean'), # Formatted string or None
                'total_under_price_clean':odds_data['clean'].get('total_under_price_clean'),
                'raw_api_response':json.dumps(game) # Store api-baseball game data as JSON string
             }
            # Basic validation before adding to upsert list
            if preview_data['game_id'] and preview_data['home_team_name'] and preview_data['away_team_name']:
                 previews_to_upsert.append(preview_data)
            else: print(f"Warn: Skip game {preview_data.get('game_id')} missing essential fields")
        except Exception as e: print(f"Error processing game {game.get('id', 'UNKNOWN')}: {e}")

    # --- Upsert to Supabase ---
    if previews_to_upsert:
        print(f"\nStep 4: Upserting {len(previews_to_upsert)} previews...")
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(previews_to_upsert, on_conflict='game_id').execute()
            print("Supabase upsert finished.")
            # Optional: Add basic check on response status if needed
            # if hasattr(upsert_response, 'data') and not upsert_response.data:
            #      if hasattr(upsert_response, 'error'): print(f"Supabase Upsert Warning/Error: {upsert_response.error}")
            #      else: print("Supabase Upsert may have had issues (no data returned).")

        except Exception as e: print(f"Error upserting to Supabase: {e}")
    else: print("\nStep 4: No valid previews generated to upsert.")

    script_end_time = time.time()
    print(f"\n--- MLB Preview Script finished (api-baseball + Odds API). Processed {len(processed_game_ids)} unique games. Pitcher data skipped. ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    return len(previews_to_upsert)

# --- Cleanup Function ---
def clear_old_mlb_games():
    """Removes games from the table whose scheduled date (ET) is before today."""
    print("\nAttempting to clear old games from mlb_game_schedule...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        response = supabase.table(SUPABASE_TABLE_NAME).select("game_id, scheduled_time_utc").execute()

        if not hasattr(response,'data') or response.data is None:
            print(f"Clear check: No games found or Supabase error: {response.get('error') or 'Unknown error'}")
            return

        rows = response.data; today_et = dt_datetime.now(ET_ZONE).date(); ids_to_delete=[]
        if not rows: print("Clear check: No games found in table."); return

        for row in rows:
            gid=row.get("game_id"); time_str=row.get("scheduled_time_utc")
            if not time_str or not gid: continue # Skip rows missing essential info
            try:
                # Convert UTC string to ET date for comparison
                if dateutil_parser.isoparse(time_str).astimezone(ET_ZONE).date() < today_et:
                    ids_to_delete.append(gid)
            except Exception as e: print(f"Error parsing date '{time_str}' for game {gid} during cleanup: {e}")

        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} old games (scheduled before {today_et})...")
            delete_response = supabase.table(SUPABASE_TABLE_NAME).delete().in_("game_id", ids_to_delete).execute()
            print("Batch delete attempted.")
            # Optional: Check delete response status
            # if hasattr(delete_response, 'data') and not delete_response.data:
            #      if hasattr(delete_response, 'error'): print(f"Supabase Delete Warning/Error: {delete_response.error}")
            #      else: print("Supabase Delete may have had issues (no data returned).")
        else: print("No old games found to delete.")
    except Exception as e: print(f"Error during Supabase cleanup: {e}")

# --- __main__ block ---
if __name__ == "__main__":
    run_cleanup = True # Set to False to disable cleanup
    if run_cleanup:
        clear_old_mlb_games()

    # Run the main function to build and upsert previews using reliable sources
    build_and_upsert_mlb_previews()

    print("\nDirect execution finished.")