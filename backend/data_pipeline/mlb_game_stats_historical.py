# backend/data_pipeline/mlb_game_stats_historical.py

import requests
import os
import json
import time
from dotenv import load_dotenv # Keep dotenv for API key loading
import datetime
from datetime import date, timedelta, datetime as dt_datetime
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional, Any

# --- Import Shared Supabase Client ---
# This imports the client instance initialized with the SERVICE KEY
try:
    from caching.supabase_client import supabase
    # Import Client class ONLY for type hinting if needed
    from supabase import Client
    print("Successfully imported shared Supabase client.")
except ImportError:
    print("FATAL ERROR: Could not import shared supabase client from caching.supabase_client or Supabase Client")
    exit(1)

# --- Import API Key ---
# Load API_SPORTS_KEY directly here
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    if not API_SPORTS_KEY: raise ImportError("API_SPORTS_KEY not found")
    print("API_SPORTS_KEY loaded.")
except Exception as e:
     print(f"FATAL ERROR: Could not load API_SPORTS_KEY: {e}")
     exit(1)


# --- Configuration ---
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1
TARGET_SEASONS = [2025] # Adjust as needed
SUPABASE_TABLE_NAME = "mlb_historical_game_stats"
REQUEST_DELAY_SECONDS = 2
SEASON_DATE_RANGES = {
    2025: ("2025-04-21", None), # Start date for 2025 regular season, end date handled dynamically
    #2024: ("2024-03-20", "2024-11-05"),
    #2023: ("2023-03-30", "2023-11-05"),
    #2022: ("2022-04-07", "2022-11-06"),
    #2021: ("2021-04-01", "2021-11-03"),
}
BATCH_SIZE = 50
HEADERS = {'x-apisports-key': API_SPORTS_KEY} # Define headers using loaded key
# --- End Configuration ---

# --- Removed load_environment() and init_supabase_client() ---

# --- API Fetching ---\
def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request and returns the JSON response."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        remaining = response.headers.get('x-ratelimit-requests-remaining')
        limit = response.headers.get('x-ratelimit-requests-limit')
        print(f"API Request successful for {params.get('date', params.get('team', '?'))}. Rate limit: {remaining}/{limit} remaining.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request to {url} with params {params}: {e}")
        if e.response is not None: print(f"Response Status: {e.response.status_code}"); print(f"Response Body: {e.response.text[:500]}...")
    except json.JSONDecodeError: print(f"Error decoding JSON from {url}")
    return None

def get_games_for_date(league_id: int, season: int, date_str: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches all games for a given league, season, and date."""
    print(f"Fetching games for league {league_id}, season {season}, date {date_str}...")
    url = f"{API_BASE_URL}/games"; params = {"league": league_id, "season": season, "date": date_str}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list): return data["response"]
    else:
        if data and data.get("results", -1) == 0: print(f"No games found for {date_str}.")
        else: print(f"Could not fetch/parse games for {date_str}.")
        return []

# --- Data Transformation ---\
def transform_game_data(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms a single raw API game object into the Supabase table format."""
    try:
        game_id = game_data.get('id'); date_str = game_data.get('date')
        if not game_id: print("Warn: Game data missing 'id'. Skipping."); return None
        game_date_time_utc = None
        # Parse date string into datetime object then format back to ISO string
        if date_str: 
            try: game_date_time_utc = dateutil_parser.isoparse(date_str) 
            except ValueError: print(f"Warn: Parse date fail '{date_str}' G:{game_id}")
        league_info = game_data.get('league', {}); status_info = game_data.get('status', {})
        teams_info = game_data.get('teams', {}); home_team_info = teams_info.get('home', {}); away_team_info = teams_info.get('away', {})
        scores_info = game_data.get('scores', {}); home_score_info = scores_info.get('home', {}); away_score_info = scores_info.get('away', {})
        home_innings = home_score_info.get('innings', {}); away_innings = away_score_info.get('innings', {})

        # Function to safely convert score values to integer, returning None if conversion fails
        def safe_int(value):
            if value is None: return None
            try: return int(value)
            except (ValueError, TypeError): return None

        transformed = {
            'game_id': game_id,
            'game_date_time_utc': game_date_time_utc.isoformat() if game_date_time_utc else None,
            'season': safe_int(league_info.get('season')), # Ensure season is INT
            'league_id': safe_int(league_info.get('id')), # Ensure league_id is INT
            'status_long': status_info.get('long'),
            'status_short': status_info.get('short'),
            'home_team_id': safe_int(home_team_info.get('id')),
            'home_team_name': home_team_info.get('name'),
            'away_team_id': safe_int(away_team_info.get('id')),
            'away_team_name': away_team_info.get('name'),
            'home_score': safe_int(home_score_info.get('total')),
            'away_score': safe_int(away_score_info.get('total')),
            'home_hits': safe_int(home_score_info.get('hits')),
            'away_hits': safe_int(away_score_info.get('hits')),
            'home_errors': safe_int(home_score_info.get('errors')),
            'away_errors': safe_int(away_score_info.get('errors')),
            # Inning scores - safely convert to INT
            'h_inn_1': safe_int(home_innings.get('1')), 'h_inn_2': safe_int(home_innings.get('2')), 'h_inn_3': safe_int(home_innings.get('3')),
            'h_inn_4': safe_int(home_innings.get('4')), 'h_inn_5': safe_int(home_innings.get('5')), 'h_inn_6': safe_int(home_innings.get('6')),
            'h_inn_7': safe_int(home_innings.get('7')), 'h_inn_8': safe_int(home_innings.get('8')), 'h_inn_9': safe_int(home_innings.get('9')),
            'h_inn_extra': safe_int(home_innings.get('extra')),
            'a_inn_1': safe_int(away_innings.get('1')), 'a_inn_2': safe_int(away_innings.get('2')), 'a_inn_3': safe_int(away_innings.get('3')),
            'a_inn_4': safe_int(away_innings.get('4')), 'a_inn_5': safe_int(away_innings.get('5')), 'a_inn_6': safe_int(away_innings.get('6')),
            'a_inn_7': safe_int(away_innings.get('7')), 'a_inn_8': safe_int(away_innings.get('8')), 'a_inn_9': safe_int(away_innings.get('9')),
            'a_inn_extra': safe_int(away_innings.get('extra')),
            'raw_api_response': json.dumps(game_data) # Store original JSON
        }
        return transformed
    except Exception as e: print(f"Error transforming game {game_data.get('id', '?')}: {e}"); return None

# --- Supabase Upsert ---\
# Modified to accept Client type hint correctly
def upsert_game_stats_batch(supabase_client: Client, game_stats_list: List[Dict[str, Any]]):
    """Upserts a batch of transformed game stats data using the provided client."""
    if not game_stats_list: return
    print(f"Upserting batch of {len(game_stats_list)} game records...")
    try:
        # Use the passed-in supabase_client (which should be the shared one)
        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(game_stats_list, on_conflict='game_id').execute() # Added on_conflict
        print(f"Successfully upserted batch.")
        # Optional: Check response content if needed
        # if hasattr(response, 'error') and response.error: print(f"  Supabase Upsert Error: {response.error}")
        # elif not hasattr(response, 'data') or not response.data: print(f"  Supabase Upsert Warning: No data returned.")
    except Exception as e:
        print(f"Error upserting batch data: {e}")
        if hasattr(e, 'message'): print(f"Supabase error message: {e.message}")

# --- Date Range Generation ---\
def generate_date_range(start_date_str: str, end_date_str: str):
    """Yields datetime.date objects for a given date range string."""
    try:
        # Use dt_datetime alias
        start_date = dt_datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = dt_datetime.strptime(end_date_str, "%Y-%m-%d").date()
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += timedelta(days=1) # Use timedelta directly
    except (ValueError, TypeError) as e: # Catch specific errors
        print(f"Error: Invalid date format or value in range '{start_date_str}' to '{end_date_str}': {e}")
        yield None # Yield None to indicate failure? Or just return/raise? Let's yield None.

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MLB Historical Game Stats Fetch Script...")
    script_start_time = time.time()

    # Use the imported shared client 'supabase' directly
    if not API_SPORTS_KEY: print("API_SPORTS_KEY not loaded. Exiting."); exit()
    if supabase is None: print("Shared Supabase client failed to initialize. Exiting."); exit()
    headers = {'x-apisports-key': API_SPORTS_KEY}
    print("Using shared Supabase client and loaded API key.")

    total_games_processed = 0; batch_to_upsert = []
    current_year = dt_datetime.now().year # Get current year for comparison
    # Calculate yesterday's date ONCE
    yesterday_date = (dt_datetime.now().date() - datetime.timedelta(days=1))

    for season in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season} ---")
        season_start_time = time.time(); games_in_season = 0

        if season not in SEASON_DATE_RANGES:
            print(f"Warn: Date range not defined for season {season}. Skipping.")
            continue

        start_date_str, end_date_str_config = SEASON_DATE_RANGES[season] # Get configured range

        # --- Determine the correct end date ---
        if season == current_year:
            # For the current year, fetch up to yesterday
            end_date_dt = yesterday_date
            end_date_str = end_date_dt.strftime("%Y-%m-%d")
            print(f"Current season ({season}) detected. Fetching data from {start_date_str} up to {end_date_str} (yesterday).")
        elif end_date_str_config is None:
             print(f"Warn: End date not configured for past season {season}. Skipping.")
             continue
        else:
            # For past seasons, use the configured end date
            end_date_str = end_date_str_config
            print(f"Processing past season {season} from {start_date_str} to {end_date_str}.")
        # --- End date determination ---

        # Use generate_date_range with the determined start/end dates
        for current_date in generate_date_range(start_date_str, end_date_str):
            # Ensure generate_date_range handles date objects correctly
            if current_date is None: continue # Skip if date generation failed

            date_str = current_date.strftime("%Y-%m-%d");
            # print(f"Processing date: {date_str}") # Keep or remove detail logging
            games_on_date = get_games_for_date(MLB_LEAGUE_ID, season, date_str, headers)

            if games_on_date:
                # print(f"Found {len(games_on_date)} games.") # Keep or remove
                for game_api_data in games_on_date:
                    transformed_game = transform_game_data(game_api_data)
                    if transformed_game: batch_to_upsert.append(transformed_game); games_in_season += 1; total_games_processed += 1
            # else: # Optional log if no games found
                # print(f"No games returned for {date_str}")

            # Check if batch is full and upsert (moved outside inner if)
            if len(batch_to_upsert) >= BATCH_SIZE:
                upsert_game_stats_batch(supabase, batch_to_upsert)
                batch_to_upsert = []

            # Delay between date requests
            # print(f"Waiting {REQUEST_DELAY_SECONDS} seconds..."); # Keep or remove
            time.sleep(REQUEST_DELAY_SECONDS) # Moved delay here

        # Upsert any remaining items for the season
        if batch_to_upsert:
            upsert_game_stats_batch(supabase, batch_to_upsert)
            batch_to_upsert = []

        season_end_time = time.time()
        print(f"--- Finished Season: {season} ({games_in_season} games processed) ---")
        print(f"Season duration: {season_end_time - season_start_time:.2f} seconds.")

    script_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Total games processed: {total_games_processed} across requested seasons.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
