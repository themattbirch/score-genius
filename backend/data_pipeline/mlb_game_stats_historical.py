# backend/data_pipeline/mlb_game_stats_historical.py

import requests
import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime
from dateutil import parser as dateutil_parser # Use alias to avoid confusion
from typing import Dict, List, Optional, Any

# --- Configuration ---\
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1  # Standard ID for MLB
# *** Define the historical seasons you want to fetch data for ***
TARGET_SEASONS = [2024, 2023, 2022, 2021] # Added 2021 as discussed
SUPABASE_TABLE_NAME = "mlb_historical_game_stats"
# *** Adjust delay between API calls (seconds) ***
REQUEST_DELAY_SECONDS = 2
# *** Define approximate date ranges for MLB regular seasons ***
# Adjust these if you need pre/post-season games or know exact dates
SEASON_DATE_RANGES = {
    2024: ("2024-03-20", "2024-11-05"), # Approx start/end including postseason
    2023: ("2023-03-30", "2023-11-05"),
    2022: ("2022-04-07", "2022-11-06"),
    2021: ("2021-04-01", "2021-11-03"),
    # Add more seasons as needed
}
# *** Number of game records to batch before upserting ***
BATCH_SIZE = 50
# --- End Configuration ---\

# --- Environment & Supabase Setup ---\
def load_environment():
    """Loads environment variables and returns credentials."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Adjust if needed
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("API_SPORTS_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not all([api_key, supabase_url, supabase_key]):
        print("Error: Missing required environment variables.")
        exit()
    return api_key, supabase_url, supabase_key

def init_supabase_client(url: str, key: str) -> Client:
    """Initializes and returns the Supabase client."""
    try:
        supabase: Client = create_client(url, key)
        print("Supabase client initialized successfully.")
        return supabase
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        exit()

# --- API Fetching ---\
def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request and returns the JSON response."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        remaining = response.headers.get('x-ratelimit-requests-remaining')
        limit = response.headers.get('x-ratelimit-requests-limit')
        print(f"API Request successful for {params}. Rate limit: {remaining}/{limit} remaining.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request to {url} with params {params}: {e}")
        if e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            print(f"Response Body: {e.response.text[:500]}...")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from {url} with params {params}.")
    return None

def get_games_for_date(league_id: int, season: int, date_str: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches all games for a given league, season, and date."""
    print(f"Fetching games for league {league_id}, season {season}, date {date_str}...")
    url = f"{API_BASE_URL}/games"
    params = {"league": league_id, "season": season, "date": date_str}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list):
        return data["response"]
    else:
        # It's normal for some dates to have no games
        if data and data.get("results", 0) == 0:
             print(f"No games found for {date_str}.")
        else:
            print(f"Could not fetch or parse games for {date_str}.")
        return []

# --- Data Transformation ---\
def transform_game_data(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms a single raw API game object into the Supabase table format."""
    try:
        game_id = game_data.get('id')
        if not game_id:
            print("Warning: Game data missing 'id'. Skipping.")
            return None

        # Parse date safely
        game_date_time_utc = None
        date_str = game_data.get('date')
        if date_str:
            try:
                game_date_time_utc = dateutil_parser.isoparse(date_str)
            except ValueError:
                print(f"Warning: Could not parse date string '{date_str}' for game {game_id}.")

        # Safely access nested data using .get()
        league_info = game_data.get('league', {})
        status_info = game_data.get('status', {})
        teams_info = game_data.get('teams', {})
        home_team_info = teams_info.get('home', {})
        away_team_info = teams_info.get('away', {})
        scores_info = game_data.get('scores', {})
        home_score_info = scores_info.get('home', {})
        away_score_info = scores_info.get('away', {})
        home_innings = home_score_info.get('innings', {})
        away_innings = away_score_info.get('innings', {})

        transformed = {
            'game_id': game_id,
            'game_date_time_utc': game_date_time_utc.isoformat() if game_date_time_utc else None,
            'season': league_info.get('season'),
            'league_id': league_info.get('id'),
            'status_long': status_info.get('long'),
            'status_short': status_info.get('short'),
            'home_team_id': home_team_info.get('id'),
            'home_team_name': home_team_info.get('name'),
            'away_team_id': away_team_info.get('id'),
            'away_team_name': away_team_info.get('name'),
            'home_score': home_score_info.get('total'),
            'away_score': away_score_info.get('total'),
            'home_hits': home_score_info.get('hits'),
            'away_hits': away_score_info.get('hits'),
            'home_errors': home_score_info.get('errors'),
            'away_errors': away_score_info.get('errors'),

            # Inning scores - use .get() to handle potentially missing innings (though API seems consistent 1-9)
            'h_inn_1': home_innings.get('1'),
            'h_inn_2': home_innings.get('2'),
            'h_inn_3': home_innings.get('3'),
            'h_inn_4': home_innings.get('4'),
            'h_inn_5': home_innings.get('5'),
            'h_inn_6': home_innings.get('6'),
            'h_inn_7': home_innings.get('7'),
            'h_inn_8': home_innings.get('8'),
            'h_inn_9': home_innings.get('9'),
            'h_inn_extra': home_innings.get('extra'),

            'a_inn_1': away_innings.get('1'),
            'a_inn_2': away_innings.get('2'),
            'a_inn_3': away_innings.get('3'),
            'a_inn_4': away_innings.get('4'),
            'a_inn_5': away_innings.get('5'),
            'a_inn_6': away_innings.get('6'),
            'a_inn_7': away_innings.get('7'),
            'a_inn_8': away_innings.get('8'),
            'a_inn_9': away_innings.get('9'),
            'a_inn_extra': away_innings.get('extra'),

            # 'updated_at' is handled by Supabase default
            'raw_api_response': json.dumps(game_data) # Store the whole game object
        }
        return transformed

    except Exception as e:
        print(f"Error transforming data for game {game_data.get('id', 'UNKNOWN')}: {e}")
        # print(f"Problematic Game Data: {json.dumps(game_data, indent=2)}")
        return None

# --- Supabase Upsert ---\
def upsert_game_stats_batch(supabase_client: Client, game_stats_list: List[Dict[str, Any]]):
    """Upserts a batch of transformed game stats data into Supabase."""
    if not game_stats_list:
        return

    print(f"Upserting batch of {len(game_stats_list)} game records...")
    try:
        # Upsert based on the primary key 'game_id'
        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(game_stats_list).execute()
        print(f"Successfully upserted batch.")
        # print(f"Supabase Response: {response}") # Optional detailed logging
    except Exception as e:
        print(f"Error upserting batch data: {e}")
        if hasattr(e, 'message'):
             print(f"Supabase error message: {e.message}")
        # Optionally log the first few failing records
        # print(f"First few records in failed batch: {json.dumps(game_stats_list[:2], indent=2)}")

# --- Date Range Generation ---\
def generate_date_range(start_date_str: str, end_date_str: str):
    """Yields datetime.date objects for a given date range string."""
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += datetime.timedelta(days=1)
    except ValueError:
        print(f"Error: Invalid date format in range '{start_date_str}' to '{end_date_str}'. Use YYYY-MM-DD.")
        return

# --- Main Execution ---\
if __name__ == "__main__":
    print("Starting MLB Historical Game Stats Fetch Script...")
    script_start_time = time.time()

    api_key, supabase_url, supabase_key = load_environment()
    supabase = init_supabase_client(supabase_url, supabase_key)
    headers = {'x-apisports-key': api_key}

    total_games_processed = 0
    batch_to_upsert: List[Dict[str, Any]] = []

    for season in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season} ---")
        season_start_time = time.time()
        games_in_season = 0

        if season not in SEASON_DATE_RANGES:
            print(f"Warning: Date range not defined for season {season}. Skipping.")
            continue

        start_date_str, end_date_str = SEASON_DATE_RANGES[season]

        for current_date in generate_date_range(start_date_str, end_date_str):
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing date: {date_str}")

            games_on_date = get_games_for_date(MLB_LEAGUE_ID, season, date_str, headers)

            # Apply delay AFTER the request for the day
            print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before next API call...")
            time.sleep(REQUEST_DELAY_SECONDS)

            if games_on_date:
                print(f"Found {len(games_on_date)} games on {date_str}.")
                for game_api_data in games_on_date:
                    transformed_game = transform_game_data(game_api_data)
                    if transformed_game:
                        batch_to_upsert.append(transformed_game)
                        games_in_season += 1
                        total_games_processed += 1

            # Check if batch is full and upsert
            if len(batch_to_upsert) >= BATCH_SIZE:
                upsert_game_stats_batch(supabase, batch_to_upsert)
                batch_to_upsert = [] # Clear the batch

        # Upsert any remaining records for the season after the loop
        if batch_to_upsert:
            upsert_game_stats_batch(supabase, batch_to_upsert)
            batch_to_upsert = []

        season_end_time = time.time()
        print(f"--- Finished processing Season: {season} ---")
        print(f"Processed {games_in_season} games for season {season}.")
        print(f"Season processing time: {season_end_time - season_start_time:.2f} seconds.")


    script_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Processed a total of {total_games_processed} game records across {len(TARGET_SEASONS)} seasons.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")# backend/data_pipeline/mlb_game_stats_historical.py

import requests
import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime
from dateutil import parser as dateutil_parser # Use alias to avoid confusion
from typing import Dict, List, Optional, Any

# --- Configuration ---\
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1  # Standard ID for MLB
# *** Define the historical seasons you want to fetch data for ***
TARGET_SEASONS = [2024, 2023, 2022, 2021] # Added 2021 as discussed
SUPABASE_TABLE_NAME = "mlb_historical_game_stats"
# *** Adjust delay between API calls (seconds) ***
REQUEST_DELAY_SECONDS = 2
# *** Define approximate date ranges for MLB regular seasons ***
# Adjust these if you need pre/post-season games or know exact dates
SEASON_DATE_RANGES = {
    2024: ("2024-03-20", "2024-11-05"), # Approx start/end including postseason
    2023: ("2023-03-30", "2023-11-05"),
    2022: ("2022-04-07", "2022-11-06"),
    2021: ("2021-04-01", "2021-11-03"),
    # Add more seasons as needed
}
# *** Number of game records to batch before upserting ***
BATCH_SIZE = 50
# --- End Configuration ---\

# --- Environment & Supabase Setup ---\
def load_environment():
    """Loads environment variables and returns credentials."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Adjust if needed
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("API_SPORTS_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not all([api_key, supabase_url, supabase_key]):
        print("Error: Missing required environment variables.")
        exit()
    return api_key, supabase_url, supabase_key

def init_supabase_client(url: str, key: str) -> Client:
    """Initializes and returns the Supabase client."""
    try:
        supabase: Client = create_client(url, key)
        print("Supabase client initialized successfully.")
        return supabase
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        exit()

# --- API Fetching ---\
def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request and returns the JSON response."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        remaining = response.headers.get('x-ratelimit-requests-remaining')
        limit = response.headers.get('x-ratelimit-requests-limit')
        print(f"API Request successful for {params}. Rate limit: {remaining}/{limit} remaining.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request to {url} with params {params}: {e}")
        if e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            print(f"Response Body: {e.response.text[:500]}...")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from {url} with params {params}.")
    return None

def get_games_for_date(league_id: int, season: int, date_str: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches all games for a given league, season, and date."""
    print(f"Fetching games for league {league_id}, season {season}, date {date_str}...")
    url = f"{API_BASE_URL}/games"
    params = {"league": league_id, "season": season, "date": date_str}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list):
        return data["response"]
    else:
        # It's normal for some dates to have no games
        if data and data.get("results", 0) == 0:
             print(f"No games found for {date_str}.")
        else:
            print(f"Could not fetch or parse games for {date_str}.")
        return []

# --- Data Transformation ---\
def transform_game_data(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms a single raw API game object into the Supabase table format."""
    try:
        game_id = game_data.get('id')
        if not game_id:
            print("Warning: Game data missing 'id'. Skipping.")
            return None

        # Parse date safely
        game_date_time_utc = None
        date_str = game_data.get('date')
        if date_str:
            try:
                game_date_time_utc = dateutil_parser.isoparse(date_str)
            except ValueError:
                print(f"Warning: Could not parse date string '{date_str}' for game {game_id}.")

        # Safely access nested data using .get()
        league_info = game_data.get('league', {})
        status_info = game_data.get('status', {})
        teams_info = game_data.get('teams', {})
        home_team_info = teams_info.get('home', {})
        away_team_info = teams_info.get('away', {})
        scores_info = game_data.get('scores', {})
        home_score_info = scores_info.get('home', {})
        away_score_info = scores_info.get('away', {})
        home_innings = home_score_info.get('innings', {})
        away_innings = away_score_info.get('innings', {})

        transformed = {
            'game_id': game_id,
            'game_date_time_utc': game_date_time_utc.isoformat() if game_date_time_utc else None,
            'season': league_info.get('season'),
            'league_id': league_info.get('id'),
            'status_long': status_info.get('long'),
            'status_short': status_info.get('short'),
            'home_team_id': home_team_info.get('id'),
            'home_team_name': home_team_info.get('name'),
            'away_team_id': away_team_info.get('id'),
            'away_team_name': away_team_info.get('name'),
            'home_score': home_score_info.get('total'),
            'away_score': away_score_info.get('total'),
            'home_hits': home_score_info.get('hits'),
            'away_hits': away_score_info.get('hits'),
            'home_errors': home_score_info.get('errors'),
            'away_errors': away_score_info.get('errors'),

            # Inning scores - use .get() to handle potentially missing innings (though API seems consistent 1-9)
            'h_inn_1': home_innings.get('1'),
            'h_inn_2': home_innings.get('2'),
            'h_inn_3': home_innings.get('3'),
            'h_inn_4': home_innings.get('4'),
            'h_inn_5': home_innings.get('5'),
            'h_inn_6': home_innings.get('6'),
            'h_inn_7': home_innings.get('7'),
            'h_inn_8': home_innings.get('8'),
            'h_inn_9': home_innings.get('9'),
            'h_inn_extra': home_innings.get('extra'),

            'a_inn_1': away_innings.get('1'),
            'a_inn_2': away_innings.get('2'),
            'a_inn_3': away_innings.get('3'),
            'a_inn_4': away_innings.get('4'),
            'a_inn_5': away_innings.get('5'),
            'a_inn_6': away_innings.get('6'),
            'a_inn_7': away_innings.get('7'),
            'a_inn_8': away_innings.get('8'),
            'a_inn_9': away_innings.get('9'),
            'a_inn_extra': away_innings.get('extra'),

            # 'updated_at' is handled by Supabase default
            'raw_api_response': json.dumps(game_data) # Store the whole game object
        }
        return transformed

    except Exception as e:
        print(f"Error transforming data for game {game_data.get('id', 'UNKNOWN')}: {e}")
        # print(f"Problematic Game Data: {json.dumps(game_data, indent=2)}")
        return None

# --- Supabase Upsert ---\
def upsert_game_stats_batch(supabase_client: Client, game_stats_list: List[Dict[str, Any]]):
    """Upserts a batch of transformed game stats data into Supabase."""
    if not game_stats_list:
        return

    print(f"Upserting batch of {len(game_stats_list)} game records...")
    try:
        # Upsert based on the primary key 'game_id'
        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(game_stats_list).execute()
        print(f"Successfully upserted batch.")
        # print(f"Supabase Response: {response}") # Optional detailed logging
    except Exception as e:
        print(f"Error upserting batch data: {e}")
        if hasattr(e, 'message'):
             print(f"Supabase error message: {e.message}")
        # Optionally log the first few failing records
        # print(f"First few records in failed batch: {json.dumps(game_stats_list[:2], indent=2)}")

# --- Date Range Generation ---\
def generate_date_range(start_date_str: str, end_date_str: str):
    """Yields datetime.date objects for a given date range string."""
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += datetime.timedelta(days=1)
    except ValueError:
        print(f"Error: Invalid date format in range '{start_date_str}' to '{end_date_str}'. Use YYYY-MM-DD.")
        return

# --- Main Execution ---\
if __name__ == "__main__":
    print("Starting MLB Historical Game Stats Fetch Script...")
    script_start_time = time.time()

    api_key, supabase_url, supabase_key = load_environment()
    supabase = init_supabase_client(supabase_url, supabase_key)
    headers = {'x-apisports-key': api_key}

    total_games_processed = 0
    batch_to_upsert: List[Dict[str, Any]] = []

    for season in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season} ---")
        season_start_time = time.time()
        games_in_season = 0

        if season not in SEASON_DATE_RANGES:
            print(f"Warning: Date range not defined for season {season}. Skipping.")
            continue

        start_date_str, end_date_str = SEASON_DATE_RANGES[season]

        for current_date in generate_date_range(start_date_str, end_date_str):
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing date: {date_str}")

            games_on_date = get_games_for_date(MLB_LEAGUE_ID, season, date_str, headers)

            # Apply delay AFTER the request for the day
            print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before next API call...")
            time.sleep(REQUEST_DELAY_SECONDS)

            if games_on_date:
                print(f"Found {len(games_on_date)} games on {date_str}.")
                for game_api_data in games_on_date:
                    transformed_game = transform_game_data(game_api_data)
                    if transformed_game:
                        batch_to_upsert.append(transformed_game)
                        games_in_season += 1
                        total_games_processed += 1

            # Check if batch is full and upsert
            if len(batch_to_upsert) >= BATCH_SIZE:
                upsert_game_stats_batch(supabase, batch_to_upsert)
                batch_to_upsert = [] # Clear the batch

        # Upsert any remaining records for the season after the loop
        if batch_to_upsert:
            upsert_game_stats_batch(supabase, batch_to_upsert)
            batch_to_upsert = []

        season_end_time = time.time()
        print(f"--- Finished processing Season: {season} ---")
        print(f"Processed {games_in_season} games for season {season}.")
        print(f"Season processing time: {season_end_time - season_start_time:.2f} seconds.")


    script_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Processed a total of {total_games_processed} game records across {len(TARGET_SEASONS)} seasons.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")