# backend/data_pipeline/mlb_team_stats_historical.py

import requests
import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Dict, List, Optional, Any

# --- Configuration ---
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1  # Standard ID for MLB
# *** Define the historical seasons you want to fetch data for ***
TARGET_SEASONS = [2025, 2024, 2023, 2022, 2021] # Example: Fetch last 3 completed seasons
SUPABASE_TABLE_NAME = "mlb_historical_team_stats"
# *** Adjust delay between API calls to respect rate limits (in seconds) ***
# Check your API plan's limits. 60 requests/minute = 1 req/sec. Add buffer.
REQUEST_DELAY_SECONDS = 2
# --- End Configuration ---

# --- Environment & Supabase Setup ---
def load_environment():
    """Loads environment variables and returns credentials."""
    # Adjust path to .env if needed, assuming script is in backend/data_pipeline
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("API_SPORTS_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not all([api_key, supabase_url, supabase_key]):
        print("Error: Missing required environment variables (API_SPORTS_KEY, SUPABASE_URL, SUPABASE_ANON_KEY)")
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

# --- API Fetching Functions ---
def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request and returns the JSON response."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Check for HTTP errors
        # Log rate limit info
        remaining = response.headers.get('x-ratelimit-requests-remaining')
        limit = response.headers.get('x-ratelimit-requests-limit')
        print(f"API Request successful for {params}. Rate limit: {remaining}/{limit} remaining.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request to {url} with params {params}: {e}")
        if e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            print(f"Response Body: {e.response.text[:500]}...") # Print beginning of error response
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from {url} with params {params}.")
    return None

def get_teams_for_season(league_id: int, season: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches all teams for a given league and season."""
    print(f"Fetching teams for league {league_id}, season {season}...")
    url = f"{API_BASE_URL}/teams"
    params = {"league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    if data and data.get("response"):
        return data["response"]
    else:
        print(f"Could not fetch teams for league {league_id}, season {season}.")
        return []

def get_team_stats(team_id: int, league_id: int, season: int, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Fetches statistics for a specific team, league, and season."""
    print(f"Fetching stats for team {team_id}, league {league_id}, season {season}...")
    url = f"{API_BASE_URL}/teams/statistics"
    params = {"team": team_id, "league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    # The actual stats are within the 'response' object for this endpoint
    return data.get("response") if data else None

# --- Data Transformation ---
def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely converts a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        # Handle potential percentage strings if API includes '%'
        if isinstance(value, str):
            value = value.replace('%', '')
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert value '{value}' to float.")
        return None

def transform_stats_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms the raw API statistics data into the Supabase table format."""
    try:
        games = api_data.get("games", {})
        points = api_data.get("points", {})
        team_info = api_data.get("team", {})
        league_info = api_data.get("league", {})

        # Helper to safely get nested values
        def get_nested(data: Dict, keys: List[str], default: Any = None) -> Any:
            current = data
            for key in keys:
                if not isinstance(current, dict): return default
                current = current.get(key)
                if current is None: return default
            return current

        transformed = {
            'team_id': team_info.get('id'),
            'team_name': team_info.get('name'),
            'season': league_info.get('season'),
            'league_id': league_info.get('id'),
            'league_name': league_info.get('name'),

            'games_played_home': get_nested(games, ['played', 'home']),
            'games_played_away': get_nested(games, ['played', 'away']),
            'games_played_all': get_nested(games, ['played', 'all']),

            'wins_home_total': get_nested(games, ['wins', 'home', 'total']),
            'wins_home_percentage': safe_float_conversion(get_nested(games, ['wins', 'home', 'percentage'])),
            'wins_away_total': get_nested(games, ['wins', 'away', 'total']),
            'wins_away_percentage': safe_float_conversion(get_nested(games, ['wins', 'away', 'percentage'])),
            'wins_all_total': get_nested(games, ['wins', 'all', 'total']),
            'wins_all_percentage': safe_float_conversion(get_nested(games, ['wins', 'all', 'percentage'])),

            # Note: API uses 'loses', mapping to 'losses'
            'losses_home_total': get_nested(games, ['loses', 'home', 'total']),
            'losses_home_percentage': safe_float_conversion(get_nested(games, ['loses', 'home', 'percentage'])),
            'losses_away_total': get_nested(games, ['loses', 'away', 'total']),
            'losses_away_percentage': safe_float_conversion(get_nested(games, ['loses', 'away', 'percentage'])),
            'losses_all_total': get_nested(games, ['loses', 'all', 'total']),
            'losses_all_percentage': safe_float_conversion(get_nested(games, ['loses', 'all', 'percentage'])),

            'runs_for_total_home': get_nested(points, ['for', 'total', 'home']),
            'runs_for_total_away': get_nested(points, ['for', 'total', 'away']),
            'runs_for_total_all': get_nested(points, ['for', 'total', 'all']),
            'runs_for_avg_home': safe_float_conversion(get_nested(points, ['for', 'average', 'home'])),
            'runs_for_avg_away': safe_float_conversion(get_nested(points, ['for', 'average', 'away'])),
            'runs_for_avg_all': safe_float_conversion(get_nested(points, ['for', 'average', 'all'])),

            'runs_against_total_home': get_nested(points, ['against', 'total', 'home']),
            'runs_against_total_away': get_nested(points, ['against', 'total', 'away']),
            'runs_against_total_all': get_nested(points, ['against', 'total', 'all']),
            'runs_against_avg_home': safe_float_conversion(get_nested(points, ['against', 'average', 'home'])),
            'runs_against_avg_away': safe_float_conversion(get_nested(points, ['against', 'average', 'away'])),
            'runs_against_avg_all': safe_float_conversion(get_nested(points, ['against', 'average', 'all'])),

            # Store the relevant parts of the raw response for potential future use/debugging
            'raw_api_response': json.dumps({"games": games, "points": points})
            # 'updated_at' is handled by Supabase default 'now()'
        }

        # Basic validation: Ensure essential IDs are present
        if not transformed['team_id'] or not transformed['season'] or not transformed['league_id']:
            print(f"Warning: Missing essential IDs in data for team {transformed.get('team_name')}, season {transformed.get('season')}. Skipping.")
            return None

        return transformed

    except Exception as e:
        print(f"Error transforming data for team {api_data.get('team', {}).get('id')}: {e}")
        print(f"Problematic API Data: {json.dumps(api_data, indent=2)}") # Log the data that caused error
        return None

# --- Supabase Upsert ---
def upsert_team_stats(supabase_client: Client, stats_data: Dict[str, Any]):
    """Upserts a single team's transformed stats data into Supabase."""
    try:
        # Upsert based on the unique constraint (team_id, season, league_id)
        # Supabase automatically handles mapping dict keys to column names
        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(stats_data).execute()
        # Check response for errors if needed, though Supabase client might raise exceptions
        print(f"Successfully upserted stats for Team ID: {stats_data['team_id']}, Season: {stats_data['season']}")
        # print(f"Supabase Response Data: {response.data}") # Optional detailed logging

    except Exception as e:
        print(f"Error upserting data for Team ID: {stats_data.get('team_id')}, Season: {stats_data.get('season')}: {e}")
        # Attempt to log specific Supabase errors if available
        if hasattr(e, 'message'):
             print(f"Supabase error message: {e.message}")
        # print(f"Data attempted to upsert: {stats_data}") # Optional: Log the data that failed

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MLB Historical Team Stats Fetch Script...")
    start_time = time.time()

    api_key, supabase_url, supabase_key = load_environment()
    supabase = init_supabase_client(supabase_url, supabase_key)

    headers = {'x-apisports-key': api_key}
    total_teams_processed = 0
    total_seasons_processed = 0

    for season in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season} ---")
        teams = get_teams_for_season(MLB_LEAGUE_ID, season, headers)
        time.sleep(REQUEST_DELAY_SECONDS) # Delay after fetching teams list

        if not teams:
            print(f"No teams found for season {season}. Skipping.")
            continue

        total_seasons_processed += 1
        print(f"Found {len(teams)} teams for season {season}.")

        for team in teams:
            team_id = team.get('id')
            team_name = team.get('name', 'Unknown') # Get name for logging
            if not team_id:
                print("Warning: Found team data without an ID. Skipping.")
                continue

            print(f"\nProcessing Team: {team_name} (ID: {team_id}) for Season: {season}")

            # Fetch stats for this specific team
            api_stats_response = get_team_stats(team_id, MLB_LEAGUE_ID, season, headers)

            # Apply delay AFTER the request
            print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before next API call...")
            time.sleep(REQUEST_DELAY_SECONDS)

            if api_stats_response:
                # Transform the data
                transformed_data = transform_stats_data(api_stats_response)

                if transformed_data:
                    # Upsert the data
                    upsert_team_stats(supabase, transformed_data)
                    total_teams_processed += 1
                else:
                    print(f"Skipping upsert for Team ID: {team_id}, Season: {season} due to transformation error.")
            else:
                 print(f"Skipping upsert for Team ID: {team_id}, Season: {season} due to fetch error.")

    end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Processed {total_teams_processed} team-season records across {total_seasons_processed} seasons.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")