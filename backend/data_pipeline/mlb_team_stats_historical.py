# backend/data_pipeline/mlb_team_stats_historical.py

import requests
import os
import json
import time
from dotenv import load_dotenv
# Import shared client AND Client class for type hinting
try:
    from caching.supabase_client import supabase
    from supabase import Client # For type hints
    print("Successfully imported shared Supabase client.")
except ImportError:
    print("FATAL ERROR: Could not import shared supabase client or Client class.")
    exit(1)
# Import only necessary components
from typing import Dict, List, Optional, Any

# --- Import API Key ---
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    if not API_SPORTS_KEY: raise ImportError("API_SPORTS_KEY not found")
    print("API_SPORTS_KEY loaded.")
except Exception as e:
     print(f"FATAL ERROR: Could not load API_SPORTS_KEY: {e}"); exit(1)


# --- Configuration ---
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1
TARGET_SEASONS = [2025, 2024, 2023, 2022, 2021] # Include current year
SUPABASE_TABLE_NAME = "mlb_historical_team_stats"
REQUEST_DELAY_SECONDS = 2
HEADERS = {'x-apisports-key': API_SPORTS_KEY} # Define headers once
# --- End Configuration ---

# --- Removed local load_environment() and init_supabase_client() functions ---

# --- API Fetching Functions ---
def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request and returns the JSON response."""
    # (Keep implementation as before)
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        remaining = response.headers.get('x-ratelimit-requests-remaining'); limit = response.headers.get('x-ratelimit-requests-limit')
        print(f"API Request successful for {params.get('team', params.get('date', '?'))}. Rate limit: {remaining}/{limit} remaining.") # Adjusted logging field
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request to {url} with params {params}: {e}")
        if e.response is not None: print(f"Status: {e.response.status_code}, Body: {e.response.text[:200]}...")
    except json.JSONDecodeError: print(f"Error decoding JSON from {url}")
    return None

def get_teams_for_season(league_id: int, season: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches all teams for a given league and season."""
    # (Keep implementation as before)
    print(f"Fetching teams for league {league_id}, season {season}...")
    url = f"{API_BASE_URL}/teams"; params = {"league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list): return data["response"]
    else: print(f"Could not fetch teams for league {league_id}, season {season}."); return []

def get_team_stats(team_id: int, league_id: int, season: int, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Fetches statistics for a specific team, league, and season."""
    # (Keep implementation as before)
    print(f"Fetching stats for team {team_id}, league {league_id}, season {season}...")
    url = f"{API_BASE_URL}/teams/statistics"; params = {"team": team_id, "league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    return data.get("response") if data else None

# --- Data Transformation ---
def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely converts a value to float, returning None on failure."""
    # (Keep implementation as before)
    if value is None: return None
    try:
        if isinstance(value, str): value = value.replace('%', '')
        return float(value)
    except (ValueError, TypeError): print(f"Warn: Could not convert '{value}' to float."); return None

def transform_stats_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms the raw API statistics data into the Supabase table format."""
    # (Keep implementation as before, ensure keys match mlb_historical_team_stats table)
    try:
        games = api_data.get("games", {}); points = api_data.get("points", {})
        team_info = api_data.get("team", {}); league_info = api_data.get("league", {})
        def get_nested(d: Dict, keys: List[str], default: Any = None) -> Any: # Simplified helper
            for k in keys: d = d.get(k) if isinstance(d, dict) else default;
            return d if d is not None else default
        transformed = {
            'team_id': team_info.get('id'), 'team_name': team_info.get('name'),
            'season': league_info.get('season'), 'league_id': league_info.get('id'),
            'league_name': league_info.get('name'),
            'games_played_home': get_nested(games, ['played', 'home']), 'games_played_away': get_nested(games, ['played', 'away']), 'games_played_all': get_nested(games, ['played', 'all']),
            'wins_home_total': get_nested(games, ['wins', 'home', 'total']), 'wins_home_percentage': safe_float_conversion(get_nested(games, ['wins', 'home', 'percentage'])),
            'wins_away_total': get_nested(games, ['wins', 'away', 'total']), 'wins_away_percentage': safe_float_conversion(get_nested(games, ['wins', 'away', 'percentage'])),
            'wins_all_total': get_nested(games, ['wins', 'all', 'total']), 'wins_all_percentage': safe_float_conversion(get_nested(games, ['wins', 'all', 'percentage'])),
            'losses_home_total': get_nested(games, ['loses', 'home', 'total']), 'losses_home_percentage': safe_float_conversion(get_nested(games, ['loses', 'home', 'percentage'])),
            'losses_away_total': get_nested(games, ['loses', 'away', 'total']), 'losses_away_percentage': safe_float_conversion(get_nested(games, ['loses', 'away', 'percentage'])),
            'losses_all_total': get_nested(games, ['loses', 'all', 'total']), 'losses_all_percentage': safe_float_conversion(get_nested(games, ['loses', 'all', 'percentage'])),
            'runs_for_total_home': get_nested(points, ['for', 'total', 'home']), 'runs_for_total_away': get_nested(points, ['for', 'total', 'away']), 'runs_for_total_all': get_nested(points, ['for', 'total', 'all']),
            'runs_for_avg_home': safe_float_conversion(get_nested(points, ['for', 'average', 'home'])), 'runs_for_avg_away': safe_float_conversion(get_nested(points, ['for', 'average', 'away'])), 'runs_for_avg_all': safe_float_conversion(get_nested(points, ['for', 'average', 'all'])),
            'runs_against_total_home': get_nested(points, ['against', 'total', 'home']), 'runs_against_total_away': get_nested(points, ['against', 'total', 'away']), 'runs_against_total_all': get_nested(points, ['against', 'total', 'all']),
            'runs_against_avg_home': safe_float_conversion(get_nested(points, ['against', 'average', 'home'])), 'runs_against_avg_away': safe_float_conversion(get_nested(points, ['against', 'average', 'away'])), 'runs_against_avg_all': safe_float_conversion(get_nested(points, ['against', 'average', 'all'])),
            'raw_api_response': json.dumps({"games": games, "points": points})
        }
        if not transformed['team_id'] or not transformed['season'] or not transformed['league_id']: return None
        return transformed
    except Exception as e: print(f"Error transforming team stats {api_data.get('team', {}).get('id')}: {e}"); return None

# --- Supabase Upsert ---
# NOTE: This function now uses the *imported shared* client instance
def upsert_team_stats(supabase_client: Client, stats_data: Dict[str, Any]):
    """Upserts a single team's transformed stats data into Supabase, updating on conflict."""
    if not stats_data: print("Warn: Received empty data for upsert."); return
    try:
        # Specify the columns involved in the UNIQUE constraint for conflict resolution
        # Assuming the constraint 'unique_team_season' is on (team_id, season, league_id)
        response = supabase_client.table(SUPABASE_TABLE_NAME)\
            .upsert(stats_data, on_conflict="team_id, season, league_id")\
            .execute() # Default behavior on conflict with this setup is UPDATE

        # Check response status/data if needed
        if hasattr(response, 'error') and response.error:
             print(f"Supabase Upsert Error for Team {stats_data.get('team_id')}, Season {stats_data.get('season')}: {response.error}")
        elif hasattr(response, 'data') and response.data:
             print(f"Successfully upserted/updated stats for Team ID: {stats_data.get('team_id')}, Season: {stats_data.get('season')}")
        else:
             print(f"Supabase Upsert for Team {stats_data.get('team_id')}, Season {stats_data.get('season')} - No data returned (maybe no change or check RLS/Policies if using anon key).")

    except Exception as e:
        print(f"Error upserting data for Team ID: {stats_data.get('team_id')}, Season: {stats_data.get('season')}: {e}")
        if hasattr(e, 'message'): print(f"Supabase error message: {e.message}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MLB Historical Team Stats Fetch Script...")
    script_start_time = time.time()

    # --- Use the imported shared client 'supabase' directly ---
    # No need to call load_environment() or init_supabase_client()
    if not API_SPORTS_KEY: print("API_SPORTS_KEY not loaded. Exiting."); exit()
    # Ensure the shared client was initialized correctly
    if supabase is None: print("Shared Supabase client failed to initialize (imported as None). Exiting."); exit()

    print("Using shared Supabase client and loaded API key.")
    # HEADERS defined in config section now uses the loaded API_SPORTS_KEY
    # --- End Change ---

    total_teams_processed = 0; total_seasons_processed = 0

    for season in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season} ---")
        season_start_time = time.time()
        teams = get_teams_for_season(MLB_LEAGUE_ID, season, HEADERS) # Use defined HEADERS
        time.sleep(REQUEST_DELAY_SECONDS) # Delay after fetching teams

        if not teams: print(f"No teams found for season {season}. Skipping."); continue

        total_seasons_processed += 1; print(f"Found {len(teams)} teams for season {season}.")

        for team in teams:
            team_id = team.get('id'); team_name = team.get('name', 'Unknown')
            if not team_id: print("Warn: Found team data without an ID. Skipping."); continue

            print(f"\nProcessing Team: {team_name} (ID: {team_id}) for Season: {season}")
            api_stats_response = get_team_stats(team_id, MLB_LEAGUE_ID, season, HEADERS)
            print(f"Waiting {REQUEST_DELAY_SECONDS} seconds..."); time.sleep(REQUEST_DELAY_SECONDS) # Delay AFTER request

            if api_stats_response:
                transformed_data = transform_stats_data(api_stats_response)
                if transformed_data:
                    # --- Pass the imported 'supabase' instance ---
                    upsert_team_stats(supabase, transformed_data) # Use imported shared client
                    total_teams_processed += 1
                else: print(f"Skipping upsert for Team ID: {team_id}, Season: {season} due to transformation error.")
            else: print(f"Skipping upsert for Team ID: {team_id}, Season: {season} due to fetch error.")

        season_end_time = time.time()
        print(f"--- Finished Season: {season} ---")
        print(f"Season processing time: {season_end_time - season_start_time:.2f} seconds.")

    script_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Processed {total_teams_processed} team-season records across {total_seasons_processed} seasons.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
