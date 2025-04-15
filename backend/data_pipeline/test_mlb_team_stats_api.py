# test_rapidapi_mlb_schedule_v2.py

import requests
import os
import json
from dotenv import load_dotenv
import datetime
from typing import Dict, Any, Optional

# --- Configuration ---
TARGET_DATE = datetime.date(2025, 4, 15) # Tomorrow (context Apr 14, 2025)
TARGET_YEAR = TARGET_DATE.year
TARGET_MONTH = f"{TARGET_DATE.month:02d}"
TARGET_DAY = f"{TARGET_DATE.day:02d}"

# --- End Configuration ---

# --- Environment Setup ---
def load_environment():
    """Loads environment variables and returns RapidAPI credentials."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Adjust if needed
    load_dotenv(dotenv_path=dotenv_path)
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = os.getenv("RAPIDAPI_HOST")
    if not all([rapidapi_key, rapidapi_host]):
        print("Error: Missing required environment variables (RAPIDAPI_KEY, RAPIDAPI_HOST)")
        exit()
    return rapidapi_key, rapidapi_host

# --- API Request Function ---
def make_rapidapi_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes an API request to a RapidAPI endpoint."""
    print(f"Attempting to fetch data from: {url}")
    print(f"Using parameters: {params}")
    print(f"Using Host: {headers.get('x-rapidapi-host')}")
    print(f"Using Key starting: {headers.get('x-rapidapi-key', '')[:5]}...")

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        remaining = response.headers.get('X-RateLimit-Requests-Remaining')
        limit = response.headers.get('X-RateLimit-Requests-Limit')
        print(f"RapidAPI Request successful. Usage: Remaining={remaining}, Limit={limit}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching from RapidAPI: {http_err}")
        print(f"Response Status Code: {http_err.response.status_code}")
        print(f"Response Body: {http_err.response.text[:1000]}...")
        if http_err.response.status_code in [401, 403]:
             print("NOTE: Received 401/403 Unauthorized/Forbidden. Check RAPIDAPI_KEY, RAPIDAPI_HOST, and your subscription.")
        elif http_err.response.status_code == 404:
             print("NOTE: Received 404 Not Found. The endpoint path might be incorrect.")
    except requests.exceptions.RequestException as e:
        print(f"Network or Request Error fetching from RapidAPI: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from RapidAPI URL {url}. Raw response text:")
        print(response.text[:1000] + "...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting RapidAPI MLB Schedule Test Script (v2) for Date: {TARGET_YEAR}-{TARGET_MONTH}-{TARGET_DAY}...")
    rapidapi_key, rapidapi_host = load_environment()

    headers = {
        "x-rapidapi-key": rapidapi_key,
	    "x-rapidapi-host": rapidapi_host
    }

    # *** MODIFIED ENDPOINT PATH ***
    endpoint_path = "/mlb/schedule"
    # *****************************

    url = f"https://{rapidapi_host}{endpoint_path}"
    params = {
        "year": TARGET_YEAR,
        "month": TARGET_MONTH,
        "day": TARGET_DAY
    }

    # Make the API call
    data = make_rapidapi_request(url, headers, params)

    if data:
        print("\n--- API Response ---")
        print(json.dumps(data, indent=2))
        print("--- End API Response ---\n")

        # Basic analysis attempt
        possible_data_keys = ['body', 'data', 'events', 'games', 'schedule']
        game_list = None
        # Check if the root object itself is the list first
        if isinstance(data, list) and data:
             game_list = data
             print(f"Analysis: The entire response is a list with {len(data)} item(s).")
        else:
            # Check common top-level keys if root isn't the list
            for key in possible_data_keys:
                content = data.get(key)
                if isinstance(content, list):
                    game_list = content
                    print(f"Analysis: Found a list under the key '{key}' with {len(game_list)} item(s). Assuming these are games.")
                    break

        if game_list is not None and len(game_list) > 0:
            print("Structure of the first game/event object appears to contain keys like:")
            print(list(game_list[0].keys()))
            pitcher_keys_found = [k for k in game_list[0].keys() if 'pitcher' in k.lower()]
            if pitcher_keys_found:
                 print(f"Potential pitcher-related keys found: {pitcher_keys_found}")
            else:
                 print("No obvious 'pitcher' related keys found at the top level of the first game object.")
                 # Check nested structures common for pitchers
                 teams_data = game_list[0].get('teams', {}) # Example common structure
                 home_team_data = teams_data.get('home', {})
                 away_team_data = teams_data.get('away', {})
                 nested_pitcher_keys = []
                 if 'pitcher' in str(home_team_data).lower(): nested_pitcher_keys.append("home team object")
                 if 'pitcher' in str(away_team_data).lower(): nested_pitcher_keys.append("away team object")
                 # Add more checks for other potential nested locations if needed
                 if nested_pitcher_keys:
                      print(f"Potential pitcher-related info might be nested within: {nested_pitcher_keys}")

        elif game_list is not None:
             print("Analysis: The primary data list is empty.")
        else:
            print("Analysis: Could not identify a list of games/events in the response structure.")

    else:
        print("Failed to retrieve data from the RapidAPI endpoint (check error messages above).")

    print("Script finished.")