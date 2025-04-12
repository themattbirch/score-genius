import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file in the same directory as this script.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Get the RapidAPI credentials from environment variables.
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST")

# Base URL for the NBA endpoints (as provided by the API documentation).
BASE_URL = "https://sports-information.p.rapidapi.com"

# Common headers for all requests.
# Using lowercase header keys here for consistency with the working MLB script; these are case-insensitive.
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST
}

def call_endpoint(endpoint, params=None):
    """
    Generic function to perform GET request to a specific endpoint.
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise error for bad responses
        return response.json()
    except requests.RequestException as error:
        print(f"Request to {url} failed: {error}")
        return None

def fetch_team_list_data():
    """
    Call the Team List endpoint: /team-list
    """
    endpoint = "/team-list"
    print("Fetching Team List Data...")
    data = call_endpoint(endpoint)
    print(data)

def fetch_team_info_data(team_id=52):
    """
    Call the Team Info endpoint: /team-info/:id
    """
    endpoint = f"/team-info/{team_id}"
    print(f"Fetching Team Info Data for team ID {team_id} ...")
    data = call_endpoint(endpoint)
    print(data)

def fetch_team_players_data(team_id=52):
    """
    Call the Team Players endpoint: /team-players/:id
    """
    endpoint = f"/team-players/{team_id}"
    print(f"Fetching Team Players Data for team ID {team_id} ...")
    data = call_endpoint(endpoint)
    print(data)

def fetch_injuries_data():
    """
    Call the Injuries endpoint: /injuries
    """
    endpoint = "/injuries"
    print("Fetching Injuries Data ...")
    data = call_endpoint(endpoint)
    print(data)

def fetch_team_roster_data(team_id=16, season=2023):
    """
    Call the Team Roster endpoint: /team-roster?teamId=16&season=2024
    """
    endpoint = "/team-roster"
    params = {
        "teamId": team_id,
        "season": season
    }
    print(f"Fetching Team Roster Data for team ID {team_id} for season {season} ...")
    data = call_endpoint(endpoint, params=params)
    print(data)

if __name__ == "__main__":
    print("Starting API tests with RapidAPI endpoints...\n")
    
    # 1. Team List Data
    fetch_team_list_data()
    print("\n" + "-"*50 + "\n")
    
    # 2. Team Info Data
    fetch_team_info_data(team_id=52)
    print("\n" + "-"*50 + "\n")
    
    # 3. Team Players Data
    fetch_team_players_data(team_id=52)
    print("\n" + "-"*50 + "\n")
    
    # 4. Injuries Data
    fetch_injuries_data()
    print("\n" + "-"*50 + "\n")
    
    # 5. Team Roster Data
    fetch_team_roster_data(team_id=16, season=2023)
    print("\nAPI tests completed.")
