import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file located in the same directory.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Get credentials from .env
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST")

# Base URL for the API
BASE_URL = f"https://{RAPIDAPI_HOST}"

# Common headers for all requests.
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST
}

def call_endpoint(endpoint, params=None):
    """Generic function to perform GET request to a specific endpoint."""
    url = BASE_URL + endpoint
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise exception for bad status codes.
        return response.json()
    except requests.RequestException as error:
        print(f"Request to {url} failed: {error}")
        try:
            print("Response:", response.text)
        except Exception:
            pass
        return None

# --- Updated MLB Endpoints Functions (with '/mlb' prefix) ---

def fetch_play_by_play(game_id="401472105"):
    """Retrieves play-by-play data for a game."""
    endpoint = f"/mlb/play-by-play/{game_id}"
    print(f"\nFetching Play-by-Play Data for Game ID: {game_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_box_score(game_id="401472105"):
    """Retrieves box score data for a game."""
    endpoint = f"/mlb/box-score/{game_id}"
    print(f"\nFetching Box Score Data for Game ID: {game_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_summary(game_id="401472105"):
    """Retrieves summary data for a game."""
    endpoint = f"/mlb/summary/{game_id}"
    print(f"\nFetching Summary Data for Game ID: {game_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_picks(game_id="401472105"):
    """Retrieves picks data for a game."""
    endpoint = f"/mlb/picks/{game_id}"
    print(f"\nFetching Picks Data for Game ID: {game_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_schedule(year="2020", month="09", day="17"):
    """Retrieves schedule data."""
    endpoint = "/mlb/schedule"
    params = {"year": year, "month": month, "day": day}
    print(f"\nFetching Schedule Data for {year}-{month}-{day}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_scoreboard(year="2023", month="09", day="17", limit="200"):
    """Retrieves scoreboard data."""
    endpoint = "/mlb/scoreboard"
    params = {"year": year, "month": month, "day": day, "limit": limit}
    print(f"\nFetching Scoreboard Data for {year}-{month}-{day} with limit {limit}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_standings(year="2023", group="league"):
    """Retrieves standings data."""
    endpoint = "/mlb/standings"
    params = {"year": year, "group": group}
    print(f"\nFetching Standings Data for {year} (group: {group})")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_team_list(limit="500"):
    """Retrieves team list data."""
    endpoint = "/mlb/team-list"
    params = {"limit": limit}
    print(f"\nFetching Team List Data with limit {limit}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_team_info(team_id="52"):
    """Retrieves team info data for a given team."""
    endpoint = f"/mlb/team-info/{team_id}"
    print(f"\nFetching Team Info Data for Team ID: {team_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_team_players(team_id="52"):
    """Retrieves team players data for a given team."""
    endpoint = f"/mlb/team-players/{team_id}"
    print(f"\nFetching Team Players Data for Team ID: {team_id}")
    data = call_endpoint(endpoint)
    print(data)

def fetch_news():
    """Retrieves the latest news data."""
    endpoint = "/mlb/news"
    print("\nFetching News Data")
    data = call_endpoint(endpoint)
    print(data)

def fetch_player_statistic(player_id="42403"):
    """Retrieves player statistics data (primary endpoint)."""
    endpoint = "/mlb/player-statistic"
    params = {"playerId": player_id}
    print(f"\nFetching Player Statistic for Player ID: {player_id}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_player_statistic2(player_id="42403"):
    """Retrieves player statistics data (alternative endpoint)."""
    endpoint = "/mlb/player-statistic2"
    params = {"playerId": player_id}
    print(f"\nFetching Player Statistic2 for Player ID: {player_id}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_player_splits(player_id="42403", season="2024"):
    """Retrieves player splits data for a given player and season."""
    endpoint = "/mlb/player-splits"
    params = {"playerId": player_id, "season": season}
    print(f"\nFetching Player Splits Data for Player ID: {player_id} Season: {season}")
    data = call_endpoint(endpoint, params=params)
    print(data)

def fetch_team_roster(team_id="16", season="2023"):
    """Retrieves team roster data for a given team and season."""
    endpoint = "/mlb/team-roster"
    params = {"teamId": team_id, "season": season}
    print(f"\nFetching Team Roster Data for Team ID: {team_id} Season: {season}")
    data = call_endpoint(endpoint, params=params)
    print(data)

if __name__ == "__main__":
    print("Starting MLB API tests with RapidAPI endpoints...\n")
    
    fetch_play_by_play()
    time.sleep(1)
    
    fetch_box_score()
    time.sleep(1)
    
    fetch_summary()
    time.sleep(1)
    
    fetch_picks()
    time.sleep(1)
    
    fetch_schedule()
    time.sleep(1)
    
    fetch_scoreboard()
    time.sleep(1)
    
    fetch_standings()
    time.sleep(1)
    
    fetch_team_list()
    time.sleep(1)
    
    fetch_team_info()
    time.sleep(1)
    
    fetch_team_players()
    time.sleep(1)
    
    fetch_news()
    time.sleep(1)
    
    fetch_player_statistic()
    time.sleep(1)
    
    fetch_player_statistic2()
    time.sleep(1)
    
    fetch_player_splits()
    time.sleep(1)
    
    fetch_team_roster()
    
    print("\nMLB API tests completed.")
