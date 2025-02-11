
import requests
from pprint import pprint
import time

# API Configuration using your API-Sports key (ensure this key is valid)
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games(league, season, date):
    """
    Fetches game data from API-Basketball for the given parameters.
    """
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched game data for {date} (Season {season})")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_player_box_stats(game_ids):
    """
    Fetches detailed player statistics (box score data) using the /games/statistics/players endpoint.
    Use the 'ids' parameter to supply one or more game IDs (as a string, e.g., "1912" or "1912-1913").
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {
        'ids': game_ids  # Supply the game ID(s) as a string.
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched player statistics for game IDs {game_ids}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player statistics for game IDs {game_ids}: {e}")
        return {}

def run_tests():
    """
    Runs test calls to fetch both game data and detailed player statistics.
    """
    # Test fetching game data using historical parameters
    test_cases = [
        {'league': '12', 'season': '2019-2020', 'date': '2019-11-23'}
    ]
    
    for test in test_cases:
        data = get_games(
            league=test.get('league'),
            season=test.get('season'),
            date=test.get('date')
        )
        print("\nGame Data:")
        pprint(data)
        print("-" * 80)
        time.sleep(2)
    
    # Test fetching player box score statistics using the "ids" parameter
    print("\nTesting player box score statistics:")
    game_ids = '1912'  # Using a single game id as a string; you can separate multiple IDs with hyphens.
    player_stats = get_player_box_stats(game_ids)
    pprint(player_stats)
    print("-" * 80)

def main():
    run_tests()

if __name__ == "__main__":
    main()