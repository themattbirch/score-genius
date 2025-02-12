# nba_stats_historical.py
import requests
from pprint import pprint
import time

API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games(league, season, date, timezone="America/Los_Angeles"):
    url = f"{BASE_URL}/games"
    params = {'league': league, 'season': season, 'date': date, 'timezone': timezone}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched game data for {date} (Season {season}, Timezone: {timezone})")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_player_box_stats(game_ids):
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_ids}
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

def post_process_stats(stats_data):
    for stat in stats_data.get('response', []):
        fg = stat.get('field_goals', {})
        print(f"Player {stat.get('player', {}).get('name')}: Field Goals -> Attempts: {fg.get('attempts')}, Total Made: {fg.get('total')}")
    return stats_data

def run_historical_games():
    test_cases = [
        {'league': '12', 'season': '2019-2020', 'date': '2019-11-23'},
        # Additional historical dates can be added here.
    ]
    
    for test in test_cases:
        data = get_games(test.get('league'), test.get('season'), test.get('date'))
        print("\nGame Data:")
        pprint(data)
        print("-" * 80)
        time.sleep(2)
        if data.get('response'):
            game_id = str(data['response'][0].get('id'))
            print(f"Fetching player statistics for game ID {game_id}")
            stats_data = get_player_box_stats(game_id)
            processed_stats = post_process_stats(stats_data)
            print("\nFull Player Statistics Data:")
            pprint(processed_stats)
            print("=" * 80)
        else:
            print("No game data found for date:", test.get('date'))

def main():
    print("Testing historical game data retrieval (nba_stats_historical.py):")
    run_historical_games()

if __name__ == "__main__":
    main()
