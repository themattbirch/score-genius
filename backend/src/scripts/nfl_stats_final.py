# nfl_stats_final.py
import requests
import json
from pprint import pprint

# API Configuration using your API-Sports key for NFL
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.american-football.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.american-football.api-sports.io'
}

def get_game_data(date):
    """
    Retrieves NFL game data for the given date.
    """
    params = {
        'league': '1',       # NFL league id
        'season': '2024-2025',    # Adjusted season for Feb 2025 games
        'date': date
    }
    try:
        response = requests.get(f"{BASE_URL}/games", headers=HEADERS, params=params)
        print(f"Fetched NFL game data for {date}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except Exception as e:
        print(f"Error fetching NFL game data for {date}: {e}")
        return None

def get_team_stats(game_id):
    """
    Retrieves team statistics for a given NFL game.
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        print(f"Fetched team statistics for NFL game ID {game_id}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except Exception as e:
        print(f"Error fetching team statistics for NFL game ID {game_id}: {e}")
        return None

def get_player_stats(game_id, stat_type):
    """
    Retrieves player statistics for a given NFL game and stat category.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'id': game_id, 'group': stat_type}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        print(f"{stat_type.title()} Stats for NFL game {game_id}: Status Code: {response.status_code}")
        print("Request URL:", response.url)
        return response.json()
    except Exception as e:
        print(f"Error fetching {stat_type} stats for NFL game {game_id}: {e}")
        return None

def run_all_games():
    date = '2024-11-11'  # Date for which to fetch NFL game data (e.g., Super Bowl day)
    games = get_game_data(date)
    if not games or not games.get('response'):
        print("No NFL game data found for the specified date.")
        return
    for game in games['response']:
        game_id = game.get('game', {}).get('id')
        teams = game.get('teams', {})
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Away: {teams.get('away', {}).get('name')} | Home: {teams.get('home', {}).get('name')}")
        scores = game.get('scores', {})
        print(f"Scores: Away {scores.get('away', {}).get('total')} - Home {scores.get('home', {}).get('total')}")
        status = game.get('status', {})
        print(f"Status: {status.get('long')}")
        team_stats = get_team_stats(game_id)
        print("Team Statistics:")
        pprint(team_stats)
        stat_groups = ['passing', 'rushing', 'receiving', 'defensive', 'kicking', 'kick_returns', 'punting']
        for group in stat_groups:
            print(f"\n{group.upper()} STATISTICS:")
            stats = get_player_stats(game_id, group)
            pprint(stats)
        print("=" * 60)

def main():
    print("Fetching ALL NFL game data for the specified day and printing statistics...")
    run_all_games()

if __name__ == "__main__":
    main()
