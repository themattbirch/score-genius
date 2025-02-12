# nba_stats_final.py
import requests
import json
from pprint import pprint

# API Configuration using your API-Sports key for NBA
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league, season, date):
    """
    Fetches all game data from API-Basketball for the given date, league, and season.
    """
    url = f"{BASE_URL}/games"
    params = {'league': league, 'season': season, 'date': date}
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

def get_team_stats(game_id):
    """
    Fetches team statistics for a specific game.
    """
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched team statistics for game ID {game_id}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team statistics for game ID {game_id}: {e}")
        return {}

def get_player_box_stats(game_id):
    """
    Fetches detailed player statistics for a specific game,
    and globally replaces &apos; with a real apostrophe in the raw JSON text
    before parsing into a Python dictionary.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_id}

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        # Get the raw JSON as text
        raw_text = response.text
        
        # Handle possible double-encoding like &amp;apos;, then handle &apos;
        raw_text = raw_text.replace("&amp;apos;", "'")
        raw_text = raw_text.replace("&apos;", "'")
        
        # Parse the cleaned text into a Python dict
        data = json.loads(raw_text)
        
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching player statistics for game ID {game_id}: {e}")
        return {}

def post_process_stats(stats_data):
    """
    Post-processes each player's shooting stats to print field-goal attempts and totals.
    """
    for stat in stats_data.get('response', []):
        fg = stat.get('field_goals', {})
        print(f"Player {stat.get('player', {}).get('name')}: Field Goals -> Attempts: {fg.get('attempts')}, Total Made: {fg.get('total')}")
    return stats_data

def run_all_games():
    league = '12'            # NBA
    season = '2024-2025'      # Current season parameter
    date = '2025-02-12'       # Date for which to fetch game data
    games_data = get_games_by_date(league, season, date)
    if not games_data.get('response'):
        print("No game data found for the specified date.")
        return
    for game in games_data['response']:
        game_id = game.get('id')
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
        player_stats = get_player_box_stats(game_id)
        print("Player Statistics:")
        post_process_stats(player_stats)
        pprint(player_stats)
        print("=" * 60)

def main():
    print("Fetching ALL NBA game data for the specified day and printing statistics...")
    run_all_games()

if __name__ == "__main__":
    main()
