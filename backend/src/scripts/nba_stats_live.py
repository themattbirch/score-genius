# nba_stats_live.py
import requests
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo

# API Configuration using your API-Sports key
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_games_by_date(league, season, date, timezone='America/Los_Angeles'):
    """
    Fetches all game data from API-Basketball for the given date, league, season, and timezone.
    """
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date,
        'timezone': timezone
    }
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

def filter_live_games(games_data):
    """
    Filters the games_data to only include live games.
    We assume that a live game will have a status with 'short' not equal to "NS" (Not Started) or "FT" (Finished).
    """
    live_games = []
    for game in games_data.get('response', []):
        status = game.get('status', {})
        status_short = status.get('short')
        if status_short not in ["NS", "FT"]:
            live_games.append(game)
    return live_games

def get_player_box_stats(game_id):
    """
    Fetches detailed player statistics for a specific game using the 'ids' parameter.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {'ids': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched player statistics for game ID {game_id}")
        print("Status Code:", response.status_code)
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player statistics for game ID {game_id}: {e}")
        return {}

def print_game_info(game):
    """
    Prints key game information.
    """
    game_id = game.get('id')
    teams = game.get('teams', {})
    score = game.get('scores', {})
    status = game.get('status', {})
    print(f"Game ID: {game_id}")
    print(f"  Away: {teams.get('away', {}).get('name')}  |  Home: {teams.get('home', {}).get('name')}")
    print(f"  Scores: Away {score.get('away', {}).get('total')} - Home {score.get('home', {}).get('total')}")
    print(f"  Status: {status.get('long')} (Short: {status.get('short')}, Timer: {status.get('timer')})")
    print(f"  Venue: {game.get('venue')}")
    print("-" * 60)

def run_live_games():
    """
    Fetches all games for today (based on Pacific Time) and prints only those that are live, along with player statistics.
    """
    # Compute today's date in America/Los_Angeles timezone (to match game scheduling)
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    league = '12'            # NBA
    season = '2024-2025'      # Adjust if needed
    timezone = 'America/Los_Angeles'
    
    games_data = get_games_by_date(league, season, today_pst, timezone)
    live_games = filter_live_games(games_data)
    
    if not live_games:
        print("No live games found for today.")
        return
    
    print(f"\nFound {len(live_games)} live game(s):\n")
    for game in live_games:
        print_game_info(game)
        game_id = game.get('id')
        player_stats = get_player_box_stats(game_id)
        print("Player Statistics:")
        pprint(player_stats)
        print("=" * 80)

def main():
    print("Fetching live NBA game data for today (using Pacific Time)...")
    run_live_games()

if __name__ == "__main__":
    main()
