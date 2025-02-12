# nba_games_preview.py
import requests
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo

# API Configuration using your API-Sports key for NBA
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

def filter_pregame_games(games_data):
    """
    Filters the games_data to include games that have not yet started.
    We check if the status 'short' equals "NS" or if the 'long' status contains "Not Started".
    Debug prints are added to see what status each game has.
    """
    pregame_games = []
    for game in games_data.get('response', []):
        status = game.get('status', {})
        status_short = status.get('short')
        status_long = status.get('long')
        # Debug: print the status for each game
        print(f"Game ID {game.get('id')} status: short='{status_short}', long='{status_long}'")
        if status_short == "NS" or (status_long and "Not Started" in status_long):
            pregame_games.append(game)
    return pregame_games

def get_odds(game_id, league, season, bookmaker=6, bet=1):
    """
    Fetches betting odds for a specific game using the /odds endpoint.
    """
    url = f"{BASE_URL}/odds"
    params = {
        'season': season,
        'bet': bet,
        'bookmaker': bookmaker,
        'game': game_id,
        'league': league
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print(f"Fetched odds for game {game_id}")
        print("Request URL:", response.url)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds for game {game_id}: {e}")
        return {}

def get_standings(league, season):
    """
    Optional: Fetch league standings to extract team records.
    """
    url = f"{BASE_URL}/standings"
    params = {'league': league, 'season': season}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        print("Fetched standings data.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching standings: {e}")
        return {}

def print_game_preview(game, odds_data, standings_data=None):
    """
    Prints key pregame information including game time, venue, teams, and betting odds.
    Optionally prints team records from standings data.
    """
    game_id = game.get('id')
    teams = game.get('teams', {})
    game_datetime = game.get('date')
    venue = game.get('venue')
    
    print("=" * 60)
    print(f"Game ID: {game_id}")
    print(f"  Scheduled Date/Time: {game_datetime}")
    print(f"  Venue: {venue}")
    print(f"  Away: {teams.get('away', {}).get('name')}")
    print(f"  Home: {teams.get('home', {}).get('name')}")
    
    if standings_data:
        print("Team Records:")
        for entry in standings_data.get('response', []):
            team = entry.get('team', {})
            if team.get('id') in [teams.get('away', {}).get('id'), teams.get('home', {}).get('id')]:
                print(f"  {team.get('name')}: {entry.get('record', 'Record not available')}")
    
    print("Betting Odds Preview:")
    pprint(odds_data)
    print("=" * 60)

def run_pregame_preview():
    league = '12'           # NBA
    season = '2024-2025'     # Season parameter
    # Use a date where games are scheduled but not started (adjust this date as needed)
    date = '2025-02-12'
    # IMPORTANT: Pass the timezone to ensure the correct schedule; we use Los Angeles time
    games_data = get_games_by_date(league, season, date, timezone='America/Los_Angeles')
    pregame_games = filter_pregame_games(games_data)
    
    if not pregame_games:
        print("No pregame games found for the specified date.")
        return
    
    # Optionally, fetch standings data if needed.
    standings_data = None
    # standings_data = get_standings(league, season)
    
    print(f"\nFound {len(pregame_games)} pregame game(s):\n")
    for game in pregame_games:
        game_id = game.get('id')
        odds_data = get_odds(game_id, league, season)
        print_game_preview(game, odds_data, standings_data)

def main():
    print("Fetching NBA pregame preview data...")
    run_pregame_preview()

if __name__ == "__main__":
    main()
