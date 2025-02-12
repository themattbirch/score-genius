# nba_stats_live_single.py
import requests
import json
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo

API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_team_id(team_name):
    url = f"{BASE_URL}/teams"
    params = {'search': team_name}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('results', 0) > 0 and data.get('response'):
            print(f"Team search '{team_name}':")
            pprint(data['response'][0])
            return str(data['response'][0]['id'])
        else:
            print(f"No team found for '{team_name}'.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team data for {team_name}: {e}")
        return None

def get_games_by_date(league, season, date, timezone="America/Los_Angeles"):
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

def filter_game_by_teams(games_data, team1_id, team2_id):
    for game in games_data.get('response', []):
        teams = game.get('teams', {})
        away_id = str(teams.get('away', {}).get('id'))
        home_id = str(teams.get('home', {}).get('id'))
        if (away_id == team1_id and home_id == team2_id) or (away_id == team2_id and home_id == team1_id):
            return str(game.get('id'))
    return None

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

def print_game_info(game):
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

def run_live_game_single():
    league = '12'
    season = '2024-2025'
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    timezone = 'America/Los_Angeles'
    
    team1_name = "San Antonio Spurs"
    team2_name = "Washington Wizards"
    
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    if not team1_id or not team2_id:
        print("Could not retrieve both team IDs; please verify team names and try again.")
        return
    print(f"Team IDs: {team1_name} -> {team1_id}, {team2_name} -> {team2_id}")
    
    games_data = get_games_by_date(league, season, today_pst, timezone)
    game_id = filter_game_by_teams(games_data, team1_id, team2_id)
    if game_id:
        print(f"Found live game between {team1_name} and {team2_name}: Game ID {game_id}")
        game = next((g for g in games_data.get('response', []) if str(g.get('id')) == game_id), None)
        if game:
            print_game_info(game)
        player_stats = get_player_box_stats(game_id)
        print("Player Statistics:")
        pprint(player_stats)
    else:
        print("No live game found for the specified matchup today.")

def main():
    print("Fetching live statistics for a single NBA game (nba_stats_live_single.py)...")
    run_live_game_single()

if __name__ == "__main__":
    main()
