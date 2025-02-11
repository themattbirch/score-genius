import requests
from pprint import pprint

# API Configuration using your direct API-Sports key for NFL
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.american-football.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.american-football.api-sports.io'
}

# Super Bowl parameters: This game (Feb 9, 2025) is between the Kansas City Chiefs and Philadelphia Eagles.
# Note: The season is updated to "2024" assuming that the free plan supports the most recent season that includes the Super Bowl.
GAME_PARAMS = {
    'league': '1',       # NFL league id (usually "1")
    'season': '2024',    # Adjusted season for a Feb 2025 game
    'date': '2025-02-09'  # Date for the Super Bowl
}

def get_game_data():
    """Retrieve basic game information for the specified NFL game (Super Bowl)."""
    try:
        response = requests.get(f"{BASE_URL}/games", headers=HEADERS, params=GAME_PARAMS)
        print(f"Game Data Status Code: {response.status_code}")
        print("Request URL:", response.url)
        return response.json()
    except Exception as e:
        print(f"Error fetching game data: {e}")
        return None

def get_player_stats(game_id, stat_type):
    """Retrieve player statistics for a given game and stat category."""
    url = f"{BASE_URL}/games/statistics/players"
    params = {
        'id': game_id,
        'group': stat_type
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        print(f"\n{stat_type.title()} Stats Status Code: {response.status_code}")
        return response.json()
    except Exception as e:
        print(f"Error fetching {stat_type} stats: {e}")
        return None

def get_team_stats(game_id):
    """Retrieve team statistics for a given game."""
    url = f"{BASE_URL}/games/statistics/teams"
    params = {'id': game_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        print(f"\nTeam Stats Status Code: {response.status_code}")
        return response.json()
    except Exception as e:
        print(f"Error fetching team stats: {e}")
        return None

def main():
    # Retrieve game data for the Super Bowl (Kansas City Chiefs vs Philadelphia Eagles)
    games = get_game_data()
    print("\nGAME DATA:")
    pprint(games)
    
    # Check if any game data was returned and use the first game's id for stats.
    # (Ideally, the response should contain only one game—the Super Bowl—on Feb 9, 2025.)
    if games and games.get('response'):
        game_id = games['response'][0]['game']['id']
        print(f"\nFound Game ID: {game_id}")
        
        # Get team statistics
        print("\nTEAM STATISTICS:")
        team_stats = get_team_stats(game_id)
        pprint(team_stats)
        
        # Define a list of stat categories to query for player stats
        stat_groups = [
            'passing',
            'rushing',
            'receiving',
            'defensive',
            'kicking',
            'kick_returns',
            'punting'
        ]
        
        # Loop through each stat category and fetch the player stats
        for group in stat_groups:
            print(f"\n{group.upper()} STATISTICS:")
            stats = get_player_stats(game_id, group)
            pprint(stats)
    else:
        print("No game data found.")

if __name__ == "__main__":
    main()
