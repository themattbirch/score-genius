import requests
from pprint import pprint

# API Configuration using your API-Sports key
API_KEY = 'd0c358b61e883d071bbc183c8fd72228'
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_team_id(team_name):
    """
    Fetches team information using the /teams endpoint with a search query.
    Returns the first matching team ID as a string, or None.
    """
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

def get_games_by_date(league, season, date, timezone='America/New_York'):
    """
    Fetches all game data from API-Basketball for the given date, league, and season.
    A timezone parameter is added to match the local time of interest.
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

def filter_game_by_teams(games_data, team1_id, team2_id):
    """
    Filters the games_data for a game that includes both team IDs.
    Returns the game ID if found, otherwise None.
    """
    for game in games_data.get('response', []):
        teams = game.get('teams', {})
        away_id = str(teams.get('away', {}).get('id'))
        home_id = str(teams.get('home', {}).get('id'))
        if (away_id == team1_id and home_id == team2_id) or (away_id == team2_id and home_id == team1_id):
            return str(game.get('id'))
    return None

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

def post_process_stats(stats_data):
    """
    Post-processes each player's shooting stats to print out the field_goals dictionary for review.
    """
    for stat in stats_data.get('response', []):
        fg = stat.get('field_goals', {})
        print(f"Player {stat.get('player', {}).get('name')}: Field Goals -> Attempts: {fg.get('attempts')}, Total Made: {fg.get('total')}")
    return stats_data

def run_test():
    """
    Tests the complete flow for in-game data fetching for the Minnesota Timberwolves vs. Cleveland Cavaliers matchup on Feb 10, 2025.
    """
    league = '12'             # NBA
    season = '2024-2025'       # Current season parameter
    date = '2025-02-10'        # Date for the game (if the game is live, live data will be returned)
    
    team1_name = "Utah Jazz"
    team2_name = "Los Angeles Lakers"
    
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print("Could not retrieve both team IDs; please verify team names and try again.")
        return

    print(f"Team IDs: {team1_name} -> {team1_id}, {team2_name} -> {team2_id}")
    
    # Fetch all games on the given date with the specified timezone
    games_data = get_games_by_date(league, season, date)
    pprint(games_data)
    print("-" * 80)
    
    # Filter for the game that includes both teams
    game_id = filter_game_by_teams(games_data, team1_id, team2_id)
    if game_id:
        print(f"Found game between {team1_name} and {team2_name}: Game ID {game_id}")
        # Fetch player statistics for this game.
        player_stats_data = get_player_box_stats(game_id)
        # Post-process and print field-goal attempts and totals
        processed_stats = post_process_stats(player_stats_data)
        print("\nFull Player Statistics Data:")
        pprint(processed_stats)
    else:
        print("No game data found for the specified matchup on that date. Please verify the date and season.")

def main():
    print("Testing in-game player statistics retrieval for Minnesota Timberwolves vs. Cleveland Cavaliers (Feb 10, 2025):")
    run_test()

if __name__ == "__main__":
    main()
