# backend/src/scripts/nba_schedule_fetcher.py

import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from config import API_SPORTS_KEY

# API configuration
API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_current_season():
    """Determine the current NBA season based on the date."""
    today = datetime.now()
    if today.month >= 10:  # October to December is new season
        return f"{today.year}-{today.year + 1}"
    else:  # January to June is latter part of current season
        return f"{today.year - 1}-{today.year}"

def get_games_by_date(date_str, timezone='America/Los_Angeles'):
    """Fetch NBA games for a specific date."""
    season = get_current_season()
    url = f"{BASE_URL}/games"
    params = {
        'league': '12',  # NBA league ID
        'season': season,
        'date': date_str,
        'timezone': timezone
    }
    
    try:
        print(f"Fetching games for {date_str}...")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'response' in data and data['response']:
            games = data['response']
            print(f"Found {len(games)} games for {date_str}")
            return games
        else:
            print(f"No games found for {date_str}")
            return []
    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
        return []

def normalize_team_name(name):
    """Normalize team names for consistent comparison."""
    if not name:
        return ""
    
    # Common NBA team name mappings
    name_map = {
        "sixers": "philadelphia 76ers",
        "76ers": "philadelphia 76ers",
        "blazers": "portland trail blazers",
        "trailblazers": "portland trail blazers",
        "cavs": "cleveland cavaliers",
        "mavs": "dallas mavericks",
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "t-wolves": "minnesota timberwolves",
        "timberwolves": "minnesota timberwolves"
    }
    
    # Check if the team name contains any known variations
    name_lower = name.lower()
    for key, value in name_map.items():
        if key in name_lower:
            return value
    
    # Otherwise return the original name in lowercase
    return name_lower.strip()

def fetch_todays_schedule():
    """Fetch today's NBA schedule and return as a DataFrame."""
    today = datetime.now().strftime('%Y-%m-%d')
    games = get_games_by_date(today)
    
    if not games:
        return pd.DataFrame()
    
    # Extract relevant info into a DataFrame
    schedule = []
    for game in games:
        try:
            # Get the game details
            game_id = game.get('id')
            status = game.get('status', {}).get('short')
            
            # Get team information
            home_team = game.get('teams', {}).get('home', {}).get('name')
            away_team = game.get('teams', {}).get('away', {}).get('name')
            
            # Normalize team names
            home_team_norm = normalize_team_name(home_team)
            away_team_norm = normalize_team_name(away_team)
            
            # Add to schedule list
            schedule.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_normalized': home_team_norm,
                'away_team_normalized': away_team_norm,
                'status': status,
                'game_date': today
            })
        except Exception as e:
            print(f"Error processing game: {e}")
    
    return pd.DataFrame(schedule)

def verify_game_teams(live_game_home, live_game_away, official_schedule_df):
    """
    Verify if a live game's teams match any game in the official schedule.
    Returns the matching game_id if found, None otherwise.
    """
    if official_schedule_df.empty:
        return None
    
    # Normalize the team names from the live game
    live_home_norm = normalize_team_name(live_game_home)
    live_away_norm = normalize_team_name(live_game_away)
    
    # Check for direct match
    direct_match = official_schedule_df[
        (official_schedule_df['home_team_normalized'] == live_home_norm) & 
        (official_schedule_df['away_team_normalized'] == live_away_norm)
    ]
    
    if not direct_match.empty:
        return direct_match.iloc[0]['game_id']
    
    # Check for reversed match (home/away swapped)
    reverse_match = official_schedule_df[
        (official_schedule_df['home_team_normalized'] == live_away_norm) & 
        (official_schedule_df['away_team_normalized'] == live_home_norm)
    ]
    
    if not reverse_match.empty:
        return reverse_match.iloc[0]['game_id']
    
    # No match found
    return None

def match_teams_to_schedule(live_games_df):
    """
    Match team names in live games to the official schedule.
    Updates game_id if necessary and adds verification status.
    """
    if live_games_df is None or live_games_df.empty:
        return live_games_df
    
    # Make a copy to avoid modifying the original
    verified_games = live_games_df.copy()
    
    # Fetch today's official schedule
    official_schedule = fetch_todays_schedule()
    
    if official_schedule.empty:
        print("No official schedule available to verify against")
        verified_games['verified'] = False
        return verified_games
    
    # Add verification column
    verified_games['verified'] = False
    
    # Process each game
    for idx, game in verified_games.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Verify against official schedule
        correct_game_id = verify_game_teams(home_team, away_team, official_schedule)
        
        if correct_game_id:
            # Update game_id if it differs from the current one
            current_game_id = game.get('game_id')
            if current_game_id != correct_game_id:
                print(f"Updating game_id for {home_team} vs {away_team}: {current_game_id} -> {correct_game_id}")
                verified_games.at[idx, 'game_id'] = correct_game_id
            
            verified_games.at[idx, 'verified'] = True
        else:
            print(f"Could not verify game: {home_team} vs {away_team}")
    
    print(f"Verified {verified_games['verified'].sum()} of {len(verified_games)} games")
    return verified_games

if __name__ == "__main__":
    # Test the schedule fetcher
    schedule = fetch_todays_schedule()
    if not schedule.empty:
        print("\nToday's NBA Schedule:")
        for _, game in schedule.iterrows():
            print(f"{game['away_team']} @ {game['home_team']} (Game ID: {game['game_id']})")
    else:
        print("No games scheduled for today.")