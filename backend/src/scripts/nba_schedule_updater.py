# backend/src/scripts/nba_schedule_updater.py

import json
import requests
import sys, os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import traceback

# Add the backend root to the Python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from caching.supabase_client import supabase
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def get_current_season():
    """
    Determine the current NBA season based on today's date.
    NBA season generally runs from October to June.
    """
    today = datetime.now()
    if today.month >= 7 and today.month <= 9:
        # July-September is off-season, return the upcoming season
        return f"{today.year}-{today.year + 1}"
    elif today.month >= 10:
        # October to December is the start of a new season
        return f"{today.year}-{today.year + 1}"
    else:
        # January to June is the latter part of the current season
        return f"{today.year - 1}-{today.year}"

def get_games_by_date_range(league: str, season: str, start_date: str, end_date: str, timezone: str = 'America/New_York') -> list:
    """
    Fetches game data for a range of dates.
    
    Args:
        league: League ID (e.g., '12' for NBA)
        season: Season in format YYYY-YYYY
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
        timezone: Timezone for game times
        
    Returns:
        List of games from the API
    """
    all_games = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Limit to 30 days maximum to prevent excessive API calls
    max_days = 30
    if (end_date_obj - current_date).days > max_days:
        print(f"Warning: Date range too large. Limiting to {max_days} days.")
        end_date_obj = current_date + timedelta(days=max_days)
        end_date = end_date_obj.strftime('%Y-%m-%d')
    
    while current_date <= end_date_obj:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Fetching games for date: {date_str}")
        
        url = f"{BASE_URL}/games"
        params = {
            'league': league,
            'season': season,
            'date': date_str,
            'timezone': timezone
        }
        
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data and data['response']:
                games = data['response']
                all_games.extend(games)
                print(f"Found {len(games)} games on {date_str}")
            else:
                print(f"No games found for {date_str}")
            
            # Sleep to avoid hitting API rate limits
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching games for {date_str}: {e}")
            # Continue with the next date even if there's an error
        
        current_date += timedelta(days=1)
    
    return all_games

def normalize_team_name(name: str) -> str:
    """
    Normalizes a team name for consistent comparison.
    
    Args:
        name: Team name to normalize
        
    Returns:
        Normalized team name
    """
    if not name:
        return ""
    
    # Common name variations to standardize
    replacements = {
        "Los Angeles Clippers": "LA Clippers",
        "Portland Trail Blazers": "Portland Trailblazers",
        "Cleveland Cavaliers": "Cleveland Cavs",
        "Philadelphia 76ers": "Philadelphia Sixers",
        # Add more variations as needed
    }
    
    # First check for direct replacements
    for original, replacement in replacements.items():
        if name.lower() == original.lower():
            return replacement
    
    # General normalization
    # Strip whitespace, convert to title case
    normalized = ' '.join(name.strip().split()).title()
    return normalized

def create_schedule_table():
    """
    Creates the nba_game_schedule table in Supabase if it doesn't exist.
    Uses SQL directly through Supabase's pg extension.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS public.nba_game_schedule (
      id SERIAL PRIMARY KEY,
      game_id INTEGER NOT NULL UNIQUE,
      game_date DATE NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      scheduled_time TIMESTAMP WITH TIME ZONE,
      venue TEXT,
      status TEXT DEFAULT 'scheduled',
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_nba_game_schedule_date ON public.nba_game_schedule(game_date);
    """
    
    try:
        # Execute the SQL using Supabase's pg extension
        result = supabase.postgrest.rpc('exec_sql', {'query': sql}).execute()
        print("Schedule table created or verified.")
        return True
    except Exception as e:
        print(f"Error creating schedule table: {e}")
        traceback.print_exc()
        return False

def upsert_game_schedule(games: list) -> int:
    """
    Upserts game schedule data into Supabase.
    
    Args:
        games: List of game data from the API
        
    Returns:
        Number of games upserted
    """
    if not games:
        return 0
    
    # Try to create table first (will do nothing if it already exists)
    create_schedule_table()
    
    count = 0
    for game in games:
        try:
            # Extract relevant data
            game_id = game.get('id')
            if not game_id:
                print("Skipping game with no ID")
                continue
            
            status = game.get('status', {}).get('short')
            game_date_str = game.get('date')
            if not game_date_str:
                print(f"Skipping game {game_id} with no date")
                continue
            
            # Convert to date object and back to string to ensure consistent format
            try:
                game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                game_date = game_datetime.date().isoformat()
                scheduled_time = game_datetime.isoformat()
            except (ValueError, TypeError):
                print(f"Skipping game {game_id} with invalid date format: {game_date_str}")
                continue
            
            home_team = game.get('teams', {}).get('home', {}).get('name')
            away_team = game.get('teams', {}).get('away', {}).get('name')
            venue = game.get('venue', {}).get('name')
            
            if not home_team or not away_team:
                print(f"Skipping game {game_id} with missing team info")
                continue
            
            # Normalize team names for consistency
            home_team = normalize_team_name(home_team)
            away_team = normalize_team_name(away_team)
            
            # Prepare data for upsert
            game_data = {
                'game_id': game_id,
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'scheduled_time': scheduled_time,
                'venue': venue,
                'status': status or 'scheduled',
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Upsert the game schedule
            response = supabase.table("nba_game_schedule").upsert(
                game_data, 
                on_conflict='game_id'
            ).execute()
            
            if response.data:
                count += 1
            
        except Exception as e:
            print(f"Error upserting game {game.get('id', 'unknown')}: {e}")
            traceback.print_exc()
            continue
    
    return count

def update_schedule(days_ahead=7, days_behind=1):
    """
    Main function to update the NBA game schedule.
    Fetches games for a date range (yesterday through upcoming week by default)
    and updates the database.
    
    Args:
        days_ahead: Number of days ahead to fetch
        days_behind: Number of days behind to fetch
        
    Returns:
        Number of games updated
    """
    try:
        # Set date range
        today = datetime.now().date()
        start_date = (today - timedelta(days=days_behind)).isoformat()
        end_date = (today + timedelta(days=days_ahead)).isoformat()
        
        # Get current season
        season = get_current_season()
        
        print(f"Updating NBA schedule for {start_date} to {end_date}, season {season}")
        
        # Fetch games from API
        games = get_games_by_date_range('12', season, start_date, end_date)
        
        if not games:
            print("No games found for the specified date range.")
            return 0
        
        # Upsert games to database
        count = upsert_game_schedule(games)
        print(f"Successfully updated {count} games in schedule.")
        
        return count
        
    except Exception as e:
        print(f"Error updating schedule: {e}")
        traceback.print_exc()
        return 0

def get_games_for_date(date_str=None):
    """
    Retrieves games from the schedule for a specific date.
    
    Args:
        date_str: Date in format YYYY-MM-DD, defaults to today
        
    Returns:
        List of games for the specified date
    """
    if not date_str:
        date_str = datetime.now().date().isoformat()
    
    try:
        # Query the Supabase table
        response = supabase.table("nba_game_schedule").select("*").eq("game_date", date_str).execute()
        
        if response.data:
            return response.data
        
        # If no games found in database, try to fetch from API
        print(f"No games found in database for {date_str}, fetching from API...")
        season = get_current_season()
        games = get_games_by_date_range('12', season, date_str, date_str)
        
        if games:
            upsert_game_schedule(games)
            # Query again after upserting
            response = supabase.table("nba_game_schedule").select("*").eq("game_date", date_str).execute()
            return response.data
        
        return []
        
    except Exception as e:
        print(f"Error retrieving games for {date_str}: {e}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Example usage
    update_schedule(days_ahead=7, days_behind=1)
    
    # Get today's games
    today = datetime.now().date().isoformat()
    games = get_games_for_date(today)
    print(f"\nGames scheduled for {today}:")
    for game in games:
        print(f"{game['away_team']} @ {game['home_team']} - {game['scheduled_time']}")