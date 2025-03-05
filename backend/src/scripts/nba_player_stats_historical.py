# backend/src/scripts/nba_player_stats_historical.py

import json
import requests
import sys, os
import time
from datetime import datetime, timedelta

# Add the backend root to the Python path so we can import from caching and config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase

API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

def convert_minutes(time_str):
    """Convert a time string in "MM:SS" format to a float representing minutes."""
    try:
        if not time_str or not isinstance(time_str, str) or ":" not in time_str:
            return float(time_str) if time_str and str(time_str).strip() != "" else 0.0
        minutes, seconds = time_str.split(":")
        return float(minutes) + float(seconds) / 60.0
    except Exception as e:
        print(f"Error converting minutes '{time_str}': {e}")
        return 0.0

def get_games_by_date(league, season, date):
    """Fetch games for a specific date from the API."""
    url = f"{BASE_URL}/games"
    params = {"league": league, "season": season, "date": date}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])

def get_player_stats(game_id):
    """Fetch player statistics for a specific game."""
    url = f"{BASE_URL}/games/statistics/players"
    params = {"ids": game_id}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    # Clean up any HTML encoding issues in the response
    raw_text = resp.text.replace("&amp;apos;", "'").replace("&apos;", "'")
    data = json.loads(raw_text)
    return data.get("response", [])

def get_additional_stats(game_id: int, player_id: int) -> dict:
    """
    Make an additional API call (as in the live script) to fetch detailed stats,
    which may include free throw data.
    """
    url = f"{BASE_URL}/games/statistics/players"
    params = {
        'game': game_id,
        'player': player_id
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched additional stats for player {player_id}, game {game_id}")
        return data.get('response', [{}])[0] if data.get('response') else {}
    except Exception as e:
        print(f"Error fetching additional stats for player {player_id}, game {game_id}: {e}")
        return {}

def parse_season_from_date(date_obj):
    """Determine the NBA season string from a date."""
    if date_obj.month >= 10:
        return f"{date_obj.year}-{date_obj.year+1}"
    else:
        return f"{date_obj.year-1}-{date_obj.year}"

def process_day(date_obj):
    """Process all games for a specific date."""
    date_str = date_obj.strftime("%Y-%m-%d")
    season = parse_season_from_date(date_obj)
    print(f"\n=== {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    processed_count = 0
    for g in games_list:
        if g.get("status", {}).get("short") != "FT":
            continue  # Process only final games

        game_id = g["id"]
        home_team = g["teams"]["home"]["name"]
        away_team = g["teams"]["away"]["name"]
        home_team_id = g["teams"]["home"]["id"]
        away_team_id = g["teams"]["away"]["id"]

        print(f"\nProcessing game: {away_team} @ {home_team} (Game ID: {game_id})")

        # Get player stats
        player_stats = get_player_stats(game_id)
        if not player_stats:
            print(f"No player stats found for game ID: {game_id}")
            continue

        players_processed = 0
        for player in player_stats:
            player_id = player.get("player", {}).get("id", 0)
            raw_player_name = player.get("player", {}).get("name", "Unknown")
            team_id = player.get("team", {}).get("id", 0)

            # Determine team name based on team ID
            if team_id == home_team_id:
                team_name = home_team
            elif team_id == away_team_id:
                team_name = away_team
            else:
                team_name = player.get("team", {}).get("name", "Unknown")

            # Fix player name format - FROM "Last First" TO "First Last"
            name_parts = raw_player_name.split()
            if len(name_parts) == 2 and not name_parts[0].endswith('.'):
                player_name = f"{name_parts[1]} {name_parts[0]}"
            else:
                player_name = raw_player_name

            # Extract minutes with fallback
            minutes_raw = player.get("minutes", "0:00")
            minutes = convert_minutes(minutes_raw)

            # Extract main stats with safe fallbacks
            points = player.get("points", 0) or 0
            rebounds = player.get("rebounds", {}).get("total", 0) or 0
            assists = player.get("assists", 0) or 0

            # Extract defensive stats with fallbacks to 1 (for visibility)
            steals = player.get("steals", 1) or 1
            blocks = player.get("blocks", 1) or 1
            turnovers = player.get("turnovers", 1) or 1
            fouls = player.get("fouls", 1) or 1

            # Extract shooting stats
            fg_made = player.get("field_goals", {}).get("total", 0) or 0
            fg_attempted = player.get("field_goals", {}).get("attempts", 0) or 0
            three_made = player.get("threepoint_goals", {}).get("total", 0) or 0
            three_attempted = player.get("threepoint_goals", {}).get("attempts", 0) or 0

            # --- Free Throw Extraction Logic ---
            # First, try extracting from the main player data.
            if 'free_throws' in player and isinstance(player['free_throws'], dict):
                ft = player['free_throws']
                ft_made = ft.get('total', 0)
                ft_attempted = ft.get('attempts', 0)
            elif 'fta' in player and 'ftm' in player:
                ft_made = player.get('ftm', 0)
                ft_attempted = player.get('fta', 0)
            elif 'free_throws' in player:
                try:
                    ft_made = float(player['free_throws'])
                    ft_attempted = ft_made  # assume equal if only one value exists
                except (ValueError, TypeError):
                    ft_made = 0
                    ft_attempted = 0
            else:
                ft_made = 0
                ft_attempted = 0

            # If no FT data found (both zero), call the additional stats endpoint
            if ft_made == 0 and ft_attempted == 0:
                additional = get_additional_stats(game_id, player_id)
                if additional:
                    ft_data = additional.get("free_throws", {})
                    ft_made = ft_data.get("made", ft_made)
                    ft_attempted = ft_data.get("attempts", ft_attempted)

            # Ensure values are integers
            ft_made = int(ft_made or 0)
            ft_attempted = int(ft_attempted or 0)

            if ft_made > 0 or ft_attempted > 0:
                print(f"FT stats for {player_name}: {ft_made}/{ft_attempted}")

            # Create the player record
            player_record = {
                "game_id": game_id,
                "player_id": player_id,
                "player_name": player_name,
                "team_id": team_id,
                "team_name": team_name,
                "minutes": minutes,
                "points": points,
                "rebounds": rebounds,
                "assists": assists,
                "steals": steals,
                "blocks": blocks,
                "turnovers": turnovers,
                "fouls": fouls,
                "fg_made": fg_made,
                "fg_attempted": fg_attempted,
                "three_made": three_made,
                "three_attempted": three_attempted,
                "ft_made": ft_made,
                "ft_attempted": ft_attempted,
                "game_date": date_str,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }

            print(f"Upserting record for {player_name}")
            try:
                res = (
                    supabase
                    .table("nba_historical_player_stats")
                    .insert(player_record)
                    .execute()
                )
                print(f"Insert successful for player {player_name}")
            except Exception as e:
                print(f"Error inserting player stats for {player_name}: {e}")
                if hasattr(e, 'response'):
                    try:
                        print(f"Raw error response: {e.response.text}")
                    except:
                        pass

            players_processed += 1

        print(f"Processed {players_processed} players for game {game_id}")
        processed_count += 1

    print(f"\nProcessed {processed_count} games for {date_str}")

def main():
    """Main function to process a range of dates."""
    if len(sys.argv) > 1:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2] if len(sys.argv) > 2 else start_date_str
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    else:
        # Default date range
        start_date = datetime(2018, 10, 19)
        end_date = datetime(2025, 3, 5)

    print(f"Starting historical player data import from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    current = start_date
    days_processed = 0
    while current <= end_date:
        process_day(current)
        # Rate limiting to avoid API issues
        wait_seconds = 60
        print(f"Waiting {wait_seconds} seconds before processing next date...")
        time.sleep(wait_seconds)
        current += timedelta(days=1)
        days_processed += 1

    print(f"\nCompleted processing {days_processed} days of games")

if __name__ == "__main__":
    main()
