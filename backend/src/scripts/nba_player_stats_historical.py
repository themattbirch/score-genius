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
    """Convert a time string in 'MM:SS' format to a float representing minutes."""
    try:
        if not time_str or not isinstance(time_str, str) or ":" not in time_str:
            return 0.0  # Default to 0.0 if time is missing or incorrect format
        minutes, seconds = map(int, time_str.split(":"))
        return minutes + (seconds / 60.0)  # Convert to decimal minutes
    except Exception as e:
        print(f"Error converting minutes '{time_str}': {e}")
        return 0.0  # Return 0.0 in case of error

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
    raw_text = resp.text.replace("&amp;apos;", "'").replace("&apos;", "'")
    data = json.loads(raw_text)

    # DEBUG: Print full player stats response
    print(f"\n=== DEBUG: Full Player Stats for Game {game_id} ===")
    print(json.dumps(data, indent=2))

    return data.get("response", [])

def process_day(date_obj):
    """Process all games for a specific date."""
    date_str = date_obj.strftime("%Y-%m-%d")
    season = f"{date_obj.year}-{date_obj.year+1}" if date_obj.month >= 10 else f"{date_obj.year-1}-{date_obj.year}"
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
        
        print(f"\nProcessing game: {away_team} @ {home_team} (Game ID: {game_id})")

        # Get player stats
        player_stats = get_player_stats(game_id)
        if not player_stats:
            print(f"No player stats found for game ID: {game_id}")
            continue

        players_processed = 0
        for player in player_stats:
            print(f"\n--- DEBUG: Raw Player Data ---")
            print(json.dumps(player, indent=2))

            player_id = player.get("player", {}).get("id", 0)
            player_name = player.get("player", {}).get("name", "Unknown")
            team_id = player.get("team", {}).get("id", 0)
            team_name = player.get("team", {}).get("name", "Unknown")

            # Extract all available fields dynamically
            all_fields = player.keys()
            print(f"\nDEBUG: Available fields for {player_name} - {all_fields}")
            print(f"\nDEBUG: Full fields for {player_name} - {json.dumps(player, indent=2)}")

            # Extract key statistics (handling nested dictionaries)
            minutes_raw = player.get("minutes", "0:00")
            minutes = convert_minutes(minutes_raw)  # Convert to float
            
            points = player.get("points", 0)
            rebounds = player.get("rebounds", {}).get("total", 0)
            assists = player.get("assists", 0)
            steals = player.get("steals", 0)
            blocks = player.get("blocks", 0)
            turnovers = player.get("turnovers", 0)
            fouls = player.get("fouls", 0)

            # Shooting stats
            fg_made = player.get("field_goals", {}).get("total", 0)
            fg_attempted = player.get("field_goals", {}).get("attempts", 0)
            three_made = player.get("threepoint_goals", {}).get("total", 0)
            three_attempted = player.get("threepoint_goals", {}).get("attempts", 0)
            ft_made = player.get("freethrows_goals", {}).get("total", 0)
            ft_attempted = player.get("freethrows_goals", {}).get("attempts", 0)

            # Dynamically print any fields not explicitly handled
            extra_stats = {key: player[key] for key in all_fields if key not in [
                "player", "team", "minutes", "points", "rebounds", "assists", "steals",
                "blocks", "turnovers", "fouls", "field_goals", "threepoint_goals", "freethrows_goals"
            ]}
            print(f"DEBUG: Extra stats for {player_name} - {json.dumps(extra_stats, indent=2)}")

            # Prepare record for database insertion
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

            print(f"\nUpserting record for {player_name}")
            try:
                supabase.table("nba_historical_player_stats").insert(player_record).execute()
                print(f"Insert successful for {player_name}")
            except Exception as e:
                print(f"Error inserting player stats for {player_name}: {e}")
                if hasattr(e, 'response'):
                    try:
                        print(f"Raw error response: {e.response.text}")
                    except:
                        pass

            players_processed += 1

        # Print all fields dynamically
        print(f"\nDEBUG: Full fields for {player_name} - {json.dumps(player, indent=2)}")

        print(f"Processed {players_processed} players for game {game_id}")
        processed_count += 1

    print(f"\nProcessed {processed_count} games for {date_str}")

def main():
    """Main function to process a range of dates."""
    start_date = datetime(2019, 11, 29)
    end_date = datetime(2025, 3, 5)

    print(f"Starting historical player data import from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    current = start_date
    while current <= end_date:
        process_day(current)
        print(f"Waiting 60 seconds before processing next date...")
        time.sleep(60)
        current += timedelta(days=1)

    print("\nCompleted processing historical player stats")

if __name__ == "__main__":
    main()
