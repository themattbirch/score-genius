# File: backend/src/scripts/archive_live_data.py

import os
from datetime import datetime
from caching.supabase_client import supabase

def archive_live_data():
    """
    Archives live game and player data from the nba_live_game_stats and nba_live_player_stats tables 
    to historical tables, then clears the live data tables.
    """
    # Define the parse_minutes function here to ensure it's available
    def parse_minutes(time_str):
        """
        Convert a 'MM:SS' string to a float representing total minutes.
        E.g., "27:21" -> 27.35. If invalid or empty, returns 0.0.
        """
        if not time_str or not isinstance(time_str, str) or ':' not in time_str:
            return 0.0
        try:
            minutes_part, seconds_part = time_str.split(':')
            total_minutes = float(minutes_part) + float(seconds_part) / 60.0
            return round(total_minutes, 2)
        except ValueError:
            return 0.0
    
    # --- PART 1: Archive and clear game stats ---
    try:
        # Fetch live game data from Supabase
        live_game_data_response = supabase.table("nba_live_game_stats").select("*").execute()
        live_game_data = live_game_data_response.data
        
        if live_game_data:
            print(f"Found {len(live_game_data)} live game stats to archive.")
            
            # Collect game IDs for deletion later
            game_record_ids = []
            
            # Iterate through each game record and archive
            for record in live_game_data:
                # Store ID for deletion
                record_id = record.get("id")
                if record_id is not None:
                    game_record_ids.append(record_id)
                    
                # Remove the 'id' key so we don't try to insert into a GENERATED ALWAYS column.
                record.pop("id", None)
                
                # Remove fields that don't exist in the historical table
                record.pop("current_quarter", None)
                record.pop("status", None)  # Remove status field which causes errors
                
                # Clean each field: convert any string "None" to actual Python None.
                for key, value in record.items():
                    if isinstance(value, str) and value.strip().lower() == "none":
                        record[key] = None
                
                game_id = record.get("game_id")
                # Check if the record has a valid 'player_id' (not None)
                if record.get("player_id") is not None:
                    result = supabase.table("nba_historical_player_stats") \
                        .upsert(record, on_conflict="game_id, player_id") \
                        .execute()
                    print(f"Archived player stat for game {game_id}, player {record.get('player_id')}")
                else:
                    result = supabase.table("nba_historical_game_stats") \
                        .upsert(record, on_conflict="game_id") \
                        .execute()
                    print(f"Archived game stat for game {game_id}")
            
            # Clear the live game stats table after archival using specific IDs
            if game_record_ids:
                for record_id in game_record_ids:
                    try:
                        supabase.table("nba_live_game_stats").delete().eq("id", record_id).execute()
                    except Exception as e:
                        print(f"Error deleting game record ID {record_id}: {e}")
                
                print(f"Deleted {len(game_record_ids)} records from live game stats table")
            else:
                print("No game record IDs to delete")
        else:
            print("No live game data to archive.")
            
    except Exception as e:
        print(f"Error during game stats archive process: {e}")
    
    # --- PART 2: Archive and clear player stats ---
    try:
        # Fetch live player data from Supabase
        live_player_data_response = supabase.table("nba_live_player_stats").select("*").execute()
        live_player_data = live_player_data_response.data
        
        if live_player_data:
            print(f"Found {len(live_player_data)} live player stats to archive.")
            
            # Collect player record IDs for deletion later
            player_record_ids = []
            
            # Iterate through each player record and archive
            for record in live_player_data:
                # Store ID for deletion
                record_id = record.get("id")
                if record_id is not None:
                    player_record_ids.append(record_id)
                    
                # Remove the 'id' key so we don't try to insert into a GENERATED ALWAYS column.
                record.pop("id", None)
                
                # Remove fields that don't exist in the historical table
                record.pop("current_quarter", None)
                record.pop("status", None)  # Remove status field which causes errors
                
                # Clean each field: convert any string "None" to actual Python None.
                for key, value in record.items():
                    if isinstance(value, str):
                        if value.strip().lower() == "none":
                            record[key] = None
                        # Convert minutes field if it's in MM:SS format
                        elif key == "minutes" and ":" in value:
                            record[key] = parse_minutes(value)
                
                game_id = record.get("game_id")
                player_id = record.get("player_id")
                
                if game_id is not None and player_id is not None:
                    result = supabase.table("nba_historical_player_stats") \
                        .upsert(record, on_conflict="game_id, player_id") \
                        .execute()
                    print(f"Archived player stat for game {game_id}, player {player_id}")
                else:
                    print(f"Skipping record with missing game_id or player_id: {record}")
            
            # Clear the live player stats table after archival using specific IDs
            if player_record_ids:
                for record_id in player_record_ids:
                    try:
                        supabase.table("nba_live_player_stats").delete().eq("id", record_id).execute()
                    except Exception as e:
                        print(f"Error deleting player record ID {record_id}: {e}")
                
                print(f"Deleted {len(player_record_ids)} records from live player stats table")
            else:
                print("No player record IDs to delete")
        else:
            print("No live player data to archive.")
            
    except Exception as e:
        print(f"Error during player stats archive process: {e}")

if __name__ == "__main__":
    archive_live_data()