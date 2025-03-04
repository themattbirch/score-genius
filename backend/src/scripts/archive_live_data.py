# File: backend/src/scripts/archive_live_data.py

import os
from datetime import datetime
from caching.supabase_client import supabase

def archive_live_data():
    """
    Archives live game data from the nba_live_game_stats table to historical tables,
    then clears the nba_live_game_stats table.
    Process:
      1. Fetch all records from the live game stats table.
      2. For each record, remove the 'id' key and clean any string "None" values.
      3. If the record contains a valid 'player_id', upsert into nba_historical_player_stats;
         otherwise, upsert into nba_2024_25_game_stats.
      4. After successful archival, clear all records from nba_live_game_stats.
    """
    # Fetch live game data from Supabase
    live_data_response = supabase.table("nba_live_game_stats").select("*").execute()
    live_data = live_data_response.data
    if not live_data:
        print("No live data to archive.")
        return
    
    # Iterate through each record and archive accordingly
    for record in live_data:
        # Remove the 'id' key so we don't try to insert into a GENERATED ALWAYS column.
        record.pop("id", None)
        
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
            print(f"Archived player stat for game {game_id}, player {record.get('player_id')}: {result}")
        else:
            result = supabase.table("nba_2024_25_game_stats") \
                .upsert(record, on_conflict="game_id") \
                .execute()
            print(f"Archived game stat for game {game_id}: {result}")
    
    # Clear the live game stats table after archival
    delete_response = supabase.table("nba_live_game_stats").delete().neq("id", None).execute()
    print("Cleared live game stats:", delete_response)

if __name__ == "__main__":
    archive_live_data()
