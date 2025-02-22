# backend/src/scripts/archive_live_data.py

import os
from datetime import datetime
from caching.supabase_client import supabase

def archive_live_data():
    """
    Archives live game data from nba_live_game_stats table to historical tables,
    then clears the nba_live_game_stats table.
    
    Process:
      1. Fetch all records from the live game stats table.
      2. For each record, if it contains a valid 'player_id', upsert it into nba_historical_player_stats.
         Otherwise, upsert it into nba_2024_25_game_stats.
      3. After successful archival, clear all records from nba_live_game_stats.
    """
    # Fetch live game data from Supabase
    live_data_response = supabase.table("nba_live_game_stats").select("*").execute()
    live_data = live_data_response.data
    if not live_data:
        print("No live data to archive.")
        return
    
    # Iterate through each record and archive accordingly
    for record in live_data:
        game_id = record.get("game_id")
        # Check if the record has a valid player_id (indicating a player stat)
        if record.get("player_id") is not None:
            # Archive to nba_historical_player_stats
            result = supabase.table("nba_historical_player_stats") \
                .upsert(record, on_conflict="game_id, player_id") \
                .execute()
            print(f"Archived player stat for game {game_id}, player {record.get('player_id')}: {result}")
        else:
            # Otherwise, archive to nba_2024_25_game_stats as a game-level stat
            result = supabase.table("nba_2024_25_game_stats") \
                .upsert(record, on_conflict="game_id") \
                .execute()
            print(f"Archived game stat for game {game_id}: {result}")
    
    # Clear the live game stats table after archival
    delete_response = supabase.table("nba_live_game_stats").delete().neq("id", None).execute()
    print("Cleared live game stats:", delete_response)

if __name__ == "__main__":
    archive_live_data()
