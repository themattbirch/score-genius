from caching.supabase_client import supabase
from datetime import datetime

def cache_game_data(game_id: int, data: dict):
    """
    Inserts or updates cached game data for a given game_id.
    """
    # Check if a record already exists
    existing = supabase.table("game_cache").select("*").eq("game_id", game_id).execute()
    if existing.data:
        # Update the existing record
        response = supabase.table("game_cache").update({
            "data": data,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("game_id", game_id).execute()
    else:
        # Insert a new record
        response = supabase.table("game_cache").insert({
            "game_id": game_id,
            "data": data,
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
    return response

def get_cached_game_data(game_id: int):
    """
    Retrieves cached game data for a given game_id.
    """
    response = supabase.table("game_cache").select("*").eq("game_id", game_id).execute()
    if response.data:
        return response.data[0]
    return None
