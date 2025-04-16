# /backend/caching/supabase_cache.py

from caching.supabase_client import supabase
from datetime import datetime
import logging
from dateutil import parser as dateutil_parser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_cache')

def cache_game_data(game_id: int, game_data: dict):
    """
    Upserts cached game data for a given game_id using Supabase upsert.
    Assumes 'game_cache' table has columns: game_id (PK), data (JSONB),
    updated_at (TIMESTAMPTZ), home_score (INT), away_score (INT).
    """
    try:
        # Extract info for logging (keep your existing try/except block for this)
        try:
            home_team = game_data.get('teams', {}).get('home', {}).get('name', 'Unknown')
            away_team = game_data.get('teams', {}).get('away', {}).get('name', 'Unknown')
            game_date_str = game_data.get('date', '')
            # Use dateutil parser safely
            game_date_log = dateutil_parser.parse(game_date_str).strftime('%Y-%m-%d') if game_date_str else 'Unknown Date'
        except Exception:
            home_team = away_team = 'Unknown'; game_date_log = 'Unknown Date'

        # Extract scores safely, defaulting to None
        home_score = None; away_score = None
        try:
            home_score_raw = game_data.get('scores', {}).get('home', {}).get('total')
            if home_score_raw is not None: home_score = int(home_score_raw)
        except (TypeError, ValueError, AttributeError): logger.warning(f"Could not parse home score for game {game_id}")
        try:
            away_score_raw = game_data.get('scores', {}).get('away', {}).get('total')
            if away_score_raw is not None: away_score = int(away_score_raw)
        except (TypeError, ValueError, AttributeError): logger.warning(f"Could not parse away score for game {game_id}")

        logger.info(f"Upserting cache for game: {game_id} - {away_team} @ {home_team} ({away_score} - {home_score}) on {game_date_log}")

        # Prepare data payload for upsert
        # Ensure keys match your exact Supabase column names
        upsert_data = {
            "game_id": game_id,
            "data": game_data, # Store the full raw JSON from the source API
            "updated_at": datetime.utcnow().isoformat(), # Use UTC now() for timestamp
            "home_score": home_score, # Store extracted score (INT or None)
            "away_score": away_score  # Store extracted score (INT or None)
        }

        # Perform a single UPSERT operation
        # on_conflict specifies the primary key or unique constraint column(s)
        response = supabase.table("game_cache").upsert(
            upsert_data,
            on_conflict="game_id" # Assumes 'game_id' is your primary key or unique constraint
        ).execute()

        # Optional: Check response for errors/status
        if hasattr(response, 'error') and response.error:
             logger.error(f"Supabase upsert error for game {game_id}: {response.error}")
        # elif not hasattr(response, 'data') or not response.data:
             # logger.warning(f"Supabase upsert for game {game_id} returned no data (might be expected if no change).")

        return response # Return the Supabase response object

    except Exception as e:
        logger.error(f"Unexpected error in cache_game_data for game {game_id}: {str(e)}", exc_info=True)
        return None  

def get_cached_game_data(game_id: int):
    """
    Retrieves cached game data for a given game_id.
    """
    try:
        logger.info(f"Retrieving cached data for game_id: {game_id}")
        response = supabase.table("game_cache").select("*").eq("game_id", game_id).execute()
        if response.data:
            logger.info(f"Found cached data for game_id: {game_id}")
            return response.data[0]
        logger.info(f"No cached data found for game_id: {game_id}")
        return None
    except Exception as e:
        logger.error(f"Error in get_cached_game_data: {str(e)}")
        return None