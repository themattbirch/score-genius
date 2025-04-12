# /backend/caching/supabase_cache.py

from caching.supabase_client import supabase
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_cache')

def cache_game_data(game_id: int, game_data: dict):
    """
    Inserts or updates cached game data for a given game_id.
    """
    try:
        # Extract team names and date for logging purposes 
        try:
            home_team = game_data.get('teams', {}).get('home', {}).get('name', 'Unknown')
            away_team = game_data.get('teams', {}).get('away', {}).get('name', 'Unknown')
            game_date = game_data.get('date', 'Unknown date')
        except Exception:
            home_team = away_team = 'Unknown'
            game_date = 'Unknown date'
            
        # Extract scores for the dedicated score columns
        try:
            home_score_raw = game_data.get('scores', {}).get('home', {}).get('total')
            home_score = int(home_score_raw or 0)
        except (TypeError, ValueError, AttributeError):
            home_score = 0
            logger.warning(f"Error extracting home score for game {game_id}")
        
        try:
            away_score_raw = game_data.get('scores', {}).get('away', {}).get('total')
            away_score = int(away_score_raw or 0)
        except (TypeError, ValueError, AttributeError):
            away_score = 0
            logger.warning(f"Error extracting away score for game {game_id}")
        
        logger.info(f"Caching game: {game_id} - {home_team} vs {away_team} ({home_score}-{away_score}) on {game_date}")
        
        existing = supabase.table("game_cache").select("*").eq("game_id", game_id).execute()
        
        data = {
            "game_id": game_id,
            "data": game_data,           
            "updated_at": datetime.utcnow().isoformat(),
            "home_score": home_score,
            "away_score": away_score
        }
        
        if existing.data:
            logger.info(f"Updating existing record for game_id: {game_id}")
            response = supabase.table("game_cache").update(data).eq("game_id", game_id).execute()
        else:
            logger.info(f"Inserting new record for game_id: {game_id}")
            response = supabase.table("game_cache").insert(data).execute()
        
        return response
    
    except Exception as e:
        logger.error(f"Error in cache_game_data: {str(e)}")
        raise

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