# /backend/caching/supabase_stats.py

from supabase_client import supabase
from datetime import datetime

def upsert_game_stats(game_id: int, player_stats: dict):
    """
    Inserts or updates player statistics for a given game.
    
    Args:
        game_id: The ID of the game.
        player_stats: Dictionary containing player statistics from the API.
    """
    try:
        # Transform API data into our database structure.
        # Adjust the key names as needed depending on your API response structure.
        stats_data = {
            'game_id': game_id,
            'player_id': player_stats['player']['id'],
            'player_name': player_stats['player']['name'],
            'team_id': player_stats.get('team', {}).get('id'),
            'team_name': player_stats.get('team', {}).get('name'),
            # Here we assume that 'statistics' is a nested dict; adjust as needed.
            'minutes': player_stats.get('statistics', {}).get('minutes', 0),
            'points': player_stats.get('statistics', {}).get('points', 0),
            'rebounds': player_stats.get('statistics', {}).get('rebounds', 0),
            'assists': player_stats.get('statistics', {}).get('assists', 0),
            'steals': player_stats.get('statistics', {}).get('steals', 0),
            'blocks': player_stats.get('statistics', {}).get('blocks', 0),
            'turnovers': player_stats.get('statistics', {}).get('turnovers', 0),
            'fouls': player_stats.get('statistics', {}).get('fouls', 0),
            'fg_made': player_stats.get('statistics', {}).get('fgm', 0),
            'fg_attempted': player_stats.get('statistics', {}).get('fga', 0),
            'three_made': player_stats.get('statistics', {}).get('tpm', 0),
            'three_attempted': player_stats.get('statistics', {}).get('tpa', 0),
            'ft_made': player_stats.get('statistics', {}).get('ftm', 0),
            'ft_attempted': player_stats.get('statistics', {}).get('fta', 0),
            'game_date': datetime.now().date(),  # or use a value from the API if available
            'updated_at': datetime.utcnow()
        }
        
        result = supabase.table('nba_game_stats') \
            .upsert(stats_data, on_conflict='game_id,player_id') \
            .execute()
        return result
    except Exception as e:
        print(f"Error upserting game stats for player {player_stats['player']['name']}: {e}")
        return None

def get_game_stats(game_id: int):
    """
    Retrieves all player statistics for a given game.
    """
    try:
        result = supabase.table('nba_game_stats') \
            .select('*') \
            .eq('game_id', game_id) \
            .execute()
        return result.data
    except Exception as e:
        print(f"Error retrieving game stats for game {game_id}: {e}")
        return None

def get_player_stats(player_id: int):
    """
    Retrieves all statistics for a given player across games.
    """
    try:
        result = supabase.table('nba_game_stats') \
            .select('*') \
            .eq('player_id', player_id) \
            .order('game_date', desc=True) \
            .execute()
        return result.data
    except Exception as e:
        print(f"Error retrieving player stats for player {player_id}: {e}")
        return None
