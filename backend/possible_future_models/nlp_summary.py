# backend/models/nlp_summary.py


import os
import requests
import time

# Simple in-memory cache
_summary_cache = {}

# Retrieve the Cloudflare Workers AI endpoint from environment variables.
CLOUDFLARE_WORKER_AI_URL = os.getenv("CLOUDFLARE_WORKER_AI_URL")
if not CLOUDFLARE_WORKER_AI_URL:
    raise Exception("CLOUDFLARE_WORKER_AI_URL environment variable is not set")

def _generate_summary_from_api(model_outputs, game_state):
    """
    A placeholder function demonstrating how to call the Cloudflare Worker AI endpoint.
    Replace or extend this function with actual logic to interact with your Cloudflare Worker.
    """
    payload = {
        "model_outputs": model_outputs,
        "game_state": game_state
    }
    response = requests.post(CLOUDFLARE_WORKER_AI_URL, json=payload)
    if response.status_code == 200:
        return response.text
    return "Unable to generate summary from Cloudflare Worker AI."

def generate_game_summary(model_outputs: dict, game_id: str, game_state: str = "post_game", cache_ttl: int = 60) -> str:
    """
    Generates a narrative game summary with caching support.
    
    Parameters:
        model_outputs (dict): Outputs from various models.
        game_id (str): Unique identifier for the game.
        game_state (str): The game state ("pre_game", "in_progress", "post_game").
        cache_ttl (int): Cache time-to-live in seconds.
    
    Returns:
        str: The generated narrative summary.
    """
    # Base cache key starts with game_id.
    base_key = f"{game_id}_"
    
    # For in-progress games, include a timestamp parameter (e.g., current game time)
    if game_state == "in_progress":
        # Retrieve game_time from model_outputs; if not present, fallback to current time.
        game_time = model_outputs.get("game_time", time.time())
        base_key += f"{game_time}_"
    
    # Append a hash of the model outputs for uniqueness.
    cache_key = f"{base_key}{hash(frozenset(model_outputs.items()))}"
    
    # Check the cache first.
    if cache_key in _summary_cache:
        timestamp, cached_summary = _summary_cache[cache_key]
        if time.time() - timestamp < cache_ttl:
            return cached_summary

    # Generate a new summary via the API call.
    summary = _generate_summary_from_api(model_outputs, game_state)
    
    # Cache the result.
    _summary_cache[cache_key] = (time.time(), summary)
    
    return summary

if __name__ == "__main__":
    # Example usage:
    example_model_outputs = {
        "win_probability": 0.82,
        "momentum_shift": 0.25,
        "projected_margin": 8,
        "player_projection": "Player X is expected to score 28 points and grab 7 rebounds."
    }
    summary = generate_game_summary(example_model_outputs, "game123", "post_game", 60)
    print("Generated Game Summary:\n", summary)
