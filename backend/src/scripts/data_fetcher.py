import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = 'd0c358b61e883d071bbc183c8fd72228'

# Use the correct host format and headers based on your subscription type
# Choose the appropriate one based on your subscription
BASE_URL = 'https://v1.basketball.api-sports.io'  # API-SPORTS
HEADERS = {
    'x-apisports-key': API_KEY  # Correct header for API-SPORTS
}

# Alternate configuration for RapidAPI users
# BASE_URL = 'https://api-basketball.p.rapidapi.com'
# HEADERS = {
#     'x-rapidapi-key': API_KEY,
#     'x-rapidapi-host': 'api-basketball.p.rapidapi.com'
# }

def fetch_live_game_data(league, season, date, timezone='America/Los_Angeles'):
    """Fetch live game data from API-Basketball with retry logic."""
    url = f"{BASE_URL}/games"
    params = {
        'league': league,
        'season': season,
        'date': date,
        'timezone': timezone
    }
    
    # Create a session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        response = session.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        # Return empty response instead of raising exception
        return {"response": []}