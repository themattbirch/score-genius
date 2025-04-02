import requests
import json
from config import API_SPORTS_KEY


API_KEY = API_SPORTS_KEY
BASE_URL = "https://v1.basketball.api-sports.io"
ENDPOINT = "games/statistics/players"
GAME_ID = "414694"

headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v1.basketball.api-sports.io"
}

params = {
    "id": GAME_ID  
    # or "ids": GAME_ID if the docs say "ids" – whichever the endpoint actually expects
}

response = requests.get(f"{BASE_URL}/{ENDPOINT}", headers=headers, params=params)
response.raise_for_status()
data = response.json()

# Pretty-print the entire JSON response so you can see exactly what fields are returned
print(json.dumps(data, indent=2))
