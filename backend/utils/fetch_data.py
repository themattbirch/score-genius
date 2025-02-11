# backend/utils/fetch_data.py
import requests
import os

API_KEY = os.getenv("API_SPORTS_KEY", "your_api_key")
BASE_URL = os.getenv("API_BASE_URL", "https://v1.basketball.api-sports.io")

def fetch_live_data():
    """
    Fetch live data from the external sports API.
    """
    url = f"{BASE_URL}/games"
    params = {"league": "12", "season": "2024-2025", "date": "today"}
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "v1.basketball.api-sports.io"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def fetch_historical_data(date: str):
    """
    Fetch historical game data for a given date.
    """
    url = f"{BASE_URL}/games"
    params = {"league": "12", "season": "2019-2020", "date": date}
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "v1.basketball.api-sports.io"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    print(fetch_live_data())
    print(fetch_historical_data("2019-11-23"))
