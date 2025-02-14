# backend/src/scripts/nba_games_preview.py

import json
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from config import API_SPORTS_KEY, ODDS_API_KEY

# API-Sports configuration (for game previews)
API_KEY_SPORTS = API_SPORTS_KEY
BASE_URL_SPORTS = 'https://v1.basketball.api-sports.io'
HEADERS_SPORTS = {
    'x-rapidapi-key': API_KEY_SPORTS,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

# The Odds API configuration
BASE_URL_ODDS = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
ODDS_PARAMS = {
    'apiKey': ODDS_API_KEY,
    'regions': 'us',
    'markets': 'h2h,spreads,totals',
    'oddsFormat': 'american',
    'bookmakers': 'draftkings'
}

def normalize_team_name(name: str) -> str:
    """Normalizes a team name by stripping extra whitespace and converting to lower-case."""
    return ' '.join(name.split()).lower()

def title_case_team_name(name: str) -> str:
    """Converts a normalized team name back to title case.
    Example: "dallas mavericks" -> "Dallas Mavericks"
    """
    return ' '.join(word.capitalize() for word in name.split())

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/New_York') -> dict:
    """
    Fetches all game data from API-Basketball for the given date, league, season, and timezone.
    """
    url = f"{BASE_URL_SPORTS}/games"
    params = {
        'league': league,
        'season': season,
        'date': date,
        'timezone': timezone
    }
    try:
        response = requests.get(url, headers=HEADERS_SPORTS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game data for {date}: {e}")
        return {}

def get_betting_odds() -> list:
    """
    Fetches betting odds for NBA events from The Odds API.
    Returns a list of odds event objects.
    """
    try:
        response = requests.get(BASE_URL_ODDS, params=ODDS_PARAMS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching betting odds: {e}")
        return []

def match_odds_for_game(game: dict, odds_events: list) -> dict:
    """
    Attempts to match an odds event from The Odds API to a game from API-Basketball.
    Matching is done by normalizing the team names and verifying the event date matches.
    Returns the matching odds event, or None if not found.
    """
    teams = game.get('teams', {})
    game_home = normalize_team_name(teams.get('home', {}).get('name', ''))
    game_away = normalize_team_name(teams.get('away', {}).get('name', ''))
    
    game_date_str = game.get('date', '')
    try:
        game_datetime = datetime.fromisoformat(game_date_str)
        game_local_date = game_datetime.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        game_local_date = None

    for event in odds_events:
        odds_home = normalize_team_name(event.get('home_team', ''))
        odds_away = normalize_team_name(event.get('away_team', ''))
        if game_home == odds_home and game_away == odds_away:
            commence_time_str = event.get('commence_time', '')
            try:
                event_datetime = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                event_local_date = event_datetime.astimezone(ZoneInfo("America/New_York")).date()
                if game_local_date == event_local_date:
                    return event
            except Exception:
                continue
    return None

def extract_simple_odds(odds_event: dict) -> list:
    """
    Extracts a simplified list of odds from an odds event.
    For each outcome, returns only "name", "price", and "point" (if available).
    """
    simple_odds = []
    bookmakers = odds_event.get('bookmakers', [])
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                simple_odds.append({
                    'name': outcome.get('name'),
                    'price': outcome.get('price'),
                    'point': outcome.get('point')
                })
    return simple_odds

def build_game_preview() -> list:
    """
    Builds and returns a list of game preview dictionaries for today's date (Eastern Time).
    Each preview includes game details from API-Basketball and, if available, simplified betting odds from The Odds API.
    """
    league = '12'
    season = '2024-2025'
    eastern_now = datetime.now(ZoneInfo("America/New_York"))
    todays_date_str = eastern_now.strftime('%Y-%m-%d')
    
    games_data = get_games_by_date(league, season, todays_date_str, timezone='America/New_York')
    pregame_games = [
        game for game in games_data.get('response', [])
        if game.get('status', {}).get('short') == "NS" or 
           ("Not Started" in (game.get('status', {}).get('long') or ""))
    ]
    if not pregame_games:
        print("No pregame games found for the specified date.")
        return []
    
    odds_events = get_betting_odds()
    previews = []
    for game in pregame_games:
        matched_event = match_odds_for_game(game, odds_events)
        odds = extract_simple_odds(matched_event) if matched_event else []
        preview = {
            "game_id": game.get("id"),
            "scheduled_time": game.get("date"),
            "venue": game.get("venue"),
            "away_team": game.get("teams", {}).get("away", {}).get("name"),
            "home_team": game.get("teams", {}).get("home", {}).get("name"),
            "betting_odds": odds
        }
        previews.append(preview)
    return previews

if __name__ == "__main__":
    preview_data = build_game_preview()
    from pprint import pprint
    pprint(preview_data)
