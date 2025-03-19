# backend/src/scripts/nba_games_preview.py

import json
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config import API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY

# API-Sports configuration (for game previews)
API_KEY_SPORTS = API_SPORTS_KEY
BASE_URL_SPORTS = 'https://v1.basketball.api-sports.io'
HEADERS_SPORTS = {
    'x-rapidapi-key': API_KEY_SPORTS,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

# The Odds API configuration
BASE_URL_ODDS = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'


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

def get_betting_odds(date: datetime) -> list:
    """
    Fetches betting odds for NBA events from The Odds API for a specific date.
    Uses a time range covering the entire day in UTC without microseconds,
    formatted as "YYYY-MM-DDTHH:MM:SSZ".
    Returns a list of odds event objects.
    """
    # Format the times correctly with a trailing "Z"
    commence_time_from = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = (date.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=ZoneInfo("UTC")) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": commence_time_from,
        "commenceTimeTo": commence_time_to,
        "apiKey": ODDS_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL_ODDS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 422:
            # Log the error and return an empty list if no odds data is available for this date.
            print(f"422 Unprocessable Entity: {response.text}")
            return []
        else:
            print(f"Error fetching betting odds: {http_err}")
            return []
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

def extract_odds_by_market(odds_event: dict) -> dict:
    """
    Extracts odds from an odds event, categorized by market type (moneyline, spread, total).
    Focuses on DraftKings bookmaker if available, otherwise uses the first bookmaker.
    """
    if not odds_event:
        return {"moneyline": {}, "spread": {}, "total": {}}
    
    # Initialize empty structures for odds
    moneyline_odds = {}
    spread_odds = {}
    total_odds = {}
    
    # Get bookmakers from the event
    bookmakers = odds_event.get('bookmakers', [])
    if not bookmakers:
        return {"moneyline": {}, "spread": {}, "total": {}}
    
    # Try to find DraftKings first, otherwise use the first bookmaker
    target_bookmaker = None
    for bm in bookmakers:
        if bm.get('key') == 'draftkings':
            target_bookmaker = bm
            break
    
    if not target_bookmaker and bookmakers:
        target_bookmaker = bookmakers[0]
    
    if not target_bookmaker:
        return {"moneyline": {}, "spread": {}, "total": {}}
    
    # Extract odds from the target bookmaker's markets
    markets = target_bookmaker.get('markets', [])
    for market in markets:
        market_key = market.get('key')
        outcomes = market.get('outcomes', [])
        
        if market_key == 'h2h':  # Moneyline
            for outcome in outcomes:
                team = outcome.get('name')
                price = outcome.get('price')
                moneyline_odds[team] = price
        
        elif market_key == 'spreads':  # Spread
            for outcome in outcomes:
                team = outcome.get('name')
                price = outcome.get('price')
                point = outcome.get('point')
                spread_odds[team] = {'price': price, 'point': point}
        
        elif market_key == 'totals':  # Over/Under
            for outcome in outcomes:
                position = outcome.get('name')  # 'Over' or 'Under'
                price = outcome.get('price')
                point = outcome.get('point')
                total_odds[position] = {'price': price, 'point': point}
    
    return {
        "moneyline": moneyline_odds,
        "spread": spread_odds,
        "total": total_odds
    }

def clear_old_games():
    """
    Clears any games from nba_game_schedule that are scheduled before today's date in PT time.
    Completed or past games are removed.
    """
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    # We'll interpret scheduled_time as ISO8601 and compare only the date portion
    # in America/Los_Angeles (PT).
    response = supabase.table("nba_game_schedule").select("*").execute()
    rows = response.data
    if not rows:
        print("No rows to consider for clearing old games.")
        return

    now_pt = datetime.now(ZoneInfo("America/Los_Angeles")).date()

    for row in rows:
        game_id = row.get("game_id")
        scheduled_time = row.get("scheduled_time")
        if not scheduled_time or not game_id:
            continue
        try:
            # Parse as ISO 8601
            dt = datetime.fromisoformat(scheduled_time)
            dt_local = dt.astimezone(ZoneInfo("America/Los_Angeles"))
            if dt_local.date() < now_pt:
                supabase.table("nba_game_schedule").delete().eq("game_id", game_id).execute()
                print(f"Cleared old game {game_id} scheduled for {scheduled_time}.")
        except Exception as e:
            print(f"Error parsing scheduled_time for game_id {game_id}: {e}")

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
    
    odds_events = get_betting_odds(eastern_now)
    previews = []
    for game in pregame_games:
        # Extract venue information
        venue_data = game.get("venue", "")
        if isinstance(venue_data, dict):
            venue_name = venue_data.get("name", "")
        else:
            venue_name = venue_data  # It's already a string
        
        # Compute game_date from the scheduled time
        scheduled_time = game.get("date")
        game_date = ""
        if scheduled_time:
            try:
                game_date = datetime.fromisoformat(scheduled_time).date().isoformat()
            except Exception:
                game_date = ""
        
        matched_event = match_odds_for_game(game, odds_events)
        odds_by_market = extract_odds_by_market(matched_event)
        
        preview = {
            "game_id": game.get("id"),
            "scheduled_time": scheduled_time,
            "game_date": game_date,
            "venue": venue_name,
            "away_team": game.get("teams", {}).get("away", {}).get("name"),
            "home_team": game.get("teams", {}).get("home", {}).get("name"),
            "moneyline": odds_by_market["moneyline"],
            "spread": odds_by_market["spread"],
            "total": odds_by_market["total"]
        }
        previews.append(preview)
    
    return previews

def upsert_previews_to_supabase(previews: list) -> None:
    """
    Upserts game previews into the Supabase nba_game_schedule table.
    Uses game_id as the unique identifier for upsert operations.
    """
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    for preview in previews:
        game_id = preview.get('game_id')
        if not game_id:
            print("Skipping preview with no game_id")
            continue
        
        upsert_data = {
            'game_id': game_id,
            'scheduled_time': preview.get('scheduled_time'),
            'game_date': preview.get('game_date'),
            'venue': preview.get('venue'),
            'away_team': preview.get('away_team'),
            'home_team': preview.get('home_team'),
            'moneyline': preview.get('moneyline'),
            'spread': preview.get('spread'),
            'total': preview.get('total')
        }
        
        try:
            # Use on_conflict="game_id" to update existing rows
            supabase.table('nba_game_schedule').upsert([upsert_data], on_conflict="game_id").execute()
            print(f"Successfully upserted game_id {game_id}")
        except Exception as e:
            print(f"Error upserting game_id {game_id}: {e}")

def main():
    """
    Main function that orchestrates clearing old games and updating preview data.
    Used for scheduled execution from scheduler_setup.py.
    """
    # First, clear old (past) games from the table
    clear_old_games()
    
    # Then build new previews for today's games
    preview_data = build_game_preview()
    if preview_data:
        upsert_previews_to_supabase(preview_data)
        print(f"Upserted {len(preview_data)} game previews to Supabase.")
        return len(preview_data)
    else:
        print("No game preview data to upsert.")
        return 0

if __name__ == "__main__":
    # First, clear old (past) games from the table
    clear_old_games()
    
    # Then build new previews for today's games
    preview_data = build_game_preview()
    if preview_data:
        upsert_previews_to_supabase(preview_data)
        print(f"Upserted {len(preview_data)} game previews to Supabase.")
        
        # Print sample data for verification
        if len(preview_data) > 0:
            print("\nSample preview data (first game):")
            import pprint
            pprint.pprint(preview_data[0])
    else:
        print("No game preview data to upsert.")
