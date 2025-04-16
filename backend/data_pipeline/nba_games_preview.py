#backend/data_pipeline/nba_games_preview.py

import json
import difflib
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config import API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
from caching.supabase_client import supabase
from supabase import create_client, Client
from typing import Dict, Any


# API-Sports configuration (for game previews)
API_KEY_SPORTS = API_SPORTS_KEY
NBA_URL_SPORTS = 'https://v1.basketball.api-sports.io'
HEADERS_SPORTS = {
    'x-rapidapi-key': API_KEY_SPORTS,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

# The Odds API configuration
NBA_URL_ODDS = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
ODDS_API_KEY = ODDS_API_KEY # Ensure this is correctly assigned from config

# Supabase client initialization (can be done once if needed globally)
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
# If you prefer initializing within functions, keep it as is.


def normalize_team_name(name: str) -> str:
    """Strips extra whitespace and converts the team name to lowercase."""
    return ' '.join(name.split()).lower()

def title_case_team_name(name: str) -> str:
    """Converts a normalized team name back to title case."""
    return ' '.join(word.capitalize() for word in name.split())

def get_betting_odds(et_date: datetime) -> list:
    """
    Fetches betting odds from The Odds API using an Eastern Time date range converted to UTC.
    This ensures games held on the same ET date are all captured even if their UTC times spill over to the next day.
    """
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not configured.")
        return []

    # Define the date range in Eastern Time
    et_zone = ZoneInfo("America/New_York")
    # Ensure et_date is in the ET timezone (or convert it)
    et_date = et_date.astimezone(et_zone)
    start_et = et_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et = et_date.replace(hour=23, minute=59, second=59, microsecond=0)

    # Convert these ET boundaries to UTC strings for the API call
    utc_zone = ZoneInfo("UTC")
    commence_time_from = start_et.astimezone(utc_zone).strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = end_et.astimezone(utc_zone).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": commence_time_from,
        "commenceTimeTo": commence_time_to,
        "apiKey": ODDS_API_KEY
    }

    try:
        response = requests.get(NBA_URL_ODDS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching betting odds: {http_err} - Status Code: {response.status_code}")
        if response.status_code == 422:
            print(f"Odds API response body: {response.text}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Network or Request Error fetching betting odds: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in get_betting_odds: {e}")
        return []

def match_odds_for_game(game: dict, odds_events: list) -> dict | None:
    """
    Matches an odds event to a game based on the normalized team names and local ET date.
    If a direct match isn’t found, it optionally attempts fuzzy matching.
    """
    if not odds_events:
        return None

    teams = game.get('teams', {})
    game_home = normalize_team_name(teams.get('home', {}).get('name', ''))
    game_away = normalize_team_name(teams.get('away', {}).get('name', ''))

    game_date_str = game.get('date', '')
    try:
        game_datetime_utc = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
        game_local_date = game_datetime_utc.astimezone(ZoneInfo("America/New_York")).date()
    except (ValueError, TypeError):
        print(f"Warning: Could not parse game date '{game_date_str}' for matching.")
        return None

    for event in odds_events:
        odds_home = normalize_team_name(event.get('home_team', ''))
        odds_away = normalize_team_name(event.get('away_team', ''))

        # First, try an exact normalized match
        if game_home == odds_home and game_away == odds_away:
            try:
                commence_time_str = event.get('commence_time', '')
                event_datetime_utc = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                event_local_date = event_datetime_utc.astimezone(ZoneInfo("America/New_York")).date()
                if game_local_date == event_local_date:
                    return event
            except (ValueError, TypeError):
                print(f"Warning: Could not parse odds commence_time '{commence_time_str}' for matching.")
                continue

    # Optional: if no exact match is found, try fuzzy matching the team names.
    for event in odds_events:
        odds_home = normalize_team_name(event.get('home_team', ''))
        odds_away = normalize_team_name(event.get('away_team', ''))
        # Use difflib to compare closeness between team names
        home_match = difflib.SequenceMatcher(None, game_home, odds_home).ratio()
        away_match = difflib.SequenceMatcher(None, game_away, odds_away).ratio()
        if home_match > 0.8 and away_match > 0.8:
            try:
                commence_time_str = event.get('commence_time', '')
                event_datetime_utc = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                event_local_date = event_datetime_utc.astimezone(ZoneInfo("America/New_York")).date()
                if game_local_date == event_local_date:
                    return event
            except (ValueError, TypeError):
                print(f"Warning: Could not parse odds commence_time '{commence_time_str}' during fuzzy matching.")
                continue
    return None

def get_games_by_date(league: str, season: str, date: str, timezone: str = 'America/New_York') -> dict:
    """
    Fetches all game data from API-Basketball for the given date, league, season, and timezone.
    """
    url = f"{NBA_URL_SPORTS}/games"
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
        return {} # Return empty dict on error

def extract_odds_by_market(odds_event: dict | None) -> dict: # Use modern type hint
    """
    Extracts odds from a matched odds event, preferring DraftKings.
    Returns a dict containing moneyline, spread, and total odds.
    """
    default_odds = {"moneyline": {}, "spread": {}, "total": {}}
    if not odds_event:
        return default_odds

    bookmakers = odds_event.get('bookmakers', [])
    if not bookmakers:
        return default_odds

    target_bookmaker = next((bm for bm in bookmakers if bm.get('key') == 'draftkings'), bookmakers[0]) # Find DK or take first

    moneyline_odds = {}
    spread_odds = {}
    total_odds = {}

    markets = target_bookmaker.get('markets', [])
    for market in markets:
        market_key = market.get('key')
        outcomes = market.get('outcomes', [])

        # Ensure outcomes is a list before iterating
        if not isinstance(outcomes, list):
            print(f"Warning: Market outcomes is not a list for market key '{market_key}'. Skipping market.")
            continue

        try: # Add error handling for outcome processing
            if market_key == 'h2h':  # Moneyline
                for outcome in outcomes:
                    # Use title case for team name keys for consistency
                    team = title_case_team_name(outcome.get('name', ''))
                    price = outcome.get('price')
                    if team and price is not None:
                         moneyline_odds[team] = price

            elif market_key == 'spreads':  # Spread
                for outcome in outcomes:
                    team = title_case_team_name(outcome.get('name', ''))
                    price = outcome.get('price')
                    point = outcome.get('point')
                    if team and price is not None and point is not None:
                        spread_odds[team] = {'price': price, 'point': point}

            elif market_key == 'totals':  # Over/Under
                for outcome in outcomes:
                    position = outcome.get('name')  # 'Over' or 'Under'
                    price = outcome.get('price')
                    point = outcome.get('point')
                    if position in ['Over', 'Under'] and price is not None and point is not None:
                         total_odds[position] = {'price': price, 'point': point}
        except Exception as e:
            print(f"Error processing outcomes for market key '{market_key}': {e}")
            # Decide whether to continue with other markets or return partially extracted odds

    return {
        "moneyline": moneyline_odds,
        "spread": spread_odds,
        "total": total_odds
    }


def clear_old_games():
    """
    Clears any games from nba_game_schedule whose scheduled date
    (interpreted in America/New_York timezone) is before today's date (ET).
    Uses batch delete for efficiency.
    """
    # Initialize Supabase client within the function or use a global one
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    try:
        # Select only necessary columns
        response = supabase.table("nba_game_schedule").select("game_id, scheduled_time").execute()
    except Exception as e:
        print(f"Error connecting to or querying Supabase: {e}")
        return # Cannot proceed without data

    # Check response structure (adapt based on actual Supabase client library version)
    # Assuming response.data holds the list of rows
    if not hasattr(response, 'data') or response.data is None:
        print(f"Error fetching games or no games found. Response: {response}")
        # Log potential errors from response if available, e.g., response.error
        return

    rows = response.data
    if not rows:
        print("No rows found in nba_game_schedule to consider for clearing.")
        return

    # Use the start of today in America/New_York as the cutoff date
    # This aligns with the timezone used in build_game_preview
    et_zone = ZoneInfo("America/New_York")
    today_et_date = datetime.now(et_zone).date()
    print(f"Current date for clearing check (ET): {today_et_date}")

    ids_to_delete = []
    for row in rows:
        game_id = row.get("game_id")
        scheduled_time_str = row.get("scheduled_time")

        if not scheduled_time_str or not game_id:
            print(f"Skipping row due to missing game_id or scheduled_time: {row}")
            continue

        try:
            # Parse the ISO 8601 timestamp string. Handles offsets like +00:00 or -04:00.
            dt_scheduled = datetime.fromisoformat(scheduled_time_str)

            # Convert the scheduled time to America/New_York timezone to get the game's *local* date
            dt_scheduled_et = dt_scheduled.astimezone(et_zone)

            # Compare the date part only
            if dt_scheduled_et.date() < today_et_date:
                ids_to_delete.append(game_id)
                # Debugging print:
                # print(f"Marking game {game_id} (Scheduled ET Date: {dt_scheduled_et.date()}) for deletion.")
            # else: # Optional: Log games being kept
                # print(f"Keeping game {game_id} (Scheduled ET Date: {dt_scheduled_et.date()}).")


        except ValueError as e:
            # Handle cases where scheduled_time is not a valid ISO 8601 string
            print(f"Error parsing scheduled_time '{scheduled_time_str}' for game_id {game_id}: {e}")
        except Exception as e: # Catch other potential errors (e.g., timezone conversion)
             print(f"Unexpected error processing game_id {game_id} with time '{scheduled_time_str}': {e}")

    if ids_to_delete:
        print(f"Attempting to batch delete {len(ids_to_delete)} old games...")
        try:
            # Perform batch delete using 'in_' filter
            delete_response = supabase.table("nba_game_schedule").delete().in_("game_id", ids_to_delete).execute()

            # Basic check if delete operation potentially succeeded (adapt based on library specifics)
            # Some libraries might have response.error or status codes to check
            print(f"Batch delete attempted for {len(ids_to_delete)} games. Response snippet: {str(delete_response)[:200]}") # Log part of response
            # You might need more specific checks depending on the Supabase client library's return value

        except Exception as e:
            print(f"Error during Supabase batch delete: {e}")
    else:
        print("No old games found to delete based on ET date comparison.")


def build_game_preview() -> list:
    """
    Builds and returns a list of game preview dictionaries for today's date (Eastern Time).
    Filters for games not yet started and matches odds.
    """
    league = '12' # NBA league ID in API-Basketball
    season = '2024-2025' # Adjust season as needed
    et_zone = ZoneInfo("America/New_York")
    eastern_now = datetime.now(et_zone)
    todays_date_str_et = eastern_now.strftime('%Y-%m-%d')

    print(f"Fetching games for date (ET): {todays_date_str_et}")
    games_data = get_games_by_date(league, season, todays_date_str_et, timezone='America/New_York')

    # Check if response structure is as expected
    api_response_games = games_data.get('response', [])
    if not isinstance(api_response_games, list):
         print(f"Warning: Expected a list of games in API response, got {type(api_response_games)}. Response: {games_data}")
         api_response_games = [] # Treat as empty list to avoid errors

    # Filter for games that haven't started based on status
    # Common statuses: "NS" (Not Started), "Scheduled"
    pregame_games = [
        game for game in api_response_games
        if game and isinstance(game.get('status'), dict) and (
            game['status'].get('short') == "NS" or
            "scheduled" in (game['status'].get('long', '') or '').lower() or # Check long status too
            "not started" in (game['status'].get('long', '') or '').lower()
        )
    ]

    if not pregame_games:
        print(f"No pregame games found via API-Basketball for {todays_date_str_et}.")
        return []
    print(f"Found {len(pregame_games)} pregame games.")

    # Fetch odds for today (use the ET date for consistency)
    # get_betting_odds expects a datetime object. We can use eastern_now.
    print("Fetching betting odds...")
    odds_events = get_betting_odds(eastern_now)
    if odds_events:
        print(f"Fetched {len(odds_events)} odds events.")
    else:
        print("No odds events fetched or available.")


    previews = []
    for game in pregame_games:
        # Extract core game details safely
        game_id = game.get("id")
        scheduled_time_str = game.get("date") # ISO8601 string
        venue_data = game.get("venue")
        teams_data = game.get("teams", {})
        away_team_data = teams_data.get("away", {})
        home_team_data = teams_data.get("home", {})

        # Basic validation
        if not all([game_id, scheduled_time_str, away_team_data.get('name'), home_team_data.get('name')]):
             print(f"Skipping game due to missing essential data: {game}")
             continue

        venue_name = venue_data.get("name", "N/A") if isinstance(venue_data, dict) else str(venue_data or "N/A")
        away_team_name = away_team_data.get("name")
        home_team_name = home_team_data.get("name")

        # Compute game_date (YYYY-MM-DD) in ET from the scheduled time
        game_date_et_str = ""
        try:
            game_datetime_utc = datetime.fromisoformat(scheduled_time_str.replace("Z", "+00:00"))
            game_date_et_str = game_datetime_utc.astimezone(et_zone).date().isoformat()
        except (ValueError, TypeError):
            print(f"Warning: Could not parse scheduled_time '{scheduled_time_str}' to generate game_date for game {game_id}.")
            # Decide if you want to proceed without game_date or skip

        # Match odds (uses ET date comparison internally now)
        matched_event = match_odds_for_game(game, odds_events)
        odds_by_market = extract_odds_by_market(matched_event)

        preview = {
            "game_id": game_id,
            "scheduled_time": scheduled_time_str, # Store original ISO string
            "game_date": game_date_et_str, # Store YYYY-MM-DD based on ET
            "venue": venue_name,
            "away_team": away_team_name,
            "home_team": home_team_name,
            # Ensure odds structures are always dicts, even if empty
            "moneyline": odds_by_market.get("moneyline", {}),
            "spread": odds_by_market.get("spread", {}),
            "total": odds_by_market.get("total", {})
        }
        previews.append(preview)

    print(f"Built {len(previews)} game previews.")
    return previews


def upsert_previews_to_supabase(previews: list) -> None:
    """
    Upserts game previews into the Supabase nba_game_schedule table.
    Uses game_id as the conflict target.
    """
    if not previews:
        print("No previews generated, skipping Supabase upsert.")
        return

    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Prepare data for upsert, ensuring required fields are present
    data_to_upsert = []
    for preview in previews:
        game_id = preview.get('game_id')
        if not game_id:
            print(f"Skipping preview with missing game_id: {preview}")
            continue

        # Ensure odds are JSON-serializable (should be dicts)
        # Add default values for robustness if needed, though build_game_preview should handle this
        upsert_data = {
            'game_id': game_id,
            'scheduled_time': preview.get('scheduled_time'),
            'game_date': preview.get('game_date'),
            'venue': preview.get('venue'),
            'away_team': preview.get('away_team'),
            'home_team': preview.get('home_team'),
            'moneyline': preview.get('moneyline', {}), # Default to empty dict
            'spread': preview.get('spread', {}),       # Default to empty dict
            'total': preview.get('total', {}),         # Default to empty dict
            # 'updated_at': datetime.now(ZoneInfo("UTC")).isoformat() # Optionally add/update timestamp
        }
        data_to_upsert.append(upsert_data)

    if not data_to_upsert: # Added check from previous refinement
        print("No valid preview data remaining after filtering, skipping upsert.")
        return

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) # Ensure client is initialized

    try:
        print(f"Upserting {len(data_to_upsert)} previews to Supabase...")
        # ***** ADD on_conflict HERE *****
        # Tell Supabase: If a row with the same 'game_id' exists, update it.
        # Otherwise, insert it.
        upsert_response = supabase.table('nba_game_schedule').upsert(
            data_to_upsert,
            on_conflict='game_id'  # Specify the column causing potential conflicts
        ).execute()

        # Add better response checking if possible (depends on client library version)
        # Check if the response indicates success or error explicitly
        print(f"Supabase upsert attempt finished. Response snippet: {str(upsert_response)[:200]}")
        # e.g., if upsert_response.error: print(f"Supabase error: {upsert_response.error}")

    except Exception as e:
        print(f"Error during Supabase upsert call: {e}")
        # You might want to log the full exception details

def parse_moneyline(moneyline_data: Dict[str, any]) -> str:
    if not moneyline_data:
        return ""
    parts = []
    for team, price in moneyline_data.items():
        price_str = f"+{price}" if price > 0 else str(price)
        parts.append(f"{team} {price_str}")
    return " / ".join(parts)

def parse_spread(spread_data: Dict[str, any]) -> str:
    if not spread_data:
        return ""
    parts = []
    for team, info in spread_data.items():
        if not isinstance(info, dict):
            continue
        point = info.get("point", 0)
        point_str = f"+{point}" if point > 0 else str(point)
        parts.append(f"{team} {point_str}")
    return " / ".join(parts)

def parse_total(total_data: Dict[str, any]) -> str:
    if not total_data:
        return ""
    parts = []
    for side, info in total_data.items():
        if not isinstance(info, dict):
            continue
        point = info.get("point", 0)
        parts.append(f"{side} {point}")
    return " / ".join(parts)

def process_odds_data_in_table() -> None:
    """
    Fetches all rows from nba_game_schedule, processes the raw odds fields into cleaned strings,
    updates each row with the clean odds, and then clears the raw odds columns.
    """
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    response = supabase.table("nba_game_schedule").select("*").execute()
    rows = response.data
    if not rows:
        print("No rows found in nba_game_schedule.")
        return

    for row in rows:
        game_id = row.get("game_id")
        if not game_id:
            continue

        moneyline_data = row.get("moneyline") or {}
        spread_data = row.get("spread") or {}
        total_data = row.get("total") or {}

        # Ensure the raw data is in dict form (convert if stored as JSON-string)
        if isinstance(moneyline_data, str):
            try:
                moneyline_data = json.loads(moneyline_data)
            except Exception as e:
                print(f"Error parsing moneyline data for game_id {game_id}: {e}")
                moneyline_data = {}
        if isinstance(spread_data, str):
            try:
                spread_data = json.loads(spread_data)
            except Exception as e:
                print(f"Error parsing spread data for game_id {game_id}: {e}")
                spread_data = {}
        if isinstance(total_data, str):
            try:
                total_data = json.loads(total_data)
            except Exception as e:
                print(f"Error parsing total data for game_id {game_id}: {e}")
                total_data = {}

        moneyline_clean = parse_moneyline(moneyline_data)
        spread_clean = parse_spread(spread_data)
        total_clean = parse_total(total_data)

        update_data = {
            "moneyline_clean": moneyline_clean,
            "spread_clean": spread_clean,
            "total_clean": total_clean,
            # Clear the raw data once processed
            "moneyline": None,
            "spread": None,
            "total": None
        }

        upsert_res = supabase.table("nba_game_schedule").update(update_data).eq("game_id", game_id).execute()
        print(f"Updated and cleared odds for game_id {game_id}. Response: {upsert_res}")
    print("Finished processing and clearing odds data.")

# Then, in your main function after your upsert call:
def main():
    """
    Main function: Clears old games, builds new previews, upserts them,
    then processes the betting odds in the table.
    """
    print("\n--- Running NBA Game Preview Script ---")

    # 1. Clear old games
    print("\nStep 1: Clearing old games...")
    clear_old_games()

    # 2. Build new previews for today's ET date games
    print("\nStep 2: Building game previews...")
    preview_data = build_game_preview()

    # 3. Upsert the new previews
    if preview_data:
        print("\nStep 3: Upserting previews to Supabase...")
        upsert_previews_to_supabase(preview_data)
        print(f"\n--- Previews upserted. Processed {len(preview_data)} games. ---")
    else:
        print("\nNo game preview data generated to upsert.")
        print("\n--- Script finished. No games processed. ---")
        return 0

    # 4. Process betting odds data using the integrated logic
    print("\nStep 4: Processing betting odds data in table...")
    process_odds_data_in_table()
    return len(preview_data)

if __name__ == "__main__":
    main()
    print("\nDirect execution finished.")
