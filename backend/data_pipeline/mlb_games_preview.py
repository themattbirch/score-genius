#!/usr/bin/env python3
"""
backend/data_pipeline/mlb_games_preview.py

This script builds MLB game previews by fetching schedule info via api‑baseball and betting odds via The Odds API,
then upserts the preview data into a Supabase table. Pitcher info is updated separately by scraping FanGraphs.
"""

import time
import re
import json
import datetime
from datetime import date, timedelta, datetime as dt_datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List, Any
from dateutil import parser as dateutil_parser

# --- 3rd-party Imports ---
try:
    import requests
    import undetected_chromedriver as uc
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    from supabase import create_client, Client
except ImportError as e:
    print(f"FATAL ERROR: Missing required third‑party module: {e}")
    exit(1)

# --- Local Config & Variables ---
try:
    from config import (
        API_SPORTS_KEY,      # for api‑baseball schedule
        ODDS_API_KEY,        # for odds data from The Odds API
        SUPABASE_URL,
        SUPABASE_ANON_KEY
    )
    print("Successfully imported configuration variables from config.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import variables from config.py: {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during config import: {e}")
    exit(1)

# --- Configuration Constants ---
API_BASEBALL_URL = "https://v1.baseball.api-sports.io"
MLB_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
SUPABASE_TABLE_NAME = "mlb_game_schedule"
ET_ZONE = ZoneInfo("America/New_York")
PREFERRED_BOOKMAKER_KEY = 'draftkings'
REQUEST_DELAY_SECONDS = 1  # Delay between api‑baseball calls
MLB_LEAGUE_ID = 1

# For Selenium/undetected‑chromedriver (FanGraphs scraping)
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"
SCROLL_DELAY_SECONDS = 3   # Delay after scrolling to bottom
LOAD_WAIT_SECONDS = 10     # Max time to wait for page elements
CHROME_HEADLESS = True     # Set to False to see the browser

# --- Helper Functions ---

def normalize_team_name(name: str) -> str:
    """
    Normalizes team names so that they match Supabase values.
    """
    if not name or not isinstance(name, str):
        return ""
    original = name.strip()
    temp = original.replace("St.Louis", "St Louis").replace("St.", "St").lower().strip()
    if temp in ["ath", "athletics", "oakland athletics"]:
        return "Athletics"
    if "st louis" in temp:
        return "St. Louis Cardinals"
    mapping = {
        "bal": "Baltimore Orioles",
        "bos": "Boston Red Sox",
        "nyy": "New York Yankees",
        "tbr": "Tampa Bay Rays",
        "tor": "Toronto Blue Jays",
        "chw": "Chicago White Sox",
        "cle": "Cleveland Guardians",
        "det": "Detroit Tigers",
        "kcr": "Kansas City Royals",
        "min": "Minnesota Twins",
        "hou": "Houston Astros",
        "laa": "Los Angeles Angels",
        "sea": "Seattle Mariners",
        "tex": "Texas Rangers",
        "atl": "Atlanta Braves",
        "mia": "Miami Marlins",
        "nym": "New York Mets",
        "phi": "Philadelphia Phillies",
        "wsn": "Washington Nationals",
        "chc": "Chicago Cubs",
        "cin": "Cincinnati Reds",
        "mil": "Milwaukee Brewers",
        "pit": "Pittsburgh Pirates",
        "ari": "Arizona Diamondbacks",
        "col": "Colorado Rockies",
        "lad": "Los Angeles Dodgers",
        "sdp": "San Diego Padres",
        "sfg": "San Francisco Giants",
    }
    if temp in mapping:
        return mapping[temp]
    return temp.title()

def title_case_team_name(name: str) -> str:
    """Converts a normalized team name back to title case for display."""
    return ' '.join(word.capitalize() for word in name.split())

def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely converts a value to float if possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_american_odds(numeric_odds: Optional[float]) -> Optional[str]:
    """Formats numeric odds into American odds string (e.g., +150, -110)."""
    if numeric_odds is None:
        return None
    odds_int = int(round(numeric_odds))
    return f"+{odds_int}" if odds_int > 0 else f"{odds_int}"

# --- Fetch MLB Scheduled Games from api-baseball ---

def make_api_request(url: str, headers: dict, params: dict):
    """Generic API request helper."""
    print(f"Querying {url.split('?')[0]} with params {params}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        print("Request successful.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error: {http_err} - Status: {http_err.response.status_code}")
        print(f"Body: {http_err.response.text[:500]}...")
    except Exception as e:
        print(f"Error in make_api_request: {e}")
    return None

def get_games_from_apibaseball(target_date: date) -> List[Dict[str, Any]]:
    """Fetches MLB schedule from api-baseball for a specific date."""
    if not API_SPORTS_KEY:
        print("Error: API_SPORTS_KEY not configured.")
        return []
    headers = {"x-apisports-key": API_SPORTS_KEY}
    url = f"{API_BASEBALL_URL}/games"
    params = {
        "league": MLB_LEAGUE_ID,
        "date": target_date.strftime("%Y-%m-%d"),
        "season": target_date.year
    }
    data = make_api_request(url, headers, params)
    if data:
        if isinstance(data.get("response"), list):
            print(f"Found {len(data['response'])} games from api-baseball for {target_date}.")
            return data["response"]
        else:
            print(f"Unexpected structure from api-baseball for {target_date}: {data}")
    else:
        print(f"No data returned from api-baseball for {target_date}.")
    return []

# --- Betting Odds API Functions ---

def get_betting_odds(target_date_dt: dt_datetime) -> List[Dict[str, Any]]:
    """Fetches betting odds for MLB events from The Odds API covering today/tomorrow ET."""
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not available.")
        return []
    utc_zone = ZoneInfo("UTC")
    start_utc = dt_datetime.combine(target_date_dt.date(), dt_time.min).replace(tzinfo=utc_zone)
    end_utc = dt_datetime.combine(target_date_dt.date() + timedelta(days=1), dt_time.max).replace(tzinfo=utc_zone)
    commence_time_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    commence_time_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": commence_time_from,
        "commenceTimeTo": commence_time_to
    }
    print(f"Fetching odds from {MLB_ODDS_URL} between {commence_time_from} and {commence_time_to}")
    try:
        response = requests.get(MLB_ODDS_URL, params=params)
        response.raise_for_status()
        odds_data = response.json()
        print(f"Fetched {len(odds_data)} odds events.")
        return odds_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching odds: {http_err} - Status: {response.status_code}")
        if response.status_code == 422:
            print(f"Odds API body: {response.text}")
    except Exception as e:
        print(f"Error in get_betting_odds: {e}")
    return []

def match_odds_for_apibaseball_game(apibaseball_game: Dict[str, Any], odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Matches an odds event from The Odds API to an api‑baseball game event by team names and date (ET).
    """
    if not odds_events or not apibaseball_game:
        return None
    teams_info = apibaseball_game.get("teams", {})
    home_team_info = teams_info.get("home", {})
    away_team_info = teams_info.get("away", {})
    game_home_name = normalize_team_name(home_team_info.get("name", ""))
    game_away_name = normalize_team_name(away_team_info.get("name", ""))
    game_date_str_utc = apibaseball_game.get("date")
    try:
        game_dt_utc = dateutil_parser.isoparse(game_date_str_utc)
        game_local_date_et = game_dt_utc.astimezone(ET_ZONE).date()
    except Exception as e:
        print(f"Warn: Could not parse date {game_date_str_utc}: {e}")
        return None
    if not game_home_name or not game_away_name:
        return None
    for odds_event in odds_events:
        odds_home = normalize_team_name(odds_event.get("home_team", ""))
        odds_away = normalize_team_name(odds_event.get("away_team", ""))
        if game_home_name == odds_home and game_away_name == odds_away:
            commence_time_str = odds_event.get("commence_time", "")
            try:
                event_dt_utc = dateutil_parser.isoparse(commence_time_str)
                event_local_date_et = event_dt_utc.astimezone(ET_ZONE).date()
                if game_local_date_et == event_local_date_et:
                    print(f"Matched Odds: {title_case_team_name(game_away_name)} @ {title_case_team_name(game_home_name)} on {game_local_date_et}")
                    return odds_event
            except Exception:
                continue
    return None

def extract_odds_data(odds_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts raw odds and clean formatted odds strings from an odds event.
    """
    raw_odds = {"moneyline": {}, "spread": {}, "total": {}}
    clean_odds = {
        "moneyline_home_clean": None,
        "moneyline_away_clean": None,
        "spread_home_line_clean": None,
        "spread_home_price_clean": None,
        "spread_away_price_clean": None,
        "total_line_clean": None,
        "total_over_price_clean": None,
        "total_under_price_clean": None
    }
    if not odds_event:
        return {"raw": raw_odds, "clean": clean_odds}
    bookmakers = odds_event.get("bookmakers", [])
    if not bookmakers:
        return {"raw": raw_odds, "clean": clean_odds}
    target_bookmaker = next((bm for bm in bookmakers if bm.get("key") == PREFERRED_BOOKMAKER_KEY), bookmakers[0])
    if not target_bookmaker:
        return {"raw": raw_odds, "clean": clean_odds}
    raw_odds["bookmaker_key"] = target_bookmaker.get("key")
    odds_home_team = title_case_team_name(odds_event.get("home_team", ""))
    odds_away_team = title_case_team_name(odds_event.get("away_team", ""))
    markets = target_bookmaker.get("markets", [])
    for market in markets:
        mkey = market.get("key")
        outcomes = market.get("outcomes", [])
        if not isinstance(outcomes, list):
            continue
        if mkey == "h2h":
            raw_odds["moneyline"] = {
                title_case_team_name(o.get("name", "")): o.get("price")
                for o in outcomes if o.get("name") and o.get("price") is not None
            }
            for o in outcomes:
                team = title_case_team_name(o.get("name", ""))
                price = safe_float_conversion(o.get("price"))
                fmt_price = format_american_odds(price)
                if team == odds_home_team:
                    clean_odds["moneyline_home_clean"] = fmt_price
                elif team == odds_away_team:
                    clean_odds["moneyline_away_clean"] = fmt_price
        elif mkey == "spreads":
            raw_odds["spread"] = {
                title_case_team_name(o.get("name", "")): {
                    "price": o.get("price"),
                    "point": o.get("point")
                }
                for o in outcomes if o.get("name") and o.get("price") is not None and o.get("point") is not None
            }
            for o in outcomes:
                team = title_case_team_name(o.get("name", ""))
                price = safe_float_conversion(o.get("price"))
                point = safe_float_conversion(o.get("point"))
                fmt_price = format_american_odds(price)
                if team == odds_home_team:
                    clean_odds["spread_home_line_clean"] = point
                    clean_odds["spread_home_price_clean"] = fmt_price
                elif team == odds_away_team:
                    clean_odds["spread_away_price_clean"] = fmt_price
        elif mkey == "totals":
            raw_odds["total"] = {
                o.get("name", ""): {
                    "price": o.get("price"),
                    "point": o.get("point")
                }
                for o in outcomes if o.get("name") in ["Over", "Under"] and o.get("price") is not None and o.get("point") is not None
            }
            num_total_line = None
            for o in outcomes:
                num_total_line = safe_float_conversion(o.get("point"))
                break
            clean_odds["total_line_clean"] = num_total_line
            for o in outcomes:
                pos = o.get("name")
                price = safe_float_conversion(o.get("price"))
                fmt_price = format_american_odds(price)
                if pos == "Over":
                    clean_odds["total_over_price_clean"] = fmt_price
                elif pos == "Under":
                    clean_odds["total_under_price_clean"] = fmt_price
    return {"raw": raw_odds, "clean": clean_odds}

# --- FanGraphs Scraping (Pitcher Data) ---
def scrape_fangraphs_probables() -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Uses Selenium (via undetected‑chromedriver) to load the FanGraphs Probables Grid,
    scroll to force lazy‑loading, and extract header dates and pitcher data.
    Returns a dictionary mapping (game_date_iso, normalized_team_name) to pitcher info.
    """
    pitcher_lookup: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    print(f"Opening {FANGRAPHS_URL} in headless browser...")
    chrome_options = Options()
    if CHROME_HEADLESS:
        chrome_options.add_argument("--headless")
    try:
        driver = uc.Chrome(options=chrome_options)
    except Exception as e:
        print(f"ERROR: Unable to initiate undetected ChromeDriver: {e}")
        return pitcher_lookup
    driver.set_page_load_timeout(LOAD_WAIT_SECONDS * 2)
    try:
        driver.get(FANGRAPHS_URL)
        time.sleep(SCROLL_DELAY_SECONDS)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_DELAY_SECONDS * 2)
        wait = WebDriverWait(driver, LOAD_WAIT_SECONDS * 2)
        _ = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table")
            )
        )
        page_html = driver.page_source
    except Exception as e:
        print(f"ERROR: Selenium encountered an issue: {e}")
        driver.quit()
        return pitcher_lookup
    finally:
        driver.quit()
    debug_filename = "fangraphs_debug.html"
    try:
        with open(debug_filename, "w", encoding="utf-8") as f:
            f.write(page_html)
        print(f"Saved debug HTML to '{debug_filename}'.")
    except Exception as ex_file:
        print(f"WARN: Could not save debug file: {ex_file}")
    soup = BeautifulSoup(page_html, "html.parser")
    # Extract Header Dates (assuming base date is fixed; adjust if needed)
    header_row = soup.select_one("#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table thead tr")
    if not header_row:
        print("ERROR: Could not find the header row in the table.")
        return pitcher_lookup
    header_cells = header_row.find_all("th")
    headers: List[str] = []
    first_date = dt_datetime(2025, 4, 15)
    for i, cell in enumerate(header_cells[1:]):  # Skip team name header
        game_date_obj = first_date + timedelta(days=i)
        headers.append(game_date_obj.strftime("%Y-%m-%d"))
    print("Extracted header dates:", headers)
    team_rows = soup.select("#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table tbody tr")
    if not team_rows:
        print("ERROR: No team rows found in the table.")
        return pitcher_lookup
    print(f"Found {len(team_rows)} team rows. Processing...")
    for row in team_rows:
        cells = row.find_all("td")
        if not cells:
            continue
        raw_team_name = cells[0].get_text(strip=True)
        norm_team_name = normalize_team_name(raw_team_name)
        pitcher_cells = cells[1:]
        for idx, cell in enumerate(pitcher_cells):
            if idx >= len(headers):
                break
            game_date_iso = headers[idx]
            raw_text = cell.get_text(" ", strip=True)
            pitcher_name = None
            handedness = None
            a_tag = cell.find("a")
            if a_tag:
                pitcher_name = a_tag.get_text(strip=True)
            else:
                if raw_text and raw_text.lower() not in ["tbd", "ppd", "off"]:
                    pitcher_name = raw_text
            if pitcher_name:
                hand_match = re.search(r'\(([RLS])\)', pitcher_name)
                if hand_match:
                    handedness = hand_match.group(1)
                    pitcher_name = re.sub(r'\s*\([RLS]\)', '', pitcher_name).strip()
                if pitcher_name.lower() in ["tbd", "ppd", "off", ""]:
                    continue
                key = (game_date_iso, norm_team_name)
                pitcher_lookup[key] = {"pitcher_name": pitcher_name, "handedness": handedness}
    print("Mapped Pitcher Data:")
    for key, value in pitcher_lookup.items():
        print(key, value)
    return pitcher_lookup

# --- Pitcher Update Function with Retry ---
def update_pitchers_for_date(target_date_et: date) -> int:
    """
    Scrapes FanGraphs for probable pitcher data, then queries Supabase for games on the target date
    that are missing pitcher info, and updates those records.
    
    If scraping fails (empty data), the function will wait two minutes and retry.
    """
    print(f"\n--- Updating pitcher info using FanGraphs for date: {target_date_et} ---")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Error: Supabase URL/Key missing.")
        return 0

    # Retry loop: keep trying until we get non-empty pitcher data.
    pitcher_lookup = {}
    retry_attempt = 0
    while not pitcher_lookup:
        pitcher_lookup = scrape_fangraphs_probables()
        if not pitcher_lookup:
            retry_attempt += 1
            print(f"No pitcher data scraped from FanGraphs. Retry attempt {retry_attempt}: waiting 2 minutes before retrying...")
            time.sleep(120)  # Wait 2 minutes before retrying

    supabase_games = []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        target_date_iso = target_date_et.isoformat()
        print(f"Querying Supabase for games on {target_date_iso} missing pitcher info...")
        response = supabase.table(SUPABASE_TABLE_NAME) \
            .select("game_id, home_team_name, away_team_name, game_date_et") \
            .eq("game_date_et", target_date_iso) \
            .is_("home_probable_pitcher_name", None) \
            .execute()
        if hasattr(response, "data") and response.data is not None:
            supabase_games = response.data
        else:
            print(f"Supabase query error or no data: {getattr(response, 'error', 'Unknown')}")
            return 0
        print(f"Found {len(supabase_games)} game(s) in Supabase needing updates for {target_date_et}.")
    except Exception as e:
        print(f"Error querying Supabase: {e}")
        return 0

    if not supabase_games:
        print("No games found needing pitcher updates for this date.")
        return 0

    updated_count = 0
    UTC_ZONE = ZoneInfo("UTC")
    for game in supabase_games:
        supa_game_id = game.get("game_id")
        print(f"\nProcessing Supabase game_id: {supa_game_id}")
        try:
            supa_home_raw = game.get("home_team_name", "")
            supa_away_raw = game.get("away_team_name", "")
            supa_date_et = game.get("game_date_et")
            home_team_norm = normalize_team_name(supa_home_raw)
            away_team_norm = normalize_team_name(supa_away_raw)
            print(f"  Details: Date='{supa_date_et}', Home='{supa_home_raw}' (Norm='{home_team_norm}'), Away='{supa_away_raw}' (Norm='{away_team_norm}')")
            if not all([supa_game_id, supa_date_et, home_team_norm, away_team_norm]):
                print("  -> Skipping due to missing fields.")
                continue
            home_key = (supa_date_et, home_team_norm)
            away_key = (supa_date_et, away_team_norm)
            print(f"  Lookup Keys: Home={home_key}, Away={away_key}")
            home_pitcher_info = pitcher_lookup.get(home_key)
            away_pitcher_info = pitcher_lookup.get(away_key)
            if home_pitcher_info:
                print(f"    Home Pitcher: {home_pitcher_info}")
            if away_pitcher_info:
                print(f"    Away Pitcher: {away_pitcher_info}")
            update_payload = {}
            if home_pitcher_info:
                update_payload["home_probable_pitcher_name"] = home_pitcher_info["pitcher_name"]
                update_payload["home_probable_pitcher_handedness"] = home_pitcher_info["handedness"]
            if away_pitcher_info:
                update_payload["away_probable_pitcher_name"] = away_pitcher_info["pitcher_name"]
                update_payload["away_probable_pitcher_handedness"] = away_pitcher_info["handedness"]
            if update_payload:
                update_payload["updated_at"] = dt_datetime.now(UTC_ZONE).isoformat()
                try:
                    print(f"  Updating game_id {supa_game_id}...")
                    update_response = supabase.table(SUPABASE_TABLE_NAME) \
                        .update(update_payload) \
                        .eq("game_id", supa_game_id) \
                        .execute()
                    if hasattr(update_response, "data") and update_response.data:
                        print(f"    Success: Updated game_id {supa_game_id}.")
                        updated_count += 1
                    else:
                        print(f"    Warning: Update may have failed for game_id {supa_game_id}.")
                except Exception as e_upd:
                    print(f"    Error updating game_id {supa_game_id}: {e_upd}")
            else:
                print(f"  -> No pitcher data found for game_id {supa_game_id}.")
        except Exception as e_loop:
            print(f"Error processing game {supa_game_id}: {e_loop}")
    print(f"--- Finished updating pitcher info for {target_date_et}. Updated {updated_count} record(s). ---")
    return updated_count

# --- MLB Games Preview & Upsert (Schedule + Odds) ---
def build_and_upsert_mlb_previews() -> int:
    """
    Builds MLB game preview data by:
      1. Fetching game schedule data for today and tomorrow from api‑baseball.
      2. Fetching betting odds for these days from The Odds API.
      3. Matching odds events to schedule games.
      4. Creating preview data (with pitcher fields set to NULL) and upserting into Supabase.
    """
    print("\n--- Running MLB Game Preview Script (Schedule + Odds) ---")
    script_start_time = time.time()
    if not all([API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY]):
        print("FATAL ERROR: Config variables missing. Exiting.")
        return 0

    eastern_now = dt_datetime.now(ET_ZONE)
    today_et = eastern_now.date()
    tomorrow_et = today_et + timedelta(days=1)

    print(f"\nStep 1a: Fetching schedule for {today_et}")
    apibaseball_games_today = get_games_from_apibaseball(today_et)
    time.sleep(REQUEST_DELAY_SECONDS)
    print(f"\nStep 1b: Fetching schedule for {tomorrow_et}")
    apibaseball_games_tomorrow = get_games_from_apibaseball(tomorrow_et)
    apibaseball_games = apibaseball_games_today + apibaseball_games_tomorrow

    if not apibaseball_games:
        print(f"No game events found for {today_et} or {tomorrow_et}. Exiting.")
        return 0

    print(f"\nStep 2: Fetching betting odds covering {today_et} & {tomorrow_et}")
    odds_events = get_betting_odds(eastern_now)

    print("\nStep 3: Processing games and matching odds...")
    previews_to_upsert = []
    processed_game_ids = set()

    for game in apibaseball_games:
        try:
            game_id = game.get("id")
            if not game_id or game_id in processed_game_ids:
                continue
            processed_game_ids.add(game_id)
            status_info = game.get("status", {})
            status_short = status_info.get("short", "").upper()
            if status_short != "NS":
                continue

            teams_info = game.get("teams", {})
            home_team_info = teams_info.get("home", {})
            away_team_info = teams_info.get("away", {})
            home_team_name_raw = home_team_info.get("name")
            away_team_name_raw = away_team_info.get("name")
            venue_info = game.get("venue", {})
            venue_name = venue_info.get("name") if isinstance(venue_info, dict) else None
            venue_city = venue_info.get("city") if isinstance(venue_info, dict) else None

            scheduled_time_utc_str = game.get("date")
            game_date_et_str = None
            if scheduled_time_utc_str:
                try:
                    game_dt_utc = dateutil_parser.isoparse(scheduled_time_utc_str)
                    game_date_et_str = game_dt_utc.astimezone(ET_ZONE).date().isoformat()
                except Exception as e:
                    print(f"Warn: Could not parse date {scheduled_time_utc_str} for game {game_id}: {e}")

            matched_odds_event = match_odds_for_apibaseball_game(game, odds_events)
            extracted_odds = extract_odds_data(matched_odds_event)
            preview_data = {
                "game_id": game_id,
                #"game_uid": None,
                "scheduled_time_utc": scheduled_time_utc_str,
                "game_date_et": game_date_et_str,
                "status_detail": status_info.get("long"),
                "status_state": "pre" if status_short == "NS" else status_short,
                "home_team_id": home_team_info.get("id"),
                "home_team_name": home_team_name_raw,
                "away_team_id": away_team_info.get("id"),
                "away_team_name": away_team_name_raw,

                # Pitcher info set to NULL
                "home_probable_pitcher_name": None,
                "away_probable_pitcher_name": None,
                # Odds Data
                "moneyline": extracted_odds["raw"].get("moneyline"),
                "spread": extracted_odds["raw"].get("spread"),
                "total": extracted_odds["raw"].get("total"),
                "moneyline_home_clean": extracted_odds["clean"].get("moneyline_home_clean"),
                "moneyline_away_clean": extracted_odds["clean"].get("moneyline_away_clean"),
                "spread_home_line_clean": extracted_odds["clean"].get("spread_home_line_clean"),
                "spread_home_price_clean": extracted_odds["clean"].get("spread_home_price_clean"),
                "spread_away_price_clean": extracted_odds["clean"].get("spread_away_price_clean"),
                "total_line_clean": extracted_odds["clean"].get("total_line_clean"),
                "total_over_price_clean": extracted_odds["clean"].get("total_over_price_clean"),
                "total_under_price_clean": extracted_odds["clean"].get("total_under_price_clean"),
                "raw_api_response": json.dumps(game)
            }
            if preview_data["game_id"] and preview_data["home_team_name"] and preview_data["away_team_name"]:
                previews_to_upsert.append(preview_data)
            else:
                print(f"Warning: Skipping game with ID: {preview_data.get('game_id')} due to missing essential fields.")
        except Exception as e:
            print(f"Error processing game {game.get('id', 'UNKNOWN')}: {e}")

    if previews_to_upsert:
        print(f"\nStep 4: Upserting {len(previews_to_upsert)} processed previews to Supabase...")
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(previews_to_upsert, on_conflict="game_id").execute()
            print("Supabase upsert completed.")
        except Exception as e:
            print(f"Error during Supabase upsert call: {e}")
    else:
        print("\nStep 4: No valid game previews generated to upsert.")

    script_end_time = time.time()
    print(f"\n--- MLB Preview Script finished. Processed {len(processed_game_ids)} unique games. ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    return len(previews_to_upsert)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MLB Games Preview Script (Schedule, Odds, and Pitcher Updates)...")
    # Optionally clear old games (if implemented)
    # clear_old_mlb_games()

    # Step 1: Build and upsert game previews
    build_and_upsert_mlb_previews()

    # Step 2: Update probable pitcher info for today's games
    today_et_date = dt_datetime.now(ET_ZONE).date()
    update_pitchers_for_date(today_et_date)

    print("\nMLB Games Preview Script finished.")
