# backend/data_pipeline/mlb_games_preview.py

"""
This script builds MLB game previews by fetching schedule info via api‑baseball and betting odds via The Odds API,
then upserts the preview data into a Supabase table. Pitcher info is updated separately by scraping FanGraphs.
"""

import time
import os
import sys
import re
import httpx
import json
from datetime import date, timedelta, datetime as dt_datetime, datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List, Any
from dateutil import parser as dateutil_parser
import subprocess, sys, shutil


import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# allow `from caching.supabase_client import supabase` no matter where we run from
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# --- Local Config & Variables ---
# First try to load a local config.py for dev, else fall back to env
try:
    from config import (
        API_SPORTS_KEY,      # for api-baseball schedule
        ODDS_API_KEY,        # odds from The Odds API
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY # Supabase service role key
    )
    print("Using config.py for credentials")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY      = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY        = os.getenv("ODDS_API_KEY")
    SUPABASE_URL        = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY= os.getenv("SUPABASE_SERVICE_KEY")

# --- 3rd-party Imports ---
try:
    import requests
    from supabase import create_client, Client
except ImportError as e:
    print(f"FATAL ERROR: Missing required third‑party module: {e}")
    exit(1)

# Now verify we have everything we need
missing = [
    name for name,val in [
        ("API_SPORTS_KEY",      API_SPORTS_KEY),
        ("ODDS_API_KEY",        ODDS_API_KEY),
        ("SUPABASE_URL",        SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY",SUPABASE_SERVICE_KEY),
    ]
    if not val
]
if missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(missing)}")
    exit(1)

# --- Configuration Constants ---
API_BASEBALL_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1
MLB_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
PREFERRED_BOOKMAKER_KEY = "draftkings"
REQUEST_DELAY_SECONDS = 1
SUPABASE_TABLE_NAME = "mlb_game_schedule"
ET_ZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
CHROME_HEADLESS = True
SCROLL_DELAY = 2
LOAD_WAIT = 15
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"

# --- Helper Functions ---

def _detect_chrome_major() -> int:
    """Return installed Chrome major version (0 if not found)."""
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chrome"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for path in filter(None, candidates):
        try:
            out = subprocess.check_output([path, "--version"]).decode()
            m = re.search(r" (\d+)\.", out)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return 0


def normalize_team_name(name: str) -> str:
    """
    Normalizes team names so that they match Supabase values.
    """
    if not name or not isinstance(name, str):
        return ""
    original = name.strip()
    temp = original.replace("St.Louis", "St Louis").replace("St.", "St").lower().strip()

    # Quick hits for Athletics
    if temp in ["ath", "athletics", "oakland athletics"]:
        return "Athletics"
    # Any spelled‑out “st louis” → the full Cardinals name
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
        # Add this line:
        "stl": "St. Louis Cardinals",
    }
    if temp in mapping:
        return mapping[temp]
    return temp.title()

def clear_old_games():
    """
    Deletes any mlb_game_schedule rows whose game_date_et is before today (ET).
    """
    supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    today_iso = dt.now(ET_ZONE).date().isoformat()
    # Delete all games strictly before today
    resp = (
        supa.table(SUPABASE_TABLE_NAME)
        .delete()
        .lt("game_date_et", today_iso)
        .execute()
    )
    deleted = getattr(resp, "count", None) or len(getattr(resp, "data", []))
    print(f"Cleared {deleted} games before {today_iso}.")

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
    """Fetches betting odds for MLB events from The Odds API covering a day's range."""
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not available.")
        return []
    utc_zone = ZoneInfo("UTC")
    # Define range using ET date boundaries converted to UTC
    et_date = target_date_dt.astimezone(ET_ZONE)
    start_et = et_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et = et_date.replace(hour=23, minute=59, second=59, microsecond=0)
    start_utc = start_et.astimezone(utc_zone)
    end_utc = end_et.astimezone(utc_zone)
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

# Supabase constants & normalize_team_name already imported…

FANGRAPHS_URL   = "https://www.fangraphs.com/roster-resource/probables-grid"
SCROLL_DELAY    = 2
LOAD_WAIT       = 15
UTC             = ZoneInfo("UTC")
CHROME_HEADLESS = True

def scrape_fangraphs_probables(base_date: date) -> dict:
    opts = Options()
    if CHROME_HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1920,1080")

    major = _detect_chrome_major()
    if major:
        print(f"Launching uc.Chrome for v{major}")
    driver = uc.Chrome(options=opts, version_main=major or None)  # fallback to default finder
    driver.set_page_load_timeout(LOAD_WAIT * 2)
    driver.set_script_timeout(LOAD_WAIT * 2)

    try:
        driver.get(FANGRAPHS_URL)
        for _ in range(4):
            time.sleep(SCROLL_DELAY)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        WebDriverWait(driver, LOAD_WAIT * 2).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
    except Exception:
        pass
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    # --- MODIFIED TABLE FINDING LOGIC ---
    target_table = None

    # 1) Find the specific header cell using its data-stat attribute
    #    We use a CSS selector: 'th[data-stat="Team"]' looks for a <th> element
    #    that has an attribute 'data-stat' with the value 'Team'.
    team_header_cell = soup.select_one('th[data-stat="Team"]')

    if team_header_cell:
        # 2) If we found the header, navigate up the HTML tree to find its parent table
        target_table = team_header_cell.find_parent('table')
        if target_table:
             print("Successfully located target table via header 'th[data-stat=\"Team\"]'.")
        else:
            # This would be odd, but possible if HTML is malformed
            print("Found the 'Team' header cell, but failed to find its parent table.")
            return {}
    else:
        # If the header cell itself wasn't found
        print("Could not find the crucial header cell: 'th[data-stat=\"Team\"]'. FanGraphs page structure has likely changed.")
        return {}

    # --- END OF MODIFIED LOGIC ---

    # The rest of your logic should proceed using the found 'target_table'
    # Make sure the subsequent selectors operate on 'target_table', not 'soup' if they were global before.

    # 2) Figure out how many days across (using the found target_table)
    first_row = next(
        (r for r in target_table.select("tbody tr") if len(r.find_all("td")) > 1),
        None
    )
    if not first_row:
        print("No data rows found in the identified table.")
        return {}
    # Subtract 1 because the first cell is the team name
    num_days = len(first_row.find_all("td")) - 1
    if num_days <= 0:
        print(f"Warning: Found table but detected {num_days} data columns. Check structure.")
        return {}

    # 3) Build the date headers
    headers = [(base_date + timedelta(days=i)).isoformat() for i in range(num_days)]

    # 4) Extract all pitchers (using the found target_table)
    lookup: Dict[Tuple[str,str],Dict[str,Optional[str]]] = {}
    # Use target_table.select to search only within the correct table
    for row in target_table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) <= 1: # Should be team + at least one day
            continue

        # Ensure the first cell contains the team name
        team_cell = cells[0]
        # You might want to add robustness here, e.g., check if team_cell looks like a team
        team = normalize_team_name(team_cell.get_text(strip=True))
        if not team: # Skip rows where team name isn't found/parsed
            continue

        # Process pitcher cells for the detected number of days
        for i, cell in enumerate(cells[1:]):
            if i >= num_days: # Don't process more cells than expected days
                break

            # Your existing pitcher name extraction logic
            links = cell.find_all("a")
            name = links[-1].get_text(strip=True) if links else cell.get_text(" ", strip=True)

            # Clean up '@ TEAM' prefixes which sometimes appear
            name = re.sub(r'^@\s*[A-Z]{2,3}\s+', "", name).strip()

            # Extract handedness
            m = re.search(r"\(([RLS])\)", name)
            hand = m.group(1) if m else None
            if hand:
                name = re.sub(r"\s*\([RLS]\)", "", name).strip()

            # Skip if no valid name found
            if not name or name.lower() in ("tbd", "off", "ppd", ""):
                continue

            # Store the found pitcher info
            lookup[(headers[i], team)] = {"pitcher_name": name, "handedness": hand}

    print(f"Mapped {len(lookup)} probables via Selenium using specific table.")
    return lookup

def get_probables_lookup(base_date: date) -> Dict[Tuple[str,str],Dict[str,Optional[str]]]:
    """Always use Selenium scraper now."""
    return scrape_fangraphs_probables(base_date)

def update_pitchers_for_date(target_date: date) -> int:
    """
    For the given date (ET), scrape FanGraphs and upsert starting pitchers
    into Supabase. Returns count of updated records.
    """
    print(f"\n--- Updating pitchers for {target_date} ---")
    supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    iso = target_date.isoformat()

    resp = (
        supa.table(SUPABASE_TABLE_NAME)
        .select(
            "game_id, home_team_name, away_team_name, "
            "home_probable_pitcher_name, away_probable_pitcher_name, game_date_et"
        )
        .eq("game_date_et", iso)
        .execute()
    )
    games = resp.data or []
    if not games:
        print("No games found.")
        return 0

    fg_lookup = get_probables_lookup(target_date)
    updated = 0

    for g in games:
        gid = g["game_id"]
        d   = g["game_date_et"]
        h   = normalize_team_name(g["home_team_name"])
        a   = normalize_team_name(g["away_team_name"])
        payload: Dict[str, Any] = {}

        new_h = fg_lookup.get((d, h), {}).get("pitcher_name")
        if new_h and new_h != (g.get("home_probable_pitcher_name") or "").strip():
            payload["home_probable_pitcher_name"]      = new_h
            payload["home_probable_pitcher_handedness"] = fg_lookup[(d, h)]["handedness"]

        new_a = fg_lookup.get((d, a), {}).get("pitcher_name")
        if new_a and new_a != (g.get("away_probable_pitcher_name") or "").strip():
            payload["away_probable_pitcher_name"]      = new_a
            payload["away_probable_pitcher_handedness"] = fg_lookup[(d, a)]["handedness"]

        # Check if payload has data (meaning an update is needed)
        if payload:
            # Set the updated_at timestamp just before attempting the update
            payload["updated_at"] = dt.now(UTC).isoformat()
            print(f" → Attempting to patch game {gid}: {payload}") # Updated print message slightly

            # --- Retry Logic Start ---
            max_retries = 3
            retry_delay_seconds = 10  # Wait 10 seconds between retries
            success = False # Flag to track if update succeeded within retries

            # Loop for retry attempts
            for attempt in range(max_retries):
                try:
                    # Attempt the Supabase update API call
                    r = (
                        supa.table(SUPABASE_TABLE_NAME)
                        .update(payload)
                        .eq("game_id", gid)
                        .execute()
                    )

                    # Check if Supabase returned data in the response.
                    # The presence of 'data' often indicates a successful update operation.
                    # Note: Behavior might vary based on RLS or if 'return=minimal' is set.
                    if getattr(r, "data", None):
                        print(f"   ✅ Successfully updated game {gid} on attempt {attempt + 1}.")
                        success = True
                        break # Exit the retry loop successfully

                    else:
                        # Handle case where execute() succeeds without exception but returns no data.
                        # This *might* be normal (e.g., using 'return=minimal') or could indicate
                        # an issue like RLS preventing the update but not raising an HTTP error.
                        # We'll log it as a warning but treat it as success for now, assuming no error means OK.
                        # You might need to adjust this based on your specific Supabase setup/expectations.
                        print(f"   ⚠️ Update command executed for game {gid} on attempt {attempt + 1}, but no data returned. Assuming success (verify if RLS/permissions are involved).")
                        success = True
                        break # Exit the retry loop

                # --- Exception Handling during the Supabase API call ---

                # Catch specific httpx network/HTTP errors (more specific than general Exception)
                # TransportError includes SSL errors like EOF, ConnectError, etc.
                # TimeoutException handles timeouts.
                # HTTPStatusError handles responses >= 400.
                except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as e_http:
                    error_type = type(e_http).__name__
                    # For HTTPStatusError, include the status code
                    status_code_info = f" (Status Code: {e_http.response.status_code})" if isinstance(e_http, httpx.HTTPStatusError) else ""
                    print(f"   ⚠️ Network/HTTP error ({error_type}{status_code_info}) on attempt {attempt + 1}/{max_retries} for game {gid}: {e_http}")
                    if attempt < max_retries - 1:
                        print(f"      Retrying in {retry_delay_seconds} seconds...")
                        time.sleep(retry_delay_seconds)
                    # Failure message is handled after the loop if 'success' remains False

                # Catch any other unexpected exceptions during the API call
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"   ⚠️ Unexpected error ({error_type}) on attempt {attempt + 1}/{max_retries} for game {gid}: {e}")
                    # Decide if this error is retryable. We'll retry for now.
                    if attempt < max_retries - 1:
                        print(f"      Retrying in {retry_delay_seconds} seconds...")
                        time.sleep(retry_delay_seconds)
                    # Failure message is handled after the loop if 'success' remains False

            # --- After the retry loop finishes ---
            if success:
                # Increment the overall count only if one of the attempts was successful
                updated += 1
            else:
                # Log the final failure only if all retries were exhausted without success
                print(f"   ✖️ Failed to update game {gid} after {max_retries} attempts.")
            # --- Retry Logic End ---

# The print and return statements outside the 'if payload:' block remain unchanged
# print(f"--- Done: {updated} updated. ---")
# return updated

    print(f"--- Done: {updated} updated. ---")
    return updated

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
    if not all([API_SPORTS_KEY, ODDS_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
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
            #venue_name = venue_info.get("name") if isinstance(venue_info, dict) else None
            #venue_city = venue_info.get("city") if isinstance(venue_info, dict) else None

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
                "scheduled_time_utc": scheduled_time_utc_str,
                "game_date_et": game_date_et_str,
                "status_detail": status_info.get("long"),
                "status_state": "pre" if status_short == "NS" else status_short,
                "home_team_id": home_team_info.get("id"),
                "home_team_name": home_team_name_raw,
                "away_team_id": away_team_info.get("id"),
                "away_team_name": away_team_name_raw,
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
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            _upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(previews_to_upsert, on_conflict="game_id").execute()
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
    print("Starting MLB Games Preview Script (Schedule, Odds, and Pitcher Updates)…")

    # Step 0: Clear old games from table
    clear_old_games()  

    # Step 1: Build and upsert game previews
    build_and_upsert_mlb_previews()

    # Step 2: Update probable pitcher info for today's games
    today_et_date = dt_datetime.now(ET_ZONE).date()
    update_pitchers_for_date(today_et_date)

    print("\nMLB Games Preview Script finished.")
