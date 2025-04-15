#!/usr/bin/env python3
"""
mlb_pitcher_updater.py

This script uses a headless Selenium browser to fetch the rendered FanGraphs Probables Grid,
parses the HTML with BeautifulSoup to extract starting pitcher info (pitcher name and handedness),
and then updates the Supabase table 'mlb_game_schedule' for games on a specified date.
"""

import time
import re
import json
from datetime import date, timedelta, datetime as dt_datetime  # Import datetime as dt_datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List, Any

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- BeautifulSoup Import ---
from bs4 import BeautifulSoup

# --- Supabase and Config Imports ---
try:
    from supabase import create_client, Client
    from config import SUPABASE_URL, SUPABASE_ANON_KEY  # Only need Supabase creds here
    print("Successfully imported config variables for pitcher updater.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import from config or supabase: {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during import: {e}")
    exit(1)

# --- Configuration ---
SUPABASE_TABLE_NAME = "mlb_game_schedule"  # Target table to update
ET_ZONE = ZoneInfo("America/New_York")
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"
FG_HEADERS = {  # Custom User-Agent
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}
SCRAPE_DELAY_SECONDS = 5  # Delay between actions, to allow lazy-loading

# --- Selenium (Headless Chrome) Setup ---
def get_headless_driver() -> webdriver.Chrome:
    """Initializes and returns a headless Chrome WebDriver."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # If necessary, specify the executable path of your chromedriver here.
    driver = webdriver.Chrome(options=options)
    return driver

# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    """Normalizes team names by stripping extra whitespace, lowercasing, and mapping known abbreviations."""
    if not name or not isinstance(name, str):
        return ""
    name_norm = name.strip().lower()
    team_map = {
        "bal": "baltimore orioles",
        "bos": "boston red sox",
        "nyy": "new york yankees",
        "tbr": "tampa bay rays",
        "tor": "toronto blue jays",
        "chw": "chicago white sox",
        "cle": "cleveland guardians",
        "det": "detroit tigers",
        "kcr": "kansas city royals",
        "min": "minnesota twins",
        "ath": "oakland athletics",
        "hou": "houston astros",
        "laa": "los angeles angels",
        "sea": "seattle mariners",
        "tex": "texas rangers",
        "atl": "atlanta braves",
        "mia": "miami marlins",
        "nym": "new york mets",
        "phi": "philadelphia phillies",
        "wsn": "washington nationals",
        "chc": "chicago cubs",
        "cin": "cincinnati reds",
        "mil": "milwaukee brewers",
        "pit": "pittsburgh pirates",
        "stl": "st louis cardinals",
        "ari": "arizona diamondbacks",
        "col": "colorado rockies",
        "lad": "los angeles dodgers",
        "sdp": "san diego padres",
        "sfg": "san francisco giants",
    }
    return " ".join(team_map.get(name_norm, name_norm).split())

# --- FanGraphs Scraper using Selenium (Headless) ---
def fetch_fangraphs_pitcher_data() -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Uses Selenium to load the FanGraphs Probables Grid, then uses BeautifulSoup to parse it
    and build a lookup dictionary keyed by (game_date_iso, normalized_team_name).
    Returns:
      { (game_date_iso, team_name): {"pitcher_name": str or None, "handedness": str or None } }
    """
    print(f"Opening {FANGRAPHS_URL} in headless browser...")
    driver = get_headless_driver()
    try:
        driver.get(FANGRAPHS_URL)
        print("Waiting for page content to load...")
        time.sleep(SCRAPE_DELAY_SECONDS)
        # Scroll to bottom to force lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCRAPE_DELAY_SECONDS)
        rendered_html = driver.page_source
        # Save HTML for debugging
        debug_filename = "fangraphs_debug.html"
        with open(debug_filename, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        print(f"Saved debug HTML to '{debug_filename}'.")
    except Exception as e:
        print(f"Error loading page with Selenium: {e}")
        driver.quit()
        return {}
    finally:
        driver.quit()

    soup = BeautifulSoup(rendered_html, "html.parser")
    pitcher_lookup: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}

    # Use the newly discovered CSS selector path
    table_selector = "#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table"
    grid_table = soup.select_one(table_selector)
    if not grid_table:
        print(f"ERROR: Unable to locate the grid table with selector '{table_selector}'.")
        return pitcher_lookup
    else:
        print("Successfully located the grid table.")

    # --- Extract Header Dates ---
    header_row = grid_table.select_one("thead tr")
    if not header_row:
        print("ERROR: Header row not found.")
        return pitcher_lookup

    header_cells = header_row.find_all("th")
    start_index = 1 if header_cells and header_cells[0].get_text(strip=True).lower() in ["", "team"] else 0

    # Define a base date—adjust this as necessary for your schedule context.
    base_date = dt_datetime(2025, 4, 15)
    headers = []
    for i, th in enumerate(header_cells[start_index:]):
        header_text = th.get_text(strip=True)
        # If header_text is a day abbreviation, use base_date + offset
        if header_text in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}:
            computed_date = base_date + timedelta(days=i)
            headers.append(computed_date.strftime("%Y-%m-%d"))
        else:
            try:
                date_obj = dt_datetime.strptime(header_text, "%b %d").replace(year=base_date.year)
                headers.append(date_obj.strftime("%Y-%m-%d"))
            except Exception as e:
                print(f"Warn: Could not parse header '{header_text}': {e}")
                headers.append(None)
    print("Extracted header dates:", headers)

    # --- Extract Team Rows ---
    team_rows_selector = "#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table tbody tr"
    team_rows = soup.select(team_rows_selector)
    if not team_rows:
        print(f"ERROR: No team rows found using selector '{team_rows_selector}'.")
        return pitcher_lookup
    print(f"Found {len(team_rows)} team rows. Processing...")

    for row in team_rows:
        cells = row.find_all("td")
        if not cells:
            continue
        team_cell = cells[0]
        team_name_raw = team_cell.get_text(strip=True)
        team_name_norm = normalize_team_name(team_name_raw)
        if not team_name_norm:
            continue
        pitcher_cells = cells[start_index:]
        num_cells = min(len(pitcher_cells), len(headers))
        for i in range(num_cells):
            cell = pitcher_cells[i]
            game_date = headers[i]
            if not game_date:
                continue
            pitcher_name = None
            handedness = None
            raw_cell_text = cell.get_text(" ", strip=True)
            a_tag = cell.find("a")
            if a_tag:
                pitcher_name = a_tag.get_text(strip=True)
                hand_match = re.search(r'\(([RLS])\)', raw_cell_text)
                if hand_match:
                    handedness = hand_match.group(1)
            elif raw_cell_text and raw_cell_text.lower() not in ['tbd', 'ppd', 'off']:
                hand_match = re.search(r'\(([RLS])\)', raw_cell_text)
                if hand_match:
                    handedness = hand_match.group(1)
                    pitcher_name = re.sub(r'\s*\(([RLS])\)', '', raw_cell_text).strip()
                else:
                    pitcher_name = raw_cell_text

            # Skip if no valid pitcher name or placeholder encountered
            if not pitcher_name or pitcher_name.lower() in ['tbd', 'ppd', 'off']:
                continue

            pitcher_name = " ".join(pitcher_name.split())
            key = (game_date, team_name_norm)
            pitcher_lookup[key] = {
                "pitcher_name": pitcher_name,
                "handedness": handedness
            }

    print("Mapped Pitcher Data:")
    for key, value in pitcher_lookup.items():
        print(key, value)

    return pitcher_lookup

# --- Supabase Update Logic ---
def update_pitchers_for_date(target_date_et: date) -> int:
    """Fetches pitcher data from FanGraphs and updates the Supabase mlb_game_schedule table for the target date."""
    print(f"\n--- Updating pitcher info using FanGraphs for date: {target_date_et} ---")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Error: Supabase URL/Key missing.")
        return 0

    # Step 1: Scrape FanGraphs pitcher data using Selenium
    fangraphs_pitcher_lookup = fetch_fangraphs_pitcher_data()
    if not fangraphs_pitcher_lookup:
        print("No pitcher data scraped from FanGraphs.")
        return 0

    # Step 2: Query Supabase for games on the target date needing updates
    supabase_games = []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        target_date_iso = target_date_et.isoformat()
        print(f"Querying Supabase for games on {target_date_iso} needing pitcher updates...")
        response = supabase.table(SUPABASE_TABLE_NAME) \
            .select("game_id, home_team_name, away_team_name, game_date_et") \
            .eq("game_date_et", target_date_iso) \
            .is_("home_probable_pitcher_name", None) \
            .execute()
        if hasattr(response, "data") and response.data is not None:
            supabase_games = response.data
        else:
            print(f"Supabase query error/no data: {getattr(response, 'error', 'Unknown')}")
            return 0
        print(f"Found {len(supabase_games)} games in Supabase needing updates for {target_date_et}.")
    except Exception as e:
        print(f"Error querying Supabase: {e}")
        return 0

    if not supabase_games:
        print("No games found needing pitcher updates for this date.")
        return 0

    # Step 3: For each game, update pitcher info if found in the lookup
    updated_count = 0
    UTC_ZONE = ZoneInfo("UTC")
    for game in supabase_games:
        try:
            supa_game_id = game.get("game_id")
            home_team_raw = game.get("home_team_name", "")
            away_team_raw = game.get("away_team_name", "")
            supa_date_et = game.get("game_date_et")
            home_team_norm = normalize_team_name(home_team_raw)
            away_team_norm = normalize_team_name(away_team_raw)
            if not all([supa_game_id, home_team_norm, away_team_norm, supa_date_et]):
                continue

            home_lookup_key = (supa_date_et, home_team_norm)
            away_lookup_key = (supa_date_et, away_team_norm)
            home_pitcher_info = fangraphs_pitcher_lookup.get(home_lookup_key)
            away_pitcher_info = fangraphs_pitcher_lookup.get(away_lookup_key)

            update_payload = {}
            if home_pitcher_info:
                update_payload['home_probable_pitcher_name'] = home_pitcher_info.get('pitcher_name')
                update_payload['home_probable_pitcher_handedness'] = home_pitcher_info.get('handedness')
            if away_pitcher_info:
                update_payload['away_probable_pitcher_name'] = away_pitcher_info.get('pitcher_name')
                update_payload['away_probable_pitcher_handedness'] = away_pitcher_info.get('handedness')

            if update_payload:
                print(f"Match found for Supabase game_id {supa_game_id}. Updating pitcher info...")
                update_payload['updated_at'] = dt_datetime.now(UTC_ZONE).isoformat()
                try:
                    update_response = supabase.table(SUPABASE_TABLE_NAME).update(update_payload).eq("game_id", supa_game_id).execute()
                    if hasattr(update_response, "data") and update_response.data:
                        print(f"Success: Updated game_id {supa_game_id}.")
                        updated_count += 1
                    else:
                        print(f"Warning: Update failed for {supa_game_id}. Response: {update_response}")
                except Exception as e_upd:
                    print(f"Error updating game_id {supa_game_id}: {e_upd}")
        except Exception as e_loop:
            print(f"Error processing Supabase game {game.get('game_id', '??')}: {e_loop}")

    print(f"--- Finished FanGraphs pitcher update for {target_date_et}. Updated {updated_count} records. ---")
    return updated_count

# --- Runner ---
if __name__ == "__main__":
    print("Starting MLB Pitcher Update Script (using FanGraphs and Selenium)...")
    today_et_date = dt_datetime.now(ET_ZONE).date()
    update_pitchers_for_date(today_et_date)
    print("Pitcher update script finished.")
