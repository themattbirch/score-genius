# backend/data_pipeline/mlb_pitcher_updater.py

"""
This script uses undetected‑chromedriver (with Selenium) to scrape the FanGraphs
Probables Grid and then upserts probable pitcher data into the Supabase table 
"mlb_game_schedule." It uses an updated normalization function so that team names
from FanGraphs (such as Athletics) match exactly the team names stored in Supabase.
"""


import time
import re
import json
import datetime
from datetime import date, timedelta, datetime as dt_datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List, Any

# --- 3rd-party Imports ---
try:
    import undetected_chromedriver as uc
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    from supabase import create_client, Client
except ImportError as e:
    print(f"FATAL ERROR: Missing a required import: {e}")
    exit(1)

# --- Local Imports ---
try:
    from config import SUPABASE_URL, SUPABASE_ANON_KEY
    print("Successfully imported config variables for pitcher updater.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import from config: {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during import: {e}")
    exit(1)

# --- Configuration ---
SUPABASE_TABLE_NAME = "mlb_game_schedule"  # Supabase table name to update
ET_ZONE = ZoneInfo("America/New_York")
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"

# Selenium/undetected‑chromedriver settings
SCROLL_DELAY_SECONDS = 5  # Increased delay after scrolling to allow lazy-loading
LOAD_WAIT_SECONDS = 30    # Increased wait time for page elements to load
CHROME_HEADLESS = True    # Set to False if you wish to see the browser window

# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    """
    Normalizes team names so that data scraped from FanGraphs and team names
    stored in Supabase match exactly.

    Special handling:
    - If the lowercased name contains "st.louis", "st louis", or equals "stl",
      returns "St. Louis Cardinals".
    - If the lowercased name contains "athletic" or equals "oakland" or "ath",
      returns "Athletics".
    - Otherwise, if the lowercased name appears in a mapping dictionary, returns the mapped value.
    - Else falls back to title-casing the name.
    """
    if not name or not isinstance(name, str):
        return ""
    original = name.strip().lower()

    if "st.louis" in original or "st louis" in original or original == "stl":
        return "St. Louis Cardinals"

    if "athletic" in original or original == "oakland" or original == "ath":
        return "Athletics"

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
    if original in mapping:
        return mapping[original]

    return original.title()


def scrape_fangraphs_probables() -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Uses Selenium (with undetected‑chromedriver) to open the FanGraphs Probables Grid,
    scrolls to force lazy-loading, and extracts header dates and team-row pitcher data.

    Returns:
        A dictionary with keys (game_date_iso, normalized_team_name) and values:
        {"pitcher_name": <str>, "handedness": <str or None>}.
    """
    pitcher_lookup: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    print(f"Opening {FANGRAPHS_URL} in headless browser...")

    chrome_options = Options()
    if CHROME_HEADLESS:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")

    try:
        driver = uc.Chrome(options=chrome_options)
    except Exception as e:
        print(f"ERROR: Unable to initiate undetected ChromeDriver: {e}")
        return pitcher_lookup

    driver.set_page_load_timeout(LOAD_WAIT_SECONDS * 2)
    try:
        driver.get(FANGRAPHS_URL)
        print("Waiting for page content to load...")
        time.sleep(SCROLL_DELAY_SECONDS)
        print("Scrolling to bottom to force lazy-loading...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_DELAY_SECONDS * 2)

        wait = WebDriverWait(driver, LOAD_WAIT_SECONDS * 2)
        table_wrapper = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table")
            )
        )
        print("Successfully located the grid table.")
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

    # --- Extract Header Dates ---
    header_row = soup.select_one("#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table thead tr")
    if not header_row:
        print("ERROR: Could not find the header row in the table.")
        return pitcher_lookup
    header_cells = header_row.find_all("th")

    headers: List[str] = []
    first_date = dt_datetime(2025, 4, 15)
    for i, cell in enumerate(header_cells[1:]):  # Skip the first column (team names)
        game_date_obj = first_date + timedelta(days=i)
        headers.append(game_date_obj.strftime("%Y-%m-%d"))
    print("Extracted header dates:", headers)

    # --- Extract Team Rows & Pitcher Data ---
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
    for k, v in pitcher_lookup.items():
        print(k, v)

    return pitcher_lookup


def update_pitchers_for_date(target_date_et: date) -> int:
    """
    1) Scrapes FanGraphs for probable pitchers using Selenium.
    2) Queries Supabase for games on the given date.
    3) Updates the corresponding Supabase records with the scraped pitcher data.
    """
    print(f"\n--- Updating pitcher info using FanGraphs for date: {target_date_et} ---")
    from supabase import create_client  # Local import to ensure scope
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Error: Supabase URL/Key missing.")
        return 0

    # Step 1: Scrape FanGraphs data
    pitcher_lookup = scrape_fangraphs_probables()
    if not pitcher_lookup:
        print("No pitcher data scraped from FanGraphs. No updates performed.")
        return 0

    # Step 2: Query Supabase for games on target_date_et
    supabase_games = []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        target_date_iso = target_date_et.isoformat()
        print(f"Querying Supabase for games on {target_date_iso}...")
        response = (
            supabase.table(SUPABASE_TABLE_NAME)
            .select("game_id, home_team_name, away_team_name, game_date_et")
            .eq("game_date_et", target_date_iso)
            .execute()
        )
        if hasattr(response, "data") and response.data is not None:
            supabase_games = response.data
        else:
            print(f"Supabase query error or no data: {getattr(response, 'error', 'Unknown')}")
            return 0
        print(f"Found {len(supabase_games)} game(s) in Supabase for {target_date_et}.")
    except Exception as e:
        print(f"Error querying Supabase: {e}")
        return 0

    if not supabase_games:
        print("No games found for this date.")
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
            print(f" Supabase Details: Date='{supa_date_et}', Home='{supa_home_raw}' (Norm='{home_team_norm}'), Away='{supa_away_raw}' (Norm='{away_team_norm}')")
            if not all([supa_game_id, supa_date_et, home_team_norm, away_team_norm]):
                print(" -> Skipping due to missing fields.")
                continue

            home_key = (supa_date_et, home_team_norm)
            away_key = (supa_date_et, away_team_norm)
            print(f" Lookup Keys: Home={home_key}, Away={away_key}")

            home_pitcher_info = pitcher_lookup.get(home_key)
            away_pitcher_info = pitcher_lookup.get(away_key)
            print(f" FanGraphs Lookup: Home Found={home_pitcher_info is not None}, Away Found={away_pitcher_info is not None}")
            if home_pitcher_info:
                print(f" Home Pitcher: {home_pitcher_info}")
            if away_pitcher_info:
                print(f" Away Pitcher: {away_pitcher_info}")

            update_payload = {}
            if home_pitcher_info:
                update_payload["home_probable_pitcher_name"] = home_pitcher_info["pitcher_name"]
                update_payload["home_probable_pitcher_handedness"] = home_pitcher_info["handedness"]
            if away_pitcher_info:
                update_payload["away_probable_pitcher_name"] = away_pitcher_info["pitcher_name"]
                update_payload["away_probable_pitcher_handedness"] = away_pitcher_info["handedness"]

            if update_payload:
                print(f" Update Payload: {update_payload}")
                update_payload["updated_at"] = dt_datetime.now(UTC_ZONE).isoformat()
                try:
                    print(f" Executing update for game_id {supa_game_id}...")
                    update_response = (
                        supabase.table(SUPABASE_TABLE_NAME)
                        .update(update_payload)
                        .eq("game_id", supa_game_id)
                        .execute()
                    )
                    print(f" Supabase Update Response for {supa_game_id}: {update_response}")
                    if hasattr(update_response, "data") and update_response.data:
                        print(f" Success: Updated game_id {supa_game_id}.")
                        updated_count += 1
                    else:
                        print(f" Warn: Update may have failed for game_id {supa_game_id}.")
                except Exception as e_upd:
                    print(f" Error updating game_id {supa_game_id}: {e_upd}")
            else:
                print(f" -> No pitcher data found in lookup for game_id {supa_game_id}.")

        except Exception as e_loop:
            print(f"Error processing Supabase game {supa_game_id}: {e_loop}")

    print(f"--- Finished FanGraphs pitcher update for {target_date_et}. Updated {updated_count} record(s). ---")
    return updated_count


if __name__ == "__main__":
    print("Starting MLB Pitcher Update Script (using FanGraphs and Selenium)...")
    today_et_date = dt_datetime.now(ET_ZONE).date()
    update_pitchers_for_date(today_et_date)
    print("\nPitcher update script finished.")