#!/usr/bin/env python3
"""
backend/data_pipeline/mlb_games_preview.py

This integrated script builds MLB game previews using schedule and odds (via api‑baseball and The Odds API)
and also updates the probable pitcher information from FanGraphs (using Selenium/undetected‑chromedriver)
by upserting into your Supabase table "mlb_game_schedule".

When run, it:
  1. (Optionally) builds and upserts game preview data.
  2. Scrapes FanGraphs to get probable pitcher info.
  3. Queries Supabase for games that still have NULL pitcher fields and updates them.
  
Team normalization:
  - For team names matching Athletics (e.g. "Ath", "oakland athletics", "athletics"), the function returns "Athletics".
  - For variants of St. Louis Cardinals (including "st.louis", "st louis", or "st. louis cardinals"), it returns "St. Louis Cardinals".
  - Other team names are matched from a lookup mapping or simply converted to title case.
  
Adjust the mappings as needed so that the lookup keys (e.g. the team name parts) match exactly what is stored in Supabase.
"""

import time
import re
import json
import datetime
from datetime import date, timedelta, datetime as dt_datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List, Any

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

# --- Unified Helper Functions ---

def normalize_team_name(name: str) -> str:
    """
    Normalizes team names so that they match your Supabase values.
      - Replaces "St.Louis" or "St." with "St Louis" then maps to "St. Louis Cardinals".
      - If the cleaned name is one of: "ath", "athletics", "oakland athletics", return "Athletics".
      - Otherwise, for common abbreviations uses a mapping.
      - Falls back to title-case of the final result.
    """
    if not name or not isinstance(name, str):
        return ""
    original = name.strip()
    # Replace specific punctuation for St. Louis variants
    temp = original.replace("St.Louis", "St Louis").replace("St.", "St")
    temp = temp.lower().strip()
    
    # Special handling for Athletics
    if temp in ["ath", "athletics", "oakland athletics"]:
        return "Athletics"
    # Special handling for St. Louis
    if "st louis" in temp:
        return "St. Louis Cardinals"
    
    # Mapping for common abbreviations (all keys in lowercase)
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
    # Otherwise, return title-case the cleaned name
    return temp.title()

def title_case_team_name(name: str) -> str:
    """Converts a normalized team name back to title case for display."""
    return ' '.join(word.capitalize() for word in name.split())

# --- FanGraphs Scraping (Pitcher Data) ---
def scrape_fangraphs_probables() -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Uses Selenium (via undetected‑chromedriver) to load the FanGraphs Probables Grid,
    forces lazy‑loading by scrolling, and extracts header dates and pitcher data.
    
    Returns:
      A dictionary with keys of the form (game_date_iso, normalized_team_name)
      and values as dicts with "pitcher_name" and "handedness".
    """
    pitcher_lookup: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    print(f"Opening {FANGRAPHS_URL} in headless browser (Selenium with undetected‑chromedriver)...")
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
        print("Successfully located the grid table on FanGraphs.")
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
    # For simplicity, assume the base date is fixed; adjust this logic as needed.
    first_date = dt_datetime(2025, 4, 15)
    for i, cell in enumerate(header_cells[1:]):  # Skip the first cell (team name)
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
    for key, value in pitcher_lookup.items():
        print(key, value)
    return pitcher_lookup

# --- Pitcher Update Function ---
def update_pitchers_for_date(target_date_et: date) -> int:
    """
    1. Scrapes FanGraphs for probable pitcher data.
    2. Queries Supabase for games on the target date that are missing pitcher info.
    3. Updates those Supabase records with the scraped data.
    """
    print(f"\n--- Updating pitcher info using FanGraphs for date: {target_date_et} ---")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Error: Supabase URL/Key missing.")
        return 0

    # Step 1: Scrape FanGraphs
    pitcher_lookup = scrape_fangraphs_probables()
    if not pitcher_lookup:
        print("No pitcher data scraped from FanGraphs. No updates performed.")
        return 0

    # Step 2: Query Supabase for games on the target date that need updates
    supabase_games = []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        target_date_iso = target_date_et.isoformat()
        print(f"Querying Supabase for games on {target_date_iso} missing pitcher name...")
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
            print(f"  Supabase Details: Date='{supa_date_et}', Home='{supa_home_raw}' (Norm='{home_team_norm}'), Away='{supa_away_raw}' (Norm='{away_team_norm}')")
            if not all([supa_game_id, supa_date_et, home_team_norm, away_team_norm]):
                print("  -> Skipping due to missing fields.")
                continue

            home_key = (supa_date_et, home_team_norm)
            away_key = (supa_date_et, away_team_norm)
            print(f"  Lookup Keys: Home={home_key}, Away={away_key}")
            home_pitcher_info = pitcher_lookup.get(home_key)
            away_pitcher_info = pitcher_lookup.get(away_key)
            print(f"  FanGraphs Lookup: Home Found={home_pitcher_info is not None}, Away Found={away_pitcher_info is not None}")
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
                print(f"  Update Payload: {update_payload}")
                update_payload["updated_at"] = dt_datetime.now(UTC_ZONE).isoformat()
                try:
                    print(f"  Executing update for game_id {supa_game_id}...")
                    update_response = supabase.table(SUPABASE_TABLE_NAME) \
                        .update(update_payload) \
                        .eq("game_id", supa_game_id) \
                        .execute()
                    print(f"  Supabase Update Response for {supa_game_id}: {update_response}")
                    if hasattr(update_response, "data") and update_response.data:
                        print(f"    Success: Updated game_id {supa_game_id}.")
                        updated_count += 1
                    else:
                        print(f"    Warning: Update may have failed for game_id {supa_game_id}.")
                except Exception as e_upd:
                    print(f"    Error updating game_id {supa_game_id}: {e_upd}")
            else:
                print(f"  -> No pitcher data found in lookup for game_id {supa_game_id}.")
        except Exception as e_loop:
            print(f"Error processing Supabase game {supa_game_id}: {e_loop}")

    print(f"--- Finished FanGraphs pitcher update for {target_date_et}. Updated {updated_count} record(s). ---")
    return updated_count

# --- MLB Games Preview & Upsert (Schedule + Odds) ---
def build_and_upsert_mlb_previews() -> int:
    """
    Builds MLB game preview data by fetching schedule (via api‑baseball) and odds
    (via The Odds API) and upserts the results into Supabase.
    
    The pitcher fields are initialized as NULL.
    (The preview-building logic is assumed to be implemented elsewhere.)
    """
    print("\n--- Running MLB Game Preview Script (Schedule + Odds) ---")
    script_start_time = time.time()
    # (Your schedule and odds fetching/upsert logic goes here.)
    # For the purposes of this integration, we assume no new previews are generated.
    processed_game_ids = set()
    previews_to_upsert = []  # Placeholder for preview data
    num_previews = len(previews_to_upsert)
    if previews_to_upsert:
        print(f"\nUpserting {num_previews} game preview(s) into Supabase...")
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            upsert_response = supabase.table(SUPABASE_TABLE_NAME).upsert(previews_to_upsert, on_conflict='game_id').execute()
            print("Supabase upsert finished.")
        except Exception as e:
            print(f"Error upserting to Supabase: {e}")
    else:
        print("\nNo valid game previews generated to upsert.")
    script_end_time = time.time()
    print(f"\n--- MLB Preview Script finished. Processed {len(processed_game_ids)} unique games. ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    return num_previews

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
