# backend/data_pipeline/mlb_pitcher_updater.py
# Fetches probable pitchers from FanGraphs and updates mlb_game_schedule table

import requests
from bs4 import BeautifulSoup
import time
import re
import json
import datetime
from datetime import date, timedelta, datetime as dt_datetime # Explicitly import datetime class
from zoneinfo import ZoneInfo
from dateutil import parser as dateutil_parser
from typing import Dict, Optional, Tuple, List, Any

# --- Import Supabase client and necessary config variables ---
try:
    from supabase import create_client, Client
    from config import ( # Only need Supabase creds now
        SUPABASE_URL,
        SUPABASE_ANON_KEY
    )
    print("Successfully imported config variables for pitcher updater.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import from config or supabase: {e}"); exit()
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during import: {e}"); exit()

# --- Configuration ---
SUPABASE_TABLE_NAME = "mlb_game_schedule" # Target table to UPDATE
ET_ZONE = ZoneInfo("America/New_York")
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"
FG_HEADERS = { # Define a User-Agent
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36" # Example
}
SCRAPE_DELAY_SECONDS = 5 # Be respectful to FanGraphs
# --- End Configuration ---

# --- Helper Functions ---
def normalize_team_name(name: str) -> str:
    """Normalizes team name: lowercase, handles St./St Louis, Athletics, std spacing."""
    if not name or not isinstance(name, str): return ""
    name = name.replace("St.Louis", "St Louis").replace("St.", "St")
    name = name.lower()
    if name == "athletics": name = "oakland athletics"
    # Add other known variations here if needed
    result = " ".join(name.split())
    return result

# --- FanGraphs Scraper ---
def fetch_and_parse_fangraphs_pitchers() -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Scrapes FanGraphs Probables Grid. Includes saving HTML for debugging.
    Returns a lookup dictionary:
    Key: (date_str_iso, team_name_normalized)
    Value: {'pitcher_name': str | None, 'handedness': str | None}
    """
    print(f"Attempting to scrape FanGraphs Probables Grid: {FANGRAPHS_URL}")
    pitcher_lookup: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    current_time_et = dt_datetime.now(ET_ZONE) # Use alias
    current_year = current_time_et.year
    current_month = current_time_et.month
    print(f"Using current year: {current_year} for date parsing.")

    try:
        print(f"Waiting {SCRAPE_DELAY_SECONDS} seconds before scraping FanGraphs...")
        time.sleep(SCRAPE_DELAY_SECONDS)

        response = requests.get(FANGRAPHS_URL, headers=FG_HEADERS)
        response.raise_for_status()
        print("Successfully fetched FanGraphs page.")

        # --- ADDED: Save HTML Content to File ---
        html_filename = "fangraphs_debug.html"
        try:
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"DEBUG: Saved fetched HTML content to '{html_filename}'")
        except Exception as e_save:
            print(f"WARN: Could not save debug HTML file: {e_save}")
        # --- End Added Section ---

        # Try parsing with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser") # Or try "lxml"

        # --- Find the main grid table using the CORRECT ID ---
        print("DEBUG: Searching for table with ID 'rr-table-probables' in downloaded HTML...")
        grid_table = soup.select_one("table#rr-table-probables")
        # --- End table finding ---

        if not grid_table:
             print("ERROR: Unable to locate table#rr-table-probables in the HTML received by the script.")
             print(f"      Please open '{html_filename}' and check if the table exists with that ID.")
             return pitcher_lookup # Return empty dict as table wasn't found
        else:
             print("DEBUG: Successfully located table with ID 'rr-table-probables'. Proceeding with parsing...")

        # --- Extract Headers (Dates) ---
        date_headers_raw: List[str] = []; date_objects: List[Optional[date]] = []
        header_row = grid_table.select_one("thead > tr")
        if header_row:
            header_cells = header_row.select("th"); start_index = 0
            if header_cells and header_cells[0].get_text(strip=True).lower() in ["", "team"]: start_index = 1
            date_headers_raw = [th.get_text(strip=True) for th in header_cells[start_index:]]
            for date_str_raw in date_headers_raw:
                 game_date_obj: Optional[date] = None
                 try:
                     cleaned_date_str = date_str_raw.split()[-1]
                     game_date_obj = dt_datetime.strptime(cleaned_date_str, "%m/%d").replace(year=current_year).date() # Use alias
                     if game_date_obj.month == 12 and current_month == 1: game_date_obj = game_date_obj.replace(year=current_year - 1)
                 except Exception as e_date: print(f"Warn: Could not parse date header '{date_str_raw}': {e_date}")
                 date_objects.append(game_date_obj)
        if not date_objects: print("ERROR: Could not parse any date headers."); return pitcher_lookup

        # --- Extract Rows (Teams and Pitchers) ---
        rows = grid_table.select("tbody > tr")
        print(f"Found {len(rows)} team rows in grid body. Processing...")
        for row in rows:
            cells = row.select("td"); 
            if not cells: continue
            team_cell = cells[0]; team_name_raw = team_cell.get_text(strip=True)
            team_name_norm = normalize_team_name(team_name_raw); 
            if not team_name_norm: continue
            probable_cells = cells[start_index:]
            num_cells_to_process = min(len(probable_cells), len(date_objects))
            for i in range(num_cells_to_process):
                cell = probable_cells[i]; game_date_obj = date_objects[i]
                if game_date_obj is None: continue
                game_date_iso = game_date_obj.isoformat()
                pitcher_name = None; handedness = None
                pitcher_link = cell.select_one("a"); raw_cell_text = cell.get_text(" ", strip=True)
                if pitcher_link:
                    pitcher_name = pitcher_link.get_text(strip=True)
                    hand_match = re.search(r'\(([RLS])\)', raw_cell_text);
                    if hand_match: handedness = hand_match.group(1)
                elif raw_cell_text and raw_cell_text.lower() not in ['tbd', 'ppd', '']:
                    hand_match = re.search(r'\(([RLS])\)', raw_cell_text)
                    if hand_match: handedness = hand_match.group(1); pitcher_name = raw_cell_text.replace(f"({handedness})", "").strip()
                    else: pitcher_name = raw_cell_text
                if pitcher_name:
                    pitcher_name = " ".join(pitcher_name.split())
                    lookup_key = (game_date_iso, team_name_norm)
                    pitcher_data = {"pitcher_name": pitcher_name, "handedness": handedness}
                    pitcher_lookup[lookup_key] = pitcher_data

    except requests.exceptions.RequestException as e: print(f"Error fetching FanGraphs page: {e}")
    except Exception as e: print(f"Error parsing FanGraphs page: {e}", exc_info=True)

    print(f"Finished scraping FanGraphs. Found pitcher entries for {len(pitcher_lookup)} team/date combinations.")
    return pitcher_lookup

# --- Supabase Update Logic ---
def update_pitchers_for_date(target_date_et: date):
    """Fetches pitcher data from FanGraphs and updates Supabase table."""
    print(f"\n--- Updating pitcher info using FanGraphs for date: {target_date_et} ---")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY: print("Error: Supabase URL/Key missing."); return 0

    # Step 1: Scrape FanGraphs
    fangraphs_pitcher_lookup = fetch_and_parse_fangraphs_pitchers()
    if not fangraphs_pitcher_lookup: print("No pitcher data scraped from FanGraphs."); return 0

    # Step 2: Fetch games from Supabase for the target date needing updates
    supabase_games = []
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        target_date_iso = target_date_et.isoformat()
        print(f"Querying Supabase for games on {target_date_iso} needing pitcher updates...")
        response = supabase.table(SUPABASE_TABLE_NAME)\
            .select("game_id, home_team_name, away_team_name, game_date_et")\
            .eq("game_date_et", target_date_iso)\
            .is_("home_probable_pitcher_name", None) \
            .execute() # Removed limit, process all for the day

        if hasattr(response, "data") and response.data is not None: supabase_games = response.data
        else: print(f"Supabase query error/no data: {getattr(response, 'error', 'Unknown')}"); return 0
        print(f"Found {len(supabase_games)} games in Supabase potentially needing updates for {target_date_et}.")
    except Exception as e: print(f"Error querying Supabase: {e}"); return 0
    if not supabase_games: print("No games found needing pitcher updates for this date."); return 0

    # Step 3: Match and Update records individually
    updated_count = 0
    UTC_ZONE = ZoneInfo("UTC") # Define UTC Zone for timestamp
    for game in supabase_games:
        try:
            supa_game_id = game.get("game_id"); supa_home_norm = normalize_team_name(game.get("home_team_name", "")); supa_away_norm = normalize_team_name(game.get("away_team_name", "")); supa_date_et = game.get("game_date_et")
            if not all([supa_game_id, supa_home_norm, supa_away_norm, supa_date_et]): continue

            home_lookup_key = (supa_date_et, supa_home_norm)
            home_pitcher_info = fangraphs_pitcher_lookup.get(home_lookup_key)
            away_lookup_key = (supa_date_et, supa_away_norm)
            away_pitcher_info = fangraphs_pitcher_lookup.get(away_lookup_key)

            update_payload = {}
            if home_pitcher_info:
                update_payload['home_probable_pitcher_name'] = home_pitcher_info.get('pitcher_name')
                update_payload['home_probable_pitcher_handedness'] = home_pitcher_info.get('handedness') # Populate new column
                # Leave 'home_probable_pitcher_record' as NULL unless FG provides it
            if away_pitcher_info:
                update_payload['away_probable_pitcher_name'] = away_pitcher_info.get('pitcher_name')
                update_payload['away_probable_pitcher_handedness'] = away_pitcher_info.get('handedness') # Populate new column

            if update_payload: # Only update if we found pitcher info
                print(f"Match found for Supabase game_id {supa_game_id}. Preparing update...")
                update_payload['updated_at'] = dt_datetime.now(UTC_ZONE).isoformat() # Update timestamp
                try:
                    update_response = supabase.table(SUPABASE_TABLE_NAME).update(update_payload).eq("game_id", supa_game_id).execute()
                    if hasattr(update_response, "data") and update_response.data: print(f"Success: Updated {supa_game_id}."); updated_count += 1
                    else: print(f"Warn: Update failed for {supa_game_id}? Resp: {update_response}")
                except Exception as e_upd: print(f"Error updating {supa_game_id}: {e_upd}")
        except Exception as e_loop: print(f"Error processing Supabase game {game.get('game_id', '??')}: {e_loop}")

    print(f"--- Finished FanGraphs pitcher update for {target_date_et}. Updated {updated_count} records. ---")
    return updated_count

# --- Runner Script Logic ---
if __name__ == "__main__":
    print("Starting MLB Pitcher Update Script (using FanGraphs)...")
    # Get today's date in ET timezone
    today_et_date = dt_datetime.now(ET_ZONE).date()

    # Attempt to update pitcher info for today's games
    update_pitchers_for_date(today_et_date)

    # Optional: Also attempt to update tomorrow's games, data *might* be there
    # tomorrow_et_date = today_et_date + timedelta(days=1)
    # print("\nAttempting update for tomorrow...")
    # update_pitchers_for_date(tomorrow_et_date)

    print("\nPitcher update script finished.")