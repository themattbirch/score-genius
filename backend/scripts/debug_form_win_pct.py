# backend/scripts/debug_form_win_pct.py

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# --- Boilerplate Path Setup ---
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- Configure logging to see detailed output ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Imports ---
try:
    from backend import config
    from supabase import create_client, Client
    from backend.mlb_features.utils import normalize_team_name
    from backend.mlb_features.form import transform as form_transform
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    sys.exit(1)


def debug_form_for_game(game_id: str):
    logger.info(f"--- Starting Form Win% Diff Debug for game_id: {game_id} ---")

    # 1. INITIALIZE SUPABASE CLIENT
    try:
        sb_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        logger.info("Successfully initialized Supabase client.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return

    # 2. FETCH DATA
    game_response = sb_client.table("mlb_game_schedule").select("*").eq("game_id", game_id).execute()
    working_df = pd.DataFrame(game_response.data)
    if working_df.empty:
        game_response = sb_client.table("mlb_historical_game_stats").select("*").eq("game_id", game_id).execute()
        working_df = pd.DataFrame(game_response.data)
        if working_df.empty:
            logger.error(f"FATAL: Could not find game_id '{game_id}' in either table.")
            return

    # For form, we need the full historical stats table
    hist_games_response = sb_client.table("mlb_historical_game_stats").select("*").execute()
    hist_games_df = pd.DataFrame(hist_games_response.data)
    logger.info(f"Fetched {len(working_df)} row for the target game and {len(hist_games_df)} historical games.")

    # 3. STANDARDIZE DATA (mimicking engine.py)
    logger.info("Standardizing data types and formats...")
    for df in [working_df, hist_games_df]:
        df['game_id'] = df['game_id'].astype(str)
        df['home_team_norm'] = df['home_team_id'].apply(normalize_team_name)
        df['away_team_norm'] = df['away_team_id'].apply(normalize_team_name)
        source_date_col = 'game_date_et' if 'game_date_et' in df.columns and df['game_date_et'].notna().any() else 'game_date_time_utc'
        dt_series = pd.to_datetime(df[source_date_col], errors='coerce')
        if dt_series.dt.tz is None: dt_series = dt_series.dt.tz_localize('UTC')
        df['game_date_et'] = dt_series.dt.tz_convert('America/New_York').dt.normalize()

    working_df.dropna(subset=['game_date_et'], inplace=True)
    hist_games_df.dropna(subset=['game_date_et', 'home_score', 'away_score'], inplace=True)


    # 4. --- INVESTIGATION POINT ---
    logger.info("--- Replicating form.py logic for diagnostics ---")

    if working_df.empty:
        logger.error("Target game data was dropped during cleaning (likely bad date). Cannot proceed.")
        return

    game_row = working_df.iloc[0]
    game_date = game_row["game_date_et"]
    home_team = game_row["home_team_norm"]
    away_team = game_row["away_team_norm"]
    logger.info(f"Analyzing Game: {home_team} vs {away_team} on {game_date.date()}")

    # Filter historical games for each team BEFORE the current game's date
    home_hist_df = hist_games_df[
        ((hist_games_df['home_team_norm'] == home_team) | (hist_games_df['away_team_norm'] == home_team)) &
        (hist_games_df['game_date_et'] < game_date)
    ].copy()

    away_hist_df = hist_games_df[
        ((hist_games_df['home_team_norm'] == away_team) | (hist_games_df['away_team_norm'] == away_team)) &
        (hist_games_df['game_date_et'] < game_date)
    ].copy()

    # NEW DEBUG LOGS: history counts
    logger.debug(f"History count for home '{home_team}': {len(home_hist_df)}")
    logger.debug(f"History count for away '{away_team}': {len(away_hist_df)}")

    logger.info(f"Found {len(home_hist_df)} historical games for home team '{home_team}' before {game_date.date()}.")
    if len(home_hist_df) > 0:
        logger.debug(f"Last 5 games for {home_team}:\n{home_hist_df.sort_values('game_date_et', ascending=False).head().to_string()}")
    else:
        logger.warning(f"FAILURE: No historical games found for home team '{home_team}'. Form string will be 'N/A'.")


    logger.info(f"Found {len(away_hist_df)} historical games for away team '{away_team}' before {game_date.date()}.")
    if len(away_hist_df) > 0:
         logger.debug(f"Last 5 games for {away_team}:\n{away_hist_df.sort_values('game_date_et', ascending=False).head().to_string()}")
    else:
        logger.warning(f"FAILURE: No historical games found for away team '{away_team}'. Form string will be 'N/A'.")


    # 5. EXECUTE THE ACTUAL TRANSFORM
    logger.info("--- Executing the actual form_transform function ---")
    output_df = form_transform(working_df.copy(), historical_df=hist_games_df, debug=True)

    # 6. ANALYZE THE FINAL OUTPUT
    logger.info("--- Final Output Analysis ---")
    if not output_df.empty:
        final_row = output_df.iloc[0]
        home_form = final_row.get("home_current_form")
        away_form = final_row.get("away_current_form")
        form_diff = final_row.get("form_win_pct_diff")

        logger.info(f"Generated Home Form String: '{home_form}'")
        logger.info(f"Generated Away Form String: '{away_form}'")
        logger.info(f"Final Form Win% Diff: {form_diff}")

        if home_form == "N/A" and away_form == "N/A":
            logger.error("CONFIRMED: Both form strings are 'N/A', leading to a 0.0 diff.")
    else:
        logger.error("The transform returned an empty DataFrame.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_id_to_debug = str(sys.argv[1])
        debug_form_for_game(game_id_to_debug)
    else:
        print("Usage: python backend/scripts/debug_form_win_pct.py <game_id>")
