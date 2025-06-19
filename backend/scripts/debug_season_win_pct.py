# backend/scripts/debug_season_win_pct.py

import os
import sys
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# --- Boilerplate Path Setup ---
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- Configure logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports ---
try:
    from backend import config
    from supabase import create_client, Client
    from backend.mlb_features.utils import normalize_team_name, determine_season
    from backend.mlb_features.season import transform as season_transform
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    sys.exit(1)


def debug_season_for_game(game_id: str):
    logger.info(f"--- Starting Prev Season Win% Diff Debug for game_id: {game_id} ---")

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
        # Fallback for historical games
        game_response = sb_client.table("mlb_historical_game_stats").select("*").eq("game_id", game_id).execute()
        working_df = pd.DataFrame(game_response.data)
        if working_df.empty:
            logger.error(f"FATAL: Could not find game_id '{game_id}' in either table.")
            return

    # For this feature, we need the historical TEAM stats table
    team_stats_response = sb_client.table("mlb_historical_team_stats").select("*").execute()
    team_stats_df = pd.DataFrame(team_stats_response.data)
    logger.info(f"Fetched {len(working_df)} row for the target game and {len(team_stats_df)} historical team stat rows.")

    # 3. STANDARDIZE DATA
    logger.info("Standardizing data...")
    working_df['game_id'] = working_df['game_id'].astype(str)
    working_df['home_team_norm'] = working_df['home_team_id'].apply(normalize_team_name)
    working_df['away_team_norm'] = working_df['away_team_id'].apply(normalize_team_name)
    
    source_date_col = 'game_date_et' if 'game_date_et' in working_df.columns and working_df['game_date_et'].notna().any() else 'game_date_time_utc'
    game_date = pd.to_datetime(working_df.iloc[0][source_date_col])
    working_df['season'] = determine_season(game_date)
    
    team_stats_df['team_norm'] = team_stats_df['team_id'].apply(normalize_team_name)

    # 4. --- CRITICAL CHECK: VERIFY MERGE KEYS ---
    logger.info("--- Verifying the keys for the merge operation ---")
    
    target_season = working_df.iloc[0]['season']
    target_prev_season = target_season - 1
    home_team = working_df.iloc[0]['home_team_norm']
    away_team = working_df.iloc[0]['away_team_norm']

    logger.info(f"Target Game Season: {target_season}. Looking for stats from: {target_prev_season}")
    logger.info(f"Home Team Norm: '{home_team}'")
    logger.info(f"Away Team Norm: '{away_team}'")

    available_seasons = team_stats_df['season'].unique()
    logger.info(f"Available seasons in historical_team_stats: {available_seasons}")

    if target_prev_season not in available_seasons:
        logger.error(f"FATAL FAILURE: The required season ({target_prev_season}) is NOT PRESENT in the historical team stats data.")
    else:
        logger.info(f"SUCCESS: Found data for the required season ({target_prev_season}).")
        
        # Check for home team
        home_lookup = team_stats_df[(team_stats_df['team_norm'] == home_team) & (team_stats_df['season'] == target_prev_season)]
        if home_lookup.empty:
            logger.warning(f"FAILURE: Could not find '{home_team}' in team stats for season {target_prev_season}.")
        else:
            logger.info(f"SUCCESS: Found '{home_team}' in team stats for season {target_prev_season}.")

        # Check for away team
        away_lookup = team_stats_df[(team_stats_df['team_norm'] == away_team) & (team_stats_df['season'] == target_prev_season)]
        if away_lookup.empty:
            logger.warning(f"FAILURE: Could not find '{away_team}' in team stats for season {target_prev_season}.")
        else:
            logger.info(f"SUCCESS: Found '{away_team}' in team stats for season {target_prev_season}.")


    # 5. EXECUTE THE ACTUAL TRANSFORM
    logger.info("--- Executing the actual season_transform function ---")
    output_df = season_transform(working_df.copy(), historical_team_stats_df=team_stats_df, debug=True)

    # 6. ANALYZE THE FINAL OUTPUT
    logger.info("--- Final Output Analysis ---")
    if not output_df.empty:
        final_row = output_df.iloc[0]
        home_win_pct = final_row.get("home_prev_season_win_pct")
        away_win_pct = final_row.get("away_prev_season_win_pct")
        pct_diff = final_row.get("prev_season_win_pct_diff")

        logger.info(f"Home Prev Season Win %: {home_win_pct}")
        logger.info(f"Away Prev Season Win %: {away_win_pct}")
        logger.info(f"Final Prev Season Win % Diff: {pct_diff}")

        if home_win_pct == 0.5 and away_win_pct == 0.5:
            logger.error("CONFIRMED: Both teams are using the default 0.5 win pct, leading to a 0.0 diff.")
    else:
        logger.error("The transform returned an empty DataFrame.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_id_to_debug = str(sys.argv[1])
        debug_season_for_game(game_id_to_debug)
    else:
        print("Usage: python backend/scripts/debug_season_win_pct.py <game_id>")