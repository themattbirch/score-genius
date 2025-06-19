# backend/scripts/debug_handedness.py

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

# --- Configure logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports ---
try:
    from backend import config
    from supabase import create_client, Client
    from backend.mlb_features.utils import normalize_team_name, determine_season
    from backend.mlb_features.handedness_for_display import transform as handedness_transform
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    sys.exit(1)


def debug_hands_for_game(game_id: str):
    logger.info(f"--- Starting Handedness Debug for game_id: {game_id} ---")

    # 1. INITIALIZE SUPABASE CLIENT
    try:
        sb_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return

    # 2. FETCH THE REQUIRED DATA (mimicking make_mlb_snapshots.py)
    # First, get the game data, which contains the pitcher hands
    game_response = sb_client.table("mlb_game_schedule").select("*").eq("game_id", game_id).execute()
    game_df = pd.DataFrame(game_response.data)
    if game_df.empty:
        logger.error(f"FATAL: Could not find game_id '{game_id}' in mlb_game_schedule.")
        return
    
    # Second, determine the season and fetch the pitcher splits data for that season
    source_date_col = 'game_date_et' if 'game_date_et' in game_df.columns and game_df['game_date_et'].notna().any() else 'scheduled_time_utc'
    game_date = pd.to_datetime(game_df.iloc[0][source_date_col])
    season_year = determine_season(game_date)
    
    logger.info(f"Game is in season {season_year}. Fetching pitcher splits data...")
    splits_response = sb_client.table("mlb_historical_team_stats").select(
        "team_id, team_name, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp"
    ).eq("season", season_year).execute()
    splits_df = pd.DataFrame(splits_response.data)
    
    # 3. --- CRITICAL CHECKS ---
    logger.info("--- Verifying data before passing to transform ---")

    # Check 1: Are the pitcher hands present in the game data?
    home_pitcher_hand = game_df.iloc[0].get("home_probable_pitcher_handedness")
    away_pitcher_hand = game_df.iloc[0].get("away_probable_pitcher_handedness")
    logger.info(f"Home pitcher hand from DB: '{home_pitcher_hand}'")
    logger.info(f"Away pitcher hand from DB: '{away_pitcher_hand}'")
    if pd.isna(home_pitcher_hand) or pd.isna(away_pitcher_hand):
        logger.warning("FAILURE: Pitcher handedness is missing for one or both teams in mlb_game_schedule.")

    # Check 2: Did we get any splits data for this season?
    if splits_df.empty:
        logger.error(f"FATAL FAILURE: No data found in mlb_historical_team_stats for season {season_year}.")
    else:
        logger.info(f"Found {len(splits_df)} rows of team stats for season {season_year}.")
        
        # Standardize team names for lookup
        game_df['home_team_norm'] = game_df['home_team_id'].apply(normalize_team_name)
        game_df['away_team_norm'] = game_df['away_team_id'].apply(normalize_team_name)
        splits_df['team_norm'] = splits_df['team_id'].apply(normalize_team_name)
        
        home_team = game_df.iloc[0]['home_team_norm']
        away_team = game_df.iloc[0]['away_team_norm']

        # Check 3: Do our specific teams have data in the splits table?
        home_split_data = splits_df[splits_df['team_norm'] == home_team]
        away_split_data = splits_df[splits_df['team_norm'] == away_team]

        if home_split_data.empty:
            logger.warning(f"FAILURE: Could not find splits data for home team '{home_team}'")
        else:
            logger.debug(f"Home team split data:\n{home_split_data.to_string()}")

        if away_split_data.empty:
            logger.warning(f"FAILURE: Could not find splits data for away team '{away_team}'")
        else:
            logger.debug(f"Away team split data:\n{away_split_data.to_string()}")


    # 4. EXECUTE THE ACTUAL TRANSFORM
    logger.info("--- Executing the actual handedness_transform function ---")
    # The snapshot script uses different column names, so we must alias them for the function call
    game_df.rename(columns={
        "home_probable_pitcher_handedness": "home_starter_pitcher_handedness",
        "away_probable_pitcher_handedness": "away_starter_pitcher_handedness"
    }, inplace=True)

    output_df = handedness_transform(
        df=game_df.copy(), 
        mlb_pitcher_splits_df=splits_df.copy(),
        home_team_col_param="home_team_norm",
        away_team_col_param="away_team_norm",
        debug=True
    )

    # 5. ANALYZE THE FINAL OUTPUT
    logger.info("--- Final Output Analysis ---")
    if not output_df.empty:
        final_row = output_df.iloc[0]
        home_val = final_row.get("h_team_off_avg_runs_vs_opp_hand")
        away_val = final_row.get("a_team_off_avg_runs_vs_opp_hand")

        logger.info(f"Home Off. vs Opp Hand: {home_val}")
        logger.info(f"Away Off. vs Opp Hand: {away_val}")

        if home_val == 0.0 or away_val == 0.0:
            logger.error("CONFIRMED: One or both features calculated to 0.0. Check failure warnings above.")
    else:
        logger.error("The transform returned an empty DataFrame.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_id_to_debug = str(sys.argv[1])
        debug_hands_for_game(game_id_to_debug)
    else:
        print("Usage: python backend/scripts/debug_handedness.py <game_id>")