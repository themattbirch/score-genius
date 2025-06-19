# backend/scripts/debug_rest_advantage.py

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
    from backend.mlb_features.rest import transform as rest_transform
    from backend.mlb_features.utils import DEFAULTS as MLB_DEFAULTS
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    sys.exit(1)


def debug_rest_for_game(game_id: str):
    logger.info(f"--- Starting Rest Advantage Debug for game_id: {game_id} ---")

    # 1. INITIALIZE SUPABASE CLIENT
    try:
        sb_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return

    # 2. FETCH DATA
    game_response = sb_client.table("mlb_game_schedule").select("game_id, game_date_et, home_team_id, away_team_id").eq("game_id", game_id).execute()
    working_df = pd.DataFrame(game_response.data)

    if working_df.empty:
        game_response = sb_client.table("mlb_historical_game_stats").select("game_id, game_date_time_utc, home_team_id, away_team_id").eq("game_id", game_id).execute()
        working_df = pd.DataFrame(game_response.data)
        if working_df.empty:
            logger.error(f"FATAL: Could not find game_id '{game_id}' in either table.")
            return

    # 2a. Historical stats for 2025 up through today
    hist_games_response = (
        sb_client
        .table("mlb_historical_game_stats")
        .select("game_id, game_date_time_utc, home_team_id, away_team_id")
        .gte("game_date_time_utc", "2025-01-01T00:00:00Z")
        .lte("game_date_time_utc", pd.Timestamp.now(tz="UTC").isoformat())
        .order("game_date_time_utc")
        .execute()
    )
    hist_games_df = pd.DataFrame(hist_games_response.data)

    # 2b. Schedule table to pick up yesterday’s games if stats lag
    sched = sb_client.table("mlb_game_schedule") \
        .select("game_id, game_date_et, home_team_id, away_team_id") \
        .gte("game_date_et", "2025-06-01") \
        .lte("game_date_et", "2025-06-17") \
        .execute()
    sched_df = pd.DataFrame(sched.data)

    # 2c. Union them (fills in any missing rows), then dedupe
    #    Note: sched_df has game_date_et but no game_date_time_utc,
    #    which our normalization step will handle.
    hist_games_df = (
        pd.concat([hist_games_df, sched_df], ignore_index=True)
          .drop_duplicates(subset="game_id")
    )
    logger.info(
        f"Fetched {len(working_df)} row(s) for the target game and "
        f"{len(hist_games_df)} historical games (2025+)."
    )
    # --- NEW CRITICAL FIX: STANDARDIZE game_id DTYPE ---
    logger.info("--- CRITICAL FIX: Standardizing game_id data types to string ---")
    logger.debug(f"working_df['game_id'] Dtype BEFORE: {working_df['game_id'].dtype}")
    logger.debug(f"hist_games_df['game_id'] Dtype BEFORE: {hist_games_df['game_id'].dtype}")

    working_df['game_id'] = working_df['game_id'].astype(str)
    hist_games_df['game_id'] = hist_games_df['game_id'].astype(str)

    logger.debug(f"working_df['game_id'] Dtype AFTER: {working_df['game_id'].dtype}")
    logger.debug(f"hist_games_df['game_id'] Dtype AFTER: {hist_games_df['game_id'].dtype}")
    # --- END FIX ---

    # 3. MIMIC THE ENGINE'S PREPARATION LOGIC
    combined_df = pd.concat([working_df, hist_games_df], ignore_index=True)

    # This check should now pass
    if game_id not in combined_df['game_id'].values:
        logger.error(f"FATAL: Game {game_id} STILL not found after concatenating. This is unexpected.")
        return
    else:
        logger.info(f"SUCCESS: Game {game_id} was found in the combined dataframe.")

    combined_df['home_team_norm'] = combined_df['home_team_id'].apply(normalize_team_name)
    combined_df['away_team_norm'] = combined_df['away_team_id'].apply(normalize_team_name)

    source_date_col = 'game_date_et' if 'game_date_et' in combined_df.columns and combined_df['game_date_et'].notna().any() else 'game_date_time_utc'
    dt_series = pd.to_datetime(combined_df[source_date_col], errors='coerce')
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize('UTC')
    combined_df['game_date_et'] = dt_series.dt.tz_convert('America/New_York').dt.normalize()
    
    combined_df.dropna(subset=['game_date_et'], inplace=True)
    final_working_df = combined_df[combined_df['game_id'] == game_id].copy()

    if final_working_df.empty:
        logger.error("The target row was dropped, likely due to a bad date. Check your database for this game's date value.")
        return

    # Normalize historical_df for rest.transform
    hist_normalized = hist_games_df.copy()
    hist_normalized['home_team_norm'] = hist_normalized['home_team_id'].apply(normalize_team_name)
    hist_normalized['away_team_norm'] = hist_normalized['away_team_id'].apply(normalize_team_name)
    source_date = 'game_date_et' if 'game_date_et' in hist_normalized.columns else 'game_date_time_utc'
    dt_hist = pd.to_datetime(hist_normalized[source_date], errors='coerce')
    if dt_hist.dt.tz is None:
        dt_hist = dt_hist.dt.tz_localize('UTC')
    hist_normalized['game_date_et'] = dt_hist.dt.tz_convert('America/New_York').dt.normalize()
    hist_normalized.dropna(subset=['game_date_et'], inplace=True)

    # Now call rest.transform with the normalized history
    output_df = rest_transform(
        df=final_working_df,
        historical_df=hist_normalized,
        debug=True
    )

    # 5. ANALYZE THE FINAL OUTPUT
    logger.info("--- Final Output Analysis ---")
    if not output_df.empty:
        final_row = output_df.iloc[0]
        rest_home = final_row.get('rest_days_home')
        rest_away = final_row.get('rest_days_away')
        rest_adv = final_row.get('rest_advantage')
        
        logger.info(f"Rest Days Home: {rest_home}")
        logger.info(f"Rest Days Away: {rest_away}")
        logger.info(f"Rest Advantage: {rest_adv}")

        DEF_REST_DAYS = float(MLB_DEFAULTS.get("rest_days", 7.0))
        if rest_home == DEF_REST_DAYS and rest_away == DEF_REST_DAYS:
            logger.error("CONFIRMED: Both teams are using the default rest days. The lookup failed.")
        else:
            logger.info("SUCCESS: Rest advantage was calculated correctly.")
    else:
        logger.error("The transform returned an empty DataFrame.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_id_to_debug = str(sys.argv[1])
        debug_rest_for_game(game_id_to_debug)
    else:
        print("Usage: python backend/scripts/debug_rest_advantage.py <game_id>")