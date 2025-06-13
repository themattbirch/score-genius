# backend/mlb_features/make_mlb_snapshots.py
"""
Generate and upsert per-game MLB feature snapshots for frontend display.
Orchestrates data fetching, runs the MLB Python feature pipeline for core features,
computes display-specific features (e.g., handedness), integrates data from specific RPCs,
and assembles payloads for headlines, bar, radar, and pie charts.
"""
import os
import sys
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import pandas as pd
from supabase import create_client, Client
from dateutil import parser as dateutil_parser # For robust date parsing
import numpy as np # For pd.NA handling

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Path Setup ---
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- Config and Utility Imports ---
try:
    from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL/key missing in make_mlb_snapshots.py")

# --- Import MLB Feature Engine and specific modules ---
try:
    from backend.mlb_features.engine import run_mlb_feature_pipeline
    from backend.mlb_features.utils import normalize_team_name, determine_season, DEFAULTS as MLB_DEFAULTS
    from backend.mlb_features.handedness_for_display import transform as handedness_transform # New module
    logger.info("Successfully imported MLB engine and feature modules.")
except ImportError as e:
    logger.error(f"Could not import MLB feature dependencies: {e}")
    logger.error("Please ensure backend/mlb_features/engine.py, utils.py, handedness_for_display.py exist and are correctly defined.")
    sys.exit(1)

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Supabase Client ---
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Fetch Helpers ---
def fetch_table(table_name: str, match_criteria: Dict[str, Any], select_cols: str = "*") -> pd.DataFrame:
    logger.debug(f"Fetching from table '{table_name}' with criteria: {match_criteria}")
    try:
        response = sb_client.table(table_name).select(select_cols).match(match_criteria).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Error fetching from table '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

def fetch_mlb_full_history() -> pd.DataFrame:
    """Entire historical game stats for H2H and rolling features for MLB."""
    logger.debug("Fetching full MLB historical game stats...")
    response = sb_client.table("mlb_historical_game_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    # Ensure date column is parsed for form strings and filtering
    if 'game_date_time_utc' in df.columns:
        df['parsed_game_date'] = pd.to_datetime(df['game_date_time_utc'], errors='coerce').dt.date
    elif 'game_date_et' in df.columns:
        df['parsed_game_date'] = pd.to_datetime(df['game_date_et'], errors='coerce').dt.date
    logger.debug(f"Fetched {len(df)} rows from mlb_historical_game_stats.")
    return df

def fetch_mlb_team_season_stats() -> pd.DataFrame:
    """All teams' season stats from mlb_historical_team_stats for MLB."""
    logger.debug("Fetching MLB historical team season stats...")
    response = sb_client.table("mlb_historical_team_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    logger.debug(f"Fetched {len(df)} rows from mlb_historical_team_stats.")
    return df

def fetch_mlb_pitcher_splits_data(season_year: int) -> pd.DataFrame:
    """Fetches pitcher splits data for a given season from mlb_historical_team_stats."""
    # Assuming mlb_historical_team_stats contains season_avg_runs_vs_lhp/rhp
    # This table has the required columns: team_id, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp
    logger.debug(f"Fetching MLB pitcher splits data for season: {season_year}")
    response = sb_client.table("mlb_historical_team_stats").select(
        "team_id, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp"
    ).eq("season", season_year).execute()
    df = pd.DataFrame(response.data or [])
    if df.empty:
        logger.warning(f"No MLB pitcher splits found for season {season_year} in mlb_historical_team_stats.")
    return df

# --- Helper Function _get_mlb_team_form_string ---
def _get_mlb_team_form_string(
    team_id_to_check: Union[str, int],
    current_game_date_et: pd.Timestamp | Any, # Can be date object or string for parsing
    all_historical_games: pd.DataFrame,
    num_form_games: int = 5
) -> str:
    team_id_str = str(team_id_to_check)
    
    current_game_date_obj = pd.to_datetime(current_game_date_et, errors='coerce').date()

    if pd.isna(current_game_date_obj) or all_historical_games.empty:
        logger.debug(f"Cannot get form for {team_id_str}: invalid date or empty historical games.")
        return "N/A"

    # Ensure parsed_date_for_form is created and used consistently
    if 'parsed_date_for_form' not in all_historical_games.columns:
        if 'game_date_et' in all_historical_games.columns:
            all_historical_games['parsed_date_for_form'] = pd.to_datetime(all_historical_games['game_date_et'], errors='coerce').dt.date
        elif 'game_date_time_utc' in all_historical_games.columns:
            all_historical_games['parsed_date_for_form'] = pd.to_datetime(all_historical_games['game_date_time_utc'], errors='coerce').dt.date # Keep as date only for comparison
        else:
            logger.warning("No suitable date column found in all_historical_games for form string generation.")
            return "N/A"
    
    # Ensure team_id columns are string for consistent comparison
    if 'home_team_id' in all_historical_games.columns:
        all_historical_games['home_team_id_str'] = all_historical_games['home_team_id'].astype(str)
    if 'away_team_id' in all_historical_games.columns:
        all_historical_games['away_team_id_str'] = all_historical_games['away_team_id'].astype(str)

    team_games = all_historical_games[
        ((all_historical_games['home_team_id_str'] == team_id_str) | \
         (all_historical_games['away_team_id_str'] == team_id_str)) & \
        (all_historical_games['parsed_date_for_form'] < current_game_date_obj) & \
        (all_historical_games['status_short'].isin(['FT', 'F']) | all_historical_games['status_long'] == 'Finished') & \
        (pd.notna(all_historical_games['home_score']) & pd.notna(all_historical_games['away_score'])) # Ensure scores are present for result
    ].copy()

    if team_games.empty:
        logger.debug(f"No completed games found for {team_id_str} before {current_game_date_obj} for form string.")
        return "N/A"

    recent_games = team_games.sort_values(by='parsed_date_for_form', ascending=False).head(num_form_games)

    if recent_games.empty:
        logger.debug(f"Not enough recent completed games found for {team_id_str} for form string.")
        return "N/A"

    form_results = []
    for _, game_row in recent_games.sort_values(by='parsed_date_for_form', ascending=True).iterrows():
        is_home = str(game_row['home_team_id_str']) == team_id_str
        team_score = pd.to_numeric(game_row['home_score'] if is_home else game_row['away_score'], errors='coerce')
        opponent_score = pd.to_numeric(game_row['away_score'] if is_home else game_row['home_score'], errors='coerce')
        
        if pd.isna(team_score) or pd.isna(opponent_score):
            form_results.append("?") # Indicate unknown result if scores are missing
        elif team_score > opponent_score:
            form_results.append("W")
        elif team_score < opponent_score:
            form_results.append("L")
        else:
            form_results.append("T") # MLB can have ties in rare cases before extra innings, or if game called due to weather/etc.

    return "".join(form_results) if form_results else "N/A"


# --- Snapshot Generator ---
def make_mlb_snapshot(
    game_id: Union[str, int],
    # Default column names used in this function that access df_game (from schedule/historical)
    input_game_date_col: str = "game_date_et", # Primary date col in mlb_game_schedule
    input_game_date_utc_col: str = "game_date_time_utc", # Fallback date col, typically in mlb_historical_game_stats
    input_home_team_id_col: str = "home_team_id",
    input_away_team_id_col: str = "away_team_id",
    input_home_pitcher_hand_col: str = "home_starter_pitcher_handedness", # Updated from home_probable_pitcher_handedness
    input_away_pitcher_hand_col: str = "away_starter_pitcher_handedness", # Updated from away_probable_pitcher_handedness
    input_home_score_col: str = "home_score", # From mlb_historical_game_stats
    input_away_score_col: str = "away_score",
    ):
    game_id_str = str(game_id)
    logger.info(f"--- Generating MLB Snapshot for game_id: {game_id_str} ---")

    # 1) Load raw game data: Try historical stats (completed games) first
    # Ensure all columns required by features (like inning scores, pitcher handedness) are selected
    game_data_cols_select = "*, h_inn_1, h_inn_2, h_inn_3, h_inn_4, h_inn_5, h_inn_6, h_inn_7, h_inn_8, h_inn_9, h_inn_extra, " \
                            "a_inn_1, a_inn_2, a_inn_3, a_inn_4, a_inn_5, a_inn_6, a_inn_7, a_inn_8, a_inn_9, a_inn_extra, " \
                            "home_starter_pitcher_handedness, away_starter_pitcher_handedness"
    df_game = fetch_table("mlb_historical_game_stats", {"game_id": game_id_str}, select_cols=game_data_cols_select)
    
    is_historical_game = not df_game.empty 
    
    if not is_historical_game:
        df_game = fetch_table("mlb_game_schedule", {"game_id": game_id_str})
        if not df_game.empty:
            logger.debug(f"Found MLB game {game_id_str} in mlb_game_schedule.")
            # Add placeholder columns for pre-game, to avoid key errors in feature pipeline
            # These columns are expected by the feature engine and later by the display-only handedness module
            for col in [
                input_home_score_col, input_away_score_col, 
                input_home_pitcher_hand_col, input_away_pitcher_hand_col,
                'h_inn_1', 'h_inn_2', 'h_inn_3', 'h_inn_4', 'h_inn_5', 'h_inn_6', 'h_inn_7', 'h_inn_8', 'h_inn_9', 'h_inn_extra',
                'a_inn_1', 'a_inn_2', 'a_inn_3', 'a_inn_4', 'a_inn_5', 'a_inn_6', 'a_inn_7', 'a_inn_8', 'a_inn_9', 'a_inn_extra',
            ]:
                if col not in df_game.columns:
                    df_game[col] = pd.NA # Use pandas NA for nullable integer/float columns
        else:
            logger.error(f"No primary data found for game_id={game_id_str}. Cannot generate snapshot.")
            return

    df_game['is_historical_game'] = is_historical_game

    # Determine game date and season
    current_game_date_et = None
    if input_game_date_col in df_game.columns and pd.notna(df_game[input_game_date_col].iloc[0]):
        current_game_date_et = pd.to_datetime(df_game[input_game_date_col].iloc[0], errors='coerce').date()
    elif input_game_date_utc_col in df_game.columns and pd.notna(df_game[input_game_date_utc_col].iloc[0]):
        try:
            dt_obj_utc = pd.to_datetime(df_game[input_game_date_utc_col].iloc[0], errors='coerce')
            if pd.isna(dt_obj_utc): # Handle cases where to_datetime yields NaT
                current_game_date_et = None
            else:
                # If the datetime object is naive (no timezone info), localize it to UTC.
                # If it's already tz-aware, tz_localize will raise an error, so handle it.
                if dt_obj_utc.tz is None: 
                    dt_obj_utc = dt_obj_utc.tz_localize('UTC')
                # Now convert to the target timezone (ET) and get the date part
                current_game_date_et = dt_obj_utc.tz_convert('America/New_York').date()
        except Exception as e:
            logger.error(f"Error parsing {input_game_date_utc_col} for {game_id_str}: {e}", exc_info=True) # Added exc_info=True for full traceback

    if current_game_date_et is None:
        logger.error(f"No valid game date found for game_id={game_id_str}. Cannot proceed.")
        return
    
    season_year = determine_season(pd.Timestamp(current_game_date_et))


    # 2) Fetch historical context for Python feature pipeline
    df_full_history = fetch_mlb_full_history() # For H2H, Form
    df_team_stats = fetch_mlb_team_season_stats() # For Season features


    # 3) Generate Form Strings and add to df_game (needs team IDs, current date, full history)
    home_team_id_val = df_game[input_home_team_id_col].iloc[0]
    away_team_id_val = df_game[input_away_team_id_col].iloc[0]

    df_game["home_current_form"] = _get_mlb_team_form_string(
        home_team_id_val, current_game_date_et, df_full_history
    )
    df_game["away_current_form"] = _get_mlb_team_form_string(
        away_team_id_val, current_game_date_et, df_full_history
    )
    logger.debug(f"Form strings: Home: {df_game['home_current_form'].iloc[0]}, Away: {df_game['away_current_form'].iloc[0]}")


    # 4) Run Main MLB Feature Pipeline (for model features)
    logger.info(f"Running core MLB feature pipeline for game {game_id_str}...")
    # This pipeline should NOT include handedness features (advanced.py is lean)
    df_features = run_mlb_feature_pipeline(
        df_game.copy(), # Pass a copy of df_game
        mlb_historical_games_df=df_full_history,
        mlb_historical_team_stats_df=df_team_stats,
        # Pass required column names for engine's internal modules like rest, h2h, advanced
        game_date_col="game_date_et",
        home_col=input_home_team_id_col,
        away_col=input_away_team_id_col,
        home_hand_col_param=input_home_pitcher_hand_col, # Passed for internal use by advanced.py if it uses it
        away_hand_col_param=input_away_pitcher_hand_col, # Passed for internal use by advanced.py if it uses it
        debug=True # Set to True for initial debugging
    )
    if df_features.empty:
        logger.error(f"MLB Feature pipeline returned empty for game_id={game_id_str}")
        return
    if len(df_features) != 1:
        logger.error(f"MLB Feature pipeline did not return 1 row for game_id={game_id_str}. Got {len(df_features)} rows.")
        if len(df_features) > 1: logger.warning("Proceeding with the first row from feature pipeline output.")
    
    row_initial_features = df_features.iloc[0] # Features generated by engine


    # 5) Fetch Pitcher Splits for Display-Only Handedness Features
    handedness_lookup_season = season_year 
    logger.debug(f"Fetching handedness data for season: {handedness_lookup_season}")
    df_pitcher_splits = fetch_mlb_pitcher_splits_data(handedness_lookup_season)

    # 6) Run Handedness Display Features (adds display-only features to df_features)
    # This runs separately AFTER the main feature pipeline, so it doesn't feed to the model.
    logger.info(f"Adding display-only handedness features for game {game_id_str}...")
    df_features_with_handedness = handedness_transform(
        df=df_features.copy(), # Pass a copy of the features DF
        mlb_pitcher_splits_df=df_pitcher_splits,
        home_team_col_param=input_home_team_id_col,
        away_team_col_param=input_away_team_id_col,
        home_pitcher_hand_col=input_home_pitcher_hand_col,
        away_pitcher_hand_col=input_away_pitcher_hand_col,
        debug=True
    )
    if df_features_with_handedness.empty or len(df_features_with_handedness) != 1:
        logger.warning(f"Handedness transform returned unexpected shape. Proceeding with features before handedness. Shape: {df_features_with_handedness.shape}")
        row = row_initial_features # Fallback to original features if handedness transform fails
    else:
        row = df_features_with_handedness.iloc[0] # Use features including handedness

        # --- ADD THESE LOGS TO INSPECT FEATURE VALUES ---
    logger.info(f"Raw feature values from MLB feature pipeline for game {game_id_str}:")
    logger.info(f"  rest_advantage: {row.get('rest_advantage')}")
    logger.info(f"  form_win_pct_diff: {row.get('form_win_pct_diff')}")
    logger.info(f"  prev_season_win_pct_diff: {row.get('prev_season_win_pct_diff')}")
    logger.info(f"  matchup_home_win_pct: {row.get('matchup_home_win_pct')}")
    logger.info(f"  h_team_off_avg_runs_vs_opp_hand: {row.get('h_team_off_avg_runs_vs_opp_hand')}")
    logger.info(f"  a_team_off_avg_runs_vs_opp_hand: {row.get('a_team_off_avg_runs_vs_opp_hand')}")
    # --- END ADD ---



    # 7) Fetch Advanced Team Stats RPC for Radar (and pre-game bar)
    logger.debug(f"Fetching MLB advanced team stats for radar from RPC for season {season_year}...")
    rpc_adv_team_data = sb_client.rpc("get_mlb_advanced_team_stats_splits", {"p_season": season_year}).execute().data or []
    df_rpc_adv_team = pd.DataFrame(rpc_adv_team_data)
    
    if not df_rpc_adv_team.empty and 'team_id' in df_rpc_adv_team.columns:
        df_rpc_adv_team["team_norm"] = df_rpc_adv_team["team_id"].apply(normalize_team_name)
    else:
        logger.warning(f"RPC 'get_mlb_advanced_team_stats_splits' returned no data or no 'team_id' column for season {season_year}.")
        df_rpc_adv_team = pd.DataFrame(columns=['team_id', 'team_name', 'team_norm'])

    home_team_norm = normalize_team_name(row.get(input_home_team_id_col))
    away_team_norm = normalize_team_name(row.get(input_away_team_id_col))
    
    home_adv_rpc_data = df_rpc_adv_team[df_rpc_adv_team["team_norm"] == home_team_norm].iloc[0] if not df_rpc_adv_team[df_rpc_adv_team["team_norm"] == home_team_norm].empty else pd.Series()
    away_adv_rpc_data = df_rpc_adv_team[df_rpc_adv_team["team_norm"] == away_team_norm].iloc[0] if not df_rpc_adv_team[df_rpc_adv_team["team_norm"] == away_team_norm].empty else pd.Series()

    # 8) Fetch Game-Specific RPCs (only for data not in Python features if needed)
    # As per strategy, these are minimal/placeholder RPCs, actual data from Python features
    # get_mlb_game_bar_data is the only one we might use directly for post-game inning scores
    game_bar_rpc_data = []
    if is_historical_game:
        logger.debug(f"Fetching MLB game bar data from RPC for game {game_id_str}...")
        game_bar_rpc_result = sb_client.rpc("get_mlb_game_bar_data", {"p_game_id": game_id_str}).execute().data
        if game_bar_rpc_result and isinstance(game_bar_rpc_result, list) and game_bar_rpc_result[0] and 'bar_chart_data' in game_bar_rpc_result[0]:
            game_bar_rpc_data = game_bar_rpc_result[0]['bar_chart_data']
            logger.debug(f"Fetched {len(game_bar_rpc_data)} bars from RPC.")
        else:
            logger.warning(f"get_mlb_game_bar_data RPC did not return expected data for {game_id_str}. Falling back to defaults.")
            game_bar_rpc_data = [] # Ensure it's an empty list if RPC failed

    # 9) Build Snapshot Components (JSON Payloads in Python)
    logger.info(f"Building MLB snapshot components for game {game_id_str}...")

    # Headline Stats (from Python features)
    headlines = [
        {"label": "Rest Advantage (Home)", "value": float(row.get("rest_advantage", MLB_DEFAULTS.get("mlb_rest_days", 0.0)))},
        {"label": "Form Win% Diff", "value": round(float(row.get("form_win_pct_diff", MLB_DEFAULTS.get("mlb_form_win_pct_diff", 0.0))), 3)},
        {"label": "Prev Season Win% Diff", "value": round(float(row.get("prev_season_win_pct_diff", MLB_DEFAULTS.get("mlb_prev_season_win_pct_diff", 0.0))), 3)},
        {"label": f"H2H Home Win% (L{int(row.get('matchup_num_games',0))})", "value": round(float(row.get("matchup_home_win_pct", MLB_DEFAULTS.get("mlb_h2h_home_win_pct", 0.0))), 3)},
        # Handedness features (now from handedness_for_display.py output)
        {"label": "Home Off. vs Opp Hand", "value": round(float(row.get("h_team_off_avg_runs_vs_opp_hand", MLB_DEFAULTS.get("mlb_avg_runs_vs_hand", 0.0))), 2)},
        {"label": "Away Off. vs Opp Hand", "value": round(float(row.get("a_team_off_avg_runs_vs_opp_hand", MLB_DEFAULTS.get("mlb_avg_runs_vs_hand", 0.0))), 2)},
    ]
    # Rounding for headlines
    for h in headlines:
        if isinstance(h['value'], (float, int)) and h['label'] != "Rest Advantage (Home)":
            h['value'] = round(h['value'], 2 if 'Opp Hand' in h['label'] else 1)
        elif h['label'] == "Rest Advantage (Home)":
            h['value'] = int(h['value']) # Ensure integer for days

    # Bar Chart Data (Inning Scores for post-game from RPC, Averages for pre-game from RPC)
    bar_chart_data = []
    if is_historical_game and game_bar_rpc_data: # Use data from RPC for post-game innings
        bar_chart_data = game_bar_rpc_data
        logger.debug(f"Bar chart (post-game, from RPC): {bar_chart_data}")
    else: # Pre-game, use average runs for/against from get_mlb_advanced_team_stats_splits RPC
        home_r_for = home_adv_rpc_data.get('runs_for_avg_overall', MLB_DEFAULTS.get('mlb_avg_runs_for', 0.0))
        away_r_for = away_adv_rpc_data.get('runs_for_avg_overall', MLB_DEFAULTS.get('mlb_avg_runs_for', 0.0))
        home_r_against = home_adv_rpc_data.get('runs_against_avg_overall', MLB_DEFAULTS.get('mlb_avg_runs_against', 0.0))
        away_r_against = away_adv_rpc_data.get('runs_against_avg_overall', MLB_DEFAULTS.get('mlb_avg_runs_against', 0.0))

        bar_chart_data = [
            {"name": "Avg Runs For", "Home": round(float(home_r_for), 2), "Away": round(float(away_r_for), 2)},
            {"name": "Avg Runs Against", "Home": round(float(home_r_against), 2), "Away": round(float(away_r_against), 2)},
        ]
        logger.debug(f"Bar chart (pre-game): {bar_chart_data}")

    # Radar Chart Data (from get_mlb_advanced_team_stats_splits RPC)
    radar_payload = []
    # Map metrics from get_mlb_advanced_team_stats_splits RPC output
    # Using placeholders (0.0) as actual calculation not feasible from schema.
    # User can populate these columns in mlb_historical_team_stats with real data later.
    radar_metrics_map = {
        "AVG": "team_batting_avg", "OBP": "team_on_base_pct", "SLG": "team_slugging_pct",
        "ERA": "team_era", "WHIP": "team_whip",
        "Win %": "win_pct_overall", # Added existing stat from RPC
        "Runs/G": "runs_for_avg_overall" # Added existing stat from RPC
    }
    # Default values for radar chart if RPC data is missing
    default_radar_values = {
        "AVG": 0.0, "OBP": 0.0, "SLG": 0.0, "ERA": 0.0, "WHIP": 0.0, # Placeholders
        "Win %": 0.5, "Runs/G": 4.5
    }

    for display_metric, rpc_col_name in radar_metrics_map.items():
        home_val = float(home_adv_rpc_data.get(rpc_col_name, default_radar_values[display_metric]))
        away_val = float(away_adv_rpc_data.get(rpc_col_name, default_radar_values[display_metric]))
        
        round_digits = 3 if display_metric in ["AVG", "OBP", "SLG", "Win %"] else 2
            
        radar_payload.append({
            'metric': display_metric, 
            'home_value': round(home_val, round_digits), 
            'away_value': round(away_val, round_digits)
        })
    logger.debug(f"Radar chart: {radar_payload}")

    # Pie Chart Data (Home Team's scoring contribution - simple proxy)
    pie_payload = []
    if is_historical_game and pd.notna(row.get(input_home_score_col)): # Post-game logic
        home_runs = float(row.get(input_home_score_col, 0))
        home_hits = float(row.get('home_hits', 0)) # From mlb_historical_game_stats
        home_errors = float(row.get('home_errors', 0)) # From mlb_historical_game_stats
        
        # Simple breakdown: Hits, Errors, and Remaining Runs as a proxy for other scoring
        # This is a creative adaptation due to limited raw score breakdown.
        pie_payload = [
            {"category": "Runs via Hits", "value": int(max(0, home_hits)), "color": "#4ade80"},
            {"category": "Runs via Errors", "value": int(max(0, home_errors)), "color": "#fbbf24"},
            {"category": "Other Runs", "value": int(max(0, home_runs - home_hits - home_errors)), "color": "#60a5fa"},
        ]
        # Filter out categories with 0 value if preferred for display
        pie_payload = [item for item in pie_payload if item['value'] > 0]
        if not pie_payload: # If all values are zero
            pie_payload = [{"category": "No Scoring Data", "value": 1, "color": "#cccccc"}]
        logger.debug(f"Pie chart (post-game): {pie_payload}")
    else: # Pre-game, use handedness offensive runs as requested
        h_off_vs_opp_hand = float(row.get("h_team_off_avg_runs_vs_opp_hand", MLB_DEFAULTS.get('mlb_avg_runs_vs_hand',0.0)))
        
        pie_payload = [
            {"category": f"Home Off. vs Opp Pitch Hand Avg Runs", "value": round(h_off_vs_opp_hand, 2), "color": "#60a5fa"},
            {"category": "Season Avg Runs For", "value": round(float(home_adv_rpc_data.get('runs_for_avg_overall', MLB_DEFAULTS.get('mlb_avg_runs_for', 0.0))), 2), "color": "#4ade80"},
        ]
        # Ensure values for pie chart sum to something if displayed as percentage (simple avg here, not percentage)
        # If this is to be a true "pie" for comparison, maybe sum to 100 or another meaningful total.
        # For now, it's just two slices of raw runs.
        logger.debug(f"Pie chart (pre-game, handedness focus): {pie_payload}")


    # 10) Upsert into Supabase
    snapshot_payload_final = {
        "game_id": game_id_str,
        "game_date": current_game_date_et.isoformat(),
        "season": str(season_year), # Use the integer season year
        "headline_stats": headlines,
        "bar_chart_data": bar_chart_data,
        "radar_chart_data": radar_payload,
        "pie_chart_data": pie_payload,
        "last_updated": pd.Timestamp.utcnow().isoformat()
    }
    
    logger.info(f"Upserting MLB snapshot for game_id: {game_id_str}")
    upsert_response = sb_client.table("mlb_snapshots").upsert(snapshot_payload_final, on_conflict="game_id").execute()
    
    if hasattr(upsert_response, 'error') and upsert_response.error:
        logger.error(f"MLB Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
        logger.error(f"Supabase response: {upsert_response}")
    elif hasattr(upsert_response, 'data') and not upsert_response.data and not (hasattr(upsert_response, 'count') and upsert_response.count is not None and upsert_response.count > 0) :
        logger.warning(f"MLB Snapshot upsert for game_id={game_id_str} may have had an issue (no data/count returned). Response: {upsert_response}")
    else:
        logger.info(f"✅ MLB Snapshot upserted for game_id={game_id_str}")

# --- CLI entry point ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python backend/mlb_features/make_mlb_snapshots.py <game_id1> [<game_id2> ...]")
        logger.info("Example: python backend/mlb_features/make_mlb_snapshots.py YOUR_MLB_GAME_ID_HERE")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_mlb_snapshot(game_id_arg)
            except Exception as e_main:
                logger.error(f"CRITICAL ERROR processing game_id {game_id_arg} in snapshot generation: {e_main}", exc_info=True)