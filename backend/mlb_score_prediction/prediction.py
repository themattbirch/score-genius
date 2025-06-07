# backend/mlb_score_prediction/prediction.py

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import pytz  # Retained for timezone consistency if needed
import pandas as pd
import numpy as np
import math
import json
import argparse
from typing import List, Dict, Optional, Any, Tuple

import logging
# Send all DEBUG+ messages to console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, # Keep this as your default for the root logger
)
logger = logging.getLogger() # Logger for the current script (__main__)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("prediction_run.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
)
logger.addHandler(file_handler)

# (Optionally still add a StreamHandler for console, if you want both)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(file_handler.formatter)
logger.addHandler(stream_handler)


# Define Paths Consistently
from backend import config

MODELS_DIR = Path(config.MAIN_MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR_PRED = Path(config.MAIN_MODELS_DIR).parent / "reports_mlb"
REPORTS_DIR_PRED.mkdir(parents=True, exist_ok=True)
logger.info(f"MLB prediction.py using model directory: {MODELS_DIR}")

# --- Third‐party clients & config ---
from supabase import create_client, Client as SupabaseClient  # Explicitly import Client

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")  # Retain if used, though ET/UTC more common
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB = 2190
DEFAULT_UPCOMING_DAYS_WINDOW_MLB = 7  # Predict for a week

ENSEMBLE_WEIGHTS_FILENAME_MLB = "mlb_ensemble_weights_optimized.json"  # MLB specific
# Fallback weights if file not found, keys are simple model names
FALLBACK_ENSEMBLE_WEIGHTS_MLB: Dict[str, float] = {"svr": 0.6, "ridge": 0.4}

# Update with columns from your mlb_historical_game_stats
REQUIRED_HISTORICAL_COLS_MLB = [
    "game_id",
    "game_date_time_utc",
    "home_team_id",
    "away_team_id",
    "home_team_name",  # <- used later for display
    "away_team_name",  # <- used later for display
    "status_short",  # <- used to keep only finished games
    "home_score",
    "away_score",
    "home_hits",
    "away_hits",
    "home_errors",
    "away_errors",
    # inning splits …
    *(f"h_inn_{i}" for i in range(1, 10)),
    "h_inn_extra",
    *(f"a_inn_{i}" for i in range(1, 10)),
    "a_inn_extra",
]

# Update with columns from your mlb_historical_team_stats
REQUIRED_TEAM_STATS_COLS_MLB = [
    'team_id', 'team_name', 'season', 'league_id', 'league_name', 'games_played_home', 'games_played_away',
    'games_played_all', 'wins_home_total', 'wins_home_percentage', 'wins_away_total', 'wins_away_percentage',
    'wins_all_total', 'wins_all_percentage', 'losses_home_total', 'losses_home_percentage', 'losses_away_total',
    'losses_away_percentage', 'losses_all_total', 'losses_all_percentage', 'runs_for_total_home',
    'runs_for_total_away', 'runs_for_total_all', 'runs_for_avg_home', 'runs_for_avg_away', 'runs_for_avg_all',
    'runs_against_total_home', 'runs_against_total_away', 'runs_against_total_all', 'runs_against_avg_home',
    'runs_against_avg_away', 'runs_against_avg_all', 'updated_at', 'raw_api_response', 'season_runs_scored_vs_lhp',
    'season_games_vs_lhp', 'season_avg_runs_vs_lhp', 'season_runs_scored_vs_rhp', 'season_games_vs_rhp',
    'season_avg_runs_vs_rhp'
]

# Update with columns from your mlb_game_schedule for upcoming games
UPCOMING_GAMES_COLS_MLB = [
    "game_id",
    "scheduled_time_utc",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    # Include probable pitcher IDs/names if used directly by feature engine
    "home_probable_pitcher_name",
    "away_probable_pitcher_name",
    "home_probable_pitcher_handedness", 
    "away_probable_pitcher_handedness", 
] 
# Box score columns expected by downstream code
BOX_SCORE_COLS = [
    "home_score",
    "away_score",
    "home_hits",
    "away_hits",
    "home_errors",
    "away_errors",
    *(f"h_inn_{i}" for i in range(1, 10)),
    "h_inn_extra",
    *(f"a_inn_{i}" for i in range(1, 10)),
    "a_inn_extra",
]

# --- Project module imports (MLB specific) ---
PROJECT_MODULES_IMPORTED = False
from backend.mlb_features.engine import run_mlb_feature_pipeline
from backend.mlb_score_prediction.models import (
    RidgeScorePredictor as MLBRidgePredictor,
    SVRScorePredictor as MLBSVRScorePredictor,
    XGBoostScorePredictor as MLBXGBoostPredictor,
)

PROJECT_MODULES_IMPORTED = True

# --- Supabase helper ---
def get_supabase_client() -> Optional[SupabaseClient]:
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:  # Prioritize service key
        logger.error("Supabase URL or Service Key not configured.")
        return None
    try:
        # Using service key for operations like updates
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}", exc_info=True)
        return None


# --- Data Loading Functions (MLB specific) ---
# backend/mlb_score_prediction/prediction.py

# Make sure pandas and datetime/pytz are imported at the top of your file
import pandas as pd
from datetime import datetime, timedelta
import pytz # If not already

# (Assuming REQUIRED_HISTORICAL_COLS_MLB is defined globally or passed appropriately)
# Example definition from prior context (ensure yours is accurate):
# REQUIRED_HISTORICAL_COLS_MLB = [
#     "game_id", "game_date_time_utc", "season", "league_id", 
#     "status_long", "status_short", # These are key string columns
#     "home_team_id", "home_team_name", "away_team_id", "away_team_name",
#     "home_score", "away_score", "home_hits", "away_hits", "home_errors", "away_errors",
#     # ... inning scores ... "h_inn_1", ..., "a_inn_extra",
#     "updated_at", 
#     # "home_starter_pitcher_handedness", # These will be added via additional_raw_hist_cols
#     # "away_starter_pitcher_handedness"  # if not already in REQUIRED_HISTORICAL_COLS_MLB
# ]


def load_recent_historical_data(supabase_client: SupabaseClient, days_lookback: int) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()

    date_column_hist = "game_date_time_utc" # This is the actual column name for date filtering
    start_date_iso = (
        datetime.now(pytz.utc) - timedelta(days=days_lookback)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"Loading MLB historical game data from {start_date_iso} onwards...")

    # Define additional raw columns you want to ensure are loaded,
    # especially if they aren't in your main REQUIRED_HISTORICAL_COLS_MLB list
    # or if you want to be absolutely sure they are fetched.
    additional_raw_hist_cols = [ 
        "home_starter_pitcher_handedness",
        "away_starter_pitcher_handedness",
        "status_short", # Ensure it's selected if not already guaranteed by REQUIRED_HISTORICAL_COLS_MLB
        "status_long"   # Ensure it's selected
    ]
    
    # Combine all columns for the SELECT query, ensuring uniqueness
    # and inclusion of the primary date column for filtering.
    temp_cols_list = REQUIRED_HISTORICAL_COLS_MLB + additional_raw_hist_cols
    if date_column_hist not in temp_cols_list: # date_column_hist should already be in REQUIRED_HISTORICAL_COLS_MLB
        temp_cols_list.append(date_column_hist)
    
    final_required_hist_cols = sorted(list(set(temp_cols_list))) # Sort for consistent query string, easier debugging
    select_cols_str = ", ".join(final_required_hist_cols)
    
    all_historical_data = []
    page_size = 1000
    start_index = 0
    has_more = True

    try:
        while has_more:
            resp = (
                supabase_client.table("mlb_historical_game_stats")
                .select(select_cols_str)
                .gte(date_column_hist, start_date_iso)
                .order(date_column_hist, desc=False) # Order by the actual date column from DB
                .range(start_index, start_index + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            all_historical_data.extend(batch)
            if len(batch) < page_size:
                has_more = False
            else:
                start_index += page_size

        if not all_historical_data:
            logger.warning(f"No MLB historical game data found since {start_date_iso}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_historical_data)
        logger.info(f"Successfully loaded {len(df)} raw historical records from Supabase.")

        # --- Data Type Handling and Cleaning ---

        # 1. Handle Date Columns and derive 'season'
        # Ensure the primary date column exists before trying to use it
        if date_column_hist not in df.columns:
            logger.error(f"Critical date column '{date_column_hist}' not found in loaded data. Aborting processing.")
            return pd.DataFrame()
            
        df["game_date"] = pd.to_datetime(df[date_column_hist], errors='coerce').dt.tz_localize(None)
        df = df.dropna(subset=["game_date"]) # Crucial to do after conversion
        if df.empty:
            logger.warning(f"DataFrame became empty after processing '{date_column_hist}' and dropping NaNs.")
            return pd.DataFrame()
        df["season"] = df["game_date"].dt.year.astype(int)

        # 2. Define columns that should NOT be converted to numeric
        # These are known ID, name, text, or specific date/category columns.
        non_numeric_cols = [
            "game_id", 
            date_column_hist, # Original UTC timestamp string
            "game_date",      # Derived datetime object
            "home_team_id", 
            "away_team_id", 
            "home_team_name", 
            "away_team_name",
            "status_short",   # CRITICAL: Must remain string
            "status_long",    # CRITICAL: Must remain string
            "home_starter_pitcher_handedness", # String (L, R, or empty)
            "away_starter_pitcher_handedness", # String (L, R, or empty)
            "league_id",      # Often a string like 'AL', 'NL', or numeric ID but not for arithmetic
            "updated_at",     # Timestamp string
            # Add any other columns from REQUIRED_HISTORICAL_COLS_MLB that are definitely not numeric
        ]
        # Ensure all non_numeric_cols are treated as strings and NaNs filled appropriately for them.
        for col in non_numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("") # Convert to string, fill any NaNs with empty string
        
        # 3. Convert remaining columns (expected to be numeric) from REQUIRED_HISTORICAL_COLS_MLB
        #    to numeric, coercing errors and filling NaNs with 0.
        #    These are typically scores, hits, errors, inning scores etc.
        logger.debug(f"Attempting numeric conversion for applicable columns...")
        for col in df.columns: # Iterate over all columns in the DataFrame
            if col in final_required_hist_cols and col not in non_numeric_cols:
                # This column was requested and is not in our explicit non_numeric_cols list
                original_dtype = df[col].dtype
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Only fill with 0 if it's now a numeric type (float/int after coerce)
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(0)
                        if original_dtype == 'object' and df[col].dtype != 'object': # Log if type changed
                             logger.debug(f"Converted column '{col}' from {original_dtype} to numeric; NaNs filled with 0.")
                    else: # pd.to_numeric with errors='coerce' might leave it as object if all were unparseable
                        logger.warning(f"Column '{col}' (original_dtype: {original_dtype}) could not be fully coerced to numeric. Check its content. NaNs not filled with 0 by numeric logic.")
                        # If it's still object and you want to fill its NaNs differently (e.g. with ""), handle here
                        if df[col].dtype == 'object':
                             df[col] = df[col].fillna("")


                except Exception as e:
                    logger.error(f"Error during numeric conversion for column '{col}': {e}", exc_info=True)


        logger.info(f"Data types processed. Final historical DataFrame shape before sort: {df.shape}")
        return df.sort_values("game_date").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error in load_recent_historical_data main try block: {e}", exc_info=True)
        return pd.DataFrame()


def load_team_stats_data(supabase_client: SupabaseClient) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    logger.info("Loading MLB team stats data from 'mlb_historical_team_stats'...")
    additional_raw_team_cols = []        
    final_required_team_cols = list(set(REQUIRED_TEAM_STATS_COLS_MLB + additional_raw_team_cols))
    select_cols_str = ", ".join(final_required_team_cols)
    logger.debug(f"Supabase team stats SELECT string: {select_cols_str}")

    try:
        resp = supabase_client.table("mlb_historical_team_stats").select(select_cols_str).execute()
        if not resp.data:
            logger.warning("No MLB team stats data found.")
            return pd.DataFrame()

        df = pd.DataFrame(resp.data)
        numeric_cols = [c for c in REQUIRED_TEAM_STATS_COLS_MLB if c not in ("team_id", "team_name", "season", "league_name", "raw_api_response", "updated_at")]

        for col in numeric_cols:
            if col in df.columns:
                ## FIX: Do NOT fill NaNs with 0 here. Let them remain as NaN.
                ## The feature engineering modules are now designed to handle proper NaNs.
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # This part for non-numeric columns is fine.
        for col in ("game_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name", "status_short"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str).fillna("")

        return df
    except Exception as e:
        logger.error(f"Error loading MLB team stats: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_upcoming_games_data(supabase_client: SupabaseClient, days_window: int) -> pd.DataFrame:
    try:
        from zoneinfo import ZoneInfo  # Python ≥ 3.9

        ET = ZoneInfo("America/New_York")
        UTC = ZoneInfo("UTC")
    except (ImportError, KeyError):
        ET = pytz.timezone("America/New_York")
        UTC = pytz.utc

    today_et = datetime.now(ET).date()
    start_et = datetime(today_et.year, today_et.month, today_et.day, tzinfo=ET)
    end_et = start_et + timedelta(days=days_window)
    start_utc_str = start_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc_str = end_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    date_column_upcoming = "scheduled_time_utc"
    logger.info(f"Fetching upcoming MLB games between {start_utc_str} and {end_utc_str} (UTC)...")

    try:
        resp = (
            supabase_client.table("mlb_game_schedule")
            .select(", ".join(list(set(UPCOMING_GAMES_COLS_MLB + [date_column_upcoming]))))
            .gte(date_column_upcoming, start_utc_str)
            .lt(date_column_upcoming, end_utc_str)
            .order(date_column_upcoming, desc=False)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            logger.warning("No upcoming MLB games found.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["scheduled_time_utc_dt"] = pd.to_datetime(df[date_column_upcoming], errors="coerce", utc=True)
        df = df.dropna(subset=["scheduled_time_utc_dt"])
        df["game_date"] = pd.to_datetime(df["scheduled_time_utc_dt"].dt.tz_convert(ET).dt.date)

        for c in ("game_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name"):
            df[c] = df.get(c).astype(str).fillna("Unknown")

        return df.sort_values("scheduled_time_utc_dt").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching upcoming MLB games: {e}", exc_info=True)
        return pd.DataFrame()


# --- Model Loading (MLB specific) ---
def load_trained_models(
    model_dir: Path = MODELS_DIR,  # Uses shared MAIN_MODELS_DIR
    load_feature_list: bool = True,
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    feature_list: Optional[List[str]] = None
    features_fp = model_dir / "mlb_selected_features.json"
    if load_feature_list and features_fp.is_file():
        try:
            raw = json.loads(features_fp.read_text())
            # Deduplicate while preserving order:
            seen = set()
            deduped = []
            for f in raw:
                if f not in seen:
                    deduped.append(f)
                    seen.add(f)
            feature_list = deduped
            logger.info(f"Loaded mlb_selected_features.json with {len(feature_list)} features")
        except Exception as e:
            logger.warning(f"Could not read mlb_selected_features.json: {e}")
    elif load_feature_list:
        logger.warning(f"{features_fp.name} not found. Model might use all features or internal list.")

    models: Dict[str, Any] = {}
    model_map = {
        "svr": MLBSVRScorePredictor,
        "ridge": MLBRidgePredictor,
        "xgb": MLBXGBoostPredictor,
    }
    for name, ClsMLB in model_map.items():
        try:
            pred = ClsMLB(model_dir=str(model_dir), model_name=f"{name}_mlb_runs_predictor")
            pred.load_model()
            models[name] = pred
            logger.info(f"Loaded MLB '{name}' predictor from {model_dir}")
        except Exception as e:
            logger.error(f"Could not load MLB '{name}' model: {e}", exc_info=True)

    if not models:
        raise RuntimeError(f"No MLB models could be loaded from {model_dir}")
    return models, feature_list

# backend/mlb_score_prediction/prediction.py

# ... (other imports and global constants) ...

def categorize_selected_features(feature_list: List[str]) -> Dict[str, List[str]]:
    """Categorizes features from the selected list."""
    categorized = {
        "ha_split": [],
        "rolling": [],
        "h2h": [],
        "imputed_flags": [],
        "other": [],
    }
    if not feature_list:
        return categorized

    for feature in feature_list:
        if "hist_HA_" in feature and not feature.endswith("_imputed"):
            categorized["ha_split"].append(feature)
        elif "rolling_" in feature and not feature.endswith("_imputed"):
            categorized["rolling"].append(feature)
        elif "matchup_" in feature: # Assuming these are your H2H features
            categorized["h2h"].append(feature)
        elif feature.endswith("_imputed"):
            categorized["imputed_flags"].append(feature)
        else:
            categorized["other"].append(feature)
    return categorized


# --- Display Summary (MLB specific) ---
def display_prediction_summary_mlb(preds: List[Dict]) -> None:
    header_line = "=" * 125
    header_title = " " * 45 + "MLB PREGAME PREDICTION SUMMARY"
    header_cols = (
        f"{'DATE':<11} {'MATCHUP':<30} {'PRED RUNS':<12} {'PRED RUN DIFF':<15} {'WIN PROB (H)':<15}"
    )
    header_sep = "-" * 125

    print(f"\n{header_line}\n{header_title}\n{header_line}\n{header_cols}\n{header_sep}")
    try:
        df = pd.DataFrame(preds)
        if "game_date" in df.columns:
            df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.dropna(subset=["game_date_dt"])
            df = df.sort_values(
                ["game_date_dt", "predicted_run_diff"],
                key=lambda x: np.abs(x) if x.name == "predicted_run_diff" else x,
            )
    except Exception as e:
        logger.error(f"Error sorting for MLB display: {e}")
        df = pd.DataFrame(preds)

    for _, g in df.iterrows():
        try:
            date_str = (
                pd.to_datetime(g.get("game_date")).strftime("%Y-%m-%d")
                if pd.notna(g.get("game_date"))
                else "N/A"
            )
            matchup = f"{str(g.get('home_team_name', '?'))[:13]} vs {str(g.get('away_team_name', '?'))[:13]}"
            final_home = g.get("predicted_home_runs", np.nan)
            final_away = g.get("predicted_away_runs", np.nan)
            final_diff = g.get("predicted_run_diff", np.nan)
            winp_home = g.get("win_probability_home", np.nan)

            final_runs_str = f"{final_home:.1f}-{final_away:.1f}" if pd.notna(final_home) and pd.notna(final_away) else "N/A"
            final_diff_str = f"{final_diff:+.1f}" if pd.notna(final_diff) else "N/A"
            win_prob_str = f"{winp_home*100:.1f}%" if pd.notna(winp_home) else "N/A"

            print(f"{date_str:<11} {matchup:<30} {final_runs_str:<12} {final_diff_str:<15} {win_prob_str:<15}")
        except Exception as e:
            logger.error(f"Error displaying MLB game {g.get('game_id')}: {e}", exc_info=True)
    print(header_line)


def compute_form_for_team(df_group: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    # Renamed from train_models.py to accept window_size parameter
    df_sorted = df_group.sort_values('game_date').reset_index(drop=True)
    results = df_sorted['result'].tolist()
    forms = []
    for idx in range(len(results)):
        # Form is based on games *before* the current one
        start_idx = max(0, idx - window_size)
        past_slice = results[start_idx:idx] # Slice up to, but not including, current game
        forms.append(''.join(past_slice))
    df_sorted['form_str'] = forms # Use a distinct name like 'form_str'
    return df_sorted


def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW_MLB,
    model_dir: Path = MODELS_DIR,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB, # Will use the updated default
    debug_mode: bool = False,
) -> Tuple[List[Dict], List[Dict]]:

    logger.info("--- Starting MLB Prediction Pipeline ---")
    start_ts = time.time()

    if not PROJECT_MODULES_IMPORTED:
        logger.critical("MLB modules not imported – aborting.")
        return [], []

    supabase = get_supabase_client()
    if not supabase:
        logger.critical("Supabase client unavailable – aborting.")
        return [], []

    # ────────────────────────── 1) LOAD DATA ──────────────────────────
    # hist_df_mlb will now have a longer lookback due to DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB change
    hist_df_mlb = load_recent_historical_data(supabase, historical_lookback)
    team_stats_df = load_team_stats_data(supabase)
    upcoming_df_raw = fetch_upcoming_games_data(supabase, days_window) # Keep original upcoming_df separate for now

    logger.info(
        f"Loaded {len(hist_df_mlb)} historical rows, seasons={sorted(hist_df_mlb['season'].unique()) if not hist_df_mlb.empty else 'N/A'}"
    )
    if upcoming_df_raw.empty:
        logger.warning("No upcoming MLB games – nothing to predict.")
        return [], []

    # Keep only finished historical games (status_short==0 might be too restrictive if 0 means something else)
    # Ensure 'status_short' is loaded and correctly interpreted. For this example, assuming 'F' or 'Final'
    # This was 'status_short == 0' in your prediction.py snippet, but 'F' or 'Final' in train_models.py comments.
    # Clarify what indicates a completed game for historical data. Let's assume 'status_short' is a string like 'FT' or 'F'.
    if "status_short" in hist_df_mlb.columns:
        logger.info(f"Debugging status_short BEFORE filtering. Number of rows: {len(hist_df_mlb)}")
        logger.info(f"Data type of status_short: {hist_df_mlb['status_short'].dtype}")
        unique_statuses = hist_df_mlb['status_short'].unique()
        logger.info(f"Unique status_short values in DataFrame: {unique_statuses}")
        
        # Check specifically for 'FT' and its variations
        ft_count_exact = hist_df_mlb[hist_df_mlb['status_short'] == 'FT'].shape[0]
        logger.info(f"Count of rows where status_short == 'FT' (exact match): {ft_count_exact}")
        
        # Check for 'FT' with potential whitespace issues
        if hist_df_mlb['status_short'].dtype == 'object': # Only apply .str if it's an object/string type
            hist_df_mlb['status_short_stripped'] = hist_df_mlb['status_short'].str.strip()
            ft_count_stripped = hist_df_mlb[hist_df_mlb['status_short_stripped'] == 'FT'].shape[0]
            logger.info(f"Count of rows where status_short.str.strip() == 'FT': {ft_count_stripped}")
            unique_statuses_stripped = hist_df_mlb['status_short_stripped'].unique()
            logger.info(f"Unique status_short_stripped values in DataFrame: {unique_statuses_stripped}")
        else:
            logger.info("status_short is not of object type, skipping .str.strip().")

    # Original filtering block (ensure completed_statuses is correct now)
    if "status_short" in hist_df_mlb.columns:
        completed_statuses = ['FT'] 
        
        # OPTION 1: Use the exact value if you are confident after debugging
        # hist_df_mlb = hist_df_mlb[hist_df_mlb["status_short"].isin(completed_statuses)].copy()
        
        # OPTION 2: Or, if stripping spaces proved necessary from debug logs:
        if 'status_short_stripped' in hist_df_mlb.columns: # Check if the debug column was created
            hist_df_mlb = hist_df_mlb[hist_df_mlb["status_short_stripped"].isin(completed_statuses)].copy()
            # Optionally drop the temporary column if you keep this method
            # hist_df_mlb = hist_df_mlb.drop(columns=['status_short_stripped'], errors='ignore')
        else: # Fallback to original if stripping wasn't applied or column doesn't exist
            hist_df_mlb = hist_df_mlb[hist_df_mlb["status_short"].isin(completed_statuses)].copy()

        logger.info(f"Filtered historical games to 'FT' status (using appropriate comparison), {len(hist_df_mlb)} rows remaining.")
    else:
        logger.warning("'status_short' column not found in hist_df_mlb. Cannot filter by status.")

    assert "home_score" in hist_df_mlb.columns, "Historical DataFrame is missing 'home_score' column!"
    assert "away_score" in hist_df_mlb.columns, "Historical DataFrame is missing 'away_score' column!"
    if hist_df_mlb["home_score"].isna().any():
        raise RuntimeError("Some rows in hist_df_mlb have NaN in 'home_score' after FT‐filter. Rolling will break.")
    if hist_df_mlb["away_score"].isna().any():
        raise RuntimeError("Some rows in hist_df_mlb have NaN in 'away_score' after FT‐filter. Rolling will break.")

    if hist_df_mlb.empty:
        logger.warning("No completed historical games after filtering. Form calculation might be impaired.")
        # Decide if to proceed; for now, we will, but form will be empty.

    # Create a working copy of upcoming_df
    upcoming_df = upcoming_df_raw.copy()
    logger.info(f"DEBUG: upcoming_df INITIAL columns after copy from raw: {upcoming_df.columns.tolist()}")
    if "home_probable_pitcher_handedness" in upcoming_df.columns:
        logger.info(f"DEBUG: Sample of home_probable_pitcher_handedness BEFORE mapping:\n{upcoming_df[['home_probable_pitcher_handedness']].head().to_string(index=False)}")
    else:
        logger.info("DEBUG: home_probable_pitcher_handedness NOT in upcoming_df BEFORE mapping.")


    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # PLACE THE PITCHER HANDEDNESS MAPPING LOGIC HERE
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    logger.info("Mapping probable pitcher handedness to starter handedness for upcoming_df...")
    # Map 'probable' handedness from schedule to 'starter' handedness for the feature engine
        # Define the source and target column names
    probable_home_hand_col = "home_probable_pitcher_handedness"
    starter_home_hand_col = "home_starter_pitcher_handedness"
    probable_away_hand_col = "away_probable_pitcher_handedness"
    starter_away_hand_col = "away_starter_pitcher_handedness"
    
    if probable_home_hand_col in upcoming_df.columns:
        upcoming_df[starter_home_hand_col] = upcoming_df[probable_home_hand_col]
        logger.debug(f"Mapped '{probable_home_hand_col}' to '{starter_home_hand_col}'.")
    else:
        upcoming_df[starter_home_hand_col] = "" 
        logger.info(f"'{probable_home_hand_col}' not found, defaulting '{starter_home_hand_col}' to empty string.")

    if probable_away_hand_col in upcoming_df.columns:
        upcoming_df[starter_away_hand_col] = upcoming_df[probable_away_hand_col]
        logger.debug(f"Mapped '{probable_away_hand_col}' to '{starter_away_hand_col}'.")
    else:
        upcoming_df[starter_away_hand_col] = "" 
        logger.info(f"'{probable_away_hand_col}' not found, defaulting '{starter_away_hand_col}' to empty string.")

    # Ensure these 'starter' columns exist and are clean strings
    if starter_home_hand_col not in upcoming_df.columns: upcoming_df[starter_home_hand_col] = ""
    upcoming_df[starter_home_hand_col] = upcoming_df[starter_home_hand_col].astype(str).fillna("")
    
    if starter_away_hand_col not in upcoming_df.columns: upcoming_df[starter_away_hand_col] = ""
    upcoming_df[starter_away_hand_col] = upcoming_df[starter_away_hand_col].astype(str).fillna("")
    
    logger.info(f"DEBUG: upcoming_df columns IMMEDIATELY AFTER mapping logic: {upcoming_df.columns.tolist()}")
    cols_to_sample_handedness = [col for col in [starter_home_hand_col, starter_away_hand_col] if col in upcoming_df.columns]
    if cols_to_sample_handedness:
        logger.info(f"DEBUG: Sample of starter handedness IMMEDIATELY AFTER mapping logic:\n{upcoming_df[cols_to_sample_handedness].head().to_string(index=False)}")
    else:
        logger.warning("DEBUG: Starter handedness columns NOT FOUND for sampling immediately after mapping logic.")
    # --- PITCHER HANDEDNESS MAPPING LOGIC ENDS ---

    # Now, ensure these 'starter' columns are clean strings
    upcoming_df['home_starter_pitcher_handedness'] = upcoming_df['home_starter_pitcher_handedness'].astype(str).fillna("")
    upcoming_df['away_starter_pitcher_handedness'] = upcoming_df['away_starter_pitcher_handedness'].astype(str).fillna("")
    logger.debug(f"Sample starter handedness after mapping: \n{upcoming_df[['home_starter_pitcher_handedness', 'away_starter_pitcher_handedness']].head().to_string()}")

    if 'game_date' in upcoming_df.columns:
         upcoming_df['game_date'] = pd.to_datetime(upcoming_df['game_date'], errors='coerce')
    else:
        logger.error("'game_date' column missing in upcoming_df before form calculation. Forms will be empty.")
        # Initialize form columns if game_date is missing to prevent errors later
        upcoming_df['home_current_form'] = ""
        upcoming_df['away_current_form'] = ""


    # MODIFICATION: Pre-calculate team form strings START
    # This logic is adapted from train_models.py
    # It calculates form based on hist_df_mlb and then maps it to upcoming_df

        # --- MODIFIED FORM PRE-CALCULATION BLOCK START ---
    logger.info("Starting pre-calculation of team form strings for upcoming_df...")

    # Initialize form columns on the *existing* upcoming_df if they aren't there.
    # This upcoming_df should NOT be reassigned from upcoming_df_raw within this block.
    if 'home_current_form' not in upcoming_df.columns:
        upcoming_df['home_current_form'] = ""
    else: # If it exists, ensure it's clear for new calculation based on new hist_df_mlb
        upcoming_df['home_current_form'] = ""
        
    if 'away_current_form' not in upcoming_df.columns:
        upcoming_df['away_current_form'] = ""
    else:
        upcoming_df['away_current_form'] = ""

    # Use hist_df_mlb (which should be populated with 'FT' games) for form context
    if hist_df_mlb is not None and not hist_df_mlb.empty and 'game_date' in hist_df_mlb.columns:
        df_for_form_context = hist_df_mlb.copy() # Use a copy of hist_df_mlb for these calculations
        
        # 1. Ensure 'game_date' is datetime in the context df
        df_for_form_context['game_date'] = pd.to_datetime(df_for_form_context['game_date'], errors='coerce')
        df_for_form_context.dropna(subset=['game_date'], inplace=True) # Drop rows if date conversion failed

        if not df_for_form_context.empty:
            logger.debug(f"Form context: Using {len(df_for_form_context)} rows from hist_df_mlb.")
            # 2. Build a long DataFrame of one row per team per game from df_for_form_context
            # Ensure scores are numeric for result calculation
            for score_col in ['home_score', 'away_score']:
                if score_col in df_for_form_context.columns:
                    df_for_form_context[score_col] = pd.to_numeric(df_for_form_context[score_col], errors='coerce')
                else: 
                    logger.warning(f"Score column {score_col} missing in hist_df_mlb for form context. Results may be incorrect.")
                    df_for_form_context[score_col] = 0 

            home_part = df_for_form_context[df_for_form_context['home_team_id'].notna() & df_for_form_context['home_score'].notna() & df_for_form_context['away_score'].notna()][[
                'game_id', 'game_date', 'home_team_id', 'home_score', 'away_score'
            ]].copy().rename(columns={
                'home_team_id': 'team_id', 'home_score': 'team_score', 'away_score': 'opp_score'
            })
            home_part['result'] = np.where(home_part['team_score'] > home_part['opp_score'], 'W', 
                                         np.where(home_part['team_score'] < home_part['opp_score'], 'L', 'T'))

            away_part = df_for_form_context[df_for_form_context['away_team_id'].notna() & df_for_form_context['home_score'].notna() & df_for_form_context['away_score'].notna()][[
                'game_id', 'game_date', 'away_team_id', 'away_score', 'home_score'
            ]].copy().rename(columns={
                'away_team_id': 'team_id', 'away_score': 'team_score', 'home_score': 'opp_score'
            })
            away_part['result'] = np.where(away_part['team_score'] > away_part['opp_score'], 'W',
                                         np.where(away_part['team_score'] < away_part['opp_score'], 'L', 'T'))

            team_games_long_for_form = pd.concat([home_part, away_part], ignore_index=True)
            team_games_long_for_form.dropna(subset=['team_id', 'game_date', 'result'], inplace=True)

            if not team_games_long_for_form.empty:
                form_window_size = 5 
                logger.debug("Form context: Calculating form strings from historical team games.")
                team_games_with_form_str = (
                    team_games_long_for_form[team_games_long_for_form['team_id'].notna()]
                    .groupby('team_id', group_keys=False)
                    .apply(compute_form_for_team, window_size=form_window_size) # compute_form_for_team must be defined
                    .reset_index(drop=True)
                )
                
                if not team_games_with_form_str.empty:
                    team_games_with_form_str = team_games_with_form_str.sort_values(by=['team_id', 'game_date'])
                    logger.debug("Form context: Historical form strings calculated. Now applying to upcoming_df.")
                    
                    # Apply forms to the existing upcoming_df
                    for idx, row in upcoming_df.iterrows(): # Operates on the existing upcoming_df
                        game_dt = row['game_date'] # Should be datetime from previous step
                        home_team_id_str = str(row['home_team_id'])
                        away_team_id_str = str(row['away_team_id'])

                        if pd.isna(game_dt):
                            continue

                        # Home team form: get most recent form string *before* this upcoming game's date
                        home_forms_hist = team_games_with_form_str[
                            (team_games_with_form_str['team_id'] == home_team_id_str) &
                            (team_games_with_form_str['game_date'] < game_dt)
                        ]
                        if not home_forms_hist.empty:
                            upcoming_df.loc[idx, 'home_current_form'] = home_forms_hist['form_str'].iloc[-1]

                        # Away team form
                        away_forms_hist = team_games_with_form_str[
                            (team_games_with_form_str['team_id'] == away_team_id_str) &
                            (team_games_with_form_str['game_date'] < game_dt)
                        ]
                        if not away_forms_hist.empty:
                            upcoming_df.loc[idx, 'away_current_form'] = away_forms_hist['form_str'].iloc[-1]
                    logger.info("Team form strings (from historical context) applied to upcoming_df.")
                else: # team_games_with_form_str was empty
                    logger.warning("Form context: No historical form strings generated (team_games_with_form_str is empty). upcoming_df forms will be empty.")
            else: # team_games_long_for_form was empty
                logger.warning("Form context: team_games_long_for_form is empty. Cannot calculate historical form strings. upcoming_df forms will be empty.")
        else: # df_for_form_context became empty after date processing
            logger.warning("Form context: hist_df_mlb became empty after date processing. Cannot calculate historical form strings. upcoming_df forms will be empty.")
    else: # hist_df_mlb was None, empty, or missing 'game_date' initially
        logger.warning("Historical data (hist_df_mlb) is not available or suitable for form calculation context. upcoming_df forms will be empty.")
    


    # MODIFICATION: Pre-calculate team form strings END
    # ─────────────────────── 2) COLUMN SANITY (MOVED AND ADAPTED) ────────────────────────
    # Ensure upcoming_df has the necessary columns for the feature engine
    # orig_game_id, game_date_et, season are created later.
    # Form columns are now potentially populated.
    # Pitcher handedness needs to be handled.

    # Ensure pitcher handedness columns exist on upcoming_df, using data from fetch_upcoming_games_data if available
    # or defaulting. The feature engine's 'advanced' module will use these.
    for hand_col_suffix in ["probable_pitcher_handedness"]:
        for team_prefix in ["home_", "away_"]:
            col_name = f"{team_prefix}{hand_col_suffix}"
            if col_name not in upcoming_df.columns:
                upcoming_df[col_name] = "" # Default to empty string if not fetched
                logger.info(f"'{col_name}' not found in upcoming_df, adding as empty string.")
            else:
                # Ensure it's string and fill NaNs if it was fetched but some are missing
                upcoming_df[col_name] = upcoming_df[col_name].astype(str).fillna("")


    # If hist_df_mlb is used by feature engine, ensure it also has these columns consistently
    if hist_df_mlb is not None and not hist_df_mlb.empty:
        for col in ("home_current_form", "away_current_form"):
             if col not in hist_df_mlb.columns: hist_df_mlb[col] = "" # Should be populated by its own form calc if needed by engine

        for hand_col in ("home_starter_pitcher_handedness", "away_starter_pitcher_handedness"):
            if hand_col not in hist_df_mlb.columns: hist_df_mlb[hand_col] = ""
            else: hist_df_mlb[hand_col] = hist_df_mlb[hand_col].astype(str).fillna("")
    
    # The old BOX_SCORE_COLS loop for upcoming_df might be redundant if feature engine doesn't need them on input
    # For hist_df_mlb, these should have been loaded by load_recent_historical_data

    # Filter hist_df_mlb to be None if empty after all ops
    hist_df_mlb = None if (hist_df_mlb is None or hist_df_mlb.empty) else hist_df_mlb
    team_stats_df = None if (team_stats_df is None or team_stats_df.empty) else team_stats_df


        # --- SECTION: FINAL PREPARATION FOR FEATURE ENGINE ---
    logger.info("Finalizing DataFrames for the feature engine...")

    # 1. Finalize 'upcoming_df' (the primary input to the feature engine)
    # It should already have:
    #   - game_id, home_team_id, away_team_id, home_team_name, away_team_name (from fetch_upcoming_games_data)
    #   - home_starter_pitcher_handedness, away_starter_pitcher_handedness (from mapping logic)
    #   - home_current_form, away_current_form (from form pre-calculation)
    #   - game_date (as datetime object from earlier steps)

    if 'game_id' not in upcoming_df.columns:
        logger.error("'game_id' is missing from upcoming_df. Cannot proceed.")
        return [], []
    upcoming_df["orig_game_id"] = upcoming_df["game_id"].astype(str)
    
    if 'game_date' not in upcoming_df.columns or upcoming_df['game_date'].isnull().all():
        logger.error("'game_date' is missing or all null in upcoming_df before creating 'game_date_et'. This indicates a problem in data loading or form calculation logic. Aborting.")
        return [], [] 
    upcoming_df["game_date_et"] = pd.to_datetime(upcoming_df["game_date"]) 
    upcoming_df["season"] = upcoming_df["game_date_et"].dt.year.astype(int)
    
    # Log critical columns of upcoming_df before it enters the feature pipeline
    logger.debug(f"upcoming_df columns prepared for pipeline: {upcoming_df.columns.tolist()}")
    key_cols_to_log_upcoming = ['orig_game_id', 'game_date_et', 'season', 
                                'home_team_name', 'away_team_name',
                                'home_starter_pitcher_handedness', 'away_starter_pitcher_handedness',
                                'home_current_form', 'away_current_form']
    existing_key_cols_upcoming = [col for col in key_cols_to_log_upcoming if col in upcoming_df.columns]
    if existing_key_cols_upcoming:
        logger.info(f"Sample of key columns in upcoming_df before pipeline: \n{upcoming_df[existing_key_cols_upcoming].head().to_string(index=False)}")


    # 2. Sanity check and prepare auxiliary DataFrames: hist_df_mlb and team_stats_df
    
    # For hist_df_mlb:
    # - Pitcher handedness columns ('*_starter_pitcher_handedness') should have been loaded by load_recent_historical_data.
    # - Form columns ('*_current_form') are not strictly necessary here if form_transform primarily uses forms from the main 'df' (upcoming_df).
    #   Initializing them as empty strings is a safeguard against errors if any module unexpectedly probes them.
    if hist_df_mlb is not None and not hist_df_mlb.empty:
        logger.debug(f"Performing final checks on hist_df_mlb (shape: {hist_df_mlb.shape})")
        for col in ["home_starter_pitcher_handedness", "away_starter_pitcher_handedness", 
                    "home_current_form", "away_current_form"]:
            if col not in hist_df_mlb.columns: 
                hist_df_mlb[col] = "" 
                logger.debug(f"Added missing safeguard column '{col}' to hist_df_mlb as empty string.")
            else: 
                # Ensure string type for handedness and form, fill NaNs
                hist_df_mlb[col] = hist_df_mlb[col].astype(str).fillna("")
        # Ensure 'game_date_et' and 'season' also exist on hist_df_mlb if any module needs them (engine.py already does this if 'season' missing)
        if 'game_date' in hist_df_mlb.columns: # game_date should already be datetime
            if 'game_date_et' not in hist_df_mlb.columns: hist_df_mlb['game_date_et'] = pd.to_datetime(hist_df_mlb['game_date'])
            if 'season' not in hist_df_mlb.columns and 'game_date_et' in hist_df_mlb.columns:
                 hist_df_mlb['season'] = hist_df_mlb['game_date_et'].dt.year


    # 3. Set DataFrames to None if they are empty (engine.py might expect this)
    hist_df_mlb = None if (hist_df_mlb is None or hist_df_mlb.empty) else hist_df_mlb
    team_stats_df = None if (team_stats_df is None or team_stats_df.empty) else team_stats_df
    
    logger.info(f"hist_df_mlb is {'None' if hist_df_mlb is None else 'Populated'}")
    logger.info(f"team_stats_df is {'None' if team_stats_df is None else 'Populated'}")

    # --- END OF FINAL PREPARATION FOR FEATURE ENGINE ---

    # ─────────────────── 4) RUN FEATURE PIPELINE ────────────────────
    # (This comment and the call to run_mlb_feature_pipeline should follow immediately)
    execution_order = ["rest", "season", "rolling", "form", "h2h", "advanced"]
    
    logger.info(f"Calling run_mlb_feature_pipeline with upcoming_df shape: {upcoming_df.shape}")
    # Ensure all columns in the list actually exist in upcoming_df to avoid KeyError in the log message itself
    cols_to_log_in_upcoming_sample = [
        'orig_game_id', 'home_team_name', 'away_team_name', 
        'home_starter_pitcher_handedness', 'away_starter_pitcher_handedness', 
        'home_current_form', 'away_current_form'
    ]
    existing_cols_for_log_sample = [col for col in cols_to_log_in_upcoming_sample if col in upcoming_df.columns]
    if existing_cols_for_log_sample:
        logger.info(f"PREDICTION_DEBUG: Sample of key columns in upcoming_df for ENGINE input: \n{upcoming_df[existing_cols_for_log_sample].head().to_string(index=False)}")
    else:
        logger.warning("PREDICTION_DEBUG: Could not log sample of key columns as none of them exist in upcoming_df.")

    features_df = run_mlb_feature_pipeline(
        df=upcoming_df, 
        mlb_historical_games_df=hist_df_mlb,
        mlb_historical_team_stats_df=team_stats_df,
        rolling_window_sizes=[15, 30, 60, 100], 
        h2h_max_games=10, 
        execution_order=execution_order,
        flag_imputations=True, 
        debug=debug_mode, # Passed from prediction.py args
    )

    if features_df.empty:
        logger.error("PREDICTION_PIPELINE: features_df is empty after run_mlb_feature_pipeline. Aborting.")
        return [], []
        
    # Initial cleanup of features_df (as you had it)
    # Note: If 'orig_game_id' is actually what's in feature_list, this rename to 'game_id'
    # might cause 'game_id' features to be "missing" if feature_list expects 'orig_game_id'.
    # Or, if 'game_id' from feature_list is expected, this is correct.
    # For now, assuming 'game_id' is the canonical key used in feature_list.
    if "orig_game_id" in features_df.columns and "game_id" not in features_df.columns:
        logger.info("PREDICTION_PIPELINE: Renaming 'orig_game_id' to 'game_id' in features_df.")
        features_df = features_df.rename(columns={"orig_game_id": "game_id"})
    elif "orig_game_id" in features_df.columns and "game_id" in features_df.columns:
        logger.warning("PREDICTION_PIPELINE: Both 'orig_game_id' and 'game_id' exist in features_df. 'game_id' will be prioritized if it's in feature_list.")
        # Or decide on a strategy, e.g., drop 'orig_game_id' if 'game_id' is primary
        # features_df = features_df.drop(columns=["orig_game_id"])


    if features_df.columns.duplicated().any():
        logger.info("PREDICTION_PIPELINE: Dropping duplicated columns from features_df.")
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    # ─────────────────── 5) LOAD MODELS / WEIGHTS AND FEATURE LIST ──────────────────
    models, feature_list = load_trained_models(model_dir, load_feature_list=True)
    if not models:
        logger.error("PREDICTION_PIPELINE: No models loaded – aborting.")
        return [], []

    if not features_df.empty and feature_list:
        ha_split_feats_in_list = [f for f in feature_list if "hist_HA_" in f]
        if ha_split_feats_in_list:
            logger.info(f"PREDICTION_DEBUG: Describing HA Split features in features_df (from 2024 data):\n"
                        f"{features_df[ha_split_feats_in_list].describe().to_string()}")
        else:
            logger.info("PREDICTION_DEBUG: No 'hist_HA_' features found in selected feature_list.")



    # Ensure team_stats_df is available in this scope if smart fill-in needs it
    # It should be passed or loaded earlier in generate_predictions
    # For this example, assuming team_stats_df is already loaded and available in the function scope.
    # if 'team_stats_df' not in locals() and 'team_stats_df' not in globals():
    #     logger.warning("PREDICTION_PIPELINE: team_stats_df not found for smart fill-in. Fill-in might be impaired.")
    #     # team_stats_df = load_team_stats_data(supabase) # Or however it should be loaded if missing

        # ─── 5A) Dedupe feature_list (already done in your load_trained_models), Log, Perform Missing Feature Checks & Smart Fill-in ───
    if feature_list: 
        logger.info(f"PREDICTION_PIPELINE: Using feature_list with {len(feature_list)} features from mlb_selected_features.json.")
        # New PREDICTION_DEBUG logging block:
        logger.info(f"PREDICTION_DEBUG: Number of columns from feature_pipeline (features_df): {len(features_df.columns)}")
        
        missing_feats_set = set(feature_list) - set(features_df.columns)
        
        if missing_feats_set:
            logger.warning(
                f"PREDICTION_PIPELINE: {len(missing_feats_set)} features from selected list are MISSING in generated features_df. "
                f"Applying smart fill-in. Missing examples (up to 10): {sorted(list(missing_feats_set))[:10]}"
            )
            
            # --- SMART FILL-IN LOGIC STARTS HERE ---
            # Ensure team_stats_df is available (it should be loaded earlier in generate_predictions)
            if team_stats_df is not None and not team_stats_df.empty: 
                logger.debug("PREDICTION_PIPELINE: Preparing team_core for smart fill-in.")
                team_core_df = team_stats_df.copy()
                # Ensure 'team_id' in team_core_df is appropriate for indexing (e.g., if it needs to be int)
                # If team_core_df.index is already 'team_id' as int, this might not be needed.
                # Assuming 'team_id' column exists and can be set as index.
                if 'team_id' in team_core_df.columns:
                    try:
                        team_core_df['team_id'] = pd.to_numeric(team_core_df['team_id'], errors='raise')
                        team_core_df = team_core_df.set_index('team_id')
                    except Exception as e_tc_idx:
                        logger.warning(f"PREDICTION_PIPELINE: Could not set 'team_id' as index for team_core. Error: {e_tc_idx}. Team lookups might fail.")
                        team_core_df = pd.DataFrame() # Make it empty to fallback to league averages

                required_team_core_cols = ["runs_for_avg_all", "runs_against_avg_all", "wins_all_percentage"]
                if 'season' not in team_core_df.columns and not team_core_df.empty:
                    logger.warning("PREDICTION_PIPELINE: 'season' column missing in team_stats_df for smart fill-in sort. Using last available stats per team (if index is team_id).")
                    team_core = (
                        team_core_df # Assuming it's already effectively 'last' per team if no season sort
                        .reindex(columns=required_team_core_cols, fill_value=np.nan) # Ensure columns exist
                        .rename(columns={
                            "runs_for_avg_all":      "avg_for",
                            "runs_against_avg_all":  "avg_against",
                            "wins_all_percentage":   "win_pct",
                        })
                    )
                elif not team_core_df.empty:
                     team_core = (
                        team_core_df
                        .sort_values("season") # Requires 'season' column
                        .groupby(level=0) # Groupby index (team_id)
                        .last()
                        .reindex(columns=required_team_core_cols, fill_value=np.nan)
                        .rename(columns={
                            "runs_for_avg_all":      "avg_for",
                            "runs_against_avg_all":  "avg_against",
                            "wins_all_percentage":   "win_pct",
                        })
                    )
                else: # team_core_df became empty (e.g. if index setting failed)
                    team_core = pd.DataFrame()

                if not team_core.empty:
                    logger.debug(f"PREDICTION_PIPELINE: team_core created for smart fill-in with shape {team_core.shape}. Index: {team_core.index.name}, Columns: {team_core.columns.tolist()}")
                else:
                    logger.warning("PREDICTION_PIPELINE: team_core is empty after preparation. Team lookups will use league averages.")

            else: # team_stats_df was None or empty
                logger.warning("PREDICTION_PIPELINE: team_stats_df is None or empty. Smart fill-in will rely on league averages only for team-specific proxies.")
                team_core = pd.DataFrame() 

            LEAGUE_AVG_FOR     = 4.60
            LEAGUE_AVG_AGAINST = 4.60
            LEAGUE_WIN_PCT     = 0.50

            # Helper function for team lookups, defined within this scope
            def _team_lookup(tid_str: str, stat_key: str, default_value: float) -> float:
                if not tid_str or not tid_str.isdigit():
                    return default_value
                team_id_int = int(tid_str)
                try:
                    if not team_core.empty and team_id_int in team_core.index: # Check if team_id_int is in the index
                        value = team_core.at[team_id_int, stat_key]
                        return float(value) if pd.notna(value) else default_value
                    else:
                        return default_value
                except KeyError: # If stat_key is not a column in team_core for some reason
                    logger.debug(f"PREDICTION_PIPELINE: _team_lookup stat_key '{stat_key}' not in team_core columns. Using default for tid {tid_str}.")
                    return default_value
                except Exception as e_lookup:
                    logger.debug(f"PREDICTION_PIPELINE: _team_lookup failed for tid {tid_str}, stat {stat_key}. Error: {e_lookup}. Using default.")
                    return default_value

            # Actual loop for filling missing features in features_df
            for mcol in missing_feats_set:
                current_default_for_mcol = LEAGUE_AVG_FOR # General default for this iteration
                try:
                    # Ensure features_df has home_team_id and away_team_id for the map/apply
                    home_team_id_col_exists = "home_team_id" in features_df.columns
                    away_team_id_col_exists = "away_team_id" in features_df.columns

                    if mcol.startswith("home_"):
                        if home_team_id_col_exists:
                            # Use .apply for vectorized operation if possible, or .map if Series.
                            # Assuming features_df["home_team_id"] is a Series.
                            if "runs_scored" in mcol:
                                features_df[mcol] = features_df["home_team_id"].apply(lambda tid: _team_lookup(str(tid), "avg_for", LEAGUE_AVG_FOR))
                            elif "runs_allowed" in mcol or "hits_allowed" in mcol:
                                features_df[mcol] = features_df["home_team_id"].apply(lambda tid: _team_lookup(str(tid), "avg_against", LEAGUE_AVG_AGAINST))
                            elif "win_pct" in mcol or "wins_" in mcol:
                                features_df[mcol] = features_df["home_team_id"].apply(lambda tid: _team_lookup(str(tid), "win_pct", LEAGUE_WIN_PCT))
                            else:
                                features_df[mcol] = LEAGUE_AVG_FOR 
                        else: 
                            features_df[mcol] = LEAGUE_AVG_FOR
                    elif mcol.startswith("away_"):
                        if away_team_id_col_exists:
                            if "runs_scored" in mcol:
                                features_df[mcol] = features_df["away_team_id"].apply(lambda tid: _team_lookup(str(tid), "avg_for", LEAGUE_AVG_FOR))
                            elif "runs_allowed" in mcol or "hits_allowed" in mcol:
                                features_df[mcol] = features_df["away_team_id"].apply(lambda tid: _team_lookup(str(tid), "avg_against", LEAGUE_AVG_AGAINST))
                            elif "win_pct" in mcol or "wins_" in mcol:
                                features_df[mcol] = features_df["away_team_id"].apply(lambda tid: _team_lookup(str(tid), "win_pct", LEAGUE_WIN_PCT))
                            else:
                                features_df[mcol] = LEAGUE_AVG_FOR
                        else:
                            features_df[mcol] = LEAGUE_AVG_FOR
                    else: 
                        # For non-team-specific features like 'rest_days_home' (though this starts with 'home_')
                        # or any truly generic selected feature not caught above.
                        # This might need more specific rules based on your actual selected features.
                        logger.debug(f"Smart fill-in: Applying general default ({current_default_for_mcol}) to non-prefix-specific missing feature: {mcol}")
                        features_df[mcol] = current_default_for_mcol
                except Exception as e_fill_loop:
                    logger.error(f"PREDICTION_PIPELINE: Unexpected error during smart fill-in for column '{mcol}': {e_fill_loop}. Filling with general default {current_default_for_mcol}.")
                    features_df[mcol] = current_default_for_mcol # Fallback assignment

            logger.info(f"PREDICTION_PIPELINE: Smart fill-in logic applied for {len(missing_feats_set)} features.")

            # Final sanity check (as you had it):
            still_missing_after_smart_fill = [col for col in feature_list if col not in features_df.columns]
            if still_missing_after_smart_fill:
                logger.error(
                    f"PREDICTION_PIPELINE ERROR: After smart fill-in, {len(still_missing_after_smart_fill)} selected features are "
                    f"STILL NOT in features_df columns: {still_missing_after_smart_fill[:10]}. "
                    "This indicates an issue in the fill-in logic for these specific columns. Aborting."
                )
                return [], [] 
            # --- END OF SMART FILL-IN LOGIC ---
            
        else: # missing_feats_set is empty
            logger.info("PREDICTION_PIPELINE: All features from feature_list are present in features_df. OK. No smart fill-in needed.")
            
    else: # feature_list is None or empty
        logger.warning(
            "PREDICTION_PIPELINE: feature_list (from mlb_selected_features.json) is not available. "
            "Skipping selected feature check and smart fill-in. "
            "X matrix will be built using all available numeric columns from features_df."
        )

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # START: NEW DIAGNOSTIC LOGGING CODE TO INSERT
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

    if feature_list and not features_df.empty:
        logger.info("--- Starting Diagnostic Logging for Selected Features (Pre-Imputation to 0.0) ---")
        
        # Ensure all selected features are actually in features_df for .describe() to avoid KeyErrors
        # This check is important because smart fill-in might add columns later if they were completely missing.
        # Here, we are interested in features *produced by the pipeline or present before smart fill-in*.
        
        # Get the intersection of feature_list and features_df.columns to only act on present columns
        # However, if smart-fill-in has already run and added them, then feature_list should be okay.
        # The critical part is that this happens *before* X = features_df.reindex(columns=feature_list, fill_value=0.0)
        
        # For safety, let's operate on features that are in both feature_list AND features_df.columns
        # (though after your smart fill-in, all of feature_list SHOULD be in features_df.columns)
        
        # Log NaN percentages for ALL selected features that are present in features_df
        logger.info("NaN Percentages for Selected Features in current features_df batch:")
        present_selected_features = [f for f in feature_list if f in features_df.columns]
        if present_selected_features:
            nan_percentages = features_df[present_selected_features].isnull().mean().sort_values(ascending=False) * 100
            if nan_percentages.empty:
                logger.info("No NaNs found in any selected features present in features_df.")
            else:
                logger.info(f"\n{nan_percentages[nan_percentages >= 0].to_string()}") # Show all, even 0%
        else:
            logger.warning("No selected features were found in features_df columns for NaN percentage logging.")

        # Categorize for targeted .describe()
        # Ensure you have defined `categorize_selected_features` function earlier in the file
        categorized_feats = categorize_selected_features(feature_list)

        # HA-Split Features (Selected Ones)
        selected_ha_split = [f for f in categorized_feats["ha_split"] if f in features_df.columns]
        if selected_ha_split:
            logger.info(f"Describing {len(selected_ha_split)} Selected HA-Split Features in features_df batch:")
            try:
                logger.info(f"\n{features_df[selected_ha_split].describe().to_string()}")
            except KeyError as e:
                logger.error(f"KeyError describing HA-Split features: {e}. Features requested: {selected_ha_split}")
        else:
            logger.info("No selected HA-Split features found in features_df to describe.")

        # Rolling Features (Selected Ones - Sample or All)
        # You might want to sample if there are too many, or describe all if manageable.
        # Here, we'll describe all selected rolling features found in features_df.
        selected_rolling = [f for f in categorized_feats["rolling"] if f in features_df.columns]
        if selected_rolling:
            logger.info(f"Describing {len(selected_rolling)} Selected Rolling Features in features_df batch:")
            try:
                logger.info(f"\n{features_df[selected_rolling].describe().to_string()}")
            except KeyError as e:
                 logger.error(f"KeyError describing Rolling features: {e}. Features requested: {selected_rolling}")
        else:
            logger.info("No selected Rolling features found in features_df to describe.")
            
        # H2H Features (Selected Ones)
        selected_h2h = [f for f in categorized_feats["h2h"] if f in features_df.columns]
        if selected_h2h:
            logger.info(f"Describing {len(selected_h2h)} Selected H2H Features in features_df batch:")
            try:
                logger.info(f"\n{features_df[selected_h2h].describe().to_string()}")
            except KeyError as e:
                logger.error(f"KeyError describing H2H features: {e}. Features requested: {selected_h2h}")
        else:
            logger.info("No selected H2H features found in features_df to describe.")
            
        # Imputed Flags (Selected Ones) - usually binary, but describe can still be informative
        selected_imputed_flags = [f for f in categorized_feats["imputed_flags"] if f in features_df.columns]
        if selected_imputed_flags:
            logger.info(f"Describing {len(selected_imputed_flags)} Selected Imputed Flag Features in features_df batch:")
            try:
                logger.info(f"\n{features_df[selected_imputed_flags].describe().to_string()}")
            except KeyError as e:
                logger.error(f"KeyError describing Imputed Flag features: {e}. Features requested: {selected_imputed_flags}")
        else:
            logger.info("No selected Imputed Flag features found in features_df to describe.")
            
        # Other Selected Features
        selected_other = [f for f in categorized_feats["other"] if f in features_df.columns]
        if selected_other:
            logger.info(f"Describing {len(selected_other)} Other Selected Features in features_df batch:")
            try:
                logger.info(f"\n{features_df[selected_other].describe().to_string()}")
            except KeyError as e:
                logger.error(f"KeyError describing Other features: {e}. Features requested: {selected_other}")
        else:
            logger.info("No Other selected features found in features_df to describe.")
            
        logger.info("--- Finished Diagnostic Logging for Selected Features ---")
    elif not feature_list:
        logger.warning("Diagnostic Logging: feature_list is not available. Skipping .describe() and NaN logs.")
    elif features_df.empty:
        logger.warning("Diagnostic Logging: features_df is empty. Skipping .describe() and NaN logs.")
        
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # END: NEW DIAGNOSTIC LOGGING CODE
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲



    # --- Load Ensemble Weights (as in your original script) ---

    weights = FALLBACK_ENSEMBLE_WEIGHTS_MLB.copy() # Ensure FALLBACK_ENSEMBLE_WEIGHTS_MLB is defined
    wfile = model_dir / ENSEMBLE_WEIGHTS_FILENAME_MLB # Ensure ENSEMBLE_WEIGHTS_FILENAME_MLB is defined
    if wfile.is_file():
        try:
            raw = json.loads(wfile.read_text()) # Ensure json is imported
            parsed = {k.replace("_mlb_runs_predictor", ""): float(v) for k, v in raw.items()}
            s = sum(parsed.values())
            if abs(s) > 1e-9: 
                weights_loaded = {model_key: parsed.get(model_key, 0.0) / s for model_key in FALLBACK_ENSEMBLE_WEIGHTS_MLB} 
                # Only update if all keys in FALLBACK_ENSEMBLE_WEIGHTS_MLB are present in parsed, or handle partial update
                # For simplicity, if all expected model keys are in parsed:
                if all(key in parsed for key in FALLBACK_ENSEMBLE_WEIGHTS_MLB.keys()):
                    weights = weights_loaded
                else:
                    logger.warning(f"Parsed weights from {wfile.name} missing some expected model keys. Using fallback for consistency or check logic.")
                    # Decide: use fallback, or use partial normalized, or error. For now, let's be cautious and log.
                    # Using fallback if not all keys present:
                    # weights = FALLBACK_ENSEMBLE_WEIGHTS_MLB.copy()
                    # Or, use what was parsed but log it:
                    weights = {k: v for k,v in weights_loaded.items() if k in FALLBACK_ENSEMBLE_WEIGHTS_MLB}


            else: # sum is zero
                logger.warning(f"Sum of parsed weights from {wfile.name} is near zero. Using fallback weights.")
            logger.info(f"Loaded/finalized ensemble weights: {weights}")
        except Exception as e:
            logger.warning(f"Error loading or parsing ensemble weights file {wfile.name} – using fallback. Error: {e}")
    else:
        logger.info(f"Ensemble weights file {wfile.name} not found. Using fallback weights: {weights}")


    # ─────────────────── 6) BUILD FEATURE MATRIX X ───────────────────

    logger.info("PREDICTION_PIPELINE: Building feature matrix X...")
    
    if features_df.columns.duplicated().any(): 
        logger.warning("PREDICTION_PIPELINE: Duplicate columns detected in features_df before creating X (should have been handled). Keeping first.")
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    if feature_list:
        X = features_df.reindex(columns=feature_list, fill_value=0.0) 
        logger.debug(f"X matrix created using reindex from feature_list. Shape: {X.shape}. Columns (up to 5): {X.columns.tolist()[:5]}")
    else: # feature_list was None or empty
        logger.warning("PREDICTION_PIPELINE: No feature_list available. Creating X from all numeric columns in features_df.")
        X = features_df.select_dtypes(include=np.number)
        if X.empty and not features_df.empty:
            logger.warning("PREDICTION_PIPELINE: X matrix is empty after selecting numeric dtypes (features_df wasn't empty). Check dtypes or content.")
        elif X.empty:
            logger.error("PREDICTION_PIPELINE: X matrix is completely empty (no numeric columns or features_df was empty). Cannot proceed with predictions.")
            return [],[]

    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0) # Final clean of X
    logger.debug(f"X matrix final clean (NaN/inf filled). Shape: {X.shape}")

    if X.empty: # Final check
        logger.error("PREDICTION_PIPELINE: X matrix is empty before passing to models. Aborting.")
        return [],[]

    # ───────────────────── 7) PREDICT PER MODEL ──────────────────────
    preds_by_model: Dict[str, pd.DataFrame] = {}
    for key, mdl in models.items():
        df_pred = mdl.predict(X)
        if df_pred is not None and not df_pred.empty:
            df_pred = df_pred.rename(
                columns={
                    "predicted_home_score": "predicted_home_runs",
                    "predicted_away_score": "predicted_away_runs",
                }
            )
            preds_by_model[key] = df_pred[["predicted_home_runs", "predicted_away_runs"]]

    if not preds_by_model:
        logger.error("All models failed to predict – aborting.")
        return [], []

    # ────────────────────── 8) ENSEMBLE + OUTPUT ─────────────────────
    meta_df = upcoming_df[["game_id", "game_date", "home_team_name", "away_team_name"]].set_index("game_id")
    final_preds, raw_preds = [], []

    for idx, row in features_df.iterrows():
        gid = str(row["game_id"])
        if gid not in meta_df.index:
            continue

        comp = {
            k: {"home": v.at[idx, "predicted_home_runs"], "away": v.at[idx, "predicted_away_runs"]}
            for k, v in preds_by_model.items()
        }

        w_sum = sum(weights.get(k, 0) for k in comp)
        if w_sum < 1e-6:
            h_ens = np.mean([v["home"] for v in comp.values()])
            a_ens = np.mean([v["away"] for v in comp.values()])
        else:
            # Use weights.get(k, 0) to avoid KeyError if a model key isn't in weights
            h_ens = sum(comp[k]["home"] * weights.get(k, 0) for k in comp) / w_sum
            a_ens = sum(comp[k]["away"] * weights.get(k, 0) for k in comp) / w_sum

        run_diff = h_ens - a_ens
        meta = meta_df.loc[gid]
        out = {
            "game_id": gid,
            "game_date": pd.to_datetime(meta["game_date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team_name": str(meta["home_team_name"]),
            "away_team_name": str(meta["away_team_name"]),
            "predicted_home_runs": round(h_ens, 2),
            "predicted_away_runs": round(a_ens, 2),
            "predicted_run_diff": round(run_diff, 2),
            "predicted_total_runs": round(h_ens + a_ens, 2),
            "win_probability_home": round(1 / (1 + math.exp(-0.3 * run_diff)), 3),
            "component_predictions": comp,
        }
        raw_preds.append(out)
        final_preds.append(out)

    display_prediction_summary_mlb(final_preds)
    logger.info(f"MLB prediction pipeline finished in {time.time() - start_ts:.1f}s")
    return final_preds, raw_preds


def upsert_run_predictions(predictions: List[Dict[str, Any]], supabase_client: SupabaseClient) -> None:
    """
    Upserts ONLY the predicted home and away runs into the mlb_game_schedule table.
    """
    if not supabase_client:
        logger.error("Supabase client not available for upsert.")
        return
    if not predictions:
        logger.warning("Prediction list is empty. Nothing to upsert.")
        return

    updated_count = 0
    logger.info(f"Starting upsert for {len(predictions)} predictions (home/away runs only)...")

    for pred in predictions:
        gid = pred.get("game_id")
        if gid is None:
            logger.warning(f"Skipping prediction with missing game_id: {pred}")
            continue

        # This payload now ONLY includes the two columns you want to update.
        update_payload = {
            "predicted_home_runs": pred.get("predicted_home_runs"),
            "predicted_away_runs": pred.get("predicted_away_runs"),
        }

        # This check ensures we don't proceed if the essential prediction values are missing.
        if update_payload["predicted_home_runs"] is None or update_payload["predicted_away_runs"] is None:
            logger.warning(f"Skipping game_id {gid} due to missing run prediction values.")
            continue

        try:
            resp = (
                supabase_client.table("mlb_game_schedule")
                .update(update_payload)
                .eq("game_id", str(gid))
                .execute()
            )
            # This logic correctly checks for success or errors.
            if resp.data:
                logger.info(f"Upserted MLB predictions for game_id {gid}.")
                updated_count += 1
            elif resp.error:
                logger.error(f"Error upserting MLB game_id {gid}: {resp.error}")
            else:
                logger.warning(f"No row found or no data returned for MLB game_id {gid}.")
        except Exception as e:
            logger.error(f"Exception during upsert for MLB game_id {gid}: {e}", exc_info=True)

    logger.info(f"Finished upserting MLB predictions for {updated_count} games.")

# --- Main Execution ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate MLB Run Predictions")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_UPCOMING_DAYS_WINDOW_MLB,
        help=f"Default: {DEFAULT_UPCOMING_DAYS_WINDOW_MLB}",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB,
        help=f"Default: {DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB}",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Default: {MODELS_DIR}",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug: # Or your equivalent check for debug_mode
        # 1. Set your application's specific loggers to DEBUG
        logger.setLevel(logging.DEBUG) # For prediction.py itself (e.g., __main__)
        logging.getLogger('backend.mlb_features.advanced').setLevel(logging.DEBUG)
        logging.getLogger('backend.mlb_features.engine').setLevel(logging.DEBUG)
        # Add any other of your own modules here if you want their DEBUG logs
        # e.g., logging.getLogger('backend.mlb_features.season').setLevel(logging.DEBUG)

        # 2. Set the ROOT logger's level to DEBUG.
        # This allows DEBUG messages from your specifically enabled loggers 
        # (and potentially others if not silenced) to reach the handlers.
        logging.getLogger().setLevel(logging.DEBUG)

        # 3. Ensure the HANDLERS on the root logger are also processing DEBUG messages.
        # (basicConfig might have set them to INFO initially if root was INFO)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

        # 4. CRITICAL STEP: Silence noisy third-party libraries by setting their loggers
        #    to a higher level like INFO or WARNING.
        noisy_library_loggers = [
            "hpack", 
            "httpcore", 
            "httpx", 
            # "supabase" # Uncomment if supabase client itself is too verbose at DEBUG
            # Add any other library names that are spamming DEBUG logs
        ]
        for lib_name in noisy_library_loggers:
            logging.getLogger(lib_name).setLevel(logging.INFO) # Or logging.WARNING

        logger.info("DEBUG logging enabled for application modules. Third-party library DEBUG logs are suppressed (set to INFO/WARNING).")
    
    else: # If not in debug mode
        # Ensure your application loggers are at INFO (or their desired non-debug level)
        logger.setLevel(logging.INFO)
        logging.getLogger('backend.mlb_features.advanced').setLevel(logging.INFO)
        logging.getLogger('backend.mlb_features.engine').setLevel(logging.INFO)
        # No need to adjust library loggers here; they'll follow root (INFO) or their own defaults.

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled by CLI argument.")

    if not PROJECT_MODULES_IMPORTED:
        sys.exit("Exiting: Required MLB project modules failed to import.")

    supabase_c = get_supabase_client()
    if not supabase_c:
        sys.exit("Exiting: Supabase client could not be initialized.")

    final_mlb_preds, _ = generate_predictions(
        days_window=args.days,
        model_dir=args.model_dir,
        historical_lookback=args.lookback,
        debug_mode=args.debug,
    )

    if final_mlb_preds:
        logger.info(f"Upserting {len(final_mlb_preds)} final MLB predictions...")
        try:
            upsert_run_predictions(final_mlb_preds, supabase_c)
            logger.info("MLB Upsert process completed.")
        except Exception as e:
            logger.error(f"Error during MLB upsert: {e}", exc_info=args.debug)

        logger.info("Sample Final MLB Prediction:")
        print(json.dumps(final_mlb_preds[0], indent=2, default=str))
    else:
        logger.info("No final MLB predictions generated to upsert.")


if __name__ == "__main__":
    main()
