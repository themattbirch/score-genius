# backend/mlb_score_prediction/prediction.py

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import pytz # Retained for timezone consistency if needed
import pandas as pd
import numpy as np
import math
import json
import argparse
from typing import List, Dict, Optional, Any, Tuple

import logging
# Send all DEBUG+ messages to console
logging.basicConfig(format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO) # Default to INFO, can be overridden by --debug
logger = logging.getLogger(__name__)

# Define Paths Consistently
SCRIPT_DIR_PRED = Path(__file__).resolve().parent
# Assuming prediction.py is in backend/mlb_score_prediction/
PROJECT_ROOT_PRED = SCRIPT_DIR_PRED.parents[2]
MODELS_DIR = PROJECT_ROOT_PRED / "models" / "saved" # This is your preferred path
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR_PRED = PROJECT_ROOT_PRED / "reports_mlb" # Consistent reports dir for MLB
REPORTS_DIR_PRED.mkdir(parents=True, exist_ok=True)
logger.info(f"MLB prediction.py using model directory: {MODELS_DIR}")

# --- Third‐party clients & config ---
from backend import config # Use the shared config
from supabase import create_client, Client as SupabaseClient # Explicitly import Client
# from backend.caching.supabase_client import supabase as supabase_client_instance # Use direct creation

PACIFIC_TZ = pytz.timezone("America/Los_Angeles") # Retain if used, though ET/UTC more common
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB = 270 # Approx 1.5 MLB seasons, adjust as needed
DEFAULT_UPCOMING_DAYS_WINDOW_MLB = 7 # Predict for a week

ENSEMBLE_WEIGHTS_FILENAME_MLB = "mlb_ensemble_weights_optimized.json" # MLB specific
# Fallback weights if file not found, keys are simple model names
FALLBACK_ENSEMBLE_WEIGHTS_MLB: Dict[str, float] = {"svr": 0.6, "ridge": 0.4} # Example for MLB

# Update with columns from your mlb_historical_game_stats
REQUIRED_HISTORICAL_COLS_MLB = [
    'game_id', 'game_date_time_utc', 'home_team_id', 'away_team_id',
    'home_score', 'away_score', 'home_hits', 'away_hits', 'home_errors', 'away_errors',
    # Add other essential raw stats your feature engine might need as direct input
    # e.g., inning scores if used for very recent form before rolling stats catch up
    'h_inn_1', 'h_inn_2', 'h_inn_3', 'h_inn_4', 'h_inn_5', 'h_inn_6', 'h_inn_7', 'h_inn_8', 'h_inn_9', 'h_inn_extra',
    'a_inn_1', 'a_inn_2', 'a_inn_3', 'a_inn_4', 'a_inn_5', 'a_inn_6', 'a_inn_7', 'a_inn_8', 'a_inn_9', 'a_inn_extra',
]
# Update with columns from your mlb_historical_team_stats
REQUIRED_TEAM_STATS_COLS_MLB = [
    'team_id', 'team_name', 'season',
    'wins_all_percentage', 'runs_for_avg_all', 'runs_against_avg_all',
    # Add other key aggregated team stats
]
# Update with columns from your mlb_game_schedule for upcoming games
UPCOMING_GAMES_COLS_MLB = [
    "game_id", "scheduled_time_utc", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name",
    # Include probable pitcher IDs/names if used directly by feature engine
    "home_probable_pitcher_name", "away_probable_pitcher_name"
]

# --- Project module imports (MLB specific) ---
PROJECT_MODULES_IMPORTED = False
from backend.mlb_features.engine import run_mlb_feature_pipeline, DEFAULT_MLB_EXECUTION_ORDER # Adjust name/path as needed
from backend.mlb_score_prediction.models import (
        RidgeScorePredictor as MLBRidgePredictor,
        SVRScorePredictor as MLBSVRPredictor,
        XGBoostScorePredictor as MLBXGBoostPredictor
    )
# --- Supabase helper ---
def get_supabase_client() -> Optional[SupabaseClient]:
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY: # Prioritize service key
        logger.error("Supabase URL or Service Key not configured.")
        return None
    try:
        # Using service key for operations like updates
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}", exc_info=True)
        return None

# --- Data Loading Functions (MLB specific) ---
def load_recent_historical_data(supabase_client: SupabaseClient, days_lookback: int) -> pd.DataFrame:
    if not supabase_client: logger.error("Supabase client unavailable."); return pd.DataFrame()

    # Use the correct date column for MLB historical data
    date_column_hist = "game_date_time_utc" # As in mlb_historical_game_stats
    start_date_iso = (datetime.now(pytz.utc) - timedelta(days=days_lookback)).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"Loading MLB historical game data from {start_date_iso} onwards...")
    
    select_cols_str = ", ".join(list(set(REQUIRED_HISTORICAL_COLS_MLB + [date_column_hist]))) # Ensure date col is selected
    all_historical_data = []
    page_size = 1000; start_index = 0; has_more = True

    try:
        while has_more:
            resp = (
                supabase_client
                .table("mlb_historical_game_stats") # MLB table
                .select(select_cols_str)
                .gte(date_column_hist, start_date_iso)
                .order(date_column_hist, desc=False)
                .range(start_index, start_index + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            all_historical_data.extend(batch)
            if len(batch) < page_size: has_more = False
            else: start_index += page_size

        if not all_historical_data:
            logger.warning(f"No MLB historical game data found since {start_date_iso}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_historical_data)
        # Standardize to 'game_date' for feature engine compatibility if it expects that
        df["game_date"] = pd.to_datetime(df[date_column_hist], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["game_date"])

        numeric_cols = [c for c in REQUIRED_HISTORICAL_COLS_MLB if c not in
                        ("game_id", date_column_hist, "home_team_id", "away_team_id",
                         "home_team_name", "away_team_name")] # Adjust non-numeric list
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
        for col in ("game_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name"):
            df[col] = df.get(col).astype(str).fillna("")
        return df.sort_values("game_date").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading MLB historical data: {e}", exc_info=True)
        return pd.DataFrame()

def load_team_stats_data(supabase_client: SupabaseClient) -> pd.DataFrame:
    if not supabase_client: logger.error("Supabase client unavailable."); return pd.DataFrame()
    logger.info("Loading MLB team stats data from 'mlb_historical_team_stats'...")
    select_cols_str = ", ".join(REQUIRED_TEAM_STATS_COLS_MLB)
    # ... (pagination logic similar to load_recent_historical_data, simplified here for brevity) ...
    try:
        resp = supabase_client.table("mlb_historical_team_stats").select(select_cols_str).execute()
        if not resp.data: logger.warning("No MLB team stats data found."); return pd.DataFrame()
        
        df = pd.DataFrame(resp.data)
        numeric_cols = [c for c in REQUIRED_TEAM_STATS_COLS_MLB if c not in ("team_id", "team_name", "season")] # Adjust
        for col in numeric_cols: df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
        for col in ("team_id", "team_name", "season"): df[col] = df.get(col).astype(str).fillna("")
        return df
    except Exception as e:
        logger.error(f"Error loading MLB team stats: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client: SupabaseClient, days_window: int) -> pd.DataFrame:
    # (Timezone logic from NBA version is fine, uses UTC for query)
    try:
        from zoneinfo import ZoneInfo          # Python ≥ 3.9
        ET = ZoneInfo("America/New_York")
        UTC = ZoneInfo("UTC")
    except (ImportError, KeyError):            # Fallback for older runtimes
        ET = pytz.timezone("America/New_York")
        UTC = pytz.utc
    today_et = datetime.now(ET).date()
    start_et = datetime(today_et.year, today_et.month, today_et.day, tzinfo=ET)
    end_et = start_et + timedelta(days=days_window)
    start_utc_str = start_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc_str = end_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Use correct date column from mlb_game_schedule: scheduled_time_utc
    date_column_upcoming = "scheduled_time_utc"
    logger.info(f"Fetching upcoming MLB games between {start_utc_str} and {end_utc_str} (UTC)...")

    try:
        resp = (
            supabase_client
            .table("mlb_game_schedule") # MLB table
            .select(", ".join(list(set(UPCOMING_GAMES_COLS_MLB + [date_column_upcoming])))) # Ensure date col selected
            .gte(date_column_upcoming, start_utc_str)
            .lt(date_column_upcoming, end_utc_str)
            .order(date_column_upcoming, desc=False)
            .execute()
        )
        rows = resp.data or []
        if not rows: logger.warning("No upcoming MLB games found."); return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["scheduled_time_utc_dt"] = pd.to_datetime(df[date_column_upcoming], errors="coerce", utc=True)
        df = df.dropna(subset=["scheduled_time_utc_dt"])
        # Standardize to 'game_date' (as date object, naive) for feature engine if it expects that
        df["game_date"] = pd.to_datetime(df["scheduled_time_utc_dt"].dt.tz_convert(ET).dt.date) # Example: Game date in ET
        
        # Ensure required columns are present and have correct types
        for c in ("game_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name"):
            df[c] = df.get(c).astype(str).fillna("Unknown") # Use get for safety
        return df.sort_values("scheduled_time_utc_dt").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching upcoming MLB games: {e}", exc_info=True)
        return pd.DataFrame()

# --- Model Loading (MLB specific) ---
def load_trained_models(
    model_dir: Path = MODELS_DIR, # Uses shared MAIN_MODELS_DIR
    load_feature_list: bool = True,
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    feature_list: Optional[List[str]] = None
    features_fp = model_dir / "mlb_selected_features.json" # MLB specific
    if load_feature_list and features_fp.is_file():
        try:
            with open(features_fp, 'r') as f: feature_list = json.load(f)
            logger.info(f"Loaded mlb_selected_features.json with {len(feature_list)} features")
        except Exception as e: logger.warning(f"Could not read mlb_selected_features.json: {e}")
    elif load_feature_list: logger.warning(f"{features_fp.name} not found. Model might use all features or internal list.")

    models: Dict[str, Any] = {}
    # Ensure class names match those imported from mlb_score_prediction.models
    # The keys 'svr', 'ridge', 'xgb' are used for ensemble weights.
    model_map = {
        "svr": MLBSVRPredictor,
        "ridge": MLBRidgePredictor,
        "xgb": MLBXGBoostPredictor # Add XGBoost if you use it
    }
    for name, ClsMLB in model_map.items():
        try:
            # Model name used for filename loading, e.g., "svr_mlb_runs_predictor"
            pred = ClsMLB(model_dir=str(model_dir), model_name=f"{name}_mlb_runs_predictor")
            pred.load_model() # load_model in BaseScorePredictor will find the latest timestamped file
            models[name] = pred
            logger.info(f"Loaded MLB '{name}' predictor from {model_dir}")
        except Exception as e:
            logger.error(f"Could not load MLB '{name}' model: {e}", exc_info=True)
    if not models: raise RuntimeError(f"No MLB models could be loaded from {model_dir}")
    return models, feature_list

# --- Betting Odds Parsing (Adapt if MLB odds structure in Supabase differs) ---
# (Assuming parsing functions are general enough, but source table and column names are critical)
def fetch_and_parse_betting_odds( # This function needs careful review for MLB table structure
    supabase_client: SupabaseClient, game_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    if not supabase_client or not game_ids: return {}
    
    odds_dict: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50
    # Your mlb_game_schedule table has 'moneyline', 'spread', 'total' (text) and also specific price/line cols
    # The NBA version constructs clean strings then parses them. For MLB, you might directly use the parsed columns
    # if they are reliable (e.g., moneyline_home_clean, spread_home_line_clean).
    # For now, adapting the NBA's "construct string then parse" logic, assuming raw odds are text/JSON.
    
    # Helper to get data from potentially nested dicts (like NBA odds structure)
    def get_odds_value(data_dict, team_key, value_key, default=None):
        if not isinstance(data_dict, dict): return default
        team_data = data_dict.get(team_key)
        if not isinstance(team_data, dict): return default
        return team_data.get(value_key, default)

    for i in range(0, len(game_ids), chunk_size):
        chunk = game_ids[i : i + chunk_size]
        try:
            # Select raw odds text columns and also pre-parsed fields if available
            # Adjust these select columns based on your mlb_game_schedule table!
            resp = (
                supabase_client
                .table("mlb_game_schedule") # MLB table
                .select("game_id, home_team_name, away_team_name, moneyline, spread, total, "
                        "moneyline_home_clean, moneyline_away_clean, spread_home_line_clean, total_line_clean") # Example parsed cols
                .in_("game_id", chunk)
                .execute()
            )
            rows = resp.data or []
            for r_dict in rows:
                gid = str(r_dict["game_id"])
                home_ml_val = r_dict.get("moneyline_home_clean")
                away_ml_val = r_dict.get("moneyline_away_clean")
                home_sp_val = r_dict.get("spread_home_line_clean")
                total_val   = r_dict.get("total_line_clean")

                # if any of those are None, skip or set defaults:
                if home_ml_val is None or away_ml_val is None:
                    logger.warning(f"Missing cleaned moneyline for game {gid}. Skipping.")
                    continue

                # Build the MLB‐specific entry:
                entry = {
                    "moneyline": {"home": float(home_ml_val), "away": float(away_ml_val)},
                    "spread": {
                        "home_line": float(home_sp_val) if home_sp_val is not None else None,
                        "away_line": (float(-home_sp_val) if home_sp_val is not None else None),
                        "home_odds": -110,  # placeholder
                        "away_odds": -110,
                    },
                    "total": {
                        "line": float(total_val) if total_val is not None else None,
                        "over_odds": -110,
                        "under_odds": -110,
                    },
                    "bookmaker": "ParsedFromSupabaseMLB",
                    "last_update": pd.Timestamp.now(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                odds_dict[gid] = entry
        except Exception as e:
            logger.error(f"Error fetching/parsing MLB odds chunk: {e}", exc_info=True)
    logger.info(f"Finished fetching/parsing odds for {len(odds_dict)} MLB games.")
    return odds_dict

# --- Display Summary (MLB specific) ---
def display_prediction_summary_mlb(preds: List[Dict]) -> None:
    # (Column widths might need adjustment for MLB team names/run values)
    header_line = "=" * 125 # Adjusted width
    header_title = " " * 45 + "MLB PREGAME PREDICTION SUMMARY"
    header_cols = ( # Removed Lower/Upper Bound, Calib?, Raw Diff/Total to simplify
        f"{'DATE':<11} {'MATCHUP':<30} {'PRED RUNS':<12} {'PRED RUN DIFF':<15} "
        f"{'WIN PROB (H)':<15}"
    ) # Removed Conf Pct
    header_sep = "-" * 125

    print(f"\n{header_line}\n{header_title}\n{header_line}\n{header_cols}\n{header_sep}")
    try:
        df = pd.DataFrame(preds)
        if 'game_date' in df.columns:
            df['game_date_dt'] = pd.to_datetime(df['game_date'], errors="coerce")
            df = df.dropna(subset=['game_date_dt'])
            # Sort by date, then by absolute predicted run differential (closer games first)
            df = df.sort_values(['game_date_dt', 'predicted_run_diff'],
                                key=lambda x: np.abs(x) if x.name == 'predicted_run_diff' else x)
    except Exception as e: logger.error(f"Error sorting for MLB display: {e}"); df = pd.DataFrame(preds)

    for _, g in df.iterrows():
        try:
            date_str = pd.to_datetime(g.get('game_date')).strftime("%Y-%m-%d") if pd.notna(g.get('game_date')) else "N/A"
            matchup = f"{str(g.get('home_team_name','?'))[:13]} vs {str(g.get('away_team_name','?'))[:13]}" # Use team_name
            
            final_home = g.get('predicted_home_runs', np.nan) # MLB key
            final_away = g.get('predicted_away_runs', np.nan) # MLB key
            final_diff = g.get('predicted_run_diff', np.nan)  # MLB key
            winp_home = g.get('win_probability_home', np.nan) # MLB key

            final_runs_str = f"{final_home:.1f}-{final_away:.1f}" if pd.notna(final_home) and pd.notna(final_away) else "N/A"
            final_diff_str = f"{final_diff:+.1f}" if pd.notna(final_diff) else "N/A"
            win_prob_str = f"{winp_home*100:.1f}%" if pd.notna(winp_home) else "N/A"

            print(f"{date_str:<11} {matchup:<30} {final_runs_str:<12} {final_diff_str:<15} {win_prob_str:<15}")
        except Exception as e: logger.error(f"Error displaying MLB game {g.get('game_id')}: {e}", exc_info=True)
    print(header_line)

# --- Core Pipeline (MLB specific) ---
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW_MLB,
    model_dir: Path = MODELS_DIR,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB,
    debug_mode: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    
    logger.info("--- Starting MLB Prediction Pipeline ---")
    start_ts = time.time()

    if not PROJECT_MODULES_IMPORTED:
        logger.critical("MLB Project modules not imported. Cannot generate predictions.")
        return [], []

    supabase = get_supabase_client()
    if not supabase:
        logger.critical("Cannot proceed: Supabase client failed.")
        return [], []

    hist_df_mlb = load_recent_historical_data(supabase, historical_lookback)
    team_stats_df_mlb = load_team_stats_data(supabase)
    upcoming_df_mlb = fetch_upcoming_games_data(supabase, days_window)

    if upcoming_df_mlb.empty:
        logger.warning("No upcoming MLB games; exiting.")
        return [], []

    # Replace empty DataFrames with None so feature engine knows to handle missing history
    hist_df_mlb = None if (hist_df_mlb is not None and hist_df_mlb.empty) else hist_df_mlb
    team_stats_df_mlb = None if (team_stats_df_mlb is not None and team_stats_df_mlb.empty) else team_stats_df_mlb

    # Load trained models + selected feature list
    models_mlb, feature_list_mlb = load_trained_models(model_dir, load_feature_list=True)
    if not models_mlb:
        logger.error("No MLB models loaded; aborting.")
        return [], []

    # Load ensemble weights (.json) or fallback
    weights_mlb = FALLBACK_ENSEMBLE_WEIGHTS_MLB.copy()
    wfile_mlb = model_dir / ENSEMBLE_WEIGHTS_FILENAME_MLB
    if wfile_mlb.is_file():
        try:
            raw_weights = json.loads(wfile_mlb.read_text())
            if not isinstance(raw_weights, dict) or not raw_weights:
                raise ValueError("Weights file not a non-empty dict")

            base_weights: Dict[str, float] = {}
            for fullname, val in raw_weights.items():
                base_key = fullname.replace("_mlb_runs_predictor", "")
                if base_key in FALLBACK_ENSEMBLE_WEIGHTS_MLB:
                    base_weights[base_key] = float(val)

            if not base_weights:
                raise ValueError("No valid model weights parsed")

            final_parsed_weights = {
                k: float(base_weights.get(k, 0.0)) for k in FALLBACK_ENSEMBLE_WEIGHTS_MLB.keys()
            }
            s = sum(final_parsed_weights.values())
            if s > 1e-9 and abs(s - 1.0) > 1e-5:
                final_parsed_weights = {k: v / s for k, v in final_parsed_weights.items()}
            elif s <= 1e-9:
                logger.warning(
                    f"Loaded weights sum to zero from {wfile_mlb.name}, using fallback."
                )
                final_parsed_weights = FALLBACK_ENSEMBLE_WEIGHTS_MLB.copy()

            weights_mlb = final_parsed_weights
            logger.info(f"Loaded MLB ensemble weights: {weights_mlb}")
        except Exception as e:
            logger.warning(
                f"{wfile_mlb.name} invalid ({e}); falling back to MLB defaults {FALLBACK_ENSEMBLE_WEIGHTS_MLB}"
            )
            weights_mlb = FALLBACK_ENSEMBLE_WEIGHTS_MLB.copy()
    else:
        logger.warning(
            f"{wfile_mlb.name} not found in {model_dir}; using MLB defaults {weights_mlb}"
        )
    logger.debug(f"Final MLB ensemble weights = {weights_mlb}")

    # --- FEATURE ENGINEERING ---
    # Combine historical + upcoming, but ensure all required columns are present
    if hist_df_mlb is not None:
        # Guarantee required columns exist in both
        required_merge_cols = ["game_id", "game_date", "home_team_id", "away_team_id"]
        for df in (hist_df_mlb, upcoming_df_mlb):
            for c in required_merge_cols:
                if c not in df.columns:
                    df[c] = pd.NA

        combined_input_df = pd.concat(
            [
                hist_df_mlb[required_merge_cols + [col for col in hist_df_mlb.columns if col not in required_merge_cols]],
                upcoming_df_mlb[required_merge_cols + [col for col in upcoming_df_mlb.columns if col not in required_merge_cols]],
            ],
            ignore_index=True,
        ).sort_values(by="game_date").reset_index(drop=True)
    else:
        combined_input_df = upcoming_df_mlb.copy()

    # Run feature pipeline
    try:
        features_all_mlb = run_mlb_feature_pipeline(
            df=combined_input_df,
            team_stats_df=team_stats_df_mlb,
            rolling_windows=[10, 30, 81],
            h2h_lookback_games=10,
            execution_order=DEFAULT_MLB_EXECUTION_ORDER,
            debug=debug_mode,
        )
    except Exception as e:
        logger.error(f"MLB Feature pipeline error: {e}", exc_info=debug_mode)
        return [], []

    # Keep only rows corresponding to upcoming games
    features_df_mlb = features_all_mlb.loc[
        features_all_mlb["game_id"].isin(upcoming_df_mlb["game_id"])
    ].reset_index(drop=True)
    if features_df_mlb.empty:
        logger.error("No features for upcoming MLB games; exiting.")
        return [], []

    # Build X_mlb (only numeric features)
    if feature_list_mlb:
        logger.info(f"Applying {len(feature_list_mlb)} selected MLB features.")
        X_mlb = features_df_mlb.reindex(columns=feature_list_mlb, fill_value=0)
    else:
        logger.warning("mlb_selected_features.json not found. Using all numeric features.")
        non_feature_cols = [
            "game_id",
            "game_date",
            "scheduled_time_utc_dt",
            "home_team_id",
            "away_team_id",
            "home_team_name",
            "away_team_name",
            "home_score",
            "away_score",
        ]
        X_mlb = features_df_mlb.drop(columns=[c for c in non_feature_cols if c in features_df_mlb.columns], errors="ignore")
        X_mlb = X_mlb.select_dtypes(include=np.number)

    X_mlb = X_mlb.fillna(0).replace([np.inf, -np.inf], 0)
    if X_mlb.empty:
        logger.error("Feature matrix X_mlb is empty after processing; exiting.")
        return [], []

    # --- PER‐MODEL PREDICTIONS ---
    preds_by_model_mlb: Dict[str, pd.DataFrame] = {}
    for name, mdl_instance in models_mlb.items():
        try:
            pred_df = mdl_instance.predict(X_mlb)
            if pred_df is None or pred_df.empty:
                logger.warning(f"MLB Model '{name}' predict() returned None or empty.")
                continue

            rename_map = {}
            if "predicted_home_score" in pred_df.columns:
                rename_map["predicted_home_score"] = "predicted_home_runs"
            if "predicted_away_score" in pred_df.columns:
                rename_map["predicted_away_score"] = "predicted_away_runs"
            pred_df = pred_df.rename(columns=rename_map)

            if not {"predicted_home_runs", "predicted_away_runs"}.issubset(pred_df.columns):
                raise KeyError(f"Model '{name}' did not return 'predicted_home_runs' & 'predicted_away_runs'")

            preds_by_model_mlb[name] = pred_df
        except Exception as e:
            logger.error(f"Error in MLB {name}.predict(): {e}", exc_info=debug_mode)

    if not preds_by_model_mlb:
        logger.error("No MLB predictions from any model; exiting.")
        return [], []

    # --- JOIN METADATA BY game_id ---
    meta_df = (
        upcoming_df_mlb[["game_id", "game_date", "home_team_name", "away_team_name"]]
        .set_index("game_id")
    )

    raw_preds_mlb: List[Dict[str, Any]] = []
    for idx, row in features_df_mlb.iterrows():
        gid = str(row["game_id"])
        if gid not in meta_df.index:
            logger.warning(f"Game ID {gid} not found in upcoming metadata. Skipping.")
            continue

        # gather each model's predictions
        component_preds: Dict[str, Dict[str, float]] = {}
        skip_game = False
        for model_key, pred_df in preds_by_model_mlb.items():
            try:
                h_val = float(pred_df.at[idx, "predicted_home_runs"])
                a_val = float(pred_df.at[idx, "predicted_away_runs"])
                component_preds[model_key] = {"home": h_val, "away": a_val}
            except KeyError:
                logger.warning(f"Index {idx} not in predictions for model {model_key}. Skipping game {gid}.")
                skip_game = True
                break
        if skip_game:
            continue

        # Blend using ensemble weights
        w_sum = sum(weights_mlb.get(m, 0.0) for m in component_preds)
        if w_sum < 1e-6:
            h_ens = np.mean([v["home"] for v in component_preds.values()])
            a_ens = np.mean([v["away"] for v in component_preds.values()])
        else:
            h_ens = sum(component_preds[m]["home"] * weights_mlb[m]
                        for m in component_preds if m in weights_mlb) / w_sum
            a_ens = sum(component_preds[m]["away"] * weights_mlb[m]
                        for m in component_preds if m in weights_mlb) / w_sum

        run_diff_ens = h_ens - a_ens
        total_runs_ens = h_ens + a_ens
        win_prob_home_ens = 1 / (1 + math.exp(-0.3 * run_diff_ens))

        meta = meta_df.loc[gid]
        raw_preds_mlb.append({
            "game_id":               gid,
            "game_date":             pd.to_datetime(meta["game_date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team_name":        str(meta["home_team_name"]),
            "away_team_name":        str(meta["away_team_name"]),
            "predicted_home_runs":   round(h_ens, 2),
            "predicted_away_runs":   round(a_ens, 2),
            "predicted_run_diff":    round(run_diff_ens, 2),
            "predicted_total_runs":  round(total_runs_ens, 2),
            "win_probability_home":  round(win_prob_home_ens, 3),
            "raw_predicted_home_runs":  round(h_ens, 2),
            "raw_predicted_away_runs":  round(a_ens, 2),
            "component_predictions": component_preds,
        })

    final_preds_mlb = raw_preds_mlb  # no calibration step
    display_prediction_summary_mlb(final_preds_mlb)
    logger.info(f"MLB Prediction Pipeline completed in {time.time() - start_ts:.1f}s")
    return final_preds_mlb, raw_preds_mlb

# --- Upsert Function (MLB specific) ---
def upsert_run_predictions(predictions: List[Dict[str, Any]], supabase_client: SupabaseClient) -> None:
    if not supabase_client: logger.error("Supabase client not available for upsert."); return
    
    updated_count = 0
    for pred in predictions:
        gid = pred.get("game_id")
        if gid is None: logger.warning(f"Skipping prediction with missing game_id: {pred}"); continue

        # Ensure payload keys match your mlb_game_schedule table's prediction columns
        update_payload = {
            "predicted_home_runs": pred.get("predicted_home_runs"), # Or whatever your DB col is
            "predicted_away_runs": pred.get("predicted_away_runs"), # Or whatever your DB col is
            "predicted_run_diff": pred.get("predicted_run_diff"),
            "predicted_total_runs": pred.get("predicted_total_runs"),
            "win_probability_home": pred.get("win_probability_home"),
            "prediction_utc": datetime.now(pytz.utc).isoformat() # Add timestamp of prediction
        }
        # Filter out None values from payload to avoid overwriting existing DB values with NULL
        update_payload = {k: v for k, v in update_payload.items() if v is not None}

        if not update_payload: logger.warning(f"Empty payload for game_id {gid}. Skipping upsert."); continue

        try:
            # Assuming game_id in mlb_game_schedule is string, if int, cast gid
            # Check if your game_id is string or int in the table
            resp = (
                supabase_client
                .table("mlb_game_schedule") # MLB table
                .update(update_payload) # returning="representation" is optional
                .eq("game_id", str(gid)) # Ensure type match with DB
                .execute()
            )
            # Supabase V2 update doesn't return data by default unless returning="representation"
            # We can check if error is None or count is > 0 if available
            if resp.data : # Or check resp.error or resp.count based on client version
                logger.info(f"Upserted MLB predictions for game_id {gid}.")
                updated_count += 1
            elif resp.error:
                 logger.error(f"Error upserting MLB game_id {gid}: {resp.error}")
            else: # No error, but no data returned (might mean no row matched)
                logger.warning(f"No row found or no data returned from upsert for MLB game_id {gid}.")
        except Exception as e:
            logger.error(f"Exception during upsert for MLB game_id {gid}: {e}", exc_info=True)
    logger.info(f"Finished upserting MLB predictions for {updated_count} games.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate MLB Run Predictions")
    parser.add_argument("--days", type=int, default=DEFAULT_UPCOMING_DAYS_WINDOW_MLB, help=f"Default: {DEFAULT_UPCOMING_DAYS_WINDOW_MLB}")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB, help=f"Default: {DEFAULT_LOOKBACK_DAYS_FOR_FEATURES_MLB}")
    parser.add_argument("--model_dir", type=Path, default=MODELS_DIR, help=f"Default: {MODELS_DIR}")
    # parser.add_argument("--no_calibrate", action="store_true", help="Skip odds calibration") # Calibration removed
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: handler.setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled by CLI argument.")

    if not PROJECT_MODULES_IMPORTED:
        sys.exit("Exiting: Required MLB project modules failed to import.")

    supabase_c = get_supabase_client()
    if not supabase_c:
        sys.exit("Exiting: Supabase client could not be initialized.")

    final_mlb_preds, _ = generate_predictions( # Ignoring raw_preds for now
        days_window=args.days,
        model_dir=args.model_dir,
        historical_lookback=args.lookback,
        debug_mode=args.debug
    )

    if final_mlb_preds:
        logger.info(f"Upserting {len(final_mlb_preds)} final MLB predictions...")
        try:
            upsert_run_predictions(final_mlb_preds, supabase_c) # Pass client
            logger.info("MLB Upsert process completed.")
        except Exception as e:
            logger.error(f"Error during MLB upsert: {e}", exc_info=args.debug)
        if final_mlb_preds: # Check again in case it became empty due to errors
             logger.info("Sample Final MLB Prediction:")
             print(json.dumps(final_mlb_preds[0], indent=2, default=str)) # Add default=str for datetime
    else:
        logger.info("No final MLB predictions generated to upsert.")

if __name__ == "__main__":
    main()