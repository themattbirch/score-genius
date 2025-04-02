# backend/nba_score_prediction/prediction.py
"""
Generates pre-game predictions for upcoming NBA games using on-the-fly feature
generation with NBAFeatureEngine and a static ensemble model (XGBoost, RandomForest, Ridge).
Optionally integrates betting odds and calibrates predictions.
Includes logging for raw component predictions and displays raw vs. final predictions.
"""
import os
import sys
import re
import math
import pytz
import joblib
import traceback
import logging
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text # Added text for literal SQL
from typing import List, Dict, Optional, Any, Tuple

# --- Add backend directory to sys.path ---
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
# --- End Path Setup ---

# --- Project Module Imports ---
try:
    from nba_score_prediction.feature_engineering import NBAFeatureEngine # Import Feature Engine
    from nba_score_prediction.models import XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor # Import specific predictors
    from caching.supabase_client import supabase as supabase_client_instance # Renamed to avoid conflict
    from backend import config # Import the config module
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Error importing project modules: {e}. Prediction script may fail.", exc_info=True)
    # Define dummy classes if imports fail to allow script to load
    PROJECT_MODULES_IMPORTED = False
    class NBAFeatureEngine:
        def __init__(self, *args, **kwargs): self.defaults={'avg_pts_for': 115.0} # Add default for fallback
        def generate_all_features(self, df, **kwargs):
             logging.error("Dummy NBAFeatureEngine used - generate_all_features returning input df.")
             return df # Return input to avoid immediate crash, but prediction will fail
    class _DummyBasePredictor:
         feature_names_in_ = []
         training_timestamp = None
         def load_model(self, *args, **kwargs): raise NotImplementedError("Dummy model - real import failed")
         def predict(self, *args, **kwargs): raise NotImplementedError("Dummy model - real import failed")
    class XGBoostScorePredictor(_DummyBasePredictor): pass
    class RandomForestScorePredictor(_DummyBasePredictor): pass
    class RidgeScorePredictor(_DummyBasePredictor): pass
    config = None
    supabase_client_instance = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', BACKEND_DIR / 'models' / 'saved')) # Use config or default
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Static weights for the pre-game ensemble (align with model_inference.py)
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,
    "random_forest": 0.30,
    "ridge": 0.20
}

TARGET_COLUMNS = ['home_score', 'away_score']

# Timezone and other constants
PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180 # How much history needed for rolling features etc.
DEFAULT_UPCOMING_DAYS_WINDOW = 2 # How many days ahead to predict

REQUIRED_HISTORICAL_COLS = [ # Columns needed from historical data for feature engine
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot',
        'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_ot',
        'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
        'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
        'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
        'home_off_reb', 'home_def_reb', 'home_total_reb',
        'away_off_reb', 'away_def_reb', 'away_total_reb',
        'home_turnovers', 'away_turnovers',
        'home_assists', 'home_steals', 'home_blocks', 'home_fouls',
        'away_assists', 'away_steals', 'away_blocks', 'away_fouls',
    ]
REQUIRED_TEAM_STATS_COLS = [ # Columns needed from team stats for feature engine
        'team_name', 'season', 'wins_all_percentage', 'points_for_avg_all',
        'points_against_avg_all', 'current_form'
    ]
UPCOMING_GAMES_COLS = ["game_id", "scheduled_time", "home_team", "away_team"] # Base columns from schedule

# --- Database and Supabase Initialization ---
def get_supabase_client():
    """Returns the imported Supabase client instance if available."""
    if supabase_client_instance:
        logger.debug("Using imported Supabase client instance.")
        return supabase_client_instance
    else:
        logger.warning("Supabase client instance not available from import.")
        return None

# --- Data Loading Functions ---
def load_recent_historical_data(supabase_client, days_lookback: int) -> pd.DataFrame:
    # ... (implementation remains the same as previous version) ...
    if not supabase_client: logger.error("Supabase client unavailable."); return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical game data from {start_date} onwards from Supabase table 'nba_historical_game_stats'...")
    select_cols_str = ", ".join(REQUIRED_HISTORICAL_COLS)
    all_historical_data = []; page_size = 1000; start_index = 0; has_more = True
    try:
        while has_more:
            logger.debug(f"Fetching historical data batch starting at index {start_index}...")
            response = supabase_client.table("nba_historical_game_stats").select(select_cols_str).gte("game_date", start_date).order('game_date', desc=False).range(start_index, start_index + page_size - 1).execute()
            batch = response.data; b_size = len(batch); all_historical_data.extend(batch); logger.debug(f"Retrieved {b_size} historical records in this batch.")
            if b_size < page_size: has_more = False
            else: start_index += page_size
        if not all_historical_data: logger.warning(f"No historical game data found since {start_date}."); return pd.DataFrame()
        historical_df = pd.DataFrame(all_historical_data); logger.info(f"Loaded {len(historical_df)} total historical games from Supabase.")
        if 'game_date' in historical_df.columns: historical_df['game_date'] = pd.to_datetime(historical_df['game_date'], errors='coerce').dt.tz_localize(None); historical_df = historical_df.dropna(subset=['game_date'])
        else: logger.error("Critical column 'game_date' missing."); return pd.DataFrame()
        numeric_cols = [col for col in REQUIRED_HISTORICAL_COLS if col not in ['game_id', 'game_date', 'home_team', 'away_team']]
        for col in numeric_cols:
             if col in historical_df.columns: historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce').fillna(0)
             else: logger.warning(f"Required historical column '{col}' missing."); historical_df[col] = 0
        for col in ['game_id', 'home_team', 'away_team']:
             if col in historical_df.columns: historical_df[col] = historical_df[col].astype(str)
             else: historical_df[col] = ""
        return historical_df.sort_values(by='game_date').reset_index(drop=True)
    except Exception as e: logger.error(f"Error loading historical data from Supabase: {e}", exc_info=True); return pd.DataFrame()

def load_team_stats_data(supabase_client) -> pd.DataFrame:
    # ... (implementation remains the same as previous version) ...
    if not supabase_client: logger.error("Supabase client unavailable."); return pd.DataFrame()
    logger.info("Loading team stats data from Supabase table 'nba_historical_team_stats'...")
    select_cols_str = ", ".join(REQUIRED_TEAM_STATS_COLS)
    try:
        response = supabase_client.table("nba_historical_team_stats").select(select_cols_str).execute()
        data = response.data
        if not data: logger.warning("No team stats data found."); return pd.DataFrame()
        team_stats_df = pd.DataFrame(data); logger.info(f"Loaded {len(team_stats_df)} team stat records.")
        numeric_cols = [col for col in REQUIRED_TEAM_STATS_COLS if col not in ['team_name', 'season', 'current_form']]
        for col in numeric_cols:
            if col in team_stats_df.columns: team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
            else: logger.warning(f"Required team stat column '{col}' missing."); team_stats_df[col] = 0.0
        for col in ['team_name', 'season', 'current_form']:
             if col in team_stats_df.columns: team_stats_df[col] = team_stats_df[col].astype(str).fillna('')
             else: team_stats_df[col] = ''
        return team_stats_df
    except Exception as e: logger.error(f"Error loading team stats from Supabase: {e}", exc_info=True); return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client, days_window: int) -> pd.DataFrame:
    # ... (implementation remains the same as previous version) ...
    if not supabase_client: logger.error("Supabase client unavailable."); return pd.DataFrame()
    now_pt = datetime.now(PACIFIC_TZ)
    start_utc_str = now_pt.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S%z'); end_utc_str = (now_pt + timedelta(days=days_window)).astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S%z')
    logger.info(f"Fetching upcoming games scheduled between {start_utc_str} and {end_utc_str} (UTC)...")
    select_cols_str = ", ".join(UPCOMING_GAMES_COLS)
    try:
        response = supabase_client.table("nba_game_schedule").select(select_cols_str).gte("scheduled_time", start_utc_str).lt("scheduled_time", end_utc_str).order("scheduled_time", desc=False).execute()
        data = response.data
        if not data: logger.warning("No upcoming games found."); return pd.DataFrame()
        games_df = pd.DataFrame(data); logger.info(f"Fetched {len(games_df)} upcoming game schedules.")
        if "scheduled_time" in games_df.columns:
             games_df["scheduled_time_utc"] = pd.to_datetime(games_df["scheduled_time"], errors='coerce', utc=True); games_df = games_df.dropna(subset=['scheduled_time_utc'])
             games_df["game_time_pt"] = games_df["scheduled_time_utc"].dt.tz_convert(PACIFIC_TZ); games_df["game_date"] = pd.to_datetime(games_df["game_time_pt"].dt.date)
        else: logger.error("Column 'scheduled_time' not found."); return pd.DataFrame()
        for col in ["home_team", "away_team"]: games_df[col] = games_df[col].astype(str)
        games_df['game_id'] = games_df['game_id'].astype(str)
        stats_cols_to_add = set(REQUIRED_HISTORICAL_COLS) - set(games_df.columns)
        for col in stats_cols_to_add:
             if col not in ['game_id', 'game_date', 'home_team', 'away_team']: games_df[col] = np.nan
        logger.info(f"Prepared {len(games_df)} upcoming games with necessary columns.")
        return games_df.sort_values(by='game_time_pt').reset_index(drop=True)
    except Exception as e: logger.error(f"Error fetching upcoming games from Supabase: {e}", exc_info=True); return pd.DataFrame()

# --- Betting Odds Parsing (No Changes Needed) ---
def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    if not line_str:
        return None, None
    line, odds = None, None
    match = re.search(
        r"([\+\-]?\d+(?:\.\d+)?)\s*(?:\(?(?:[ou]?)\s*([\+\-]\d+)\)?)?",
        str(line_str).strip()
    )
    if match:
        try:
            line = float(match.group(1))
        except (ValueError, TypeError):
            pass
        if match.group(2):
            try:
                odds = int(match.group(2))
            except (ValueError, TypeError):
                pass
        else:
            odds = -110
    return line, odds
def parse_moneyline_str(ml_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[int], Optional[int]]:
    # ... (implementation remains the same) ...
    home_ml, away_ml = None, None;
    if not ml_str or not isinstance(ml_str, str): return home_ml, away_ml
    try:
        parts = ml_str.split('/'); team_map = {}
        if len(parts) != 2: return home_ml, away_ml
        for part in parts:
            part = part.strip(); match_num = re.search(r"([\+\-]\d+)$", part)
            if match_num:
                ml_value = int(match_num.group(1)); team_name_part = part.replace(match_num.group(0), '').strip()
                if home_team.lower() in team_name_part.lower(): team_map['home'] = ml_value
                elif away_team.lower() in team_name_part.lower(): team_map['away'] = ml_value
                elif 'home' not in team_map and 'away' not in team_map: team_map['home' if ml_value < 0 else 'away'] = ml_value
        home_ml = team_map.get('home'); away_ml = team_map.get('away')
    except Exception as e: logger.warning(f"Could not parse moneyline string '{ml_str}': {e}")
    return home_ml, away_ml
def parse_spread_str(spread_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    # ... (implementation remains the same) ...
    home_line, away_line = None, None
    if not spread_str or not isinstance(spread_str, str): return home_line, away_line
    try:
        parts = spread_str.split('/'); team_map = {}
        if len(parts) != 2: return home_line, away_line
        for part in parts:
             part = part.strip(); match_num = re.search(r"([\+\-]?\d+(?:\.\d+)?)$", part)
             if match_num:
                  line_value = float(match_num.group(1)); team_name_part = part.replace(match_num.group(0), '').strip()
                  if home_team.lower() in team_name_part.lower(): team_map['home'] = line_value
                  elif away_team.lower() in team_name_part.lower(): team_map['away'] = line_value
                  elif 'home' not in team_map and 'away' not in team_map: team_map['home'] = line_value
        home_line = team_map.get('home'); away_line = team_map.get('away')
        if home_line is not None and away_line is None: away_line = -home_line
        elif away_line is not None and home_line is None: home_line = -away_line
        elif home_line is not None and away_line is not None and abs(home_line + away_line) > 0.1: logger.warning(f"Inconsistent spread: H={home_line}, A={away_line}"); away_line = -home_line
    except Exception as e: logger.warning(f"Could not parse spread string '{spread_str}': {e}")
    return home_line, away_line
def parse_total_str(total_str: Optional[str]) -> Optional[float]:
    # ... (implementation remains the same) ...
    total_line = None
    if not total_str or not isinstance(total_str, str): return total_line
    try:
        numbers = re.findall(r"(\d+(?:\.\d+)?)", total_str)
        if numbers: total_line = float(numbers[0])
    except Exception as e: logger.warning(f"Could not parse total string '{total_str}': {e}")
    return total_line
def fetch_and_parse_betting_odds(supabase_client, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    # ... (implementation remains the same as previous version) ...
    if not supabase_client or not game_ids: logger.warning("Skipping odds fetch."); return {}
    odds_dict = {}; chunk_size = 50; game_id_chunks = [game_ids[i:i + chunk_size] for i in range(0, len(game_ids), chunk_size)]
    logger.info(f"Fetching betting odds from 'nba_game_schedule' for {len(game_ids)} game IDs...")
    select_cols = "game_id, home_team, away_team, moneyline_clean, spread_clean, total_clean"
    for i, chunk in enumerate(game_id_chunks):
        logger.debug(f"Fetching odds chunk {i+1}/{len(game_id_chunks)}...")
        try:
            response = supabase_client.table("nba_game_schedule").select(select_cols).in_("game_id", chunk).execute()
            if not response.data: continue
            df = pd.DataFrame(response.data)
            for _, row in df.iterrows():
                game_id = str(row.get('game_id')); home_team = row.get('home_team', ''); away_team = row.get('away_team', '')
                if not game_id: continue
                home_ml, away_ml = parse_moneyline_str(row.get('moneyline_clean'), home_team, away_team)
                home_spread, away_spread = parse_spread_str(row.get('spread_clean'), home_team, away_team)
                total_line = parse_total_str(row.get('total_clean')); default_odds = -110
                odds_dict[game_id] = {'moneyline': {'home': home_ml, 'away': away_ml}, 'spread': {'home_line': home_spread, 'away_line': away_spread, 'home_odds': default_odds, 'away_odds': default_odds}, 'total': {'line': total_line, 'over_odds': default_odds, 'under_odds': default_odds}, 'bookmaker': 'Parsed from Supabase', 'last_update': pd.Timestamp.now(tz='UTC')}
            logger.debug(f"Processed odds for {len(df)} games in chunk.")
        except Exception as e: logger.error(f"Error fetching/parsing betting odds chunk: {e}", exc_info=True)
    logger.info(f"Finished fetching odds. Parsed data for {len(odds_dict)} games.")
    return odds_dict

# --- Model Loading Function (No Changes Needed)---
def load_trained_models(model_dir: Path = MODELS_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    # ... (implementation remains the same as previous version) ...
    models = {}; model_configs = {"xgboost": XGBoostScorePredictor, "random_forest": RandomForestScorePredictor, "ridge": RidgeScorePredictor}
    loaded_feature_names = None; all_models_loaded = True; inconsistent_features_found = False
    for name, PredictorClass in model_configs.items():
        if not PROJECT_MODULES_IMPORTED or getattr(PredictorClass, '__module__', '').startswith('_Dummy'): logger.error(f"Cannot load '{name}': Module not imported."); all_models_loaded = False; continue
        try:
            logger.info(f"Loading latest '{name}' model component from: {model_dir}")
            predictor = PredictorClass(model_dir=str(model_dir), model_name=f"{name}_score_predictor"); predictor.load_model()
            models[name] = predictor; logger.info(f"Loaded {name} trained {predictor.training_timestamp or 'unknown'}.")
            current_features = getattr(predictor, 'feature_names_in_', None)
            if loaded_feature_names is None and current_features: loaded_feature_names = current_features; logger.info(f"Using feature list from '{name}' ({len(loaded_feature_names)} features).")
            elif loaded_feature_names and current_features != loaded_feature_names: logger.error(f"CRITICAL Feature mismatch! '{name}' (features: {len(current_features)}) != '{list(models.keys())[0]}' ({len(loaded_feature_names)}). Retrain models consistently."); inconsistent_features_found = True; all_models_loaded = False
        except FileNotFoundError: logger.error(f"No model file found for '{name}' in {model_dir}."); all_models_loaded = False
        except Exception as e: logger.error(f"Error loading model '{name}': {e}", exc_info=True); all_models_loaded = False
    if inconsistent_features_found: logger.error("Aborting due to inconsistent features."); return None, None
    if not all_models_loaded or not models: logger.error("Failed to load all models."); return None, None
    if not loaded_feature_names: logger.error("Failed to extract required feature list from models."); return models, None
    logger.info(f"All models loaded successfully with consistent feature list ({len(loaded_feature_names)} features).")
    return models, loaded_feature_names

# --- Calibration Function (UPDATED) ---
def calibrate_prediction_with_odds(
    prediction: Dict, # A single raw prediction dict
    odds_info: Optional[Dict], # Odds dict for this game_id
    blend_factor: float = 0.3
    ) -> Dict:
    """
    Calibrates a prediction using market odds. Adds 'raw_' prefixed keys
    to store the original model prediction values before blending.
    """
    calibrated = prediction.copy()

    # Store raw values FIRST, before attempting calibration
    calibrated['raw_predicted_home_score'] = prediction.get('predicted_home_score')
    calibrated['raw_predicted_away_score'] = prediction.get('predicted_away_score')
    calibrated['raw_predicted_point_diff'] = prediction.get('predicted_point_diff')
    calibrated['raw_predicted_total_score'] = prediction.get('predicted_total_score')
    calibrated['raw_win_probability'] = prediction.get('win_probability')

    calibrated['betting_odds'] = odds_info # Store odds info used (or None)
    calibrated['calibration_blend_factor'] = blend_factor
    calibrated['is_calibrated'] = False # Default to False

    if not odds_info:
        logger.debug(f"No odds info for game {prediction.get('game_id')}. Calibration skipped.")
        return calibrated # Return dict with raw values in main keys, raw_ keys, and is_calibrated=False

    try:
        # Extract Market Lines
        market_spread, market_total = None, None
        spread_data = odds_info.get('spread', {})
        if spread_data.get('home_line') is not None: market_spread = -spread_data['home_line'] # Market spread perspective (negative of home line)
        elif spread_data.get('away_line') is not None: market_spread = spread_data['away_line']
        total_data = odds_info.get('total', {}); market_total = total_data.get('line')

        if market_spread is None or market_total is None:
             logger.warning(f"Market spread ({market_spread}) or total ({market_total}) is None for {prediction.get('game_id')}. Cannot calibrate.")
             return calibrated # Return with raw values, is_calibrated=False

        logger.info(f"Calibrating game {prediction.get('game_id')} with blend_factor: {blend_factor}")

        # Use stored raw predictions for blending
        pred_diff = calibrated['raw_predicted_point_diff']
        pred_total = calibrated['raw_predicted_total_score']

        # Simple Blend Calibration
        cal_total = (blend_factor * market_total) + ((1 - blend_factor) * pred_total)
        cal_diff = (blend_factor * market_spread) + ((1 - blend_factor) * pred_diff)

        # Recalculate home/away scores based on calibrated total and diff
        cal_home = (cal_total + cal_diff) / 2
        cal_away = cal_total - cal_home

        # Recalculate win probability based on calibrated difference
        cal_win_prob = 1 / (1 + math.exp(-0.15 * cal_diff)) # Use same factor

        # Update PRIMARY Dictionary Keys with Calibrated Values
        calibrated['predicted_home_score'] = round(cal_home, 1)
        calibrated['predicted_away_score'] = round(cal_away, 1)
        calibrated['predicted_point_diff'] = round(cal_diff, 1)
        calibrated['predicted_total_score'] = round(cal_total, 1)
        calibrated['win_probability'] = round(cal_win_prob, 3)
        calibrated['is_calibrated'] = True # Mark as calibrated

        logger.info(f" -> Calibrated: H {cal_home:.1f}-A {cal_away:.1f} (Diff: {cal_diff:+.1f}, Total: {cal_total:.1f}, WP: {cal_win_prob:.1%})")

    except Exception as e:
        logger.error(f"Error during calibration for {prediction.get('game_id')}: {e}", exc_info=True)
        # is_calibrated remains False, primary keys still hold raw values
    return calibrated

# --- Reporting Utility (UPDATED) ---
def display_prediction_summary(predictions: List[Dict]):
    """
    Displays a summary of predictions, including raw vs. final (calibrated)
    scores and diffs, along with betting edges.
    """
    if not predictions:
        logger.info("No predictions to display.")
        return

    # Prepare header with Raw columns
    header_line_1 = "="*115
    header_title = " " * 40 + "NBA PREGAME PREDICTION SUMMARY"
    header_cols = (f"{'DATE':<11} {'MATCHUP':<30} {'FINAL SCORE':<12} {'FINAL DIFF':<10} "
                   f"{'RAW SCORE':<12} {'RAW DIFF':<10} {'WIN PROB':<10} {'CALIB?':<6}")
    header_sep = "-"*115

    print(f"\n{header_line_1}\n{header_title}\n{header_line_1}")
    print(header_cols)
    print(header_sep)

    try:
        predictions_df = pd.DataFrame(predictions)
        predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'], errors='coerce')
        predictions_df = predictions_df.dropna(subset=['game_date'])
        # Sort by date then maybe predicted diff magnitude?
        predictions_df = predictions_df.sort_values(['game_date', 'predicted_point_diff'], key=lambda x: abs(x) if x.name == 'predicted_point_diff' else x)
    except Exception as e:
        logger.error(f"Error sorting predictions for display: {e}")
        predictions_df = pd.DataFrame(predictions) # Fallback to unsorted

    for _, game in predictions_df.iterrows():
        try:
            date_str = game['game_date'].strftime('%Y-%m-%d') if pd.notna(game['game_date']) else "N/A"
            matchup = f"{game.get('home_team','?')[:14]} vs {game.get('away_team','?')[:14]}"[:30] # Truncate team names too

            is_calib = game.get('is_calibrated', False)
            calib_str = "Yes" if is_calib else "No "

            # Final values (might be raw if not calibrated)
            final_score = f"{game.get('predicted_home_score', np.nan):.1f}-{game.get('predicted_away_score', np.nan):.1f}"
            final_diff = game.get('predicted_point_diff', np.nan)
            final_diff_str = f"{final_diff:<+10.1f}" if pd.notna(final_diff) else f"{'N/A':<10}"
            final_win_prob = game.get('win_probability', np.nan)
            win_prob_str = f"{final_win_prob*100:.1f}%" if pd.notna(final_win_prob) else f"{'N/A':<10}"

            # Raw values (should exist due to change in calibrate func)
            raw_home = game.get('raw_predicted_home_score')
            raw_away = game.get('raw_predicted_away_score')
            raw_score = f"{raw_home:.1f}-{raw_away:.1f}" if pd.notna(raw_home) and pd.notna(raw_away) else "N/A"

            raw_diff = game.get('raw_predicted_point_diff', np.nan)
            raw_diff_str = f"{raw_diff:<+10.1f}" if pd.notna(raw_diff) else f"{'N/A':<10}"

            # Print main line with Raw vs Final comparison
            print(f"{date_str:<11} {matchup:<30} {final_score:<12} {final_diff_str:<10} "
                  f"{raw_score:<12} {raw_diff_str:<10} {win_prob_str:<10} {calib_str:<6}")

            # Betting edge logic (uses final/calibrated values)
            odds_info = game.get('betting_odds')
            if odds_info:
                 market_spread, market_total = None, None
                 spread_data = odds_info.get('spread', {})
                 if spread_data.get('home_line') is not None: market_spread = -spread_data['home_line']
                 elif spread_data.get('away_line') is not None: market_spread = spread_data['away_line']
                 if odds_info.get('total'): market_total = odds_info['total'].get('line')

                 spread_edge_str, total_edge_str = "N/A", "N/A"
                 # Ensure final_diff and final_total are valid numbers before calculating edge
                 final_total = game.get('predicted_total_score', np.nan)
                 if market_spread is not None and pd.notna(final_diff): spread_edge = final_diff - market_spread; spread_edge_str = f"{spread_edge:+.1f}"
                 if market_total is not None and pd.notna(final_total): total_edge = final_total - market_total; total_edge_str = f"{total_edge:+.1f}"

                 market_spread_disp = f"{market_spread:.1f}" if market_spread is not None else "N/A"
                 market_total_disp = f"{market_total:.1f}" if market_total is not None else "N/A"
                 print(f"  -> Betting Edge: Spread={spread_edge_str}, Total={total_edge_str} (vs Market Spd={market_spread_disp}, Tot={market_total_disp})")
            elif is_calib: print("  -> Betting Edge: Odds Used for Calibration") # If calibrated but odds dict somehow missing
            else: print("  -> Betting Edge: Market Odds Unavailable / Calibration Skipped")

        except Exception as display_e:
            logger.error(f"Error displaying summary row for game {game.get('game_id', 'N/A')}: {display_e}")

    print("="*115) # Match header length

# --- Main Prediction Orchestration Function (UPDATED) ---
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW,
    model_dir: Path = MODELS_DIR,
    calibrate_with_odds: bool = True,
    blend_factor: float = 0.3,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES
    ) -> Tuple[List[Dict], List[Dict]]:
    """
    Orchestrates the prediction process using on-the-fly feature generation.
    Fetches data, generates features, loads models, predicts, optionally calibrates.
    Returns:
        Tuple: (list of final prediction dictionaries, list of raw prediction dictionaries before calibration)
    """
    logger.info("--- Starting NBA Prediction Pipeline (On-the-Fly Features) ---")
    start_time = time.time()
    if not PROJECT_MODULES_IMPORTED: logger.critical("Core project modules not imported."); return [], []

    # --- Setup ---
    supabase_client = get_supabase_client()
    if not supabase_client: logger.error("Supabase Client initialization failed."); return [], []
    try: feature_engine = NBAFeatureEngine(supabase_client=supabase_client, debug=False) # Instantiate feature engine
    except Exception as fe_e: logger.error(f"Failed to initialize NBAFeatureEngine: {fe_e}"); return [], []

    # --- Load Data ---
    logger.info("Step 1: Loading data...")
    historical_data = load_recent_historical_data(supabase_client, days_lookback=historical_lookback)
    team_stats_data = load_team_stats_data(supabase_client)
    upcoming_games_df = fetch_upcoming_games_data(supabase_client, days_window=days_window)
    if upcoming_games_df.empty: logger.warning("No upcoming games found."); return [], []
    if historical_data.empty: logger.warning("Historical data empty; feature quality reduced.")
    if team_stats_data.empty: logger.warning("Team stats data empty; context features reduced.")

    # --- Load Models ---
    logger.info("Step 2: Loading trained models...")
    try: prediction_models, required_features = load_trained_models(model_dir)
    except Exception as load_e: logger.error(f"Model loading failed: {load_e}", exc_info=True); return [], []
    if not prediction_models or not required_features: logger.error("Models or required features list missing after load attempt."); return [], []

    # --- Feature Generation ---
    logger.info("Step 3: Generating features for prediction...")
    try:
        logger.debug(f"Combining {len(historical_data)} historical and {len(upcoming_games_df)} upcoming games for feature gen.")
        # Ensure required columns exist in both dataframes before concat
        all_cols_needed = set(REQUIRED_HISTORICAL_COLS) | set(REQUIRED_TEAM_STATS_COLS) | set(UPCOMING_GAMES_COLS)
        for df, name in [(historical_data, 'historical'), (upcoming_games_df, 'upcoming')]:
            if df is None:
                continue
            for col in all_cols_needed:
                if col not in df.columns:
                    df[col] = np.nan if col not in ['game_id', 'home_team', 'away_team', 'game_date', 'season', 'current_form'] else ''
        # Filter only relevant columns before concat to avoid potential type issues from extra cols
        hist_cols_to_concat = [c for c in historical_data.columns if c in all_cols_needed or c in TARGET_COLUMNS]  # Keep targets if present in hist
        upcoming_cols_to_concat = [c for c in upcoming_games_df.columns if c in all_cols_needed]
        common_cols = list(set(hist_cols_to_concat) & set(upcoming_cols_to_concat))

        combined_df_for_features = pd.concat(
            [historical_data[common_cols], upcoming_games_df[common_cols]],
            ignore_index=True
        ).sort_values(by='game_date', kind='mergesort').reset_index(drop=True)  # Use stable sort

        prediction_rolling_windows = [5, 10, 20]  # <<< HARDCODED - MATCH TRAINING ARGUMENTS
        logger.info(f"Using rolling windows {prediction_rolling_windows} for prediction feature generation.")
        features_df_full = feature_engine.generate_all_features(
            df=combined_df_for_features,
            historical_games_df=historical_data.copy() if not historical_data.empty else None,
            team_stats_df=team_stats_data.copy() if not team_stats_data.empty else None,
            rolling_windows=prediction_rolling_windows
        )
        if features_df_full.empty:
            raise RuntimeError("Feature generation returned empty dataframe.")
    except Exception as feat_e:
        logger.error(f"Feature generation failed: {feat_e}", exc_info=True)
        return [], []

    # Define predict_features_df by filtering features_df_full for upcoming games
    upcoming_game_ids = upcoming_games_df['game_id'].unique()
    predict_features_df = features_df_full[features_df_full['game_id'].isin(upcoming_game_ids)].copy()
    logger.info(f"Features generated and isolated for {len(predict_features_df)} upcoming games.")

    # --- Feature Selection and Validation ---
    logger.info(f"Step 4: Selecting and validating {len(required_features)} features...") # Use count from loaded list
    missing_in_df = [f for f in required_features if f not in predict_features_df.columns]
    if missing_in_df: logger.error(f"FATAL: Generated features missing required model columns: {missing_in_df}"); return [], []
    X_predict = predict_features_df[required_features].copy()
    if X_predict.isnull().any().any():
        nan_cols = X_predict.columns[X_predict.isnull().any()].tolist(); logger.warning(f"NaNs found in features: {nan_cols}. Filling with 0.0."); X_predict = X_predict.fillna(0.0)

    # --- Generate Raw Ensemble Predictions ---
    logger.info("Step 5: Generating raw predictions from ensemble components...")
    raw_predictions_list = []
    component_predictions_all_games = {}
    model_preds_dfs = {}
    all_components_successful = True
    for name, predictor in prediction_models.items():
         try:
              preds_df = predictor.predict(X_predict)
              if preds_df is None or preds_df.empty or 'predicted_home_score' not in preds_df.columns: raise ValueError("Invalid prediction output")
              preds_df.index = X_predict.index; model_preds_dfs[name] = preds_df[['predicted_home_score', 'predicted_away_score']]
         except Exception as e: logger.error(f"Error predicting with {name}: {e}", exc_info=True); model_preds_dfs[name] = None; all_components_successful = False

    # Assemble raw predictions game by game
    for idx in X_predict.index:
         game_info = predict_features_df.loc[idx]; game_id = str(game_info['game_id'])
         home_team = game_info['home_team']; away_team = game_info['away_team']; game_date = pd.to_datetime(game_info['game_date'])
         component_predictions_this_game = {}; ensemble_home_score = 0.0; ensemble_away_score = 0.0; total_weight = 0.0
         for name, preds_df in model_preds_dfs.items():
              home_pred_game, away_pred_game = np.nan, np.nan
              if preds_df is not None and idx in preds_df.index: home_pred_game = preds_df.loc[idx, 'predicted_home_score']; away_pred_game = preds_df.loc[idx, 'predicted_away_score']
              else: avg_score = feature_engine.defaults.get('avg_pts_for', 115.0); home_pred_game = avg_score; away_pred_game = avg_score; logger.debug(f"Using fallback for {name} on game {game_id}")
              component_predictions_this_game[name] = {'home': home_pred_game, 'away': away_pred_game}
              weight = ENSEMBLE_WEIGHTS.get(name, 0)
              if weight > 0 and not (np.isnan(home_pred_game) or np.isnan(away_pred_game)): ensemble_home_score += home_pred_game * weight; ensemble_away_score += away_pred_game * weight; total_weight += weight
         if total_weight > 1e-6: ensemble_home_score /= total_weight; ensemble_away_score /= total_weight
         else:
             logger.warning(f"Total weight zero for game {game_id}. Using simple average fallback."); home_scores = [p['home'] for p in component_predictions_this_game.values() if not np.isnan(p['home'])]; away_scores = [p['away'] for p in component_predictions_this_game.values() if not np.isnan(p['away'])]
             avg_score = feature_engine.defaults.get('avg_pts_for', 115.0); ensemble_home_score = np.mean(home_scores) if home_scores else avg_score; ensemble_away_score = np.mean(away_scores) if away_scores else avg_score
         point_diff = ensemble_home_score - ensemble_away_score; win_prob = 1 / (1 + math.exp(-0.15 * point_diff))
         raw_pred_dict = {'game_id': game_id, 'game_date': game_date.strftime('%Y-%m-%d'), 'home_team': home_team, 'away_team': away_team, 'predicted_home_score': round(ensemble_home_score, 1), 'predicted_away_score': round(ensemble_away_score, 1), 'predicted_point_diff': round(point_diff, 1), 'predicted_total_score': round(ensemble_home_score + ensemble_away_score, 1), 'win_probability': round(win_prob, 3), 'component_predictions': {name: {'home': round(preds['home'],1), 'away': round(preds['away'],1)} for name, preds in component_predictions_this_game.items()}, 'component_weights': ENSEMBLE_WEIGHTS, 'is_calibrated': False}
         raw_predictions_list.append(raw_pred_dict); component_predictions_all_games[game_id] = component_predictions_this_game

    if not raw_predictions_list: logger.error("No raw predictions generated."); return [], []

    # --- Log Raw Component Predictions ---
    logger.info("--- Raw Component Predictions (Before Calibration) ---")
    for raw_pred in raw_predictions_list:
        game_id = raw_pred.get('game_id', 'N/A')
        comp_preds = raw_pred.get('component_predictions', {})
        log_str = f"Game {game_id} ({raw_pred.get('home_team', '')} vs {raw_pred.get('away_team', '')}): "
        preds_parts = []
        # Use ENSEMBLE_WEIGHTS keys for consistent order if possible
        for model_name in ENSEMBLE_WEIGHTS.keys():
            if model_name in comp_preds:
                 pred_vals = comp_preds[model_name]
                 home_val = pred_vals.get('home', 'NaN'); away_val = pred_vals.get('away', 'NaN')
                 home_str = f"{home_val:.1f}" if isinstance(home_val, (int, float)) else str(home_val)
                 away_str = f"{away_val:.1f}" if isinstance(away_val, (int, float)) else str(away_val)
                 preds_parts.append(f"{model_name.split('_')[0].upper()}=H{home_str}/A{away_str}")
            else:
                 preds_parts.append(f"{model_name.split('_')[0].upper()}=N/A") # Log if component missing
        log_str += ", ".join(preds_parts)
        logger.info(log_str)
    logger.info("--- End Raw Component Predictions ---")

    # --- Fetch Odds & Calibrate (Optional) ---
    logger.info("Step 6: Fetching odds and calibrating predictions (optional)...")
    final_predictions = []; odds_dict = {}
    if calibrate_with_odds:
        game_ids_to_predict = [str(p['game_id']) for p in raw_predictions_list]
        odds_dict = fetch_and_parse_betting_odds(supabase_client, game_ids_to_predict)
        if not odds_dict: logger.warning("No betting odds fetched. Using raw predictions.")
        logger.info(f"Calibrating {len(raw_predictions_list)} predictions with odds...")
        for pred in raw_predictions_list:
             game_id = str(pred.get('game_id')); odds_info = odds_dict.get(game_id)
             calibrated_pred = calibrate_prediction_with_odds(pred, odds_info, blend_factor) # Adds raw_ keys
             final_predictions.append(calibrated_pred)
    else:
         logger.info("Skipping odds fetching and calibration.")
         # Need to manually add raw_ keys if calibration is skipped
         for pred in raw_predictions_list:
             pred_with_raw = pred.copy()
             pred_with_raw['raw_predicted_home_score'] = pred.get('predicted_home_score')
             pred_with_raw['raw_predicted_away_score'] = pred.get('predicted_away_score')
             pred_with_raw['raw_predicted_point_diff'] = pred.get('predicted_point_diff')
             pred_with_raw['raw_predicted_total_score'] = pred.get('predicted_total_score')
             pred_with_raw['raw_win_probability'] = pred.get('win_probability')
             pred_with_raw['is_calibrated'] = False # Ensure it's marked false
             final_predictions.append(pred_with_raw)

    # --- Display Summary & Return ---
    logger.info("Step 7: Displaying prediction summary...")
    display_prediction_summary(final_predictions) # Display final (calibrated or raw with raw_ keys)
    end_time = time.time()
    logger.info(f"--- NBA Prediction Pipeline Finished in {end_time - start_time:.2f} seconds ---")
    return final_predictions, raw_predictions_list # Return final list (with raw_ keys) and original raw list

# --- Script Execution ---
if __name__ == "__main__":
    logger.info("--- Running Prediction Script Directly ---")
    # Define parameters for the prediction run
    config_days_window = 2
    config_calibrate = True # Set to False if you don't want odds calibration
    config_blend_factor = 0.3 # Lower blend factor = more trust in model
    config_hist_lookback = 365 # How much history to use for feature gen
    # Run the main prediction function
    final_preds, raw_preds = generate_predictions(
        days_window=config_days_window,
        model_dir=MODELS_DIR,
        calibrate_with_odds=config_calibrate,
        blend_factor=config_blend_factor,
        historical_lookback=config_hist_lookback
    )
    # --- Post-Prediction Actions (Saving to CSV / Display) ---
    if final_preds:
        logger.info(f"Successfully generated {len(final_preds)} predictions.")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"predictions_output_{ts}.csv"
            df_to_save = pd.DataFrame(final_preds)
            # Clean up complex columns before saving to CSV if necessary
            cols_to_drop_for_csv = ['component_predictions', 'component_weights', 'betting_odds']
            df_to_save = df_to_save.drop(columns=cols_to_drop_for_csv, errors='ignore')
            df_to_save.to_csv(output_path, index=False)
            logger.info(f"Final predictions saved to {output_path}")
        except Exception as e:
             logger.error(f"Failed to save predictions to CSV: {e}")
    else:
        logger.warning("Prediction pipeline finished but produced no results.")
    logger.info("--- Prediction Script Finished ---")