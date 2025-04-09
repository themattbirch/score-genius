# backend/nba_score_prediction/prediction.py

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
import json
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from typing import List, Dict, Optional, Any, Tuple

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --- Project Module Imports ---
try:
    from nba_score_prediction.feature_engineering import NBAFeatureEngine
    from nba_score_prediction.models import XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor
    from nba_score_prediction.simulation import PredictionUncertaintyEstimator
    from caching.supabase_client import supabase as supabase_client_instance
    from backend import config
    from . import utils # Relative import for utils module
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Error importing project modules: {e}. Prediction script may fail.", exc_info=True)
    PROJECT_MODULES_IMPORTED = False
    # Define dummy classes if imports fail
    class NBAFeatureEngine:
        def __init__(self, *args, **kwargs): self.defaults={'avg_pts_for': 115.0}
        def generate_all_features(self, df, **kwargs):
            logging.error("Dummy NBAFeatureEngine used - generate_all_features returning empty df.")
            return pd.DataFrame() # Return empty to signal failure
    class _DummyBasePredictor:
        feature_names_in_ = []; training_timestamp = None
        def load_model(self, *args, **kwargs): raise NotImplementedError("Dummy model")
        def predict(self, *args, **kwargs): raise NotImplementedError("Dummy model")
    class XGBoostScorePredictor(_DummyBasePredictor): pass
    class RandomForestScorePredictor(_DummyBasePredictor): pass
    class RidgeScorePredictor(_DummyBasePredictor): pass
    class PredictionUncertaintyEstimator:
        def __init__(self, *args, **kwargs): pass
        def dynamically_adjust_interval(self, *args, **kwargs):
            logging.error("Dummy Uncertainty Estimator used!")
            return np.nan, np.nan, np.nan
    class utils: # Keep dummy utils as fallback
        @staticmethod
        def slice_dataframe_by_features(df, fl, fill_value=0.0):
            logging.error("Dummy utils.slice_dataframe_by_features used!")
            if df is None or fl is None: return pd.DataFrame()
            cols_to_select = [col for col in fl if col in df.columns]
            missing_cols = [col for col in fl if col not in df.columns]
            res_df = df[cols_to_select].copy()
            for col in missing_cols: res_df[col] = fill_value
            try: return res_df[fl].fillna(fill_value)
            except KeyError: logging.error(f"Dummy utils: KeyError selecting final columns {fl}"); return res_df.fillna(fill_value)
    config = None
    supabase_client_instance = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths and Constants ---
MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', PROJECT_ROOT / 'models' / 'saved'))
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = ['home_score', 'away_score']
PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180
DEFAULT_UPCOMING_DAYS_WINDOW = 2

# --- Ensemble Configuration --- # <<< RENAMED SECTION
# META_MODEL_FILENAME = "stacking_meta_model_xgb_hs.joblib" # <<< REMOVE or COMMENT OUT
ENSEMBLE_WEIGHTS_FILENAME = "ensemble_weights.json" # <<< ADD Filename for weights

# Fallback weights if the JSON file cannot be loaded
FALLBACK_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "xgboost": 0.30,
    "random_forest": 0.40,
    "ridge": 0.30
}
# --- Data Column Requirements (Restored from Old Script) ---
REQUIRED_HISTORICAL_COLS = [
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
    'away_assists', 'away_steals', 'away_blocks', 'away_fouls'
]
REQUIRED_TEAM_STATS_COLS = [
    'team_name', 'season', 'wins_all_percentage', 'points_for_avg_all',
    'points_against_avg_all', 'current_form'
]
UPCOMING_GAMES_COLS = ["game_id", "scheduled_time", "home_team", "away_team"]

# --- Database and Supabase Initialization ---
def get_supabase_client() -> Optional[Any]:
    """Returns the Supabase client instance if available."""
    if supabase_client_instance:
        logger.debug("Using imported Supabase client instance.")
        return supabase_client_instance
    else:
        logger.warning("Supabase client instance not available from import.")
        return None

# --- Data Loading Functions (Restored Implementations if changed, checked for consistency) ---
def load_recent_historical_data(supabase_client: Any, days_lookback: int) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical game data from {start_date} onwards...")
    # Ensure REQUIRED_HISTORICAL_COLS is correct before this join
    select_cols_str = ", ".join(map(str, REQUIRED_HISTORICAL_COLS)) # Use map(str,...) for extra safety if list contains non-strings
    all_historical_data = []
    page_size = 1000
    start_index = 0
    has_more = True
    try:
        while has_more:
            logger.debug(f"Fetching historical data batch starting at index {start_index}...")
            response = (
                supabase_client
                .table("nba_historical_game_stats")
                .select(select_cols_str)
                .gte("game_date", start_date)
                .order('game_date', desc=False) # Added desc=False for clarity
                .range(start_index, start_index + page_size - 1)
                .execute()
            )
            batch = response.data
            b_size = len(batch)
            all_historical_data.extend(batch)
            logger.debug(f"Retrieved {b_size} historical records in this batch.")
            if b_size < page_size:
                has_more = False
            else:
                start_index += page_size
        if not all_historical_data:
            logger.warning(f"No historical game data found since {start_date}.")
            return pd.DataFrame()
        historical_df = pd.DataFrame(all_historical_data)
        logger.info(f"Loaded {len(historical_df)} total historical games from Supabase.")
        if 'game_date' not in historical_df.columns:
            logger.error("Critical column 'game_date' missing.")
            return pd.DataFrame()
        historical_df['game_date'] = (
            pd.to_datetime(historical_df['game_date'], errors='coerce')
              .dt.tz_localize(None) # Ensure timezone is stripped for consistency
        )
        historical_df = historical_df.dropna(subset=['game_date'])
        numeric_cols = [
            col for col in REQUIRED_HISTORICAL_COLS
            if col not in ['game_id', 'game_date', 'home_team', 'away_team']
        ]
        for col in numeric_cols:
            if col in historical_df.columns:
                historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Required historical column '{col}' missing. Filling with 0.")
                historical_df[col] = 0
        for col in ['game_id', 'home_team', 'away_team']:
            if col in historical_df.columns:
                historical_df[col] = historical_df[col].astype(str)
            else:
                logger.warning(f"Required historical column '{col}' missing. Filling with empty string.")
                historical_df[col] = "" # Fill with empty string
        return historical_df.sort_values(by='game_date').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading historical data from Supabase: {e}", exc_info=True)
        return pd.DataFrame()

def load_team_stats_data(supabase_client: Any) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    logger.info("Loading team stats data from Supabase table 'nba_historical_team_stats'...")
    # Ensure REQUIRED_TEAM_STATS_COLS is correct before this join
    select_cols_str = ", ".join(map(str, REQUIRED_TEAM_STATS_COLS))
    try:
        response = supabase_client.table("nba_historical_team_stats").select(select_cols_str).execute()
        data = response.data
        if not data:
            logger.warning("No team stats data found.")
            return pd.DataFrame()
        team_stats_df = pd.DataFrame(data)
        logger.info(f"Loaded {len(team_stats_df)} team stat records.")
        numeric_cols = [
            col for col in REQUIRED_TEAM_STATS_COLS
            if col not in ['team_name', 'season', 'current_form']
        ]
        for col in numeric_cols:
            if col in team_stats_df.columns:
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Required team stat column '{col}' missing. Filling with 0.0.")
                team_stats_df[col] = 0.0
        for col in ['team_name', 'season', 'current_form']:
            if col in team_stats_df.columns:
                team_stats_df[col] = team_stats_df[col].astype(str).fillna('')
            else:
                logger.warning(f"Required team stat column '{col}' missing. Filling with empty string.")
                team_stats_df[col] = '' # Fill with empty string
        return team_stats_df
    except Exception as e:
        logger.error(f"Error loading team stats from Supabase: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client: Any, days_window: int) -> pd.DataFrame:
    """
    Fetches upcoming game schedules from Supabase for a given days window.
    """
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    now_pt = datetime.now(PACIFIC_TZ)
    start_utc = now_pt.astimezone(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_utc = (start_utc + timedelta(days=days_window)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc_str = start_utc.strftime('%Y-%m-%dT%H:%M:%S%z')
    end_utc_str = end_utc.strftime('%Y-%m-%dT%H:%M:%S%z')
    logger.info(f"Fetching upcoming games between {start_utc_str} and {end_utc_str} (UTC)...")
    select_cols_str = ", ".join(UPCOMING_GAMES_COLS)
    try:
        response = supabase_client.table("nba_game_schedule") \
            .select(select_cols_str) \
            .gte("scheduled_time", start_utc_str) \
            .lt("scheduled_time", end_utc_str) \
            .order("scheduled_time", desc=False) \
            .execute()
        data = response.data
        if not data:
            logger.warning("No upcoming games found.")
            return pd.DataFrame()
        games_df = pd.DataFrame(data)
        logger.info(f"Fetched {len(games_df)} upcoming games.")
        if "scheduled_time" in games_df.columns:
            games_df["scheduled_time_utc"] = pd.to_datetime(games_df["scheduled_time"], errors='coerce', utc=True)
            games_df = games_df.dropna(subset=['scheduled_time_utc'])
            games_df["game_time_pt"] = games_df["scheduled_time_utc"].dt.tz_convert(PACIFIC_TZ)
            games_df["game_date"] = pd.to_datetime(games_df["game_time_pt"].dt.date)
        else:
            logger.error("Column 'scheduled_time' missing from schedule data.")
            return pd.DataFrame()
        for col in ["home_team", "away_team", "game_id"]:
            games_df[col] = games_df[col].astype(str)
        for col in set(REQUIRED_HISTORICAL_COLS) - {"game_id", "game_date", "home_team", "away_team"}:
            if col not in games_df.columns:
                games_df[col] = np.nan
        return games_df.sort_values(by='game_time_pt').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}", exc_info=True)
        return pd.DataFrame()

# --- Betting Odds Parsing Functions (Restored Implementations) ---
def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    if not line_str:
        return None, None
    # Regex improved slightly for flexibility
    match = re.search(r"([\+\-]?\d+(?:\.\d+)?)\s*(?:\(?\s*(?:[ou])?\s*([\+\-]\d+)\s*\)?)?", str(line_str).strip())
    if match:
        try:
            line = float(match.group(1))
        except (ValueError, TypeError):
            line = None
        try:
            # Default odds if not explicitly found
            odds = int(match.group(2)) if match.group(2) else -110
        except (ValueError, TypeError):
             odds = -110
        return line, odds
    return None, None

def parse_moneyline_str(ml_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[int], Optional[int]]:
    home_ml, away_ml = None, None
    if not ml_str or not isinstance(ml_str, str):
        return home_ml, away_ml
    try:
        # Find all moneyline values first
        mls = re.findall(r"([\+\-]\d{3,})", ml_str)
        if len(mls) == 2:
            # Assume order might be inconsistent, try to identify favorite/underdog
            ml1, ml2 = int(mls[0]), int(mls[1])
            if ml1 < 0 and ml2 > 0: # Standard fav/dog
                 home_ml = ml1; away_ml = ml2
            elif ml2 < 0 and ml1 > 0: # Reversed fav/dog
                 home_ml = ml2; away_ml = ml1
            elif ml1 < 0 and ml2 < 0: # Both negative (rare, maybe error) - assign arbitrarily
                 home_ml = ml1; away_ml = ml2
                 logger.warning(f"Both moneylines negative in '{ml_str}'. Assigning {ml1} to home.")
            elif ml1 > 0 and ml2 > 0: # Both positive (rare, maybe error) - assign arbitrarily
                 home_ml = ml1; away_ml = ml2
                 logger.warning(f"Both moneylines positive in '{ml_str}'. Assigning {ml1} to home.")
            else: # e.g., one is zero? Unlikely but handle
                 home_ml = ml1; away_ml = ml2
            # TODO: Could add team name matching here as a secondary check if needed
        elif len(mls) == 1:
             # Cannot reliably determine which team without more context or format assumption
             logger.warning(f"Only one moneyline found in '{ml_str}'. Cannot assign reliably.")
        else:
            logger.warning(f"Could not find two moneylines in '{ml_str}'.")

    except Exception as e:
        logger.warning(f"Could not parse moneyline string '{ml_str}': {e}")
    return home_ml, away_ml

def parse_spread_str(spread_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    home_line, away_line = None, None
    if not spread_str or not isinstance(spread_str, str):
        return home_line, away_line
    try:
        # Find all potential spread numbers
        numbers = re.findall(r"([\+\-]?\d+(?:\.\d+)?)", spread_str)
        if len(numbers) == 2:
            val1, val2 = float(numbers[0]), float(numbers[1])
            # Check if they sum close to zero (standard spread format)
            if abs(val1 + val2) < 0.1:
                 # Assume the negative value usually belongs to the favorite (home for now)
                 home_line = val1 if val1 < 0 else val2
                 away_line = -home_line
            else: # Doesn't sum to zero - maybe just one line listed? Assume it's home
                home_line = val1
                away_line = -val1
                logger.warning(f"Spread values in '{spread_str}' don't sum to zero. Assuming {val1} is home line.")
        elif len(numbers) == 1:
            # Only one number found, assume it's the home line
            home_line = float(numbers[0])
            away_line = -home_line
        else:
            logger.debug(f"Could not parse spread from '{spread_str}'.")
            return None, None
    except Exception as e:
        logger.warning(f"Could not parse spread string '{spread_str}': {e}")
    return home_line, away_line

def parse_total_str(total_str: Optional[str]) -> Optional[float]:
    total_line = None
    if not total_str or not isinstance(total_str, str):
        return total_line
    try:
        # Find the first number that looks like a typical total (e.g., > 150)
        numbers = re.findall(r"(\d{3}(?:\.\d+)?)", total_str) # Look for 3+ digits
        if numbers:
            total_line = float(numbers[0])
        else: # Fallback to any number if 3+ digits not found
             numbers = re.findall(r"(\d+(?:\.\d+)?)", total_str)
             if numbers:
                  total_line = float(numbers[0])
    except Exception as e:
        logger.warning(f"Could not parse total string '{total_str}': {e}")
    return total_line

def fetch_and_parse_betting_odds(supabase_client: Any, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not supabase_client or not game_ids:
        logger.warning("Skipping odds fetch due to missing client or game IDs.")
        return {}
    odds_dict = {}
    chunk_size = 50 # Supabase might have limits on 'in' filter size
    game_id_chunks = [game_ids[i:i + chunk_size] for i in range(0, len(game_ids), chunk_size)]

    logger.info(f"Fetching betting odds for {len(game_ids)} games in {len(game_id_chunks)} chunk(s)...")
    select_cols = "game_id, home_team, away_team, moneyline_clean, spread_clean, total_clean" # Adjust column names if needed

    for i, chunk in enumerate(game_id_chunks):
        logger.debug(f"Fetching odds chunk {i+1}/{len(game_id_chunks)}...")
        try:
            response = supabase_client.table("nba_game_schedule").select(select_cols).in_("game_id", chunk).execute()
            if not response.data:
                logger.debug(f"No odds data found for chunk {i+1}.")
                continue

            df = pd.DataFrame(response.data)
            for _, row in df.iterrows():
                game_id = str(row.get('game_id'))
                home_team = row.get('home_team', '')
                away_team = row.get('away_team', '')
                if not game_id: continue # Skip if no game_id

                # Use the parsing functions defined above
                home_ml, away_ml = parse_moneyline_str(row.get('moneyline_clean'), home_team, away_team)
                home_spread, _ = parse_spread_str(row.get('spread_clean'), home_team, away_team) # Use the parsed home spread
                total_line = parse_total_str(row.get('total_clean'))

                default_odds = -110 # Standard juice

                odds_entry = {
                    'moneyline': {'home': home_ml, 'away': away_ml},
                    'spread': {'home_line': home_spread, 'away_line': -home_spread if home_spread is not None else None, 'home_odds': default_odds, 'away_odds': default_odds},
                    'total': {'line': total_line, 'over_odds': default_odds, 'under_odds': default_odds},
                    'bookmaker': 'Parsed from Supabase',
                    'last_update': pd.Timestamp.now(tz='UTC') # Record fetch time
                }
                # Only add if at least one line was successfully parsed
                if odds_entry['moneyline']['home'] or odds_entry['moneyline']['away'] or odds_entry['spread']['home_line'] or odds_entry['total']['line']:
                    odds_dict[game_id] = odds_entry
                else:
                     logger.debug(f"No usable odds lines parsed for game {game_id}.")


            logger.debug(f"Processed odds for {len(df)} games in this chunk.")
        except Exception as e:
            logger.error(f"Error fetching/parsing betting odds chunk: {e}", exc_info=True)

    logger.info(f"Finished fetching odds. Parsed data for {len(odds_dict)} games.")
    return odds_dict

# --- Model Loading Function ---
def load_trained_models(model_dir: Path = MODELS_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    models = {}
    model_configs = {"xgboost": XGBoostScorePredictor, "random_forest": RandomForestScorePredictor, "ridge": RidgeScorePredictor}
    loaded_feature_names = None
    all_models_loaded = True
    inconsistent_features_found = False
    for name, PredictorClass in model_configs.items():
        if not PROJECT_MODULES_IMPORTED or getattr(PredictorClass, '__module__', '').startswith('_Dummy'):
            logger.error(f"Cannot load '{name}': Module not imported.")
            all_models_loaded = False
            continue
        try:
            logger.info(f"Loading latest '{name}' model from {model_dir}")
            predictor = PredictorClass(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            predictor.load_model()
            if predictor.pipeline_home is None or predictor.pipeline_away is None:
                raise ValueError(f"Pipelines not loaded correctly for {name}")
            models[name] = predictor
            logger.info(f"Loaded {name} model trained on {predictor.training_timestamp or 'unknown date'}.")
            current_features = getattr(predictor, 'feature_names_in_', None)
            if current_features is None:
                logger.error(f"Model '{name}' missing feature list ('feature_names_in_').")
                inconsistent_features_found = True
                all_models_loaded = False
                continue
            if loaded_feature_names is None:
                loaded_feature_names = current_features
                logger.info(f"Using feature list from '{name}' ({len(loaded_feature_names)} features).")
            elif set(current_features) != set(loaded_feature_names):
                logger.error(f"Feature mismatch: '{name}' features differ from previously loaded model.")
                inconsistent_features_found = True
                all_models_loaded = False
            elif len(current_features) != len(loaded_feature_names):
                logger.warning(f"Feature length mismatch for '{name}' though sets match. Check for duplicates.")
        except FileNotFoundError:
            logger.error(f"No model file found for '{name}' in {model_dir}.")
            all_models_loaded = False
        except Exception as e:
            logger.error(f"Error loading model '{name}': {e}", exc_info=True)
            all_models_loaded = False

    if inconsistent_features_found:
        logger.error("Aborting due to inconsistent features across models.")
        return None, None
    if not all_models_loaded or not models:
        logger.error("Failed to load one or more required models.")
        return None, None
    if not loaded_feature_names:
        logger.error("No feature list found from loaded models.")
        return models, None
    logger.info(f"All models loaded with consistent feature list ({len(loaded_feature_names)} features).")
    return models, loaded_feature_names

# --- Calibration Function (Restored Implementation) ---
def calibrate_prediction_with_odds(
    prediction: Dict, # Expects the raw prediction dict
    odds_info: Optional[Dict],
    blend_factor: float = 0.3
) -> Dict:
    """ Calibrates raw ensemble predictions with market odds if available. """
    calibrated = prediction.copy() # Start with raw prediction values

    # Store raw values before potential overwrite
    calibrated['raw_predicted_home_score'] = prediction.get('predicted_home_score')
    calibrated['raw_predicted_away_score'] = prediction.get('predicted_away_score')
    calibrated['raw_predicted_point_diff'] = prediction.get('predicted_point_diff')
    calibrated['raw_predicted_total_score'] = prediction.get('predicted_total_score')
    calibrated['raw_win_probability'] = prediction.get('win_probability')
    # Keep uncertainty estimates based on raw prediction for now
    calibrated['raw_lower_bound'] = prediction.get('lower_bound')
    calibrated['raw_upper_bound'] = prediction.get('upper_bound')
    calibrated['raw_confidence_pct'] = prediction.get('confidence_pct')

    calibrated['betting_odds'] = odds_info # Store odds used for calibration
    calibrated['calibration_blend_factor'] = blend_factor
    calibrated['is_calibrated'] = False # Default

    game_id = prediction.get('game_id', 'N/A')

    if not odds_info:
        logger.debug(f"No odds info for game {game_id}. Calibration skipped.")
        return calibrated # Return with raw values and is_calibrated=False

    try:
        market_spread, market_total = None, None

        # Extract market spread (negative means home team is favored)
        spread_data = odds_info.get('spread', {})
        if spread_data.get('home_line') is not None:
            market_spread = spread_data['home_line'] # Use home line directly
        elif spread_data.get('away_line') is not None:
             # Infer home line if only away line exists
            market_spread = -spread_data['away_line']
        else:
             logger.debug(f"No spread line found in odds for game {game_id}.")

        # Extract market total
        total_data = odds_info.get('total', {})
        market_total = total_data.get('line')
        if market_total is None:
             logger.debug(f"No total line found in odds for game {game_id}.")

        # Get raw predictions needed for calibration
        pred_diff_raw = calibrated.get('raw_predicted_point_diff')
        pred_total_raw = calibrated.get('raw_predicted_total_score')

        # Check if we have the necessary components
        can_calibrate = True
        if market_spread is None: logger.debug(f"Cannot calibrate spread for game {game_id}: Missing market spread."); can_calibrate = False
        if market_total is None: logger.debug(f"Cannot calibrate total for game {game_id}: Missing market total."); can_calibrate = False
        if pred_diff_raw is None: logger.debug(f"Cannot calibrate spread for game {game_id}: Missing raw point diff."); can_calibrate = False
        if pred_total_raw is None: logger.debug(f"Cannot calibrate total for game {game_id}: Missing raw total score."); can_calibrate = False

        if not can_calibrate:
            logger.warning(f"Calibration skipped for game {game_id} due to missing values.")
            return calibrated # Return with raw values and is_calibrated=False

        logger.info(f"Calibrating game {game_id} with blend factor {blend_factor} (Market Spread: {market_spread:+.1f}, Total: {market_total:.1f})")

        # Blend total and spread separately
        # Note: Market Spread convention needs care. If market_spread = -7.5 (home favored by 7.5),
        # the predicted outcome equivalent is AwayScore - HomeScore = 7.5, or HomeScore - AwayScore = -7.5.
        # Our pred_diff_raw = home - away. So they align.
        cal_total = blend_factor * market_total + (1 - blend_factor) * pred_total_raw
        cal_diff = blend_factor * market_spread + (1 - blend_factor) * pred_diff_raw # Home - Away

        # Recalculate scores based on calibrated total and difference
        cal_home = (cal_total + cal_diff) / 2.0
        cal_away = (cal_total - cal_diff) / 2.0 # = cal_total - cal_home

        # Recalculate win probability based on calibrated difference
        cal_win_prob = 1 / (1 + math.exp(-0.15 * cal_diff)) # Using the same simple logistic function

        # Update the prediction dictionary with calibrated values
        calibrated['predicted_home_score'] = round(cal_home, 1)
        calibrated['predicted_away_score'] = round(cal_away, 1)
        calibrated['predicted_point_diff'] = round(cal_diff, 1)
        calibrated['predicted_total_score'] = round(cal_total, 1)
        calibrated['win_probability'] = round(cal_win_prob, 3)
        # Keep uncertainty based on raw prediction for now
        calibrated['lower_bound'] = calibrated['raw_lower_bound']
        calibrated['upper_bound'] = calibrated['raw_upper_bound']
        calibrated['confidence_pct'] = calibrated['raw_confidence_pct']
        calibrated['is_calibrated'] = True # Mark as calibrated

        logger.info(f"Calibrated game {game_id}: H {cal_home:.1f}, A {cal_away:.1f}, Diff {cal_diff:+.1f}, Total {cal_total:.1f}, WP {cal_win_prob:.1%}")

    except Exception as e:
        logger.error(f"Error during calibration for game {game_id}: {e}", exc_info=True)
        # Return original prediction dict but mark calibration failed
        calibrated['is_calibrated'] = False # Ensure it's marked as not calibrated on error

    return calibrated


# --- Reporting Utility (Restored Implementation) ---
def display_prediction_summary(predictions: List[Dict]) -> None:
    """ Displays a formatted summary of final predictions. """
    header_line = "=" * 145
    header_title = " " * 55 + "NBA PREGAME PREDICTION SUMMARY"
    header_cols = (f"{'DATE':<11} {'MATCHUP':<30} {'FINAL SCORE':<12} {'FINAL DIFF':<10} "
                   f"{'L/U BOUND':<12} {'CONF %':<7} {'RAW SCORE':<12} {'RAW DIFF':<10} {'WIN PROB':<10} {'CALIB?':<6}")
    header_sep = "-" * 145

    print(f"\n{header_line}\n{header_title}\n{header_line}")
    print(header_cols)
    print(header_sep)
    try:
        # Create DataFrame for sorting
        predictions_df = pd.DataFrame(predictions)
        if 'game_date' not in predictions_df.columns:
             logger.error("Missing 'game_date' column for display sorting.")
        else:
             predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'], errors='coerce')
             predictions_df = predictions_df.dropna(subset=['game_date'])
             # Sort by date, then by absolute predicted difference (closer games first)
             predictions_df = predictions_df.sort_values(
                 ['game_date', 'predicted_point_diff'],
                 key=lambda x: abs(x) if x.name == 'predicted_point_diff' else x,
                 na_position='last' # Put games with NaN diff last
            )
    except Exception as e:
        logger.error(f"Error sorting predictions for display: {e}")
        predictions_df = pd.DataFrame(predictions) # Display unsorted if error

    # Iterate and print rows safely
    for _, game in predictions_df.iterrows():
        try:
            date_str = game['game_date'].strftime('%Y-%m-%d') if pd.notna(game.get('game_date')) else "N/A       "
            matchup = f"{game.get('home_team','?')[:14]} vs {game.get('away_team','?')[:14]}"
            is_calib = game.get('is_calibrated', False)
            calib_str = "Yes" if is_calib else "No "

            # Use .get with default=np.nan for numeric fields for safe formatting
            final_home = game.get('predicted_home_score', np.nan)
            final_away = game.get('predicted_away_score', np.nan)
            final_diff = game.get('predicted_point_diff', np.nan)
            final_win_prob = game.get('win_probability', np.nan)
            lower_b = game.get('lower_bound', np.nan)
            upper_b = game.get('upper_bound', np.nan)
            conf_pct = game.get('confidence_pct', np.nan)
            raw_home = game.get('raw_predicted_home_score', np.nan)
            raw_away = game.get('raw_predicted_away_score', np.nan)
            raw_diff = game.get('raw_predicted_point_diff', np.nan)

            # Format safely
            final_score_str = f"{final_home:.1f}-{final_away:.1f}" if pd.notna(final_home) and pd.notna(final_away) else "N/A"
            final_diff_str = f"{final_diff:<+10.1f}" if pd.notna(final_diff) else "   N/A    "
            bound_str = f"{lower_b:.1f}/{upper_b:.1f}" if pd.notna(lower_b) and pd.notna(upper_b) else "  N/A/N/A   "
            conf_str = f"{conf_pct:.1f}" if pd.notna(conf_pct) else "N/A"
            raw_score_str = f"{raw_home:.1f}-{raw_away:.1f}" if pd.notna(raw_home) and pd.notna(raw_away) else "N/A"
            raw_diff_str = f"{raw_diff:<+10.1f}" if pd.notna(raw_diff) else "   N/A    "
            win_prob_str = f"{final_win_prob*100:.1f}%" if pd.notna(final_win_prob) else "   N/A   "

            print(f"{date_str:<11} {matchup:<30} {final_score_str:<12} {final_diff_str:<10} "
                  f"{bound_str:<12} {conf_str:<7} {raw_score_str:<12} {raw_diff_str:<10} {win_prob_str:<10} {calib_str:<6}")
        except Exception as display_e:
            logger.error(f"Error displaying summary row for game {game.get('game_id', 'N/A')}: {display_e}")

    print("=" * 145)


# --- Main Prediction Orchestration Function (Using Stacking/Fallback) ---
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW,
    model_dir: Path = MODELS_DIR,
    calibrate_with_odds: bool = True,
    blend_factor: float = 0.3,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES
) -> Tuple[List[Dict], List[Dict]]:
    """
    Orchestrates the prediction process using on-the-fly feature generation
    and a weighted ensemble based on validation performance.
    Returns a tuple: (final prediction list, raw prediction list).
    """
    logger.info("--- Starting NBA Prediction Pipeline (Weighted Ensemble) ---") # <<< UPDATED Log
    start_time = time.time()
    final_predictions: List[Dict] = []
    raw_predictions_list: List[Dict] = []

    # --- Initial Setup (Client, Engines) ---
    if not PROJECT_MODULES_IMPORTED:
        logger.critical("Core project modules not imported. Cannot proceed.")
        return [], []
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error("Supabase Client initialization failed. Cannot proceed.")
        return [], []
    feature_engine = None
    uncertainty_estimator = None
    hist_stats_df = None
    try:
        feature_engine = NBAFeatureEngine(supabase_client=supabase_client, debug=False)
        logger.info(f"Initialized FeatureEngine: {type(feature_engine)}")
        # Load coverage stats (keep existing logic)
        coverage_stats_path = REPORTS_DIR / "historical_coverage_stats.csv"
        if coverage_stats_path.is_file():
            try:
                hist_stats_df = pd.read_csv(coverage_stats_path)
                if 'quarter' in hist_stats_df.columns and 'actual_coverage' in hist_stats_df.columns:
                    logger.info(f"Loaded historical coverage stats from {coverage_stats_path}")
                else:
                    logger.warning(f"Coverage stats file {coverage_stats_path} missing required columns. Ignoring.")
                    hist_stats_df = None
            except Exception as csv_e:
                 logger.warning(f"Could not read coverage stats file {coverage_stats_path}: {csv_e}")
                 hist_stats_df = None
        else:
            logger.info(f"Coverage stats file not found at {coverage_stats_path}. Using default uncertainty settings.")
            hist_stats_df = None
        uncertainty_estimator = PredictionUncertaintyEstimator(
            historical_coverage_stats=hist_stats_df,
            debug=False
        )
        logger.info("Successfully initialized FeatureEngine and PredictionUncertaintyEstimator.")
    except Exception as init_e:
        logger.error(f"FATAL: Failed to initialize core engines: {init_e}", exc_info=True)
        return [], []

    # --- Load Data (Step 1) ---
    logger.info("Step 1: Loading data...")
    historical_data = load_recent_historical_data(supabase_client, days_lookback=historical_lookback)
    team_stats_data = load_team_stats_data(supabase_client)
    upcoming_games_df = fetch_upcoming_games_data(supabase_client, days_window=days_window)

    if upcoming_games_df.empty:
        logger.warning("No upcoming games found or error fetching them. Exiting.")
        return [], []
    if historical_data.empty: logger.warning("Historical data empty; feature quality reduced.")
    if team_stats_data.empty: logger.warning("Team stats data empty; context features reduced.")

        # --- Load Models & Weights (Step 2) --- # <<< UPDATED Step Name
    logger.info("Step 2: Loading trained base models & ensemble weights...")
    prediction_models: Optional[Dict[str, Any]] = None
    required_features: Optional[List[str]] = None
    ensemble_weights: Dict[str, float] = {} # <<< Initialize weights dict

    try:
        # Load base models using the existing function
        prediction_models, required_features = load_trained_models(model_dir)
        if not prediction_models or not required_features:
            raise RuntimeError("Failed to load base models or required feature list.")
        logger.info(f"Loaded base models: {list(prediction_models.keys())} (requires {len(required_features)} features).")

        # --- Load Ensemble Weights from JSON --- # <<< NEW Block
        weights_path = model_dir / ENSEMBLE_WEIGHTS_FILENAME
        if not weights_path.is_file():
             logger.warning(f"Ensemble weights file '{ENSEMBLE_WEIGHTS_FILENAME}' not found at {weights_path}.")
             logger.warning(f"Using FALLBACK static weights: {FALLBACK_ENSEMBLE_WEIGHTS}")
             ensemble_weights = FALLBACK_ENSEMBLE_WEIGHTS.copy() # Use fallback
        else:
             logger.info(f"Attempting to load ensemble weights from {weights_path}...")
             try:
                 with open(weights_path, 'r') as f:
                     loaded_weights = json.load(f)
                 # Basic validation
                 if isinstance(loaded_weights, dict) and all(isinstance(k, str) and isinstance(v, (float, int)) for k, v in loaded_weights.items()):
                     # Ensure weights approximately sum to 1 (allow for float precision)
                     if abs(sum(loaded_weights.values()) - 1.0) > 1e-5:
                          logger.warning(f"Loaded weights sum to {sum(loaded_weights.values()):.4f}, not 1.0. Check calculation/file.")
                          # Option: Renormalize here, or use as is, or use fallback? Let's use as is for now.
                     ensemble_weights = {k: float(v) for k, v in loaded_weights.items()} # Ensure float
                     logger.info(f"Ensemble weights loaded successfully: {ensemble_weights}")
                 else:
                      raise ValueError("Invalid format in ensemble weights JSON file.")
             except Exception as weights_load_e:
                  logger.error(f"Error loading or validating ensemble weights from {weights_path}: {weights_load_e}.")
                  logger.warning(f"Using FALLBACK static weights: {FALLBACK_ENSEMBLE_WEIGHTS}")
                  ensemble_weights = FALLBACK_ENSEMBLE_WEIGHTS.copy() # Use fallback on error

        # Check if loaded weights cover the loaded models
        missing_weights = set(prediction_models.keys()) - set(ensemble_weights.keys())
        if missing_weights:
             logger.warning(f"Loaded weights missing for models: {missing_weights}. These models will have zero weight.")
             for model_key in missing_weights:
                  ensemble_weights[model_key] = 0.0 # Assign zero weight explicitly

    except Exception as load_e:
        logger.error(f"Model/Weight loading failed: {load_e}", exc_info=True)
        return [], []

    # --- Feature Generation (Step 3) ---
    logger.info("Step 3: Generating features...")
    features_df_full = pd.DataFrame()
    predict_features_df = pd.DataFrame()
    try:
        logger.info("Calling feature_engine.generate_all_features...")
        features_df_full = feature_engine.generate_all_features(
            df=upcoming_games_df.copy(), # Pass only upcoming games
            historical_games_df=historical_data.copy() if not historical_data.empty else None,
            team_stats_df=team_stats_data.copy() if not team_stats_data.empty else None,
            rolling_windows=[5, 10, 20] # Make configurable if needed
        )
        if features_df_full is None or features_df_full.empty:
            raise RuntimeError("Feature generation returned empty DataFrame.")
        logger.info(f"Feature generation call completed. Output shape: {features_df_full.shape}")

        # Validate base features
        missing_base_features = set(required_features) - set(features_df_full.columns)
        if missing_base_features:
             raise RuntimeError(f"Generated features missing base model columns: {missing_base_features}")

        # Filter to upcoming games (should already be the case if generate_all_features uses df index)
        upcoming_game_ids = upcoming_games_df['game_id'].unique()
        predict_features_df = features_df_full[features_df_full['game_id'].isin(upcoming_game_ids)].copy()
        if predict_features_df.empty:
             raise RuntimeError(f"No feature rows found for upcoming games after generation.")
        logger.info(f"Filtered features to {len(predict_features_df)} upcoming games.")

    except Exception as feat_e:
        logger.error(f"FATAL: Feature generation or validation failed: {feat_e}", exc_info=True)
        return [], []

    # --- Feature Selection for Base Models (Step 4) ---
    logger.info(f"Step 4: Selecting/validating {len(required_features)} features for base models...")
    X_predict = pd.DataFrame()
    try:
        # Use utils helper for consistent slicing and filling
        X_predict = utils.slice_dataframe_by_features(
            df=predict_features_df, # Use the filtered features df
            feature_list=required_features,
            fill_value=0.0 # Or use training mean/median if available and appropriate
        )
        if X_predict is None or X_predict.empty or len(X_predict) != len(predict_features_df):
            raise ValueError("Feature slicing failed or resulted in wrong number of rows.")
        if X_predict.isnull().any().any():
            logger.warning(f"NaNs remaining after slice/fill. Filling with 0.0.")
            X_predict = X_predict.fillna(0.0)
        logger.info(f"Base model feature selection complete. Shape: {X_predict.shape}")

    except Exception as e:
        logger.error(f"Error during feature selection/validation: {e}", exc_info=True)
        return [], []

        # --- Generate Predictions (Step 5: Weighted Ensemble) --- # <<< UPDATED Step Name
    logger.info("Step 5: Generating predictions (Weighted Ensemble)...")
    model_preds_dfs: Dict[str, Optional[pd.DataFrame]] = {} # Store base model predictions
    logger.info("Generating predictions from base models...")
    base_models_to_predict_with = list(prediction_models.keys())
    for name in base_models_to_predict_with:
         try:
              logger.debug(f"Predicting with base model: {name}")
              preds_df = prediction_models[name].predict(X_predict)
              if preds_df is None or preds_df.empty or not all(c in preds_df.columns for c in ['predicted_home_score', 'predicted_away_score']):
                   raise ValueError(f"Invalid/empty prediction output DataFrame from {name}")
              preds_df.index = X_predict.index # Align index
              model_preds_dfs[name] = preds_df[['predicted_home_score', 'predicted_away_score']]
              logger.debug(f"Base model {name} prediction successful. Shape: {preds_df.shape}")
         except Exception as e:
              logger.error(f"Error predicting with base model {name}: {e}", exc_info=False)
              model_preds_dfs[name] = None # Mark as failed

    successful_base_models = [name for name, df in model_preds_dfs.items() if df is not None]
    logger.info(f"Base models with successful predictions: {successful_base_models}")
    if not successful_base_models:
         logger.error("FATAL: No base models generated predictions."); return [], []

    logger.info(f"Assembling final predictions for {len(predict_features_df)} games...")
    processed_game_count = 0
    for idx in predict_features_df.index:
        game_id = 'unknown_idx_' + str(idx)
        try:
            game_info = predict_features_df.loc[idx]
            game_id = str(game_info.get('game_id', 'N/A'))
            home_team = game_info.get('home_team', 'N/A')
            away_team = game_info.get('away_team', 'N/A')
            game_date = pd.to_datetime(game_info.get('game_date'))

            # --- Get Component Predictions for this Game ---
            component_predictions_this_game = {}
            valid_base_preds_for_game = True
            for name in base_models_to_predict_with:
                home_pred_game, away_pred_game = np.nan, np.nan
                preds_df = model_preds_dfs.get(name)
                if preds_df is not None and idx in preds_df.index:
                    home_pred_game = preds_df.loc[idx, 'predicted_home_score']
                    away_pred_game = preds_df.loc[idx, 'predicted_away_score']
                    if pd.isna(home_pred_game) or pd.isna(away_pred_game):
                         logger.warning(f"NaN prediction value from base model {name} game {game_id}.")
                         valid_base_preds_for_game = False
                else:
                     logger.warning(f"Prediction missing/failed for base model {name} game {game_id}.")
                     valid_base_preds_for_game = False
                component_predictions_this_game[name] = {'home': home_pred_game, 'away': away_pred_game}

            # --- REMOVED Meta-Model Prediction Logic ---

            # --- Implement Weighted Average Logic --- # <<< NEW Block
            logger.debug(f"Calculating weighted average for game {game_id} using weights: {ensemble_weights}")
            ensemble_home_score, ensemble_away_score = 0.0, 0.0
            total_weight_applied = 0.0
            ensemble_method_used = "weighted_avg" # Default type

            for name, preds in component_predictions_this_game.items():
                weight = ensemble_weights.get(name, 0.0) # Get weight from loaded dict
                h_pred, a_pred = preds.get('home'), preds.get('away')

                if weight > 1e-6 and pd.notna(h_pred) and pd.notna(a_pred): # Use small threshold for weight check
                    ensemble_home_score += h_pred * weight
                    ensemble_away_score += a_pred * weight
                    total_weight_applied += weight
                elif weight > 1e-6:
                     logger.warning(f"Skipping {name} (weight={weight:.3f}) in weighted avg due to NaN prediction for game {game_id}.")

            if total_weight_applied > 1e-6: # Normalize if weights were applied
                ensemble_home_score /= total_weight_applied
                ensemble_away_score /= total_weight_applied
                # Add detail if fallback weights were potentially used
                if ensemble_weights == FALLBACK_ENSEMBLE_WEIGHTS:
                     ensemble_method_used = "weighted_avg_fallback_static"
                else:
                     ensemble_method_used = "weighted_avg_inv_mae" # Assume loaded from file based on MAE
                logger.debug(f"Weighted avg calculated game {game_id} (Total Weight Applied: {total_weight_applied:.3f})")
            else: # Ultimate fallback: simple average of available preds
                logger.warning(f"No valid weighted predictions for game {game_id} (Total Weight Applied: {total_weight_applied:.3f}). Using simple average.")
                hs = [p['home'] for p in component_predictions_this_game.values() if pd.notna(p.get('home'))]
                aws = [p['away'] for p in component_predictions_this_game.values() if pd.notna(p.get('away'))]
                avg_score = feature_engine.defaults.get('avg_pts_for', 115.0)
                ensemble_home_score = np.mean(hs) if hs else avg_score
                ensemble_away_score = np.mean(aws) if aws else avg_score
                ensemble_method_used = "simple_average_fallback"
                if pd.isna(ensemble_home_score) or pd.isna(ensemble_away_score): # Final safety net
                     logger.error(f"Simple average fallback also resulted in NaN for game {game_id}. Assigning default score.")
                     ensemble_home_score = avg_score
                     ensemble_away_score = avg_score
                     ensemble_method_used = "default_score_fallback"
            # --- End Weighted Average Logic ---

            # --- Calculate derived values + uncertainty (Keep this logic) ---
            point_diff = ensemble_home_score - ensemble_away_score if pd.notna(ensemble_home_score) and pd.notna(ensemble_away_score) else np.nan
            total_score = ensemble_home_score + ensemble_away_score if pd.notna(ensemble_home_score) and pd.notna(ensemble_away_score) else np.nan
            win_prob = 1 / (1 + math.exp(-0.15 * point_diff)) if pd.notna(point_diff) else np.nan
            lower_b, upper_b, conf_pct = np.nan, np.nan, np.nan
            if pd.notna(total_score):
                try:
                    hist_acc_dict = None # Reset/Ensure not using stale data
                    if uncertainty_estimator is not None:
                         # Get historical stats if available from the estimator
                         if hasattr(uncertainty_estimator, 'get_coverage_stats'):
                              stats_df = uncertainty_estimator.get_coverage_stats()
                              if stats_df is not None and not stats_df.empty:
                                   hist_acc_dict = stats_df.set_index('quarter').to_dict('index')
                         # Calculate bounds
                         lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
                             prediction=total_score, current_quarter=0, historic_accuracy=hist_acc_dict
                         )
                    else: logger.warning("Uncertainty estimator not available.")
                except Exception as unc_e: logger.error(f"Error calculating uncertainty game {game_id}: {unc_e}")
            else: logger.warning(f"Cannot calc uncertainty game {game_id}: total score NaN.")


            # Assemble raw prediction dictionary
            raw_pred_dict = {
                'game_id': game_id, 'game_date': game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else None,
                'home_team': home_team, 'away_team': away_team,
                'predicted_home_score': round(ensemble_home_score, 1) if pd.notna(ensemble_home_score) else None,
                'predicted_away_score': round(ensemble_away_score, 1) if pd.notna(ensemble_away_score) else None,
                'predicted_point_diff': round(point_diff, 1) if pd.notna(point_diff) else None,
                'predicted_total_score': round(total_score, 1) if pd.notna(total_score) else None,
                'win_probability': round(win_prob, 3) if pd.notna(win_prob) else None,
                'lower_bound': round(lower_b, 1) if pd.notna(lower_b) else None,
                'upper_bound': round(upper_b, 1) if pd.notna(upper_b) else None,
                'confidence_pct': round(conf_pct, 1) if pd.notna(conf_pct) else None,
                'component_predictions': { name: {'home': round(preds.get('home', np.nan), 1), 'away': round(preds.get('away', np.nan), 1)} for name, preds in component_predictions_this_game.items() },
                'ensemble_method': ensemble_method_used, # <<< UPDATED Method Name
                'is_calibrated': False
            }
            raw_predictions_list.append(raw_pred_dict)
            processed_game_count += 1

        except Exception as game_proc_e:
            logger.error(f"FATAL: Failed processing game index {idx} (GameID: {game_id}): {game_proc_e}", exc_info=True)
            continue

    # --- Post-prediction Processing ---
    if processed_game_count == 0:
         logger.error("No games successfully processed for predictions."); return [], []
    logger.info(f"Successfully assembled raw predictions for {processed_game_count} games.")


    # --- Fetch Odds & Calibrate (Step 6) ---
    logger.info("Step 6: Fetching odds & calibrating...")
    final_predictions = []; odds_dict = {}
    if calibrate_with_odds:
        gids = [str(p['game_id']) for p in raw_predictions_list if pd.notna(p.get('game_id'))]
        if gids: odds_dict = fetch_and_parse_betting_odds(supabase_client, gids)
        if not odds_dict: logger.warning("No odds fetched. Using raw predictions.")
        else: logger.info(f"Fetched odds for {len(odds_dict)} games. Attempting calibration...")
        for pred in raw_predictions_list: final_predictions.append(calibrate_prediction_with_odds(pred, odds_dict.get(str(pred.get('game_id'))), blend_factor))
    else:
        logger.info("Skipping calibration.")
        # Ensure structure matches calibrated output even if not calibrating
        for pred in raw_predictions_list: final_predictions.append(calibrate_prediction_with_odds(pred, None, blend_factor))

    # --- Final Checks, Display Summary (Step 7) & Return ---
    if not final_predictions: logger.error("Pipeline finished: final_predictions list empty."); return [], raw_predictions_list
    logger.info("Step 7: Displaying prediction summary...")
    display_prediction_summary(final_predictions)
    end_time = time.time(); logger.info(f"--- NBA Prediction Pipeline Finished in {end_time - start_time:.2f} seconds ---")
    return final_predictions, raw_predictions_list

# --- Script Execution ---
if __name__ == "__main__":
    logger.info("--- Running Prediction Script Directly ---")
    # Use defaults defined at top of script
    config_days_window = DEFAULT_UPCOMING_DAYS_WINDOW
    config_calibrate = True
    config_blend_factor = 0.3 # Make configurable if needed
    config_hist_lookback = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES

    logger.info(f"Config: Days={config_days_window}, Calibrate={config_calibrate}, Blend={config_blend_factor}, Lookback={config_hist_lookback}")

    final_preds, raw_preds = generate_predictions(
        days_window=config_days_window,
        model_dir=MODELS_DIR,
        calibrate_with_odds=config_calibrate,
        blend_factor=config_blend_factor,
        historical_lookback=config_hist_lookback
    )

    if final_preds:
        logger.info(f"Successfully generated {len(final_preds)} final predictions.")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"predictions_output_{ts}.csv"
            df_to_save = pd.DataFrame(final_preds)
            # Define columns to drop - ensure 'ensemble_method' isn't crucial for CSV output
            cols_to_drop_for_csv = ['component_predictions', 'betting_odds']
            df_to_save = df_to_save.drop(columns=cols_to_drop_for_csv, errors='ignore')
            # Ensure all expected numeric cols exist before conversion
            numeric_cols_out = [ # Define expected numeric columns
                 'predicted_home_score', 'predicted_away_score', 'predicted_point_diff',
                 'predicted_total_score', 'win_probability', 'lower_bound', 'upper_bound',
                 'confidence_pct', 'raw_predicted_home_score', 'raw_predicted_away_score',
                 'raw_predicted_point_diff', 'raw_predicted_total_score', 'raw_win_probability',
                 'raw_lower_bound', 'raw_upper_bound', 'raw_confidence_pct', 'calibration_blend_factor'
            ]
            for nc in numeric_cols_out:
                 if nc in df_to_save.columns:
                     df_to_save[nc] = pd.to_numeric(df_to_save[nc], errors='coerce')
                 else:
                      logger.debug(f"Numeric column '{nc}' not found in final DataFrame for CSV conversion.")

            df_to_save.to_csv(output_path, index=False, float_format="%.3f")
            logger.info(f"Final predictions saved to {output_path}")
        except Exception as e:
             logger.error(f"Failed to save predictions to CSV: {e}", exc_info=True)
    else:
        logger.warning("Prediction pipeline finished but produced no final results.")
    logger.info("--- Prediction Script Finished ---")