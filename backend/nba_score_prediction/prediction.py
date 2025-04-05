"""
Generates pre-game predictions for upcoming NBA games using on-the-fly feature
generation with NBAFeatureEngine and a static ensemble model (XGBoost, RandomForest, Ridge).
Calculates prediction intervals and confidence using PredictionUncertaintyEstimator.
Optionally integrates betting odds and calibrates point predictions.
Includes logging for raw component predictions and displays raw vs. final predictions with uncertainty.
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
from sqlalchemy import create_engine, text  # Added text for literal SQL
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
    from nba_score_prediction.feature_engineering import NBAFeatureEngine  # Import Feature Engine
    from nba_score_prediction.models import XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor  # Specific predictors
    from nba_score_prediction.simulation import PredictionUncertaintyEstimator
    from caching.supabase_client import supabase as supabase_client_instance  # Renamed to avoid conflict
    from backend import config  # Import the config module
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Error importing project modules: {e}. Prediction script may fail.", exc_info=True)
    PROJECT_MODULES_IMPORTED = False
    class NBAFeatureEngine:
        def __init__(self, *args, **kwargs): 
            self.defaults = {'avg_pts_for': 115.0}
        def generate_all_features(self, df, **kwargs):
            logging.error("Dummy NBAFeatureEngine used - generate_all_features returning input df.")
            return df
    class _DummyBasePredictor:
        feature_names_in_ = []
        training_timestamp = None
        def load_model(self, *args, **kwargs): 
            raise NotImplementedError("Dummy model - real import failed")
        def predict(self, *args, **kwargs): 
            raise NotImplementedError("Dummy model - real import failed")
    class XGBoostScorePredictor(_DummyBasePredictor): pass
    class RandomForestScorePredictor(_DummyBasePredictor): pass
    class RidgeScorePredictor(_DummyBasePredictor): pass
    class PredictionUncertaintyEstimator:
        def __init__(self, *args, **kwargs): pass
        def dynamically_adjust_interval(self, prediction, current_quarter, historic_accuracy=None):
            logging.error("Dummy Uncertainty Estimator used!")
            return prediction - 10, prediction + 10, 50.0
    config = None
    supabase_client_instance = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', BACKEND_DIR / 'models' / 'saved'))
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "xgboost": 0.30,
    "random_forest": 0.40,
    "ridge": 0.30
}

TARGET_COLUMNS = ['home_score', 'away_score']

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180  # days for rolling features
DEFAULT_UPCOMING_DAYS_WINDOW = 2           # days ahead to predict

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

# --- Data Loading Functions ---
def load_recent_historical_data(supabase_client: Any, days_lookback: int) -> pd.DataFrame:
    """
    Loads historical game data from Supabase using the REQUIRED_HISTORICAL_COLS.
    """
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical game data from {start_date} onwards...")
    select_cols_str = ", ".join(REQUIRED_HISTORICAL_COLS)
    all_historical_data = []
    page_size = 1000
    start_index = 0
    has_more = True
    try:
        while has_more:
            logger.debug(f"Fetching historical data batch starting at index {start_index}...")
            response = supabase_client.table("nba_historical_game_stats") \
                .select(select_cols_str) \
                .gte("game_date", start_date) \
                .order('game_date') \
                .range(start_index, start_index + page_size - 1) \
                .execute()
            batch = response.data
            b_size = len(batch)
            all_historical_data.extend(batch)
            logger.debug(f"Retrieved {b_size} records in this batch.")
            if b_size < page_size:
                has_more = False
            else:
                start_index += page_size
        if not all_historical_data:
            logger.warning(f"No historical game data found since {start_date}.")
            return pd.DataFrame()
        historical_df = pd.DataFrame(all_historical_data)
        logger.info(f"Loaded {len(historical_df)} historical games.")
        historical_df['game_date'] = pd.to_datetime(historical_df['game_date'], errors='coerce').dt.tz_localize(None)
        historical_df = historical_df.dropna(subset=['game_date'])
        for col in [col for col in REQUIRED_HISTORICAL_COLS if col not in ['game_id', 'game_date', 'home_team', 'away_team']]:
            if col in historical_df.columns:
                historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Column '{col}' missing; filling with 0.")
                historical_df[col] = 0
        for col in ['game_id', 'home_team', 'away_team']:
            historical_df[col] = historical_df[col].astype(str)
        return historical_df.sort_values(by='game_date').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {e}", exc_info=True)
        return pd.DataFrame()

def load_team_stats_data(supabase_client: Any) -> pd.DataFrame:
    """
    Loads team stats data from Supabase using the REQUIRED_TEAM_STATS_COLS.
    """
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    logger.info("Loading team stats data from Supabase...")
    select_cols_str = ", ".join(REQUIRED_TEAM_STATS_COLS)
    try:
        response = supabase_client.table("nba_historical_team_stats").select(select_cols_str).execute()
        data = response.data
        if not data:
            logger.warning("No team stats data found.")
            return pd.DataFrame()
        team_stats_df = pd.DataFrame(data)
        logger.info(f"Loaded {len(team_stats_df)} team stat records.")
        for col in [c for c in REQUIRED_TEAM_STATS_COLS if c not in ['team_name', 'season', 'current_form']]:
            if col in team_stats_df.columns:
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Column '{col}' missing; filling with 0.0.")
                team_stats_df[col] = 0.0
        for col in ['team_name', 'season', 'current_form']:
            team_stats_df[col] = team_stats_df[col].astype(str).fillna('')
        return team_stats_df
    except Exception as e:
        logger.error(f"Error loading team stats data: {e}", exc_info=True)
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

# --- Betting Odds Parsing Functions (Unchanged) ---
def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    if not line_str:
        return None, None
    match = re.search(r"([\+\-]?\d+(?:\.\d+)?)\s*(?:\(?(?:[ou]?)\s*([\+\-]\d+)\)?)?", str(line_str).strip())
    if match:
        try:
            line = float(match.group(1))
        except (ValueError, TypeError):
            line = None
        odds = int(match.group(2)) if match.group(2) else -110
        return line, odds
    return None, None

def parse_moneyline_str(ml_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[int], Optional[int]]:
    home_ml, away_ml = None, None
    if not ml_str or not isinstance(ml_str, str):
        return home_ml, away_ml
    try:
        parts = ml_str.split('/')
        team_map = {}
        if len(parts) != 2:
            match_num = re.findall(r"([\+\-]\d{3,})", ml_str)
            if len(match_num) == 2:
                home_ml = int(match_num[0])
                away_ml = int(match_num[1])
                return home_ml, away_ml
            else:
                return home_ml, away_ml
        for part in parts:
            part = part.strip()
            match_num = re.search(r"([\+\-]\d+)$", part)
            if match_num:
                ml_value = int(match_num.group(1))
                team_name_part = part.replace(match_num.group(0), '').strip()
                if home_team and home_team.lower() in team_name_part.lower():
                    team_map['home'] = ml_value
                elif away_team and away_team.lower() in team_name_part.lower():
                    team_map['away'] = ml_value
                elif 'home' not in team_map and 'away' not in team_map:
                    team_map['home' if ml_value < 0 else 'away'] = ml_value
        home_ml = team_map.get('home')
        away_ml = team_map.get('away')
    except Exception as e:
        logger.warning(f"Could not parse moneyline string '{ml_str}': {e}")
    return home_ml, away_ml

def parse_spread_str(spread_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    home_line, away_line = None, None
    if not spread_str or not isinstance(spread_str, str):
        return home_line, away_line
    try:
        numbers = re.findall(r"([\+\-]?\d+(?:\.\d+)?)", spread_str)
        if len(numbers) == 2:
            val1, val2 = float(numbers[0]), float(numbers[1])
            if abs(val1 + val2) < 0.1:
                home_line = val1
                away_line = val2
            else:
                home_line = val1 if val1 < val2 else val2
                away_line = -home_line
        elif len(numbers) == 1:
            home_line = float(numbers[0])
            away_line = -home_line
        else:
            return None, None
    except Exception as e:
        logger.warning(f"Could not parse spread string '{spread_str}': {e}")
    return home_line, away_line

def parse_total_str(total_str: Optional[str]) -> Optional[float]:
    total_line = None
    if not total_str or not isinstance(total_str, str):
        return total_line
    try:
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
    chunk_size = 50
    game_id_chunks = [game_ids[i:i + chunk_size] for i in range(0, len(game_ids), chunk_size)]
    logger.info(f"Fetching betting odds for {len(game_ids)} games...")
    select_cols = "game_id, home_team, away_team, moneyline_clean, spread_clean, total_clean"
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
                if not game_id:
                    continue
                home_ml, away_ml = parse_moneyline_str(row.get('moneyline_clean'), home_team, away_team)
                home_spread, _ = parse_spread_str(row.get('spread_clean'), home_team, away_team)
                total_line = parse_total_str(row.get('total_clean'))
                default_odds = -110
                odds_dict[game_id] = {
                    'moneyline': {'home': home_ml, 'away': away_ml},
                    'spread': {'home_line': home_spread, 'away_line': -home_spread if home_spread is not None else None, 'home_odds': default_odds, 'away_odds': default_odds},
                    'total': {'line': total_line, 'over_odds': default_odds, 'under_odds': default_odds},
                    'bookmaker': 'Parsed from Supabase',
                    'last_update': pd.Timestamp.now(tz='UTC')
                }
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

# --- Calibration Function ---
def calibrate_prediction_with_odds(
    prediction: Dict,
    odds_info: Optional[Dict],
    blend_factor: float = 0.3
) -> Dict:
    calibrated = prediction.copy()
    calibrated['raw_predicted_home_score'] = prediction.get('predicted_home_score')
    calibrated['raw_predicted_away_score'] = prediction.get('predicted_away_score')
    calibrated['raw_predicted_point_diff'] = prediction.get('predicted_point_diff')
    calibrated['raw_predicted_total_score'] = prediction.get('predicted_total_score')
    calibrated['raw_win_probability'] = prediction.get('win_probability')
    calibrated['raw_lower_bound'] = prediction.get('lower_bound')
    calibrated['raw_upper_bound'] = prediction.get('upper_bound')
    calibrated['raw_confidence_pct'] = prediction.get('confidence_pct')
    calibrated['betting_odds'] = odds_info
    calibrated['calibration_blend_factor'] = blend_factor
    calibrated['is_calibrated'] = False

    if not odds_info:
        logger.debug(f"No odds info for game {prediction.get('game_id')}. Calibration skipped.")
        return calibrated

    try:
        market_spread, market_total = None, None
        spread_data = odds_info.get('spread', {})
        if spread_data.get('home_line') is not None:
            market_spread = -spread_data['home_line']
        elif spread_data.get('away_line') is not None:
            market_spread = spread_data['away_line']
        total_data = odds_info.get('total', {})
        market_total = total_data.get('line')
        pred_diff_raw = calibrated.get('raw_predicted_point_diff')
        pred_total_raw = calibrated.get('raw_predicted_total_score')
        if market_spread is None or market_total is None or pred_diff_raw is None or pred_total_raw is None:
            logger.warning(f"Calibration skipped for game {prediction.get('game_id')}: Missing required values.")
            return calibrated
        logger.info(f"Calibrating game {prediction.get('game_id')} with blend factor {blend_factor}.")
        cal_total = blend_factor * market_total + (1 - blend_factor) * pred_total_raw
        cal_diff = blend_factor * market_spread + (1 - blend_factor) * pred_diff_raw
        cal_home = (cal_total + cal_diff) / 2.0
        cal_away = cal_total - cal_home
        cal_win_prob = 1 / (1 + math.exp(-0.15 * cal_diff))
        calibrated['predicted_home_score'] = round(cal_home, 1)
        calibrated['predicted_away_score'] = round(cal_away, 1)
        calibrated['predicted_point_diff'] = round(cal_diff, 1)
        calibrated['predicted_total_score'] = round(cal_total, 1)
        calibrated['win_probability'] = round(cal_win_prob, 3)
        calibrated['lower_bound'] = calibrated['raw_lower_bound']
        calibrated['upper_bound'] = calibrated['raw_upper_bound']
        calibrated['confidence_pct'] = calibrated['raw_confidence_pct']
        calibrated['is_calibrated'] = True
        logger.info(f"Calibrated game {prediction.get('game_id')}: H {cal_home:.1f}, A {cal_away:.1f}, Diff {cal_diff:+.1f}, Total {cal_total:.1f}, WP {cal_win_prob:.1%}")
    except Exception as e:
        logger.error(f"Error during calibration for game {prediction.get('game_id')}: {e}", exc_info=True)
    return calibrated

# --- Reporting Utility ---
def display_prediction_summary(predictions: List[Dict]) -> None:
    header_line = "=" * 145
    header_title = " " * 55 + "NBA PREGAME PREDICTION SUMMARY"
    header_cols = (f"{'DATE':<11} {'MATCHUP':<30} {'FINAL SCORE':<12} {'FINAL DIFF':<10} "
                   f"{'L/U BOUND':<12} {'CONF %':<7} {'RAW SCORE':<12} {'RAW DIFF':<10} {'WIN PROB':<10} {'CALIB?':<6}")
    header_sep = "-" * 145

    print(f"\n{header_line}\n{header_title}\n{header_line}")
    print(header_cols)
    print(header_sep)
    try:
        predictions_df = pd.DataFrame(predictions)
        predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'], errors='coerce')
        predictions_df = predictions_df.dropna(subset=['game_date'])
        predictions_df = predictions_df.sort_values(['game_date', 'predicted_point_diff'], key=lambda x: abs(x) if x.name == 'predicted_point_diff' else x)
    except Exception as e:
        logger.error(f"Error sorting predictions for display: {e}")
        predictions_df = pd.DataFrame(predictions)
    for _, game in predictions_df.iterrows():
        try:
            date_str = game['game_date'].strftime('%Y-%m-%d') if pd.notna(game['game_date']) else "N/A"
            matchup = f"{game.get('home_team','?')[:14]} vs {game.get('away_team','?')[:14]}"[:30]
            is_calib = game.get('is_calibrated', False)
            calib_str = "Yes" if is_calib else "No "
            final_score = f"{game.get('predicted_home_score', np.nan):.1f}-{game.get('predicted_away_score', np.nan):.1f}"
            final_diff = game.get('predicted_point_diff', np.nan)
            final_diff_str = f"{final_diff:<+10.1f}" if pd.notna(final_diff) else "N/A     "
            final_win_prob = game.get('win_probability', np.nan)
            win_prob_str = f"{final_win_prob*100:.1f}%" if pd.notna(final_win_prob) else "N/A       "
            lower_b = game.get('lower_bound', np.nan)
            upper_b = game.get('upper_bound', np.nan)
            bound_str = f"{lower_b:.1f}/{upper_b:.1f}" if pd.notna(lower_b) and pd.notna(upper_b) else "N/A         "
            conf_str = f"{game.get('confidence_pct', np.nan):.1f}" if pd.notna(game.get('confidence_pct', np.nan)) else "N/A   "
            raw_home = game.get('raw_predicted_home_score')
            raw_away = game.get('raw_predicted_away_score')
            raw_score = f"{raw_home:.1f}-{raw_away:.1f}" if pd.notna(raw_home) and pd.notna(raw_away) else "N/A"
            raw_diff = game.get('raw_predicted_point_diff', np.nan)
            raw_diff_str = f"{raw_diff:<+10.1f}" if pd.notna(raw_diff) else "N/A     "
            print(f"{date_str:<11} {matchup:<30} {final_score:<12} {final_diff_str:<10} "
                  f"{bound_str:<12} {conf_str:<7} {raw_score:<12} {raw_diff_str:<10} {win_prob_str:<10} {calib_str:<6}")
        except Exception as display_e:
            logger.error(f"Error displaying summary row for game {game.get('game_id', 'N/A')}: {display_e}")
    print("=" * 145)

# --- Main Prediction Orchestration Function ---
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW,
    model_dir: Path = MODELS_DIR,
    calibrate_with_odds: bool = True,
    blend_factor: float = 0.3,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES
) -> Tuple[List[Dict], List[Dict]]:
    """
    Orchestrates the prediction process using on-the-fly feature generation.
    Returns a tuple: (final prediction list, raw prediction list).
    """
    logger.info("--- Starting NBA Prediction Pipeline (On-the-Fly Features) ---")
    start_time = time.time()
    if not PROJECT_MODULES_IMPORTED:
        logger.critical("Core project modules not imported.")
        return [], []
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error("Supabase Client initialization failed.")
        return [], []
    try:
        feature_engine = NBAFeatureEngine(supabase_client=supabase_client, debug=False)
        logger.debug("FeatureEngine initialized.")
        coverage_stats_path = REPORTS_DIR / "historical_coverage_stats.csv"
        if coverage_stats_path.is_file():
            hist_stats_df = pd.read_csv(coverage_stats_path)
            if 'quarter' in hist_stats_df.columns and 'actual_coverage' in hist_stats_df.columns:
                logger.info(f"Loaded historical coverage stats from {coverage_stats_path}")
            else:
                logger.warning(f"Coverage stats file {coverage_stats_path} missing required columns. Ignoring.")
                hist_stats_df = None
        else:
            logger.info(f"Coverage stats file not found at {coverage_stats_path}. Using default uncertainty settings.")
            hist_stats_df = None
        uncertainty_estimator = PredictionUncertaintyEstimator(
            historical_coverage_stats=hist_stats_df,
            debug=False
        )
        logger.debug("PredictionUncertaintyEstimator initialized.")
    except Exception as init_e:
        logger.error(f"FATAL: Failed to initialize core engines: {init_e}", exc_info=True)
        return [], []
    if feature_engine is None or uncertainty_estimator is None:
        logger.error("Core engine initialization failed. Cannot proceed.")
        return [], []
    logger.info("Step 1: Loading data...")
    historical_data = load_recent_historical_data(supabase_client, days_lookback=historical_lookback)
    team_stats_data = load_team_stats_data(supabase_client)
    upcoming_games_df = fetch_upcoming_games_data(supabase_client, days_window=days_window)
    if upcoming_games_df.empty:
        logger.warning("No upcoming games found.")
        return [], []
    if historical_data.empty:
        logger.warning("Historical data empty; feature quality may be reduced.")
    if team_stats_data.empty:
        logger.warning("Team stats data empty; context features may use defaults.")
    logger.info("Step 2: Loading trained models...")
    try:
        prediction_models, required_features = load_trained_models(model_dir)
    except Exception as load_e:
        logger.error(f"Model loading failed: {load_e}", exc_info=True)
        return [], []
    if not prediction_models or not required_features:
        logger.error("Models or required feature list missing after load attempt.")
        return [], []
    logger.info("Step 3: Generating features for prediction...")
    try:
        all_cols_needed = set(REQUIRED_HISTORICAL_COLS) | set(REQUIRED_TEAM_STATS_COLS) | set(UPCOMING_GAMES_COLS)
        for df, name in [(historical_data, 'historical'), (upcoming_games_df, 'upcoming')]:
            if df is None:
                continue
            for col in all_cols_needed:
                if col not in df.columns:
                    df[col] = np.nan if col not in ['game_id', 'home_team', 'away_team', 'game_date', 'season', 'current_form'] else ''
        hist_cols_to_concat = [c for c in historical_data.columns if c in all_cols_needed or c in TARGET_COLUMNS]
        upcoming_cols_to_concat = [c for c in upcoming_games_df.columns if c in all_cols_needed]
        common_cols = list(set(hist_cols_to_concat) & set(upcoming_cols_to_concat))
        combined_df_for_features = pd.concat(
            [historical_data[common_cols], upcoming_games_df[common_cols]],
            ignore_index=True
        ).sort_values(by='game_date', kind='mergesort').reset_index(drop=True)
        prediction_rolling_windows = [5, 10, 20]
        logger.info(f"Using rolling windows {prediction_rolling_windows} for feature generation.")
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
    upcoming_game_ids = upcoming_games_df['game_id'].unique()
    predict_features_df = features_df_full[features_df_full['game_id'].isin(upcoming_game_ids)].copy()
    logger.info(f"Features generated for {len(predict_features_df)} upcoming games.")
    logger.info(f"Step 4: Selecting and validating {len(required_features)} features...")
    missing_in_df = [f for f in required_features if f not in predict_features_df.columns]
    if missing_in_df:
        logger.error(f"FATAL: Missing required model columns: {missing_in_df}")
        return [], []
    try:
        X_predict = predict_features_df[required_features].copy()
        if X_predict.isnull().any().any():
            nan_cols = X_predict.columns[X_predict.isnull().any()].tolist()
            logger.warning(f"NaNs in features: {nan_cols}. Filling with 0.0.")
            X_predict = X_predict.fillna(0.0)
    except Exception as e:
        logger.error(f"Error during feature selection/validation: {e}", exc_info=True)
        return [], []
    logger.info("Step 5: Generating raw predictions and uncertainty...")
    raw_predictions_list = []
    model_preds_dfs = {}
    for name, predictor in prediction_models.items():
        try:
            preds_df = predictor.predict(X_predict)
            if preds_df is None or preds_df.empty or 'predicted_home_score' not in preds_df.columns:
                raise ValueError("Invalid prediction output")
            preds_df.index = X_predict.index
            model_preds_dfs[name] = preds_df[['predicted_home_score', 'predicted_away_score']]
        except Exception as e:
            logger.error(f"Error predicting with {name}: {e}", exc_info=True)
            model_preds_dfs[name] = None
    for idx in X_predict.index:
        game_info = predict_features_df.loc[idx]
        game_id = str(game_info['game_id'])
        home_team = game_info['home_team']
        away_team = game_info['away_team']
        game_date = pd.to_datetime(game_info['game_date'])
        component_predictions_this_game = {}
        ensemble_home_score = 0.0
        ensemble_away_score = 0.0
        total_weight = 0.0
        for name, preds_df in model_preds_dfs.items():
            if preds_df is not None and idx in preds_df.index:
                home_pred = preds_df.loc[idx, 'predicted_home_score']
                away_pred = preds_df.loc[idx, 'predicted_away_score']
            else:
                avg_score = feature_engine.defaults.get('avg_pts_for', 115.0)
                home_pred = avg_score
                away_pred = avg_score
                logger.debug(f"Using fallback for {name} on game {game_id}")
            component_predictions_this_game[name] = {'home': home_pred, 'away': away_pred}
            weight = ENSEMBLE_WEIGHTS.get(name, 0)
            if weight > 0 and pd.notna(home_pred) and pd.notna(away_pred):
                ensemble_home_score += home_pred * weight
                ensemble_away_score += away_pred * weight
                total_weight += weight
        if total_weight > 1e-6:
            ensemble_home_score /= total_weight
            ensemble_away_score /= total_weight
        else:
            logger.warning(f"Total weight zero for game {game_id}. Using fallback average.")
            home_scores = [p['home'] for p in component_predictions_this_game.values() if pd.notna(p['home'])]
            away_scores = [p['away'] for p in component_predictions_this_game.values() if pd.notna(p['away'])]
            avg_score = feature_engine.defaults.get('avg_pts_for', 115.0)
            ensemble_home_score = np.mean(home_scores) if home_scores else avg_score
            ensemble_away_score = np.mean(away_scores) if away_scores else avg_score
        point_diff = ensemble_home_score - ensemble_away_score
        ensemble_total_score = ensemble_home_score + ensemble_away_score
        win_prob = 1 / (1 + math.exp(-0.15 * point_diff))
        lower_b, upper_b, conf_pct = np.nan, np.nan, np.nan
        try:
            hist_acc_dict = None  # Optionally load historical accuracy stats
            lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
                prediction=ensemble_total_score,
                current_quarter=0,
                historic_accuracy=hist_acc_dict
            )
            logger.debug(f"Game {game_id}: Total={ensemble_total_score:.1f}, Interval=[{lower_b:.1f}, {upper_b:.1f}], Conf={conf_pct:.1f}%")
        except Exception as uncertainty_e:
            logger.error(f"Error calculating uncertainty for game {game_id}: {uncertainty_e}", exc_info=True)
        raw_pred_dict = {
            'game_id': game_id,
            'game_date': game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else None,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_score': round(ensemble_home_score, 1),
            'predicted_away_score': round(ensemble_away_score, 1),
            'predicted_point_diff': round(point_diff, 1),
            'predicted_total_score': round(ensemble_total_score, 1),
            'win_probability': round(win_prob, 3),
            'lower_bound': round(lower_b, 1) if pd.notna(lower_b) else None,
            'upper_bound': round(upper_b, 1) if pd.notna(upper_b) else None,
            'confidence_pct': round(conf_pct, 1) if pd.notna(conf_pct) else None,
            'component_predictions': {name: {'home': round(preds['home'], 1), 'away': round(preds['away'], 1)}
                                        for name, preds in component_predictions_this_game.items()},
            'component_weights': ENSEMBLE_WEIGHTS,
            'is_calibrated': False
        }
        raw_predictions_list.append(raw_pred_dict)
    if not raw_predictions_list:
        logger.error("No raw predictions generated.")
        return [], []
    logger.info("--- Raw Component Predictions (Before Calibration) ---")
    for raw_pred in raw_predictions_list:
        game_id = raw_pred.get('game_id', 'N/A')
        comp_preds = raw_pred.get('component_predictions', {})
        log_str = f"Game {game_id}: "
        preds_parts = []
        for model_name in ENSEMBLE_WEIGHTS.keys():
            if model_name in comp_preds:
                pred_vals = comp_preds[model_name]
                home_str = f"{pred_vals.get('home', 'NaN'):.1f}" if isinstance(pred_vals.get('home'), (int, float)) and pd.notna(pred_vals.get('home')) else str(pred_vals.get('home'))
                away_str = f"{pred_vals.get('away', 'NaN'):.1f}" if isinstance(pred_vals.get('away'), (int, float)) and pd.notna(pred_vals.get('away')) else str(pred_vals.get('away'))
                preds_parts.append(f"{model_name.split('_')[0].upper()}=H{home_str}/A{away_str}")
            else:
                preds_parts.append(f"{model_name.split('_')[0].upper()}=N/A")
        log_str += ", ".join(preds_parts)
        logger.info(log_str)
    logger.info("--- End Raw Component Predictions ---")
    logger.info("Step 6: Fetching odds and calibrating predictions (optional)...")
    final_predictions = []
    if calibrate_with_odds:
        game_ids_to_predict = [str(p['game_id']) for p in raw_predictions_list]
        odds_dict = fetch_and_parse_betting_odds(supabase_client, game_ids_to_predict)
        if not odds_dict:
            logger.warning("No betting odds fetched. Using raw predictions.")
        logger.info(f"Calibrating {len(raw_predictions_list)} predictions with odds...")
        for pred in raw_predictions_list:
            game_id = str(pred.get('game_id'))
            odds_info = odds_dict.get(game_id)
            calibrated_pred = calibrate_prediction_with_odds(pred, odds_info, blend_factor)
            final_predictions.append(calibrated_pred)
    else:
        logger.info("Skipping odds fetching and calibration.")
        for pred in raw_predictions_list:
            pred_with_raw = pred.copy()
            pred_with_raw['raw_predicted_home_score'] = pred.get('predicted_home_score')
            pred_with_raw['raw_predicted_away_score'] = pred.get('predicted_away_score')
            pred_with_raw['raw_predicted_point_diff'] = pred.get('predicted_point_diff')
            pred_with_raw['raw_predicted_total_score'] = pred.get('predicted_total_score')
            pred_with_raw['raw_win_probability'] = pred.get('win_probability')
            pred_with_raw['raw_lower_bound'] = pred.get('lower_bound')
            pred_with_raw['raw_upper_bound'] = pred.get('upper_bound')
            pred_with_raw['raw_confidence_pct'] = pred.get('confidence_pct')
            pred_with_raw['is_calibrated'] = False
            final_predictions.append(pred_with_raw)
    logger.info("Step 7: Displaying prediction summary...")
    display_prediction_summary(final_predictions)
    end_time = time.time()
    logger.info(f"--- NBA Prediction Pipeline Finished in {end_time - start_time:.2f} seconds ---")
    return final_predictions, raw_predictions_list

# --- Script Execution ---
if __name__ == "__main__":
    logger.info("--- Running Prediction Script Directly ---")
    config_days_window = DEFAULT_UPCOMING_DAYS_WINDOW
    config_calibrate = True
    config_blend_factor = 0.3
    config_hist_lookback = 365
    final_preds, raw_preds = generate_predictions(
        days_window=config_days_window,
        model_dir=MODELS_DIR,
        calibrate_with_odds=config_calibrate,
        blend_factor=config_blend_factor,
        historical_lookback=config_hist_lookback
    )
    if final_preds:
        logger.info(f"Successfully generated {len(final_preds)} predictions.")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"predictions_output_{ts}.csv"
            df_to_save = pd.DataFrame(final_preds)
            cols_to_drop_for_csv = ['component_predictions', 'component_weights', 'betting_odds']
            df_to_save = df_to_save.drop(columns=cols_to_drop_for_csv, errors='ignore')
            numeric_cols_out = ['predicted_home_score', 'predicted_away_score', 'predicted_point_diff',
                                'predicted_total_score', 'win_probability', 'lower_bound', 'upper_bound',
                                'confidence_pct', 'raw_predicted_home_score', 'raw_predicted_away_score',
                                'raw_predicted_point_diff', 'raw_predicted_total_score', 'raw_win_probability',
                                'raw_lower_bound', 'raw_upper_bound', 'raw_confidence_pct', 'calibration_blend_factor']
            for nc in numeric_cols_out:
                if nc in df_to_save.columns:
                    df_to_save[nc] = pd.to_numeric(df_to_save[nc], errors='coerce')
            df_to_save.to_csv(output_path, index=False, float_format="%.3f")
            logger.info(f"Final predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions to CSV: {e}", exc_info=True)
    else:
        logger.warning("Prediction pipeline finished but produced no results.")
    logger.info("--- Prediction Script Finished ---")
