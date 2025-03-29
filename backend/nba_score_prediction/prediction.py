# backend/nba_score_prediction/prediction.py
"""
Generates pre-game predictions for upcoming NBA games using the trained ensemble model,
integrates betting odds, and calibrates predictions.
"""

import os
import re
import math
import pytz
import joblib
import traceback
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text # Added text for literal SQL
from sklearn.impute import SimpleImputer # Keep for feature handling if needed
from typing import List, Dict, Optional, Any, Tuple

# --- Project Module Imports ---
try:
    # Assuming standard package structure allows these relative imports
    from .feature_engineering import NBAFeatureEngine, UncertaintyEstimator # Import necessary classes
    from .models import (
        XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor, BaseScorePredictor,
        QuarterSpecificModelSystem # Import base and specific predictors if needed for loading
    )
    from .ensemble import EnsembleWeightManager # Assuming ensemble logic lives here
    # from .data import load_historical_data_db # Ideal: Use data module function
except ImportError as e:
    logging.error(f"Error importing project modules: {e}. Prediction script may fail.", exc_info=True)
    # Define dummy classes/functions if imports fail to allow script to load
    class NBAFeatureEngine: pass
    class UncertaintyEstimator: pass
    class BaseScorePredictor: pass
    class XGBoostScorePredictor(BaseScorePredictor): pass
    class RandomForestScorePredictor(BaseScorePredictor): pass
    class RidgeScorePredictor(BaseScorePredictor): pass
    class EnsembleWeightManager: pass

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent # Project root assuming backend/ is one level down
MODELS_DIR = BASE_DIR / "models" / "saved"
DEFAULT_ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_metadata.joblib" # Example path for ensemble metadata/components
DEFAULT_PREDICTOR_MODEL_DIR = MODELS_DIR # Predictor models likely saved here directly by models.py

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
BETTING_THRESHOLD = 2.0 # Example threshold for highlighting betting edges
HOME_ADVANTAGE = 3.0 # Standard home court advantage points

# --- Database and Supabase (Optional) ---
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_db_engine():
    """Creates a SQLAlchemy engine if DATABASE_URL is set."""
    if DATABASE_URL:
        try:
            engine = create_engine(DATABASE_URL)
            logger.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logger.error(f"Error creating database engine: {e}", exc_info=True)
    else:
        logger.warning("DATABASE_URL not set. Database operations will be skipped.")
    return None

def get_supabase_client():
    """Creates a Supabase client if credentials are set."""
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client, Client
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client created successfully.")
            return supabase
        except ImportError:
             logger.error("Supabase library not found. Install with 'pip install supabase'")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}", exc_info=True)
    else:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Supabase operations will be skipped.")
    return None

# --- Data Loading Functions ---

def load_historical_data(engine, days_lookback=90) -> pd.DataFrame:
    """Loads recent historical game data from the database."""
    if not engine:
        logger.error("Database engine not available. Cannot load historical data.")
        return pd.DataFrame()

    # Use a wider lookback for feature calculation stability
    start_date = (datetime.now(PACIFIC_TZ) - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical data from {start_date} onwards...")

    query = text(f"""
        SELECT * FROM nba_historical_game_stats
        WHERE game_date >= :start_date
        ORDER BY game_date ASC
    """) # Use ORDER BY ASC for time-series features
    try:
        with engine.connect() as connection:
             historical_df = pd.read_sql(query, connection, params={'start_date': start_date}, parse_dates=['game_date'])
        logger.info(f"Loaded {len(historical_df)} historical games from database.")
        if 'game_date' in historical_df.columns:
             historical_df['game_date'] = pd.to_datetime(historical_df['game_date'], errors='coerce').dt.tz_localize(None) # Ensure timezone naive for consistency
        return historical_df
    except Exception as e:
        logger.error(f"Error loading historical data from DB: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

def fetch_upcoming_games(supabase_client, days_window=2) -> pd.DataFrame:
    """Fetches upcoming games schedule from Supabase."""
    if not supabase_client:
        logger.error("Supabase client not available. Cannot fetch upcoming games.")
        return pd.DataFrame()

    now_pt = datetime.now(PACIFIC_TZ)
    # Fetch games starting from today PT date
    start_date_pt_str = now_pt.strftime('%Y-%m-%d')
    # Fetch games up to 'days_window' days from now
    end_date_pt_str = (now_pt + timedelta(days=days_window + 1)).strftime('%Y-%m-%d') # Go one day further for buffer

    logger.info(f"Fetching upcoming games from {start_date_pt_str} up to {end_date_pt_str}...")
    try:
        response = supabase_client.table("nba_game_schedule") \
            .select("game_id, scheduled_time, home_team, away_team, game_date") \
            .gte("game_date", start_date_pt_str) \
            .lt("game_date", end_date_pt_str) \
            .execute()

        data = response.data
        if not data:
            logger.warning("No upcoming games found in Supabase for the specified window.")
            return pd.DataFrame()

        games_df = pd.DataFrame(data)
        # Convert scheduled_time to PT datetime
        if "scheduled_time" in games_df.columns:
             games_df["scheduled_time"] = pd.to_datetime(games_df["scheduled_time"], errors='coerce')
             games_df["game_time_pt"] = games_df["scheduled_time"].dt.tz_convert(PACIFIC_TZ)
             # Use game_date directly from DB if available, otherwise derive from PT time
             if 'game_date' not in games_df.columns or games_df['game_date'].isnull().all():
                  games_df["game_date"] = games_df["game_time_pt"].dt.date
             else:
                  games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors='coerce').dt.date
        else:
             logger.warning("Column 'scheduled_time' not found in nba_game_schedule data.")
             # Attempt to use game_date if available
             if 'game_date' in games_df.columns:
                  games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors='coerce').dt.date
                  # Estimate a game time if needed elsewhere (though usually date is sufficient for pregame)
                  games_df["game_time_pt"] = games_df["game_date"].apply(lambda d: PACIFIC_TZ.localize(datetime.combine(d, datetime.min.time().replace(hour=18)))) # Estimate 6pm PT
             else:
                  logger.error("Neither 'scheduled_time' nor 'game_date' found. Cannot process upcoming games.")
                  return pd.DataFrame()


        # Ensure required columns exist and have correct types
        for col in ["home_team", "away_team"]: games_df[col] = games_df[col].astype(str)
        games_df['game_id'] = games_df['game_id'].astype(str)
        games_df['game_date'] = pd.to_datetime(games_df['game_date'], errors='coerce') # Ensure game_date is datetime object

        # Filter games that haven't started yet based on current PT time
        games_df = games_df[games_df['game_time_pt'] > now_pt].copy()
        logger.info(f"Found {len(games_df)} upcoming games that haven't started.")
        return games_df.sort_values(by='game_time_pt')

    except Exception as e:
        logger.error(f"Error fetching upcoming games from Supabase: {e}", exc_info=True)
        return pd.DataFrame()

# --- Betting Odds Parsing ---

def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    """Parses a combined line and odds string (e.g., '-3.5 (-110)')"""
    if not line_str: return None, None
    line, odds = None, None
    # Match patterns like -3.5 (-110), +7 (+100), 220.5 o(-115) etc.
    match = re.search(r"([\+\-]?[\d\.]+)\s*(?:\(?(?:[ou]?)([\+\-]\d+)\)?)?", str(line_str).strip())
    if match:
        try: line = float(match.group(1))
        except (ValueError, TypeError): pass
        try: odds = int(match.group(2))
        except (ValueError, TypeError): odds = -110 # Default odds if not found
    return line, odds

def fetch_and_parse_betting_odds(supabase_client, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches and parses betting odds for given game IDs from Supabase."""
    if not supabase_client or not game_ids:
        logger.warning("Supabase client or game_ids list empty. Skipping odds fetch.")
        return {}

    odds_dict = {}
    # Fetch data in chunks if game_ids list is very large (e.g., > 50)
    chunk_size = 50
    game_id_chunks = [game_ids[i:i + chunk_size] for i in range(0, len(game_ids), chunk_size)]

    logger.info(f"Fetching betting odds for {len(game_ids)} game IDs in {len(game_id_chunks)} chunks...")

    for chunk in game_id_chunks:
        try:
            response = supabase_client.table("nba_betting_odds") \
                .select("game_id, bookmaker, home_team, away_team, moneyline_home, moneyline_away, "
                        "spread_home_line, spread_home_odds, spread_away_line, spread_away_odds, "
                        "total_line, total_over_odds, total_under_odds, last_update") \
                .in_("game_id", chunk) \
                .order("last_update", desc=True) \
                .execute()

            data = response.data
            if not data:
                logger.warning(f"No betting odds data found for game IDs chunk: {chunk}")
                continue

            df = pd.DataFrame(data)
            # Get the most recent odds per game_id (assuming multiple bookmakers might exist)
            # Prioritize certain bookmakers if needed, otherwise take latest update
            df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
            df = df.sort_values('last_update', ascending=False).drop_duplicates('game_id')

            for _, row in df.iterrows():
                game_id = str(row.get('game_id'))
                if not game_id: continue # Should not happen if fetched by game_id

                # Convert raw odds/lines using flexible parsing
                home_spread, home_spread_odds = parse_odds_line(row.get('spread_home_line'))
                away_spread, away_spread_odds = parse_odds_line(row.get('spread_away_line'))
                total_line, total_over_odds = parse_odds_line(row.get('total_line')) # Assume total_line holds the main line
                _, total_under_odds = parse_odds_line(row.get('total_under_odds')) # Only extract odds for under

                odds_dict[game_id] = {
                    'moneyline': {
                        'home': int(row['moneyline_home']) if pd.notna(row.get('moneyline_home')) else None,
                        'away': int(row['moneyline_away']) if pd.notna(row.get('moneyline_away')) else None
                    },
                    'spread': {
                        'home_line': home_spread,
                        'away_line': away_spread,
                        'home_odds': int(row['spread_home_odds']) if pd.notna(row.get('spread_home_odds')) else home_spread_odds or -110,
                        'away_odds': int(row['spread_away_odds']) if pd.notna(row.get('spread_away_odds')) else away_spread_odds or -110
                    },
                    'total': {
                        'line': total_line,
                        'over_odds': int(row['total_over_odds']) if pd.notna(row.get('total_over_odds')) else total_over_odds or -110,
                        'under_odds': int(row['total_under_odds']) if pd.notna(row.get('total_under_odds')) else total_under_odds or -110
                    },
                    'bookmaker': row.get('bookmaker'),
                    'last_update': row.get('last_update')
                }
            logger.info(f"Processed odds for {len(df)} games in chunk.")

        except Exception as e:
            logger.error(f"Error fetching/parsing betting odds chunk: {e}", exc_info=True)

    logger.info(f"Finished fetching odds. Found data for {len(odds_dict)} games.")
    return odds_dict


# --- Prediction Logic ---

def load_ensemble_model(model_path: Path = DEFAULT_ENSEMBLE_MODEL_PATH) -> Optional[Dict]:
    """Loads the ensemble model metadata and component information."""
    if not model_path.exists():
        logger.error(f"Ensemble model file not found: {model_path}")
        return None
    try:
        ensemble_data = joblib.load(model_path)
        logger.info(f"Loaded ensemble model data from {model_path}")
        # Basic validation
        if not isinstance(ensemble_data, dict) or 'model_paths' not in ensemble_data or 'weights' not in ensemble_data:
             logger.error("Loaded ensemble data is missing required keys ('model_paths', 'weights').")
             return None
        return ensemble_data
    except Exception as e:
        logger.error(f"Error loading ensemble model from {model_path}: {e}", exc_info=True)
        return None

def make_single_game_prediction(
    game_data: pd.Series,
    feature_engine: NBAFeatureEngine,
    historical_data: pd.DataFrame,
    ensemble_data: Dict,
    team_metrics: Optional[pd.DataFrame] = None
) -> Optional[Dict]:
    """Generates features and predictions for a single upcoming game."""
    try:
        home_team = game_data['home_team']
        away_team = game_data['away_team']
        game_date = pd.to_datetime(game_data['game_date']) # Ensure datetime
        game_id = str(game_data.get('game_id', f"{home_team}_{away_team}_{game_date.strftime('%Y%m%d')}"))
        logger.debug(f"Starting prediction for game_id: {game_id}")

        # --- Feature Generation ---
        upcoming_df = pd.DataFrame([game_data]) # Create DataFrame for feature engine
        # Ensure team metrics are available
        if team_metrics is None:
             logger.warning("Team metrics not provided, calculating on the fly (less efficient).")
             team_metrics = feature_engine.create_team_metrics(historical_data)

        # Use feature engine's pipeline
        pregame_features_df = feature_engine.generate_pregame_features(upcoming_df, historical_data, team_metrics)
        final_features_df, _ = feature_engine.finalize_features_for_prediction(
            pregame_features_df,
            required_features=ensemble_data.get('features', None) # Pass required features from loaded ensemble data
        )

        if final_features_df.empty:
            logger.error(f"Feature generation failed for game_id: {game_id}")
            return None

        X_predict = final_features_df.drop(columns=['game_id', 'game_date', 'home_team', 'away_team'], errors='ignore')
        logger.debug(f"Generated {X_predict.shape[1]} features for prediction.")

        # --- Model Prediction ---
        component_predictions = {}
        component_models = {}
        weights = ensemble_data['weights']
        model_paths = ensemble_data['model_paths']

        # Load component models
        for name, path_str in model_paths.items():
             path = Path(path_str)
             if not path.exists():
                  # Attempt to find latest model if specific path fails (robustness)
                  logger.warning(f"Model file {path} not found. Trying to load latest '{name}' model...")
                  try:
                       predictor_map = {
                           'xgboost': XGBoostScorePredictor, 'random_forest': RandomForestScorePredictor, 'ridge': RidgeScorePredictor
                       }
                       if name in predictor_map:
                           predictor = predictor_map[name](model_dir=DEFAULT_PREDICTOR_MODEL_DIR, model_name=f"{name}_score_predictor")
                           predictor.load_model() # Loads latest by convention
                           component_models[name] = predictor
                           logger.info(f"Loaded latest {name} model successfully.")
                       else: raise FileNotFoundError("Unknown model type for auto-load")
                  except Exception as load_err:
                       logger.error(f"Failed to load component model '{name}' from path {path} or latest: {load_err}", exc_info=True)
                       return None # Cannot proceed without all components
             else:
                  try:
                       # Assuming models saved using structure from models.py (joblib dict)
                       model_data = joblib.load(path)
                       # Simple check, assumes structure from models.py save
                       if 'pipeline_home' in model_data and 'pipeline_away' in model_data:
                            # We need a predictor instance to use predict method correctly
                            predictor_map = {
                               'xgboost': XGBoostScorePredictor, 'random_forest': RandomForestScorePredictor, 'ridge': RidgeScorePredictor
                            }
                            base_name = name.split('_')[0] # e.g., 'xgboost' from 'xgboost_score_predictor'
                            if base_name in predictor_map:
                                 predictor = predictor_map[base_name]()
                                 predictor.load_model(filepath=path) # Load the specific saved model
                                 component_models[name] = predictor
                            else:
                                 logger.error(f"Unknown model type '{name}' found at {path}")
                                 return None
                       else:
                            logger.error(f"Loaded model file {path} has unexpected structure.")
                            return None
                  except Exception as e:
                       logger.error(f"Error loading model component {name} from {path}: {e}", exc_info=True)
                       return None

        # Get individual predictions (assuming model classes handle predict)
        for name, predictor in component_models.items():
             try:
                 preds_df = predictor.predict(X_predict) # Returns DataFrame['predicted_home_score', 'predicted_away_score']
                 if preds_df is not None and not preds_df.empty:
                      # Assuming the ensemble primarily predicts TOTAL score based on previous logic
                      # We will average the component total score predictions
                      component_predictions[name] = preds_df['predicted_home_score'].iloc[0] + preds_df['predicted_away_score'].iloc[0]
                 else:
                      logger.warning(f"Predictor {name} returned None or empty DataFrame.")
                      component_predictions[name] = 220.0 # Fallback average total
             except Exception as e:
                  logger.error(f"Error predicting with component {name}: {e}", exc_info=True)
                  component_predictions[name] = 220.0 # Fallback

        # Calculate weighted ensemble total score
        ensemble_total_score = 0.0
        total_weight = 0.0
        for name, pred_total in component_predictions.items():
            weight = weights.get(name, 0)
            if weight > 0:
                ensemble_total_score += pred_total * weight
                total_weight += weight
        if total_weight > 1e-6:
            ensemble_total_score /= total_weight
        else:
             logger.warning(f"Total weight for ensemble is zero or near-zero for game {game_id}. Using average.")
             ensemble_total_score = np.mean(list(component_predictions.values()))


        # Estimate individual scores and point differential using home advantage
        home_score = (ensemble_total_score + HOME_ADVANTAGE) / 2
        away_score = ensemble_total_score - home_score
        point_diff = home_score - away_score

        # Compute win probability (simple logistic function based on point diff)
        # The constant 0.15 can be tuned based on historical data fit
        win_prob = 1 / (1 + math.exp(-0.15 * point_diff))

        # Get uncertainty/confidence if estimator available (example integration)
        lower_bound, upper_bound, confidence = None, None, None
        if 'UncertaintyEstimator' in globals(): # Check if class was imported/defined
            try:
                estimator = UncertaintyEstimator() # Needs instantiation based on its design
                # Needs game state features if used in-game; pre-game might use defaults
                # Example call structure, adjust based on UncertaintyEstimator design
                # lower_bound, upper_bound, _ = estimator.calculate_prediction_interval(predicted_score=home_score, ...)
                lower_bound, upper_bound, confidence = estimator.dynamically_adjust_interval(prediction=home_score, current_quarter=0) # Use pregame (Q0)
            except Exception as e:
                 logger.warning(f"Could not calculate uncertainty: {e}")


        # --- Assemble Result ---
        prediction_result = {
            'game_id': game_id,
            'game_date': game_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_score': round(home_score, 1),
            'predicted_away_score': round(away_score, 1),
            'predicted_point_diff': round(point_diff, 1),
            'predicted_total_score': round(ensemble_total_score, 1),
            'win_probability': round(win_prob, 3),
            'component_predictions': {k: round(v, 1) for k,v in component_predictions.items()}, # Store component totals
            'component_weights': weights,
            # Add uncertainty if calculated
            'lower_bound': round(lower_bound, 1) if lower_bound is not None else None,
            'upper_bound': round(upper_bound, 1) if upper_bound is not None else None,
            'prediction_confidence': round(confidence, 2) if confidence is not None else None,
        }
        logger.info(f"Prediction generated for {game_id}: H {home_score:.1f} - A {away_score:.1f} (WP: {win_prob:.1%})")
        return prediction_result

    except Exception as e:
        logger.error(f"Error during prediction process for game {game_data.get('game_id', 'N/A')}: {e}", exc_info=True)
        return None

# --- Calibration ---

def calibrate_prediction_with_odds(
    prediction: Dict,
    odds_info: Optional[Dict],
    blend_factor: float = 0.7 # How much to trust market odds (0=model only, 1=market only)
    ) -> Dict:
    """Calibrates a single prediction dictionary with betting odds using a simple blend."""

    calibrated = prediction.copy() # Start with original prediction
    calibrated['betting_odds'] = odds_info # Store odds used for calibration
    calibrated['calibration_blend_factor'] = blend_factor

    if not odds_info:
        logger.debug(f"No odds info provided for game {prediction.get('game_id')}. Returning original prediction.")
        return calibrated

    try:
        # --- Extract Market Lines ---
        market_spread = None
        if odds_info.get('spread') and odds_info['spread'].get('home_line') is not None:
            # Use home line; market spread is negative of home team's line handicap
             market_spread = -odds_info['spread']['home_line']
        elif odds_info.get('spread') and odds_info['spread'].get('away_line') is not None:
             # Use away line; market spread is positive of away team's line handicap
             market_spread = odds_info['spread']['away_line']

        market_total = odds_info.get('total', {}).get('line')

        if market_spread is None or market_total is None:
             logger.warning(f"Market spread ({market_spread}) or total ({market_total}) is None for {prediction.get('game_id')}. Cannot calibrate.")
             return calibrated # Return original prediction if market data incomplete

        logger.debug(f"Game {prediction.get('game_id')}: Market Spread={market_spread:.1f}, Market Total={market_total:.1f}")

        # --- Extract Model Predictions ---
        pred_diff = prediction['predicted_point_diff']
        pred_total = prediction['predicted_total_score']

        # --- Simple Blend Calibration ---
        # Blend total score and point difference separately
        cal_total = (blend_factor * market_total) + ((1 - blend_factor) * pred_total)
        cal_diff = (blend_factor * market_spread) + ((1 - blend_factor) * pred_diff)

        # Recalculate home/away scores based on calibrated total and diff
        cal_home = (cal_total + cal_diff) / 2
        cal_away = cal_total - cal_home

        # Recalculate win probability based on calibrated difference
        cal_win_prob = 1 / (1 + math.exp(-0.15 * cal_diff))

        # --- Update Calibrated Dictionary ---
        calibrated['predicted_home_score'] = round(cal_home, 1)
        calibrated['predicted_away_score'] = round(cal_away, 1)
        calibrated['predicted_point_diff'] = round(cal_diff, 1)
        calibrated['predicted_total_score'] = round(cal_total, 1)
        calibrated['win_probability'] = round(cal_win_prob, 3)
        calibrated['is_calibrated'] = True

        # Adjust confidence bounds if they exist (simple shift)
        if prediction.get('lower_bound') is not None and prediction.get('upper_bound') is not None:
             score_shift = cal_home - prediction['predicted_home_score'] # Use home score shift as example
             calibrated['lower_bound'] = round(prediction['lower_bound'] + score_shift, 1)
             calibrated['upper_bound'] = round(prediction['upper_bound'] + score_shift, 1)

        logger.info(f"Calibrated {prediction.get('game_id')}: H {cal_home:.1f} - A {cal_away:.1f} (WP: {cal_win_prob:.1%})")
        return calibrated

    except Exception as e:
        logger.error(f"Error calibrating prediction for {prediction.get('game_id')}: {e}", exc_info=True)
        calibrated['is_calibrated'] = False
        return calibrated # Return original on error, but mark as failed


# --- Reporting Utility ---

def display_prediction_summary(predictions: List[Dict], odds_dict: Dict = {}):
    """Displays a summary of predictions and betting edges in the console."""
    if not predictions:
        logger.info("No predictions to display.")
        return

    print("\n" + "="*90)
    print(" " * 30 + "NBA PREGAME PREDICTION SUMMARY")
    print("=" * 90)
    print(f"{'DATE':<11} {'MATCHUP':<30} {'PRED SCORE':<12} {'PRED DIFF':<10} {'WIN PROB':<10} {'CONF':<6}")
    print("-"*90)

    predictions_df = pd.DataFrame(predictions).sort_values('game_date')

    for _, game in predictions_df.iterrows():
        date_str = pd.to_datetime(game['game_date']).strftime('%Y-%m-%d')
        matchup = f"{game['home_team']} vs {game['away_team']}"
        pred_score = f"{game['predicted_home_score']:.1f}-{game['predicted_away_score']:.1f}"
        pred_diff = game['predicted_point_diff']
        win_prob_str = f"{game['win_probability']*100:.1f}%"
        conf_str = f"{game.get('prediction_confidence', 0.5)*100:.0f}%" if game.get('prediction_confidence') is not None else "N/A"

        print(f"{date_str:<11} {matchup:<30} {pred_score:<12} {pred_diff:<+10.1f} {win_prob_str:<10} {conf_str:<6}")

        # Display Betting Edge if odds available and calibration occurred
        game_id = str(game.get('game_id'))
        odds_info = game.get('betting_odds') # Get from calibrated prediction if present
        if not odds_info and game_id in odds_dict: # Fallback to original odds dict
             odds_info = odds_dict[game_id]

        if odds_info:
             market_spread, market_total = None, None
             if odds_info.get('spread') and odds_info['spread'].get('home_line') is not None: market_spread = -odds_info['spread']['home_line']
             elif odds_info.get('spread') and odds_info['spread'].get('away_line') is not None: market_spread = odds_info['spread']['away_line']
             if odds_info.get('total'): market_total = odds_info['total'].get('line')

             spread_edge_str, total_edge_str = "N/A", "N/A"
             if market_spread is not None:
                 spread_edge = pred_diff - market_spread
                 spread_edge_str = f"{spread_edge:+.1f}"
             if market_total is not None:
                 total_edge = game['predicted_total_score'] - market_total
                 total_edge_str = f"{total_edge:+.1f}"

             print(f"  -> Betting Edge: Spread={spread_edge_str}, Total={total_edge_str} "
                   f"(vs Market Spread={market_spread:.1f}, Total={market_total:.1f})")
        else:
             print("  -> Betting Edge: Market Odds Unavailable")

    print("="*90)


# --- Main Orchestration Function ---

def run_predictions(
    days_window: int = 2,
    model_path: Path = DEFAULT_ENSEMBLE_MODEL_PATH,
    calibrate_with_odds: bool = True,
    blend_factor: float = 0.7,
    historical_lookback: int = 90
    ) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs the full pre-game prediction pipeline.

    Args:
        days_window: How many days ahead to fetch upcoming games.
        model_path: Path to the saved ensemble model data.
        calibrate_with_odds: Whether to fetch odds and calibrate predictions.
        blend_factor: Blend factor for calibration (0=model, 1=market).
        historical_lookback: Days of historical data to load for features.

    Returns:
        Tuple: (list of calibrated prediction dictionaries, list of raw prediction dictionaries)
    """
    logger.info("--- Starting NBA Pregame Prediction Pipeline ---")

    # --- Setup ---
    engine = get_db_engine()
    supabase_client = get_supabase_client()
    feature_engine = NBAFeatureEngine(debug=False) # Instantiate feature engine

    # --- Load Data ---
    historical_data = load_historical_data(engine, days_lookback=historical_lookback)
    if historical_data.empty:
        logger.error("Failed to load historical data. Cannot proceed.")
        return [], []
    # Pre-calculate team metrics
    team_metrics = feature_engine.create_team_metrics(historical_data)

    upcoming_games_df = fetch_upcoming_games(supabase_client, days_window=days_window)
    if upcoming_games_df.empty:
        logger.warning("No upcoming games found to predict.")
        return [], []

    # --- Load Model ---
    ensemble_data = load_ensemble_model(model_path)
    if not ensemble_data:
        logger.error("Failed to load ensemble model data. Cannot proceed.")
        return [], []

    # --- Generate Raw Predictions ---
    raw_predictions = []
    logger.info(f"Generating predictions for {len(upcoming_games_df)} games...")
    for _, game_row in upcoming_games_df.iterrows():
        pred = make_single_game_prediction(game_row, feature_engine, historical_data, ensemble_data, team_metrics)
        if pred:
            raw_predictions.append(pred)

    if not raw_predictions:
        logger.error("No raw predictions were successfully generated.")
        return [], []

    # --- Fetch Odds & Calibrate (Optional) ---
    calibrated_predictions = raw_predictions # Default to raw if no calibration
    if calibrate_with_odds:
        game_ids_to_predict = [str(p['game_id']) for p in raw_predictions]
        odds_dict = fetch_and_parse_betting_odds(supabase_client, game_ids_to_predict)

        if not odds_dict:
            logger.warning("No betting odds fetched. Predictions will not be calibrated.")
        else:
            logger.info(f"Calibrating {len(raw_predictions)} predictions with odds...")
            calibrated_predictions = []
            for pred in raw_predictions:
                 game_id = str(pred.get('game_id'))
                 odds_info = odds_dict.get(game_id)
                 calibrated_pred = calibrate_prediction_with_odds(pred, odds_info, blend_factor)
                 calibrated_predictions.append(calibrated_pred)
    else:
         logger.info("Skipping odds fetching and calibration.")
         # Add flag indicating no calibration
         for pred in calibrated_predictions:
              pred['is_calibrated'] = False


    # --- Display Summary ---
    display_prediction_summary(calibrated_predictions)

    logger.info("--- NBA Pregame Prediction Pipeline Finished ---")
    return calibrated_predictions, raw_predictions


# --- Script Execution ---
if __name__ == "__main__":
    # Example: Run predictions for the next 2 days and calibrate
    calibrated_preds, raw_preds = run_predictions(
        days_window=2,
        calibrate_with_odds=True,
        blend_factor=0.7
    )

    if calibrated_preds:
        logger.info(f"Successfully generated {len(calibrated_preds)} calibrated predictions.")
        # Optionally save results to CSV or database here
        # pd.DataFrame(calibrated_preds).to_csv("predictions_output.csv", index=False)
    else:
        logger.error("Prediction pipeline failed to produce results.")