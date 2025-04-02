# File: backend/src/scripts/model_inference.py
# (Your provided code is sound for its intended purpose, assuming
# the nba_precomputed_features table is correctly populated externally)

import os
import sys
import joblib
import pandas as pd
import numpy as np
# import datetime # datetime used via pd.Timestamp or imported below
import logging
import pytz # Needed for save_predictions_to_db timestamp
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime # Explicit import for timestamp

# --- Add backend directory to sys.path ---
# Assumes this script is in backend/src/scripts/
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
# --- End Path Setup ---

# --- Project Imports ---
try:
    from nba_score_prediction.models import ( # Corrected path
        XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor
    )
    from caching.supabase_client import supabase # Corrected path
    import config # Corrected path
    PROJECT_MODULES_IMPORTED = True
    # Import Supabase types if needed within try block
    try: from supabase import Client
    except ImportError: Client = None; logging.warning("Supabase client library not found.")

except ImportError as e:
    logging.error(f"Error importing project modules in model_inference: {e}", exc_info=True)
    PROJECT_MODULES_IMPORTED = False
    # Define dummy classes if imports fail
    class BaseScorePredictorPlaceholder:
         def load_model(self, *args, **kwargs): raise NotImplementedError("Dummy model used")
         def predict(self, *args, **kwargs): raise NotImplementedError("Dummy model used")
         feature_names_in_ = None # Add placeholder attribute
    XGBoostScorePredictor = BaseScorePredictorPlaceholder
    RandomForestScorePredictor = BaseScorePredictorPlaceholder
    RidgeScorePredictor = BaseScorePredictorPlaceholder
    supabase = None
    config = None
    Client = None # Define Client as None if import fails

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"), # Consider standardized logging location
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Get model directory from config or use default path structure
MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', Path(backend_dir) / 'models' / 'saved'))
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure exists

# Static ensemble weights
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,
    "random_forest": 0.30,
    "ridge": 0.20
}

# --- Functions ---

def load_ensemble_components(model_dir: Path = MODELS_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Loads the latest XGBoost, RandomForest, and Ridge model components."""
    # (Implementation seems correct - keeping as provided)
    models = {}
    component_classes = { "xgboost": XGBoostScorePredictor, "random_forest": RandomForestScorePredictor, "ridge": RidgeScorePredictor }
    loaded_feature_names = None; success = True
    for name, PredictorClass in component_classes.items():
        # Skip if class is dummy (due to import error)
        if PredictorClass.__name__ == 'BaseScorePredictorPlaceholder': success = False; continue
        # Skip XGBoost if library not available
        if name == 'xgboost' and not XGBOOST_AVAILABLE: logger.warning("XGBoost unavailable, skipping load."); continue
        try:
            logger.info(f"Loading latest '{name}' model component from {model_dir}...")
            predictor = PredictorClass(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            predictor.load_model()
            models[name] = predictor
            logger.info(f"Loaded {name} component trained {predictor.training_timestamp or 'unknown time'}.")
            if loaded_feature_names is None and predictor.feature_names_in_: loaded_feature_names = predictor.feature_names_in_; logger.info(f"Using feature list from '{name}' model ({len(loaded_feature_names)} features).")
        except FileNotFoundError: logger.error(f"No trained '{name}' model file found in {model_dir}."); success = False
        except Exception as e: logger.error(f"Error loading '{name}': {e}", exc_info=True); success = False
    if not success or not loaded_feature_names: logger.error("Failed to load all required components or get feature list."); return None, None
    # Feature consistency check (optional)
    for name, predictor in models.items():
        if predictor.feature_names_in_ != loaded_feature_names: logger.warning(f"Feature mismatch! Model '{name}' differs."); break # Use first list
    logger.info("All loadable model components loaded successfully.")
    return models, loaded_feature_names


def fetch_precomputed_features() -> Optional[pd.DataFrame]:
    """ Fetches the latest precomputed features from Supabase table 'nba_precomputed_features'. """
    # (Implementation seems correct - keeping as provided)
    if not supabase: logger.error("Supabase client unavailable."); return None
    target_table = "nba_precomputed_features"
    try:
        logger.info(f"Fetching precomputed features from Supabase table '{target_table}'...")
        response = supabase.table(target_table).select("*").execute()
        if not hasattr(response, 'data'): logger.error("Invalid response fetching features."); return None
        data = response.data
        if data:
            df = pd.DataFrame(data); logger.info(f"Fetched {len(df)} precomputed feature records.")
            if 'game_id' not in df.columns: logger.error("Features missing 'game_id'."); return None
            df['game_id'] = df['game_id'].astype(str)
            return df
        else: logger.warning("No precomputed features found."); return pd.DataFrame()
    except Exception as e: logger.error(f"Error fetching precomputed features: {e}", exc_info=True); return pd.DataFrame()


def generate_ensemble_predictions(models: Dict[str, Any], features_df: pd.DataFrame, required_features: List[str]) -> Optional[pd.DataFrame]:
    """ Generates ensemble predictions using static weights, requires precomputed features. """
    # (Implementation seems correct - keeping as provided)
    if features_df.empty: logger.warning("Empty features DataFrame received."); return None
    logger.info(f"Generating ensemble predictions for {len(features_df)} games...")
    missing_features = [f for f in required_features if f not in features_df.columns]
    if missing_features: logger.error(f"Precomputed features missing required columns: {missing_features}. Cannot predict."); return None
    extra_features = [f for f in features_df.columns if f not in required_features and f != 'game_id'];
    if extra_features: logger.warning(f"Precomputed features have extra columns: {extra_features[:5]}...")
    try: X_predict = features_df[required_features].copy().astype(float)
    except KeyError: logger.error("Failed selecting required features (column mismatch?)."); return None
    except Exception as e: logger.error(f"Error preparing feature matrix: {e}", exc_info=True); return None
    logger.debug(f"Feature shape for prediction: {X_predict.shape}")
    component_preds_df: Dict[str, pd.DataFrame] = {}
    for name, predictor in models.items():
        try:
            preds_df = predictor.predict(X_predict)
            if preds_df is not None and not preds_df.empty: component_preds_df[name] = preds_df; logger.debug(f"Got preds from {name}.")
            else: logger.warning(f"Predictor {name} returned None/empty. Skipping.")
        except Exception as e: logger.error(f"Error predicting with {name}: {e}", exc_info=True)
    if not component_preds_df: logger.error("No component models produced predictions."); return None
    ensemble_home_score = np.zeros(len(X_predict)); ensemble_away_score = np.zeros(len(X_predict)); total_weight = 0.0
    for name, weight in ENSEMBLE_WEIGHTS.items():
        if name in component_preds_df and weight > 0:
            preds = component_preds_df[name]; ensemble_home_score += preds['predicted_home_score'].values * weight; ensemble_away_score += preds['predicted_away_score'].values * weight; total_weight += weight; logger.debug(f"Added {name} (weight {weight}).")
    if total_weight < 1e-6: logger.error("Total weight is zero."); return None
    elif abs(total_weight - 1.0) > 1e-6: logger.warning(f"Total weight {total_weight:.3f} != 1.0. Normalizing."); ensemble_home_score /= total_weight; ensemble_away_score /= total_weight
    results_df = pd.DataFrame({'game_id': features_df['game_id'], 'predicted_home_score': np.round(ensemble_home_score, 1), 'predicted_away_score': np.round(ensemble_away_score, 1), 'predicted_point_diff': np.round(ensemble_home_score - ensemble_away_score, 1), 'predicted_total_score': np.round(ensemble_home_score + ensemble_away_score, 1)}, index=features_df.index)
    logger.info(f"Generated ensemble predictions for {len(results_df)} games.")
    return results_df


def save_predictions_to_db(predictions_df: pd.DataFrame):
    """ Saves predictions to Supabase table 'nba_score_predictions'. """
    # (Implementation seems correct - keeping as provided)
    if not supabase: logger.error("Supabase client unavailable."); return
    if predictions_df is None or predictions_df.empty: logger.warning("No predictions to save."); return
    logger.info(f"Saving {len(predictions_df)} predictions to Supabase table 'nba_score_predictions'...")
    try:
        records_to_save = predictions_df[['game_id', 'predicted_home_score', 'predicted_away_score']].copy()
        records_to_save['prediction_timestamp'] = datetime.now(pytz.utc).isoformat()
        records = records_to_save.to_dict('records')
        response = supabase.table("nba_score_predictions").upsert(records, on_conflict="game_id").execute()
        logger.info(f"Upsert request for {len(records)} predictions sent to Supabase.")
    except Exception as e: logger.error(f"Error saving predictions to Supabase: {e}", exc_info=True)


def run_model_inference():
    """ Main function to run model inference on precomputed features. """
    logger.info("=" * 50); logger.info("RUNNING MODEL INFERENCE SCRIPT"); logger.info("=" * 50)
    if not PROJECT_MODULES_IMPORTED: logger.critical("Core modules not imported. Aborting."); return
    if not supabase: logger.critical("Supabase client not available. Aborting."); return

    try:
        models, required_features = load_ensemble_components()
        if not models or not required_features: logger.error("Failed load models/features. Aborting."); return
        features_df = fetch_precomputed_features()
        if features_df is None or features_df.empty: logger.warning("No precomputed features. Skipping cycle."); return
        predictions_df = generate_ensemble_predictions(models, features_df, required_features)
        if predictions_df is None or predictions_df.empty: logger.error("Prediction generation failed."); return
        logger.info("Generated Predictions (Head):"); logger.info("\n" + predictions_df.head().to_string())
        save_predictions_to_db(predictions_df)
        logger.info("Model inference cycle complete.")
    except Exception as e: logger.error(f"Unhandled error during model inference: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Running model inference directly...")
    run_model_inference()