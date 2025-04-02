# File: backend/src/scripts/model_inference.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
import datetime
import logging
import pytz
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# --- Add backend directory to sys.path ---
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
# --- End Path Setup ---

# --- Project Imports ---
try:
    # Import model classes using path relative to backend/
    from nba_score_prediction.models import (
        XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor
    )
    # Import supabase client using path relative to backend/
    from caching.supabase_client import supabase
    # Import config directly as backend/ is in path
    import config
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Error importing project modules in model_inference: {e}", exc_info=True)
    PROJECT_MODULES_IMPORTED = False
    # Define dummy classes if imports fail
    class BaseScorePredictorPlaceholder:
         def load_model(self, *args, **kwargs): raise NotImplementedError("Dummy model used")
         def predict(self, *args, **kwargs): raise NotImplementedError("Dummy model used")
    XGBoostScorePredictor = BaseScorePredictorPlaceholder
    RandomForestScorePredictor = BaseScorePredictorPlaceholder
    RidgeScorePredictor = BaseScorePredictorPlaceholder
    supabase = None
    config = None


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"), # Consider logging to a more central location
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Get model directory from config or use default path structure
MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / '..' / 'models' / 'saved')) # Fallback path added

# Static ensemble weights
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,
    "random_forest": 0.30,
    "ridge": 0.20
}

# --- Functions ---

def load_ensemble_components(model_dir: Path = MODELS_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Loads the latest XGBoost, RandomForest, and Ridge model components."""
    models = {}
    component_classes = {
        "xgboost": XGBoostScorePredictor,
        "random_forest": RandomForestScorePredictor,
        "ridge": RidgeScorePredictor
    }
    loaded_feature_names = None
    success = True

    for name, PredictorClass in component_classes.items():
        try:
            # Ensure PredictorClass is not the placeholder if modules failed import
            if PredictorClass.__name__ == 'BaseScorePredictorPlaceholder':
                 logger.error(f"Cannot load '{name}': Required model module was not imported.")
                 success = False; continue

            logger.info(f"Loading latest '{name}' model component...")
            # Instantiate with specific model name convention expected by load_model
            predictor = PredictorClass(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            predictor.load_model() # Loads latest by convention
            models[name] = predictor
            logger.info(f"Loaded {name} component trained {predictor.training_timestamp or 'unknown time'}.")
            # Store feature names from the first successfully loaded model
            if loaded_feature_names is None and predictor.feature_names_in_:
                loaded_feature_names = predictor.feature_names_in_
                logger.info(f"Using feature list from '{name}' model ({len(loaded_feature_names)} features).")
        except FileNotFoundError:
             logger.error(f"No trained model file found for '{name}' in {model_dir}. Cannot generate predictions.")
             success = False
        except Exception as e:
            logger.error(f"Error loading model component '{name}': {e}", exc_info=True)
            success = False

    if not success or len(models) != len(component_classes):
         logger.error("Failed to load all required model components.")
         return None, None

    # Optional: Verify feature consistency across loaded models
    for name, predictor in models.items():
         if predictor.feature_names_in_ != loaded_feature_names:
              logger.warning(f"Feature mismatch! Model '{name}' feature list differs from '{list(models.keys())[0]}'. Using first list.")
              # Implement handling if needed, e.g., error out or try to reconcile

    logger.info("All model components loaded successfully.")
    return models, loaded_feature_names




def fetch_precomputed_features() -> Optional[pd.DataFrame]:
    """ Fetches the latest precomputed features from Supabase. """
    if not supabase:
         logger.error("Supabase client unavailable. Cannot fetch features.")
         return None
    try:
        logger.info("Fetching precomputed features from Supabase table 'nba_precomputed_features'...")
        # Select all columns for now, or specify if only a subset is needed
        response = supabase.table("nba_precomputed_features").select("*").execute()

        if not hasattr(response, 'data'):
             logger.error("Invalid response from Supabase fetching features.")
             return None

        data = response.data
        if data:
            df = pd.DataFrame(data)
            # Basic cleaning - ensure game_id exists, handle dates if needed
            if 'game_id' not in df.columns:
                 logger.error("Fetched features missing 'game_id' column.")
                 return None
            df['game_id'] = df['game_id'].astype(str)
            # Add any other necessary type conversions here
            logger.info(f"Fetched {len(df)} precomputed feature records.")
            return df
        else:
            logger.warning("No precomputed features found in database.")
            return pd.DataFrame() # Return empty DataFrame instead of None
    except Exception as e:
        logger.error(f"Error fetching precomputed features: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

def generate_ensemble_predictions(
    models: Dict[str, Any],
    features_df: pd.DataFrame,
    required_features: List[str]
    ) -> Optional[pd.DataFrame]:

    if features_df.empty:
         logger.warning("Received empty features DataFrame, cannot generate predictions.")
         return None

    logger.info(f"Generating ensemble predictions for {len(features_df)} games...")

    # --- Add this logging ---
    logger.info(f"DEBUG: Required features count: {len(required_features)}")
    logger.info(f"DEBUG: Available features count from DB: {len(features_df.columns)}")
    logger.info(f"DEBUG: Required features (model): {sorted(required_features)}")
    logger.info(f"DEBUG: Available features (DB): {sorted(list(features_df.columns))}")
    # --- End logging ---

    # Validate and select required features
    missing_features = [f for f in required_features if f not in features_df.columns]
    if missing_features:
        # Log only a subset if the list is huge
        log_limit = 50
        logger.error(f"Precomputed features missing {len(missing_features)} required columns. Examples: {missing_features[:log_limit]}{'...' if len(missing_features) > log_limit else ''}. Cannot predict.")
        # You might want to log the full list to a file if needed for detailed comparison
        # with open("missing_features.log", "w") as f:
        #     f.write("\n".join(missing_features))
        return None

    extra_features = [f for f in features_df.columns if f not in required_features and f != 'game_id'] # Keep game_id
    if extra_features:
         logger.warning(f"Precomputed features have extra columns not used by models: {extra_features[:5]}...")

    # Ensure features are in the correct order and only required ones are used
    try:
        X_predict = features_df[required_features].copy()
        # Ensure numeric types (models expect float internally after scaler)
        X_predict = X_predict.astype(float)
    except KeyError:
         logger.error("Failed to select required features from DataFrame. Column names mismatch?")
         return None
    except Exception as e:
         logger.error(f"Error preparing feature matrix for prediction: {e}", exc_info=True)
         return None

    logger.debug(f"Feature shape for prediction: {X_predict.shape}")

    # Store individual model predictions (DataFrames with home/away scores)
    component_preds_df: Dict[str, pd.DataFrame] = {}

    for name, predictor in models.items():
        try:
            preds_df = predictor.predict(X_predict) # Returns DataFrame with home/away
            if preds_df is not None and not preds_df.empty:
                 component_preds_df[name] = preds_df
                 logger.debug(f"Generated predictions with {name} component.")
            else:
                 logger.warning(f"Predictor {name} returned None or empty DataFrame. Skipping component.")
        except Exception as e:
             logger.error(f"Error predicting with component {name}: {e}", exc_info=True)
             # Decide how to handle component failure: skip or use fallback? Skipping for now.

    if not component_preds_df:
         logger.error("No component models produced predictions.")
         return None

    # Combine predictions using static weights
    ensemble_home_score = np.zeros(len(X_predict))
    ensemble_away_score = np.zeros(len(X_predict))
    total_weight = 0.0

    for name, weight in ENSEMBLE_WEIGHTS.items():
        if name in component_preds_df and weight > 0:
            preds = component_preds_df[name]
            ensemble_home_score += preds['predicted_home_score'].values * weight
            ensemble_away_score += preds['predicted_away_score'].values * weight
            total_weight += weight
            logger.debug(f"Added {name} (weight {weight}) to ensemble.")

    if total_weight < 1e-6: # Should not happen if weights are correct and models predict
         logger.error("Total weight is zero, cannot calculate ensemble average.")
         return None
    elif abs(total_weight - 1.0) > 1e-6:
         logger.warning(f"Total weight used ({total_weight:.3f}) is not 1.0. Normalizing scores.")
         # Normalize scores if weights don't sum to 1 (e.g., if a component failed)
         ensemble_home_score /= total_weight
         ensemble_away_score /= total_weight

    # Create result DataFrame
    results_df = pd.DataFrame({
        'game_id': features_df['game_id'], # Assumes game_id was kept or is index
        'predicted_home_score': np.round(ensemble_home_score, 1),
        'predicted_away_score': np.round(ensemble_away_score, 1),
        'predicted_point_diff': np.round(ensemble_home_score - ensemble_away_score, 1),
        'predicted_total_score': np.round(ensemble_home_score + ensemble_away_score, 1)
    }, index=features_df.index) # Maintain original index if needed

    logger.info(f"Generated ensemble predictions for {len(results_df)} games.")
    return results_df


def save_predictions_to_db(predictions_df: pd.DataFrame):
    """ Saves predictions to Supabase table 'nba_score_predictions'. """
    if not supabase:
        logger.error("Supabase client unavailable. Cannot save predictions.")
        return
    if predictions_df is None or predictions_df.empty:
        logger.warning("No predictions to save.")
        return

    logger.info(f"Saving {len(predictions_df)} predictions to Supabase table 'nba_score_predictions'...")
    try:
        # Prepare records for upsert
        records_to_save = predictions_df[['game_id', 'predicted_home_score', 'predicted_away_score']].copy()
        records_to_save['prediction_timestamp'] = datetime.now(pytz.utc).isoformat() # Add timestamp

        records = records_to_save.to_dict('records')

        # Upsert based on game_id (assuming game_id is unique constraint or PK)
        response = supabase.table("nba_score_predictions").upsert(
            records,
            on_conflict="game_id" # Specify the column for conflict resolution
            ).execute()

        # Check response for errors (PostgREST API v1+)
        # Note: As of early 2024, upsert might not return detailed error objects directly in response.data/error
        # It's safer to assume success unless an exception was raised.
        # if hasattr(response, 'error') and response.error:
        #    logger.error(f"Error saving predictions to database: {response.error}")
        # elif hasattr(response, 'data') and response.data:
        #     logger.info(f"Successfully saved/upserted {len(response.data)} predictions to database")
        # else:
        #      logger.warning("Prediction save request sent, but response format unclear or empty.")
        logger.info(f"Upsert request for {len(records)} predictions sent to Supabase.")

    except Exception as e:
        logger.error(f"Error saving predictions to Supabase: {e}", exc_info=True)


def run_model_inference():
    """ Main function to run model inference on precomputed features. """
    logger.info("=" * 50)
    logger.info("RUNNING MODEL INFERENCE SCRIPT")
    logger.info("=" * 50)

    if not PROJECT_MODULES_IMPORTED:
        logger.critical("Core project modules not imported. Aborting inference.")
        return
    if not supabase:
        logger.critical("Supabase client not available. Aborting inference.")
        return

    try:
        # Step 1: Load the trained model components
        models, required_features = load_ensemble_components()
        if not models or not required_features:
            logger.error("Failed to load necessary model components or feature list. Aborting.")
            return

        # Step 2: Fetch the latest precomputed features
        features_df = fetch_precomputed_features()
        if features_df is None or features_df.empty:
            # Changed check to handle empty DataFrame return value
            logger.warning("No precomputed feature data available. Skipping inference cycle.")
            return

        # Step 3: Generate predictions
        predictions_df = generate_ensemble_predictions(models, features_df, required_features)
        if predictions_df is None or predictions_df.empty:
             logger.error("Prediction generation failed.")
             return

        # Step 4: Log and persist the predictions
        logger.info("Generated Predictions (Head):")
        logger.info("\n" + predictions_df.head().to_string())
        save_predictions_to_db(predictions_df)

        logger.info("Model inference cycle complete.")

    except Exception as e:
        logger.error(f"Unhandled error during model inference: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Running model inference directly...")
    run_model_inference()