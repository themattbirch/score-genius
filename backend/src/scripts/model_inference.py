# File: backend/src/scripts/model_inference.py

import os
import joblib
import pandas as pd
import numpy as np
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from models.score_prediction import load_training_data, preprocess_data, train_model, save_model
from caching.supabase_client import supabase
from config import MODEL_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_inference")

# Global cache for model object
_MODEL_CACHE: Dict[str, Any] = {}

def get_model(force_reload: bool = False) -> Any:
    """
    Get the model from cache or load it from disk.
    If model doesn't exist or force_reload is True, train a new one.
    
    Args:
        force_reload: If True, ignore cache and reload model from disk
        
    Returns:
        The loaded model
    """
    global _MODEL_CACHE

    if "pregame_model" in _MODEL_CACHE and not force_reload:
        logger.info("Using cached model")
        return _MODEL_CACHE["pregame_model"]

    if os.path.exists(MODEL_PATH) and not force_reload:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Loaded model from: {MODEL_PATH}")
            logger.info(f"Model type: {model.__class__.__name__}")
            _MODEL_CACHE["pregame_model"] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model from disk: {e}")
            logger.info("Proceeding to retrain the model...")

    logger.info("Training new model...")
    model = retrain_model()
    _MODEL_CACHE["pregame_model"] = model
    return model

def fetch_new_features():
    """
    Fetches the latest engineered features from Supabase.
    Assumes the features are stored in a table called "nba_precomputed_features".
    
    Returns:
        DataFrame containing the features or None if no data is found
    """
    try:
        logger.info("Fetching new features from Supabase...")
        response = supabase.table("nba_precomputed_features").select("*").execute()
        data = response.data
        
        if data:
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} new feature records")
            return df
        else:
            logger.warning("No new features found in database")
            return None
    except Exception as e:
        logger.error(f"Error fetching features: {str(e)}")
        return None

def generate_predictions(model, features_df):
    """
    Generate predictions using model features extracted from the model itself
    """
    # Extract features from model instead of hardcoding
    if isinstance(model, dict) and 'features' in model:
        expected_features = model.get('features', [])
        print(f"Using {len(expected_features)} features from model definition")
    else:
        # Fallback for older models
        print("No features defined in model, checking component models...")
        if isinstance(model, dict) and 'models' in model:
            # For ensemble models, try to extract features from first component
            first_model_name = next(iter(model['models']))
            first_model = model['models'][first_model_name]
            if hasattr(first_model, 'feature_names_in_'):
                expected_features = first_model.feature_names_in_.tolist()
                print(f"Extracted {len(expected_features)} features from {first_model_name}")
            else:
                # Last resort fallback
                print("WARNING: Could not extract features from model, using features from data")
                expected_features = features_df.columns.tolist()
    
    print(f"Expected features: {expected_features}")
    
    # Check feature overlap
    missing_features = [f for f in expected_features if f not in features_df.columns]
    extra_features = [f for f in features_df.columns if f not in expected_features]
    
    if missing_features:
        print(f"WARNING: Missing {len(missing_features)} required features: {missing_features[:5]}...")
        # Add missing features with zeros
        for feature in missing_features:
            features_df[feature] = 0
    
    if extra_features:
        print(f"NOTE: {len(extra_features)} extra features not used by model: {extra_features[:5]}...")
    
    # Ensure the features are in the same order as during training
    X_features = features_df[expected_features].copy()
    X_features = X_features.astype(float)
        
    logger.info(f"Feature shape for prediction: {X_features.shape}")
    
    # Handle different model types
    if isinstance(model, dict) and 'models' in model:
        # This is an ensemble model
        models = model.get('models', {})
        weights = model.get('weights', {})
        
        logger.info(f"Making predictions with ensemble model ({len(models)} components)")
        
        # Make predictions with each component
        component_predictions = {}
        for name, component_model in models.items():
            if name in weights and weights[name] > 0:
                try:
                    component_predictions[name] = component_model.predict(X_features)
                    logger.info(f"Generated predictions with {name} component")
                except Exception as e:
                    logger.error(f"Error predicting with {name} component: {str(e)}")
        
        # Combine predictions according to weights
        predictions = np.zeros(len(X_features))
        total_weight = 0
        
        for name, weight in weights.items():
            if name in component_predictions:
                predictions += weight * component_predictions[name]
                total_weight += weight
        
        if total_weight > 0:
            predictions /= total_weight
            
        return predictions
    else:
        # This is a single model
        logger.info(f"Making predictions with single model")
        return model.predict(X_features)

def save_predictions_to_db(features_df, predictions):
    """
    Save predictions to the database for historical tracking.
    
    Args:
        features_df: DataFrame with features and game IDs
        predictions: Array of predictions
    """
    try:
        # Create a dataframe with predictions
        prediction_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        pred_df = pd.DataFrame({
            'prediction_date': prediction_date,
            'unique_game_id': features_df['unique_game_id'] if 'unique_game_id' in features_df.columns else None,
            'predicted_home_score': predictions
        })
        
        # Also save to local CSV for backup
        csv_path = f"predictions_{prediction_date}.csv"
        pred_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Save to Supabase
        records = pred_df.to_dict('records')
        response = supabase.table("nba_score_predictions").upsert(records).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error saving predictions to database: {response.error}")
        else:
            logger.info(f"Successfully saved {len(records)} predictions to database")
            
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")

def evaluate_ensemble_weights(model, recent_games_df):
    """
    Evaluate recent performance of each component model and potentially adjust weights.
    
    Args:
        model: The ensemble model
        recent_games_df: DataFrame with recent game data including actual outcomes
        
    Returns:
        Updated model with potentially adjusted weights
    """
    # Only proceed if we have an ensemble model
    if not isinstance(model, dict) or 'models' not in model:
        return model
        
    try:
        if 'actual_home_score' not in recent_games_df.columns:
            logger.warning("Cannot evaluate model components: missing actual scores")
            return model
            
        models = model.get('models', {})
        current_weights = model.get('weights', {})
        features = model.get('features', [])
        
        # Prepare features for evaluation
        missing_features = [f for f in features if f not in recent_games_df.columns]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features for evaluation")
            for feature in missing_features:
                recent_games_df[feature] = 0
                
        X_eval = recent_games_df[features].copy()
        y_eval = recent_games_df['actual_home_score']
        
        # Evaluate each component
        component_scores = {}
        for name, component_model in models.items():
            try:
                component_preds = component_model.predict(X_eval)
                mse = np.mean((component_preds - y_eval) ** 2)
                component_scores[name] = 1.0 / (mse + 1e-10)  # Higher score for lower MSE
            except Exception as e:
                logger.error(f"Error evaluating {name} component: {str(e)}")
                component_scores[name] = 0
                
        # Normalize scores to get new weights
        total_score = sum(component_scores.values())
        if total_score > 0:
            new_weights = {name: score / total_score for name, score in component_scores.items()}
            
            # Log weight changes
            logger.info("Ensemble weight evaluation:")
            for name in models.keys():
                old_weight = current_weights.get(name, 0)
                new_weight = new_weights.get(name, 0)
                logger.info(f"  - {name}: {old_weight:.2f} -> {new_weight:.2f}")
                
            # Update model weights (alternative: blend with current weights for stability)
            blend_factor = 0.8  # 80% existing weights, 20% new weights
            blended_weights = {
                name: (blend_factor * current_weights.get(name, 0) + 
                      (1 - blend_factor) * new_weights.get(name, 0))
                for name in models.keys()
            }
            
            # Normalize blended weights
            total_blended = sum(blended_weights.values())
            if total_blended > 0:
                blended_weights = {name: weight / total_blended for name, weight in blended_weights.items()}
                
            # Update model
            model['weights'] = blended_weights
            logger.info("Updated ensemble weights based on recent performance")
            
        return model
    
    except Exception as e:
        logger.error(f"Error evaluating ensemble weights: {str(e)}")
        return model

def retrain_model():
    """
    Retrains the model using the entire updated historical dataset.
    
    Returns:
        The newly trained model
    """
    logger.info("Retraining the model with updated historical data...")
    try:
        df = load_training_data()
        X, y = preprocess_data(df)
        model = train_model(X, y)
        save_model(model)
        logger.info("Model retraining complete")
        
        # Log model type
        if isinstance(model, dict) and 'models' in model:
            components = model.get('models', {})
            weights = model.get('weights', {})
            logger.info(f"Trained ensemble model with {len(components)} components:")
            
            for name, component in components.items():
                weight = weights.get(name, 0)
                logger.info(f"  - {name} ({component.__class__.__name__}): {weight*100:.1f}% weight")
        else:
            logger.info(f"Trained single model: {model.__class__.__name__}")
            
        return model
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise

def run_model_inference(force_retrain=False):
    """
    Runs model inference and optionally retrains.
    
    Args:
        force_retrain: If True, force model retraining regardless of existing model
    """
    logger.info("=" * 50)
    logger.info("RUNNING MODEL INFERENCE")
    logger.info("=" * 50)

    try:
        # Step 1: Load new engineered features from Supabase
        new_features_df = fetch_new_features()
        if new_features_df is None or new_features_df.empty:
            logger.warning("No new feature data available. Skipping inference.")
            return

        # Step 2: Load the model (from cache or disk or train new)
        model = get_model(force_reload=force_retrain)

        # Step 3: Generate predictions on new features
        predictions = generate_predictions(model, new_features_df)
        new_features_df['predicted_home_score'] = predictions

        # Step 4: Log and persist the predictions
        if 'unique_game_id' in new_features_df.columns:
            logger.info("Predictions for new games:")
            prediction_df = new_features_df[['unique_game_id', 'predicted_home_score']]
            logger.info("\n" + prediction_df.head().to_string())
            
            # Save predictions to database
            save_predictions_to_db(new_features_df, predictions)
        else:
            logger.info("Predictions:")
            logger.info("\n" + new_features_df[['predicted_home_score']].head().to_string())

        # Step 5: Only retrain if specifically requested
        if force_retrain:
            logger.info("Forcing model retraining...")
            updated_model = retrain_model()
            _MODEL_CACHE["pregame_model"] = updated_model
        
        logger.info("Model inference complete")
        
    except Exception as e:
        logger.error(f"Error in model inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # By default, don't retrain the model every run
    run_model_inference(force_retrain=False)