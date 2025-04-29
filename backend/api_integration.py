# backend/api_integration.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import logging
from pathlib import Path
import numpy as np

from typing import Optional, List, Dict, Any  # Added missing type imports

# --- Add backend directory to sys.path (Consider using python packages instead) ---
# This approach can sometimes cause issues depending on deployment environment
# SCRIPT_DIR = Path(__file__).resolve().parent
# BACKEND_DIR = SCRIPT_DIR.parent
# PROJECT_ROOT = BACKEND_DIR.parent
# if str(BACKEND_DIR) not in sys.path:
#     sys.path.insert(0, str(BACKEND_DIR))
# --- End Path Setup ---

# --- Refined Project Module Imports ---
try:
    # Import classes/functions needed AT THE TOP LEVEL for instantiation
    from backend.features.legacy.feature_engineering import FeatureEngine    
    from backend.nba_score_prediction.models import QuarterSpecificModelSystem
    from backend.nba_score_prediction.simulation import PredictionUncertaintyEstimator
    from backend.nba_score_prediction.ensemble import EnsembleWeightManager  # Import the weight manager

    # Import classes/functions needed primarily WITHIN ENDPOINTS
    # Use TYPE_CHECKING to potentially avoid circular imports if needed, though often better structure avoids it
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from backend.nba_score_prediction.models import XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor
        from backend.nba_score_prediction.prediction import generate_predictions  # For pre-game static ensemble
        from backend.nba_score_prediction.ensemble import generate_enhanced_predictions  # For live/dynamic ensemble

except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.error(f"CRITICAL ERROR importing project modules: {e}. API cannot start.", exc_info=True)
    raise RuntimeError(f"API failed to start due to missing core modules: {e}") from e

# Define Pydantic models for API requests
class ScorePredictionRequest(BaseModel):
    features: list  # DEPRECATED? This seems like passing raw features, usually avoided

class EnhancedScorePredictionRequest(BaseModel):
    game_data: dict  # Should contain current scores, quarter, time, etc.

class QuarterAnalysisRequest(BaseModel):
    game_state: dict  # e.g., {'current_quarter': 2, 'home_q1': 28, 'away_q1': 27, ...}

class BatchPredictionRequestItem(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_date: str  # Or datetime
    # Include other base fields needed for feature generation if not live
    current_quarter: Optional[int] = 0
    home_score: Optional[int] = 0
    away_score: Optional[int] = 0
    # ... potentially other state fields ...

# Initialize FastAPI app
app = FastAPI(
    title="ScoreGenius API",
    description="API for live sports analytics and predictive modeling.",
    version="1.0.0"
)

# Create a top-level logger instance
logger = logging.getLogger(__name__)

# --- Global State / Component Management ---
app_state: Dict[str, Any] = {}

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Load models and initialize components when the API starts.
    This is more efficient than loading inside each request handler
    and helps mitigate circular import issues related to instantiation order.
    """
    logger.info("API starting up...")

    try:
        feature_generator = FeatureEngine(debug=False)
        app_state["feature_generator"] = feature_generator
        logger.info("NBAFeatureEngine initialized.")

        weight_manager = EnsembleWeightManager(debug=False)
        app_state["weight_manager"] = weight_manager
        logger.info("EnsembleWeightManager initialized.")

        quarter_system = QuarterSpecificModelSystem(
            feature_generator=feature_generator,
            weight_manager=weight_manager,
            debug=False
        )
        try:
            quarter_system.load_models()
            logger.info("QuarterSpecificModelSystem initialized and models loaded.")
        except Exception as q_load_e:
            logger.warning(f"Could not load quarter-specific models during startup: {q_load_e}", exc_info=True)
        app_state["quarter_system"] = quarter_system

        uncertainty_estimator = PredictionUncertaintyEstimator(debug=False)
        app_state["uncertainty_estimator"] = uncertainty_estimator
        logger.info("PredictionUncertaintyEstimator initialized.")

        app_state["main_models_loaded"] = False  # Flag for main model loading
        try:
            logger.info("Placeholder: Main models would be loaded here if needed directly by API.")
        except Exception as main_load_e:
            logger.warning(f"Could not load main prediction models during startup: {main_load_e}")

        logger.info("API startup sequence complete.")

    except Exception as startup_e:
        logger.error(f"FATAL ERROR during API startup: {startup_e}", exc_info=True)
        # Optionally, prevent the API from starting completely if critical
        # raise RuntimeError("API Startup Failed") from startup_e

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    required_components = ["feature_generator", "quarter_system", "uncertainty_estimator", "weight_manager"]
    available = all(comp in app_state for comp in required_components)
    return {"status": "ok" if available else "degraded", "components_loaded": available}

# --- API Endpoints ---

@app.post("/predict/pregame")
async def pregame_prediction(game_ids: Optional[List[str]] = None, days_window: int = 2):
    """Generates pre-game predictions using static ensemble weights."""
    try:
        from backend.nba_score_prediction.prediction import generate_predictions

        final_preds, _ = generate_predictions(
            days_window=days_window,
            calibrate_with_odds=True,
        )
        if game_ids:
            final_preds = [p for p in final_preds if p.get('game_id') in game_ids]

        if not final_preds:
            raise HTTPException(status_code=404, detail="No predictions generated or found for the requested criteria.")

        return {"predictions": final_preds}

    except FileNotFoundError as e:
        logger.error(f"File not found during pregame prediction (check model paths?): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error: Model or data file not found.")
    except Exception as e:
        logger.error(f"Error during pregame prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during pregame prediction: {str(e)}")

@app.post("/predict/live")
async def live_prediction(request: EnhancedScorePredictionRequest):
    """
    Generates live predictions using dynamic ensemble weights, quarter system,
    and uncertainty estimation. Requires game_data with current state.
    """
    feature_generator = app_state.get("feature_generator")
    quarter_system = app_state.get("quarter_system")
    uncertainty_estimator = app_state.get("uncertainty_estimator")
    weight_manager = app_state.get("weight_manager")

    if not all([feature_generator, quarter_system, uncertainty_estimator, weight_manager]):
        raise HTTPException(status_code=503, detail="Core prediction components not available.")

    try:
        try:
            from backend.nba_score_prediction.models import XGBoostScorePredictor
            logger.warning("Loading XGBoost predictor on demand for live prediction.")
            predictor = XGBoostScorePredictor()
            predictor.load_model()
        except Exception as load_e:
            logger.error(f"Failed to load XGBoost model for live prediction: {load_e}")
            raise HTTPException(status_code=503, detail="Main prediction model unavailable.")

        game_data_df = pd.DataFrame([request.game_data])
        features_df = feature_generator.generate_all_features(game_data_df)
        if features_df.empty:
            raise ValueError("Feature generation failed for live prediction input.")

        required_main_features = predictor.feature_names_in_
        if not required_main_features:
            raise ValueError("Main model missing required feature list.")
        missing_live_features = [f for f in required_main_features if f not in features_df.columns]
        if missing_live_features:
            logger.warning(f"Live features missing for main model: {missing_live_features}. Filling with 0.")
            for f in missing_live_features:
                features_df[f] = 0
        X_main_live = features_df[required_main_features]
        if X_main_live.isnull().any().any():
            logger.warning("NaNs found in live main model features. Filling with 0.")
            X_main_live = X_main_live.fillna(0.0)
        main_pred = predictor.predict(X_main_live)
        main_pred_home = main_pred['predicted_home_score'].iloc[0]

        if not quarter_system.models:
            quarter_system.load_models()

        ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
            game_data_dict=request.game_data,
            main_model_prediction=float(main_pred_home)
        )

        current_quarter = int(request.game_data.get('current_quarter', 0))
        hist_dict = None
        lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
            prediction=float(ensemble_pred),
            current_quarter=current_quarter,
            historic_accuracy=hist_dict
        )

        result = {
            "main_model_prediction": float(main_pred_home),
            "quarter_model_prediction": float(breakdown.get('quarter_model_pred', np.nan)),
            "ensemble_prediction": float(ensemble_pred),
            "confidence_pct": float(conf_pct),
            "prediction_interval": {
                "lower_bound": float(lower_b),
                "upper_bound": float(upper_b)
            },
            "weights": breakdown.get('weights'),
            "breakdown": breakdown
        }
        return result

    except FileNotFoundError as e:
        logger.error(f"File not found during live prediction (check model paths?): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error: Model file not found.")
    except ValueError as e:
        logger.warning(f"Value error during live prediction (check input data?): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(e)}")
    except Exception as e:
        logger.error(f"Error during live prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during live prediction: {str(e)}")

@app.post("/quarter/predict")
async def quarter_prediction_endpoint(request: QuarterAnalysisRequest):
    """Predicts scores for remaining quarters based on current game state."""
    quarter_system = app_state.get("quarter_system")
    feature_generator = app_state.get("feature_generator")
    if not quarter_system or not feature_generator:
        raise HTTPException(status_code=503, detail="Quarter system or feature generator not available.")

    try:
        game_state_dict = request.game_state
        current_q = game_state_dict.get('current_quarter')
        if current_q is None:
            raise ValueError("Missing 'current_quarter' in game_state")
        current_q = int(current_q)

        if not quarter_system.models:
            logger.info("Loading quarter models on demand for /quarter/predict")
            quarter_system.load_models()

        remaining_preds = quarter_system.predict_remaining_quarters(
            game_data=game_state_dict,
            current_quarter=current_q
        )

        return {"quarter_predictions": {q: float(score) for q, score in remaining_preds.items()}}

    except ValueError as e:
        logger.warning(f"Value error during quarter prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(e)}")
    except Exception as e:
        logger.error(f"Error predicting quarters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error predicting quarters: {str(e)}")

@app.post("/predict/batch")
async def batch_predictions_endpoint(request: List[BatchPredictionRequestItem]):
    """
    Generates batch predictions. Currently configured for PREGAME style.
    Input should be a list of basic game info.
    """
    try:
        from backend.nba_score_prediction.prediction import generate_predictions

        requested_game_ids = [item.game_id for item in request]
        final_preds, _ = generate_predictions(days_window=7, calibrate_with_odds=True)

        batch_results = [p for p in final_preds if p.get('game_id') in requested_game_ids]

        if not batch_results:
            raise HTTPException(status_code=404, detail="No predictions generated for the requested batch game IDs.")

        return {"predictions": batch_results}

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in batch prediction: {str(e)}")

@app.post("/predict/score")
async def score_prediction_deprecated(request: ScorePredictionRequest):
    raise HTTPException(status_code=410, detail="Endpoint deprecated. Use /predict/pregame or /predict/live.")

@app.get("/quarter/momentum_shifts")
async def momentum_shifts_deprecated():
    raise HTTPException(status_code=410, detail="Endpoint deprecated. Momentum analysis should be integrated elsewhere.")

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')
    logger = logging.getLogger(__name__)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
