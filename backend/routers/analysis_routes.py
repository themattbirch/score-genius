# backend/routers/analysis_routes.py
from fastapi import APIRouter, HTTPException, Depends # Added Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel # Keep BaseModel if needed for request bodies here
import logging

# --- Potentially import shared components via Dependency Injection ---
# This requires defining 'getter' functions elsewhere, often near your main app setup
# from ..dependencies import get_feature_engine # Example dependency function

# --- Import specific analysis functions ---
try:
    from backend.ml.llama_inference import generate_recap
    from backend.analytics.win_probability import calculate_win_probability
    # Import the feature engine class if needed for type hinting Dependencies
    #from backend.nba_score_prediction.feature_engineering import FeatureEngine
    from backend.features.legacy.feature_engineering import FeatureEngine as _L
    
except ImportError as e:
    logging.error(f"Error importing analysis dependencies: {e}", exc_info=True)
    # Handle missing dependencies appropriately
    raise RuntimeError(f"Analysis routes failed to load dependencies: {e}") from e

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/analysis", # Add a prefix for organization
    tags=["Analysis"]   # Tag endpoints in OpenAPI docs
)

# Define request/response models IF NEEDED by these specific endpoints
class RecapRequest(BaseModel):
     game_data: dict # Or a more specific model

class WinProbRequest(BaseModel):
     game_data: dict # Or a more specific model


# --- Analysis Specific Endpoints ---

@router.post("/recap", summary="Generate Game Recap")
async def generate_game_recap_endpoint(request: RecapRequest):
    """Generates a textual recap for a given game's data."""
    # If generate_recap needs shared components, inject them:
    # feature_engine: FeatureEngine = Depends(get_feature_engine)
    try:
        # Pass necessary data or components to the analysis function
        recap = generate_recap(request.game_data) # Assuming it takes the dict
        return JSONResponse(content={"recap": recap})
    except Exception as e:
        logger.error(f"Error generating recap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating recap: {str(e)}")


@router.post("/win-probability", summary="Calculate Win Probability")
async def get_win_probability_endpoint(request: WinProbRequest):
    """Calculates win probability based on game data/state."""
    # If calculate_win_probability needs shared components, inject them
    try:
        # Pass necessary data or components
        probability = calculate_win_probability(request.game_data) # Assuming it takes the dict
        # Basic validation
        if probability is None or not (0 <= probability <= 1):
             logger.warning(f"Win probability calculation returned invalid value: {probability}")
             raise HTTPException(status_code=500, detail="Win probability calculation failed.")
        return JSONResponse(content={"win_probability": float(probability)})
    except Exception as e:
        logger.error(f"Error calculating win probability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculating win probability: {str(e)}")
