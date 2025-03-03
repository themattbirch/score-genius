# backend/routers/analysis_routes.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import pandas as pd

# Import existing endpoints
from backend.ml.llama_inference import generate_recap
from backend.analytics.win_probability import calculate_win_probability

# Import new predictive model endpoints
from models.score_prediction import load_model as load_score_model, predict_final_score
from models.player_projection import load_models as load_player_projection_models, predict_player_performance
from models.nlp_summary import generate_game_summary
from models.ensemble_model import (
    ensemble_predictions,
    ensemble_with_confidence,
    get_context_weights,
    calculate_dynamic_weights
)
from models.quarter_analysis import (
    analyze_quarter_differences,
    train_momentum_models,
    identify_momentum_shifts,
    predict_remaining_quarters
)

# Define Pydantic models for API requests
class ScorePredictionRequest(BaseModel):
    features: list  # e.g., [home_q1, home_q2, home_q3, home_q4, score_ratio, rolling_home_score, rolling_away_score]

class PlayerProjectionRequest(BaseModel):
    player_id: int
    opponent_id: str = None
    method: str = "xgboost"  # or "arima"
    current_live_data: dict = None  # e.g., {"points": 20, "minutes_played": 30}

class DynamicRecommendationRequest(BaseModel):
    model_outputs: dict  # e.g., win_probability, momentum_shift, projected_margin, total_projected_score, quarter, time_remaining, etc.
    player_projection: dict = None  # Optional: mapping of player stats for fantasy recommendations

class QuarterAnalysisRequest(BaseModel):
    current_quarter: int
    quarter_scores: dict  # e.g., {'home_q1': 28, 'home_q2': 27, 'away_q1': 27, 'away_q2': 26, 'home_avg': 28, 'away_avg': 27}
    team_stats: dict      # e.g., {'home_off_rating': 30, 'away_off_rating': 29, 'is_home': True}

router = APIRouter()

# Existing endpoints
@router.post("/recap")
async def generate_game_recap(game_data: dict):
    recap = generate_recap(game_data)
    return JSONResponse(content={"recap": recap})

@router.post("/win-probability")
async def get_win_probability(game_data: dict):
    probability = calculate_win_probability(game_data)
    return JSONResponse(content={"win_probability": probability})

# New predictive endpoints

@router.post("/predict/score")
def score_prediction(request: ScorePredictionRequest):
    model = load_score_model()
    if not model:
        raise HTTPException(status_code=500, detail="Score prediction model not available.")
    try:
        prediction = predict_final_score(model, request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_score": prediction}

@router.post("/predict/player")
def player_projection(request: PlayerProjectionRequest):
    models = load_player_projection_models(request.player_id)
    if not models:
        raise HTTPException(status_code=500, detail="Player projection models not available.")
    try:
        forecasts = predict_player_performance(
            models,
            steps=1,
            method=request.method,
            opponent_id=request.opponent_id,
            current_live_data=request.current_live_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"player_forecasts": forecasts}

@router.post("/recommendations")
def recommendations(request: DynamicRecommendationRequest):
    try:
        recs = generate_game_summary(request.model_outputs, request.player_projection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"recommendations": recs}

@router.get("/quarter/momentum_shifts")
def momentum_shifts():
    data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_games.csv')
    df = pd.read_csv(data_path)
    shifts = identify_momentum_shifts(df)
    return {"momentum_shifts": shifts}

@router.post("/quarter/predict")
def quarter_prediction(request: QuarterAnalysisRequest):
    try:
        preds = predict_remaining_quarters(request.current_quarter, request.quarter_scores, request.team_stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"quarter_predictions": preds}
