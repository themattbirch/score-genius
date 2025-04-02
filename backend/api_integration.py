# backend/api_integration.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd

# Import modules from our models directory
from models.score_prediction import load_model as load_score_model, predict_final_score
from models.player_projection import load_models as load_player_projection_models, predict_player_performance
from models.nlp_summary import generate_game_summary

# Import the new features and ensemble models from features.py
from predict_score.feature_engineering import (
    NBAFeatureGenerator, 
    QuarterSpecificModelSystem, 
    PredictionUncertaintyEstimator,
    generate_enhanced_predictions,
    dynamic_ensemble_predictions
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

class EnhancedScorePredictionRequest(BaseModel):
    game_data: dict  # Complete game data with all features needed

# Initialize FastAPI app
app = FastAPI(
    title="ScoreGenius API",
    description="API for live sports analytics and predictive modeling.",
    version="1.0.0"
)

# Initialize core components
feature_generator = NBAFeatureGenerator(debug=False)
quarter_system = QuarterSpecificModelSystem(feature_generator)
uncertainty_estimator = PredictionUncertaintyEstimator()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Endpoint for score prediction
@app.post("/predict/score")
def score_prediction(request: ScorePredictionRequest):
    model = load_score_model()
    if not model:
        raise HTTPException(status_code=500, detail="Score prediction model not available.")
    try:
        prediction = predict_final_score(model, request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_score": prediction}

# Enhanced endpoint for score prediction with quarter-specific models
@app.post("/predict/score/enhanced")
def enhanced_score_prediction(request: EnhancedScorePredictionRequest):
    try:
        # Load the main model
        main_model = load_score_model()
        if not main_model:
            raise HTTPException(status_code=500, detail="Main prediction model not available.")
        
        # Load quarter-specific models (if not already loaded)
        if not quarter_system.models:
            quarter_system.load_models()
        
        # Create DataFrame from request data
        game_data = pd.DataFrame([request.game_data])
        
        # Generate features
        features_df = feature_generator.generate_all_features(game_data)
        
        # Get main model prediction
        expected_features = feature_generator.get_expected_features(
            enhanced=hasattr(main_model, 'feature_importances_') and len(main_model.feature_importances_) > 8
        )
        main_pred = main_model.predict(features_df[expected_features])[0]
        
        # Get ensemble prediction
        ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(features_df.iloc[0], main_pred)
        
        # Calculate uncertainty bounds
        current_quarter = int(features_df.iloc[0].get('current_quarter', 0))
        score_margin = abs(float(features_df.iloc[0].get('score_differential', 0)))
        lower_bound, upper_bound, interval_width = uncertainty_estimator.calculate_prediction_interval(
            ensemble_pred, current_quarter, score_margin
        )
        
        result = {
            "main_model_prediction": float(main_pred),
            "quarter_model_prediction": float(breakdown['quarter_model']),
            "ensemble_prediction": float(ensemble_pred),
            "confidence": float(confidence),
            "prediction_interval": {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "width": float(interval_width)
            },
            "weights": breakdown['weights']
        }
        
        # Add quarter predictions if available
        if 'quarter_predictions' in breakdown:
            result["quarter_predictions"] = {
                q: float(score) for q, score in breakdown['quarter_predictions'].items()
            }
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in enhanced prediction: {str(e)}")

# Endpoint for player performance projection
@app.post("/predict/player")
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

# Endpoint for dynamic recommendations
@app.post("/recommendations")
def recommendations(request: DynamicRecommendationRequest):
    try:
        recs = generate_game_summary(request.model_outputs, request.player_projection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"recommendations": recs}

# Endpoint for quarter analysis: momentum shifts
@app.get("/quarter/momentum_shifts")
def momentum_shifts():
    data_path = os.path.join(os.path.dirname(__file__), '../data/historical_games.csv')
    try:
        df = pd.read_csv(data_path)
        # Process data to find momentum shifts using the feature generator
        processed_df = feature_generator.generate_all_features(df)
        # Extract momentum-related features
        shifts = processed_df[['game_id', 'home_team', 'away_team', 
                              'q1_to_q2_momentum', 'q2_to_q3_momentum', 
                              'q3_to_q4_momentum', 'cumulative_momentum']]
        return {"momentum_shifts": shifts.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing momentum shifts: {str(e)}")

# Endpoint for predicting remaining quarter scores
@app.post("/quarter/predict")
def quarter_prediction(request: QuarterAnalysisRequest):
    try:
        # Convert request data to dataframe format
        game_data = {
            'current_quarter': request.current_quarter,
            'home_team': request.team_stats.get('home_team', 'Home'),
            'away_team': request.team_stats.get('away_team', 'Away'),
        }
        
        # Add quarter scores from request
        for quarter in range(1, 5):
            game_data[f'home_q{quarter}'] = request.quarter_scores.get(f'home_q{quarter}', 0)
            game_data[f'away_q{quarter}'] = request.quarter_scores.get(f'away_q{quarter}', 0)
        
        # Add team stats
        for key, value in request.team_stats.items():
            game_data[key] = value
            
        # Create DataFrame
        df = pd.DataFrame([game_data])
        
        # Process with feature generator
        processed_df = feature_generator.generate_all_features(df)
        
        # Use quarter system to predict remaining quarters
        if not quarter_system.models:
            quarter_system.load_models()
            
        predictions = quarter_system.predict_remaining_quarters(
            processed_df.iloc[0], 
            request.current_quarter
        )
        
        return {"quarter_predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting quarters: {str(e)}")

# Endpoint for batch predictions for multiple games
@app.post("/predict/batch")
def batch_predictions(request: list):
    try:
        # Load models if not already loaded
        main_model = load_score_model()
        if not main_model:
            raise HTTPException(status_code=500, detail="Main prediction model not available.")
            
        if not quarter_system.models:
            quarter_system.load_models()
            
        # Convert request to DataFrame
        games_df = pd.DataFrame(request)
        
        # Generate enhanced predictions
        predictions = generate_enhanced_predictions(games_df, main_model)
        
        return {"predictions": predictions.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Load models during startup
    try:
        quarter_system.load_models()
        print("Quarter-specific models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load quarter-specific models: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)