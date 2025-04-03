# backend/api_integration.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd

# Import the correct model loading and prediction functions
from backend.nba_score_prediction.models import XGBoostScorePredictor, RandomForestScorePredictor, RidgeScorePredictor
from backend.nba_score_prediction.prediction import predict_final_score, generate_enhanced_predictions

# Import the feature engineering components
from backend.nba_score_prediction.feature_engineering import (
    NBAFeatureEngine,
    QuarterSpecificModelSystem,
    PredictionUncertaintyEstimator )

# Define Pydantic models for API requests
class ScorePredictionRequest(BaseModel):
    features: list  # e.g., [home_q1, home_q2, home_q3, home_q4, score_ratio, rolling_home_score, rolling_away_score]

class EnhancedScorePredictionRequest(BaseModel):
    game_data: dict  # Complete game data with all features needed

class QuarterAnalysisRequest(BaseModel):
    current_quarter: int
    quarter_scores: dict  # e.g., {'home_q1': 28, 'home_q2': 27, 'away_q1': 27, 'away_q2': 26, 'home_avg': 28, 'away_avg': 27}
    team_stats: dict      # e.g., {'home_off_rating': 30, 'away_off_rating': 29, 'is_home': True}


# Initialize FastAPI app
app = FastAPI(
    title="ScoreGenius API",
    description="API for live sports analytics and predictive modeling.",
    version="1.0.0"
)

# Initialize core components
feature_generator = NBAFeatureEngine(debug=False)
quarter_system = QuarterSpecificModelSystem(feature_generator)
uncertainty_estimator = PredictionUncertaintyEstimator()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Endpoint for score prediction
@app.post("/predict/score")
def score_prediction(request: ScorePredictionRequest):
    # Ensemble model loading logic
    predictors = {
        "xgboost": XGBoostScorePredictor(),
        "random_forest": RandomForestScorePredictor(),
        "ridge": RidgeScorePredictor()
    }
    models = {}
    for name, predictor in predictors.items():
        predictor.load_model()  # Load the model for each predictor
        models[name] = predictor.pipeline_home  # Access the actual model

    if not models:
        raise HTTPException(status_code=500, detail="No score prediction models available.")

    try:
        # Ensemble prediction logic (simplified example)
        ensemble_prediction = 0
        for name, model in models.items():
            if model: # Ensure model loaded correctly
                prediction = predict_final_score(model, request.features)
                if prediction is not None:
                    ensemble_prediction += prediction  # Simple average (can be weighted)
                else:
                    raise HTTPException(status_code=500, detail=f"Prediction failed for {name} model.")
        ensemble_prediction /= len(models)
        return {"predicted_score": round(ensemble_prediction, 1)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during score prediction: {str(e)}")


# Enhanced endpoint for score prediction with quarter-specific models
@app.post("/predict/score/enhanced")
def enhanced_score_prediction(request: EnhancedScorePredictionRequest):
    try:
        # Ensemble model loading logic
        predictors = {
            "xgboost": XGBoostScorePredictor(),
            "random_forest": RandomForestScorePredictor(),
            "ridge": RidgeScorePredictor()
        }
        models = {}
        for name, predictor in predictors.items():
            predictor.load_model()  # Load the model for each predictor
            models[name] = predictor.pipeline_home  # Access the actual model

        if not models:
            raise HTTPException(status_code=500, detail="Main prediction model not available.")

        # Load the main model (example: XGBoost, adjust as needed)
        main_model = models.get("xgboost")
        if not main_model:
            raise HTTPException(status_code=500, detail="Main prediction model (XGBoost) not available.")

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
        ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
            game_data=request.game_data, # Pass original game data
            main_model_prediction=main_pred,
            weight_manager=quarter_system.weight_manager # Ensure weight_manager is available
        )

        # Calculate uncertainty bounds
        current_quarter = int(features_df.iloc[0].get('current_quarter', 0))
        score_margin = abs(float(features_df.iloc[0].get('score_differential', 0)))
        lower_bound, upper_bound, interval_width = uncertainty_estimator.calculate_prediction_interval(
            ensemble_pred, current_quarter, score_margin
        )

        result = {
            "main_model_prediction": float(main_pred),
            "quarter_model_prediction": float(breakdown['quarter_model_pred']),
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

        return {"quarter_predictions": predictions.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting quarters: {str(e)}")

# Endpoint for batch predictions for multiple games
@app.post("/predict/batch")
def batch_predictions(request: list):
    try:
        # Ensemble model loading logic
        predictors = {
            "xgboost": XGBoostScorePredictor(),
            "random_forest": RandomForestScorePredictor(),
            "ridge": RidgeScorePredictor()
        }
        models = {}
        for name, predictor in predictors.items():
            predictor.load_model()  # Load the model for each predictor
            models[name] = predictor.pipeline_home  # Access the actual model

        if not models:
            raise HTTPException(status_code=500, detail="Main prediction model not available.")

        games_df = pd.DataFrame(request)
        predictions = generate_enhanced_predictions(games_df, models)

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