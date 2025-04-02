# from fastapi import APIRouter, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import os
# import pandas as pd

# from backend.ml.llama_inference import generate_recap
# from backend.analytics.win_probability import calculate_win_probability

# class ScorePredictionRequest(BaseModel):
#     features: list

# class PlayerProjectionRequest(BaseModel):
#     player_id: int
#     opponent_id: str = None
#     method: str = "xgboost"
#     current_live_data: dict = None

# class DynamicRecommendationRequest(BaseModel):
#     model_outputs: dict
#     player_projection: dict = None

# class QuarterAnalysisRequest(BaseModel):
#     current_quarter: int
#     quarter_scores: dict
#     team_stats: dict

# class EnhancedScorePredictionRequest(BaseModel):
#     game_data: dict

# router = APIRouter()

# @router.post("/recap")
# async def generate_game_recap(game_data: dict):
#     recap = generate_recap(game_data)
#     return JSONResponse(content={"recap": recap})

# @router.post("/win-probability")
# async def get_win_probability(game_data: dict):
#     probability = calculate_win_probability(game_data)
#     return JSONResponse(content={"win_probability": probability})

# @router.post("/predict/score")
# def score_prediction(request: ScorePredictionRequest):
#     model = load_score_model()
#     if not model:
#         raise HTTPException(status_code=500, detail="Score prediction model not available.")
#     try:
#         prediction = predict_final_score(model, request.features)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     return {"predicted_score": prediction}

# @router.post("/predict/score/enhanced")
# def enhanced_score_prediction(request: EnhancedScorePredictionRequest):
#     try:
#         main_model = load_score_model()
#         if not main_model:
#             raise HTTPException(status_code=500, detail="Main prediction model not available.")
#         game_data = pd.DataFrame([request.game_data])
#         features_df = feature_generator.generate_all_features(game_data)
#         expected_features = feature_generator.get_expected_features(
#             enhanced=hasattr(main_model, 'feature_importances_') and len(main_model.feature_importances_) > 8
#         )
#         main_pred = main_model.predict(features_df[expected_features])[0]
#         ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(features_df.iloc[0], main_pred)
#         current_quarter = int(features_df.iloc[0].get('current_quarter', 0))
#         score_margin = abs(float(features_df.iloc[0].get('score_differential', 0)))
#         lower_bound, upper_bound, interval_width = uncertainty_estimator.calculate_prediction_interval(
#             ensemble_pred, current_quarter, score_margin
#         )
#         result = {
#             "main_model_prediction": float(main_pred),
#             "quarter_model_prediction": float(breakdown['quarter_model']),
#             "ensemble_prediction": float(ensemble_pred),
#             "confidence": float(confidence),
#             "prediction_interval": {
#                 "lower_bound": float(lower_bound),
#                 "upper_bound": float(upper_bound),
#                 "width": float(interval_width)
#             },
#             "weights": breakdown['weights']
#         }
#         if 'quarter_predictions' in breakdown:
#             result["quarter_predictions"] = {
#                 q: float(score) for q, score in breakdown['quarter_predictions'].items()
#             }
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error in enhanced prediction: {str(e)}")

# @router.post("/predict/player")
# def player_projection(request: PlayerProjectionRequest):
#     models = load_player_projection_models(request.player_id)
#     if not models:
#         raise HTTPException(status_code=500, detail="Player projection models not available.")
#     try:
#         forecasts = predict_player_performance(
#             models,
#             steps=1,
#             method=request.method,
#             opponent_id=request.opponent_id,
#             current_live_data=request.current_live_data
#         )
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     return {"player_forecasts": forecasts}

# @router.post("/recommendations")
# def recommendations(request: DynamicRecommendationRequest):
#     try:
#         recs = generate_game_summary(request.model_outputs, request.player_projection)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     return {"recommendations": recs}

# @router.get("/quarter/momentum_shifts")
# def momentum_shifts():
#     try:
#         data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_games.csv')
#         df = pd.read_csv(data_path)
#         processed_df = feature_generator.generate_all_features(df)
#         shifts = processed_df[['game_id', 'home_team', 'away_team',
#                               'q1_to_q2_momentum', 'q2_to_q3_momentum',
#                               'q3_to_q4_momentum', 'cumulative_momentum']]
#         return {"momentum_shifts": shifts.to_dict(orient='records')}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing momentum shifts: {str(e)}")

# @router.post("/quarter/predict")
# def quarter_prediction(request: QuarterAnalysisRequest):
#     try:
#         game_data = {
#             'current_quarter': request.current_quarter,
#             'home_team': request.team_stats.get('home_team', 'Home'),
#             'away_team': request.team_stats.get('away_team', 'Away'),
#         }
#         for quarter in range(1, 5):
#             game_data[f'home_q{quarter}'] = request.quarter_scores.get(f'home_q{quarter}', 0)
#             game_data[f'away_q{quarter}'] = request.quarter_scores.get(f'away_q{quarter}', 0)
#         for key, value in request.team_stats.items():
#             game_data[key] = value
#         df = pd.DataFrame([game_data])
#         processed_df = feature_generator.generate_all_features(df)
#         predictions = quarter_system.predict_remaining_quarters(
#             processed_df.iloc[0],
#             request.current_quarter
#         )
#         return {"quarter_predictions": predictions}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error predicting quarters: {str(e)}")

# @router.post("/predict/batch")
# def batch_predictions(request: list):
#     try:
#         main_model = load_score_model()
#         if not main_model:
#             raise HTTPException(status_code=500, detail="Main prediction model not available.")
#         games_df = pd.DataFrame(request)
#         predictions = generate_enhanced_predictions(games_df, main_model)
#         return {"predictions": predictions.to_dict(orient='records')}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error in batch prediction: {str(e)}")
