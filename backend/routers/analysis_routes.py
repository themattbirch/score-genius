# backend/routers/analysis_routes.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.ml.llama_inference import generate_recap
from backend.analytics.win_probability import calculate_win_probability  # Replace with your actual function

router = APIRouter()

@router.post("/recap")
async def generate_game_recap(game_data: dict):
    recap = generate_recap(game_data)
    return JSONResponse(content={"recap": recap})

@router.post("/win-probability")
async def get_win_probability(game_data: dict):
    probability = calculate_win_probability(game_data)
    return JSONResponse(content={"win_probability": probability})
