# backend/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# Pydantic models for new endpoints
from pydantic import BaseModel

# Supabase client
from caching.supabase_client import supabase

# Redis config
from caching.redis_config import get_redis_client

# Example function from your scripts
from src.scripts.nba_games_preview import build_game_preview

import uvicorn
import json

###############################################################################
#                        APP & CORS CONFIGURATION
###############################################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
#                       REDIS CLIENT (Caching)
###############################################################################
redis_client = get_redis_client()

###############################################################################
#               MODELS FOR WIN PROB / MOMENTUM / PLAYER PROJECTION
###############################################################################
class WinProbRequest(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    time_remaining: int  # e.g. seconds left in the game

class WinProbResponse(BaseModel):
    win_probability: float

class MomentumShiftRequest(BaseModel):
    """
    Example fields that might help detect a momentum shift.
    In reality, you'll tailor these to the actual data your model needs.
    """
    home_team: str
    away_team: str
    recent_plays: list[str]  # e.g., a list of textual play descriptions or events
    current_time: int        # e.g., game time in seconds

class MomentumShiftResponse(BaseModel):
    """
    A simple response showing the shift direction or intensity.
    """
    shift_score: float  # e.g., a numeric representation of how big the momentum shift is
    shift_team: str     # which team is gaining momentum

class PlayerProjectionRequest(BaseModel):
    """
    Basic info for projecting a player's performance.
    """
    player_name: str
    team_name: str
    upcoming_opponent: str
    recent_stats: dict  # e.g. { "points": [10, 18, 22], "rebounds": [5, 7, 9], ... }

class PlayerProjectionResponse(BaseModel):
    """
    Example output from a mock player-projection model.
    """
    predicted_points: float
    predicted_rebounds: float
    predicted_assists: float

###############################################################################
#                         HEALTH CHECK ENDPOINT
###############################################################################
@app.get("/health")
def health_check():
    return {"status": "ok"}

###############################################################################
#                      WIN PROBABILITY ENDPOINT (MOCK)
###############################################################################
@app.post("/predict/win-probability", response_model=WinProbResponse)
def predict_win_probability(request: WinProbRequest):
    """
    Mock example of a model endpoint that calculates (or pretends to calculate)
    the probability of the home team winning.
    """
    # Create a unique cache key based on request data
    cache_key = (
        f"winprob:{request.home_team}:{request.away_team}:"
        f"{request.home_score}:{request.away_score}:{request.time_remaining}"
    )

    # Check Redis cache
    cached_value = redis_client.get(cache_key)
    if cached_value:
        cached_data = json.loads(cached_value)
        return WinProbResponse(**cached_data)

    # If not cached, do mock model inference (replace with real logic)
    mock_probability = 0.75
    result = {"win_probability": mock_probability}

    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(result))

    return WinProbResponse(**result)

###############################################################################
#                    MOMENTUM SHIFT ENDPOINT (MOCK)
###############################################################################
@app.post("/predict/momentum-shift", response_model=MomentumShiftResponse)
def predict_momentum_shift(request: MomentumShiftRequest):
    """
    Mock example endpoint that detects a momentum shift in the game.
    We'll do minimal logic and caching here.
    """
    # Create a unique cache key
    cache_key = (
        f"momentum:{request.home_team}:{request.away_team}:"
        f"{request.current_time}:{','.join(request.recent_plays)}"
    )

    # Check Redis cache
    cached_value = redis_client.get(cache_key)
    if cached_value:
        cached_data = json.loads(cached_value)
        return MomentumShiftResponse(**cached_data)

    # Mock model logic
    # e.g., if 'recent_plays' includes “steal” or “fast break,” we might guess shift_score is higher
    shift_score = 0.0
    if any("steal" in play.lower() for play in request.recent_plays):
        shift_score += 0.2
    if any("fast break" in play.lower() for play in request.recent_plays):
        shift_score += 0.3

    # Decide which team is benefiting. (Completely arbitrary for demonstration!)
    shift_team = request.home_team if shift_score > 0.3 else request.away_team

    result = {
        "shift_score": shift_score,
        "shift_team": shift_team
    }

    # Cache the result
    redis_client.setex(cache_key, 300, json.dumps(result))

    return MomentumShiftResponse(**result)

###############################################################################
#                  PLAYER PROJECTION ENDPOINT (MOCK)
###############################################################################
@app.post("/predict/player-projection", response_model=PlayerProjectionResponse)
def predict_player_projection(request: PlayerProjectionRequest):
    """
    Mock endpoint for forecasting a player's performance in an upcoming game.
    """
    # Create a cache key
    cache_key = (
        f"proj:{request.player_name}:{request.team_name}:{request.upcoming_opponent}"
    )

    # Check Redis
    cached_value = redis_client.get(cache_key)
    if cached_value:
        cached_data = json.loads(cached_value)
        return PlayerProjectionResponse(**cached_data)

    # Mock model logic:
    # For demonstration, we'll average the player's recent points and add a random offset
    points_avg = 0
    rebounds_avg = 0
    assists_avg = 0

    # If user provided something like:
    # request.recent_stats = {"points": [10, 18, 22], "rebounds": [5, 7, 9], "assists": [2, 3, 6]}
    # We'll do a naive average:
    if "points" in request.recent_stats:
        points_avg = sum(request.recent_stats["points"]) / len(request.recent_stats["points"])
    if "rebounds" in request.recent_stats:
        rebounds_avg = sum(request.recent_stats["rebounds"]) / len(request.recent_stats["rebounds"])
    if "assists" in request.recent_stats:
        assists_avg = sum(request.recent_stats["assists"]) / len(request.recent_stats["assists"])

    # Some arbitrary “projection” formula
    # e.g. just add 2 points, 1 rebound, 1 assist to the average
    predicted_points = points_avg + 2
    predicted_rebounds = rebounds_avg + 1
    predicted_assists = assists_avg + 1

    result = {
        "predicted_points": predicted_points,
        "predicted_rebounds": predicted_rebounds,
        "predicted_assists": predicted_assists
    }

    # Cache
    redis_client.setex(cache_key, 300, json.dumps(result))

    return PlayerProjectionResponse(**result)

###############################################################################
#                    EXAMPLE DATA ENDPOINT (HISTORICAL)
###############################################################################
@app.get("/data/historical/{game_id}")
def get_historical_data(game_id: int):
    """
    Placeholder endpoint for retrieving historical data for a specific game.
    Right now it just returns mock data. 
    You could integrate Supabase logic here if you want.
    """
    return {
        "game_id": game_id,
        "status": "Not Implemented",
        "message": "Integrate your Supabase calls or other logic here."
    }

###############################################################################
#                          /GAME-PREVIEW
###############################################################################
@app.get("/game-preview")
def game_preview():
    """
    Returns pregame info, including betting odds, basic team data, etc.
    """
    try:
        previews = build_game_preview()
        return {"data": previews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                         /HISTORICAL-STATS
###############################################################################
@app.get("/historical-stats")
def get_historical_stats(
    start_date: Optional[str] = Query(None, description="Filter: game_date >= start_date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter: game_date <= end_date (YYYY-MM-DD)"),
    team_id: Optional[int] = Query(None, description="Filter results by team_id"),
):
    try:
        query = supabase.table("nba_historical_game_stats").select("*")
        if start_date:
            query = query.gte("game_date", start_date)
        if end_date:
            query = query.lte("game_date", end_date)
        if team_id is not None:
            query = query.eq("team_id", team_id)

        response = query.execute()
        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                          /LIVE-STATS
###############################################################################
@app.get("/live-stats")
def get_live_stats(
    team_id: Optional[int] = Query(None, description="Filter live stats by team_id"),
):
    try:
        query = supabase.table("nba_live_game_stats").select("*")
        if team_id is not None:
            query = query.eq("team_id", team_id)

        response = query.execute()
        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                         /FINAL-STATS
###############################################################################
@app.get("/final-stats")
def get_final_stats(
    start_date: Optional[str] = Query(None, description="Filter: game_date >= start_date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter: game_date <= end_date (YYYY-MM-DD)"),
    team_id: Optional[int] = Query(None, description="Filter results by team_id"),
):
    try:
        query = supabase.table("nba_2024_25_game_stats").select("*")
        if start_date:
            query = query.gte("game_date", start_date)
        if end_date:
            query = query.lte("game_date", end_date)
        if team_id is not None:
            query = query.eq("team_id", team_id)

        response = query.execute()
        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                           RUN THE SERVER
###############################################################################
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
