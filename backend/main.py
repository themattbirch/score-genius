# backend/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# Import the Supabase client
# Adjust if your path to supabase_client is different
from caching.supabase_client import supabase

# Import your existing pre-game preview function
from src.scripts.nba_games_preview import build_game_preview

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your actual frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
#                               /game-preview                                 #
###############################################################################

@app.get("/game-preview")
def game_preview():
    """
    Returns pregame information, including betting odds and basic team data.
    """
    try:
        previews = build_game_preview()
        return {"data": previews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                             /historical-stats                               #
###############################################################################

@app.get("/historical-stats")
def get_historical_stats(
    start_date: Optional[str] = Query(None, description="Filter results where game_date >= start_date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter results where game_date <= end_date (YYYY-MM-DD)"),
    team_id: Optional[int] = Query(None, description="Filter results by team_id"),
):
    """
    Returns historical (2017–2023) NBA game stats from 'nba_historical_game_stats'.
    You can optionally provide start_date, end_date, and/or team_id to filter.
    """
    try:
        # Start building the query
        query = supabase.table("nba_historical_game_stats").select("*")

        # Filter by start_date
        if start_date:
            # Since game_date is text, a direct .gte("game_date", start_date)
            # works lexicographically if your date is stored as YYYY-MM-DD.
            query = query.gte("game_date", start_date)
        
        # Filter by end_date
        if end_date:
            query = query.lte("game_date", end_date)
        
        # Filter by team_id
        if team_id is not None:
            query = query.eq("team_id", team_id)

        response = query.execute()
        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                                /live-stats                                  #
###############################################################################

@app.get("/live-stats")
def get_live_stats(
    team_id: Optional[int] = Query(None, description="Filter live stats by team_id"),
):
    """
    Returns current (in-progress) NBA game stats from 'nba_live_game_stats'.
    Optionally filter by team_id. (Date filter can be added if desired.)
    """
    try:
        query = supabase.table("nba_live_game_stats").select("*")

        if team_id is not None:
            query = query.eq("team_id", team_id)

        response = query.execute()
        return {"data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###############################################################################
#                               /final-stats                                  #
###############################################################################

@app.get("/final-stats")
def get_final_stats(
    start_date: Optional[str] = Query(None, description="Filter final results where game_date >= start_date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter final results where game_date <= end_date (YYYY-MM-DD)"),
    team_id: Optional[int] = Query(None, description="Filter final results by team_id"),
):
    """
    Returns final NBA game stats from 'nba_2024_25_game_stats'.
    Optionally provide start_date, end_date, and team_id for filtering.
    """
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
#                               RUN THE SERVER                                #
###############################################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
