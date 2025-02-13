# backend/routers/data_routes.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.utils.fetch_data import fetch_live_data, fetch_historical_data  # Replace with your actual implementations

router = APIRouter()

@router.get("/live")
async def get_live_data():
    data = fetch_live_data()
    return JSONResponse(content=data)

@router.get("/historical")
async def get_historical_data(date: str):
    data = fetch_historical_data(date)
    return JSONResponse(content=data)

