# backend/routers/user_routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/profile")
async def get_user_profile(user_id: int):
    dummy_user = {
        "id": user_id,
        "username": f"user{user_id}",
        "email": f"user{user_id}@example.com"
    }
    return JSONResponse(content=dummy_user)

@router.post("/login")
async def login(username: str, password: str):
    if username == "test" and password == "test":
        return JSONResponse(content={"token": "dummy_jwt_token"})
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
