# backend/app.py
from fastapi import FastAPI
from routers import data_routes, analysis_routes, user_routes

app = FastAPI(
    title="ScoreGenius API",
    description="API for live sports analytics and predictive modeling.",
    version="1.0.0"
)

app.include_router(data_routes.router, prefix="/data", tags=["data"])
app.include_router(analysis_routes.router, prefix="/analysis", tags=["analysis"])
app.include_router(user_routes.router, prefix="/users", tags=["users"])

@app.get("/")
async def root():
    return {"message": "Welcome to ScoreGenius API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
