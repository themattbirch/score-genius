import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from caching.redis_config import get_redis_client

app = FastAPI(
    title="ScoreGenius API",
    description="API for live sports analytics and predictive modeling.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing endpoints, health check, etc.
@app.get("/health")
def health_check():
    return {"status": "ok"}
