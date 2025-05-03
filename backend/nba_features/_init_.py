# backend/nba_features/__init__.py

# Import the main pipeline function from the engine module within this package
from .engine import run_feature_pipeline

# Define what gets imported when someone does 'from backend.nba_features import *'
# Or what is accessible via 'from backend import nba_features; nba_features.run_feature_pipeline'
__all__ = ["run_feature_pipeline"]