# backend/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Define the project root explicitly. For example, if this file is in the backend directory:
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Now define MODEL_PATH relative to the project root.
MODEL_PATH = str(PROJECT_ROOT / "notebooks" / "models" / "pregame_model.pkl")

# Optionally load other environment variables
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MODEL_PATH:", MODEL_PATH)
