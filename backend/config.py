# backend/config.py


from dotenv import load_dotenv
import os
import pathlib

# Load environment variables from the .env file located in the project root
env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
print("Loading .env from:", env_path.resolve())