import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("Supabase URL and anon key must be set in the .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# In a temporary test script or in supabase_client.py
result = supabase.table("nba_game_stats").select("*").execute()
print(result.data)
