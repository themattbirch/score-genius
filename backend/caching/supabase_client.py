# /backend/caching/supabase_client.py


from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Adjust the path to your .env file as needed:
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("Supabase URL and anon key must be set in the .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
