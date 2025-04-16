# /backend/caching/supabase_client.py

from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Define path relative to this file to get to project root .env
# Assumes this file is in backend/caching, so needs to go up two levels
try:
    # Construct path relative to the current file's directory
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    print(f"Attempting to load .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
except Exception as e:
    print(f"Warning: Error loading .env file in supabase_client.py: {e}")
    # Continue, hoping environment variables are set system-wide

SUPABASE_URL = os.getenv("SUPABASE_URL")
# --- Load SERVICE KEY instead of ANON KEY ---
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# --- End Change ---

# --- Update Validation Check ---
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # Make sure SUPABASE_SERVICE_KEY exists in your .env file!
    print("FATAL ERROR: Supabase URL and/or SUPABASE_SERVICE_KEY not found in environment variables.")
    print("Ensure your root .env file exists and contains these variables.")
    exit(1) # Exit if essential keys are missing
# --- End Change ---

# --- Initialize Client with SERVICE KEY ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) # <-- USE SERVICE KEY HERE
    print("Shared Supabase client initialized successfully (using Service Key).")
except Exception as e:
     print(f"FATAL ERROR: Could not initialize Supabase client: {e}")
     supabase = None # Set to None or handle error as needed
     exit(1)
