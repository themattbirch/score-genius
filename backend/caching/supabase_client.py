# /backend/caching/supabase_client.py

import os
from supabase import create_client, Client


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
