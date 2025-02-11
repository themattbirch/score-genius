# backend/caching/supabase_cache.py
import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-anon-key")

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def cache_set(key: str, value: dict):
    supabase = get_supabase_client()
    data = {"key": key, "value": value}
    response = supabase.table("cache").upsert(data).execute()
    return response

def cache_get(key: str):
    supabase = get_supabase_client()
    response = supabase.table("cache").select("*").eq("key", key).execute()
    if response.data:
        return response.data[0]["value"]
    return None

if __name__ == "__main__":
    test_key = "test_key"
    test_value = {"example": "data"}
    print("Setting cache:", cache_set(test_key, test_value))
    print("Getting cache:", cache_get(test_key))
