# backend/config.py

from pathlib import Path
import os
from dotenv import load_dotenv
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Definitions ---
try:
    # Define the project root explicitly. Assumes config.py is in the backend directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    logger.debug(f"PROJECT_ROOT automatically determined as: {PROJECT_ROOT}")

    # Define MODEL_PATH relative to the project root. (Using path from your working example)
    MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "pregame_model.pkl"
    logger.info(f"Using MODEL_PATH: {MODEL_PATH}")

    # Define path to the .env file in the project root
    env_path = PROJECT_ROOT / ".env"
    # logger.info(f"Expecting .env file at: {env_path}") # Optional: uncomment for debugging path

except Exception as e:
    logger.exception("CRITICAL: Failed to define essential paths (PROJECT_ROOT, MODEL_PATH). Check file structure.")
    raise

# --- Load Environment Variables ---
try:
    # Use the calculated env_path from PROJECT_ROOT
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Successfully loaded environment variables from: {env_path}")
    else:
        # Log the path it tried to find
        logger.warning(f".env file not found at: {env_path}. Relying on system environment variables.")
except Exception as e:
    logger.error(f"Error loading .env file from {env_path}: {e}", exc_info=True)

# --- Read Configuration Variables ---
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# --- NEW: Add RapidAPI Credentials ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
# --- End New Section ---

# --- Log Status of Key Variables ---
config_status = {
    "API_SPORTS_KEY": "Set" if API_SPORTS_KEY else "Not Set",
    "ODDS_API_KEY": "Set" if ODDS_API_KEY else "Not Set",
    "DATABASE_URL": "Set" if DATABASE_URL else "Not Set",
    "SUPABASE_URL": "Set" if SUPABASE_URL else "Not Set",
    "SUPABASE_KEY": "Set" if SUPABASE_ANON_KEY else "Not Set",
     # --- NEW: Add RapidAPI status ---
    "RAPIDAPI_KEY": "Set" if RAPIDAPI_KEY else "Not Set",
    "RAPIDAPI_HOST": "Set" if RAPIDAPI_HOST else "Not Set",
    # --- End New Section ---
}
logger.info(f"Configuration Loading Status: {config_status}")

# --- Warnings for Missing Variables ---
# Keep warnings as they were in your working example, plus add new ones
if not DATABASE_URL:
    logger.warning("DATABASE_URL is not configured.")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.warning("Supabase URL or Key is not configured.")
# --- NEW: Add RapidAPI warning ---
if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
    logger.warning("RapidAPI Key or Host is not configured. MLB schedule/pitcher features may be unavailable.")
# --- End New Section ---

# Convert MODEL_PATH to string only if needed by specific libraries
# MODEL_PATH_STR = str(MODEL_PATH)!