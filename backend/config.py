# backend/config.py

import os
import logging
from pathlib import Path
from dotenv import load_dotenv # Ensure dotenv is installed: pip install python-dotenv

# --- Basic Logging Setup ---
# Configure logging early. If other modules also configure it, ensure consistency
# or use a central logging setup function imported here.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Path Definitions ---
try:
    # Define paths relative to this config.py file
    BACKEND_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BACKEND_DIR.parent
    logger.debug(f"PROJECT_ROOT determined as: {PROJECT_ROOT}")

    # Define MODEL_PATH relative to the project root (adjust path if needed)
    MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "pregame_model.pkl"
    logger.info(f"Using MODEL_PATH: {MODEL_PATH}")

    # project‐root/models/saved
    MAIN_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
    REPORTS_DIR     = Path(__file__).resolve().parent.parent / "reports"

    # ensure they exist on import
    MAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Define path to the .env file IN THE BACKEND DIRECTORY ---
    env_path = BACKEND_DIR / ".env"

except Exception as path_e:
    logger.exception("CRITICAL: Failed to define essential paths. Check file structure relative to config.py.")
    # Depending on how critical paths are, you might raise the error
    # raise path_e
    # For now, let's allow continuing to see if env vars load standalone

# --- Load Environment Variables from backend/.env ---
try:
    logger.info(f"Looking for .env file at: {env_path}")
    if env_path.is_file():
        # Load environment variables from backend/.env
        # override=True means variables in .env will overwrite existing system env vars
        dotenv_loaded = load_dotenv(dotenv_path=env_path, override=True)
        if dotenv_loaded:
            logger.info(f"Successfully loaded environment variables from: {env_path}")
        else:
            # This might happen if the file is empty or has issues
             logger.warning(f".env file found at {env_path}, but load_dotenv returned False.")
    else:
        logger.warning(f".env file not found at: {env_path}. Relying on system environment variables if available.")
except Exception as e:
    logger.error(f"Error loading .env file from {env_path}: {e}", exc_info=True)

# --- Read Configuration Variables & Perform Strict Validation ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
# Add any other CRITICAL keys needed for basic operation here

# Validate critical variables - Raise Error if missing to halt execution
if not SUPABASE_URL:
    raise EnvironmentError("FATAL: SUPABASE_URL not found in environment variables. Check backend/.env.")
if not SUPABASE_SERVICE_KEY:
    raise EnvironmentError("FATAL: SUPABASE_SERVICE_KEY not found in environment variables. Check backend/.env.")
if not API_SPORTS_KEY:
    raise EnvironmentError("FATAL: API_SPORTS_KEY not found in environment variables. Check backend/.env.")
if not ODDS_API_KEY:
     raise EnvironmentError("FATAL: ODDS_API_KEY not found in environment variables. Check backend/.env.")

# --- Read Optional/Less Critical Variables ---
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") # Often same as Supabase URL but might differ
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

# --- Log Status of All Variables ---
config_status = {
    "API_SPORTS_KEY": "Set" if API_SPORTS_KEY else "Not Set",
    "ODDS_API_KEY": "Set" if ODDS_API_KEY else "Not Set",
    "DATABASE_URL": "Set" if DATABASE_URL else "Not Set",
    "SUPABASE_URL": "Set" if SUPABASE_URL else "Not Set",
    "SUPABASE_ANON_KEY": "Set" if SUPABASE_ANON_KEY else "Not Set",
    "SUPABASE_SERVICE_KEY": "Set" if SUPABASE_SERVICE_KEY else "Not Set",
    "RAPIDAPI_KEY": "Set" if RAPIDAPI_KEY else "Not Set",
    "RAPIDAPI_HOST": "Set" if RAPIDAPI_HOST else "Not Set",
}
logger.info(f"Configuration Loading Status (Read from Environment): {config_status}")

# --- Warnings for Missing Non-Critical Variables ---
if not SUPABASE_ANON_KEY:
    logger.warning("SUPABASE_ANON_KEY not configured (might be needed for some operations).")
if not DATABASE_URL:
    logger.warning("DATABASE_URL is not configured (might be needed for non-Supabase DB access).")
if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
    logger.warning("RapidAPI Key or Host is not configured. Features relying on it may be unavailable.")

# You can add other config variables, type conversions, etc. below