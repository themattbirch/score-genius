# backend/config.py

from pathlib import Path
import os
from dotenv import load_dotenv
import logging # Import the logging library

# --- Basic Logging Setup ---
# Configure logging basics. Level INFO means INFO, WARNING, ERROR, CRITICAL messages will be shown.
# You might configure this more centrally in your main application entry point (e.g., main.py for FastAPI)
# but having a basic setup here makes config.py usable standalone or early in import chain.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__) # Get a logger specific to this module

# --- Path Definitions ---
try:
    # Define the project root explicitly. Assumes config.py is in the backend directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # Use DEBUG level for this, as it's very low-level path info, less likely needed in production logs
    logger.debug(f"PROJECT_ROOT automatically determined as: {PROJECT_ROOT}")

    # Define MODEL_PATH relative to the project root.
    # *** IMPORTANT: Ensure this path is correct for your deployed model! ***
    MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "pregame_model.pkl"
    logger.info(f"Using MODEL_PATH: {MODEL_PATH}") # INFO level seems appropriate for key paths

    # Define path to the .env file
    env_path = PROJECT_ROOT / ".env"

except Exception as e:
    logger.exception("CRITICAL: Failed to define essential paths (PROJECT_ROOT, MODEL_PATH). Check file structure.")
    # Optionally re-raise or exit if paths are absolutely critical for the app to even start
    raise

# --- Load Environment Variables ---
try:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Successfully loaded environment variables from: {env_path}")
    else:
        logger.warning(f".env file not found at: {env_path}. Relying on system environment variables.")
except Exception as e:
    logger.error(f"Error loading .env file from {env_path}: {e}", exc_info=True)


# --- Read Configuration Variables ---
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") # Or SUPABASE_KEY depending on need

# --- Log Status of Key Variables (without logging secrets) ---
config_status = {
    "API_SPORTS_KEY": "Set" if API_SPORTS_KEY else "Not Set",
    "ODDS_API_KEY": "Set" if ODDS_API_KEY else "Not Set",
    "DATABASE_URL": "Set" if DATABASE_URL else "Not Set",
    "SUPABASE_URL": "Set" if SUPABASE_URL else "Not Set",
    "SUPABASE_KEY": "Set" if SUPABASE_ANON_KEY else "Not Set" # Check the correct key name
}
logger.info(f"Configuration Loading Status: {config_status}")

# Add warnings for missing critical variables if desired
if not DATABASE_URL:
    logger.warning("DATABASE_URL is not configured. Database features will be unavailable.")
if not SUPABASE_URL or not SUPABASE_ANON_KEY: # Adjust key name if necessary
    logger.warning("Supabase URL or Key is not configured. Supabase features will be unavailable.")

# Convert MODEL_PATH to string only if needed by specific libraries, Path object is often better
# MODEL_PATH_STR = str(MODEL_PATH)!