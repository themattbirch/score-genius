# backend/scripts/rebuild_feature_store.py

import pandas as pd
import logging
import time # To time the process
import os
import sys

# --- 1. Configuration & Setup ---

# Add backend directory to sys.path to allow finding config.py
# Assumes this script is run from the project root (e.g., score-genius/)
HERE = os.path.dirname(__file__) # Should be backend/scripts
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir)) # Should be backend/
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir)) # Optional: project root

# Ensure backend directory is in path for config import
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
# Ensure project root is in path for backend.nba_features import if needed
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)


# --- Load Credentials (from config.py or environment) ---
try:
    # Assumes config.py is in the backend/ directory
    from config import (
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY # Service role key for admin tasks
        # Add other keys if needed by other feature modules
        # API_SPORTS_KEY,
        # ODDS_API_KEY,
    )
    print("Using config.py for credentials.")
except ImportError:
    print("config.py not found -> loading credentials from environment variables.")
    # Attempt to load from environment (e.g., .env file loaded by shell or Docker)
    # You might need `from dotenv import load_dotenv; load_dotenv()` here
    # if running locally and relying on a .env file at the project root.
    # from dotenv import load_dotenv # Example
    # load_dotenv(os.path.join(PROJECT_ROOT, '.env')) # Example if using python-dotenv

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    # API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    # ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Validate essential credentials
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("FATAL ERROR: Supabase URL and/or SUPABASE_SERVICE_KEY not found.")
    print("Ensure your backend/config.py or environment variables are set.")
    sys.exit(1) # Exit if essential credentials are missing

# --- Supabase Client Initialization (optional but good practice) ---
try:
    from supabase import create_client, Client
    # Add schema option if your table is not in 'public'
    # options = {"schema": "your_schema_name"}
    # supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY, options=options)
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Supabase client initialized successfully.")
except ImportError:
    print("WARNING: 'supabase' library not found. Cannot initialize client.")
    print("Install it using: pip install supabase")
    supabase_client = None
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    supabase_client = None


# --- Import Feature Modules ---
# Now that sys.path is potentially updated, try importing features
try:
    # Import necessary exceptions from Supabase client library for better error handling
    from postgrest.exceptions import APIError
    from backend.nba_features import rolling # Use the updated rolling.py
    # Import other feature modules you have (e.g., h2h, team_stats_features, etc.)
    # from backend.nba_features import h2h
    # from backend.nba_features import some_other_feature
except ImportError as e:
    print(f"FATAL ERROR: Could not import feature modules or Supabase exceptions: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. Load Raw Data ---
# Replace this section with your actual data loading logic
def load_raw_nba_data(client: Client | None):
    """
    Loads raw NBA game data from the 'nba_historical_game_stats' table.
    Includes basic pagination for potentially large tables.
    """
    logging.info("Loading raw NBA game data...")
    if not client:
        logging.error("Supabase client not available. Cannot load data.")
        return pd.DataFrame()

    table_name = "nba_historical_game_stats"
    all_data = []
    offset = 0
    limit = 1000 # Supabase default limit per request
    logging.info(f"Querying Supabase table: {table_name} with pagination (limit={limit})...")

    while True:
        try:
            response = client.table(table_name).select("*").range(offset, offset + limit - 1).execute()

            if hasattr(response, 'data') and response.data:
                logging.debug(f"Fetched {len(response.data)} records (offset={offset}).")
                all_data.extend(response.data)
                offset += limit
                # Stop if last page was not full (or exactly full)
                if len(response.data) < limit:
                    break
            else:
                # No more data or empty table
                logging.debug(f"No more data found at offset {offset}.")
                break
        except APIError as api_error:
            logging.error(f"Supabase API Error during data fetch: {api_error.message}", exc_info=True)
            logging.error(f"APIError details: Code={api_error.code}, Hint={api_error.hint}, Message={api_error.message}")
            return pd.DataFrame() # Return empty DataFrame on API error
        except Exception as e:
            logging.error(f"Error loading raw data chunk from Supabase (offset={offset}): {e}", exc_info=True)
            return pd.DataFrame() # Return empty DF on other errors

    if not all_data:
        logging.warning(f"No data loaded from Supabase table {table_name}.")
        return pd.DataFrame()

    raw_games_df = pd.DataFrame(all_data)
    logging.info(f"Loaded a total of {len(raw_games_df)} raw game records from {table_name}.")


    # --- IMPORTANT: Ensure data is sorted correctly before feature engineering ---
    if 'game_date' in raw_games_df.columns:
         raw_games_df['game_date'] = pd.to_datetime(raw_games_df['game_date'])
         sort_columns = ['game_date']
         if 'game_id' in raw_games_df.columns:
             sort_columns.append('game_id')
         try:
            raw_games_df = raw_games_df.sort_values(by=sort_columns).reset_index(drop=True)
            logging.info(f"Raw data sorted by {sort_columns}.")
         except Exception as sort_e:
             logging.error(f"Failed to sort DataFrame by {sort_columns}: {sort_e}", exc_info=True)
             # Decide if you want to proceed with unsorted data or stop
             # return pd.DataFrame() # Example: Stop if sorting fails
    else:
         logging.warning("Column 'game_date' not found for sorting.")

    # --- Add required columns if missing (e.g., for rolling features) ---
    required_rating_cols = {
        "home_offensive_rating", "away_offensive_rating",
        "home_defensive_rating", "away_defensive_rating",
        "home_net_rating", "away_net_rating"
    }
    # Check if score columns exist first
    has_home_score = 'home_score' in raw_games_df.columns
    has_away_score = 'away_score' in raw_games_df.columns

    for col in required_rating_cols:
        if col not in raw_games_df.columns:
            logging.warning(f"Column '{col}' missing. Adding placeholder. Adjust logic as needed.")
            # Derive placeholder only if score columns exist
            if has_home_score and has_away_score:
                if 'offensive_rating' in col:
                     source_col = 'home_score' if 'home' in col else 'away_score'
                     raw_games_df[col] = raw_games_df[source_col]
                elif 'defensive_rating' in col:
                     # Defensive rating often mirrors opponent's offensive rating (score)
                     source_col = 'away_score' if 'home' in col else 'home_score'
                     raw_games_df[col] = raw_games_df[source_col]
                elif 'net_rating' in col:
                     raw_games_df[col] = raw_games_df['home_score'] - raw_games_df['away_score'] if 'home' in col else raw_games_df['away_score'] - raw_games_df['home_score']
                else: # Default placeholder if logic doesn't match
                     raw_games_df[col] = 100 # Or np.nan potentially
            else: # Fallback if score columns are also missing
                logging.warning(f"Score columns missing, cannot derive placeholder for {col}. Setting to 100.")
                raw_games_df[col] = 100 # Or np.nan

    return raw_games_df


# --- 3. Apply Feature Transformations ---
# (No changes needed in this function based on the error)
def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in sequence.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty, skipping feature generation.")
        return df

    logging.info("Starting feature generation...")
    features_df = df.copy()
    start_time = time.time()

    # --- Apply Rolling Features (using the updated script) ---
    try:
        logging.info("Applying rolling features...")
        # Define window sizes you want
        windows = [5, 10, 20, 30] # Example window sizes
        features_df = rolling.transform(features_df, window_sizes=windows, debug=False) # Ensure rolling.py is the updated version
        logging.info(f"Rolling features applied. Shape: {features_df.shape}")
    except Exception as e:
        logging.error(f"Error applying rolling features: {e}", exc_info=True) # Log traceback
        # Decide if you want to stop or continue if one step fails
        # raise

    # --- Apply Other Features ---
    # Add calls to your other feature transformation functions here
    # Example:
    # try:
    #     logging.info("Applying H2H features...")
    #     # features_df = h2h.transform(features_df, ...) # Add necessary args
    #     logging.info(f"H2H features applied. Shape: {features_df.shape}")
    # except Exception as e:
    #     logging.error(f"Error applying H2H features: {e}", exc_info=True)
    #     # raise

    # try:
    #     logging.info("Applying Some Other features...")
    #     # features_df = some_other_feature.transform(features_df, ...) # Add necessary args
    #     logging.info(f"Some Other features applied. Shape: {features_df.shape}")
    # except Exception as e:
    #     logging.error(f"Error applying Some Other features: {e}", exc_info=True)
    #     # raise

    end_time = time.time()
    logging.info(f"Feature generation completed in {end_time - start_time:.2f} seconds.")
    return features_df


# --- 4. Save Processed Features ---
def save_features(df: pd.DataFrame, destination_table: str, client: Client | None):
    """
    Saves the DataFrame with all features to Supabase, handling data types and errors.
    """
    if df.empty:
        logging.warning("Feature DataFrame is empty, nothing to save.")
        return
    if not client:
        logging.error("Supabase client not available. Cannot save data.")
        return

    logging.info(f"Saving features to Supabase table: {destination_table}...")
    df_processed = df.copy() # Work on a copy

    # --- Data Type Conversion for Supabase Compatibility ---
    # Convert datetime columns to ISO format strings
    for col in df_processed.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
         logging.debug(f"Converting column {col} to ISO format string for Supabase.")
         # Use .dt accessor to apply isoformat() element-wise
         # Handle NaT (Not a Time) values gracefully by converting them to None
         df_processed[col] = df_processed[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

    # Convert potentially problematic types (like numpy types) to standard Python types
    # Convert remaining NaNs/NaTs/Pd.NA to None for JSON/database compatibility
    df_processed = df_processed.astype(object).where(pd.notnull(df_processed), None)

    # Convert DataFrame to list of dictionaries (records)
    # This should now handle None values correctly
    try:
        records = df_processed.to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error converting DataFrame to records (dict): {e}", exc_info=True)
        # Log columns and dtypes for debugging
        logging.error(f"DataFrame dtypes:\n{df_processed.dtypes}")
        return # Stop if conversion fails

    # Optional: Add chunking for very large uploads
    chunk_size = 500 # Adjust as needed based on performance/limits
    num_chunks = (len(records) + chunk_size - 1) // chunk_size
    logging.info(f"Preparing to upsert {len(records)} records in {num_chunks} chunk(s) of size {chunk_size}.")
    errors_occurred = False

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = records[start_index:end_index]
        logging.info(f"Upserting chunk {i+1}/{num_chunks} ({len(chunk)} records)...")

        try:
            # Upsert is generally safer for rebuilds if rows might already exist:
            # Ensure your destination table has a primary key defined (e.g., 'game_id' or 'id')
            # If the primary key is NOT 'id', specify it using `on_conflict`
            # Example: response = client.table(destination_table).upsert(chunk, on_conflict='game_id').execute()
            response = client.table(destination_table).upsert(chunk).execute()
            # Add more robust response checking if needed
            # Check response status, etc. if the library provides it
            logging.debug(f"Chunk {i+1} upsert completed.")

        except APIError as api_error:
            # Log more detailed API error information
            errors_occurred = True
            logging.error(f"Supabase API Error saving chunk {i+1} to {destination_table}: {api_error.message}", exc_info=False) # Set exc_info=False for brevity
            logging.error(f"APIError details: Code={api_error.code}, Hint={api_error.hint}, Message={api_error.message}, Full Error: {api_error}")
            # break # Optional: Stop processing further chunks on error
        except Exception as e:
            errors_occurred = True
            logging.error(f"Generic error saving chunk {i+1} to Supabase table {destination_table}: {e}", exc_info=True)
            # break # Optional: Stop processing further chunks on error

    if errors_occurred:
         logging.warning(f"Finished saving features to Supabase table: {destination_table}, but errors occurred during the process.")
    else:
         logging.info(f"Finished saving features successfully to Supabase table: {destination_table}.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    logging.info("--- Starting Feature Store Rebuild ---")

    # Define where to save the final features (Supabase table name)
    # *** IMPORTANT: Verify this table name exists in your Supabase project! ***
    feature_store_destination_table = "nba_feature_store" # Customize this table name

    # Step 1: Load Data
    # Pass the initialized Supabase client to the loading function
    raw_data = load_raw_nba_data(supabase_client)

    # Step 2: Apply Features
    if raw_data is not None and not raw_data.empty:
        features_data = apply_features(raw_data)
    else:
        logging.warning("Raw data loading failed or returned empty, skipping feature generation.")
        features_data = pd.DataFrame() # Ensure it's an empty DF

    # Step 3: Save Features
    if not features_data.empty:
        # Pass the client and destination table name to the saving function
        save_features(features_data, feature_store_destination_table, supabase_client)
    else:
        logging.warning("Skipping save because feature generation resulted in an empty DataFrame.")

    logging.info("--- Feature Store Rebuild Script Finished ---")