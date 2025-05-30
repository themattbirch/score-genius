# backend/nba_features/engine.py

"""
Orchestrates the execution of the modular feature-engineering pipeline for NBA games.
Each module exports a `transform(df, **kwargs)` function; we call them in order.
The engine can run fully offline (pass your own DataFrames) or auto-fetch the
needed helpers (season stats, historical games, rolling windows) from Supabase
when you hand in a `db_conn`.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict, Sequence # Added Sequence

import pandas as pd
# numpy is used by pandas and potentially by some feature modules indirectly
# import numpy as np # Not directly used in this version of engine.py

# Load .env (backend/.env or project-root/.env)
# This pattern is fine for loading .env files.
for env_path in (
    Path(__file__).resolve().parents[1] / '.env', # .../backend/.env
    Path(__file__).resolve().parents[2] / '.env', # project-root/.env
):
    if env_path.is_file():
        load_dotenv(env_path, override=True)
        # logger.info(f"Loaded .env from: {env_path}") # Logging here might be too early if logger not yet configured
        break

# Supabase singleton (if available)
try:
    from caching.supabase_client import supabase as supabase_client_singleton # Renamed for clarity
    SUPABASE_AVAILABLE = True
except ImportError:
    supabase_client_singleton = None
    SUPABASE_AVAILABLE = False

# Feature modules - ensure these .py files exist in the same directory (nba_features)
from .game_advanced_metrics import transform as game_adv_transform
from .rest import transform as rest_transform
from .h2h import transform as h2h_transform
from .form import transform as form_transform
from .season import transform as season_transform
from .advanced import transform as adv_splits_transform # This is the refactored one
from .rolling import transform as rolling_transform

from .utils import determine_season
# from .base_windows import fetch_rolling # Uncomment if pre-computed rolling stats are also used

# Type hint for Supabase client
from supabase import Client

logger = logging.getLogger(__name__) # Get logger after basicConfig if it's set elsewhere
__all__ = ['run_feature_pipeline']

# Default module order and mapping
DEFAULT_ORDER: List[str] = [ # Changed to List[str] for more specific typing
    'game_advanced_metrics', 'rest', 'h2h', 'form', 'season', 'adv_splits', 'rolling'
]
TRANSFORMS: Dict[str, Any] = { # Any for callable is practical here
    'game_advanced_metrics': game_adv_transform,
    'rest': rest_transform,
    'h2h': h2h_transform,
    'form': form_transform,
    'season': season_transform,
    'adv_splits': adv_splits_transform,
    'rolling': rolling_transform,
}

# --- Helper Fetchers with Improved Logging ---
def _fetch_table_from_supabase(
    db: Client,
    table_name: str,
    filter_params: Optional[Dict[str, Any]] = None,
    select_cols: str = '*'
) -> pd.DataFrame:
    """Generic helper to fetch data from a Supabase table with optional filters."""
    log_params = f"with params: {filter_params}" if filter_params else "all rows"
    logger.info(f"Fetching data from '{table_name}' ({log_params})...")
    try:
        query = db.from_(table_name).select(select_cols)
        if filter_params:
            for key, value in filter_params.items():
                query = query.eq(key, value)
        
        response = query.execute()
        
        if getattr(response, 'error', None):
            logger.error(f"Error fetching from '{table_name}': {response.error}")
            return pd.DataFrame()
        
        data = getattr(response, 'data', [])
        logger.debug(f"Fetched {len(data)} rows from '{table_name}'.")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Exception while fetching from '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

def _fetch_team_stats(db: Client) -> pd.DataFrame:
    return _fetch_table_from_supabase(db, "nba_historical_team_stats")

def _fetch_game_stats(db: Client) -> pd.DataFrame:
    return _fetch_table_from_supabase(db, "nba_historical_game_stats")

def _fetch_adv_splits(db: Client, season_year: int) -> pd.DataFrame:
    return _fetch_table_from_supabase(
        db,
        "nba_team_seasonal_advanced_splits",
        filter_params={"season": season_year}
    )

# --- Main Pipeline Function ---
def run_feature_pipeline(
    df: pd.DataFrame,
    *,
    db_conn: Optional[Client] = None,
    execution_order: Optional[List[str]] = None,
    h2h_lookback: int = 7,
    rolling_windows: Optional[List[int]] = None, # Pass None to use default, empty list to disable
    adv_splits_lookup_offset: int = -1, # Renamed for clarity
    flag_imputations_all: bool = True, # Single flag for all modules that support it
    debug: bool = False,
) -> pd.DataFrame:
    """
    Enrich `df` with feature-engineering transforms executed in sequence.
    """
    # Configure logger for this run if debug is True
    # Note: If basicConfig was called at module level, setLevel here might be overridden
    # or cause unexpected behavior if not managed carefully. Ideally, setLevel on specific logger.
    original_logger_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"engine.run_feature_pipeline: DEBUG mode activated. Initial df shape: {df.shape}")
    
    pipeline_start_time = time.time()

    # Determine effective execution order
    active_order = (execution_order or DEFAULT_ORDER).copy()
    
    # Handle rolling_windows: None means use default, [] means disable.
    effective_rolling_windows = rolling_windows if rolling_windows is not None else [5, 10, 20]
    if not effective_rolling_windows and "rolling" in active_order: # Empty list explicitly disables
        logger.info("Rolling module disabled due to empty 'rolling_windows' list.")
        active_order.remove("rolling")
    
    logger.info(f"Effective feature module execution order: {active_order}")

    processed_df = df.copy() # Work on a copy

    # Validate essential input columns in df
    required_df_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if processed_df.empty: # Check after copy
        logger.warning("engine.run_feature_pipeline: Input DataFrame 'df' is empty. Aborting.")
        if debug: logger.setLevel(original_logger_level)
        return processed_df # Return the empty copy
        
    missing_input_cols = [col for col in required_df_cols if col not in processed_df.columns]
    if missing_input_cols:
        logger.error(f"engine.run_feature_pipeline: Input 'df' missing required columns: {missing_input_cols}. Aborting.")
        if debug: logger.setLevel(original_logger_level)
        return processed_df # Return copy as is (it's unusable for pipeline)

    # Standardize game_date early
    processed_df['game_date'] = pd.to_datetime(processed_df['game_date'], errors='coerce')
    if processed_df['game_date'].isnull().any():
        logger.warning("engine.run_feature_pipeline: 'game_date' column contains NaT values after coercion. This might affect date-sensitive features.")
        # Consider dropping rows with NaT game_date if they cannot be processed
        # processed_df.dropna(subset=['game_date'], inplace=True)
        # if processed_df.empty:
        #     logger.error("DataFrame empty after removing rows with invalid game_date. Aborting.")
        #     if debug: logger.setLevel(original_logger_level)
        #     return processed_df


    # Determine active Supabase connection
    active_db_conn = db_conn if db_conn is not None else supabase_client_singleton
    if active_db_conn is None:
        logger.warning("No Supabase connection available. Data fetching for helper tables will be skipped.")

    # --- Pre-fetch or prepare helper data based on active modules ---
    # These are fetched once if needed by any module in the execution order.
    _general_team_stats_df: Optional[pd.DataFrame] = None
    if active_db_conn and any(m in active_order for m in ['season', 'form']):
        _general_team_stats_df = _fetch_team_stats(active_db_conn)

    _historical_games_df: Optional[pd.DataFrame] = None
    if active_db_conn and 'h2h' in active_order:
        _historical_games_df = _fetch_game_stats(active_db_conn)

    _adv_splits_cache: Dict[int, pd.DataFrame] = {} # Cache for fetched advanced splits
    _lookup_season_for_adv_splits: Optional[int] = None
    if 'adv_splits' in active_order:
        if not processed_df['game_date'].dropna().empty:
            # Determine lookup season based on the minimum game date in the current batch
            min_date_in_batch = processed_df['game_date'].dropna().min()
            season_of_min_date = int(determine_season(min_date_in_batch).split('-')[0])
            _lookup_season_for_adv_splits = season_of_min_date + adv_splits_lookup_offset
            
            if active_db_conn and _lookup_season_for_adv_splits not in _adv_splits_cache:
                fetched_data = _fetch_adv_splits(active_db_conn, _lookup_season_for_adv_splits)
                _adv_splits_cache[_lookup_season_for_adv_splits] = fetched_data # Cache even if empty
        else:
            logger.warning("Cannot determine lookup season for 'adv_splits': 'game_date' is empty or all NaT.")


    # --- Execute Feature Engineering Modules in Order ---
    for module_name in active_order:
        transform_function = TRANSFORMS.get(module_name)
        if not transform_function:
            logger.warning(f"No transform function found for module '{module_name}'. Skipping.")
            continue

        module_start_time = time.time()
        logger.info(f"Running module: '{module_name}'...")

        # Prepare kwargs for the current module
        current_module_kwargs: Dict[str, Any] = {'debug': debug}
        if module_name == 'h2h':
            current_module_kwargs.update({
                'historical_df': _historical_games_df,
                'max_games': h2h_lookback
            })
        elif module_name == 'season':
            current_module_kwargs.update({
                'team_stats_df': _general_team_stats_df,
                'flag_imputations': flag_imputations_all
            })
        elif module_name == 'form': # form.py also uses team_stats_df
            current_module_kwargs.update({'team_stats_df': _general_team_stats_df})
        elif module_name == 'adv_splits':
            current_module_kwargs.update({
                'stats_df': _adv_splits_cache.get(_lookup_season_for_adv_splits, pd.DataFrame()), # Pass cached or empty DF
                'season': _lookup_season_for_adv_splits, # Pass the season year it pertains to
                'flag_imputations': flag_imputations_all
            })
        elif module_name == 'rolling':
            current_module_kwargs.update({
                'window_sizes': effective_rolling_windows,
                'flag_imputation': flag_imputations_all
            })
        # 'game_advanced' and 'rest' might only need df and debug, handled by initial current_module_kwargs

        try:
            processed_df = transform_function(processed_df, **current_module_kwargs)
            logger.debug(f"Module '{module_name}' completed in {time.time() - module_start_time:.2f}s. DF shape: {processed_df.shape}")
        except Exception as e:
            logger.error(f"Error executing module '{module_name}': {e}", exc_info=debug)
            logger.error("Feature pipeline halted due to an error.")
            # Restore logger level before exiting due to error
            if debug: logger.setLevel(original_logger_level)
            return processed_df # Return df in its current state or df.copy() if preferred

        if processed_df.empty: # Check if a module unexpectedly emptied the DataFrame
            logger.error(f"DataFrame became empty after module '{module_name}'. Halting pipeline.")
            break
            
    logger.info(
        f"Feature pipeline completed in {time.time() - pipeline_start_time:.2f}s. Final DataFrame shape: {processed_df.shape}"
    )
    if debug: # Restore original logger level
        logger.setLevel(original_logger_level)
    return processed_df

# --- CLI Smoke Test (Example) ---
if __name__ == '__main__':
    # For direct execution, ensure logging is configured
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | [%(name)s:%(funcName)s:%(lineno)d] – %(message)s',
    )
    logger.info("Running smoke test for feature pipeline...")

    # Ensure SUPABASE_AVAILABLE and supabase_client_singleton are correctly evaluated
    db_connection_for_test = supabase_client_singleton if SUPABASE_AVAILABLE else None
    if db_connection_for_test is None:
         logger.warning("Smoke test: No actual Supabase connection; helper data fetching will be skipped or use mocked data.")


    dummy_games_data = pd.DataFrame({
        "game_id": ["G201", "G202"],
        "game_date": pd.to_datetime(["2023-12-01", "2023-12-01"]),
        "home_team": ["Team Foo", "Team Bar"], 
        "away_team": ["Team Baz", "Team Qux"],
        
        # Existing box score stats
        'home_score': [100, 110], 'away_score': [90, 115], 
        'home_fg_made': [35, 40], 'home_fg_attempted': [80, 85], 
        'away_fg_made': [33, 42], 'away_fg_attempted': [78, 88],
        'home_3pm': [10,12], 
        'away_3pm': [8,11], 
        'home_ft_attempted': [20, 25], 'away_ft_attempted': [18, 22],
        'home_off_reb': [8, 10], 'home_def_reb': [30,32], 
        'away_off_reb': [7, 9], 'away_def_reb':[28,30],
        'home_turnovers': [10, 12], 'away_turnovers': [11, 13],
        'home_ot': [0,0], 'away_ot': [0,0],

        # --- ADD THESE MISSING COLUMNS ---
        'home_3pa': [25, 30],          # Example values
        'away_3pa': [22, 28],          # Example values
        'home_ft_made': [15, 20],      # Example values
        'away_ft_made': [14, 19],      # Example values
        'home_total_reb': [38, 42],    # home_off_reb + home_def_reb
        'away_total_reb': [35, 39],    # away_off_reb + away_def_reb
        # ----------------------------------
    })

    # --- To properly test `adv_splits`, you'd need to either:
    # 1. Ensure your `nba_team_seasonal_advanced_splits` table is populated for relevant lookup seasons
    #    (e.g., 2022 if your dummy games are in the 2023-24 season and offset is -1).
    # 2. Or, create a dummy `_adv_splits_cache` and pass it directly or simulate its population.
    # For this example, we rely on the fetching logic if db_connection_for_test is live.
    # If not, `adv_splits.transform` will receive an empty DataFrame for `stats_df`.

    features_df_output = run_feature_pipeline(
        dummy_games_data,
        db_conn=db_connection_for_test,
        adv_splits_lookup_offset=-1, # Use previous season's stats
        debug=True # Enable debug logging for the smoke test run
    )

    print("\n--- Smoke Test: Generated Features DataFrame ---")
    print(f"Shape: {features_df_output.shape}")
    print(features_df_output.head())
    
    print("\nColumns generated:")
    for col in features_df_output.columns:
        if col not in dummy_games_data.columns: # Print only new columns
            print(f"  - {col}")

    # Example check for NaNs in a key new column if it exists
    if "h_rolling_score_for_mean_5" in features_df_output.columns:
        nan_count = features_df_output["h_rolling_score_for_mean_5"].isnull().sum()
        print(f"\nNaNs in 'h_rolling_score_for_mean_5': {nan_count} (expected if insufficient history for all games)")
    if "h_adv_splits_pace_home" in features_df_output.columns: # Column from refactored advanced
        nan_count_adv = features_df_output["h_adv_splits_pace_home"].isnull().sum()
        print(f"NaNs in 'h_adv_splits_pace_home': {nan_count_adv} (expected if lookup season/team missing)")