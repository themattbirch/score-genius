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
for env_path in (
    Path(__file__).resolve().parents[1] / '.env', # .../backend/.env
    Path(__file__).resolve().parents[2] / '.env', # project-root/.env
):
    if env_path.is_file():
        load_dotenv(env_path, override=True)
        break

try:
    from caching.supabase_client import supabase as supabase_client_singleton
    SUPABASE_AVAILABLE = True
except ImportError:
    supabase_client_singleton = None
    SUPABASE_AVAILABLE = False

from .game_advanced_metrics import transform as game_adv_transform
from .rest import transform as rest_transform
from .h2h import transform as h2h_transform
from .form import transform as form_transform
from .season import transform as season_transform
from .advanced import transform as adv_splits_transform # This is advanced.py
from .rolling import transform as rolling_transform

from .utils import determine_season # Ensure this utility correctly determines NBA season start year

from supabase import Client

logger = logging.getLogger(__name__)
__all__ = ['run_feature_pipeline']

DEFAULT_ORDER: List[str] = [
    'game_advanced_metrics', 'rest', 'h2h', 'form', 'season', 'adv_splits', 'rolling'
]
TRANSFORMS: Dict[str, Any] = {
    'game_advanced_metrics': game_adv_transform,
    'rest': rest_transform,
    'h2h': h2h_transform,
    'form': form_transform,
    'season': season_transform,
    'adv_splits': adv_splits_transform,
    'rolling': rolling_transform,
}

def _fetch_table_from_supabase(
    db: Client,
    table_name: str,
    filter_params: Optional[Dict[str, Any]] = None,
    select_cols: str = '*'
) -> pd.DataFrame:
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

def _fetch_adv_splits_for_season(db: Client, season_year: int) -> pd.DataFrame:
    """Fetches advanced splits for a single specified season."""
    return _fetch_table_from_supabase(
        db,
        "nba_team_seasonal_advanced_splits",
        filter_params={"season": season_year} # Assumes 'season' column in Supabase table is the start year
    )

def run_feature_pipeline(
    df: pd.DataFrame,
    *,
    db_conn: Optional[Client] = None,
    execution_order: Optional[List[str]] = None,
    h2h_lookback: int = 7,
    rolling_windows: Optional[List[int]] = None,
    adv_splits_lookup_offset: int = -1, # Default to -1 for prior season
    flag_imputations_all: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    original_logger_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"engine.run_feature_pipeline: DEBUG mode activated. Initial df shape: {df.shape}")
    
    pipeline_start_time = time.time()
    active_order = (execution_order or DEFAULT_ORDER).copy()
    effective_rolling_windows = rolling_windows if rolling_windows is not None else [5, 10, 20]
    if not effective_rolling_windows and "rolling" in active_order:
        logger.info("Rolling module disabled due to empty 'rolling_windows' list.")
        active_order.remove("rolling")
    
    logger.info(f"Effective feature module execution order: {active_order}")
    processed_df = df.copy()

    required_df_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if processed_df.empty:
        logger.warning("engine.run_feature_pipeline: Input DataFrame 'df' is empty. Aborting.")
        if debug: logger.setLevel(original_logger_level)
        return processed_df
        
    missing_input_cols = [col for col in required_df_cols if col not in processed_df.columns]
    if missing_input_cols:
        logger.error(f"engine.run_feature_pipeline: Input 'df' missing required columns: {missing_input_cols}. Aborting.")
        if debug: logger.setLevel(original_logger_level)
        return processed_df

    processed_df['game_date'] = pd.to_datetime(processed_df['game_date'], errors='coerce')
    if processed_df['game_date'].isnull().any():
        logger.warning("engine.run_feature_pipeline: 'game_date' column contains NaT values after coercion.")

    active_db_conn = db_conn if db_conn is not None else supabase_client_singleton
    if active_db_conn is None:
        logger.warning("No Supabase connection available. Data fetching for helper tables will be skipped.")

    _general_team_stats_df: Optional[pd.DataFrame] = None
    if active_db_conn and any(m in active_order for m in ['season', 'form']):
        _general_team_stats_df = _fetch_team_stats(active_db_conn)

    _historical_games_df: Optional[pd.DataFrame] = None
    if active_db_conn and 'h2h' in active_order:
        _historical_games_df = _fetch_game_stats(active_db_conn)

    # --- MODIFIED LOGIC FOR adv_splits DATA FETCHING ---
    _all_fetched_adv_splits_df: Optional[pd.DataFrame] = None
    if 'adv_splits' in active_order:
        if active_db_conn:
            if not processed_df['game_date'].dropna().empty:
                # Determine the season for which advanced stats should be looked up, for each game
                processed_df['game_nba_season_start_year'] = processed_df['game_date'].dropna().apply(
                    lambda date: int(determine_season(date).split('-')[0]) if pd.notna(date) else pd.NA
                )
                processed_df['adv_stats_lookup_season'] = (
                    pd.to_numeric(processed_df['game_nba_season_start_year'], errors='coerce') + adv_splits_lookup_offset
                ).astype('Int64') # Keep as nullable integer

                unique_lookup_seasons = processed_df['adv_stats_lookup_season'].dropna().unique().tolist()
                
                if unique_lookup_seasons:
                    logger.info(f"Need to fetch advanced splits for lookup seasons: {sorted(unique_lookup_seasons)}")
                    all_splits_list = []
                    for season_year_to_fetch in unique_lookup_seasons:
                        logger.debug(f"Fetching adv_splits for lookup season: {season_year_to_fetch}")
                        fetched_data_for_season = _fetch_adv_splits_for_season(active_db_conn, season_year_to_fetch)
                        if not fetched_data_for_season.empty:
                            # IMPORTANT: Ensure fetched_data_for_season['season'] correctly reflects season_year_to_fetch
                            # If 'season' column is already correct from Supabase, this is fine.
                            # If not, you might need to assign it: fetched_data_for_season['season_of_stats'] = season_year_to_fetch
                            all_splits_list.append(fetched_data_for_season)
                        else:
                            logger.warning(f"No advanced splits data found for lookup season: {season_year_to_fetch}")
                    
                    if all_splits_list:
                        _all_fetched_adv_splits_df = pd.concat(all_splits_list, ignore_index=True)
                        logger.info(f"Concatenated advanced splits for {len(unique_lookup_seasons)} lookup season(s). Total rows: {len(_all_fetched_adv_splits_df)}")
                        # Ensure the 'season' column in _all_fetched_adv_splits_df correctly identifies
                        # which season the stats themselves pertain to. This is crucial for advanced.py.
                        # The _fetch_adv_splits_for_season already filters by season, so the 'season' column
                        # in fetched_data_for_season should be the season_year_to_fetch.
                    else:
                        logger.warning("No advanced splits data could be fetched for any required lookup season.")
                        _all_fetched_adv_splits_df = pd.DataFrame() # Pass empty if nothing found
                else:
                    logger.warning("No valid lookup seasons determined for 'adv_splits'.")
                    _all_fetched_adv_splits_df = pd.DataFrame()
            else:
                logger.warning("Cannot determine lookup seasons for 'adv_splits': 'game_date' is empty or all NaT.")
                _all_fetched_adv_splits_df = pd.DataFrame()
        else:
            logger.warning("'adv_splits' module active, but no database connection. Will pass empty DataFrame to advanced.py.")
            _all_fetched_adv_splits_df = pd.DataFrame()
    # --- END OF MODIFIED LOGIC FOR adv_splits ---

    for module_name in active_order:
        transform_function = TRANSFORMS.get(module_name)
        if not transform_function:
            logger.warning(f"No transform function found for module '{module_name}'. Skipping.")
            continue

        module_start_time = time.time()
        logger.info(f"Running module: '{module_name}'...")

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
        elif module_name == 'form':
            current_module_kwargs.update({'team_stats_df': _general_team_stats_df})
        elif module_name == 'adv_splits':
            # Pass the DataFrame containing all fetched prior seasons' stats.
            # advanced.py will also need processed_df itself to access 'adv_stats_lookup_season' per game.
            current_module_kwargs.update({
                'all_historical_splits_df': _all_fetched_adv_splits_df if _all_fetched_adv_splits_df is not None else pd.DataFrame(),
                'flag_imputations': flag_imputations_all
                # processed_df (which now includes 'adv_stats_lookup_season') is passed as the first arg to transform
            })
        elif module_name == 'rolling':
            current_module_kwargs.update({
                'window_sizes': effective_rolling_windows,
                'flag_imputation': flag_imputations_all
            })
        
        try:
            processed_df = transform_function(processed_df, **current_module_kwargs)
            logger.debug(f"Module '{module_name}' completed in {time.time() - module_start_time:.2f}s. DF shape: {processed_df.shape}")
        except Exception as e:
            logger.error(f"Error executing module '{module_name}': {e}", exc_info=debug)
            logger.error("Feature pipeline halted due to an error.")
            if debug: logger.setLevel(original_logger_level)
            return processed_df 

        if processed_df.empty:
            logger.error(f"DataFrame became empty after module '{module_name}'. Halting pipeline.")
            break
            
    logger.info(
        f"Feature pipeline completed in {time.time() - pipeline_start_time:.2f}s. Final DataFrame shape: {processed_df.shape}"
    )
    # Clean up added columns if they are no longer needed and not meant to be output by the engine
    if 'game_nba_season_start_year' in processed_df.columns:
         processed_df.drop(columns=['game_nba_season_start_year'], inplace=True, errors='ignore')
    # 'adv_stats_lookup_season' is used by advanced.py, which then drops its own helper columns.
    # If 'adv_stats_lookup_season' is not needed downstream of advanced.py, it could also be dropped here
    # or by advanced.py itself. For now, let advanced.py manage it if it uses it.

    if debug:
        logger.setLevel(original_logger_level)
    return processed_df

# CLI Smoke Test (Example - needs adjustment if run directly due to new adv_splits logic complexity)
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | [%(name)s:%(funcName)s:%(lineno)d] – %(message)s',
    )
    logger.info("Running smoke test for feature pipeline...")
