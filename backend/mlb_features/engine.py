# backend/mlb_features/engine.py

"""
Orchestrates the MLB feature engineering pipeline by sequentially applying
transform functions from each feature module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .rest import transform as rest_transform
from .season import transform as season_transform
from .rolling import transform as rolling_transform
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .momentum import transform as momentum_transform
from .advanced import transform as advanced_transform

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# Default execution order
DEFAULT_ORDER: List[str] = [
    "rest",
    "season",
    "rolling",
    "form",
    "h2h",
    "momentum",
    "advanced",
]

# Map module names to transform functions
TRANSFORMS: Dict[str, Any] = {
    "rest": rest_transform,
    "season": season_transform,
    "rolling": rolling_transform,
    "form": form_transform,
    "h2h": h2h_transform,
    "momentum": momentum_transform,
    "advanced": advanced_transform,
}


def run_mlb_feature_pipeline(
    df: pd.DataFrame,  # Main game DataFrame, can span multiple seasons
    *,
    mlb_historical_games_df: Optional[pd.DataFrame] = None,
    mlb_historical_team_stats_df: Optional[pd.DataFrame] = None, # Full historical team stats (multi-season)
    rolling_window_sizes: List[int] = [15, 30, 60, 100], # Example defaults
    form_home_col: str = "home_current_form",
    form_away_col: str = "away_current_form",
    h2h_max_games: int = 10,
    momentum_num_innings: int = 9,
    execution_order: List[str] = DEFAULT_ORDER, # Make sure DEFAULT_ORDER is defined in this file
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Execute the MLB feature modules, processing game data season by season
    to ensure advanced.py uses the correct preceding season's historical stats.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Starting MLB feature pipeline with debug mode ON. Initial df shape: {df.shape}")

    if df is None or df.empty:
        logger.error("Input DataFrame (df) for game data is empty in run_mlb_feature_pipeline.")
        return pd.DataFrame()

    # Ensure 'season' column exists in the main game DataFrame and is numeric
    if 'season' not in df.columns:
        logger.error("'season' column is required in the input game DataFrame (df) for per-season processing. Aborting.")
        return df.copy()
    try:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')
        df.dropna(subset=['season'], inplace=True)
        if df.empty:
            logger.error("Input DataFrame (df) is empty after 'season' column processing. Aborting.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing 'season' column in input game DataFrame (df): {e}. Aborting", exc_info=True)
        return pd.DataFrame()

    # Prepare the full mlb_historical_team_stats_df (ensure its 'season' column is numeric)
    if mlb_historical_team_stats_df is not None and not mlb_historical_team_stats_df.empty:
        if 'season' in mlb_historical_team_stats_df.columns:
            mlb_historical_team_stats_df['season'] = pd.to_numeric(mlb_historical_team_stats_df['season'], errors='coerce').astype('Int64')
            mlb_historical_team_stats_df.dropna(subset=['season'], inplace=True)
        else:
            logger.warning("'season' column missing in mlb_historical_team_stats_df. Advanced historical stats may be incorrect or use defaults.")
            # Treat as empty if season column is vital and missing for filtering
            mlb_historical_team_stats_df = pd.DataFrame() 
    else:
        logger.warning("mlb_historical_team_stats_df is None or empty. Advanced stats will likely use defaults.")
        mlb_historical_team_stats_df = pd.DataFrame() # Ensure it's an empty DataFrame if None

    all_processed_chunks = []
    unique_game_seasons = sorted(df['season'].unique())
    
    logger.info(f"Feature pipeline will process game data for seasons: {unique_game_seasons}")
    pipeline_overall_start_time = time.time()

    for current_game_season in unique_game_seasons:
        logger.info(f"--- Processing feature engineering for games in season: {current_game_season} ---")
        df_season_chunk = df[df['season'] == current_game_season].copy()
        
        if df_season_chunk.empty:
            logger.warning(f"No game data for season {current_game_season} after initial chunking. Skipping.")
            continue
            
        historical_stats_lookup_season = current_game_season - 1
        logger.info(f"For game season {current_game_season}, the 'advanced' module will target historical stats from season: {historical_stats_lookup_season}")

        filtered_hist_stats_for_advanced = pd.DataFrame() # Default to empty DF
        if not mlb_historical_team_stats_df.empty and 'season' in mlb_historical_team_stats_df.columns:
            filtered_hist_stats_for_advanced = mlb_historical_team_stats_df[
                mlb_historical_team_stats_df['season'] == historical_stats_lookup_season
            ].copy()
            if filtered_hist_stats_for_advanced.empty:
                logger.warning(f"For game season {current_game_season}, no historical team stats found for lookup season {historical_stats_lookup_season}. 'advanced' module might use defaults.")
        
        processed_chunk_for_this_season = df_season_chunk.copy() 
        chunk_pipeline_successful = True

        for module_key_name in execution_order: # Ensure execution_order is defined (e.g., DEFAULT_ORDER)
            transform_function_to_call = TRANSFORMS.get(module_key_name) # Ensure TRANSFORMS is defined
            if not transform_function_to_call:
                logger.warning(f"Unknown module '{module_key_name}' in execution_order. Skipping for game season {current_game_season}.")
                continue

            logger.info(f"Running module: '{module_key_name}' for game season: {current_game_season}")
            
            module_start_time = time.time()
            current_kwargs: Dict[str, Any] = {"debug": debug}

            if module_key_name == "season":
                current_kwargs.update({"team_stats_df": mlb_historical_team_stats_df}) # season.py might need broader historical access
            elif module_key_name == "rolling":
                current_kwargs.update({"window_sizes": rolling_window_sizes})
            elif module_key_name == "form":
                current_kwargs.update({"home_form_col": form_home_col, "away_form_col": form_away_col})
            elif module_key_name == "h2h":
                current_kwargs.update({"historical_df": mlb_historical_games_df, "max_games": h2h_max_games})
            elif module_key_name == "momentum":
                current_kwargs.update({"num_innings": momentum_num_innings})
            elif module_key_name == "advanced": # Make sure this string matches your DEFAULT_ORDER and TRANSFORMS keys
                current_kwargs.update({
                    "historical_team_stats_df": filtered_hist_stats_for_advanced, # CRITICAL: Use S-1 filtered data
                    "home_team_col_param": "home_team_id",
                    "away_team_col_param": "away_team_id",
                    "home_hand_col_param": "home_starter_pitcher_handedness",
                    "away_hand_col_param": "away_starter_pitcher_handedness",
                })
            
            if module_key_name in ["rolling", "season", "form", "advanced"]: # Adjust if other modules take this
                current_kwargs["flag_imputations"] = flag_imputations
            
            try:
                processed_chunk_for_this_season = transform_function_to_call(processed_chunk_for_this_season, **current_kwargs)
                logger.debug(f"Module '{module_key_name}' for game season {current_game_season} completed in {time.time() - module_start_time:.2f}s; chunk shape: {None if processed_chunk_for_this_season is None else processed_chunk_for_this_season.shape}")
            except Exception as e:
                 logger.error(f"Error in module '{module_key_name}' for game season {current_game_season}: {e}", exc_info=debug)
                 chunk_pipeline_successful = False
                 break 
            if processed_chunk_for_this_season is None or processed_chunk_for_this_season.empty:
                 logger.error(f"Module '{module_key_name}' returned empty DataFrame for game season {current_game_season}. Aborting processing for this chunk.")
                 chunk_pipeline_successful = False
                 break
        
        if chunk_pipeline_successful and not processed_chunk_for_this_season.empty:
            all_processed_chunks.append(processed_chunk_for_this_season)
        else:
            logger.warning(f"Feature engineering for game season {current_game_season} was not fully successful or resulted in an empty chunk. This chunk will be excluded.")

    if not all_processed_chunks:
        logger.error("No data chunks successfully processed across any input season. Returning an empty DataFrame.")
        return pd.DataFrame(columns=df.columns if not df.empty else None) # Return empty with original columns if possible
    
    final_df = pd.concat(all_processed_chunks, ignore_index=True, sort=False)
    logger.info(f"Full feature pipeline completed. Processed {len(all_processed_chunks)} season chunk(s). Final combined DataFrame shape: {final_df.shape}. Total time: {time.time() - pipeline_overall_start_time:.2f}s")
    return final_df
