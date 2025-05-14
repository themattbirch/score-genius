# backend/nba_features/engine.py
"""
Orchestrates the execution of the modular feature engineering pipeline.
Calls transform function of each individual feature module in sequence.
"""

from __future__ import annotations
import logging
import time
from typing import Any, List, Optional # Keep Any for DEFAULTS typing if needed

import pandas as pd
import numpy as np # Keep numpy if needed for NaN checks etc.

# Import the transform function from each feature module, aliasing for clarity
from .advanced import transform as advanced_transform
from .rolling import transform as rolling_transform
from .rest import transform as rest_transform
from .h2h import transform as h2h_transform
from .season import transform as season_transform
from .form import transform as form_transform

# Import utilities if needed (e.g., for profile_time, though removed for now)
# from .utils import profile_time, DEFAULTS # DEFAULTS might not be needed directly here

# --- Logger Configuration ---
# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly export the main pipeline function
__all__ = ["run_feature_pipeline"]

# -- Constants --
# Define the standard execution order
# Adjust this order based on actual feature dependencies if they differ from legacy
DEFAULT_EXECUTION_ORDER = [
    "advanced",
    "rest",
    "h2h",
    "form",
]

# Map module names to their transform functions
TRANSFORM_MAP = {
    "advanced": advanced_transform,
    "rest": rest_transform,
    "h2h": h2h_transform,
    "form": form_transform,
}

# -- Main Pipeline Function --
def run_feature_pipeline(
    df: pd.DataFrame,
    *,
    historical_games_df: pd.DataFrame | None = None,
    team_stats_df: pd.DataFrame | None = None,
    rolling_windows: list[int] = [5,10,20],
    h2h_window: int = 7,
    execution_order: list[str] = DEFAULT_EXECUTION_ORDER,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Runs feature transformations in the specified order.
    """
    # Adjust order if rolling is disabled
    if not rolling_windows and 'rolling' in execution_order:
        execution_order = [m for m in execution_order if m != 'rolling']

    # Setup logging level
    original_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for feature pipeline.")

    start_total = time.time()
    logger.info(f"Starting feature engineering pipeline (Order: {execution_order})...")

    # Validate input
    if df is None or df.empty:
        logger.error("Input DataFrame is empty--nothing to process.")
        if debug: logger.setLevel(original_level)
        return pd.DataFrame()
    for col in ['game_id','game_date','home_team','away_team']:
        if col not in df.columns:
            logger.error(f"Missing essential column: {col}")
            if debug: logger.setLevel(original_level)
            return pd.DataFrame()

    processed_df = df.copy()
    # Execute each transform
    for module_name in execution_order:
        if module_name not in TRANSFORM_MAP:
            logger.warning(f"Unknown module '{module_name}', skipping.")
            continue
        transform_func = TRANSFORM_MAP[module_name]
        logger.info(f"Running module: {module_name}...")
        t0 = time.time()

        # build module-specific kwargs
        kwargs = {'debug': debug}
        if module_name == "rolling":
            # rolling only needs the combined DataFrame + window sizes
            kwargs['window_sizes'] = rolling_windows
        elif module_name == "h2h":
            kwargs['historical_df'] = historical_games_df
            kwargs['max_games']     = h2h_window
        elif module_name == "season":
            kwargs['team_stats_df'] = team_stats_df

        try:
            processed_df = transform_func(processed_df, **kwargs)
            elapsed = time.time() - t0
            logger.debug(f"Module '{module_name}' completed in {elapsed:.3f}s; shape={processed_df.shape}")
            if processed_df is None or processed_df.empty:
                logger.error(f"After '{module_name}', DataFrame empty. Aborting.")
                if debug: logger.setLevel(original_level)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in module '{module_name}': {e}", exc_info=debug)
            if debug: logger.setLevel(original_level)
            return processed_df

    total_elapsed = time.time() - start_total
    logger.info(f"Pipeline completed in {total_elapsed:.2f}s. Final shape: {processed_df.shape}")
    if debug: logger.setLevel(original_level)
    return processed_df

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    logger.info("Feature engine script executed directly (usually imported).")
    # Example of how to potentially use the pipeline function
    # This requires setting up sample dataframes (df, historical_df, team_stats_df)

    # Placeholder for demonstration
    # sample_df = pd.DataFrame({
    #     'game_id': ['1', '2'],
    #     'game_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    #     'home_team': ['Team A', 'Team C'],
    #     'away_team': ['Team B', 'Team D'],
    #     'home_q1': [25, 30], 'away_q1': [20, 28],
    #     'home_q2': [28, 25], 'away_q2': [26, 27],
    #     'home_q3': [30, 29], 'away_q3': [28, 31],
    #     'home_q4': [22, 26], 'away_q4': [24, 25],
    # })
    # sample_hist_df = pd.DataFrame(...) # Needs historical data for H2H
    # sample_team_stats_df = pd.DataFrame(...) # Needs seasonal stats for Season

    # logger.info("Running pipeline with sample data (placeholders)...")
    # try:
    #     # final_features = run_feature_pipeline(
    #     #     df=sample_df,
    #     #     historical_games_df=None, # Provide sample data here
    #     #     team_stats_df=None,       # Provide sample data here
    #     #     debug=True
    #     # )
    #     # logger.info("Sample pipeline run complete.")
    #     # print(final_features.info())
    #     pass # Avoid running without actual sample data
    # except Exception as ex:
    #      logger.error(f"Error running example usage: {ex}")
    pass
