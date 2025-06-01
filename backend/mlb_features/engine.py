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
    df: pd.DataFrame,
    *,
    mlb_historical_games_df: Optional[pd.DataFrame] = None,
    mlb_historical_team_stats_df: Optional[pd.DataFrame] = None,
    season_to_lookup: Optional[int] = None,
    rolling_window_sizes: List[int] = [5, 10, 20],
    form_home_col: str = "home_current_form",
    form_away_col: str = "away_current_form",
    h2h_max_games: int = 10,
    momentum_num_innings: int = 9,
    execution_order: List[str] = DEFAULT_ORDER,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Execute the MLB feature modules in the specified order.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Starting MLB feature pipeline with debug mode ON.")

    # ←– HERE is the only change: log exactly "Input DataFrame is empty"
    if df is None or df.empty:
        logger.error("Input DataFrame is empty")
        return pd.DataFrame()

    required = ["game_id", "game_date", "home_team_id", "away_team_id"] # Changed 'game_date_et' to 'game_date'

    missing = set(required) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}. Aborting pipeline.")
        return pd.DataFrame() # Return empty DataFrame on critical error

    processed = df.copy()
    start_time = time.time()

    for module in execution_order:
        transform = TRANSFORMS.get(module)
        if not transform:
            logger.warning(f"Unknown module '{module}' in execution_order. Skipping.")
            continue

        logger.info(f"Running module: {module}")
        t0 = time.time()
        kwargs: Dict[str, Any] = {"debug": debug}

        # Common kwargs
        if module == "season":
            kwargs.update({
                "team_stats_df": mlb_historical_team_stats_df,
            })

        if module == "rolling":
            kwargs.update({
                "window_sizes": rolling_window_sizes,
            })

        if module == "form":
            kwargs.update({
                "home_form_col": form_home_col,
                "away_form_col": form_away_col,
            })

        if module == "h2h":
            kwargs.update({
                "historical_df": mlb_historical_games_df,
                "max_games": h2h_max_games,
            })

        if module == "momentum":
            kwargs.update({
                "num_innings": momentum_num_innings,
            })

        if module == "advanced":
            kwargs.update({
                "historical_team_stats_df": mlb_historical_team_stats_df,
                "season_to_lookup": season_to_lookup,
            })

        # Imputation flag
        if module in ["rolling", "season", "form", "advanced"]:
            kwargs["flag_imputations"] = flag_imputations

        try:
            processed = transform(processed, **kwargs)
            elapsed = time.time() - t0
            logger.debug(f"{module} completed in {elapsed:.2f}s; df shape: {processed.shape}")

            if processed is None or processed.empty:
                logger.error(f"Pipeline aborted: '{module}' returned empty DataFrame.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in module '{module}': {e}", exc_info=debug)
            return processed

    total_time = time.time() - start_time
    logger.info(f"Pipeline completed in {total_time:.2f}s; final shape: {processed.shape}")
    return processed
