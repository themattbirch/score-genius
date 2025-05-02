# backend/features/momentum.py
"""
Calculates intra-game momentum features based on quarter-by-quarter scores.
Features include quarter margins, cumulative differences, margin changes,
and EWMA (Exponentially Weighted Moving Average) momentum proxies.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Mapping, Optional # Keep Any for DEFAULTS typing

import numpy as np
import pandas as pd

# Import necessary components from the utils module
from .utils import DEFAULTS, convert_and_fill # Import DEFAULTS and convert_and_fill

# --- Logger Configuration ---
# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly export the transform function
__all__ = ["transform"]

# -- Constants --
# EPSILON = 1e-6 # Moved to utils.py if needed globally

# -- Main Transformation Function --

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    defaults: Mapping[str, float] = DEFAULTS, # Allow passing custom defaults, fallback to global DEFAULTS
) -> pd.DataFrame:
    """
    Adds intra-game momentum features based on quarter scores and returns a new DataFrame.

    Requires columns 'home_q1', 'away_q1', ..., 'home_q4', 'away_q4'.

    Features added:
    * `q{i}_margin`: Per-quarter margin (home_qi - away_qi).
    * `end_q{n}_diff`: Running cumulative margin at the end of each quarter.
    * `q{n}_margin_change`: Change in margin compared to the previous quarter.
    * `momentum_score_ewma_q3`: EWMA of margins through Q3.
    * `momentum_score_ewma_q4`: EWMA of margins through Q4.

    Args:
        df: Input DataFrame with quarter score columns.
        debug: If True, sets logging level to DEBUG for this function call.
        defaults: A dictionary of default values to use for filling NaNs.

    Returns:
        DataFrame with added intra-game momentum features.
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for momentum.transform")

    if df is None or df.empty:
        logger.warning("momentum.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return pd.DataFrame()

    logger.debug("Adding intra-game momentum features...")
    # Work on a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # --- Input Validation and Preparation ---
    # Define required quarter score columns
    qtr_nums = range(1, 5)
    qtr_home = [f"home_q{i}" for i in qtr_nums]
    qtr_away = [f"away_q{i}" for i in qtr_nums]
    required_cols = qtr_home + qtr_away

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        logger.warning(f"Missing required quarter columns: {missing_cols}. Cannot calculate momentum features accurately. Returning original DataFrame.")
        # Optionally, fill placeholder columns with defaults if desired, but calculation is impossible.
        # For now, return the original df to signal the issue.
        if debug: logger.setLevel(current_level) # Restore logger level
        return df # Return original df

    # Ensure quarter columns are numeric and fill NaNs using the utility function
    # Use 0.0 as the default fill value for missing quarter scores
    logger.debug("Ensuring quarter columns are numeric and filling NaNs...")
    result_df = convert_and_fill(result_df, required_cols, default=0.0)

    # --- Calculate Momentum Features ---
    try:
        # Calculate quarter margins (home score - away score)
        logger.debug("Calculating quarter margins...")
        for i in qtr_nums:
            result_df[f"q{i}_margin"] = result_df[f"home_q{i}"] - result_df[f"away_q{i}"]

        # Calculate cumulative score differentials at the end of each quarter
        logger.debug("Calculating cumulative differentials...")
        result_df["end_q1_diff"] = result_df["q1_margin"]
        result_df["end_q2_diff"] = result_df["end_q1_diff"] + result_df["q2_margin"]
        result_df["end_q3_diff"] = result_df["end_q2_diff"] + result_df["q3_margin"]
        result_df["end_q4_reg_diff"] = result_df["end_q3_diff"] + result_df["q4_margin"] # End of regulation diff

        # Calculate quarter-to-quarter margin changes
        logger.debug("Calculating margin changes...")
        result_df["q2_margin_change"] = result_df["q2_margin"] - result_df["q1_margin"]
        result_df["q3_margin_change"] = result_df["q3_margin"] - result_df["q2_margin"]
        result_df["q4_margin_change"] = result_df["q4_margin"] - result_df["q3_margin"]

        # Calculate EWMA (Exponentially Weighted Moving Average) of margins as momentum proxy
        logger.debug("Calculating EWMA momentum scores...")
        q_margin_cols = [f"q{i}_margin" for i in qtr_nums] # ['q1_margin', 'q2_margin', 'q3_margin', 'q4_margin']

        # EWMA through Q4 (using all 4 quarter margins)
        # span=3 gives more weight to recent quarters
        result_df["momentum_score_ewma_q4"] = (
            result_df[q_margin_cols]
            .ewm(span=3, axis=1, adjust=False) # Calculate EWMA across columns (axis=1)
            .mean()
            .iloc[:, -1] # Select the EWMA value after the last quarter (Q4)
            .fillna(defaults.get("momentum_ewma", 0.0)) # Fill NaNs with default
        )

        # EWMA through Q3 (using first 3 quarter margins)
        # span=2 gives even more weight to recent quarters (Q2, Q3)
        result_df["momentum_score_ewma_q3"] = (
            result_df[q_margin_cols[:3]] # Select only Q1, Q2, Q3 margins
            .ewm(span=2, axis=1, adjust=False)
            .mean()
            .iloc[:, -1] # Select the EWMA value after Q3
            .fillna(defaults.get("momentum_ewma", 0.0)) # Fill NaNs with default
        )

    except Exception as e_calc:
        logger.error(f"Error during momentum calculation: {e_calc}", exc_info=debug)
        # If calculation fails, return the DataFrame state before the error
        # (or potentially fill remaining placeholders with defaults)
        logger.warning("Returning DataFrame in state before momentum calculation error.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return result_df # Or handle more gracefully by filling defaults for expected output columns

    # --- Final Logging and Return ---
    logger.debug("Finished adding intra-game momentum features.")
    logger.debug("momentum.transform: done, output shape=%s", result_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return result_df