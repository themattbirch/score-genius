# backend/features/form.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional # Keep Any for DEFAULTS typing

import numpy as np
import pandas as pd

# Import necessary components from the utils module
from .utils import DEFAULTS # Import DEFAULTS

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

# -- Helper Function --

def _extract_form_metrics_single(form_string: Optional[str]) -> Dict[str, float]:
    """
    Extracts metrics (win_pct, current_streak, momentum_direction)
    from a team's recent form string (e.g., 'WWLWL').

    Args:
        form_string: The form string (e.g., 'WWLLW'). Should contain 'W' or 'L'.

    Returns:
        A dictionary containing 'form_win_pct', 'current_streak', and 'momentum_direction'.
    """
    # Define default values using the imported DEFAULTS constant
    defaults: Dict[str, float] = {
        'form_win_pct': DEFAULTS.get('form_win_pct', 0.5),
        'current_streak': DEFAULTS.get('current_streak', 0),
        'momentum_direction': DEFAULTS.get('momentum_direction', 0.0),
    }

    # Handle invalid input
    if not form_string or pd.isna(form_string) or not isinstance(form_string, str):
        return defaults

    # Clean the input string: uppercase, remove spaces, hyphens, question marks
    s = form_string.upper().strip().replace('-', '').replace('?', '')
    # Return defaults if the string is empty or 'N/A' after cleaning
    if not s or s == 'N/A':
        return defaults

    length = len(s)
    wins = s.count('W')
    # Calculate win percentage based on the form string length
    form_win_pct = float(wins / length) if length > 0 else defaults['form_win_pct']

    # Calculate Current Streak
    current_streak = 0.0 # Use float for consistency, convert to int later if needed
    if length > 0:
        last_char = s[-1] # Get the result of the most recent game
        streak_count = 0
        # Count consecutive identical results from the end of the string
        for ch in reversed(s):
            if ch == last_char:
                streak_count += 1
            else:
                break # Stop counting when the result changes
        # Streak is positive for wins, negative for losses
        current_streak = float(streak_count) if last_char == 'W' else float(-streak_count)

    # Calculate Momentum Direction (comparing recent half vs older half)
    momentum_direction = 0.0
    if length >= 4: # Need at least 4 games to compare halves
        split_point = length // 2 # Integer division to find midpoint
        # Slice the string into recent and older halves
        recent_half = s[-split_point:]
        older_half = s[:length - split_point]

        # Ensure both halves have games to compare
        if recent_half and older_half:
            len_r, len_o = len(recent_half), len(older_half)
            # Calculate win percentages for each half
            pct_r = float(recent_half.count('W') / len_r) if len_r > 0 else 0.0
            pct_o = float(older_half.count('W') / len_o) if len_o > 0 else 0.0
            # Determine momentum direction based on comparison
            if pct_r > pct_o:
                momentum_direction = 1.0 # Improving form
            elif pct_r < pct_o:
                momentum_direction = -1.0 # Declining form
            # else: momentum_direction remains 0.0 (stable form)

    # Return the calculated metrics
    return {
        'form_win_pct': form_win_pct,
        'current_streak': current_streak, # Keep as float, finalize type later
        'momentum_direction': momentum_direction
    }

# -- Main Transformation Function --

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Calculates features based on team form strings (e.g., 'WWLWL') found in
    'home_current_form' and 'away_current_form' columns.

    Adds columns for form win percentage, current streak, momentum direction
    for both home and away teams, and calculates difference features.

    Args:
        df: Input DataFrame, expected to have 'home_current_form' and
            'away_current_form' columns.
        debug: If True, sets logging level to DEBUG for this function call.

    Returns:
        DataFrame with added form-based feature columns.
    """
    # Set logger level based on debug flag for this specific call
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for form.transform")

    if df is None or df.empty:
        logger.warning("form.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug: logger.setLevel(current_level) # Restore logger level
        return pd.DataFrame()

    logger.debug("Adding form string derived features...")
    result_df = df.copy() # Work on a copy

    # Define the keys for the metrics calculated by the helper function
    form_metric_keys = list(_extract_form_metrics_single("").keys()) # ['form_win_pct', 'current_streak', 'momentum_direction']
    # Define expected output column names
    home_cols = [f"home_{k}" for k in form_metric_keys] # ['home_form_win_pct', ...]
    away_cols = [f"away_{k}" for k in form_metric_keys] # ['away_form_win_pct', ...]
    diff_cols = ["form_win_pct_diff", "streak_advantage", "momentum_diff"]
    # All columns this function aims to add or update
    placeholder_cols = home_cols + away_cols + diff_cols

    # Input columns expected in the DataFrame
    home_form_col = "home_current_form"
    away_form_col = "away_current_form"

    # --- Calculate Form Metrics ---
    # Check if required input columns exist
    if home_form_col not in result_df.columns or away_form_col not in result_df.columns:
        logger.warning(f"Missing one or both required form columns ('{home_form_col}', '{away_form_col}'). Form metrics will be filled with defaults.")
        # Ensure placeholder columns exist even if calculation is skipped
        for col in placeholder_cols:
             if col not in result_df.columns: result_df[col] = np.nan # Fill with NaN initially
    else:
        # Proceed with calculation if columns exist
        try:
            logger.debug("Processing form strings...")
            # Clean the form string columns
            result_df[home_form_col] = (
                result_df[home_form_col]
                .fillna("") # Replace NaN with empty string
                .astype(str)
                .replace("N/A", "") # Replace 'N/A' placeholder
            )
            result_df[away_form_col] = (
                result_df[away_form_col]
                .fillna("")
                .astype(str)
                .replace("N/A", "")
            )

            # Apply the helper function to extract metrics for home and away teams
            home_metrics_list = result_df[home_form_col].apply(_extract_form_metrics_single)
            away_metrics_list = result_df[away_form_col].apply(_extract_form_metrics_single)

            # Convert the list of dictionaries into DataFrames and join them back
            # Add 'home_' prefix to columns from home_metrics
            result_df = result_df.join(
                pd.DataFrame(home_metrics_list.tolist(), index=result_df.index)
                .add_prefix("home_")
            )
            # Add 'away_' prefix to columns from away_metrics
            result_df = result_df.join(
                pd.DataFrame(away_metrics_list.tolist(), index=result_df.index)
                .add_prefix("away_")
            )

            logger.debug("Calculating form difference features...")
            # Calculate difference features using the newly added columns
            # Use .get() with DEFAULTS fallback in case join failed or columns are missing
            result_df["form_win_pct_diff"] = (
                result_df.get("home_form_win_pct", DEFAULTS.get("form_win_pct", 0.5)) -
                result_df.get("away_form_win_pct", DEFAULTS.get("form_win_pct", 0.5))
            )
            result_df["streak_advantage"] = (
                result_df.get("home_current_streak", DEFAULTS.get("current_streak", 0)) -
                result_df.get("away_current_streak", DEFAULTS.get("current_streak", 0))
            )
            result_df["momentum_diff"] = (
                result_df.get("home_momentum_direction", DEFAULTS.get("momentum_direction", 0.0)) -
                result_df.get("away_momentum_direction", DEFAULTS.get("momentum_direction", 0.0))
            )

            logger.debug("Successfully calculated form metrics and diffs.")

        except Exception as e:
            logger.error(f"Error processing form strings or calculating diffs: {e}", exc_info=debug)
            # Ensure placeholder columns exist even if calculation fails
            for col in placeholder_cols:
                if col not in result_df.columns: result_df[col] = np.nan # Fill with NaN initially

    # --- Final Processing (Filling NaNs and Enforcing Types) ---
    logger.debug("Finalizing form features (filling defaults/types)...")

    for col in placeholder_cols:
        base_key = col.replace("home_", "").replace("away_", "")
        diff_key = base_key.replace("_diff", "").replace("_advantage", "")

        # Decide the default value
        if "_diff" in base_key or "_advantage" in base_key:
            default_val = 0.0                         # difference/advantage ⇒ 0
        elif diff_key == "form_win_pct":
            default_val = DEFAULTS.get("form_win_pct", 0.5)
        elif diff_key == "current_streak":
            default_val = DEFAULTS.get("current_streak", 0)
        elif diff_key == "momentum_direction":
            default_val = DEFAULTS.get("momentum_direction", 0.0)
        else:
            default_val = DEFAULTS.get(diff_key, 0.0)  # fallback

        # Ensure column exists and fill NaNs
        if col not in result_df.columns:
            logger.warning(
                f"Column '{col}' missing before final fill. Adding with default: {default_val}"
            )
            result_df[col] = default_val
        else:
            result_df[col] = result_df[col].fillna(default_val)

        # Numeric enforcement
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(default_val)

        # Cast streak columns (not advantages) to int
        if "streak" in col and "advantage" not in col:
            result_df[col] = result_df[col].round().astype(int)

    logger.debug("Finished adding form features.")
    logger.debug("form.transform: done, output shape=%s", result_df.shape)

    # Restore original logger level if it was changed
    if debug: logger.setLevel(current_level)

    return result_df