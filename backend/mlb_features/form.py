# backend/mlb_features/form.py

"""
Calculates form-based features for MLB games from W/L form strings.

Focus
-----
1.  Robust parsing of form strings (W/L sequences).
2.  Calculation of win percentage, current streak, and momentum direction.
3.  Handling of missing or malformed input form strings using defaults.
4.  Computation of differential features between home and away teams.

MLB Specifics:
-   Assumes input DataFrame contains columns with form strings (e.g., 'home_current_form').
-   Uses MLB-specific defaults from .utils.DEFAULTS if available.
-   The definition of "form" (e.g., L5, L10 games) is determined by the
    input form string itself.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Defaults Configuration ---
try:
    from .utils import DEFAULTS as MLB_DEFAULTS
    logger.info("Successfully imported MLB_DEFAULTS from .utils")
except ImportError:
    logger.warning("Could not import DEFAULTS from .utils for MLB form.py. Using local fallbacks.")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]

def _extract_form_metrics_single(form_string: Optional[str]) -> Dict[str, float]:
    """
    Extracts form metrics (win %, current streak, momentum) from a single form string.
    """
    # Define default metrics, trying to fetch from MLB_DEFAULTS first
    defaults: Dict[str, float] = {
        "form_win_pct": float(MLB_DEFAULTS.get("mlb_form_win_pct", 0.5)),
        "current_streak": float(MLB_DEFAULTS.get("mlb_current_streak", 0.0)),
        "momentum_direction": float(MLB_DEFAULTS.get("mlb_momentum_direction", 0.0)),
    }

    if not form_string or pd.isna(form_string) or not isinstance(form_string, str):
        logger.debug("Invalid or missing form string, returning defaults.")
        return defaults

    # Clean the string: W, L, D (Draws are rare in MLB regular season but good to sanitize if present)
    # For MLB, primarily 'W' and 'L' are expected. Other characters are ignored.
    s = "".join(filter(lambda x: x in ['W', 'L'], form_string.upper().strip()))

    if not s: # If string becomes empty after filtering
        logger.debug(f"Form string '{form_string}' became empty after sanitization, returning defaults.")
        return defaults

    length = len(s)
    wins = s.count("W")
    form_win_pct = float(wins / length) if length > 0 else defaults["form_win_pct"]

    # Current streak
    current_streak_val = 0.0
    if length > 0:
        last_char = s[-1]
        streak_count = 0
        for char_idx in range(length - 1, -1, -1): # Iterate backwards
            if s[char_idx] == last_char:
                streak_count += 1
            else:
                break
        current_streak_val = float(streak_count) if last_char == "W" else float(-streak_count)
    else: # Should not happen if `if not s:` check is effective
        current_streak_val = defaults["current_streak"]


    # Momentum direction (compare recent half vs. older half)
    momentum_direction_val = 0.0
    if length >= 4: # Need at least 4 games to compare two halves of at least 2 games each
        # Ensure integer division for split_point, favoring larger recent half if odd length
        split_point = length // 2
        
        # Correct slicing for older and recent halves
        older_half_str = s[:length - split_point] # First N-split_point characters
        recent_half_str = s[length - split_point:]   # Last split_point characters

        if recent_half_str and older_half_str: # Both halves must be non-empty
            wins_recent = recent_half_str.count("W")
            len_recent = len(recent_half_str)
            pct_recent = float(wins_recent / len_recent) if len_recent > 0 else 0.0

            wins_older = older_half_str.count("W")
            len_older = len(older_half_str)
            pct_older = float(wins_older / len_older) if len_older > 0 else 0.0
            
            if pct_recent > pct_older:
                momentum_direction_val = 1.0
            elif pct_recent < pct_older:
                momentum_direction_val = -1.0
    else:
        momentum_direction_val = defaults["momentum_direction"]


    return {
        "form_win_pct": form_win_pct,
        "current_streak": current_streak_val,
        "momentum_direction": momentum_direction_val,
    }

def transform(df: pd.DataFrame, 
              home_form_col: str = "home_current_form", 
              away_form_col: str = "away_current_form",
              **kwargs) -> pd.DataFrame:
    """
    Adds form-based features to the DataFrame.

    Args:
        df: Input DataFrame.
        home_form_col (str): Column name for home team's W/L form string.
        away_form_col (str): Column name for away team's W/L form string.
        **kwargs: Additional keyword arguments (not used in this version).

    Returns:
        DataFrame with added form features.
    """
    logger.info(f"Starting MLB form feature transformation. Input df shape: {df.shape}")
    df_out = df.copy()
    n = len(df_out)

    # Pull raw form strings (or NaN Series if column missing)
    # This allows flexibility if column names change slightly for MLB datasets
    home_raw_form = df_out.get(home_form_col, pd.Series([np.nan] * n, index=df_out.index))
    away_raw_form = df_out.get(away_form_col, pd.Series([np.nan] * n, index=df_out.index))
    
    if home_raw_form.isnull().all() and home_form_col in df_out.columns:
        logger.warning(f"Column '{home_form_col}' is present but all values are NaN.")
    elif home_form_col not in df_out.columns:
        logger.warning(f"Home form column '{home_form_col}' not found in DataFrame. Features will use defaults.")

    if away_raw_form.isnull().all() and away_form_col in df_out.columns:
        logger.warning(f"Column '{away_form_col}' is present but all values are NaN.")
    elif away_form_col not in df_out.columns:
        logger.warning(f"Away form column '{away_form_col}' not found in DataFrame. Features will use defaults.")

    # Extract per-row metrics using a list comprehension for efficiency
    logger.debug("Extracting home team form metrics...")
    home_metrics_list = [ _extract_form_metrics_single(str(x) if pd.notna(x) else None) for x in home_raw_form ]
    logger.debug("Extracting away team form metrics...")
    away_metrics_list = [ _extract_form_metrics_single(str(x) if pd.notna(x) else None) for x in away_raw_form ]

    # Build DataFrames from the lists of dictionaries and rename columns
    df_home_metrics = (
        pd.DataFrame(home_metrics_list, index=df_out.index)
        .rename(columns={
            "form_win_pct":        "home_form_win_pct",
            "current_streak":      "home_current_streak",
            "momentum_direction":  "home_momentum_direction"
        })
    )
    df_away_metrics = (
        pd.DataFrame(away_metrics_list, index=df_out.index)
        .rename(columns={
            "form_win_pct":        "away_form_win_pct",
            "current_streak":      "away_current_streak",
            "momentum_direction":  "away_momentum_direction"
        })
    )

    # Concatenate new feature DataFrames to the output DataFrame
    df_out = pd.concat([df_out, df_home_metrics, df_away_metrics], axis=1)
    logger.debug(f"Shape after adding home/away metrics: {df_out.shape}")


    # Compute differential features
    df_out["form_win_pct_diff"] = df_out["home_form_win_pct"] - df_out["away_form_win_pct"]
    df_out["streak_advantage"]   = df_out["home_current_streak"] - df_out["away_current_streak"] # Home streak - Away streak
    df_out["momentum_diff"]      = df_out["home_momentum_direction"] - df_out["away_momentum_direction"]

    # Final fillna for all created columns, ensuring they exist and have correct types/defaults.
    # This handles cases where original form strings were missing/invalid for all rows.
    # It's generally better to ensure columns are created then fill, rather than conditionally creating.

    default_form_win_pct = float(MLB_DEFAULTS.get("mlb_form_win_pct", 0.5))
    default_current_streak = float(MLB_DEFAULTS.get("mlb_current_streak", 0.0))
    default_momentum_direction = float(MLB_DEFAULTS.get("mlb_momentum_direction", 0.0))

    cols_to_fill = {
        "home_form_win_pct": default_form_win_pct, "away_form_win_pct": default_form_win_pct,
        "home_current_streak": default_current_streak, "away_current_streak": default_current_streak,
        "home_momentum_direction": default_momentum_direction, "away_momentum_direction": default_momentum_direction,
        "form_win_pct_diff": 0.0, "streak_advantage": 0.0, "momentum_diff": 0.0
    }

    for col, default_val in cols_to_fill.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(default_val)
        else: # Should not happen if concat was successful
            logger.warning(f"Expected column '{col}' not found after transformations. Creating with default.")
            df_out[col] = default_val
            
    # Ensure correct dtypes (especially for streak and momentum which should be float if they can be non-integer)
    # Win pct and momentum are inherently float. Streak can be float due to defaults.
    float_cols = [
        "home_form_win_pct", "away_form_win_pct",
        "home_current_streak", "away_current_streak",
        "home_momentum_direction", "away_momentum_direction",
        "form_win_pct_diff", "streak_advantage", "momentum_diff"
    ]
    for col in float_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(float)

    logger.info(f"MLB form feature transformation complete. Output df shape: {df_out.shape}")
    return df_out
