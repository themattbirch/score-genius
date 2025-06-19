# backend/mlb_features/momentum.py
"""
Calculates intra-game momentum features for MLB based on inning-by-inning scores
using efficient, vectorized pandas operations.
"""
from __future__ import annotations
import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("momentum.py: Could not import DEFAULTS from .utils; using local fallbacks.")
    MLB_DEFAULTS: Mapping[str, float] = {}

__all__ = ["transform"]

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None, # Kept for API compatibility with engine
    debug: bool = False,
    num_innings: int = 9,
    span_ewma: int = 4,
) -> pd.DataFrame:
    """
    Adds intra-game momentum features based on inning scores.
    Gracefully skips games where inning-by-inning scores are not available.
    """
    if debug: logger.setLevel(logging.DEBUG)
    logger.debug("Starting momentum.transform")

    result = df.copy()

    # --- 1. Guard Clause: Check for required data ---
    innings = range(1, num_innings + 1)
    h_cols = [f"h_inn_{i}" for i in innings]
    a_cols = [f"a_inn_{i}" for i in innings]
    required_cols = h_cols + a_cols

    if not all(col in result.columns for col in required_cols):
        logger.debug("Momentum: Missing required inning score columns. Skipping calculations.")
        return result

    # Convert all inning columns to numeric, filling missing scores with 0
    for col in required_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)

    # --- 2. Vectorized Calculations ---

    # Per-inning margins (Home - Away)
    margin_cols = [f"inn_{i}_margin" for i in innings]
    for i, h_col, a_col in zip(innings, h_cols, a_cols):
        result[f"inn_{i}_margin"] = result[h_col] - result[a_col]

    # Cumulative run differential at the end of each inning
    # .cumsum(axis=1) efficiently calculates the running total across columns
    cumulative_diffs = result[margin_cols].cumsum(axis=1)
    result[[f"end_inn_{i}_run_diff" for i in innings]] = cumulative_diffs

    # Change in margin from the previous inning
    # .diff(axis=1) efficiently calculates the difference between columns
    margin_changes = result[margin_cols].diff(axis=1)
    result[[f"inn_{i}_margin_change" for i in innings]] = margin_changes

    # Exponentially Weighted Moving Average (EWMA) of inning margins
    ewma_col = f"momentum_runs_ewma_inn_{num_innings}"
    try:
        ewma_series = result[margin_cols].ewm(span=span_ewma, axis=1, adjust=False, min_periods=1).mean()
        # The final EWMA is the value in the last inning's column
        result[ewma_col] = ewma_series.iloc[:, -1]
    except Exception as e:
        logger.warning(f"Momentum: EWMA calculation failed: {e}. Defaulting to 0.")
        result[ewma_col] = 0.0

    # --- 3. Final Cleanup ---
    # Fill any NaNs that might have been created (e.g., in margin_change for inning 1)
    final_feature_cols = [col for col in result.columns if "inn_" in col or "momentum_" in col and col not in df.columns]
    result[final_feature_cols] = result[final_feature_cols].fillna(0.0)

    logger.debug(f"Finished momentum.transform; output shape={result.shape}")
    return result