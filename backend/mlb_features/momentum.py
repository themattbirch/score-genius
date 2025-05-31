# backend/mlb_features/momentum.py
"""
Calculates intra-game momentum features for MLB based on inning-by-inning scores.
Features include inning margins, cumulative run differences, margin changes,
and EWMA (Exponentially Weighted Moving Average) of inning run margins.
"""

from __future__ import annotations
import logging
from typing import Mapping, List

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS, convert_and_fill
    logger.info("Imported MLB_DEFAULTS and convert_and_fill from .utils")
except ImportError:
    logger.warning("Could not import DEFAULTS or convert_and_fill; using local fallbacks")
    MLB_DEFAULTS: Mapping[str, float] = {}

    def convert_and_fill(df: pd.DataFrame, cols: List[str], default: float = 0.0) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
            else:
                df[col] = default # Create column with default if missing
        return df


__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    defaults: Mapping[str, float] = MLB_DEFAULTS,
    num_innings: int = 9,
    span_inn6: int = 3,
    span_inn9: int = 4,
    default_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Adds intra-game momentum features based on MLB inning scores.

    Requires 'h_inn_#' and 'a_inn_#' for innings 1..num_innings.

    Returns a DataFrame with new columns:
      - inn_{i}_margin
      - end_inn_{i}_run_diff
      - inn_{i}_margin_change
      - momentum_runs_ewma_inn_6
      - momentum_runs_ewma_inn_{num_innings}
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG: Starting momentum.transform")

    innings = range(1, num_innings + 1)
    
    # Define required raw input columns (inning scores)
    req_cols = [f"h_inn_{i}" for i in innings] + [f"a_inn_{i}" for i in innings]
    missing_req_cols = [c for c in req_cols if c not in df.columns]

    if missing_req_cols:
        logger.warning(f"Missing inning score cols - specifically {missing_req_cols}. Input DataFrame returned as is. Skipping momentum feature calculations.")
        if debug: 
            logger.setLevel(orig_level)
        return df.copy() # Return a copy of the original DataFrame

    # --- If all required columns are present, proceed: ---

    # Define all columns that this transform might create for idempotency and final enforcement
    derived_cols_temp = (
        [f"inn_{i}_margin" for i in innings] +
        [f"end_inn_{i}_run_diff" for i in innings] +
        [f"inn_{i}_margin_change" for i in range(2, num_innings + 1)] 
    )
    ewma_col_final_name = f"momentum_runs_ewma_inn_{num_innings}"
    derived_cols_temp.append(ewma_col_final_name)
    ewma_col_6_name_local = "momentum_runs_ewma_inn_6" # Used as a consistent key
    if num_innings >= 6:
        if ewma_col_6_name_local != ewma_col_final_name: 
             derived_cols_temp.append(ewma_col_6_name_local)
    
    derived_cols = sorted(list(set(derived_cols_temp)))

    # Idempotency: drop any existing derived columns to ensure fresh calculation
    to_drop = [c for c in derived_cols if c in df.columns]
    if to_drop:
        logger.debug(f"Dropping existing derived columns: {to_drop}")
    df_base = df.drop(columns=to_drop, errors="ignore")

    if df_base.empty: # Check if df became empty after dropping (or was empty but had req_cols)
        logger.warning("momentum.transform: DataFrame is empty after ensuring required columns and dropping existing derived columns. Processing will add default derived columns.")
        # If df_base is empty, result will be empty. Calculations might not run as expected.
        # The final loop will add derived_cols with defaults.
        result = df_base.copy()
        # Fall through to final enforcement loop to populate derived_cols with defaults
    else:
        result = df_base.copy()


    # Identify rows where all original required inning scores were NaN *before* filling them.
    # This is done after we've confirmed all req_cols exist.
    originally_all_nan_mask = result[req_cols].isnull().all(axis=1)

    # Fill any raw NaNs in inning scores using the general default_fill
    result = convert_and_fill(result, req_cols, default=default_fill)

    # 1. Per-inning margins
    for i in innings:
        result[f"inn_{i}_margin"] = result[f"h_inn_{i}"] - result[f"a_inn_{i}"]

    # 2. Cumulative run differentials
    if f"inn_1_margin" in result.columns : 
        result["end_inn_1_run_diff"] = result["inn_1_margin"]
        for i in range(2, num_innings + 1):
            result[f"end_inn_{i}_run_diff"] = (
                result.get(f"end_inn_{i-1}_run_diff", 0.0) + result.get(f"inn_{i}_margin", 0.0)
            )
    # else block for missing 'inn_1_margin' handled by final enforcement if column doesn't get created

    # 3. Inning margin changes
    if num_innings >= 2 :
        if "inn_1_margin" in result.columns and "inn_2_margin" in result.columns:
             result["inn_2_margin_change"] = result["inn_2_margin"] - result["inn_1_margin"]
        # else this column might not be created, final enforcement will handle it.
        for i in range(3, num_innings + 1): 
            if f"inn_{i}_margin" in result.columns and f"inn_{i-1}_margin" in result.columns:
                result[f"inn_{i}_margin_change"] = (
                    result[f"inn_{i}_margin"] - result[f"inn_{i-1}_margin"]
                )
            # else this column might not be created for this i, final enforcement.

    # 4. EWMA momentum
    margin_cols = [f"inn_{i}_margin" for i in innings if f"inn_{i}_margin" in result.columns]
    
    default_ewma_final = defaults.get(f"mlb_{ewma_col_final_name}", default_fill)
    default_ewma_6 = defaults.get(f"mlb_{ewma_col_6_name_local}", default_fill)

    if margin_cols and not result[margin_cols].empty: # Check if result[margin_cols] is not empty
        # Final inning EWMA
        ewma_final_series = result[margin_cols].ewm(span=span_inn9, axis=1, adjust=False, min_periods=1).mean()
        if not ewma_final_series.empty:
            result[ewma_col_final_name] = ewma_final_series.iloc[:, -1].fillna(default_ewma_final)
        else: # Should not happen if result[margin_cols] not empty
            result[ewma_col_final_name] = default_ewma_final

        # Inning-6 EWMA
        if num_innings >= 6 and ewma_col_6_name_local in derived_cols:
            if len(margin_cols) >= 6 and not result[margin_cols[:6]].empty:
                ewma6_series = result[margin_cols[:6]].ewm(span=span_inn6, axis=1, adjust=False, min_periods=1).mean()
                if not ewma6_series.empty:
                    result[ewma_col_6_name_local] = ewma6_series.iloc[:, -1].fillna(default_ewma_6)
                else:
                    result[ewma_col_6_name_local] = default_ewma_6
            else: # Not enough margin columns for inn_6_ewma or empty slice
                result[ewma_col_6_name_local] = default_ewma_6
        
        # Override for rows that were originally all NaNs
        if originally_all_nan_mask.any():
            if ewma_col_final_name in result.columns:
                result.loc[originally_all_nan_mask, ewma_col_final_name] = default_ewma_final
            
            if num_innings >= 6 and ewma_col_6_name_local in derived_cols and ewma_col_6_name_local in result.columns:
                 result.loc[originally_all_nan_mask, ewma_col_6_name_local] = default_ewma_6
    else: # margin_cols is empty or result[margin_cols] was empty
        logger.warning("No valid inning margins columns found or data was empty; EWMA calculations will use defaults.")
        result[ewma_col_final_name] = default_ewma_final
        if num_innings >= 6 and ewma_col_6_name_local in derived_cols:
            result[ewma_col_6_name_local] = default_ewma_6

    # Final enforcement: ensure each derived col exists, fill any remaining NaNs, enforce numeric type
    for col_name in derived_cols:
        specific_default = defaults.get(f"mlb_{col_name}", default_fill)
        if col_name not in result.columns:
            result[col_name] = specific_default # Create column if unexpectedly missing
        
        # Ensure column is numeric and fill NaNs
        # NaNs here could be from operations on empty df_base or if specific features couldn't be calculated
        result[col_name] = pd.to_numeric(result[col_name], errors="coerce").fillna(specific_default)

    if debug:
        logger.setLevel(orig_level)
    logger.debug(f"Finished momentum.transform; output shape={result.shape}")
    return result