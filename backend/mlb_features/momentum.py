# backend/mlb_features/momentum.py
"""
Calculates intra-game momentum features for MLB based on inning-by-inning scores.
Features include inning margins, cumulative run differences, margin changes,
and EWMA (Exponentially Weighted Moving Average) of inning run margins.

NOTE: As currently designed, these are INTRA-GAME features. They can only be
calculated for games where inning-by-inning scores are already present in the
input DataFrame 'df'. For future games without scores, this module will gracefully
return the DataFrame without adding these specific momentum features.
The 'historical_df' param is accepted for API compatibility but not yet used
to generate historical summary momentum for future games.
"""

from __future__ import annotations
import logging
from typing import Mapping, List, Optional # Added Optional

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
# logging.basicConfig(...) # Assuming this is handled by the calling script (engine.py)
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS, convert_and_fill
    # logger.info("Imported MLB_DEFAULTS and convert_and_fill from .utils") # Reduced verbosity
except ImportError:
    logger.warning("momentum.py: Could not import DEFAULTS or convert_and_fill from .utils; using local fallbacks.")
    MLB_DEFAULTS: Mapping[str, float] = {}

    def convert_and_fill(df: pd.DataFrame, cols: List[str], default: float = 0.0) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
            else:
                df[col] = default 
        return df


__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *, # Make subsequent arguments keyword-only
    debug: bool = False,
    defaults: Mapping[str, float] = MLB_DEFAULTS,
    num_innings: int = 9,
    span_inn6: int = 3,
    span_inn9: int = 4,
    default_fill: float = 0.0,
    historical_df: Optional[pd.DataFrame] = None, # MODIFICATION: Added to accept argument from engine.py
) -> pd.DataFrame:
    """
    Adds intra-game momentum features based on MLB inning scores present in 'df'.

    Requires 'h_inn_#' and 'a_inn_#' for innings 1..num_innings in the input 'df'.
    If these are not present (e.g., for future games in 'df'), no momentum features are added.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("MOMENTUM: Starting momentum.transform")

    # Log if historical_df is passed, even if not directly used by current intra-game logic
    if historical_df is not None and not historical_df.empty:
        logger.debug(f"MOMENTUM: Received historical_df (shape {historical_df.shape}). "
                     "Current logic calculates intra-game momentum based on input 'df'.")
    elif historical_df is not None and historical_df.empty:
         logger.debug("MOMENTUM: Received an empty historical_df.")


    if df is None or df.empty:
        logger.warning("MOMENTUM: Input DataFrame (df) is empty. Returning as is.")
        if debug: logger.setLevel(orig_level)
        return pd.DataFrame() if df is None else df.copy()

    result_df = df.copy() # Operate on a copy

    innings = range(1, num_innings + 1)
    req_cols = [f"h_inn_{i}" for i in innings] + [f"a_inn_{i}" for i in innings]
    missing_req_cols = [c for c in req_cols if c not in result_df.columns]

    if missing_req_cols:
        logger.warning(f"MOMENTUM: Missing inning score cols: {missing_req_cols}. Skipping momentum feature calculations and returning df as is.")
        if debug: 
            logger.setLevel(orig_level)
        return result_df # Return the copy of the original DataFrame

    # --- If all required columns are present in result_df, proceed: ---
    logger.debug(f"MOMENTUM: All required inning score columns found in input df (shape {result_df.shape}). Proceeding.")

    # Define all columns that this transform might create for idempotency and final default filling
    derived_cols_temp = (
        [f"inn_{i}_margin" for i in innings] +
        [f"end_inn_{i}_run_diff" for i in innings] +
        [f"inn_{i}_margin_change" for i in range(2, num_innings + 1)] 
    )
    ewma_col_final_name = f"momentum_runs_ewma_inn_{num_innings}"
    derived_cols_temp.append(ewma_col_final_name)
    ewma_col_6_name_local = "momentum_runs_ewma_inn_6"
    if num_innings >= 6:
        if ewma_col_6_name_local != ewma_col_final_name: 
             derived_cols_temp.append(ewma_col_6_name_local)
    
    all_derived_momentum_cols = sorted(list(set(derived_cols_temp)))

    # Idempotency: drop any existing derived columns
    cols_to_drop_for_idempotency = [c for c in all_derived_momentum_cols if c in result_df.columns]
    if cols_to_drop_for_idempotency:
        logger.debug(f"MOMENTUM: Dropping existing derived momentum columns: {cols_to_drop_for_idempotency}")
        result_df = result_df.drop(columns=cols_to_drop_for_idempotency, errors="ignore")

    if result_df.empty:
        logger.warning("MOMENTUM: DataFrame became empty (possibly after dropping columns). Final enforcement will add default derived columns.")
        # Fall through to final enforcement which will add columns with defaults to the (now empty) result_df
    
    # Identify rows where all original required inning scores were NaN *before* filling them.
    all_inning_scores_originally_nan_mask = result_df[req_cols].isnull().all(axis=1)

    # Fill any NaNs in the raw inning score columns using the general default_fill
    result_df = convert_and_fill(result_df, req_cols, default=default_fill)

    # 1. Per-inning margins
    for i in innings:
        result_df[f"inn_{i}_margin"] = result_df[f"h_inn_{i}"] - result_df[f"a_inn_{i}"]

    # 2. Cumulative run differentials
    if "inn_1_margin" in result_df.columns: 
        result_df["end_inn_1_run_diff"] = result_df["inn_1_margin"]
        for i in range(2, num_innings + 1):
            prev_diff_series = result_df.get(f"end_inn_{i-1}_run_diff", pd.Series(0.0, index=result_df.index))
            curr_margin_series = result_df.get(f"inn_{i}_margin", pd.Series(0.0, index=result_df.index))
            result_df[f"end_inn_{i}_run_diff"] = prev_diff_series + curr_margin_series
    else:
        logger.warning("MOMENTUM: 'inn_1_margin' not created, cannot calculate all cumulative run diffs.")


    # 3. Inning margin changes
    if num_innings >= 2:
        if "inn_1_margin" in result_df.columns and "inn_2_margin" in result_df.columns:
             result_df["inn_2_margin_change"] = result_df["inn_2_margin"] - result_df["inn_1_margin"]
        else:
            logger.debug("MOMENTUM: Not creating 'inn_2_margin_change' due to missing inn_1 or inn_2 margin.")

        for i in range(3, num_innings + 1): 
            if f"inn_{i}_margin" in result_df.columns and f"inn_{i-1}_margin" in result_df.columns:
                result_df[f"inn_{i}_margin_change"] = result_df[f"inn_{i}_margin"] - result_df[f"inn_{i-1}_margin"]
            else:
                logger.debug(f"MOMENTUM: Not creating 'inn_{i}_margin_change' due to missing current or previous inning margin.")


    # 4. EWMA momentum
    valid_margin_cols = [f"inn_{i}_margin" for i in innings if f"inn_{i}_margin" in result_df.columns]
    
    default_ewma_final = defaults.get(f"mlb_{ewma_col_final_name}", default_fill)
    default_ewma_6 = defaults.get(f"mlb_{ewma_col_6_name_local}", default_fill)

    if valid_margin_cols and not result_df[valid_margin_cols].empty:
        try:
            ewma_final_series_df = result_df[valid_margin_cols].ewm(span=span_inn9, axis=1, adjust=False, min_periods=1).mean()
            if not ewma_final_series_df.empty:
                result_df[ewma_col_final_name] = ewma_final_series_df.iloc[:, -1] 
        except Exception as e_ewma:
            logger.warning(f"MOMENTUM: Error calculating EWMA for {ewma_col_final_name}: {e_ewma}. Column will be default filled.")
            result_df[ewma_col_final_name] = np.nan # Mark for default filling

        if num_innings >= 6 and ewma_col_6_name_local in all_derived_momentum_cols:
            if len(valid_margin_cols) >= 6 and not result_df[valid_margin_cols[:6]].empty:
                try:
                    ewma6_series_df = result_df[valid_margin_cols[:6]].ewm(span=span_inn6, axis=1, adjust=False, min_periods=1).mean()
                    if not ewma6_series_df.empty:
                        result_df[ewma_col_6_name_local] = ewma6_series_df.iloc[:, -1]
                except Exception as e_ewma6:
                    logger.warning(f"MOMENTUM: Error calculating EWMA for {ewma_col_6_name_local}: {e_ewma6}. Column will be default filled.")
                    result_df[ewma_col_6_name_local] = np.nan
            else: 
                logger.debug(f"MOMENTUM: Not enough margin columns or empty slice for {ewma_col_6_name_local}. It will be default filled.")
                result_df[ewma_col_6_name_local] = np.nan # Mark for default filling
        
        # For rows where ALL input inning scores were originally NaN, EWMA should be the specific default
        if all_inning_scores_originally_nan_mask.any():
            if ewma_col_final_name in result_df.columns:
                result_df.loc[all_inning_scores_originally_nan_mask, ewma_col_final_name] = default_ewma_final
            
            if num_innings >= 6 and ewma_col_6_name_local in all_derived_momentum_cols and ewma_col_6_name_local in result_df.columns:
                result_df.loc[all_inning_scores_originally_nan_mask, ewma_col_6_name_local] = default_ewma_6

    else: 
        logger.warning("MOMENTUM: No valid inning margin columns or data empty for EWMA calculations. EWMA columns will be default filled.")
        result_df[ewma_col_final_name] = np.nan # Mark for default filling
        if num_innings >= 6 and ewma_col_6_name_local in all_derived_momentum_cols:
            result_df[ewma_col_6_name_local] = np.nan # Mark for default filling

    # Final enforcement: ensure each derived col exists, fill any remaining NaNs, enforce numeric type
    for col_name in all_derived_momentum_cols:
        specific_col_default = defaults.get(f"mlb_{col_name}", default_fill)
        if col_name not in result_df.columns:
            result_df[col_name] = specific_col_default 
        
        result_df[col_name] = pd.to_numeric(result_df[col_name], errors='coerce').fillna(specific_col_default)

    if debug:
        logger.setLevel(orig_level)
    logger.debug(f"MOMENTUM: Finished momentum.transform; output shape={result_df.shape}. Input df was shape {df.shape}.")
    return result_df