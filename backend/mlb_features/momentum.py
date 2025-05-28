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
                df[col] = default
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

    # Define columns
    innings = range(1, num_innings + 1)
    derived_cols = (
        [f"inn_{i}_margin" for i in innings] +
        [f"end_inn_{i}_run_diff" for i in innings] +
        [f"inn_{i}_margin_change" for i in range(2, num_innings + 1)] +
        ["momentum_runs_ewma_inn_6", f"momentum_runs_ewma_inn_{num_innings}"]
    )
    # Idempotency: drop any existing derived columns
    to_drop = [c for c in derived_cols if c in df.columns]
    df_base = df.drop(columns=to_drop, errors="ignore")

    if df_base.empty:
        logger.warning("momentum.transform: empty input → returning input unchanged")
        if debug:
            logger.setLevel(orig_level)
        return df_base

    result = df_base.copy()

    # Ensure inning score columns exist
    req_cols = [f"h_inn_{i}" for i in innings] + [f"a_inn_{i}" for i in innings]
    missing = [c for c in req_cols if c not in result.columns]
    if missing:
        logger.warning(f"Missing inning score cols {missing} → skipping momentum features")
        if debug:
            logger.setLevel(orig_level)
        return result

    # Fill any raw NaNs in inning scores
    result = convert_and_fill(result, req_cols, default=default_fill)

    # 1. Per-inning margins
    for i in innings:
        result[f"inn_{i}_margin"] = result[f"h_inn_{i}"] - result[f"a_inn_{i}"]

    # 2. Cumulative run differentials
    result["end_inn_1_run_diff"] = result["inn_1_margin"]
    for i in range(2, num_innings + 1):
        result[f"end_inn_{i}_run_diff"] = (
            result[f"end_inn_{i-1}_run_diff"] + result[f"inn_{i}_margin"]
        )

    # 3. Inning margin changes
    for i in range(2, num_innings + 1):
        result[f"inn_{i}_margin_change"] = (
            result[f"inn_{i}_margin"] - result[f"inn_{i-1}_margin"]
        )

    # 4. EWMA momentum
    margin_cols = [f"inn_{i}_margin" for i in innings if f"inn_{i}_margin" in result.columns]
    if margin_cols:
        # Final inning EWMA
        ewma_final = result[margin_cols] \
            .ewm(span=span_inn9, axis=1, adjust=False, min_periods=1) \
            .mean().iloc[:, -1] \
            .fillna(defaults.get(f"mlb_momentum_runs_ewma_inn_{num_innings}", default_fill))
        result[f"momentum_runs_ewma_inn_{num_innings}"] = ewma_final

        # Inning-6 EWMA, if applicable and distinct
        if num_innings >= 6:
            ewma6_cols = margin_cols[:6]
            ewma6 = result[ewma6_cols] \
                .ewm(span=span_inn6, axis=1, adjust=False, min_periods=1) \
                .mean().iloc[:, -1] \
                .fillna(defaults.get("mlb_momentum_runs_ewma_inn_6", default_fill))
            result["momentum_runs_ewma_inn_6"] = ewma6
    else:
        logger.warning("No inning margins found → EWMA skipped")

    # Final enforcement: ensure each derived col exists, fill NaNs, enforce numeric
    for col in derived_cols:
        val = defaults.get(f"mlb_{col}", default_fill)
        if col not in result.columns:
            result[col] = val
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(val)

    if debug:
        logger.setLevel(orig_level)
    logger.debug(f"Finished momentum.transform; output shape={result.shape}")
    return result
