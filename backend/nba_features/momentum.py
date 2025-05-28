# backend/nba_features/momentum.py
"""
Calculates intra-game momentum features based on quarter-by-quarter scores.
Features include quarter margins, cumulative differences, margin changes,
and EWMA (Exponentially Weighted Moving Average) momentum proxies.
"""

from __future__ import annotations
import logging
from typing import Mapping

import numpy as np
import pandas as pd

from .utils import DEFAULTS, convert_and_fill

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    defaults: Mapping[str, float] = DEFAULTS,
    span_q3: int = 2,
    span_q4: int = 3,
    default_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Adds intra-game momentum features based on quarter scores and returns a new DataFrame.

    Requires columns 'home_q1', 'away_q1', ..., 'home_q4', 'away_q4'.

    Features added:
      - q{i}_margin: Per-quarter margin (home_qi - away_qi).
      - end_q{n}_diff: Cumulative margin at end of each quarter (n = 1..3).
      - end_q4_reg_diff: Cumulative margin at end of Q4 (regulation).
      - q{n}_margin_change: Change vs. previous quarter (n = 2..4).
      - momentum_score_ewma_q3: EWMA of Q1–Q3 margins (span=span_q3).
      - momentum_score_ewma_q4: EWMA of Q1–Q4 margins (span=span_q4).

    Args:
        df: Input DataFrame.
        debug: If True, enable DEBUG logging for this call.
        defaults: Mapping of default fallback values.
        span_q3: EWMA span for Q3 momentum.
        span_q4: EWMA span for Q4 momentum.
        default_fill: Value to use when filling raw NaNs.

    Returns:
        DataFrame with added momentum features.
    """
    # adjust logging level if requested
    original_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for momentum.transform")

    # drop any existing momentum features so we start clean
    derived_cols = (
        [f"q{i}_margin" for i in range(1, 5)]
        + [f"end_q{i}_diff" for i in range(1, 4)]
        + ["end_q4_reg_diff"]
        + [f"q{i}_margin_change" for i in range(2, 5)]
        + ["momentum_score_ewma_q3", "momentum_score_ewma_q4"]
    )
    df_base = df.drop(columns=[c for c in derived_cols if c in df.columns], errors="ignore")

    if df_base is None or df_base.empty:
        logger.warning("momentum.transform: Input DataFrame is empty. Returning empty DataFrame.")
        if debug:
            logger.setLevel(original_level)
        return pd.DataFrame()

    result_df = df_base.copy()

    # required columns
    quarters = range(1, 5)
    required = [f"home_q{i}" for i in quarters] + [f"away_q{i}" for i in quarters]
    missing = [c for c in required if c not in result_df.columns]
    if missing:
        logger.warning(f"Missing required quarter columns {missing}. Returning original DataFrame.")
        if debug:
            logger.setLevel(original_level)
        return df_base

    # ensure numeric and fill raw NaNs
    logger.debug("Filling raw NaNs in quarter columns with %s...", default_fill)
    result_df = convert_and_fill(result_df, required, default=default_fill)

    try:
        # quarter margins
        logger.debug("Calculating quarter margins...")
        for i in quarters:
            result_df[f"q{i}_margin"] = result_df[f"home_q{i}"] - result_df[f"away_q{i}"]

        # cumulative diffs
        logger.debug("Calculating cumulative differentials...")
        result_df["end_q1_diff"] = result_df["q1_margin"]
        result_df["end_q2_diff"] = result_df["end_q1_diff"] + result_df["q2_margin"]
        result_df["end_q3_diff"] = result_df["end_q2_diff"] + result_df["q3_margin"]
        result_df["end_q4_reg_diff"] = result_df["end_q3_diff"] + result_df["q4_margin"]

        # margin changes
        logger.debug("Calculating margin changes...")
        result_df["q2_margin_change"] = result_df["q2_margin"] - result_df["q1_margin"]
        result_df["q3_margin_change"] = result_df["q3_margin"] - result_df["q2_margin"]
        result_df["q4_margin_change"] = result_df["q4_margin"] - result_df["q3_margin"]

        # EWMA momentum
        q_margins = [f"q{i}_margin" for i in quarters]
        logger.debug("Calculating EWMA momentum (span_q4=%d)...", span_q4)
        result_df["momentum_score_ewma_q4"] = (
            result_df[q_margins]
            .ewm(span=span_q4, axis=1, adjust=False)
            .mean()
            .iloc[:, -1]
            .fillna(defaults.get("momentum_score_ewma_q4", default_fill))
        )

        logger.debug("Calculating EWMA momentum (span_q3=%d)...", span_q3)
        result_df["momentum_score_ewma_q3"] = (
            result_df[q_margins[:3]]
            .ewm(span=span_q3, axis=1, adjust=False)
            .mean()
            .iloc[:, -1]
            .fillna(defaults.get("momentum_score_ewma_q3", default_fill))
        )

    except Exception as e:
        logger.error("Error in momentum.transform: %s", e, exc_info=debug)
        logger.warning("Returning DataFrame state before error.")
        if debug:
            logger.setLevel(original_level)
        return result_df

    logger.debug("Finished momentum.transform; output shape=%s", result_df.shape)

    if debug:
        logger.setLevel(original_level)

    return result_df