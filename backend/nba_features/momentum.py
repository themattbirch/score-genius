# backend/nba_features/momentum.py
"""
Calculates intra-game momentum features based on quarter-by-quarter scores.
Features include quarter margins, cumulative differences, margin changes,
and EWMA (Exponentially Weighted Moving Average) momentum proxies.
"""

from __future__ import annotations
import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULTS, convert_and_fill

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Default fallback values for momentum features
_MOMENTUM_DEFAULTS: Mapping[str, float] = {
    "momentum_score_ewma_q3": DEFAULTS.get("momentum_score_ewma_q3", 0.0),
    "momentum_score_ewma_q4": DEFAULTS.get("momentum_score_ewma_q4", 0.0),
    "std_dev_q_margins":     DEFAULTS.get("std_dev_q_margins", 0.0),
}


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
    Add intra-game momentum features based on quarter scores.

    Features:
      - q{i}_margin          : home_qi - away_qi for i in 1..4
      - end_q{i}_diff        : cumulative margin at end of quarter i for i in 1..4
      - q{i}_margin_change   : q{i}_margin - q{i-1}_margin for i in 2..4
      - momentum_score_ewma_q3: EWMA over q1..q3 margins (span=span_q3)
      - momentum_score_ewma_q4: EWMA over q1..q4 margins (span=span_q4)
      - std_dev_q_margins    : standard deviation of q1..q4 margins

    Relies on columns 'home_q1'..'home_q4', 'away_q1'..'away_q4'.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Drop any pre-existing derived columns
        margins = [f"q{i}_margin" for i in range(1, 5)]
        ends    = [f"end_q{i}_reg_diff" for i in range(1, 5)]
        changes = [f"q{i}_margin_change" for i in range(2, 5)]
        extras  = ["momentum_score_ewma_q3", "momentum_score_ewma_q4", "std_dev_q_margins"]
        derived = margins + ends + changes + extras
        base = df.drop(columns=[c for c in derived if c in df.columns], errors="ignore").copy()

        if base.empty:
            logger.warning("momentum.transform: input empty after dropping derived cols.")
            return base

        # Required raw columns
        quarters = range(1, 5)
        raw_cols: Sequence[str] = [f"home_q{i}" for i in quarters] + [f"away_q{i}" for i in quarters]
        missing = [c for c in raw_cols if c not in base.columns]
        if missing:
            logger.warning(f"momentum.transform: missing quarter cols {missing}, skipping momentum.")
            return base

        # Fill raw NaNs
        df_work = convert_and_fill(base, raw_cols, default=default_fill)

        # Compute per-quarter margins
        for i in quarters:
            df_work[f"q{i}_margin"] = df_work[f"home_q{i}"] - df_work[f"away_q{i}"]

        # Compute cumulative end-of-quarter diffs
        # Compute cumulative end-of-quarter diffs (regular-season)
        df_work["end_q1_reg_diff"] = df_work["q1_margin"]
        for i in range(2, 5):
            prev = f"end_q{i-1}_reg_diff"
            curr = f"q{i}_margin"
            df_work[f"end_q{i}_reg_diff"] = df_work[prev] + df_work[curr]

        # Compute quarter-to-quarter margin changes
        for i in range(2, 5):
            df_work[f"q{i}_margin_change"] = (
                df_work[f"q{i}_margin"] - df_work[f"q{i-1}_margin"]
            )

        # EWMA momentum: Q1..Q4
        margin_cols = [f"q{i}_margin" for i in quarters]
        df_work["momentum_score_ewma_q4"] = (
            df_work[margin_cols]
                .ewm(span=span_q4, axis=1, adjust=False)
                .mean()
                .iloc[:, -1]
        )
        df_work["momentum_score_ewma_q4"].fillna(
            defaults.get("momentum_score_ewma_q4", default_fill), inplace=True
        )

        # EWMA momentum: Q1..Q3
        df_work["momentum_score_ewma_q3"] = (
            df_work[margin_cols[:3]]
                .ewm(span=span_q3, axis=1, adjust=False)
                .mean()
                .iloc[:, -1]
        )
        df_work["momentum_score_ewma_q3"].fillna(
            defaults.get("momentum_score_ewma_q3", default_fill), inplace=True
        )

        # Standard deviation of quarter margins
        df_work["std_dev_q_margins"] = df_work[margin_cols].std(axis=1)
        df_work["std_dev_q_margins"].fillna(
            defaults.get("std_dev_q_margins", default_fill), inplace=True
        )

    except Exception as e:
        logger.error("momentum.transform failed: %s", e, exc_info=debug)
        return base

    finally:
        if debug:
            logger.setLevel(orig_level)

    return df_work