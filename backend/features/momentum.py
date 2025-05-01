# backend/features/momentum.py
"""
Adds intra-game momentum features (quarter margins, cumulative
diffs, EWMA momentum proxies). Thin wrapper around pure-Python logic
so it can be hot-swapped in the modular pipeline.
"""

from __future__ import annotations
import logging
from typing import Mapping, Dict

import pandas as pd

from .utils import DEFAULTS, convert_and_fill

logger = logging.getLogger(__name__)
__all__ = ["transform"]


# --------------------------------------------------------------------------- #
#  core helper (mutates df in-place)
# --------------------------------------------------------------------------- #
def _add_intra_game_momentum(df: pd.DataFrame, defaults: Dict[str, float]) -> None:
    """
    Mutates *df* in-place ― expects home_q{i}/away_q{i} (i=1-4) to be present
    and numeric.
    """
    # quarter margins
    for i in range(1, 5):
        df[f"q{i}_margin"] = df[f"home_q{i}"] - df[f"away_q{i}"]

    # cumulative diffs
    df["end_q1_diff"] = df["q1_margin"]
    df["end_q2_diff"] = df["end_q1_diff"] + df["q2_margin"]
    df["end_q3_diff"] = df["end_q2_diff"] + df["q3_margin"]
    df["end_q4_reg_diff"] = df["end_q3_diff"] + df["q4_margin"]

    # inter-quarter changes
    df["q2_margin_change"] = df["q2_margin"] - df["q1_margin"]
    df["q3_margin_change"] = df["q3_margin"] - df["q2_margin"]
    df["q4_margin_change"] = df["q4_margin"] - df["q3_margin"]

    # EWMA momentum
    qcols = [f"q{i}_margin" for i in range(1, 5)]

    df["momentum_score_ewma_q4"] = (
        df[qcols]
        .ewm(span=3, axis=1, adjust=False)
        .mean()
        .iloc[:, -1]
        .fillna(defaults.get("momentum_ewma", 0.0))
    )

    df["momentum_score_ewma_q3"] = (
        df[qcols[:3]]
        .ewm(span=2, axis=1, adjust=False)
        .mean()
        .iloc[:, -1]
        .fillna(defaults.get("momentum_ewma", 0.0))
    )


# --------------------------------------------------------------------------- #
#  public transform()
# --------------------------------------------------------------------------- #
def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    defaults: Mapping[str, float] = DEFAULTS,
) -> pd.DataFrame:
    """
    Add intra-game momentum features and return a **new** DataFrame.

    Features added
    --------------
    * `q{i}_margin`                – per-quarter margin (home − away)
    * `end_q{n}_diff`              – running cumulative margin
    * `q{n}_margin_change`         – Δ margin vs previous quarter
    * `momentum_score_ewma_q{3,4}` – EWMA momentum proxy
    """
    if df is None or df.empty:
        if debug:
            logger.debug("momentum.transform: empty input, returning as-is.")
        return df

    # shallow copy → cheap & leaves caller’s df untouched
    out = df.copy(deep=False)

    # ensure quarter columns exist & numeric
    q_home = [f"home_q{i}" for i in range(1, 5)]
    q_away = [f"away_q{i}" for i in range(1, 5)]
    out = convert_and_fill(out, q_home + q_away, default=0.0)

    # compute momentum features (in-place)
    _add_intra_game_momentum(out, dict(defaults))

    if debug:
        logger.debug("momentum.transform: done, shape=%s", out.shape)

    return out
