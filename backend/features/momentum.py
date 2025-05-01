from __future__ import annotations
import pandas as pd
import logging

from .utils import safe_divide, DEFAULTS
logger = logging.getLogger(__name__)

__all__ = ['transform']

def _convert_and_fill(
    df: pd.DataFrame,
    cols: list[str],
    default: float = 0.0
) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    return df

def add_intra_game_momentum(
    df: pd.DataFrame,
    defaults: dict[str, float]
) -> pd.DataFrame:
    """Core legacy logic: assumes all home_q*/away_q* exist & numeric."""
    out = df.copy()
    # quarter margins
    for i in range(1, 5):
        out[f'q{i}_margin'] = out[f'home_q{i}'] - out[f'away_q{i}']

    # cumulative
    out['end_q1_diff']     = out['q1_margin']
    out['end_q2_diff']     = out['end_q1_diff'] + out['q2_margin']
    out['end_q3_diff']     = out['end_q2_diff'] + out['q3_margin']
    out['end_q4_reg_diff'] = out['end_q3_diff'] + out['q4_margin']

    # inter-quarter deltas
    out['q2_margin_change'] = out['q2_margin'] - out['q1_margin']
    out['q3_margin_change'] = out['q3_margin'] - out['q2_margin']
    out['q4_margin_change'] = out['q4_margin'] - out['q3_margin']

    # EWMA scores
    qcols = [f'q{i}_margin' for i in range(1, 5)]
    out['momentum_score_ewma_q4'] = (
        out[qcols]
        .ewm(span=3, axis=1, adjust=False)
        .mean()
        .iloc[:, -1]
        .fillna(defaults['momentum_ewma'])
    )
    out['momentum_score_ewma_q3'] = (
        out[qcols[:3]]
        .ewm(span=2, axis=1, adjust=False)
        .mean()
        .iloc[:, -1]
        .fillna(defaults['momentum_ewma'])
    )

    return out

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    defaults: dict[str, float] = DEFAULTS
) -> pd.DataFrame:
    """
    Add intra-game momentum features:
      - quarter margins
      - end_q{n}_diff
      - q{n}_margin_change
      - EWMA momentum scores
    """
    if df is None or df.empty:
        if debug:
            logger.debug("momentum.transform: empty input, returning as-is.")
        return df

    out = df.copy()
    if debug:
        logger.debug("momentum.transform: filling quarter columns and computing margins…")

    # 1️⃣ Ensure quarters are numeric (and present)
    q_base      = [f'q{i}' for i in range(1, 5)]
    q_home_cols = [f'home_{c}' for c in q_base]
    q_away_cols = [f'away_{c}' for c in q_base]
    out = _convert_and_fill(out, q_home_cols + q_away_cols, default=0.0)

    # 2️⃣ Delegate the heavy lifting
    out = add_intra_game_momentum(out, defaults)

    if debug:
        logger.debug("momentum.transform: done.")

    return out
