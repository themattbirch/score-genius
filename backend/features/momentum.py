import pandas as pd
from .utils import safe_divide, DEFAULTS

__all__ = ['add_intra_game_momentum']

def add_intra_game_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter-margin EWMAs & deltas."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for i in range(1, 5):
        out[f'q{i}_margin'] = out[f'home_q{i}'] - out[f'away_q{i}']
    out['end_q1_diff'] = out['q1_margin']
    out['end_q2_diff'] = out['end_q1_diff'] + out['q2_margin']
    out['end_q3_diff'] = out['end_q2_diff'] + out['q3_margin']
    out['end_q4_reg_diff'] = out['end_q3_diff'] + out['q4_margin']
    out['q2_margin_change'] = out['q2_margin'] - out['q1_margin']
    out['q3_margin_change'] = out['q3_margin'] - out['q2_margin']
    out['q4_margin_change'] = out['q4_margin'] - out['q3_margin']
    qcols = [f'q{i}_margin' for i in range(1, 5)]
    out['momentum_score_ewma_q4'] = (
        out[qcols].ewm(span=3, axis=1, adjust=False).mean().iloc[:, -1]
        .fillna(DEFAULTS['momentum_ewma'])
    )
    out['momentum_score_ewma_q3'] = (
        out[qcols[:3]].ewm(span=2, axis=1, adjust=False).mean().iloc[:, -1]
        .fillna(DEFAULTS['momentum_ewma'])
    )
    return out
