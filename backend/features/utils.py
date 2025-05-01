# backend/features/utils.py

from __future__ import annotations
import numpy as np
import pandas as pd

# -------- Defaults --------
DEFAULTS: dict[str, float] = {
    'efg_pct': 0.54,
    'ft_rate': 0.20,
    'pace': 100.0,
    'tov_rate': 13.0,
    'oreb_pct': 0.23,
    'dreb_pct': 0.77,
    'trb_pct': 0.50,
    'score_for': 115.0,
    'score_against': 115.0,
    'net_rating': 0.0,
    'momentum_ewma': 0.0,
    'rest_days': 3.0,
    'win_pct': 0.5,
}

EPSILON = 1e-6

# -------- Helpers --------
def safe_divide(
    num: pd.Series | float,
    den: pd.Series | float,
    default_val: float = 0.0
) -> pd.Series | float:
    """Division that never explodes."""
    res = pd.to_numeric(num, errors='coerce') / (
        pd.to_numeric(den, errors='coerce').replace(0, np.nan)
    )
    if isinstance(res, pd.Series):
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        return res.fillna(default_val)
    return default_val if np.isnan(res) or np.isinf(res) else res

def generate_rolling_column_name(
    prefix: str,
    base: str,
    stat_type: str,
    window: int
) -> str:
    """Generate standardized rolling feature column names."""
    pre = f'{prefix}_' if prefix else ''
    return f'{pre}rolling_{base}_{stat_type}_{window}'

def convert_and_fill(
    df: pd.DataFrame,
    cols: list[str],
    default: float = 0.0
) -> pd.DataFrame:
    """
    Ensure specified columns exist, coerce them to numeric, 
    and fill any NaNs with the given default.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(default)
    return df
