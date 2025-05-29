# backend/mlb_features/rolling.py
"""
Calculates leakage-free rolling mean and standard deviation features
for MLB games based on team-level game statistics (runs, hits, errors).

Focus:
1. Strict no-lookahead: only prior games contribute.
2. Handles multiple games per day by excluding same-day duplicates.
3. Provides fallback defaults and imputation flags.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

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
    from .utils import DEFAULTS, normalize_team_name
    logger.info("Imported DEFAULTS and normalize_team_name from mlb_features.utils")
except ImportError:
    logger.warning("Could not import DEFAULTS or normalize_team_name; using fallbacks.")
    DEFAULTS: Dict[str, Any] = {}
    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip() if pd.notna(team_id) else "unknown"


def _lagged_rolling_stat(
    s: pd.Series,
    window: int,
    min_periods: int,
    stat: str
) -> pd.Series:
    """
    Compute a leakage-free rolling statistic (mean or std) on the series `s`:
    - Shifted by 1 to exclude the current game
    - Excludes values from the same date as the immediately preceding game
    - Provides fallback when primary min_periods not met
    """
    if s.empty:
        return pd.Series([], dtype=float, index=s.index)

    shifted = s.shift(1)
    dates = pd.Series(s.index, index=s.index)
    # Exclude same-day duplicates
    same_day = dates == dates.shift(1)
    shifted.loc[same_day] = np.nan

    if stat == 'mean':
        primary = shifted.rolling(window=window, min_periods=min_periods).mean()
        fallback = shifted.rolling(window=window, min_periods=1).mean()
    elif stat == 'std':
        primary = shifted.rolling(window=window, min_periods=min_periods).std()
        fallback = shifted.rolling(window=window, min_periods=1).std()
    else:
        raise ValueError(f"Unsupported stat: {stat}")

    result = primary.fillna(fallback)
    return pd.Series(result.values, index=s.index, name=s.name)


def transform(
    df: pd.DataFrame,
    *,
    window_sizes: List[int] = (5, 10, 20),
    flag_imputation: bool = True,
    debug: bool = False,
    # Column names
    game_date_col: str = "game_date_et",
    home_team_col: str = "home_team_id",
    away_team_col: str = "away_team_id",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    home_hits_col: str = "home_hits",
    away_hits_col: str = "away_hits",
    home_errors_col: str = "home_errors",
    away_errors_col: str = "away_errors",
) -> pd.DataFrame:
    """
    Adds rolling mean/std features for MLB teams.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Starting rolling.transform; df shape={df.shape}")

    if df is None or df.empty:
        logger.warning("Empty input to rolling.transform; returning empty copy.")
        return df.copy()

    out = df.copy()
    required = [
        'game_id', game_date_col, home_team_col, away_team_col,
        home_score_col, away_score_col,
        home_hits_col, away_hits_col,
        home_errors_col, away_errors_col
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return df

    # Parse dates
    out['game_date'] = (
        pd.to_datetime(out[game_date_col], errors='coerce')
        .dt.tz_localize(None)
    )
    out = out.dropna(subset=['game_date']).copy()

    # Normalize team
    out['home_norm'] = out[home_team_col].apply(normalize_team_name)
    out['away_norm'] = out[away_team_col].apply(normalize_team_name)

    # Generic stat mapping: stat -> (home_col, away_col)
    stat_map: Dict[str, tuple] = {
        'runs_scored':      (home_score_col, away_score_col),
        'runs_allowed':     (away_score_col, home_score_col),
        'hits_for':         (home_hits_col, away_hits_col),
        'hits_allowed':     (away_hits_col, home_hits_col),
        'errors_committed': (home_errors_col, away_errors_col),
        'errors_by_opponent':(away_errors_col, home_errors_col)
    }

    # Build long format
    records: List[Dict[str, Any]] = []
    for idx, row in out.iterrows():
        for side, team_key in (('home', 'home_norm'), ('away', 'away_norm')):
            team = row[team_key]
            for stat, (h_col, a_col) in stat_map.items():
                val = row[h_col] if side == 'home' else row[a_col]
                records.append({
                    'game_id': row['game_id'],
                    'team_norm': team,
                    'game_date': row['game_date'],
                    'stat': stat,
                    'value': pd.to_numeric(val, errors='coerce')
                })
    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        logger.warning("Long DF empty in rolling.transform; no stats to calculate.")
        return out

    # Sort for stable rolling
    long_df = long_df.sort_values(
        ['team_norm', 'stat', 'game_date'],
        kind='mergesort', ignore_index=True
    )

    # Compute rolling for each window
    for w in window_sizes:
        min_p = max(1, w // 2)
        long_df[f'mean_{w}'] = long_df.groupby(
            ['team_norm', 'stat'], observed=True
        )['value'].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, 'mean')
        )
        long_df[f'std_{w}'] = long_df.groupby(
            ['team_norm', 'stat'], observed=True
        )['value'].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, 'std')
        )
        if flag_imputation:
            long_df[f'mean_{w}_imputed'] = long_df[f'mean_{w}'].isna()
            long_df[f'std_{w}_imputed'] = long_df[f'std_{w}'].isna()

        # Fill defaults
        long_df[f'mean_{w}'] = long_df.apply(
            lambda r: r[f'mean_{w}']
                      if pd.notna(r[f'mean_{w}'])
                      else DEFAULTS.get(r['stat'], 0.0), axis=1
        )
        long_df[f'std_{w}'] = long_df.apply(
            lambda r: r[f'std_{w}']
                      if pd.notna(r[f'std_{w}'])
                      else DEFAULTS.get(f"{r['stat']}_std", 0.0), axis=1
        )

    # Pivot wide
    pivots = []
    for w in window_sizes:
        mean_pivot = long_df.pivot_table(
            index=['game_id', 'team_norm'],
            columns='stat', values=f'mean_{w}', aggfunc='first'
        )
        mean_pivot.columns = [f'rolling_{stat}_mean_{w}' for stat in mean_pivot.columns]

        std_pivot = long_df.pivot_table(
            index=['game_id', 'team_norm'],
            columns='stat', values=f'std_{w}', aggfunc='first'
        )
        std_pivot.columns = [f'rolling_{stat}_std_{w}' for stat in std_pivot.columns]

        dfs = [mean_pivot, std_pivot]
        if flag_imputation:
            imp_mean = long_df.pivot_table(
                index=['game_id', 'team_norm'],
                columns='stat', values=f'mean_{w}_imputed', aggfunc='first'
            )
            imp_mean.columns = [f'rolling_{stat}_mean_{w}_imputed' for stat in imp_mean.columns]
            imp_std = long_df.pivot_table(
                index=['game_id', 'team_norm'],
                columns='stat', values=f'std_{w}_imputed', aggfunc='first'
            )
            imp_std.columns = [f'rolling_{stat}_std_{w}_imputed' for stat in imp_std.columns]
            dfs.extend([imp_mean, imp_std])

        pivots.append(pd.concat(dfs, axis=1))

    rolling_wide = pd.concat(pivots, axis=1).reset_index()

    # Merge back into out
    rolling_wide['key'] = rolling_wide['game_id'].astype(str) + '_' + rolling_wide['team_norm']
    out['home_key'] = out['game_id'].astype(str) + '_' + out['home_norm']
    out['away_key'] = out['game_id'].astype(str) + '_' + out['away_norm']

    # Merge home
    home_cols = [c for c in rolling_wide.columns if c.startswith('rolling_')]
    out = out.merge(
        rolling_wide[['key'] + home_cols].rename(
            columns={c: f'home_{c}' for c in home_cols}
        ),
        how='left', left_on='home_key', right_on='key'
    )
    # Merge away
    out = out.merge(
        rolling_wide[['key'] + home_cols].rename(
            columns={c: f'away_{c}' for c in home_cols}
        ),
        how='left', left_on='away_key', right_on='key'
    )

    # Cleanup
    to_drop = ['home_norm', 'away_norm', 'home_key', 'away_key', 'key', 'game_date']
    out = out.drop(columns=[c for c in to_drop if c in out.columns], errors='ignore')

    if debug:
        logger.debug(f"Finished rolling.transform; output shape={out.shape}")
    return out
