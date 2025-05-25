# ---------------------------------------------------------------------
# backend/nba_features/rolling.py - Using robust helper functions
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]


def _lagged_rolling_stat(
    s: pd.Series,
    w: int,
    min_p: int,
    stat_func: str
) -> pd.Series:
    """
    Leakage-free rolling statistic (mean or std).

    • Excludes *all* games played on the same calendar date as the current row.
    • Falls back to the mean/std of the available history (>=1 game).
    """
    if s.empty:
        return s

    # s is already ordered by game_date
    idx = s.index
    # drop current row via shift
    shifted = s.shift(1)
    # remove same-day duplicates
    dates = pd.Series(idx, index=idx)
    same_day = dates == dates.shift(1)
    shifted.loc[same_day.values] = np.nan

    # compute rolling
    if stat_func == 'mean':
        primary = shifted.rolling(window=w, min_periods=min_p).mean()
        fallback = shifted.rolling(window=w, min_periods=1).mean()
    elif stat_func == 'std':
        primary = shifted.rolling(window=w, min_periods=min_p).std()
        fallback = shifted.rolling(window=w, min_periods=1).std()
    else:
        raise ValueError(f"Unsupported stat_func: {stat_func}")

    result = primary.fillna(fallback)
    # return aligned to original group order without reindex
    return pd.Series(result.values, index=idx, name=s.name)


def transform(
    df: pd.DataFrame,
    *,
    window_sizes: List[int] = (5, 10, 20),
    flag_imputation: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add leakage-free rolling mean/std features for each team & stat,
    plus optional boolean flags where defaults were imputed.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Input df shape: {df.shape}")

    if df.empty:
        return df.copy()

    out = df.copy()
    # Validate required columns
    for col in ['game_id', 'game_date', 'home_team', 'away_team']:
        if col not in out.columns:
            logger.error(f"Missing required column: {col}, skipping rolling features.")
            return out

    out['game_date'] = pd.to_datetime(out['game_date'], errors='coerce').dt.tz_localize(None)
    out = out.dropna(subset=['game_date']).copy()
    out['home_norm'] = out['home_team'].map(normalize_team_name)
    out['away_norm'] = out['away_team'].map(normalize_team_name)

    # Define stat mapping for long format
    mapping = {
        ('home_score','away_score'): 'score_for',
        ('home_offensive_rating','away_offensive_rating'): 'off_rating',
        ('home_defensive_rating','away_defensive_rating'): 'def_rating',
        ('home_net_rating','away_net_rating'): 'net_rating',
    }

    # Build long-format DataFrame
    recs = []
    for idx, row in out.iterrows():
        for (hc, ac), stat in mapping.items():
            recs.append({
                'idx': idx,
                'game_id': row['game_id'],
                'game_date': row['game_date'],
                'team': row['home_norm'],
                'stat': stat,
                'val': pd.to_numeric(row.get(hc), errors='coerce')
            })
            recs.append({
                'idx': idx,
                'game_id': row['game_id'],
                'game_date': row['game_date'],
                'team': row['away_norm'],
                'stat': stat,
                'val': pd.to_numeric(row.get(ac), errors='coerce')
            })
    long = pd.DataFrame.from_records(recs)
    long = long.set_index('game_date')
    long = long.sort_values(['team','stat','game_date'], kind='mergesort')

    pieces = []
    for w in window_sizes:
        min_p = max(1, w // 2)
        grp = long.groupby(['team','stat'], group_keys=False, observed=True)

        # Rolling computations
        long[f'mean_{w}'] = grp['val'].transform(lambda s: _lagged_rolling_stat(s, w, min_p, 'mean'))
        long[f'std_{w}']  = grp['val'].transform(lambda s: _lagged_rolling_stat(s, w, min_p, 'std'))

        # Flags before filling defaults
        if flag_imputation:
            long[f'imp_mean_{w}'] = long[f'mean_{w}'].isna()
            long[f'imp_std_{w}']  = long[f'std_{w}'].isna()

        # Fill with DEFAULTS
        long[f'mean_{w}'] = long[f'mean_{w}'].fillna(long['stat'].map(lambda s: DEFAULTS.get(s, 0.0)))
        long[f'std_{w}']  = long[f'std_{w}'].fillna(long['stat'].map(lambda s: DEFAULTS.get(f"{s}_std", 0.0))).clip(lower=0)

        # Pivot to wide
        temp = long.reset_index()
        mean_w = temp.pivot_table(index=['game_id','team'], columns='stat', values=f'mean_{w}', aggfunc='first')
        mean_w.columns = [f'rolling_{c}_mean_{w}' for c in mean_w.columns]
        std_w  = temp.pivot_table(index=['game_id','team'], columns='stat', values=f'std_{w}',  aggfunc='first')
        std_w.columns  = [f'rolling_{c}_std_{w}' for c in std_w.columns]
        df_w = pd.concat([mean_w, std_w], axis=1)

        if flag_imputation:
            imp_mean_w = temp.pivot_table(index=['game_id','team'], columns='stat', values=f'imp_mean_{w}', aggfunc='first')
            imp_mean_w.columns = [f'rolling_{c}_mean_{w}_imputed' for c in imp_mean_w.columns]
            imp_std_w  = temp.pivot_table(index=['game_id','team'], columns='stat', values=f'imp_std_{w}',  aggfunc='first')
            imp_std_w.columns  = [f'rolling_{c}_std_{w}_imputed'  for c in imp_std_w.columns]
            df_w = pd.concat([df_w, imp_mean_w, imp_std_w], axis=1)

        pieces.append(df_w)

    # Merge all window pieces
    tidy = pd.concat(pieces, axis=1).reset_index()
    tidy['merge_key']     = tidy['game_id'].astype(str) + '_' + tidy['team']
    out['merge_key_home'] = out['game_id'].astype(str) + '_' + out['home_norm']
    out['merge_key_away'] = out['game_id'].astype(str) + '_' + out['away_norm']

    roll_cols = [c for c in tidy.columns if c.startswith('rolling_')]
    home_map = {c: c.replace('rolling_', 'home_rolling_') for c in roll_cols}
    away_map = {c: c.replace('rolling_', 'away_rolling_') for c in roll_cols}

    out = out.merge(
        tidy[['merge_key', *roll_cols]].rename(columns=home_map),
        how='left', left_on='merge_key_home', right_on='merge_key'
    )
    out = out.merge(
        tidy[['merge_key', *roll_cols]].rename(columns=away_map),
        how='left', left_on='merge_key_away', right_on='merge_key'
    )

    # Cleanup helpers
    drop_cols = ['home_norm','away_norm','merge_key_home','merge_key_away','merge_key']
    return out.drop(columns=drop_cols, errors='ignore')
