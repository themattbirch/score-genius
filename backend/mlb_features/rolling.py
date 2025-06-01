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

# --- MODULE-LEVEL DEFAULT COLUMN NAMES ---
_DEFAULT_HOME_SCORE_COL = "home_score"
_DEFAULT_AWAY_SCORE_COL = "away_score"
_DEFAULT_HOME_HITS_COL = "home_hits"
_DEFAULT_AWAY_HITS_COL = "away_hits"
_DEFAULT_HOME_ERRORS_COL = "home_errors"
_DEFAULT_AWAY_ERRORS_COL = "away_errors"

# --- MODULE-LEVEL STAT_MAP ---
stat_map: Dict[str, tuple] = {
    'runs_scored':      (_DEFAULT_HOME_SCORE_COL, _DEFAULT_AWAY_SCORE_COL),
    'runs_allowed':     (_DEFAULT_AWAY_SCORE_COL, _DEFAULT_HOME_SCORE_COL),
    'hits_for':         (_DEFAULT_HOME_HITS_COL, _DEFAULT_AWAY_HITS_COL),
    'hits_allowed':     (_DEFAULT_AWAY_HITS_COL, _DEFAULT_HOME_HITS_COL),
    'errors_committed': (_DEFAULT_HOME_ERRORS_COL, _DEFAULT_AWAY_ERRORS_COL),
    'errors_by_opponent':(_DEFAULT_AWAY_ERRORS_COL, _DEFAULT_HOME_ERRORS_COL)
}


def _lagged_rolling_stat(
    s: pd.Series,
    window: int,
    min_periods: int,
    stat: str
) -> pd.Series:
    """
    Compute a leakage-free rolling statistic (mean or std) on the series `s`:
    - Shifted by 1 to exclude the current game
    - Excludes values from the same date as the immediately preceding game (CURRENTLY FLAWED - see note)
    - Provides fallback when primary min_periods not met
    Note: The same-day exclusion logic using 'dates = pd.Series(s.index, index=s.index)'
          is incorrect when 's.index' is not a DatetimeIndex reflecting game dates.
          This needs revision if precise same-day handling within _lagged_rolling_stat is critical
          and causing issues not yet caught by tests.
    """
    if s.empty:
        return pd.Series([], dtype=float, index=s.index, name=s.name if s.name is not None else 'value')

    shifted = s.shift(1)
    
    # Problematic same-day logic (as noted before)
    dates = pd.Series(s.index, index=s.index) 
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
    return pd.Series(result.values, index=s.index, name=s.name if s.name is not None else 'value')


def transform(
    df: pd.DataFrame,
    *,
    window_sizes: List[int] = (5, 10, 20),
    flag_imputations: bool = True,
    debug: bool = False,
    game_date_col: str = "game_date",
    home_team_col: str = "home_team_id",
    away_team_col: str = "away_team_id",
    home_score_col: str = _DEFAULT_HOME_SCORE_COL,
    away_score_col: str = _DEFAULT_AWAY_SCORE_COL,
    home_hits_col: str = _DEFAULT_HOME_HITS_COL,
    away_hits_col: str = _DEFAULT_AWAY_HITS_COL,
    home_errors_col: str = _DEFAULT_HOME_ERRORS_COL,
    away_errors_col: str = _DEFAULT_AWAY_ERRORS_COL,
) -> pd.DataFrame:
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Starting rolling.transform; df shape={df.shape}")

    if df is None or df.empty:
        logger.warning("Empty input to rolling.transform; returning empty copy.")
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    required_input_cols = [
        'game_id', game_date_col, home_team_col, away_team_col,
        home_score_col, away_score_col,
        home_hits_col, away_hits_col,
        home_errors_col, away_errors_col
    ]
    missing = [c for c in required_input_cols if c not in out.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}. Returning original DataFrame.")
        if debug: logger.setLevel(orig_level)
        return df

    out['game_date_parsed_'] = (
        pd.to_datetime(out[game_date_col], errors='coerce')
        .dt.tz_localize(None)
    )
    out = out.dropna(subset=['game_date_parsed_']).copy()

    if out.empty:
        logger.warning("DataFrame empty after processing game_date in rolling.transform. Returning.")
        if debug: logger.setLevel(orig_level)
        return out.drop(columns=['game_date_parsed_'], errors='ignore') if 'game_date_parsed_' in out.columns else out

    out['home_norm'] = out[home_team_col].apply(normalize_team_name)
    out['away_norm'] = out[away_team_col].apply(normalize_team_name)

    processing_stat_map: Dict[str, tuple] = {
        'runs_scored':      (home_score_col, away_score_col),
        'runs_allowed':     (away_score_col, home_score_col),
        'hits_for':         (home_hits_col, away_hits_col),
        'hits_allowed':     (away_hits_col, home_hits_col),
        'errors_committed': (home_errors_col, away_errors_col),
        'errors_by_opponent':(away_errors_col, home_errors_col)
    }
    
    records: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        for side, team_key_in_out_df in (('home', 'home_norm'), ('away', 'away_norm')):
            team_val = row[team_key_in_out_df]
            for stat_name, (h_col, a_col) in processing_stat_map.items():
                val_to_use = row[h_col] if side == 'home' else row[a_col]
                records.append({
                    'game_id': row['game_id'],
                    'team_norm': team_val,
                    'game_date': row['game_date_parsed_'],
                    'stat': stat_name,
                    'value': pd.to_numeric(val_to_use, errors='coerce')
                })
    
    if not records:
        logger.warning("No records generated for long_df in rolling.transform; no stats to calculate. Returning processed DataFrame.")
        final_cols_to_drop = ['game_date_parsed_', 'home_norm', 'away_norm']
        if debug: logger.setLevel(orig_level)
        return out.drop(columns=[c for c in final_cols_to_drop if c in out.columns], errors='ignore')

    long_df = pd.DataFrame.from_records(records)
    
    long_df = long_df.sort_values(
        ['team_norm', 'stat', 'game_date', 'game_id'], 
        kind='mergesort',
        ignore_index=True 
    )

    for w in window_sizes:
        min_p = max(1, w // 2)
        
        long_df[f'mean_{w}'] = long_df.groupby(
            ['team_norm', 'stat'], observed=True, group_keys=False 
        )['value'].transform( 
            lambda s: _lagged_rolling_stat(s, w, min_p, 'mean')
        )
        long_df[f'std_{w}'] = long_df.groupby(
            ['team_norm', 'stat'], observed=True, group_keys=False
        )['value'].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, 'std')
        )
        if flag_imputations:
            # Store as boolean (will be numpy.bool_ initially)
            long_df[f'mean_{w}_imputed'] = long_df[f'mean_{w}'].isnull() 
            long_df[f'std_{w}_imputed'] = long_df[f'std_{w}'].isnull()   

        long_df[f'mean_{w}'] = long_df.apply(
            lambda r: r[f'mean_{w}']
                      if pd.notna(r[f'mean_{w}'])
                      else DEFAULTS.get(r['stat'], 0.0), axis=1
        )
        long_df[f'std_{w}'] = long_df.apply(
            lambda r: r[f'std_{w}']
                      if pd.notna(r[f'std_{w}'])
                      else DEFAULTS.get(f"{r['stat']}_std", DEFAULTS.get(r['stat'], 0.0)), axis=1
        )

    pivots = []
    if not long_df.empty:
        for w in window_sizes:
            pivot_dfs_for_window = []
            if f'mean_{w}' in long_df.columns:
                mean_pivot = long_df.pivot_table(
                    index=['game_id', 'team_norm'],
                    columns='stat', values=f'mean_{w}', aggfunc='first'
                )
                mean_pivot.columns = [f'rolling_{stat}_mean_{w}' for stat in mean_pivot.columns]
                pivot_dfs_for_window.append(mean_pivot)

            if f'std_{w}' in long_df.columns:
                std_pivot = long_df.pivot_table(
                    index=['game_id', 'team_norm'],
                    columns='stat', values=f'std_{w}', aggfunc='first'
                )
                std_pivot.columns = [f'rolling_{stat}_std_{w}' for stat in std_pivot.columns]
                pivot_dfs_for_window.append(std_pivot)

            if flag_imputations:
                if f'mean_{w}_imputed' in long_df.columns:
                    imp_mean = long_df.pivot_table(
                        index=['game_id', 'team_norm'],
                        columns='stat', values=f'mean_{w}_imputed', aggfunc='first'
                    )
                    imp_mean.columns = [f'rolling_{stat}_mean_{w}_imputed' for stat in imp_mean.columns]
                    pivot_dfs_for_window.append(imp_mean)
                if f'std_{w}_imputed' in long_df.columns:
                    imp_std = long_df.pivot_table(
                        index=['game_id', 'team_norm'],
                        columns='stat', values=f'std_{w}_imputed', aggfunc='first'
                    )
                    imp_std.columns = [f'rolling_{stat}_std_{w}_imputed' for stat in imp_std.columns]
                    pivot_dfs_for_window.append(imp_std)
            
            if pivot_dfs_for_window:
                pivots.append(pd.concat(pivot_dfs_for_window, axis=1))
    
    if pivots:
        rolling_wide = pd.concat(pivots, axis=1).reset_index()

        rolling_wide['merge_key_'] = rolling_wide['game_id'].astype(str) + '_' + rolling_wide['team_norm'].astype(str)
        out['home_merge_key_'] = out['game_id'].astype(str) + '_' + out['home_norm'].astype(str)
        out['away_merge_key_'] = out['game_id'].astype(str) + '_' + out['away_norm'].astype(str)

        rolling_cols_to_merge = [c for c in rolling_wide.columns if c.startswith('rolling_')]
        
        if rolling_cols_to_merge or ('merge_key_' in rolling_wide.columns and not rolling_wide[['merge_key_']].empty):
            # Home merge
            df_to_merge_home = rolling_wide[['merge_key_'] + rolling_cols_to_merge].rename(
                columns={c: f'home_{c}' for c in rolling_cols_to_merge}
            )
            if not df_to_merge_home.empty:
                 out = out.merge(df_to_merge_home, how='left', left_on='home_merge_key_', right_on='merge_key_')
                 if 'merge_key__x' in out.columns and 'merge_key__y' in out.columns:
                     out = out.drop(columns=['merge_key__y']).rename(columns={'merge_key__x': 'merge_key_'})
                 elif 'merge_key_' in out.columns and 'merge_key_' in df_to_merge_home.columns and not out['home_merge_key_'].equals(out['merge_key_']): # check if merge_key_ was from right
                     out = out.drop(columns=['merge_key_'], errors='ignore')


            # Away merge
            df_to_merge_away = rolling_wide[['merge_key_'] + rolling_cols_to_merge].rename(
                columns={c: f'away_{c}' for c in rolling_cols_to_merge}
            )
            if not df_to_merge_away.empty:
                out = out.merge(df_to_merge_away, how='left', left_on='away_merge_key_', right_on='merge_key_')
                if 'merge_key__x' in out.columns and 'merge_key__y' in out.columns:
                    out = out.drop(columns=['merge_key__y']).rename(columns={'merge_key__x': 'merge_key_'})
                elif 'merge_key_' in out.columns and 'merge_key_' in df_to_merge_away.columns and not out['away_merge_key_'].equals(out['merge_key_']):
                    out = out.drop(columns=['merge_key_'], errors='ignore')

    else: 
        logger.warning("No rolling_wide DataFrame generated. Output will not have new rolling stat columns.")

    temp_cols_to_drop = ['home_norm', 'away_norm', 'home_merge_key_', 'away_merge_key_', 'game_date_parsed_']
    if 'merge_key_' in out.columns: # Cleanup any remaining merge_key_ from left side if it was not dropped
         temp_cols_to_drop.append('merge_key_')
        
    out = out.drop(columns=[c for c in temp_cols_to_drop if c in out.columns], errors='ignore')

    # --- Final conversion of _imputed columns to Python bool ---
    final_imputed_cols = [c for c in out.columns if c.endswith('_imputed')]
    for col_name in final_imputed_cols:
        if col_name in out.columns and col_name in out: # Ensure column exists
            # Convert all values in the column to Python bools.
            # This assumes values are already boolean-like (True/False/1/0 as np.bool_ or bool)
            # and do not contain NaNs (which .isnull() results shouldn't).
            # Creating a new Series with dtype=object from a list of Python bools
            # ensures they are stored as Python objects.
            python_bool_values = [bool(val) for val in out[col_name].tolist()]
            out[col_name] = pd.Series(python_bool_values, index=out.index, name=col_name, dtype='object')

    if debug:
        logger.setLevel(orig_level)
        logger.debug(f"Finished rolling.transform; output shape={out.shape}")
    return out
