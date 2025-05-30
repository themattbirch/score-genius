# backend/nba_features/rolling.py
"""
Calculates leakage-free rolling mean and standard deviation features for team statistics.

This module transforms game-level data into a long format to compute rolling
statistics (mean, std) for various performance metrics (e.g., score, ratings)
over specified window sizes. It ensures that rolling calculations do not "look ahead"
by excluding any data from the same calendar date as the game being calculated for.
The results are then pivoted back and merged onto the original DataFrame.
Imputation flags can also be generated.
"""

from __future__ import annotations
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]


def _lagged_rolling_stat(s: pd.Series, window: int, min_periods: int, stat_func: str) -> pd.Series:
    """
    Compute a leakage-free rolling statistic (mean or std) for series `s`.
    Excludes the current row (lag=1) and any same-date duplicates from the shifted series.
    Falls back to min_periods=1 window if primary yields NaN.
    """
    if s.empty:
        return s.copy()

    shifted = s.shift(1) # Exclude current row's actual value

    # If the index (game_date for this team/stat) has duplicates after shifting,
    # it means multiple games on the same day. NaN out later same-day games' shifted values.
    # This ensures no game on a given date influences another game on that same date.
    if shifted.index.has_duplicates:
        # Mark all but the first occurrence of each date in the index as duplicated
        # True means it's a duplicate of a preceding identical index value
        is_duplicated_date_in_shifted_index = shifted.index.duplicated(keep='first')
        shifted.loc[is_duplicated_date_in_shifted_index] = np.nan

    if stat_func == 'mean':
        primary = shifted.rolling(window=window, min_periods=min_periods).mean()
        fallback = shifted.rolling(window=window, min_periods=1).mean()
    elif stat_func == 'std':
        primary = shifted.rolling(window=window, min_periods=min_periods).std()
        fallback = shifted.rolling(window=window, min_periods=1).std()
    else:
        raise ValueError(f"Unsupported stat_func: {stat_func}")
    
    # Return series re-aligned to original index `s.index`
    return pd.Series(primary.fillna(fallback).values, index=s.index, name=s.name)


def transform(
    df: pd.DataFrame,
    *,
    window_sizes: Sequence[int] = (5, 10, 20),
    flag_imputation: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add leakage-free rolling mean/std features for home/away team stats.

    Args:
        df: DataFrame, must contain 'game_id','game_date','home_team','away_team',
            and the stat columns specified in the internal stat_mapping.
        window_sizes: Sequence of window sizes for rolling calculations.
        flag_imputation: If True, add boolean _imputed flags for rolling stats.
        debug: If True, enable DEBUG logging for this function call.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"rolling.transform: Input df shape: {df.shape}")

    try:
        if df.empty:
            logger.warning("rolling.transform: empty input, returning copy")
            return df.copy()

        out = df.copy()
        stat_mapping = {
            ('home_score', 'away_score'): 'score_for',
            ('home_offensive_rating', 'away_offensive_rating'): 'off_rating',
            ('home_defensive_rating', 'away_defensive_rating'): 'def_rating',
            ('home_net_rating', 'away_net_rating'): 'net_rating',
        }
        needed_cols = {'game_id', 'game_date', 'home_team', 'away_team'}
        for home_col, away_col in stat_mapping:
            needed_cols.update({home_col, away_col})
        
        missing = needed_cols - set(out.columns)
        if missing:
            logger.error(f"rolling.transform: missing required columns {missing}, skipping.")
            if debug: logger.setLevel(orig_level)
            return out # Return original if missing crucial columns

        out['game_date'] = pd.to_datetime(out['game_date'], errors='coerce').dt.tz_localize(None)
        out = out.dropna(subset=['game_date']).copy()
        if out.empty:
            logger.warning("rolling.transform: DataFrame empty after game_date processing.")
            if debug: logger.setLevel(orig_level)
            return out

        out['home_norm'] = out['home_team'].astype(str).map(normalize_team_name)
        out['away_norm'] = out['away_team'].astype(str).map(normalize_team_name)

        records = []
        for _, row in out.iterrows(): # Using original index from `out` is fine
            game_date = row['game_date']
            game_id = row['game_id']
            for (h_col, a_col), stat_name in stat_mapping.items():
                home_val = pd.to_numeric(row.get(h_col), errors='coerce')
                if pd.notna(home_val) and pd.notna(row['home_norm']) and row['home_norm'] is not None : # Check for None team_norm
                    records.append({'game_id': game_id, 'team': row['home_norm'], 'game_date': game_date, 'stat': stat_name, 'val': home_val})
                
                away_val = pd.to_numeric(row.get(a_col), errors='coerce')
                if pd.notna(away_val) and pd.notna(row['away_norm']) and row['away_norm'] is not None:
                    records.append({'game_id': game_id, 'team': row['away_norm'], 'game_date': game_date, 'stat': stat_name, 'val': away_val})
        
        if not records:
            logger.warning("rolling.transform: no long-format records created. Check input data (e.g., all NaNs, missing stat_mapping cols, or NaN team_norm).")
            # Clean up helper columns before returning
            cols_to_drop_temp = ['home_norm', 'away_norm']
            return out.drop(columns=[c for c in cols_to_drop_temp if c in out.columns], errors='ignore')

        long_df = pd.DataFrame.from_records(records)
        # Set game_date as index for _lagged_rolling_stat. Sort carefully.
        # game_id is included in sort for stable ordering if a team has multiple games on same date.
        long_df = long_df.set_index('game_date').sort_values(['team', 'stat', 'game_date', 'game_id'])

        # Precompute default values for each unique stat present
        unique_stats = long_df['stat'].unique()
        default_means = {s: DEFAULTS.get(s, 0.0) for s in unique_stats}
        default_stds = {s: DEFAULTS.get(f"{s}_std", 0.0) for s in unique_stats}

        window_feature_pieces = []
        for w_size in window_sizes:
            min_p = max(1, w_size // 2)
            df_window_long = long_df.copy() # Use a copy for each window's calculations

            logger.debug(f"Calculating rolling stats for window: {w_size}, min_periods: {min_p}")
            grouped = df_window_long.groupby(['team', 'stat'], observed=True, group_keys=False) # group_keys=False is often good practice

            mean_col_name = f'mean_{w_size}'
            std_col_name = f'std_{w_size}'
            df_window_long[mean_col_name] = grouped['val'].transform(lambda s: _lagged_rolling_stat(s, w_size, min_p, 'mean'))
            df_window_long[std_col_name] = grouped['val'].transform(lambda s: _lagged_rolling_stat(s, w_size, min_p, 'std'))

            if flag_imputation:
                df_window_long[f'imp_mean_{w_size}'] = df_window_long[mean_col_name].isnull()
                df_window_long[f'imp_std_{w_size}'] = df_window_long[std_col_name].isnull()
            
            # Fill NaNs using the precomputed mapped defaults (more efficient than .apply)
            df_window_long[mean_col_name] = df_window_long[mean_col_name].fillna(df_window_long['stat'].map(default_means))
            df_window_long[std_col_name] = df_window_long[std_col_name].fillna(df_window_long['stat'].map(default_stds)).clip(lower=0.0)

            # Pivot back to wide format
            df_window_long = df_window_long.reset_index() # game_date becomes a column
            pivot_index = ['game_id', 'team']
            
            mean_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=mean_col_name, aggfunc='first')
            mean_pivot.columns = [f'rolling_{stat_name}_mean_{w_size}' for stat_name in mean_pivot.columns]
            
            std_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=std_col_name, aggfunc='first')
            std_pivot.columns = [f'rolling_{stat_name}_std_{w_size}' for stat_name in std_pivot.columns]
            
            current_piece = pd.concat([mean_pivot, std_pivot], axis=1)

            if flag_imputation:
                imp_mean_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'imp_mean_{w_size}', aggfunc='first')
                imp_mean_pivot.columns = [f'rolling_{stat_name}_mean_{w_size}_imputed' for stat_name in imp_mean_pivot.columns]
                
                imp_std_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'imp_std_{w_size}', aggfunc='first')
                imp_std_pivot.columns = [f'rolling_{stat_name}_std_{w_size}_imputed' for stat_name in imp_std_pivot.columns]
                current_piece = pd.concat([current_piece, imp_mean_pivot, imp_std_pivot], axis=1)
            
            window_feature_pieces.append(current_piece.reset_index()) # Reset index (game_id, team) to columns for merging

        # Combine pieces from all window sizes
        if not window_feature_pieces: # Should not happen if window_sizes is not empty and long_df was processed
             logger.warning("rolling.transform: No feature pieces generated from window_sizes loop.")
             cols_to_drop_temp = ['home_norm', 'away_norm']
             return out.drop(columns=[c for c in cols_to_drop_temp if c in out.columns], errors='ignore')

        wide_features_df = window_feature_pieces[0]
        for piece in window_feature_pieces[1:]:
            wide_features_df = wide_features_df.merge(piece, on=['game_id', 'team'], how='outer')

        # Merge rolling features back to the original 'out' DataFrame
        feature_cols_from_wide = [col for col in wide_features_df.columns if col not in ['game_id', 'team']]

        for side in ('home', 'away'):
            side_rename_map = {col: f'{side}_{col}' for col in feature_cols_from_wide}
            # Also need to rename 'team' from wide_features_df to avoid conflict/for clarity in merge
            data_to_merge_for_side = wide_features_df.rename(columns={'team': 'merge_team_norm', **side_rename_map})
            
            # Select only necessary columns for merging to avoid duplicate game_id etc.
            cols_for_this_merge = ['game_id', 'merge_team_norm'] + [side_rename_map[fc] for fc in feature_cols_from_wide]

            out = out.merge(
                data_to_merge_for_side[cols_for_this_merge],
                left_on=['game_id', f'{side}_norm'],
                right_on=['game_id', 'merge_team_norm'],
                how='left'
            )
            if 'merge_team_norm' in out.columns: # Clean up the merge key from right
                out = out.drop(columns=['merge_team_norm'])
        
        # Final fillna and type coercion for all newly added rolling columns in 'out'
        all_added_rolling_cols = [col for col in out.columns if col.startswith(('home_rolling_', 'away_rolling_'))]
        for col_name in all_added_rolling_cols:
            if col_name.endswith('_imputed'):
                 out[col_name] = (
                     out[col_name]
                     .fillna(False) 
                     .astype(object) 
                    ) 
            else:
                default_val_for_col = 0.0 # Generic fallback default
                try:
                    # Attempt to parse stat name from col_name for specific default
                    # e.g., 'home_rolling_score_for_mean_5' or 'away_rolling_net_rating_std_10'
                    parts = col_name.split('_') # ['home', 'rolling', 'score', 'for', 'mean', '5']
                    if 'rolling' in parts:
                        rolling_idx = parts.index('rolling')
                        stat_name_parts = []
                        is_std_col = False
                        for i in range(rolling_idx + 1, len(parts)):
                            part_val = parts[i]
                            if part_val == 'mean': break
                            if part_val == 'std':
                                is_std_col = True
                                break
                            stat_name_parts.append(part_val)
                        
                        base_stat_name = "_".join(stat_name_parts)
                        if is_std_col:
                            default_val_for_col = default_stds.get(base_stat_name, 0.0)
                        else: # is mean_col
                            default_val_for_col = default_means.get(base_stat_name, 0.0)
                except (ValueError, IndexError):
                    logger.debug(f"Could not parse base stat for default for {col_name}, using generic 0.0")

                out[col_name] = pd.to_numeric(out[col_name], errors='coerce').fillna(default_val_for_col)
                if 'std' in col_name and not col_name.endswith('_imputed'): # Check if it's an std feature column
                    out[col_name] = out[col_name].clip(lower=0.0)
                out[col_name] = out[col_name].astype(float) # Ensure all numeric rolling stats are float

        # Final cleanup of helper columns from 'out'
        helper_cols = ['home_norm', 'away_norm'] # Add any other temp keys if created directly on 'out'
        out = out.drop(columns=[c for c in helper_cols if c in out.columns], errors='ignore')
        
        logger.debug(f"rolling.transform completed. Output df shape: {out.shape}")
        return out

    finally:
        if debug:
            logger.setLevel(orig_level)