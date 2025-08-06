# backend/nba_features/rolling.py
from __future__ import annotations
import logging
from typing import Sequence, Optional
import pandas as pd
import numpy as np

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]


def _lagged_rolling_stat(s: pd.Series, window: int, min_periods: int, stat_func: str) -> pd.Series:
    # (This original helper function is preserved)
    if s.empty:
        return s.copy()
    shifted = s.shift(1)
    if shifted.index.has_duplicates:
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
    return pd.Series(primary.fillna(fallback).values, index=s.index, name=s.name)


def transform(
    df: pd.DataFrame,
    *,
    # --- MODIFIED: Added our new argument ---
    precomputed_rolling_features_df: Optional[pd.DataFrame] = None,
    window_sizes: Sequence[int] = (5, 10, 20),
    flag_imputation: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add leakage-free rolling mean/std features for home/away team stats.
    If precomputed_rolling_features_df is provided, it will be used as a shortcut.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        if df.empty:
            logger.warning("rolling.transform: empty input, returning copy")
            return df.copy()

        # --- NEW: Shortcut logic using pre-computed features from our RPC ---
        if precomputed_rolling_features_df is not None and not precomputed_rolling_features_df.empty:
            logger.info("Using pre-computed NBA rolling features DataFrame passed from engine.")
            
            out = df.copy()
            features = precomputed_rolling_features_df.copy()

            if 'home_norm' not in out.columns:
                out['home_norm'] = out['home_team'].astype(str).map(normalize_team_name)
            if 'away_norm' not in out.columns:
                out['away_norm'] = out['away_team'].astype(str).map(normalize_team_name)
            
            features = features.rename(columns={'team_id': 'team_norm'})
            feature_cols = [col for col in features.columns if col.startswith('rolling_')]
            
            home_features_to_merge = features[['team_norm'] + feature_cols].add_prefix('home_')
            out = pd.merge(out, home_features_to_merge, left_on='home_norm', right_on='home_team_norm', how='left')

            away_features_to_merge = features[['team_norm'] + feature_cols].add_prefix('away_')
            out = pd.merge(out, away_features_to_merge, left_on='away_norm', right_on='away_team_norm', how='left')
            
            logger.debug(f"Successfully merged pre-computed NBA rolling features. Final shape: {out.shape}")
            return out.drop(columns=['home_team_norm', 'away_team_norm'], errors='ignore')
        # --- End of new logic ---

        # --- ORIGINAL LOGIC (now serves as a fallback, preserved from your file) ---
        else:
            logger.info("No pre-computed NBA rolling features provided. Calculating from scratch...")
            out = df.copy()
            stat_mapping = {
                ('home_score', 'away_score'): 'score_for',
                ('home_offensive_rating', 'away_offensive_rating'): 'off_rating',
                ('home_defensive_rating', 'away_defensive_rating'): 'def_rating',
                ('home_net_rating', 'away_net_rating'): 'net_rating',
            }
            needed_cols = {'game_id', 'game_date', 'home_team', 'away_team'}.union(*stat_mapping.keys())
            
            missing = needed_cols - set(out.columns)
            if missing:
                logger.error(f"rolling.transform: missing required columns {missing}, skipping.")
                return out

            out['game_date'] = pd.to_datetime(out['game_date'], errors='coerce').dt.tz_localize(None)
            out = out.dropna(subset=['game_date']).copy()
            if out.empty: return out

            out['home_norm'] = out['home_team'].astype(str).map(normalize_team_name)
            out['away_norm'] = out['away_team'].astype(str).map(normalize_team_name)

            records = []
            for _, row in out.iterrows():
                for (h_col, a_col), stat_name in stat_mapping.items():
                    home_val = pd.to_numeric(row.get(h_col), errors='coerce')
                    if pd.notna(home_val) and pd.notna(row['home_norm']):
                        records.append({'game_id': row['game_id'], 'team': row['home_norm'], 'game_date': row['game_date'], 'stat': stat_name, 'val': home_val})
                    
                    away_val = pd.to_numeric(row.get(a_col), errors='coerce')
                    if pd.notna(away_val) and pd.notna(row['away_norm']):
                        records.append({'game_id': row['game_id'], 'team': row['away_norm'], 'game_date': row['game_date'], 'stat': stat_name, 'val': away_val})
            
            if not records:
                return out.drop(columns=[c for c in ['home_norm', 'away_norm'] if c in out.columns], errors='ignore')

            long_df = pd.DataFrame.from_records(records)
            long_df = long_df.set_index('game_date').sort_values(['team', 'stat', 'game_date', 'game_id'])

            unique_stats = long_df['stat'].unique()
            default_means = {s: DEFAULTS.get(s, 0.0) for s in unique_stats}
            default_stds = {s: DEFAULTS.get(f"{s}_std", 0.0) for s in unique_stats}

            window_feature_pieces = []
            for w_size in window_sizes:
                min_p = max(1, w_size // 2)
                df_window_long = long_df.copy()
                grouped = df_window_long.groupby(['team', 'stat'], observed=True, group_keys=False)
                df_window_long[f'mean_{w_size}'] = grouped['val'].transform(lambda s: _lagged_rolling_stat(s, w_size, min_p, 'mean'))
                df_window_long[f'std_{w_size}'] = grouped['val'].transform(lambda s: _lagged_rolling_stat(s, w_size, min_p, 'std'))
                if flag_imputation:
                    df_window_long[f'imp_mean_{w_size}'] = df_window_long[f'mean_{w_size}'].isnull()
                    df_window_long[f'imp_std_{w_size}'] = df_window_long[f'std_{w_size}'].isnull()
                df_window_long[f'mean_{w_size}'] = df_window_long[f'mean_{w_size}'].fillna(df_window_long['stat'].map(default_means))
                df_window_long[f'std_{w_size}'] = df_window_long[f'std_{w_size}'].fillna(df_window_long['stat'].map(default_stds)).clip(lower=0.0)
                df_window_long = df_window_long.reset_index()
                pivot_index = ['game_id', 'team']
                mean_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'mean_{w_size}', aggfunc='first').rename(columns=lambda c: f'rolling_{c}_mean_{w_size}')
                std_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'std_{w_size}', aggfunc='first').rename(columns=lambda c: f'rolling_{c}_std_{w_size}')
                current_piece = pd.concat([mean_pivot, std_pivot], axis=1)
                if flag_imputation:
                    imp_mean_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'imp_mean_{w_size}', aggfunc='first').rename(columns=lambda c: f'rolling_{c}_mean_{w_size}_imputed')
                    imp_std_pivot = df_window_long.pivot_table(index=pivot_index, columns='stat', values=f'imp_std_{w_size}', aggfunc='first').rename(columns=lambda c: f'rolling_{c}_std_{w_size}_imputed')
                    current_piece = pd.concat([current_piece, imp_mean_pivot, imp_std_pivot], axis=1)
                window_feature_pieces.append(current_piece.reset_index())
            
            if not window_feature_pieces:
                 return out.drop(columns=[c for c in ['home_norm', 'away_norm'] if c in out.columns], errors='ignore')

            wide_features_df = window_feature_pieces[0]
            for piece in window_feature_pieces[1:]:
                wide_features_df = wide_features_df.merge(piece, on=['game_id', 'team'], how='outer')

            for side in ('home', 'away'):
                side_rename_map = {col: f'{side}_{col}' for col in wide_features_df.columns if col not in ['game_id', 'team']}
                data_to_merge_for_side = wide_features_df.rename(columns={'team': 'merge_team_norm', **side_rename_map})
                cols_for_this_merge = ['game_id', 'merge_team_norm'] + list(side_rename_map.values())
                out = out.merge(data_to_merge_for_side[cols_for_this_merge], left_on=['game_id', f'{side}_norm'], right_on=['game_id', 'merge_team_norm'], how='left')
                if 'merge_team_norm' in out.columns:
                    out = out.drop(columns=['merge_team_norm'])
            
            all_added_rolling_cols = [col for col in out.columns if col.startswith(('home_rolling_', 'away_rolling_'))]
            for col_name in all_added_rolling_cols:
                if col_name.endswith('_imputed'):
                     out[col_name] = out[col_name].fillna(False).astype(bool)
                else:
                    default_val_for_col = 0.0
                    try:
                        parts = col_name.split('_'); rolling_idx = parts.index('rolling')
                        stat_name_parts = []; is_std_col = False
                        for i in range(rolling_idx + 1, len(parts)):
                            part_val = parts[i]
                            if part_val == 'mean': break
                            if part_val == 'std': is_std_col = True; break
                            stat_name_parts.append(part_val)
                        base_stat_name = "_".join(stat_name_parts)
                        if is_std_col: default_val_for_col = default_stds.get(base_stat_name, 0.0)
                        else: default_val_for_col = default_means.get(base_stat_name, 0.0)
                    except (ValueError, IndexError): pass
                    out[col_name] = pd.to_numeric(out[col_name], errors='coerce').fillna(default_val_for_col)
                    if 'std' in col_name and not col_name.endswith('_imputed'):
                        out[col_name] = out[col_name].clip(lower=0.0)
                    out[col_name] = out[col_name].astype(float)
            
            return out.drop(columns=[c for c in ['home_norm', 'away_norm'] if c in out.columns], errors='ignore')

    finally:
        if debug:
            logger.setLevel(orig_level)