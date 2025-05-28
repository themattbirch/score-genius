# backend/nba_features/season.py

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, determine_season

# Get logger for this module
logger = logging.getLogger(__name__)

__all__ = ["transform"]

# Helper to compute previous season string
def _previous_season(season_str: str) -> str:
    try:
        start, end = season_str.split('-')
        prev_start = int(start) - 1
        prev_end = (prev_start + 1) % 100
        return f"{prev_start}-{prev_end:02d}"
    except Exception:
        return season_str


def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Attach previous-season statistical context for home and away teams.
    - Falls back to defaults if stats are unavailable or missing.
    - Creates boolean flags '<feature>_imputed' when values come from DEFAULTS.

    Args:
        df: Game-level DataFrame with columns ['game_id','game_date','home_team','away_team']
        team_stats_df: Optional DataFrame with ['team_name','season','wins_all_percentage',
                       'points_for_avg_all','points_against_avg_all','current_form']
        flag_imputations: If True, create boolean flags marking default-filled features.
        debug: If True, enable debug logging for this function.

    Returns:
        DataFrame with added season context features, optional flags, and diffs/net ratings.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Adding season context features...")
    result = df.copy()

    # Ensure essential columns exist
    for col in ['game_id', 'game_date', 'home_team', 'away_team']:
        if col not in result.columns:
            logger.error(f"Missing essential column: {col}, skipping season features.")
            if debug:
                logger.setLevel(orig_level)
            return result

    # Coerce to datetime and drop invalid
    result['game_date'] = pd.to_datetime(result['game_date'], errors='coerce')
    result = result.dropna(subset=['game_date'])
    if result.empty:
        if debug:
            logger.setLevel(orig_level)
        return result

    # Default placeholders
    placeholder_defaults = {
        'home_season_win_pct': DEFAULTS['win_pct'],
        'away_season_win_pct': DEFAULTS['win_pct'],
        'home_season_avg_pts_for': DEFAULTS['avg_pts_for'],
        'away_season_avg_pts_for': DEFAULTS['avg_pts_for'],
        'home_season_avg_pts_against': DEFAULTS['avg_pts_against'],
        'away_season_avg_pts_against': DEFAULTS['avg_pts_against'],
        'home_current_form': 'N/A',
        'away_current_form': 'N/A'
    }

    # Quick path: no stats
    if team_stats_df is None or team_stats_df.empty:
        logger.warning("No team_stats_df provided - using defaults for all season features.")
        for feat, default in placeholder_defaults.items():
            result[feat] = default
            if flag_imputations:
                result[f"{feat}_imputed"] = True
        # Diffs and net ratings all zero
        zeros = ['season_win_pct_diff','season_pts_for_diff','season_pts_against_diff',
                 'home_season_net_rating','away_season_net_rating','season_net_rating_diff']
        for z in zeros:
            result[z] = 0.0
        if debug:
            logger.setLevel(orig_level)
        return result

    # Prepare season lookup keys
    result['season'] = result['game_date'].apply(determine_season)
    result['season_stats'] = result['season'].apply(_previous_season)
    result['home_norm'] = result['home_team'].map(normalize_team_name)
    result['away_norm'] = result['away_team'].map(normalize_team_name)

    # Prepare team_stats_df
    ts = team_stats_df.copy()
    ts['team_norm'] = ts['team_name'].map(normalize_team_name)
    ts['season'] = ts['season'].astype(str)
    ts['key'] = ts['team_norm'] + '_' + ts['season']
    ts = ts.drop_duplicates(subset=['key'], keep='last').set_index('key')
    sub = ts[['wins_all_percentage','points_for_avg_all','points_against_avg_all','current_form']]

    # Merge for home and away sides
    for side in ['home','away']:
        prefix = f"{side}_"
        # rename columns for merge
        renamed = sub.rename(columns={
            'wins_all_percentage': f"{prefix}season_win_pct",
            'points_for_avg_all': f"{prefix}season_avg_pts_for",
            'points_against_avg_all': f"{prefix}season_avg_pts_against",
            'current_form': f"{prefix}current_form"
        })
        # merge on composite key = norm + '_' + season_stats
        key_col = result[f"{side}_norm"] + '_' + result['season_stats']
        result = result.merge(
            renamed,
            how='left',
            left_on=key_col,
            right_index=True
        )
        # imputation flags
        if flag_imputations:
            for feat in [f"{prefix}season_win_pct", f"{prefix}season_avg_pts_for",
                         f"{prefix}season_avg_pts_against", f"{prefix}current_form"]:
                result[f"{feat}_imputed"] = result[feat].isna().astype(bool)

    # Fill defaults where missing
    for feat, default in placeholder_defaults.items():
        if feat not in result.columns:
            result[feat] = default
            if flag_imputations:
                result[f"{feat}_imputed"] = True
        else:
            if isinstance(default, str):
                result[feat] = result[feat].fillna(default).astype(str)
            else:
                result[feat] = pd.to_numeric(result[feat], errors='coerce').fillna(default)

    # Compute diffs & net ratings
    result['season_win_pct_diff'] = result['home_season_win_pct'] - result['away_season_win_pct']
    result['season_pts_for_diff'] = result['home_season_avg_pts_for'] - result['away_season_avg_pts_for']
    result['season_pts_against_diff'] = result['home_season_avg_pts_against'] - result['away_season_avg_pts_against']
    result['home_season_net_rating'] = result['home_season_avg_pts_for'] - result['home_season_avg_pts_against']
    result['away_season_net_rating'] = result['away_season_avg_pts_for'] - result['away_season_avg_pts_against']
    result['season_net_rating_diff'] = result['home_season_net_rating'] - result['away_season_net_rating']

    # Drop helper columns
    result.drop(columns=['season','season_stats','home_norm','away_norm'], inplace=True, errors='ignore')

    # Reorder columns: original columns, seasonal feats, flags, diffs/net
    original = list(df.columns)
    seasonal = list(placeholder_defaults.keys())
    flags = [c for c in result.columns if c.endswith('_imputed')]
    diffs = ['season_win_pct_diff','season_pts_for_diff','season_pts_against_diff',
             'home_season_net_rating','away_season_net_rating','season_net_rating_diff']
    ordered = original + seasonal + flags + diffs
    # Keep only columns that exist
    result = result[[c for c in ordered if c in result.columns]]

    logger.info("Finished adding season context features.")
    if debug:
        logger.setLevel(orig_level)
    return result
