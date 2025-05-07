# backend/features/h2h.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

# --- Logger Configuration ---
logger = logging.getLogger(__name__)

__all__ = ["transform"]

H2H_PLACEHOLDER_COLS: List[str] = [
    'matchup_num_games', 'matchup_avg_point_diff', 'matchup_home_win_pct',
    'matchup_avg_total_score', 'matchup_avg_home_score', 'matchup_avg_away_score',
    'matchup_last_date', 'matchup_streak'
]


def _default_val(col: str) -> Any:
    """
    Return default for a given placeholder column.
    Dates default to NaT, others use DEFAULTS with full key or stripped key.
    """
    if col == 'matchup_last_date':
        return pd.NaT
    return DEFAULTS.get(col, DEFAULTS.get(col.replace('matchup_', ''), 0.0))


def _get_matchup_history_single(
    *,
    home_team_norm: str,
    away_team_norm: str,
    historical_subset: pd.DataFrame,
    max_games: int = 7,
    current_game_date: Optional[pd.Timestamp] = None,
    loop_index: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    default_result: Dict[str, Any] = {col: _default_val(col) for col in H2H_PLACEHOLDER_COLS}

    # Early exit if no valid history or params
    if historical_subset is None or historical_subset.empty or max_games <= 0 or pd.isna(current_game_date):
        return default_result

    last_date_any = historical_subset['game_date'].max()

    recent_matchups = historical_subset.sort_values('game_date', ascending=False).head(max_games)
    if recent_matchups.empty:
        default_result['matchup_last_date'] = last_date_any
        return default_result

    recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce')
    recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
    recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score', 'home_team_norm', 'away_team_norm'])
    if recent_matchups.empty:
        default_result['matchup_last_date'] = last_date_any
        return default_result

    diffs_list, total_scores_list = [], []
    home_persp_scores, away_persp_scores = [], []
    home_persp_wins = 0
    current_streak = 0
    last_winner_norm = None

    for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
        h_score, a_score = game['home_score'], game['away_score']
        g_home_norm, g_away_norm = game['home_team_norm'], game['away_team_norm']
        if g_home_norm == home_team_norm:
            diff, won = h_score - a_score, h_score > a_score
            hp, ap = h_score, a_score
        elif g_away_norm == home_team_norm:
            diff, won = a_score - h_score, a_score > h_score
            hp, ap = a_score, h_score
        else:
            logger.warning(f"H2H Helper[{loop_index}]: Team mismatch: {home_team_norm} vs {away_team_norm}.")
            continue
        diffs_list.append(diff)
        total_scores_list.append(h_score + a_score)
        home_persp_scores.append(hp)
        away_persp_scores.append(ap)
        if won:
            home_persp_wins += 1
        winner_norm = home_team_norm if won else away_team_norm
        if last_winner_norm is None or winner_norm == last_winner_norm:
            current_streak = (current_streak + (1 if won else -1)) if last_winner_norm is not None else (1 if won else -1)
        else:
            current_streak = 1 if won else -1
        last_winner_norm = winner_norm

    if not diffs_list:
        default_result['matchup_last_date'] = last_date_any
        return default_result

    final_stats = {
        'matchup_num_games': len(diffs_list),
        'matchup_avg_point_diff': float(np.mean(diffs_list)),
        'matchup_home_win_pct': float(home_persp_wins / len(diffs_list)),
        'matchup_avg_total_score': float(np.mean(total_scores_list)),
        'matchup_avg_home_score': float(np.mean(home_persp_scores)),
        'matchup_avg_away_score': float(np.mean(away_persp_scores)),
        'matchup_last_date': recent_matchups['game_date'].max(),
        'matchup_streak': int(current_streak),
    }
    for col in H2H_PLACEHOLDER_COLS:
        final_stats.setdefault(col, default_result[col])
    return final_stats


def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 7,
    debug: bool = False,
) -> pd.DataFrame:
    current_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for h2h.transform")
        logger.debug(f"H2H lookback window: {max_games}")

    if df is None or df.empty:
        logger.warning("h2h.transform: Input DataFrame empty.")
        if debug: logger.setLevel(current_level)
        return pd.DataFrame()

    # Drop any existing H2H columns to ensure idempotency and avoid merge conflicts
    result_df = df.copy()
    result_df = result_df.drop(columns=H2H_PLACEHOLDER_COLS, errors='ignore')

    essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    if not all(col in result_df.columns for col in essential_cols):
        missing = set(essential_cols) - set(result_df.columns)
        logger.error(f"Missing columns: {missing}.")
        for col in H2H_PLACEHOLDER_COLS:
            result_df[col] = _default_val(col)
        if debug: logger.setLevel(current_level)
        return result_df

    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: No historical_df provided.")
        for col in H2H_PLACEHOLDER_COLS:
            result_df[col] = _default_val(col)
        if debug: logger.setLevel(current_level)
        return result_df

    # Prepare historical data
    try:
        hist_df = historical_df.dropna(subset=['game_date'], how='any').copy()
        hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
        hist_df['home_team_norm'] = hist_df['home_team'].astype(str).map(normalize_team_name)
        hist_df['away_team_norm'] = hist_df['away_team'].astype(str).map(normalize_team_name)
        hist_df['matchup_key'] = hist_df.apply(
            lambda r: "_vs_".join(sorted([r['home_team_norm'], r['away_team_norm']])), axis=1
        )
        historical_lookup = {k: grp.sort_values('game_date') for k, grp in hist_df.groupby('matchup_key', observed=True)}
    except Exception as e:
        logger.error(f"Error preparing historical data: {e}", exc_info=debug)
        for col in H2H_PLACEHOLDER_COLS:
            result_df[col] = _default_val(col)
        if debug: logger.setLevel(current_level)
        return result_df

    # Prepare target
    try:
        result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
        result_df = result_df.dropna(subset=['game_date'])
        result_df['home_team_norm'] = result_df['home_team'].astype(str).map(normalize_team_name)
        result_df['away_team_norm'] = result_df['away_team'].astype(str).map(normalize_team_name)
        result_df['matchup_key'] = result_df.apply(
            lambda r: "_vs_".join(sorted([r['home_team_norm'], r['away_team_norm']])), axis=1
        )
    except Exception as e:
        logger.error(f"Error preparing target: {e}", exc_info=debug)
        for col in H2H_PLACEHOLDER_COLS:
            result_df[col] = _default_val(col)
        if debug: logger.setLevel(current_level)
        return result_df

    logger.info(f"Calculating H2H for {len(result_df)} games...")
    h2h_results: List[Dict[str, Any]] = []
    for idx, row in result_df.iterrows():
        subset = historical_lookup.get(row['matchup_key'], pd.DataFrame())
        if not subset.empty:
            subset = subset[subset['game_date'] < row['game_date']]
        stats = _get_matchup_history_single(
            home_team_norm=row['home_team_norm'],
            away_team_norm=row['away_team_norm'],
            historical_subset=subset,
            max_games=max_games,
            current_game_date=row['game_date'],
            loop_index=idx,
            debug=debug,
        )
        h2h_results.append(stats)

    # Merge
    try:
        h2h_stats_df = pd.DataFrame(h2h_results, index=result_df.index)
        result_df = result_df.join(h2h_stats_df)
        for col in H2H_PLACEHOLDER_COLS:
            if col == 'matchup_last_date':
                result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                result_df[col] = result_df[col].fillna(result_df['game_date'])
            elif col in ['matchup_num_games', 'matchup_streak']:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(_default_val(col)).astype(int)
            else:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(_default_val(col)).astype(float)
    except Exception as e:
        logger.error(f"Error merging H2H results: {e}", exc_info=debug)
        for col in H2H_PLACEHOLDER_COLS:
            result_df[col] = _default_val(col)

    # Cleanup
    result_df = result_df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')
    logger.info("Finished adding head-to-head features.")
    if debug:
        logger.setLevel(current_level)
    return result_df
