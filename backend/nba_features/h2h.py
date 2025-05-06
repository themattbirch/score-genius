# backend/features/h2h.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ["transform"]

# Placeholder columns
H2H_PLACEHOLDER_COLS: List[str] = [
    'matchup_num_games',
    'matchup_avg_point_diff',
    'matchup_home_win_pct',
    'matchup_avg_total_score',
    'matchup_avg_home_score',
    'matchup_avg_away_score',
    'matchup_last_date',
    'matchup_streak'
]


def _get_matchup_history_single(
    *,
    home_team_norm: str,
    away_team_norm: str,
    historical_subset: pd.DataFrame,
    max_games: int = 7,
    current_game_date: Optional[pd.Timestamp] = None,
    loop_index: Optional[int] = None,
    debug: bool = False
) -> Dict[str, Any]:
    # Build defaults
    default_result: Dict[str, Any] = {}
    for col in H2H_PLACEHOLDER_COLS:
        if col == 'matchup_last_date':
            default_result[col] = pd.NaT
        else:
            default_result[col] = DEFAULTS.get(col, DEFAULTS.get(col.replace('matchup_', ''), 0.0))

    # No history or invalid date
    if historical_subset is None or historical_subset.empty or pd.isna(current_game_date):
        if debug:
            logger.debug(f"H2H Helper[{loop_index}]: returning defaults")
        return default_result

    # Take most recent max_games
    recent = (
        historical_subset
        .sort_values('game_date', ascending=False)
        .head(max_games)
    )
    if recent.empty:
        return default_result

    # Ensure numeric and fill missing scores as zero
    recent['home_score'] = pd.to_numeric(recent['home_score'], errors='coerce').fillna(0.0)
    recent['away_score'] = pd.to_numeric(recent['away_score'], errors='coerce').fillna(0.0)

    # Compute per-game stats
    diffs, totals, hp, ap = [], [], [], []
    wins = 0
    streak = 0
    last_winner = None

    # Iterate oldest→newest for streak
    for _, g in recent.sort_values('game_date', ascending=True).iterrows():
        h_score = g['home_score']
        a_score = g['away_score']

        if g['home_team_norm'] == home_team_norm:
            diff = h_score - a_score
            won = h_score > a_score
            home_persp, away_persp = h_score, a_score
        else:
            diff = a_score - h_score
            won = a_score > h_score
            home_persp, away_persp = a_score, h_score

        diffs.append(diff)
        totals.append(h_score + a_score)
        hp.append(home_persp)
        ap.append(away_persp)
        if won:
            wins += 1

        winner = home_team_norm if won else away_team_norm
        if last_winner is None:
            streak = 1 if won else -1
        elif winner == last_winner:
            streak += 1 if won else -1
        else:
            streak = 1 if won else -1
        last_winner = winner

    n = len(diffs)
    stats: Dict[str, Any] = {
        'matchup_num_games': n,
        'matchup_avg_point_diff': float(np.mean(diffs)),
        'matchup_home_win_pct': float(wins / n),
        'matchup_avg_total_score': float(np.mean(totals)),
        'matchup_avg_home_score': float(np.mean(hp)),
        'matchup_avg_away_score': float(np.mean(ap)),
        'matchup_last_date': recent['game_date'].max(),
        'matchup_streak': int(streak),
    }

    # Ensure all placeholders present
    for col in H2H_PLACEHOLDER_COLS:
        stats.setdefault(col, default_result[col])

    return stats


def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 7,
    debug: bool = False
) -> pd.DataFrame:
    # Optional DEBUG
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for h2h.transform")

    if df is None or df.empty:
        if debug:
            logger.setLevel(orig_level)
        return pd.DataFrame()

    out = df.copy()

    # Drop any existing H2H columns to start fresh
    for col in H2H_PLACEHOLDER_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])

    # Ensure essential columns
    essential = ['game_id', 'game_date', 'home_team', 'away_team']
    if not all(c in out.columns for c in essential):
        for c in H2H_PLACEHOLDER_COLS:
            val = pd.NaT if c == 'matchup_last_date' else DEFAULTS.get(c, DEFAULTS.get(c.replace('matchup_', ''), 0.0))
            out[c] = val
        if debug:
            logger.setLevel(orig_level)
        return out

    # No historical data → fill defaults
    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: No historical_df provided. Filling defaults.")
        for c in H2H_PLACEHOLDER_COLS:
            out[c] = pd.NaT if c == 'matchup_last_date' else DEFAULTS.get(c, DEFAULTS.get(c.replace('matchup_', ''), 0.0))
        if debug:
            logger.setLevel(orig_level)
        return out

    # Prepare historical DataFrame
    hist = (
        historical_df
        .dropna(subset=['game_date', 'home_team', 'away_team'])
        .copy()
    )
    hist['game_date'] = pd.to_datetime(hist['game_date'], errors='coerce').dt.tz_localize(None)
    hist['home_score'] = pd.to_numeric(hist['home_score'], errors='coerce').fillna(0.0)
    hist['away_score'] = pd.to_numeric(hist['away_score'], errors='coerce').fillna(0.0)
    hist['home_team_norm'] = hist['home_team'].astype(str).map(normalize_team_name)
    hist['away_team_norm'] = hist['away_team'].astype(str).map(normalize_team_name)
    hist['matchup_key'] = hist.apply(
        lambda r: '_vs_'.join(sorted([r['home_team_norm'], r['away_team_norm']])),
        axis=1
    )
    lookup = {
        key: group.sort_values('game_date')
        for key, group in hist.groupby('matchup_key', observed=True)
    }

    # Prepare target DataFrame
    out['game_date'] = pd.to_datetime(out['game_date'], errors='coerce').dt.tz_localize(None)
    out['home_team_norm'] = out['home_team'].astype(str).map(normalize_team_name)
    out['away_team_norm'] = out['away_team'].astype(str).map(normalize_team_name)
    out['matchup_key'] = out.apply(
        lambda r: '_vs_'.join(sorted([r['home_team_norm'], r['away_team_norm']])),
        axis=1
    )

    # Compute H2H for each row
    results: List[Dict[str, Any]] = []
    for idx, row in out.iterrows():
        key = row['matchup_key']
        hist_sub = lookup.get(key, pd.DataFrame()).copy()
        hist_sub = hist_sub[hist_sub['game_date'] < row['game_date']]
        stats = _get_matchup_history_single(
            home_team_norm=row['home_team_norm'],
            away_team_norm=row['away_team_norm'],
            historical_subset=hist_sub,
            max_games=max_games,
            current_game_date=row['game_date'],
            loop_index=idx,
            debug=debug
        )
        results.append(stats)

    stats_df = pd.DataFrame(results, index=out.index)
    out = out.join(stats_df)

    # Final fill, type enforcement, no NaNs
    for c in H2H_PLACEHOLDER_COLS:
        if c not in out.columns:
            out[c] = pd.NaT if c == 'matchup_last_date' else DEFAULTS.get(c, DEFAULTS.get(c.replace('matchup_', ''), 0.0))
        if c == 'matchup_last_date':
            out[c] = pd.to_datetime(out[c], errors='coerce')
        elif c in ['matchup_num_games', 'matchup_streak']:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(int)
        else:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).astype(float)

    # Cleanup
    out = out.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')

    if debug:
        logger.setLevel(orig_level)

    return out
