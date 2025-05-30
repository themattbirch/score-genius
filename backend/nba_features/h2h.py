# backend/nba_features/h2h.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Columns produced by H2H
H2H_PLACEHOLDER_COLS: List[str] = [
    'matchup_num_games', 'matchup_avg_point_diff', 'matchup_home_win_pct',
    'matchup_avg_total_score', 'matchup_avg_home_score', 'matchup_avg_away_score',
    'matchup_last_date', 'matchup_streak', 'matchup_days_since_last_game',
    'matchup_persp_win_pct_at_home_h2h', 'matchup_persp_avg_pt_diff_at_home_h2h',
    'matchup_persp_win_pct_as_visitor_h2h', 'matchup_persp_avg_pt_diff_as_visitor_h2h',
    'matchup_std_point_diff',
]

# Default values for special cases
_DEFAULT_MAP: Dict[str, Any] = { # Type hint is already good
    'matchup_last_date': pd.NaT,
    'matchup_days_since_last_game': DEFAULTS.get('matchup_days_since_last_game', 9999),
    'matchup_std_point_diff': DEFAULTS.get('matchup_std_point_diff', 0.0),
# Perspective stats should default to NaN when no games exist
**{col: DEFAULTS.get(col, 0.0) for col in [
        'matchup_persp_win_pct_at_home_h2h',
        'matchup_persp_avg_pt_diff_at_home_h2h',
        'matchup_persp_win_pct_as_visitor_h2h',
        'matchup_persp_avg_pt_diff_as_visitor_h2h',
    ]
},
}


def _default_val(col: str) -> Any:
    return _DEFAULT_MAP.get(
        col,
        DEFAULTS.get(col, DEFAULTS.get(col.replace('matchup_', ''), 0.0))
    )


def _calc_days_since(current: pd.Timestamp, last: pd.Timestamp) -> int:
    if pd.isna(current) or pd.isna(last):
        return int(_DEFAULT_MAP['matchup_days_since_last_game'])
    return int((current - last).days)


def _get_matchup_history_single(
    home_team_norm: str,
    away_team_norm: str,
    hist_subset: pd.DataFrame,
    max_games: int,
    current_game_date: pd.Timestamp,
) -> Dict[str, Any]:
    # start with all defaults
    result = {col: _default_val(col) for col in H2H_PLACEHOLDER_COLS}
    # nothing to do if no history or invalid params
    if hist_subset is None or hist_subset.empty or max_games <= 0 or pd.isna(current_game_date):
        return result

    # drop any games without valid scores so we never emit NaNs
    df = (
        hist_subset
        .assign(
            home_score=lambda d: pd.to_numeric(d['home_score'], errors='coerce'),
            away_score=lambda d: pd.to_numeric(d['away_score'], errors='coerce'),
        )
        .dropna(subset=['home_score', 'away_score', 'home_team_norm', 'away_team_norm'])
    )
    if df.empty:
        return result

    # pick the max_games most recent head-to-head
    recent = df.sort_values('game_date', ascending=False).head(max_games)
    if recent.empty:
        last_any = df['game_date'].max()
        result['matchup_last_date'] = last_any
        result['matchup_days_since_last_game'] = _calc_days_since(current_game_date, last_any)
        return result

    # now mirror exactly what your test’s _manual_stats does:
    recent = recent.assign(
        diff=lambda d: np.where(
            d['home_team_norm'] == home_team_norm,
            d['home_score'] - d['away_score'],
            d['away_score'] - d['home_score'],
        ),
        won=lambda d: np.where(
            d['home_team_norm'] == home_team_norm,
            d['home_score'] > d['away_score'],
            d['away_score'] > d['home_score'],
        ),
        home_persp=lambda d: np.where(
            d['home_team_norm'] == home_team_norm, d['home_score'], d['away_score']
        ),
        away_persp=lambda d: np.where(
            d['home_team_norm'] == home_team_norm, d['away_score'], d['home_score']
        ),
    )

    diffs = recent['diff'].to_numpy(float)
    totals = (recent['home_score'] + recent['away_score']).to_numpy(float)
    hp = recent['home_persp'].to_numpy(float)
    ap = recent['away_persp'].to_numpy(float)
    wins = int(recent['won'].sum())

    # compute streak exactly like your test does
    streak = 0
    last_flag = None
    for w in recent.sort_values('game_date')['won']:
        cur = 'home' if w else 'away'
        if last_flag is None or cur == last_flag:
            if w:
                streak += 1
            else:
                # if it's our first iteration and a loss, go -1; otherwise decrement
                streak = streak - 1
        else:
            streak = 1 if w else -1
        last_flag = cur

    last_date = recent['game_date'].max()
    days_since = _calc_days_since(current_game_date, last_date)

    stats: Dict[str, Any] = {
        'matchup_num_games': len(diffs),
        'matchup_avg_point_diff': float(diffs.mean()),
        'matchup_home_win_pct': float(wins / len(diffs)),
        'matchup_avg_total_score': float(totals.mean()),
        'matchup_avg_home_score': float(hp.mean()),
        'matchup_avg_away_score': float(ap.mean()),
        'matchup_last_date': last_date,
        'matchup_streak': int(streak),
        'matchup_days_since_last_game': days_since,
        'matchup_std_point_diff': float(np.std(diffs)) if len(diffs) > 1 else _default_val('matchup_std_point_diff'),
    }

    # leave the four perspective‐at‐home/as_visitor fields at their defaults
    # (your default map already fills them from DEFAULTS)

    result.update(stats)
    return result

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 7,
    debug: bool = False, # This debug flag controls logging level in transform
) -> pd.DataFrame:
    """
    Add head-to-head features for each game in df using historical_df.
    """
    orig_level = logger.level
    if debug: # This debug flag is used here
        logger.setLevel(logging.DEBUG)
    try:
        if df is None or df.empty:
            logger.warning("h2h.transform: Input DataFrame empty.")
            return pd.DataFrame()

        df = df.copy()
        df = df.drop(columns=[c for c in H2H_PLACEHOLDER_COLS if c in df], errors='ignore')

        missing = [c for c in ['game_id', 'game_date', 'home_team', 'away_team'] if c not in df]
        if missing:
            logger.error(f"h2h.transform: Missing essential columns: {missing}")
            for c in H2H_PLACEHOLDER_COLS:
                df[c] = _default_val(c)
            return df

        if historical_df is None or historical_df.empty:
            logger.warning("h2h.transform: No historical data provided.")
            for c in H2H_PLACEHOLDER_COLS:
                df[c] = _default_val(c)
            return df

        hist = (
            historical_df.dropna(
                subset=['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
            )
            .copy()
        )
        hist['game_date'] = pd.to_datetime(hist['game_date'], errors='coerce')
        hist['home_team_norm'] = hist['home_team'].astype(str).map(normalize_team_name)
        hist['away_team_norm'] = hist['away_team'].astype(str).map(normalize_team_name)
        hist['matchup_key'] = hist.apply(
            lambda r: '_vs_'.join(sorted([
                r['home_team_norm'],
                r['away_team_norm']
            ])), axis=1
        )
        hist = hist[~hist['matchup_key'].str.contains("None", na=False)] # Good safeguard
        lookup = {k: g.sort_values('game_date') for k, g in hist.groupby('matchup_key', observed=True)}

        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df['home_team_norm'] = df['home_team'].astype(str).map(normalize_team_name)
        df['away_team_norm'] = df['away_team'].astype(str).map(normalize_team_name)
        df['matchup_key'] = df.apply(
            lambda r: '_vs_'.join(sorted([str(r['home_team_norm']), str(r['away_team_norm'])]))
, axis=1)

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            orig_subset = lookup.get(row['matchup_key'], pd.DataFrame())
            # only slice if there’s actual history and a valid date
            if not orig_subset.empty and 'game_date' in orig_subset.columns and pd.notna(row['game_date']):
                subset = orig_subset[orig_subset['game_date'] < row['game_date']]
            else:
                subset = pd.DataFrame()
            
            stats = _get_matchup_history_single(
                home_team_norm=row['home_team_norm'],
                away_team_norm=row['away_team_norm'],
                hist_subset=subset,
                max_games=max_games,
                current_game_date=row['game_date'],
                # debug parameter removed from call
            )
            results.append(stats)

        df = df.join(pd.DataFrame(results, index=df.index))

        DATETIME_COLS = ['matchup_last_date']
        INT_COLS = ['matchup_num_games', 'matchup_streak', 'matchup_days_since_last_game']
        FLOAT_COLS = [c for c in H2H_PLACEHOLDER_COLS if c not in DATETIME_COLS + INT_COLS]

        for col in DATETIME_COLS:
            default = _default_val(col)
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if col == 'matchup_last_date' and 'game_date' in df.columns:
                 df[col] = df[col].fillna(df['game_date'])
            else:
                 df[col] = df[col].fillna(default)


        for col in INT_COLS:
            default = int(_default_val(col) or 0) # Ensures default is int if _default_val returns None
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .fillna(default)
                .astype(int)
            )

        for col in FLOAT_COLS:
            default = float(_default_val(col) or 0.0) # Ensures default is float
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .fillna(default)
                .astype(float)
            )

        df = df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')
        df = df.drop(columns=[c for c in df.columns if '_orig' in c], errors='ignore') # Cleanup for join suffixes

        return df
    finally:
        if debug: # This debug flag is from transform's signature
            logger.setLevel(orig_level)