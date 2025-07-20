# backend/mlb_features/advanced.py

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _precompute_two_season_aggregates(
    historical_team_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Efficiently calculates two-season rolling aggregated stats for all teams.
    This is called ONCE by the engine before the main processing loop.
    """
    logger.info("Pre-computing two-season aggregated stats for all teams...")
    if historical_team_stats_df.empty:
        return pd.DataFrame()

    stats = (
        historical_team_stats_df.copy()
        .sort_values(["team_norm", "season"])
        .reset_index(drop=True)
    )

    total_cols = [
        'wins_all_total', 'games_played_all', 'runs_for_total_all',
        'runs_against_total_all', 'wins_home_total', 'games_played_home',
        'wins_away_total', 'games_played_away'
    ]

    grp = stats.groupby("team_norm", group_keys=False)
    for col in total_cols:
        stats[f"agg_{col}"] = grp[col].rolling(2, min_periods=1).sum().values

    # Calculate metrics from aggregated totals, safely handling division by zero
    stats['season_win_pct'] = np.divide(stats['agg_wins_all_total'], stats['agg_games_played_all'], where=stats['agg_games_played_all']!=0, out=np.full_like(stats['agg_wins_all_total'], 0.5, dtype=float))
    stats['season_runs_for_avg'] = np.divide(stats['agg_runs_for_total_all'], stats['agg_games_played_all'], where=stats['agg_games_played_all']!=0, out=np.full_like(stats['agg_runs_for_total_all'], 0, dtype=float))
    stats['season_runs_against_avg'] = np.divide(stats['agg_runs_against_total_all'], stats['agg_games_played_all'], where=stats['agg_games_played_all']!=0, out=np.full_like(stats['agg_runs_against_total_all'], 0, dtype=float))
    stats['venue_win_pct_home'] = np.divide(stats['agg_wins_home_total'], stats['agg_games_played_home'], where=stats['agg_games_played_home']!=0, out=np.full_like(stats['agg_wins_home_total'], 0.5, dtype=float))
    stats['venue_win_pct_away'] = np.divide(stats['agg_wins_away_total'], stats['agg_games_played_away'], where=stats['agg_games_played_away']!=0, out=np.full_like(stats['agg_wins_away_total'], 0.5, dtype=float))
    
    final_cols = [
        'team_norm', 'season', 'season_win_pct', 'season_runs_for_avg', 'season_runs_against_avg',
        'venue_win_pct_home', 'venue_win_pct_away'
    ]
    return stats[final_cols].fillna(0)


def transform(
    df: pd.DataFrame,
    *,
    precomputed_stats: Optional[pd.DataFrame] = None,
    debug: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Attaches pre-computed aggregated team statistics via a fast merge."""
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.info("advanced.transform: Attaching pre-computed stats.")

    if precomputed_stats is None or precomputed_stats.empty:
        logger.warning("Advanced: No precomputed_stats provided. Skipping.")
        return df

    pre = precomputed_stats.set_index(["team_norm", "season"])
    result = df.copy()

    # Join home stats and rename columns
    home_cols = {col: f'home_{col}' for col in pre.columns}
    result = result.join(pre.rename(columns=home_cols), on=["home_team_norm", "season"])
    
    # Join away stats and rename columns
    away_cols = {col: f'away_{col}' for col in pre.columns}
    result = result.join(pre.rename(columns=away_cols), on=["away_team_norm", "season"])

    # Calculate the derived 'advantage' metric after merges
    result['home_venue_win_advantage'] = result['home_venue_win_pct_home'] - result['home_season_win_pct']
    result['away_venue_win_advantage'] = result['away_venue_win_pct_away'] - result['away_season_win_pct']
    
    result.fillna(0, inplace=True)
    logger.info("advanced.transform: Finished attaching stats.")
    return result