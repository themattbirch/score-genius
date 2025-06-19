# backend/mlb_features/advanced.py

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

def _precompute_two_season_aggregates(
    historical_team_stats_df: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Pre-computing two-season aggregated stats for all teams...")
    if historical_team_stats_df.empty:
        return pd.DataFrame()

    stats = (
        historical_team_stats_df
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

    stats['season_win_pct'] = stats['agg_wins_all_total'] / stats['agg_games_played_all']
    stats['season_runs_for_avg'] = stats['agg_runs_for_total_all'] / stats['agg_games_played_all']
    stats['season_runs_against_avg'] = stats['agg_runs_against_total_all'] / stats['agg_games_played_all']
    stats['venue_win_pct_home'] = stats['agg_wins_home_total'] / stats['agg_games_played_home']
    stats['venue_win_pct_away'] = stats['agg_wins_away_total'] / stats['agg_games_played_away']
    stats['venue_win_advantage'] = stats['venue_win_pct_home'] - stats['season_win_pct']
    stats['venue_win_advantage_away'] = stats['venue_win_pct_away'] - stats['season_win_pct']

    final_cols = [
        'team_norm', 'season',
        'season_win_pct', 'season_runs_for_avg', 'season_runs_against_avg',
        'venue_win_pct_home', 'venue_win_pct_away',
        'venue_win_advantage', 'venue_win_advantage_away'
    ]
    return stats[final_cols].fillna(0)


def transform(
    df: pd.DataFrame,
    *,
    precomputed_stats: Optional[pd.DataFrame] = None,
    debug: bool = False,
    **kwargs
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.info("advanced.transform: Attaching pre-computed stats.")

    if precomputed_stats is None or precomputed_stats.empty:
        logger.warning("Advanced: No precomputed_stats provided. Skipping.")
        return df

    # 1) set up for fast index-join
    pre = precomputed_stats.set_index(["team_norm", "season"])
    result = df.copy()

    # 2) join home stats
    result = result.join(
        pre,
        on=["home_team_norm", "season"],
        rsuffix="_home"
    )
    # 3) join away stats
    result = result.join(
        pre,
        on=["away_team_norm", "season"],
        rsuffix="_away"
    )

    # 4) rename the joined columns to match old names
    result = result.rename(columns={
        'season_win_pct':            'home_season_win_pct',
        'season_runs_for_avg':       'home_season_runs_for_avg',
        'season_runs_against_avg':   'home_season_runs_against_avg',
        'venue_win_pct_home':        'home_venue_win_pct_home',
        'venue_win_pct_away':        'home_venue_win_pct_away',
        'venue_win_advantage':       'home_venue_win_advantage',
        'venue_win_advantage_away':  'home_venue_win_advantage_away',
        # away side
        'season_win_pct_away':       'away_season_win_pct',
        'season_runs_for_avg_away':  'away_season_runs_for_avg',
        'season_runs_against_avg_away': 'away_season_runs_against_avg',
        'venue_win_pct_home_away':   'away_venue_win_pct_home',
        'venue_win_pct_away_away':   'away_venue_win_pct_away',
        'venue_win_advantage_away':  'away_venue_win_advantage',
        'venue_win_advantage_away_away': 'away_venue_win_advantage_away',
    })

    result.fillna(0, inplace=True)
    logger.info("advanced.transform: Finished attaching stats.")
    return result
