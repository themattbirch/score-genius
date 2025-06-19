# backend/mlb_features/handedness_for_display.py

import time
import logging
from typing import Optional, Any

import numpy as np
import pandas as pd

from .utils import normalize_team_name

logger = logging.getLogger(__name__)

def transform(
    df: pd.DataFrame,
    *,
    historical_team_stats_df: Optional[pd.DataFrame] = None,
    mlb_pitcher_splits_df: Optional[pd.DataFrame] = None,
    home_team_col_param: str = "home_team_norm",
    away_team_col_param: str = "away_team_norm",
    home_pitcher_hand_col: str = "home_starter_pitcher_handedness",
    away_pitcher_hand_col: str = "away_starter_pitcher_handedness",
    flag_imputations: bool = True,
    debug: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    start_total = time.time()
    logger.info("handedness: starting full transform")

    result = df.copy()
    default = 4.3

    # --- SNAPSHOT MODE ---
    if mlb_pitcher_splits_df is not None:
        t0 = time.time()
        splits = mlb_pitcher_splits_df.copy()

        if 'team_norm' not in splits.columns and 'team_id' in splits.columns:
            splits['team_norm'] = splits['team_id'].apply(normalize_team_name)
        lookup = splits.set_index('team_norm')[['season_avg_runs_vs_lhp', 'season_avg_runs_vs_rhp']]

        # vectorized map
        home_norm = result[home_team_col_param]
        away_norm = result[away_team_col_param]
        lhp = lookup['season_avg_runs_vs_lhp']
        rhp = lookup['season_avg_runs_vs_rhp']

        home_l = home_norm.map(lhp).fillna(0.0).to_numpy()
        home_r = home_norm.map(rhp).fillna(0.0).to_numpy()
        away_l = away_norm.map(lhp).fillna(0.0).to_numpy()
        away_r = away_norm.map(rhp).fillna(0.0).to_numpy()

        logger.info(f"handedness(snapshot): built lookup arrays in {time.time() - t0:.2f}s")

        t1 = time.time()
        home_opp = result[away_pitcher_hand_col].astype(str).str[:1].str.upper().to_numpy()
        away_opp = result[home_pitcher_hand_col].astype(str).str[:1].str.upper().to_numpy()

        result['h_team_off_avg_runs_vs_opp_hand'] = np.where(home_opp=='L', home_l, home_r)
        result['a_team_off_avg_runs_vs_opp_hand'] = np.where(away_opp=='L', away_l, away_r)

        logger.info(f"handedness(snapshot): computed selections in {time.time() - t1:.2f}s")

    # --- PIPELINE MODE (merge-based) ---
    elif historical_team_stats_df is not None:
        t0 = time.time()
        # Keep only one row per team_norm
        stats = (
            historical_team_stats_df[["team_norm", "season_avg_runs_vs_lhp", "season_avg_runs_vs_rhp"]]
            .drop_duplicates(subset="team_norm", keep="last")
        )

        # Merge home stats
        result = result.merge(
            stats.add_prefix("home_"),
            left_on=home_team_col_param,
            right_on="home_team_norm",
            how="left",
        )
        # Merge away stats
        result = result.merge(
            stats.add_prefix("away_"),
            left_on=away_team_col_param,
            right_on="away_team_norm",
            how="left",
        )
        logger.info(f"handedness(pipeline): merged stats in {time.time() - t0:.2f}s")

        t1 = time.time()
        # Determine opponent hand
        home_opp = result[away_pitcher_hand_col].fillna("").str[:1].str.upper()
        away_opp = result[home_pitcher_hand_col].fillna("").str[:1].str.upper()

        result["h_team_off_avg_runs_vs_opp_hand"] = np.where(
            home_opp == "L", result["home_season_avg_runs_vs_lhp"], result["home_season_avg_runs_vs_rhp"]
        )
        result["a_team_off_avg_runs_vs_opp_hand"] = np.where(
            away_opp == "L", result["away_season_avg_runs_vs_lhp"], result["away_season_avg_runs_vs_rhp"]
        )
        logger.info(f"handedness(pipeline): np.where selections in {time.time() - t1:.2f}s")

        # Clean up merge artifacts
        result.drop(
            columns=[
                "home_team_norm", "away_team_norm",
                "home_season_avg_runs_vs_lhp", "home_season_avg_runs_vs_rhp",
                "away_season_avg_runs_vs_lhp", "away_season_avg_runs_vs_rhp"
            ],
            inplace=True, errors=True
        )

    else:
        logger.warning("Handedness: no splits or historical stats—defaulting to zero")
        result['h_team_off_avg_runs_vs_opp_hand'] = 0.0
        result['a_team_off_avg_runs_vs_opp_hand'] = 0.0

    # Fill and flag
    for col in ('h_team_off_avg_runs_vs_opp_hand','a_team_off_avg_runs_vs_opp_hand'):
        if flag_imputations:
            result[f'{col}_imputed'] = result[col].isna().astype(int)
        result[col] = result[col].fillna(default)

    logger.info(f"handedness: total transform took {time.time() - start_total:.2f}s")
    return result
