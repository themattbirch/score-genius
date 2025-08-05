# backend/nfl_features/season.py
"""Previous-season context features for NFL matchups.

-- MODIFIED to use pre-aggregated data from the season_stats_df RPC --

This transformer joins team-level aggregates from the **previous season** onto
an input `games` DataFrame.
"""
from __future__ import annotations

from typing import Optional
import logging
import time

import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


def _append_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Appends the three diff columns to the feature DataFrame."""
    df["prev_season_win_pct_diff"] = (
        df["home_prev_season_win_pct"] - df["away_prev_season_win_pct"]
    )
    df["prev_season_point_diff_avg_diff"] = (
        df["home_prev_season_point_diff_avg"] - df["away_prev_season_point_diff_avg"]
    )
    df["prev_season_srs_lite_diff"] = (
        df["home_prev_season_srs_lite"] - df["away_prev_season_srs_lite"]
    )
    return df


def transform(
    games: pd.DataFrame,
    *,
    # --- MODIFIED --- Argument now accepts the pre-aggregated DataFrame from our RPC
    season_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Attach previous-season team metrics to each upcoming game."""
    if debug:
        logger.setLevel(logging.DEBUG)
    start_ts = time.time()
    logger.info("season: start – input shape %s", games.shape)

    if games.empty:
        logger.warning("season: received empty games DataFrame – returning passthrough")
        return games.copy()

    required = {"game_id", "season", "home_team_norm", "away_team_norm"}
    if not required.issubset(games.columns):
        missing = required - set(games.columns)
        logger.error("season: missing required columns %s – skipping transform", missing)
        return games.copy()

    out = games.copy()
    default_vals = {
        "prev_season_win_pct": DEFAULTS["win_pct"],
        "prev_season_points_for_avg": DEFAULTS["points_for_avg"],
        "prev_season_points_against_avg": DEFAULTS["points_against_avg"],
        "prev_season_point_diff_avg": DEFAULTS["point_differential_avg"],
        "prev_season_srs_lite": DEFAULTS["srs_lite"],
    }

    if season_stats_df is None or season_stats_df.empty:
        logger.warning("season: no pre-fetched season_stats_df provided – applying league defaults")
        for side in ("home", "away"):
            for feat, dval in default_vals.items():
                col = f"{side}_{feat}"
                out[col] = dval
                if flag_imputations:
                    out[f"{col}_imputed"] = 1
        return _append_differentials(out)

    # --- MODIFIED --- Simplified logic to use pre-aggregated data directly
    # The _compute_team_metrics helper is no longer needed.
    
    # 1. Prepare lookup from the pre-aggregated RPC data
    lookup_df = season_stats_df.copy()
    if "team_norm" not in lookup_df.columns:
        lookup_df["team_norm"] = lookup_df["team_id"].apply(normalize_team_name)

    # 2. Rename columns to match the feature names expected downstream
    lookup_df = lookup_df.rename(columns={
        "wins_all_percentage": "prev_season_win_pct",
        "points_for_avg_all": "prev_season_points_for_avg",
        "points_against_avg_all": "prev_season_points_against_avg",
        "srs_lite": "prev_season_srs_lite",
    })
    
    # Derive point differential if it's not present
    if 'prev_season_point_diff_avg' not in lookup_df.columns:
        lookup_df['prev_season_point_diff_avg'] = (
            lookup_df['prev_season_points_for_avg'] - lookup_df['prev_season_points_against_avg']
        )
    
    # 3. Create the final lookup table indexed by (team, season)
    feature_cols = list(default_vals.keys())
    lookup = lookup_df.set_index(["team_norm", "season"])[feature_cols]

    # 4. Join home/away via a reindex on (team_norm, season-1)
    for side in ("home", "away"):
        team_series = out[f"{side}_team_norm"]
        prev_season = out["season"] - 1
        joined = lookup.reindex(pd.MultiIndex.from_arrays([team_series, prev_season]))
        joined = joined.add_prefix(f"{side}_").reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)

    # 5. Fill missing values + optional imputation flags
    for side in ("home", "away"):
        for feat, dval in default_vals.items():
            col = f"{side}_{feat}"
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna().astype(int)
            out[col] = out[col].fillna(dval)

    # 6. Append the final differential columns and return
    out = _append_differentials(out)
    logger.info("season: complete in %.2f s – output shape %s", time.time() - start_ts, out.shape)
    return out