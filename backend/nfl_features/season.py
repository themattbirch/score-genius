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
from typing import Dict

import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]

def _compute_team_metrics(row: pd.Series) -> Dict[str, float]:
    """Derive per-game rates from a raw team-season row."""
    games_played = row["won"] + row["lost"] + row.get("ties", 0)
    if games_played == 0:
        games_played = 17  # Fallback for malformed rows

    # NFL win pct counts a tie as half-win, half-loss
    wins_adj = row["won"] + 0.5 * row.get("ties", 0)
    win_pct = wins_adj / games_played
    pf_avg = row["points_for"] / games_played
    pa_avg = row["points_against"] / games_played
    pdiff_avg = row["points_difference"] / games_played
    
    return {
        "prev_season_win_pct": win_pct,
        "prev_season_points_for_avg": pf_avg,
        "prev_season_points_against_avg": pa_avg,
        "prev_season_point_diff_avg": pdiff_avg,
        "prev_season_srs_lite": row.get("srs_lite", 0.0),
    }

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
    historical_team_stats_df: Optional[pd.DataFrame] = None,
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

    if historical_team_stats_df is None or historical_team_stats_df.empty:
        logger.warning("season: no historical stats provided – applying league defaults")
        for side in ("home", "away"):
            for feat, dval in default_vals.items():
                col = f"{side}_{feat}"
                out[col] = dval
                if flag_imputations:
                    out[f"{col}_imputed"] = 1
        return _append_differentials(out)

    hts = historical_team_stats_df.copy()
    if "team_norm" not in hts.columns:
        hts["team_norm"] = hts["team_id"].apply(normalize_team_name)

    # --- RESTORED: This block correctly calculates metrics from raw data ---
    feature_rows = []
    for _, row in hts.iterrows():
        metrics = _compute_team_metrics(row)
        metrics.update(team_norm=row["team_norm"], season=row["season"])
        feature_rows.append(metrics)
    lookup = pd.DataFrame(feature_rows).set_index(["team_norm", "season"])
    # --- END RESTORED BLOCK ---

    # The rest of the function correctly joins and processes this lookup table
    for side in ("home", "away"):
        team_series = out[f"{side}_team_norm"]
        prev_season = out["season"] - 1
        joined = lookup.reindex(pd.MultiIndex.from_arrays([team_series, prev_season]))
        joined = joined.add_prefix(f"{side}_").reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)

    for side in ("home", "away"):
        for feat, dval in default_vals.items():
            col = f"{side}_{feat}"
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna().astype(int) # Using .astype(int) to match test
            out[col] = out[col].fillna(dval)

    out = _append_differentials(out)
    logger.info("season: complete in %.2f s – output shape %s", time.time() - start_ts, out.shape)
    return out
