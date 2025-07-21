# backend/nfl_features/season.py
"""Previous-season context features for NFL matchups.

This transformer joins team-level aggregates from the **previous season** onto
an input `games` DataFrame. It looks up stats from the immediate prior season — 
2024 games look at 2023 stats, etc.
"""
from __future__ import annotations

from typing import Optional, Dict
import logging
import time

import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

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
    """Appends differential columns to the feature DataFrame."""
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


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------

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

    required_cols = {"game_id", "season", "home_team_norm", "away_team_norm"}
    if not required_cols.issubset(games.columns):
        missing = required_cols - set(games.columns)
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

    # If no lookup table, fill defaults and bail early
    if historical_team_stats_df is None or historical_team_stats_df.empty:
        logger.warning("season: no historical stats provided – applying league defaults")
        for side in ("home", "away"):
            for feat, dval in default_vals.items():
                col = f"{side}_{feat}"
                out[col] = dval
                if flag_imputations:
                    out[f"{col}_imputed"] = 1
        # ** Call the helper to ensure consistent output schema **
        return _append_differentials(out)

    # Build a lookup DataFrame keyed by canonical team name + season
    hts = historical_team_stats_df.copy()
    
    if "team_name" in hts.columns:
        hts["team_norm"] = hts["team_name"].apply(normalize_team_name)
    elif "team_id" in hts.columns:
        hts["team_norm"] = hts["team_id"].apply(normalize_team_name)
    else:
        raise ValueError("historical_team_stats_df must contain 'team_name' or 'team_id'")

    feature_rows = [_compute_team_metrics(row) | {"team_norm": row["team_norm"], "season": row["season"]} for _, row in hts.iterrows()]
    lookup = pd.DataFrame(feature_rows).set_index(["team_norm", "season"])

    # Join home and away sides (lookup uses season-1)
    for side in ("home", "away"):
        side_team = out[f"{side}_team_norm"]
        side_season = out["season"] - 1
        joined = lookup.reindex(pd.MultiIndex.from_arrays([side_team, side_season]))
        joined = joined.add_prefix(f"{side}_")
        out = pd.concat([out.reset_index(drop=True), joined.reset_index(drop=True)], axis=1)

    # Fill missing with defaults and flag imputations
    for side in ("home", "away"):
        for feat, dval in default_vals.items():
            col = f"{side}_{feat}"
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna().astype(int)
            out[col] = out[col].fillna(dval)

    # ** Call the helper before returning **
    out = _append_differentials(out)

    logger.info("season: complete in %.2f s – output shape %s", time.time() - start_ts, out.shape)
    return out