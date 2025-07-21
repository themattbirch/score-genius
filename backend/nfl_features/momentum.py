# backend/nfl_features/momentum.py
"""
Inter-game *momentum* features for NFL teams.

We measure momentum as an exponentially-weighted moving average (EWMA) of
point differential, shifted so the value for each game reflects only
information known *before* kickoff.

Output columns:
    home_momentum_ewma_<span>
    away_momentum_ewma_<span>
    momentum_ewma_<span>_diff
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import (
    normalize_team_name,
    prefix_columns,
    DEFAULTS,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _long_format(historical: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame: one row per **team–game**."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    # Canonical team keys for robust grouping
    hist["home_team_norm"] = hist["home_team_norm"].apply(normalize_team_name)
    hist["away_team_norm"] = hist["away_team_norm"].apply(normalize_team_name)

    home_rows = hist.rename(
        columns={
            "home_team_norm": "team",
            "away_team_norm": "opponent",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    )
    away_rows = hist.rename(
        columns={
            "away_team_norm": "team",
            "home_team_norm": "opponent",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    )
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df["point_diff"] = long_df["team_score"] - long_df["opp_score"]
    return long_df.sort_values(["team", "game_date"])


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    span: int = 5,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """
    Attach leakage-free momentum features to ``games``.

    Parameters
    ----------
    games : pd.DataFrame
        Upcoming games (must include ``game_id``, ``game_date``,
        ``home_team_norm``, ``away_team_norm``).
    historical_df : pd.DataFrame | None
        Past game results. Required columns: *_team_norm*, *_score*, *game_date*.
    span : int, default 5
        Span for EWMA. Lower → more weight on recent games.
    flag_imputations : bool, default True
        Adds ``_imputed`` flags when momentum is filled with default (0.0).

    Returns
    -------
    pd.DataFrame
        ``games`` with momentum columns appended.
    """
    if historical_df is None or historical_df.empty:
        logger.warning("momentum: no historical data – defaulting to neutral momentum.")
        base = games[["game_id"]].copy()
        for side in ("home", "away"):
            col = f"{side}_momentum_ewma_{span}"
            base[col] = DEFAULTS.get("momentum_direction", 0.0)
            if flag_imputations:
                base[f"{col}_imputed"] = 1
        base[f"momentum_ewma_{span}_diff"] = 0.0
        return base

    long_df = _long_format(historical_df)

    # EWMA of point differential, shifted to avoid leakage
    grouped = long_df.groupby("team")
    long_df[f"momentum_ewma_{span}"] = (
        grouped["point_diff"]
        .shift(1)  # exclude current game
        .ewm(span=span, adjust=False, min_periods=1)
        .mean()
    )

    features = long_df[["team", "game_date", f"momentum_ewma_{span}"]]

    games_sorted = games.copy()
    games_sorted["game_date"] = pd.to_datetime(games_sorted["game_date"])
    games_sorted.sort_values("game_date", inplace=True)

    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        df_side = pd.merge_asof(
            games_sorted,
            features.rename(columns={"team": team_col}),
            on="game_date",
            by=team_col,
            direction="backward",
        )
        merged[side] = prefix_columns(
            df_side[["game_id", f"momentum_ewma_{span}"]],
            prefix=side,
            exclude=["game_id"],
        )

    out = pd.merge(merged["home"], merged["away"], on="game_id")

    # Defaults & imputation flags
    default_val = DEFAULTS.get("momentum_direction", 0.0)
    for side in ("home", "away"):
        col = f"{side}_momentum_ewma_{span}"
        if flag_imputations:
            out[f"{col}_imputed"] = out[col].isna().astype(int)
        out[col] = out[col].fillna(default_val)

    # Differential
    diff_col = f"momentum_ewma_{span}_diff"
    out[diff_col] = out[f"home_momentum_ewma_{span}"] - out[f"away_momentum_ewma_{span}"]

    return out
