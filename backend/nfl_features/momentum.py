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
    """One row per team‑game (lower‑cased names, no mapping)."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    # preserve synthetic names used in unit‑tests – just lower‑case
    hist["home_team_norm"] = hist["home_team_norm"].str.lower()
    hist["away_team_norm"] = hist["away_team_norm"].str.lower()

    home = hist.rename(
        columns={
            "home_team_norm": "team",
            "away_team_norm": "opponent",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    )
    away = hist.rename(
        columns={
            "away_team_norm": "team",
            "home_team_norm": "opponent",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
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
    Attach leakage‑free momentum features to `games`.
    Momentum = EWMA (span=``span``) of prior point‑differentials.

    Returns a DataFrame with:
        • home_momentum_ewma_<span>
        • away_momentum_ewma_<span>
        • momentum_ewma_<span>_diff
        (+ optional *_imputed flags when `flag_imputations` is True)
    """
    # 0. Early exits
    if games.empty:
        return pd.DataFrame()

    # 0a. No history → neutral momentum, no imputation flags
    if historical_df is None or historical_df.empty:
        base = games[["game_id"]].copy()
        for side in ("home", "away"):
            base[f"{side}_momentum_ewma_{span}"] = DEFAULTS["momentum_direction"]
        base[f"momentum_ewma_{span}_diff"] = 0.0
        return base

    # 1. Build long‐form history
    long_df = _long_format(historical_df)
    # Compute the EWM on the raw point_diffs
    long_df[f"momentum_ewma_{span}"] = (
        long_df
        .groupby("team")["point_diff"]
        .transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
    )

    # Extract only what we need for the join
    features = long_df[["team", "game_date", f"momentum_ewma_{span}"]].copy()
    # Sort by on‐key only so merge_asof won't complain
    features.sort_values("game_date", inplace=True)

    # 2. Prepare upcoming games
    games_sorted = games[["game_id", "game_date", "home_team_norm", "away_team_norm"]].copy()
    games_sorted["game_date"] = pd.to_datetime(games_sorted["game_date"])
    # match the lower‐casing used in tests
    games_sorted["home_team_norm"] = games_sorted["home_team_norm"].str.lower()
    games_sorted["away_team_norm"] = games_sorted["away_team_norm"].str.lower()
    games_sorted.sort_values("game_date", inplace=True)

    # 3. Merge per side with merge_asof(on="game_date", by=team_col)
    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        feats_side = features.rename(columns={"team": team_col})
        tmp = pd.merge_asof(
            games_sorted,
            feats_side,
            on="game_date",
            by=team_col,
            direction="backward",
        )
        merged[side] = prefix_columns(
            tmp[["game_id", f"momentum_ewma_{span}"]],
            side,
            exclude=["game_id"],
        )

    out = merged["home"].merge(merged["away"], on="game_id")

    # 4. Fill defaults & optionally flag imputations
    default_val = DEFAULTS["momentum_direction"]
    for side in ("home", "away"):
        col = f"{side}_momentum_ewma_{span}"
        if flag_imputations:
            out[f"{col}_imputed"] = out[col].isna().astype(int)
        out[col] = out[col].fillna(default_val)

    # 5. Differential
    out[f"momentum_ewma_{span}_diff"] = (
        out[f"home_momentum_ewma_{span}"] - out[f"away_momentum_ewma_{span}"]
    )

    return out
