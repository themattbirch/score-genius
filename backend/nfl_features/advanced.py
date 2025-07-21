# backend/nfl_features/advanced.py

"""Win‑loss *form* and current streak features for NFL teams.

The goal is to quantify recent momentum without leaking future information.
We work from a **historical* games DataFrame (all past results) and attach
rolling win‑percentage and streak metrics to an *upcoming* games DataFrame.

Key steps:
1. Convert the historical schedule to **long format** so each row represents a
   single team in a single game.
2. Compute outcome (win=1, loss=‑1, tie=0) and build a streak counter.
3. For each team, roll over the previous ``lookback_window`` games *excluding*
   the current match (via ``shift(1)``) to produce leakage‑free win%.
4. Merge the resulting features onto upcoming games for both home and away
   sides, then derive differentials.

Output columns (for ``lookback_window=5``):
- ``home_form_win_pct_5`` / ``away_…``
- ``home_current_streak`` / ``away_…`` (positive = winning streak, negative = losing)
- ``form_win_pct_5_diff``
- ``current_streak_diff``

Missing values (early season or expansion teams) are filled with ``utils.DEFAULTS``
and flagged if desired.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, prefix_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


def _prepare_long_format(historical: pd.DataFrame) -> pd.DataFrame:
    """Return a long‑form DataFrame with one row per **team‑game**."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    # Ensure canonical team keys for join consistency
    hist["home_team_norm"] = hist["home_team_norm"].apply(normalize_team_name)
    hist["away_team_norm"] = hist["away_team_norm"].apply(normalize_team_name)

    home_games = hist.rename(
        columns={
            "home_team_norm": "team",
            "away_team_norm": "opponent",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    )
    away_games = hist.rename(
        columns={
            "away_team_norm": "team",
            "home_team_norm": "opponent",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    )
    long_df = pd.concat([home_games, away_games], ignore_index=True)
    return long_df.sort_values(["team", "game_date"])


def _compute_outcomes(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome (+1 win, ‑1 loss, 0 tie) and streak columns."""
    long_df = long_df.copy()
    long_df["outcome"] = np.select(
        [long_df["team_score"] > long_df["opp_score"], long_df["team_score"] < long_df["opp_score"]],
        [1, -1],
        default=0,
    )

    # Continuous streak counter within same outcome blocks
    long_df["_streak_len"] = (
        long_df.groupby("team")["outcome"].apply(
            lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
        )
    )
    long_df["current_streak"] = long_df["_streak_len"] * long_df["outcome"]
    long_df.drop(columns="_streak_len", inplace=True)
    return long_df


def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    lookback_window: int = 5,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Attach leakage‑free *form* features to ``games``.

    Parameters
    ----------
    games : pd.DataFrame
        Upcoming games (must have ``game_id``, ``game_date``, ``home_team_norm``, ``away_team_norm``).
    historical_df : pd.DataFrame | None
        Past games with the same columns + scores. If ``None`` or empty, defaults are used.
    lookback_window : int, default 5
        Rolling window length for win‑percentage.
    flag_imputations : bool, default True
        If True, add ``*_imputed`` columns when defaults are inserted.
    """
    if historical_df is None or historical_df.empty:
        logger.warning("form: No historical data – defaulting all form features.")
        base = games[["game_id"]].copy()
        for side in ("home", "away"):
            base[f"{side}_form_win_pct_{lookback_window}"] = DEFAULTS["form_win_pct"]
            base[f"{side}_current_streak"] = DEFAULTS["current_streak"]
            if flag_imputations:
                base[f"{side}_form_imputed"] = 1
                base[f"{side}_streak_imputed"] = 1
        base[f"form_win_pct_{lookback_window}_diff"] = 0.0
        base["current_streak_diff"] = 0
        return base

    long_df = _compute_outcomes(_prepare_long_format(historical_df))

    # Rolling win% (exclude current game)
    long_df[f"form_win_pct_{lookback_window}"] = (
        long_df.groupby("team")["outcome"].shift(1)
        .rolling(window=lookback_window, min_periods=1)
        .apply(lambda x: (x > 0).mean(), raw=True)
    )

    feat_cols = [
        "team",
        "game_date",
        f"form_win_pct_{lookback_window}",
        "current_streak",
    ]
    features_long = long_df[feat_cols]

    # Ensure upcoming games DataFrame is sorted for merge_asof
    upcoming = games.copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"])
    upcoming.sort_values("game_date", inplace=True)

    # Merge for each side
    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        tmp = pd.merge_asof(
            upcoming,
            features_long.rename(columns={"team": team_col}),
            on="game_date",
            by=team_col,
            direction="backward",
        )
        merged[side] = prefix_columns(
            tmp[["game_id", f"form_win_pct_{lookback_window}", "current_streak"]],
            f"{side}",
            exclude=["game_id"],
        )

    result = merged["home"].merge(merged["away"], on="game_id")

    # Fill defaults + flag
    fills = {
        f"home_form_win_pct_{lookback_window}": DEFAULTS["form_win_pct"],
        f"away_form_win_pct_{lookback_window}": DEFAULTS["form_win_pct"],
        "home_current_streak": DEFAULTS["current_streak"],
        "away_current_streak": DEFAULTS["current_streak"],
    }
    for col, dval in fills.items():
        if flag_imputations:
            result[f"{col}_imputed"] = result[col].isna().astype(int)
        result[col] = result[col].fillna(dval)

    # Differentials
    result[f"form_win_pct_{lookback_window}_diff"] = (
        result[f"home_form_win_pct_{lookback_window}"]
        - result[f"away_form_win_pct_{lookback_window}"]
    )
    result["current_streak_diff"] = (
        result["home_current_streak"] - result["away_current_streak"]
    )

    return result
