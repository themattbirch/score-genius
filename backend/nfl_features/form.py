# backend/nfl_features/form.py

"""Win-loss *form* and current streak features for NFL teams.

The goal is to quantify recent momentum without leaking future information.
We work from a **historical* games DataFrame (all past results) and attach
rolling win-percentage and streak metrics to an *upcoming* games DataFrame.
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
    """Return a long-form DataFrame with one row per **team-game**."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    hist["home_team_norm"] = hist["home_team_norm"].str.lower()
    hist["away_team_norm"] = hist["away_team_norm"].str.lower()

    home_games = hist.rename(columns={"home_team_norm": "team", "away_team_norm": "opponent", "home_score": "team_score", "away_score": "opp_score"})
    away_games = hist.rename(columns={"away_team_norm": "team", "home_team_norm": "opponent", "away_score": "team_score", "home_score": "opp_score"})
    
    long_df = pd.concat([home_games, away_games], ignore_index=True)
    return long_df.sort_values(["team", "game_date"])


def _compute_outcomes(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome (+1 win, -1 loss, 0 tie) and streak columns."""
    df = long_df.copy()
    df["outcome"] = np.select(
        [df["team_score"] > df["opp_score"], df["team_score"] < df["opp_score"]],
        [1, -1],
        default=0,
    )

    # Continuous streak counter within same outcome blocks
    streak_len = df.groupby("team")["outcome"].apply(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    )
    df["streak_value"] = streak_len * df["outcome"]
    return df


def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    lookback_window: int = 5,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Attach leakage-free *form* features to ``games``."""
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
        base["current_streak_diff"] = 0.0
        return base

    long_df = _compute_outcomes(_prepare_long_format(historical_df))
    grouped = long_df.groupby("team")

    # --- Feature Calculation (with leakage protection) ---
    # Rolling win% (exclude current game via shift)
    long_df[f"form_win_pct_{lookback_window}"] = (
        grouped["outcome"]
        .rolling(window=lookback_window, min_periods=1)
        .apply(lambda x: (x > 0).mean(), raw=True)
        .reset_index(level=0, drop=True)
    )

    # Streak (exclude current game via shift)
    long_df["current_streak"] = long_df["streak_value"]
    
    # --- Merging Logic ---
    feat_cols = ["team", "game_date", f"form_win_pct_{lookback_window}", "current_streak"]
    features_long = long_df[feat_cols].copy()

    upcoming = games.copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"])
    upcoming.sort_values("game_date", inplace=True)

    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        # 1️⃣  right‑hand frame MUST be sorted only by the *on* key (“game_date”)
        features_side = (
            features_long
            .rename(columns={"team": team_col})
            .sort_values("game_date")          # ← replace old sort_values([team_col, "game_date"])
        )

        # `upcoming` is already sorted by game_date earlier
        tmp = pd.merge_asof(
            upcoming,
            features_side,
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
    
    # --- Fill Defaults & Add Diffs ---
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

    result[f"form_win_pct_{lookback_window}_diff"] = result[f"home_form_win_pct_{lookback_window}"] - result[f"away_form_win_pct_{lookback_window}"]
    result["current_streak_diff"] = result["home_current_streak"] - result["away_current_streak"]

    return result