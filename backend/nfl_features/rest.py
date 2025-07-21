# backend/nfl_features/rest.py

"""
Rest‑day & schedule‑spacing features for NFL games.

For each upcoming matchup we calculate:
  • rest_days             – full days between a team’s previous game and kickoff
  • is_on_short_week      – rest_days ≤ 3   (Thu after Sun / Sat after Mon, etc.)
  • is_off_bye            – rest_days ≥ 13  (the week after a bye)
  • rest_advantage        – home_rest_days – away_rest_days
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .utils import DEFAULTS, normalize_team_name, prefix_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _timeline(historical: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Return long DataFrame: one row = team‑game, historical+upcoming."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    upc = upcoming.copy()
    upc["game_date"] = pd.to_datetime(upc["game_date"])
    upc["_is_upcoming"] = True  # marker for extraction later

    # Canonical team keys for grouping
    for col in ("home_team_norm", "away_team_norm"):
        hist[col] = hist[col].apply(normalize_team_name)
        upc[col] = upc[col].apply(normalize_team_name)

    home = pd.concat(
        [
            hist.rename(columns={"home_team_norm": "team"}),
            upc.rename(columns={"home_team_norm": "team"}),
        ],
        ignore_index=True,
    )
    away = pd.concat(
        [
            hist.rename(columns={"away_team_norm": "team"}),
            upc.rename(columns={"away_team_norm": "team"}),
        ],
        ignore_index=True,
    )
    long_df = pd.concat([home, away], ignore_index=True, sort=False)
    return long_df.sort_values(["team", "game_date"])


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """
    Attach rest‑day & schedule features to ``games``.

    Parameters
    ----------
    games : DataFrame
        Upcoming schedule (game_id / game_date / *_team_norm).
    historical_df : DataFrame | None
        Past results for rest‑day lookup.
    flag_imputations : bool, default True
        Adds *_imputed flags when defaults are injected.

    Returns
    -------
    DataFrame with home_/away_ rest columns and rest_advantage diff.
    """
    if historical_df is None or historical_df.empty:
        logger.warning("rest: no historical data – defaulting to 7‑day rest.")
        base = games[["game_id"]].copy()
        for side in ("home", "away"):
            base[f"{side}_rest_days"] = DEFAULTS["days_since_last_game"]
            base[f"{side}_is_on_short_week"] = 0
            base[f"{side}_is_off_bye"] = 0
            if flag_imputations:
                base[f"{side}_rest_days_imputed"] = 1
        base["rest_advantage"] = 0.0
        return games.merge(base, on="game_id")

    long_df = _timeline(historical_df, games)

    grp = long_df.groupby("team")
    long_df["prev_game_date"] = grp["game_date"].shift(1)

    # Full rest days between prior kickoff and current kickoff
    long_df["rest_days"] = (long_df["game_date"] - long_df["prev_game_date"]).dt.days - 1
    # First game for a team: rest_days will be NaN

    # Flags
    long_df["is_on_short_week"] = (long_df["rest_days"] <= 3).astype(int)
    long_df["is_off_bye"] = (long_df["rest_days"] >= 13).astype(int)

    # Upcoming slice
    feats = long_df[long_df["_is_upcoming"]].loc[
        :,
        ["game_id", "team", "rest_days", "is_on_short_week", "is_off_bye"],
    ]

    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        df_side = games.merge(
            feats.rename(columns={"team": team_col}),
            on=["game_id", team_col],
            how="left",
        )
        merged[side] = prefix_columns(
            df_side[["game_id", "rest_days", "is_on_short_week", "is_off_bye"]],
            side,
            exclude=["game_id"],
        )

    out = merged["home"].merge(merged["away"], on="game_id")

    # Defaults & flags
    default_rd = DEFAULTS["days_since_last_game"]
    for side in ("home", "away"):
        rd_col = f"{side}_rest_days"
        if flag_imputations:
            out[f"{rd_col}_imputed"] = out[rd_col].isna().astype(int)
        out[rd_col] = out[rd_col].fillna(default_rd)
        # bool flags default to 0
        for flag_col in (f"{side}_is_on_short_week", f"{side}_is_off_bye"):
            out[flag_col] = out[flag_col].fillna(0).astype(int)

    # Differential
    out["rest_advantage"] = out["home_rest_days"] - out["away_rest_days"]

    return out
