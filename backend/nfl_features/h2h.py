# backend/nfl_features/h2h.py
"""
Head‑to‑head (H2H) matchup features for NFL games.

The transformer looks at *prior* meetings between the same two franchises and
computes aggregate statistics, shifted so that each game only “sees” results
from earlier dates.  A matchup key is built by alphabetically sorting the two
team identifiers, so DAL‑PHI and PHI‑DAL reference the same series.

Output columns
--------------
    h2h_games_played
    h2h_home_win_pct           (win% from the scheduled home team’s perspective)
    h2h_avg_point_diff         (home – away, averaged over prior games)
    h2h_avg_total_points
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _build_matchup_key(df: pd.DataFrame) -> pd.Series:
    """Return a Series where each element is 'teamA_vs_teamB' (alpha‑sorted)."""
    arr = (
        df[["home_team_norm", "away_team_norm"]]
        .apply(lambda row: sorted(row.values.tolist()), axis=1)
        .tolist()
    )
    return pd.Series([f"{a}_vs_{b}" for a, b in arr], index=df.index)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    max_games: int = 10,
) -> pd.DataFrame:
    """
    Attach leakage‑free H2H features to `games`.

    Parameters
    ----------
    games : DataFrame
        Upcoming games (must include game_id, game_date, home_team_norm, away_team_norm).
    historical_df : DataFrame | None
        Past NFL results with the same columns plus scores.
    max_games : int, default 10
        Rolling window of prior meetings to average over.

    Returns
    -------
    DataFrame with H2H columns added (defaults when no history), or empty if no upcoming games.
    """
    # 0. Fast‑exit if no upcoming games
    if games.empty:
        return pd.DataFrame()

    # 1. If no history, just merge defaults
    if historical_df is None or historical_df.empty:
        logger.warning("h2h: no historical data – defaulting all H2H features.")
        base = games[["game_id"]].copy()
        base["h2h_games_played"]    = 0
        base["h2h_home_win_pct"]    = DEFAULTS["matchup_home_win_pct"]
        base["h2h_avg_point_diff"]  = DEFAULTS["matchup_avg_point_diff"]
        base["h2h_avg_total_points"]= DEFAULTS["matchup_avg_total_points"]
        return games.merge(base, on="game_id")

    # 2. Build a single timeline with both history and upcoming
    hist = historical_df.copy()
    hist["game_date"]       = pd.to_datetime(hist["game_date"])
    hist["home_team_norm"]  = hist["home_team_norm"].apply(normalize_team_name)
    hist["away_team_norm"]  = hist["away_team_norm"].apply(normalize_team_name)
    hist["_is_upcoming"]    = False

    upcoming = games.copy()
    upcoming["game_date"]       = pd.to_datetime(upcoming["game_date"])
    upcoming["home_team_norm"]  = upcoming["home_team_norm"].apply(normalize_team_name)
    upcoming["away_team_norm"]  = upcoming["away_team_norm"].apply(normalize_team_name)
    upcoming["_is_upcoming"]    = True

    combined = pd.concat([hist, upcoming], ignore_index=True)
    combined["_is_upcoming"] = combined["_is_upcoming"].astype(bool)  # no NaNs in mask
    combined["matchup_key"]  = _build_matchup_key(combined)
    combined.sort_values(["matchup_key", "game_date"], inplace=True)

    # 3. Compute per‑game stats
    combined["home_win"]      = (combined["home_score"] > combined["away_score"]).astype(float)
    combined["point_diff"]    = combined["home_score"] - combined["away_score"]
    combined["total_points"]  = combined["home_score"] + combined["away_score"]

    # 4. Rolling aggregates without leakage
    grp = combined.groupby("matchup_key")
    combined["h2h_games_played"] = grp.cumcount()

    def _roll(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(max_games, min_periods=1).mean()

    combined["h2h_home_win_pct"]    = grp["home_win"].apply(_roll)
    combined["h2h_avg_point_diff"]  = grp["point_diff"].apply(_roll)
    combined["h2h_avg_total_points"]= grp["total_points"].apply(_roll)

    # 5. Extract only the upcoming rows and merge back
    h2h_feats = combined.loc[
        combined["_is_upcoming"],
        ["game_id", "h2h_games_played", "h2h_home_win_pct", "h2h_avg_point_diff", "h2h_avg_total_points"],
    ]
    out = games.merge(h2h_feats, on="game_id", how="left")

    # 6. Fill defaults for brand‑new matchups
    fills = {
        "h2h_games_played": 0,
        "h2h_home_win_pct": DEFAULTS["matchup_home_win_pct"],
        "h2h_avg_point_diff": DEFAULTS["matchup_avg_point_diff"],
        "h2h_avg_total_points": DEFAULTS["matchup_avg_total_points"],
    }
    for col, default in fills.items():
        out[col] = out[col].fillna(default)

    return out
