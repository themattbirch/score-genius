# backend/nba_features/season.py  ── replace the existing transform() with this
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, determine_season

logger = logging.getLogger(__name__)
__all__ = ["transform"]


def _previous_season(season_str: str) -> str:
    """'2023‑24' → '2022‑23'."""
    first = int(season_str[:4]) - 1
    return f"{first}-{str(first + 1)[-2:]}"


def _choose_lookup_season(dates: pd.Series, seasons: pd.Series) -> pd.Series:
    """
    Before 1 Dec → previous season, otherwise current.
    (Adjust cutoff easily if desired.)
    """
    early = dates.dt.month < 12
    return np.where(early, seasons.map(_previous_season), seasons)


def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)

    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.tz_localize(
        None
    )
    out = out.dropna(subset=["game_date"])
    out["season"] = out["game_date"].apply(determine_season)

    # Pick season to look up (prev season for early‑season games)
    out["lookup_season"] = _choose_lookup_season(out["game_date"], out["season"])

    # Normalise team strings
    out["home_team_norm"] = out["home_team"].astype(str).map(normalize_team_name)
    out["away_team_norm"] = out["away_team"].astype(str).map(normalize_team_name)
    out["merge_key_home"] = out["home_team_norm"] + "_" + out["lookup_season"]
    out["merge_key_away"] = out["away_team_norm"] + "_" + out["lookup_season"]

    # ---------------------------------------------------------------- stats prep
    placeholder_cols = [
        "home_season_win_pct",
        "away_season_win_pct",
        "home_season_avg_pts_for",
        "away_season_avg_pts_for",
        "home_season_avg_pts_against",
        "away_season_avg_pts_against",
        "home_current_form",
        "away_current_form",
    ]
# ⬇ replace ONLY the `if team_stats_df is None or team_stats_df.empty:` block

    if team_stats_df is None or team_stats_df.empty:
        logger.info("No season‑stats dataframe supplied – falling back to DEFAULTS.")
        for col in placeholder_cols:
            if "win_pct" in col:
                out[col] = DEFAULTS.get("win_pct", 0.5)
            elif "avg_pts_for" in col:
                out[col] = DEFAULTS.get("avg_pts_for", 115.0)
            elif "avg_pts_against" in col:
                out[col] = DEFAULTS.get("avg_pts_against", 115.0)
            elif "current_form" in col:
                out[col] = "N/A"
            else:
                out[col] = 0.0

        # explicit diff / net‑rating zeros
        out["season_win_pct_diff"]      = 0.0
        out["season_pts_for_diff"]      = 0.0
        out["season_pts_against_diff"]  = 0.0
        out["home_season_net_rating"]   = 0.0
        out["away_season_net_rating"]   = 0.0
        out["season_net_rating_diff"]   = 0.0
        return out

    ts = team_stats_df.copy()
    ts["team_name_norm"] = ts["team_name"].astype(str).map(normalize_team_name)
    ts["merge_key"] = ts["team_name_norm"] + "_" + ts["season"].astype(str)
    ts = (
        ts.sort_values(["team_name_norm", "season"])
        .drop_duplicates(subset=["merge_key"], keep="last")
        .loc[
            :,
            [
                "merge_key",
                "wins_all_percentage",
                "points_for_avg_all",
                "points_against_avg_all",
                "current_form",
            ],
        ]
    )

    home_ren = {
        "wins_all_percentage": "home_season_win_pct",
        "points_for_avg_all": "home_season_avg_pts_for",
        "points_against_avg_all": "home_season_avg_pts_against",
        "current_form": "home_current_form",
    }
    away_ren = {k: v.replace("home_", "away_") for k, v in home_ren.items()}

    # ---------------------------------------------------------------- merging
    out = out.merge(
        ts.rename(columns=home_ren),
        how="left",
        left_on="merge_key_home",
        right_on="merge_key",
    ).drop(columns="merge_key")
    out = out.merge(
        ts.rename(columns=away_ren),
        how="left",
        left_on="merge_key_away",
        right_on="merge_key",
        suffixes=("", "_dup"),
    ).drop(columns="merge_key")

    # ---------------------------------------------------------------- fallbacks
    out["home_season_win_pct"] = out["home_season_win_pct"].fillna(
        DEFAULTS.get("win_pct", 0.5)
    )
    out["away_season_win_pct"] = out["away_season_win_pct"].fillna(
        DEFAULTS.get("win_pct", 0.5)
    )
    out["home_season_avg_pts_for"] = out["home_season_avg_pts_for"].fillna(
        DEFAULTS.get("avg_pts_for", 115.0)
    )
    out["away_season_avg_pts_for"] = out["away_season_avg_pts_for"].fillna(
        DEFAULTS.get("avg_pts_for", 115.0)
    )
    out["home_season_avg_pts_against"] = out["home_season_avg_pts_against"].fillna(
        DEFAULTS.get("avg_pts_against", 115.0)
    )
    out["away_season_avg_pts_against"] = out["away_season_avg_pts_against"].fillna(
        DEFAULTS.get("avg_pts_against", 115.0)
    )
    out["home_current_form"] = out["home_current_form"].fillna("N/A")
    out["away_current_form"] = out["away_current_form"].fillna("N/A")

    # ---------------------------------------------------------------- diffs
    out["season_win_pct_diff"] = (
        out["home_season_win_pct"] - out["away_season_win_pct"]
    )
    out["season_pts_for_diff"] = (
        out["home_season_avg_pts_for"] - out["away_season_avg_pts_for"]
    )
    out["season_pts_against_diff"] = (
        out["home_season_avg_pts_against"] - out["away_season_avg_pts_against"]
    )
    out["home_season_net_rating"] = (
        out["home_season_avg_pts_for"] - out["home_season_avg_pts_against"]
    )
    out["away_season_net_rating"] = (
        out["away_season_avg_pts_for"] - out["away_season_avg_pts_against"]
    )
    out["season_net_rating_diff"] = (
        out["home_season_net_rating"] - out["away_season_net_rating"]
    )

    # ---------------------------------------------------------------- cleanup
    out = out.drop(
        columns=[
            "season",
            "lookup_season",
            "home_team_norm",
            "away_team_norm",
            "merge_key_home",
            "merge_key_away",
        ],
        errors="ignore",
    )

    return out
