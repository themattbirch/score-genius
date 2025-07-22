# backend/nfl_features/rolling.py

"""Rolling‑window (recent form) feature loader.

This module *does not compute* rolling averages – the heavy lifting is handled
by the `mv_nfl_recent_form` materialised view inside Supabase.  We simply fetch
the latest rows for all teams involved in a list of upcoming games, reshape
those rows into home/away columns, apply sensible defaults for early‑season
edge cases, and derive home‑minus‑away differentials.
"""
from __future__ import annotations

import logging
from typing import List, Mapping

import pandas as pd
from supabase import Client

from .utils import DEFAULTS, prefix_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Configuration constants (column naming contract with the DB view)
# ---------------------------------------------------------------------------
_ROLLING_VIEW: str = "mv_nfl_recent_form"
_BASE_FEATURE_COLS: Mapping[str, str] = {
    # DB column → logical key used for defaults & naming
    "rolling_points_for_avg": "points_for_avg",
    "rolling_points_against_avg": "points_against_avg",
    "rolling_yards_per_play_avg": "yards_per_play_avg",
    "rolling_turnover_differential_avg": "turnover_differential_avg",
}
# Derived columns computed locally after fetch
_DERIVED_COLS = {
    "rolling_point_differential_avg": (
        "rolling_points_for_avg",
        "rolling_points_against_avg",
    )
}


def load_rolling_features(game_ids: List[int], supabase: Client) -> pd.DataFrame:
    """Fetches recent‑form metrics for each team in the provided ``game_ids``."""
    if not game_ids:
        return pd.DataFrame()

    # 1. Pull schedule (home/away team IDs) from Supabase
    schedule_resp = (
        supabase.table("nfl_game_schedule")
        .select("game_id, home_team_id, away_team_id")
        .in_("game_id", game_ids)
        .execute()
    )
    # support both dict-like and attribute .data responses
    if hasattr(schedule_resp, "data"):
        sched_rows = schedule_resp.data or []
    else:
        sched_rows = schedule_resp.get("data", [])  # type: ignore

    if not sched_rows:
        logger.warning("rolling: No games found for game_ids=%s", game_ids)
        return pd.DataFrame()

    games_df = pd.DataFrame(sched_rows)
    if not {"home_team_id", "away_team_id"}.issubset(games_df.columns):
        logger.warning("rolling: schedule records missing home/away IDs")
        return pd.DataFrame()

    team_ids = pd.unique(
        games_df[["home_team_id", "away_team_id"]]
        .values
        .ravel()
    ).tolist()

    # 2. Pull rolling stats for those teams
    stats_cols = ["team_id"] + list(_BASE_FEATURE_COLS.keys())
    stats_resp = (
        supabase.table(_ROLLING_VIEW)
        .select(",".join(stats_cols))
        .in_("team_id", team_ids)
        .execute()
    )
    if hasattr(stats_resp, "data"):
        rolling_rows = stats_resp.data or []
    else:
        rolling_rows = stats_resp.get("data", [])  # type: ignore

    stats_df = (
        pd.DataFrame(rolling_rows)
        if rolling_rows
        else pd.DataFrame(columns=stats_cols)
    )

    # 3. Compute any derived columns locally
    for new_col, (num_col, den_col) in _DERIVED_COLS.items():
        stats_df[new_col] = (
            stats_df.get(num_col, 0.0) - stats_df.get(den_col, 0.0)
        )

    # 4. Merge in home/away and prefix
    home_join = games_df.merge(
        stats_df,
        how="left",
        left_on="home_team_id",
        right_on="team_id",
    )
    home_pref = prefix_columns(
        home_join.drop(columns=["team_id"]),
        "home",
        exclude=["game_id", "home_team_id", "away_team_id"],
    )

    away_join = games_df.merge(
        stats_df,
        how="left",
        left_on="away_team_id",
        right_on="team_id",
    )
    away_pref = prefix_columns(
        away_join.drop(columns=["team_id"]),
        "away",
        exclude=["game_id", "home_team_id", "away_team_id"],
    )

    combined = (
        home_pref
        .drop(columns=["home_team_id", "away_team_id"], errors="ignore")
        .merge(
            away_pref.drop(columns=["home_team_id", "away_team_id"], errors="ignore"),
            on="game_id",
        )
    )

    # 5. Fill missing with defaults
    for db_col, generic_key in _BASE_FEATURE_COLS.items():
        home_col = f"home_{db_col}"
        away_col = f"away_{db_col}"
        default_val = DEFAULTS.get(generic_key, 0.0)
        if home_col in combined:
            combined[home_col] = combined[home_col].astype(float).fillna(default_val)
        if away_col in combined:
            combined[away_col] = combined[away_col].astype(float).fillna(default_val)
    # any remaining NaNs → 0.0
    combined.fillna(0.0, inplace=True)

    # 6. Compute differentials
    for db_col in list(_BASE_FEATURE_COLS.keys()) + list(_DERIVED_COLS.keys()):
        h, a = f"home_{db_col}", f"away_{db_col}"
        if h in combined and a in combined:
            combined[f"{db_col}_diff"] = combined[h] - combined[a]

    # 7. Return only game_id + rolling_*-prefixed columns
    keep = ["game_id"] + [c for c in combined if c.startswith("home_rolling_") or c.startswith("away_rolling_") or c.endswith("_diff")]
    return combined[keep].copy()
