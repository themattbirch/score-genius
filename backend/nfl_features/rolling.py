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
from supabase import create_client, Client

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

__all__ = ["load_rolling_features"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rolling_features(game_ids: List[int], supabase: Client) -> pd.DataFrame:
    """Fetches recent‑form metrics for each team in the provided ``game_ids``.

    Parameters
    ----------
    game_ids : List[int]
        List of upcoming game identifiers.
    supabase : Client
        An already‑initialised Supabase client (service or anon key works –
        read‑only queries only).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame keyed by ``game_id`` with:
        ``home_<stat>``, ``away_<stat>``, and ``<stat>_diff`` columns for every
        rolling metric plus point differential and yards‑per‑play differential.
    """
    if not game_ids:
        return pd.DataFrame()

    # ---------------------------------------------------------------------
    # 1. Get the home/away team_ids for the requested games
    # ---------------------------------------------------------------------
    games_resp = (
        supabase.table("nfl_game_schedule")
        .select("game_id, home_team_id, away_team_id")
        .in_("game_id", game_ids)
        .execute()
    )

    games_data = games_resp.get("data", [])
    if not games_data:
        logger.warning("rolling: No games found for game_ids=%s", game_ids)
        return pd.DataFrame()

    games_df = pd.DataFrame(games_data)
    team_ids = pd.unique(games_df[["home_team_id", "away_team_id"]].values.ravel()).tolist()

    # ---------------------------------------------------------------------
    # 2. Pull rolling stats for all relevant teams from the materialised view
    # ---------------------------------------------------------------------
    stats_cols = ["team_id"] + list(_BASE_FEATURE_COLS.keys())
    stats_resp = (
        supabase.table(_ROLLING_VIEW)
        .select(",".join(stats_cols))
        .in_("team_id", team_ids)
        .execute()
    )

    stats_data = stats_resp.get("data", [])
    stats_df = pd.DataFrame(stats_data) if stats_data else pd.DataFrame(columns=stats_cols)

    # ---------------------------------------------------------------------
    # 3. Compute derived columns (e.g. rolling_point_differential_avg)
    # ---------------------------------------------------------------------
    for new_col, (num_col, den_col) in _DERIVED_COLS.items():
        if new_col not in stats_df.columns:
            stats_df[new_col] = (
                stats_df.get(num_col, 0.0) - stats_df.get(den_col, 0.0)
            )

    # Re‑prefix: already has "rolling_" so skip; we'll prefix home/away later.

    # ---------------------------------------------------------------------
    # 4. Melt into home/away features
    # ---------------------------------------------------------------------
    # Merge for home side
    home_join = games_df.merge(stats_df, how="left", left_on="home_team_id", right_on="team_id")
    home_pref = prefix_columns(home_join.drop(columns=["team_id"]), "home", exclude=["game_id", "home_team_id", "away_team_id"])

    # Merge for away side
    away_join = games_df.merge(stats_df, how="left", left_on="away_team_id", right_on="team_id")
    away_pref = prefix_columns(away_join.drop(columns=["team_id"]), "away", exclude=["game_id", "home_team_id", "away_team_id"])

    # Combine & deduplicate game meta cols
    combined = (
        home_pref
        .drop(columns=["home_team_id", "away_team_id"], errors="ignore")
        .merge(
            away_pref.drop(columns=["home_team_id", "away_team_id"], errors="ignore"),
            on="game_id",
        )
    )

    # ---------------------------------------------------------------------
    # 5. Fill missing with early‑season defaults
    # ---------------------------------------------------------------------
    for db_col, generic_key in _BASE_FEATURE_COLS.items():
        rolling_col = db_col  # includes "rolling_" prefix
        home_col = f"home_{rolling_col}"
        away_col = f"away_{rolling_col}"
        default_val = DEFAULTS.get(generic_key, 0.0)
        if home_col in combined.columns:
            combined[home_col] = combined[home_col].astype(float).fillna(default_val)
        if away_col in combined.columns:
            combined[away_col] = combined[away_col].astype(float).fillna(default_val)

    # Derived default for differential columns (should be 0 by construction)
    combined.fillna(0.0, inplace=True)

    # ---------------------------------------------------------------------
    # 6. Compute differentials (home minus away)
    # ---------------------------------------------------------------------
    for db_col in list(_BASE_FEATURE_COLS.keys()) + list(_DERIVED_COLS.keys()):
        h_col, a_col = f"home_{db_col}", f"away_{db_col}"
        if h_col in combined.columns and a_col in combined.columns:
            combined[f"{db_col}_diff"] = combined[h_col] - combined[a_col]

    # ---------------------------------------------------------------------
    # 7. Return only game_id + rolling columns (home, away, diff)
    # ---------------------------------------------------------------------
    keep_cols = [c for c in combined.columns if c == "game_id" or "rolling_" in c]
    return combined[keep_cols].copy()
