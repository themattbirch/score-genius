# backend/mlb_features/rest.py
"""
Calculates rest-based features for MLB games.

Focus
-----
1.  Strict “no-look-ahead” guarantee when multiple games share a date.
2.  Correct rest-day math, B2B flags, 7-/14-day density, and advantages.
3.  Idempotency + NaN-free outputs.

MLB Specifics:
-   Uses 'game_date_et' for game dates.
-   Uses 'home_team_id' and 'away_team_id' for team identification.
-   Rest and schedule density metrics are generally applicable, though MLB game
    frequency is high. B2B (rest_days == 1) is a common scenario.
"""

from __future__ import annotations
import logging
from typing import List

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("Could not import DEFAULTS from .utils. Using local fallbacks.")
    MLB_DEFAULTS: dict = {}

DEF_REST_MLB = float(MLB_DEFAULTS.get("mlb_rest_days", 3.0))

PLACEHOLDER_COLS: List[str] = [
    "rest_days_home", "rest_days_away",
    "games_last_7_days_home", "games_last_14_days_home",
    "games_last_7_days_away", "games_last_14_days_away",
    "is_back_to_back_home", "is_back_to_back_away",
    "rest_advantage", "schedule_advantage",
]


def _fill_defaults_mlb(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure placeholder columns exist and contain no NaNs."""
    for col in PLACEHOLDER_COLS:
        if col.startswith("is_back_to_back"):
            default = 0
            dtype = int
        elif col.startswith("games_last_"):
            default = 0
            dtype = int
        elif col.startswith("rest_days"):
            default = DEF_REST_MLB
            dtype = float
        else:  # advantages
            default = 0
            dtype = float

        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(dtype)
    return df


def transform(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Calculates rest and schedule-based features for MLB games.

    Args:
        df: DataFrame with columns:
            - 'game_id'
            - 'game_date_et'
            - 'home_team_id', 'away_team_id'
        debug: enable DEBUG logging for this run.

    Returns:
        df with all PLACEHOLDER_COLS added.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Starting mlb_features.rest.transform in DEBUG mode")

    if df is None or df.empty:
        logger.warning("Empty input to rest.transform; returning placeholders only.")
        return _fill_defaults_mlb(pd.DataFrame(columns=PLACEHOLDER_COLS))

    # Clean up any prior runs
    drop_cols = PLACEHOLDER_COLS + ["prev_home_game_date", "prev_away_game_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.reset_index(drop=True).copy()

    # Required MLB columns → internal names
    col_map = {
        "game_id": "game_id",
        "game_date_et": "game_date",
        "home_team_id": "home_team",
        "away_team_id": "away_team",
    }
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns for rest.py: {missing}")
        return _fill_defaults_mlb(df)

    df = df.rename(columns=col_map)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.tz_localize(None)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    if df["game_date"].isna().any():
        n_bad = int(df["game_date"].isna().sum())
        logger.warning(f"Dropping {n_bad} rows with invalid game_date")
        df = df.dropna(subset=["game_date"])
        if df.empty:
            return _fill_defaults_mlb(pd.DataFrame(columns=PLACEHOLDER_COLS))

    # Build team–game log
    home_log = df[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away_log = df[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    full_log = pd.concat([home_log, away_log], ignore_index=True)

    # Compute previous game date per team
    prev_log = (
        full_log
        .sort_values(["team", "game_date", "game_id"], ignore_index=True)
        .drop_duplicates(["team", "game_date"], keep="first")
    )
    prev_log["prev_game_date"] = prev_log.groupby("team")["game_date"].shift(1)
    prev_map = prev_log.set_index(["team", "game_date"])["prev_game_date"]

    # Merge prev_game_date for both sides
    for side in ("home", "away"):
        df = df.merge(
            prev_map.rename("prev_game_date"),
            left_on=(f"{side}_team", "game_date"),
            right_index=True,
            how="left",
        ).rename(columns={"prev_game_date": f"prev_{side}_game_date"})

    # Calculate raw rest days
    df["rest_days_home"] = (df["game_date"] - df["prev_home_game_date"]).dt.days
    df["rest_days_away"] = (df["game_date"] - df["prev_away_game_date"]).dt.days

    # Identify each team’s first game_id
    first_game_for_team_map = ( # Series: team_id (index) -> first_game_id (values)
        full_log
        .sort_values(["team", "game_date", "game_id"])
        .drop_duplicates("team", keep="first")
        .set_index("team")["game_id"]
    )

    # Get the first game_id for the actual home team in each row
    home_team_first_game = df["home_team"].map(first_game_for_team_map)
    # Get the first game_id for the actual away team in each row
    away_team_first_game = df["away_team"].map(first_game_for_team_map)

    # Apply DEF_REST_MLB if the current game_id matches the home team's specific first game_id
    df.loc[df["game_id"] == home_team_first_game, "rest_days_home"] = DEF_REST_MLB
    # Apply DEF_REST_MLB if the current game_id matches the away team's specific first game_id
    df.loc[df["game_id"] == away_team_first_game, "rest_days_away"] = DEF_REST_MLB

    # Then, the general fillna for any other NaNs (e.g., if a team wasn't in first_game_for_team_map, though unlikely)
    df["rest_days_home"].fillna(DEF_REST_MLB, inplace=True)
    df["rest_days_away"].fillna(DEF_REST_MLB, inplace=True)

    # Schedule density: count games in the past 7/14 days (exclusive)
    sched_log = (
        full_log
        .drop_duplicates(["team", "game_date"], keep="first")
        .set_index("game_date")
        .sort_index()
    )
    if not sched_log.empty:
        cnt7 = (
            sched_log.groupby("team")["game_id"]
            .rolling("7D", closed="neither")
            .count()
            .rename("cnt7")
        )
        cnt14 = (
            sched_log.groupby("team")["game_id"]
            .rolling("14D", closed="neither")
            .count()
            .rename("cnt14")
        )
        sched_counts = pd.concat([cnt7, cnt14], axis=1).reset_index()
        for side in ("home", "away"):
            df = df.merge(
                sched_counts,
                left_on=(f"{side}_team", "game_date"),
                right_on=("team", "game_date"),
                how="left",
            ).rename(columns={
                "cnt7": f"games_last_7_days_{side}",
                "cnt14": f"games_last_14_days_{side}",
            }).drop(columns=["team"], errors="ignore")
    else:
        for side in ("home", "away"):
            df[f"games_last_7_days_{side}"] = 0
            df[f"games_last_14_days_{side}"] = 0

    # Fill missing density and enforce types
    for side in ("home", "away"):
        df[f"games_last_7_days_{side}"] = df[f"games_last_7_days_{side}"].fillna(0).astype(int)
        df[f"games_last_14_days_{side}"] = df[f"games_last_14_days_{side}"].fillna(0).astype(int)

    # B2B flags and advantage metrics
    df["is_back_to_back_home"] = (df["rest_days_home"] == 1).astype(int)
    df["is_back_to_back_away"] = (df["rest_days_away"] == 1).astype(int)
    df["rest_advantage"] = df["rest_days_home"] - df["rest_days_away"]
    df["schedule_advantage"] = df["games_last_7_days_away"] - df["games_last_7_days_home"]

    # Cleanup intermediate columns
    df.drop(columns=["prev_home_game_date", "prev_away_game_date"], errors="ignore", inplace=True)

    # Final default-filling
    out = _fill_defaults_mlb(df)

    logger.debug("Finished rest.transform; output shape %s", out.shape)
    if debug:
        logger.setLevel(orig_level)
    return out
