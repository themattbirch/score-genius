# backend/nba_features/rest.py

"""
Wide‑coverage tests for backend.nba_features.rest.transform

Focus
-----
1.  Strict “no‑look‑ahead” guarantee when multiple games share a date.
2.  Correct rest‑day math, B2B flags, 7‑/14‑day density, and advantages.
3.  Idempotency + NaN‑free outputs.
"""

from __future__ import annotations
import logging
from typing import List

import numpy as np
import pandas as pd

from .utils import DEFAULTS

# LOGGER
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

__all__ = ["transform"]

# Placeholder/output columns
PLACEHOLDER_COLS: List[str] = [
    "rest_days_home", "rest_days_away",
    "games_last_7_days_home", "games_last_14_days_home",
    "games_last_7_days_away", "games_last_14_days_away",
    "is_back_to_back_home", "is_back_to_back_away",
    "rest_advantage", "schedule_advantage",
]

# Default rest days
DEF_REST = float(DEFAULTS.get("rest_days", 3))


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure placeholders exist and no NaNs remain."""
    for c in PLACEHOLDER_COLS:
        base = c.replace("_home", "").replace("_away", "")
        val = DEFAULTS.get(base, DEF_REST)
        if c.startswith("is_back_to_back"):
            val = 0
        if c not in df.columns:
            df[c] = val
        else:
            df[c] = df[c].fillna(val)
    return df


def transform(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("rest.transform – DEBUG ON")

    if df is None or df.empty:
        logger.warning("rest.transform: empty input.")
        return _fill_defaults(pd.DataFrame())

    # Drop any prior runs for idempotency
    df = df.drop(columns=[*PLACEHOLDER_COLS, "prev_home_game_date", "prev_away_game_date"], errors="ignore")
    df = df.copy().reset_index(drop=True)

    # Required columns
    req = {"game_id", "game_date", "home_team", "away_team"}
    if not req.issubset(df.columns):
        logger.error("Missing required columns: %s", req - set(df.columns))
        return _fill_defaults(df)

    # Normalize types
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.tz_localize(None)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    # Build log of every team appearance
    home_log = df[["game_date", "home_team", "game_id"]].rename(columns={"home_team": "team"})
    away_log = df[["game_date", "away_team", "game_id"]].rename(columns={"away_team": "team"})
    full_log = pd.concat([home_log, away_log], ignore_index=True)

    # Previous-game dates (drop same-day duplicates)
    prev_dates = (
        full_log[["team", "game_date", "game_id"]]
        .sort_values(["team", "game_date", "game_id"], ignore_index=True)
        .drop_duplicates(["team", "game_date"], keep="first")
        .assign(prev=lambda d: d.groupby("team")["game_date"].shift(1))
    )[['team', 'game_date', 'prev']]

    # Merge back to get rest days
    for side in ("home", "away"):
        df = df.merge(
            prev_dates,
            left_on=[f"{side}_team", "game_date"],
            right_on=["team", "game_date"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"prev": f"prev_{side}_game_date"}).drop(columns=["team"], errors="ignore")

    df["rest_days_home"] = (df["game_date"] - df["prev_home_game_date"]).dt.days.fillna(DEF_REST)
    df["rest_days_away"] = (df["game_date"] - df["prev_away_game_date"]).dt.days.fillna(DEF_REST)

    # Only the very first overall game for each team gets the default
    first_ids = (
        full_log
        .sort_values(["team", "game_date", "game_id"], ignore_index=True)
        .drop_duplicates(subset="team", keep="first")
        .set_index("team")["game_id"]
        .to_dict()
    )
    df["rest_days_home"] = np.where(
        df["game_id"] == df["home_team"].map(first_ids),
        DEF_REST,
        df["rest_days_home"],
    )
    df["rest_days_away"] = np.where(
        df["game_id"] == df["away_team"].map(first_ids),
        DEF_REST,
        df["rest_days_away"],
    )

    # Schedule density: drop duplicates, sort index, exclusive window
    sched_log = (
        full_log
        .drop_duplicates(subset=["team", "game_date"], keep="first")
        .set_index("game_date")
        .sort_index()
    )
    cnt7 = (
        sched_log.groupby("team")["game_id"]
        .rolling("7D", closed="neither")
        .count()
        .rename("cnt7")
        .reset_index()
    )
    cnt14 = (
        sched_log.groupby("team")["game_id"]
        .rolling("14D", closed="neither")
        .count()
        .rename("cnt14")
        .reset_index()
    )
    sched_counts = cnt7.merge(cnt14, on=["team", "game_date"]).drop_duplicates(["team","game_date"], keep="first")

    for side in ("home", "away"):
        df = df.merge(
            sched_counts,
            left_on=[f"{side}_team", "game_date"],
            right_on=["team", "game_date"],
            how="left",
            validate="many_to_one",
        ).rename(columns={
            "cnt7": f"games_last_7_days_{side}",
            "cnt14": f"games_last_14_days_{side}",
        }).drop(columns=["team"], errors="ignore")

        df[f"games_last_7_days_{side}"] = df[f"games_last_7_days_{side}"].fillna(0).astype(int)
        df[f"games_last_14_days_{side}"] = df[f"games_last_14_days_{side}"].fillna(0).astype(int)

    # Flags & advantages
    df["is_back_to_back_home"] = (df["rest_days_home"] == 1).astype(int)
    df["is_back_to_back_away"] = (df["rest_days_away"] == 1).astype(int)
    df["rest_advantage"] = df["rest_days_home"] - df["rest_days_away"]
    df["schedule_advantage"] = (
        df["games_last_7_days_away"] - df["games_last_7_days_home"]
    )

    # Cleanup & defaults
    df = df.drop(columns=["prev_home_game_date", "prev_away_game_date"], errors="ignore")
    df = _fill_defaults(df)

    logger.debug("rest.transform finished; shape=%s", df.shape)
    if debug:
        logger.setLevel(orig_level)
    return df
