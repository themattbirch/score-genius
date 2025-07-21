# backend/nfl_features/situational.py

from __future__ import annotations

"""Situational / game‑context feature generation for NFL games.

This module produces Boolean or categorical indicators that describe the *setting*
of a matchup rather than the quality of the teams themselves.  Features here are
cheap to compute (pure pandas) and do **not** require pulling additional box‑score
data.

The output is a *wide* DataFrame keyed by ``game_id`` so it can be merged
straight into the master feature table assembled in ``engine.py``.
"""

from typing import Mapping, List
import logging

import pandas as pd

from . import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Divisional alignment map  ▸  canonical_team_name  →  division string
# ---------------------------------------------------------------------------
TEAM_DIVISIONS: Mapping[str, str] = {
    # NFC East
    "cowboys": "NFC East", "giants": "NFC East", "eagles": "NFC East", "commanders": "NFC East",
    # NFC North
    "bears": "NFC North", "lions": "NFC North", "packers": "NFC North", "vikings": "NFC North",
    # NFC South
    "falcons": "NFC South", "panthers": "NFC South", "saints": "NFC South", "buccaneers": "NFC South",
    # NFC West
    "cardinals": "NFC West", "rams": "NFC West", "49ers": "NFC West", "seahawks": "NFC West",
    # AFC East
    "bills": "AFC East", "dolphins": "AFC East", "patriots": "AFC East", "jets": "AFC East",
    # AFC North
    "ravens": "AFC North", "bengals": "AFC North", "browns": "AFC North", "steelers": "AFC North",
    # AFC South
    "texans": "AFC South", "colts": "AFC South", "jaguars": "AFC South", "titans": "AFC South",
    # AFC West
    "broncos": "AFC West", "chiefs": "AFC West", "raiders": "AFC West", "chargers": "AFC West",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_situational_features(games_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Derive situational (context) features for each game.

    **Required columns** (case‑sensitive):
        - ``game_id`` : int
        - ``home_team_name`` / ``away_team_name`` : str  … raw names as stored
        - Either ``game_timestamp`` (UTC) **or** ``game_date`` *and* ``game_time``
        - ``stage`` : str  … "REG" | "PST" | "PRE"

    The function is deliberately *pure* (no DB access), making it safe to call
    inside parallel feature pipelines.
    """
    if games_df.empty:
        return pd.DataFrame()

    df = games_df.copy()

    # ------------------------------------------------------------------
    # 1. Normalise datetime information
    # ------------------------------------------------------------------
    if "game_timestamp" in df.columns:
        # Ensure tz‑aware timestamp and convert to US/Eastern for primetime calc.
        ts = pd.to_datetime(df["game_timestamp"], utc=True, errors="coerce")
    else:
        # Combine separate date + time columns; assume they are already Eastern.
        ts = pd.to_datetime(
            df["game_date"].astype(str) + " " + df["game_time"].fillna("00:00:00"),
            errors="coerce",
        ).dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        ts = ts.dt.tz_convert("UTC")  # Keep internal rep in UTC

    df["_ts_eastern"] = ts.dt.tz_convert("America/New_York")
    df["dow"] = df["_ts_eastern"].dt.dayofweek       # Monday=0 … Sunday=6
    df["hour"] = df["_ts_eastern"].dt.hour

    # Primetime heuristic: MNF (Mon), TNF (Thu), SNF (Sun after 19 ET)
    df["is_primetime"] = (
        (df["dow"].isin([0, 3])) | ((df["dow"] == 6) & (df["hour"] >= 19))
    ).astype(int)

    # Extra situational flags
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # Stage of season
    df["stage"] = df["stage"].fillna("").str.upper()
    df["is_regular_season"] = (df["stage"] == "REG").astype(int)
    df["is_playoffs"] = (df["stage"] == "PST").astype(int)

    # ------------------------------------------------------------------
    # 2. Rivalry indicators (division / conference)
    # ------------------------------------------------------------------
    df["home_team_canon"] = df["home_team_name"].apply(utils.normalize_team_name)
    df["away_team_canon"] = df["away_team_name"].apply(utils.normalize_team_name)

    df["home_div"] = df["home_team_canon"].map(TEAM_DIVISIONS)
    df["away_div"] = df["away_team_canon"].map(TEAM_DIVISIONS)

    # Conference = first token ("AFC"/"NFC")
    df["home_conf"] = df["home_div"].str.split().str[0]
    df["away_conf"] = df["away_div"].str.split().str[0]

    df["is_division_game"] = (df["home_div"] == df["away_div"]).astype(int)
    df["is_conference_game"] = (
        (df["home_conf"] == df["away_conf"]) & (df["is_division_game"] == 0)
    ).astype(int)

    # ------------------------------------------------------------------
    # 3. Select & return final columns
    # ------------------------------------------------------------------
    feature_cols: List[str] = [
        "game_id",
        "is_primetime",
        "is_weekend",
        "is_regular_season",
        "is_playoffs",
        "is_division_game",
        "is_conference_game",
    ]

    return df[feature_cols]


__all__ = ["compute_situational_features"]