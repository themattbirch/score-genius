# backend/nfl_features/situational.py
from __future__ import annotations
import logging
from typing import Mapping
import pandas as pd

from . import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Divisional alignment map
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


def compute_situational_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive situational (context) features for each game.

    Output columns:
      - game_id
      - is_primetime
      - is_weekend
      - is_regular_season
      - is_playoffs
      - is_division_game
      - is_conference_game
    """
    if games_df.empty:
        return pd.DataFrame()

    df = games_df.copy()

    # ------------------------------
    # 1. Build a UTC timestamp series, per-row fallback from game_timestamp -> date+time
    # ------------------------------
    # Parse any UTC timestamps (may be NaT)
    if "game_timestamp" in df.columns:
        ts_ts = pd.to_datetime(df["game_timestamp"], utc=True, errors="coerce")
    else:
        ts_ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # Parse Eastern date+time
    dt_combo = df.get("game_date", pd.Series(dtype=str)).astype(str) + " " + df.get("game_time", pd.Series()).fillna("00:00:00")
    ts_dt = (
        pd.to_datetime(dt_combo, errors="coerce")
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    # Combine per-row: prefer ts_ts when present, else ts_dt
    ts = ts_ts.fillna(ts_dt)

    # Convert to Eastern for dow/hour
    ts_east = ts.dt.tz_convert("America/New_York")
    df["dow"] = ts_east.dt.dayofweek   # Mon=0 … Sun=6
    df["hour"] = ts_east.dt.hour

    # Primetime heuristic: Mon (0) or Thu (3), or Sun (6) after 19:00
    df["is_primetime"] = (
        (df["dow"].isin([0, 3])) | ((df["dow"] == 6) & (df["hour"] >= 19))
    ).astype(int)

    # Weekend = Sat(5) or Sun(6)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # ------------------------------
    # 2. Season stage flags (default missing->"")
    # ------------------------------
    if "stage" not in df.columns:
        df["stage"] = ""
    df["stage"] = df["stage"].fillna("").str.upper()
    df["is_regular_season"] = (df["stage"] == "REG").astype(int)
    df["is_playoffs"] = (df["stage"] == "PST").astype(int)

    # ------------------------------
    # 3. Rivalry flags via division / conference
    # ------------------------------
    df["home_team_canon"] = df["home_team_norm"].apply(utils.normalize_team_name)
    df["away_team_canon"] = df["away_team_norm"].apply(utils.normalize_team_name)

    df["home_div"] = df["home_team_canon"].map(TEAM_DIVISIONS)
    df["away_div"] = df["away_team_canon"].map(TEAM_DIVISIONS)

    df["home_conf"] = df["home_div"].str.split().str[0]
    df["away_conf"] = df["away_div"].str.split().str[0]

    df["is_division_game"] = (df["home_div"] == df["away_div"]).astype(int)
    df["is_conference_game"] = (
        (df["home_conf"] == df["away_conf"]) & (df["is_division_game"] == 0)
    ).astype(int)

    # ------------------------------
    # 4. Prune to exactly the 7 test‑expected columns
    # ------------------------------
    return df[
        [
            "game_id",
            "is_primetime",
            "is_weekend",
            "is_regular_season",
            "is_playoffs",
            "is_division_game",
            "is_conference_game",
        ]
    ].copy()
