# backend/mlb_features/rest.py

from __future__ import annotations
import logging
from typing import Dict, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

# Columns always produced by this transform
PLACEHOLDER_COLS: List[str] = [
    "rest_days_home",
    "rest_days_away",
    "is_back_to_back_home",
    "is_back_to_back_away",
    "rest_advantage",
    "games_last_7_days_home",
    "games_last_7_days_away",
    "games_last_14_days_home",
    "games_last_14_days_away",
    "schedule_advantage",
]

# Default rest days; test fixture will monkeypatch this
try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    MLB_DEFAULTS: Dict[str, float] = {}
DEF_REST_MLB = float(MLB_DEFAULTS.get("mlb_rest_days", 7.0))


def _first_ts(row: pd.Series):
    for c in ("game_date_time_utc", "scheduled_time_utc", "game_date_et"):
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT


def _utc_midnight(ts: pd.Series | pd.Timestamp | str | None):
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
          .dt.tz_localize(None)
          .dt.normalize()
    )


def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> pd.DataFrame:
    # Missing required columns guard
    required = ["game_id", "game_date_et", "home_team_id", "away_team_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns for rest.py: {missing}")
        out = df.copy()
        for col in PLACEHOLDER_COLS:
            if col.startswith(("is_back_to_back", "games_last")):
                out[col] = pd.Series([0] * len(out), dtype=int)
            else:
                out[col] = pd.Series([0.0] * len(out), dtype=float)
        return out

    # Empty input: return empty df with placeholder columns
    if df.empty:
        out = df.copy()
        for col in PLACEHOLDER_COLS:
            if col.startswith(("is_back_to_back", "games_last")):
                out[col] = pd.Series(dtype=int)
            else:
                out[col] = pd.Series(dtype=float)
        return out

    if debug:
        logger.setLevel(logging.DEBUG)
    log = logger.debug

    cur = df.copy()
    # normalize team codes
    for norm_col, src_col in [("home_team_norm", "home_team_id"), ("away_team_norm", "away_team_id")]:
        cur[norm_col] = cur[src_col].astype(str).str.lower().str.strip()

    # compute game-date and season
    cur["_gdate"] = cur.apply(_first_ts, axis=1).pipe(_utc_midnight)
    cur["_season"] = cur["_gdate"].dt.year

    # drop invalid dates
    invalid = cur["_gdate"].isna()
    if invalid.any():
        n = invalid.sum()
        logger.warning(f"Dropping {n} rows with invalid game_date")
        cur = cur.loc[~invalid].copy()

    # prepare historical and current rows for date mapping
    def _rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
        return frame[[f"{side}_team_norm", "_season", "_gdate"]].rename(
            columns={f"{side}_team_norm": "team"}
        )

    hist_rows = (pd.concat(
        [_rows(historical_df, "home"), _rows(historical_df, "away")],
        ignore_index=True
    ) if historical_df is not None and not historical_df.empty else
        pd.DataFrame(columns=["team","_season","_gdate"]))

    cur_rows = pd.concat(
        [_rows(cur, "home"), _rows(cur, "away")],
        ignore_index=True
    )

    # only include hist rows strictly before first current game date
    if not hist_rows.empty:
        cutoff = cur["_gdate"].min()
        hist_rows = hist_rows.loc[hist_rows["_gdate"] < cutoff]

    # unique dates per team-season
    date_df = pd.concat([hist_rows, cur_rows], ignore_index=True)
    date_df = date_df.drop_duplicates(["team","_season","_gdate"])
    date_df.sort_values(["team","_season","_gdate"], inplace=True)

    # build date->prev_date mapping
    date_map: Dict[tuple, pd.Timestamp] = {}
    for (team, season), group in date_df.groupby(["team","_season"]):
        dates = list(group["_gdate"])
        for i, d in enumerate(dates):
            prev = dates[i-1] if i > 0 else pd.NaT
            date_map[(team, season, d)] = prev

    # map prev_date
    cur["prev_home"] = cur.apply(
        lambda r: date_map.get((r["home_team_norm"], r["_season"], r["_gdate"]), pd.NaT),
        axis=1
    )
    cur["prev_away"] = cur.apply(
        lambda r: date_map.get((r["away_team_norm"], r["_season"], r["_gdate"]), pd.NaT),
        axis=1
    )

    # rest metrics: difference in days
    cur["rest_days_home"] = (
        cur["_gdate"] - cur["prev_home"]
    ).dt.days.fillna(DEF_REST_MLB).astype(float)
    cur["rest_days_away"] = (
        cur["_gdate"] - cur["prev_away"]
    ).dt.days.fillna(DEF_REST_MLB).astype(float)

    # back-to-back if rest_days <= 1
    cur["is_back_to_back_home"] = (cur["rest_days_home"] <= 1).astype(int)
    cur["is_back_to_back_away"] = (cur["rest_days_away"] <= 1).astype(int)
    cur["rest_advantage"] = (
        cur["rest_days_home"] - cur["rest_days_away"]
    ).astype(float)

    # schedule metrics: last 7/14 days counts per team-season
    for side in ("home", "away"):
        games7: List[int] = []
        games14: List[int] = []
        tcol = f"{side}_team_norm"
        for _, row in cur.iterrows():
            team = row[tcol]
            today = row["_gdate"]
            past_dates = [d for (tm, ss, d) in date_map if tm==team and ss==row["_season"] and d < today]
            games7.append(sum(d >= today - pd.Timedelta(days=7) for d in past_dates))
            games14.append(sum(d >= today - pd.Timedelta(days=14) for d in past_dates))
        cur[f"games_last_7_days_{side}"] = pd.Series(games7, dtype=int)
        cur[f"games_last_14_days_{side}"] = pd.Series(games14, dtype=int)

    cur["schedule_advantage"] = (
        (cur["games_last_7_days_away"] - cur["games_last_7_days_home"]).astype(float)
    )

    # cleanup
    cur.drop(columns=["prev_home","prev_away","_gdate","_season"], inplace=True, errors="ignore")

    # ensure placeholder cols exist
    for col in PLACEHOLDER_COLS:
        if col not in cur.columns:
            if col.startswith(("is_back_to_back","games_last")):
                cur[col] = pd.Series([0] * len(cur), dtype=int)
            else:
                cur[col] = pd.Series([0.0] * len(cur), dtype=float)

    log("Finished rest.transform; shape=%s", cur.shape)
    return cur
