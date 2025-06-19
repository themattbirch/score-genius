# backend/mlb_features/rest.py

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    MLB_DEFAULTS: Dict[str, float] = {}

DEF_REST_DAYS = float(MLB_DEFAULTS.get("rest_days", 7.0))


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _first_ts(row: pd.Series):
    """Return the first non-null timestamp in priority order."""
    for c in ("game_date_time_utc", "scheduled_time_utc", "game_date_et"):
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT


def _utc_midnight(ts: pd.Series | pd.Timestamp | str | None):
    """tz-aware or naive → tz-naive UTC midnight"""
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
          .dt.tz_localize(None)
          .dt.normalize()
    )


# --------------------------------------------------------------------------- #
def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    log = logger.debug

    if df.empty:
        return df
    cur = df.copy()

    # ── helpers ───────────────────────────────────────────────────
    def first_ts(r):
        for c in ("game_date_time_utc", "scheduled_time_utc", "game_date_et"):
            if c in r and pd.notna(r[c]): return r[c]
        return pd.NaT
    def midnight(ts):
        return (
            pd.to_datetime(ts, utc=True, errors="coerce")
              .tz_localize(None)
              .normalize()
        )

    for col in ("home_team_norm", "away_team_norm"):
        cur[col] = cur[col].astype(str).str.lower().str.strip()

    cur["_gdate"]  = _utc_midnight(cur.apply(_first_ts, axis=1))
    cur["_season"] = cur["_gdate"].dt.year

    # historical_df → “hist” block
    if historical_df is not None and not historical_df.empty:
        hist = historical_df.copy()
        # normalize and compute _gdate/_season
        for col in ("home_team_norm","away_team_norm"):
            hist[col] = hist[col].astype(str).str.lower().str.strip()
        hist["_gdate"]  = _utc_midnight(hist.apply(_first_ts, axis=1))
        hist["_season"] = hist["_gdate"].dt.year
        hist = hist.loc[hist["_gdate"].notna()]        # drop bad dates

        # **NEW**: only keep strictly earlier games
        cutoff = cur["_gdate"].min()
        hist = hist.loc[hist["_gdate"] < cutoff]
    else:
        hist = pd.DataFrame(columns=cur.columns)


    # ── long format ---------------------------------------------------------
    def rows(frame, side):
        col = f"{side}_team_norm"
        return frame[[ "game_id", col, "_gdate", "_season" ]].rename(
            columns={col:"team"}
        )

    long = pd.concat(
        [rows(hist,"home"), rows(hist,"away"), rows(cur,"home"), rows(cur,"away")],
        ignore_index=True
    ).dropna(subset=["_gdate"])

    long.sort_values(["team","_season","_gdate"], inplace=True)
    long.drop_duplicates(["team","_gdate","game_id"], keep="first", inplace=True)

    long["prev_date"] = (
        long.groupby(["team","_season"])["_gdate"].shift(1)
    )

    # unique (game_id, team) → prev_date
    prev_lookup = { (gid, t): p for gid, t, p
                    in zip(long["game_id"], long["team"], long["prev_date"]) }

    cur["prev_home"] = [
        prev_lookup.get((gid, t), pd.NaT)
        for gid, t in zip(cur["game_id"], cur["home_team_norm"])
    ]
    cur["prev_away"] = [
        prev_lookup.get((gid, t), pd.NaT)
        for gid, t in zip(cur["game_id"], cur["away_team_norm"])
    ]

    # ── rest-day metrics ----------------------------------------------------
    dh = (cur["_gdate"] - cur["prev_home"]).dt.days
    da = (cur["_gdate"] - cur["prev_away"]).dt.days

    cur["rest_days_home"] = dh.sub(1).clip(lower=0).fillna(DEF_REST_DAYS)
    cur["rest_days_away"] = da.sub(1).clip(lower=0).fillna(DEF_REST_DAYS)
    cur["is_back_to_back_home"] = (cur["rest_days_home"] == 0).astype(int)
    cur["is_back_to_back_away"] = (cur["rest_days_away"] == 0).astype(int)
    cur["rest_advantage"] = cur["rest_days_home"] - cur["rest_days_away"]

    cur.drop(columns=["prev_home","prev_away","_gdate","_season"],
             inplace=True, errors="ignore")
    log("Finished rest.transform; shape=%s", cur.shape)
    return cur
