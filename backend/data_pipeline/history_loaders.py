# backend/data_pipeline/history_loaders.py

import os
from datetime import datetime, timedelta, timezone
import pandas as pd

# reuse the one true client
from backend.caching.supabase_client import supabase as SUPA

def load_recent_box_scores(lookback_days: int = 365) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = (
        SUPA.table("nba_historical_game_stats")
            .select("*")
            .gt("game_date", cutoff)
            .execute()
            .data
        or []
    )
    df = pd.DataFrame(rows)

    # --- 1) build home_score / away_score from the quarters ---
    home_qs = ["home_q1","home_q2","home_q3","home_q4","home_ot"]
    away_qs = ["away_q1","away_q2","away_q3","away_q4","away_ot"]
    df["home_score"] = df[home_qs].sum(axis=1)
    df["away_score"] = df[away_qs].sum(axis=1)

    # --- 2) (optional) compute the advanced metrics once here, so future steps don't have to ---
    from backend.nba_features.advanced import transform as advanced_transform
    df = advanced_transform(df, debug=False)

    return df


def load_team_season_stats() -> pd.DataFrame:
    """
    Grab the latest snapshot(s) from your historical team‐stats table.
    """
    # you probably want the most recent snapshot_date per team;
    # here we’ll just pull all ordered desc so you can dedupe later.
    resp = (
        SUPA
        .table("nba_historical_team_stats")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )
    rows = resp.data or []
    return pd.DataFrame(rows)


def load_schedule_next_14_days(days: int = 14) -> pd.DataFrame:
    """
    Pull the next `days` of scheduled games (for resting calculations).
    """
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=days)

    resp = (
        SUPA
        .table("nba_game_schedule")
        .select("*")
        .gte("scheduled_time", now_utc.isoformat())
        .lt("scheduled_time", end_utc.isoformat())
        .order("scheduled_time", desc=False)
        .execute()
    )
    rows = resp.data or []
    return pd.DataFrame(rows)
