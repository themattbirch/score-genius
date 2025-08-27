# backend/data/weather_pull.py
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

# --- minimal copy of the kickoff builder (kept local to avoid import cycles)
def _build_ts_utc(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if "scheduled_time" in df.columns:
        try:
            st = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)
            out = out.fillna(st)
        except Exception:
            pass

    if "kickoff_ts" in df.columns:
        try:
            kt = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
            out = out.fillna(kt)
        except Exception:
            pass

    date_str = df.get("game_date", pd.Series("", index=df.index)).astype(str)
    time_str = df.get("game_time", pd.Series("00:00:00", index=df.index)).astype(str)
    combo = (date_str + " " + time_str).str.strip()

    try:
        et = (
            pd.to_datetime(combo, errors="coerce")
            .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        )
        utc = et.dt.tz_convert("UTC")
    except Exception:
        utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    return out.fillna(utc)

def fetch_latest_weather_for_games(engine, games_df: pd.DataFrame, buffer_hours: int = 24) -> pd.DataFrame:
    """
    Pulls the *latest pre-kick* forecast for every NFL game whose kickoff falls
    in the window [min_kickoff - buffer, max_kickoff + buffer], using the raw
    snapshots table to avoid leakage.

    Returns columns aligned to map.py's adapter:
      [game_id, scheduled_time, is_indoor, fore_temp_f, fore_wind_mph,
       fore_humidity_pct, fore_pop_pct, forecast_captured_at, forecast_valid_at]
    """
    if games_df is None or games_df.empty:
        return pd.DataFrame(columns=[
            "game_id", "scheduled_time", "is_indoor",
            "fore_temp_f", "fore_wind_mph", "fore_humidity_pct", "fore_pop_pct",
            "forecast_captured_at", "forecast_valid_at"
        ])

    spine = games_df.copy()
    kickoff_utc = _build_ts_utc(spine)
    if kickoff_utc.isna().all():
        return pd.DataFrame(columns=[
            "game_id", "scheduled_time", "is_indoor",
            "fore_temp_f", "fore_wind_mph", "fore_humidity_pct", "fore_pop_pct",
            "forecast_captured_at", "forecast_valid_at"
        ])

    tmin = (kickoff_utc.min() - pd.Timedelta(hours=buffer_hours)).tz_convert("UTC")
    tmax = (kickoff_utc.max() + pd.Timedelta(hours=buffer_hours)).tz_convert("UTC")

    # NOTE: this uses your schedule table to define the game universe, and
    # picks the latest snapshot <= scheduled_time by home_team_norm.
    sql = text("""
    WITH gm AS (
      SELECT
        g.game_id,
        g.scheduled_time,
        lower(trim(both from g.home_team)) AS home_team_norm,
        lower(trim(both from g.venue))     AS venue_norm
      FROM public.nfl_game_schedule g
      WHERE g.scheduled_time BETWEEN :tmin AND :tmax
        AND g.scheduled_time IS NOT NULL
    ),
    snaps AS (
      SELECT
        gm.game_id,
        gm.scheduled_time,
        gm.venue_norm,
        s.is_indoor,
        s.temperature_f        AS fore_temp_f,
        s.wind_speed_mph       AS fore_wind_mph,
        s.humidity_pct         AS fore_humidity_pct,
        s.precip_prob_pct      AS fore_pop_pct,
        s.captured_at          AS forecast_captured_at,
        s.forecast_valid_at    AS forecast_valid_at,
        row_number() OVER (
          PARTITION BY gm.game_id
          ORDER BY s.captured_at DESC
        ) AS rn
      FROM gm
      JOIN public.weather_forecast_snapshots s
        ON s.sport = 'NFL'
       AND s.team_name_norm = gm.home_team_norm
       AND s.captured_at   <= gm.scheduled_time
    )
    SELECT
      game_id,
      scheduled_time,
      is_indoor,
      fore_temp_f,
      fore_wind_mph,
      fore_humidity_pct,
      fore_pop_pct,
      forecast_captured_at,
      forecast_valid_at,
      venue_norm
    FROM snaps
    WHERE rn = 1
    """)
    wx = pd.read_sql(sql, con=engine, params={"tmin": tmin, "tmax": tmax})

    # Ensure expected dtypes
    for c in ("fore_temp_f", "fore_wind_mph", "fore_humidity_pct", "fore_pop_pct"):
        if c in wx.columns:
            wx[c] = pd.to_numeric(wx[c], errors="coerce")

    # map.py can use venue_norm if present for climo join
    return wx
