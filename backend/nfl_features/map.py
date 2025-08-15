# backend/nfl_features/map.py
"""
MAP stage – exogenous, leakage-safe, pre-game context features.

What it does:
  1) Join static venue metadata (roof, city, lat/lon, orientation) by home team.
  2) Optionally attach the latest pre-kick forecast (temp/wind/humidity/POP).
  3) Optionally attach venue/month climatology and compute forecast deltas.
  4) Emit one row per game_id with all columns prefixed `map_`.

Inputs:
  - games_df: schedule spine with at least [game_id] and team columns.
              We try, in order, to identify home/away via:
                home_team_norm/home_team_name/home_team
                away_team_norm/away_team_name/away_team
              For kickoff time, we prefer:
                scheduled_time (UTC)  OR  kickoff_ts (UTC)  OR  game_date (+ game_time)
  - weather_latest_df (optional): dataframe mirroring the matview
      `weather_nfl_latest_forecast_per_game` with columns:
        [game_id, fore_temp_f, fore_wind_mph, fore_humidity_pct, fore_precip_prob_pct, is_indoor]
      If provided, merged by game_id.
  - climatology_df (optional): dataframe mirroring the view `nfl_weather_climatology`
      with columns (wide or long):
        venue_city_norm, month, p50_temp_f, p50_wind_mph, p50_precip_prob_pct
      If provided, joined by (venue_city_norm, month).

Notes:
  - For *indoor* games (domes), weather vars are zeroed and flagged not applicable.
  - If forecast is missing for an OUTDOOR game but climatology is available, we backfill
    forecast values with climatology and set `map_weather_imputed=1`.
  - If nothing weather-related is available: emit static venue features, and keep
    `map_weather_applicable` consistent with roof type.

Return:
  DataFrame with one row per game_id. All new columns are prefixed `map_`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["compute_map_features"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_ts_utc(df: pd.DataFrame) -> pd.Series:
    """
    Build a UTC kickoff timestamp for each row:
      - Prefer 'scheduled_time' (UTC) if present.
      - Else 'kickoff_ts' (UTC).
      - Else combine 'game_date' + 'game_time' and assume America/New_York → UTC.
    """
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # scheduled_time (UTC; supabase schedule)
    if "scheduled_time" in df.columns:
        try:
            st = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)
            out = out.fillna(st)
        except Exception:
            pass

    # kickoff_ts (UTC)
    if "kickoff_ts" in df.columns:
        try:
            kt = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
            out = out.fillna(kt)
        except Exception:
            pass

    # Fallback: naive local (ET) → UTC
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

    out = out.fillna(utc)
    return out


def _load_stadium_data(stadium_data_path: Optional[str] = None) -> Dict:
    """
    Load data/stadium_data.json (supports multiple fallback paths).
    """
    candidates = []
    if stadium_data_path:
        candidates.append(Path(stadium_data_path))
    # typical project layouts
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "data" / "stadium_data.json")   # backend/data/...
    candidates.append(here.parents[2] / "data" / "stadium_data.json")   # project_root/data/...

    for p in candidates:
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            continue

    raise FileNotFoundError("stadium_data.json not found in expected locations. "
                            "Pass stadium_data_path=... or place file under backend/data/.")


def _resolve_team_cols(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Return (home_team_norm, away_team_norm) as lower-cased normalized strings.
    """
    def pick(*names: str) -> Optional[pd.Series]:
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    h = pick("home_team_norm", "home_team_name", "home_team")
    a = pick("away_team_norm", "away_team_name", "away_team")

    if h is None or a is None:
        # best-effort empty series to avoid crashes
        h = pd.Series("", index=df.index)
        a = pd.Series("", index=df.index)

    h_norm = h.astype(str).apply(normalize_team_name).str.lower()
    a_norm = a.astype(str).apply(normalize_team_name).str.lower()
    return h_norm, a_norm


def _team_to_venue_map(stadium_json: Dict) -> Dict[str, Dict]:
    """
    Build a normalized mapping: team_name_norm -> venue dict for NFL only.
    """
    nfl = stadium_json.get("NFL") or stadium_json.get("nfl") or {}
    out: Dict[str, Dict] = {}
    for raw_team, venue in nfl.items():
        key = normalize_team_name(str(raw_team)).lower()
        out[key] = dict(venue)  # copy
    return out


def _profile(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    if df is None or df.empty:
        logger.debug("[MAP][PROFILE] %s: empty", name)
        return
    nulls = df.isna().sum().sort_values(ascending=False)
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    logger.debug("[MAP][PROFILE] %s rows=%d cols=%d | null_top=%s | const=%s",
                 name, len(df), df.shape[1], {k:int(v) for k,v in nulls.head(8).items() if v>0}, const[:8])


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def compute_map_features(
    games_df: pd.DataFrame,
    *,
    weather_latest_df: Optional[pd.DataFrame] = None,
    climatology_df: Optional[pd.DataFrame] = None,
    stadium_data_path: Optional[str] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build exogenous, leakage-safe features with `map_` prefix.
    Returns one row per game_id.
    """
    if games_df is None or games_df.empty:
        logger.warning("MAP: received empty games_df; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    # --- spine setup ---
    spine = games_df.copy()
    if "game_id" not in spine.columns:
        logger.warning("MAP: games_df missing game_id; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    spine["map_kickoff_utc"] = _build_ts_utc(spine)
    spine["map_month"] = spine["map_kickoff_utc"].dt.month.fillna(
        pd.to_datetime(spine.get("game_date", pd.NaT), errors="coerce").dt.month
    )

    home_norm, away_norm = _resolve_team_cols(spine)
    spine["home_team_norm"] = home_norm
    spine["away_team_norm"] = away_norm

    # --- load static venue data (home team => venue) ---
    try:
        stadium_data = _load_stadium_data(stadium_data_path)
    except Exception as e:
        logger.error("MAP: failed to load stadium_data.json: %s", e)
        return pd.DataFrame(columns=["game_id"])

    team2venue = _team_to_venue_map(stadium_data)

    # Build static venue frame keyed by game_id
    def _lookup_home(row) -> Dict:
        v = team2venue.get(str(row.get("home_team_norm", "")).lower())
        return v or {}

    static_cols = [
        "stadium", "city", "latitude", "longitude", "orientation", "is_indoor",
        # If surface/retractable ever appear in stadium_data.json, we pick them up transparently:
        "surface", "is_retractable",
    ]
    static_rows = []
    for i, row in spine.iterrows():
        v = _lookup_home(row)
        rec = {
            "game_id": row["game_id"],
            "map_venue_name": v.get("stadium"),
            "map_venue_city": v.get("city"),
            "map_latitude": v.get("latitude", np.nan),
            "map_longitude": v.get("longitude", np.nan),
            "map_orientation_deg": v.get("orientation", np.nan),
            "map_is_dome": int(bool(v.get("is_indoor", False))),
            "map_is_retractable": int(bool(v.get("is_retractable", False))),
            "map_surface": v.get("surface", None),
        }
        static_rows.append(rec)

    venue_df = pd.DataFrame(static_rows)
    venue_df["map_is_outdoor"] = ((venue_df["map_is_dome"] == 0) & (venue_df["map_is_retractable"] == 0)).astype(int)

    # Surface one-hots if available; else mark unknown
    if "map_surface" in venue_df.columns:
        surf = venue_df["map_surface"].astype(str).str.lower()
        venue_df["map_surface_grass"] = (surf == "grass").astype(int)
        venue_df["map_surface_turf"] = (surf == "turf").astype(int)
        venue_df["map_surface_unknown"] = ((venue_df["map_surface_grass"] == 0) & (venue_df["map_surface_turf"] == 0)).astype(int)
    else:
        venue_df["map_surface_grass"] = 0
        venue_df["map_surface_turf"] = 0
        venue_df["map_surface_unknown"] = 1

    # Join static to spine
    out = spine[["game_id", "map_kickoff_utc", "map_month", "home_team_norm", "away_team_norm"]].merge(
        venue_df, on="game_id", how="left"
    )

    # Normalized venue city for climatology join
    out["map_venue_city_norm"] = out["map_venue_city"].astype(str).str.strip().str.lower()

    # --- attach latest forecast if provided ---
    if weather_latest_df is not None and not weather_latest_df.empty:
        wf = weather_latest_df.copy()
        # Be liberal with column names; support either the ones we created in SQL or sensible variants
        rename_map = {}
        if "fore_temp_f" in wf.columns: rename_map["fore_temp_f"] = "map_fore_temp_f"
        if "fore_wind_mph" in wf.columns: rename_map["fore_wind_mph"] = "map_fore_wind_mph"
        if "fore_humidity_pct" in wf.columns: rename_map["fore_humidity_pct"] = "map_fore_humidity_pct"
        if "fore_precip_prob_pct" in wf.columns: rename_map["fore_precip_prob_pct"] = "map_fore_pop_pct"
        if "is_indoor" in wf.columns: rename_map["is_indoor"] = "map_is_dome_from_wf"

        # also tolerate capitalizations
        for src, dst in list(rename_map.items()):
            if src not in wf.columns:
                alt = src.upper()
                if alt in wf.columns:
                    rename_map[alt] = dst

        wf = wf.rename(columns=rename_map)

        cols_keep = ["game_id"] + list(rename_map.values())
        wf = wf[[c for c in cols_keep if c in wf.columns]].drop_duplicates("game_id", keep="last")

        out = out.merge(wf, on="game_id", how="left")

        # If the matview also exposes indoor flag, reconcile (prefer stadium_data.json first)
        if "map_is_dome_from_wf" in out.columns:
            out["map_is_dome"] = np.where(out["map_is_dome"].isna(), out["map_is_dome_from_wf"], out["map_is_dome"])
            out.drop(columns=["map_is_dome_from_wf"], inplace=True, errors="ignore")

    # --- attach climatology if provided ---
    if climatology_df is not None and not climatology_df.empty:
        cl = climatology_df.copy()
        # Expect: venue_city_norm, month, p50_temp_f, p50_wind_mph, p50_precip_prob_pct
        # Make sure keys exist/lower
        for k in ("venue_city_norm", "month"):
            if k not in cl.columns:
                logger.warning("MAP: climatology_df missing key column '%s'; skipping climo join.", k)
                cl = None
                break

        if cl is not None:
            rename_cl = {}
            if "p50_temp_f" in cl.columns: rename_cl["p50_temp_f"] = "map_climo_temp_f"
            if "p50_wind_mph" in cl.columns: rename_cl["p50_wind_mph"] = "map_climo_wind_mph"
            if "p50_precip_prob_pct" in cl.columns: rename_cl["p50_precip_prob_pct"] = "map_climo_pop_pct"
            cl = cl.rename(columns=rename_cl)
            keep = ["venue_city_norm", "month"] + list(rename_cl.values())
            cl = cl[keep].drop_duplicates(["venue_city_norm", "month"], keep="last")

            out = out.merge(
                cl,
                left_on=["map_venue_city_norm", "map_month"],
                right_on=["venue_city_norm", "month"],
                how="left",
                suffixes=("", "_climo"),
            ).drop(columns=["venue_city_norm", "month"], errors="ignore")

    # -----------------------------------------------------------------------------
    # Weather applicability & imputation
    # -----------------------------------------------------------------------------
    # Applicability: outdoor OR retractable treated as potentially weather-impacted.
    out["map_weather_applicable"] = ((out["map_is_dome"] == 0) | (out["map_is_retractable"] == 1)).astype(int)

    # Initialize forecast columns if missing
    for col in ("map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct", "map_fore_pop_pct"):
        if col not in out.columns:
            out[col] = np.nan

    # Imputation flag (1 if we backfilled missing forecast with climatology or zeroed for dome)
    out["map_weather_imputed"] = 0

    # 1) For outdoor/retractable: if forecast missing and climo present, backfill and flag
    mask_outdoor = out["map_weather_applicable"] == 1
    for fore_col, clm_col in [
        ("map_fore_temp_f", "map_climo_temp_f"),
        ("map_fore_wind_mph", "map_climo_wind_mph"),
        ("map_fore_humidity_pct", None),           # no climo humidity in our view
        ("map_fore_pop_pct", "map_climo_pop_pct"),
    ]:
        if clm_col and clm_col in out.columns:
            need = mask_outdoor & out[fore_col].isna() & out[clm_col].notna()
            if need.any():
                out.loc[need, fore_col] = out.loc[need, clm_col]
                out.loc[need, "map_weather_imputed"] = 1

    # 2) For domes: zero-out forecasts and flag imputed (not applicable)
    mask_dome = out["map_weather_applicable"] == 0
    for col in ("map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct", "map_fore_pop_pct"):
        out.loc[mask_dome, col] = 0.0
    out.loc[mask_dome, "map_weather_imputed"] = 1

    # -----------------------------------------------------------------------------
    # Derived: anomalies and condition flags (only meaningful for applicable games)
    # -----------------------------------------------------------------------------
    # anomalies (forecast - climatology) where both exist
    if "map_climo_temp_f" in out.columns:
        out["map_fore_temp_anom_f"] = out["map_fore_temp_f"] - out["map_climo_temp_f"]
    else:
        out["map_fore_temp_anom_f"] = np.nan

    if "map_climo_wind_mph" in out.columns:
        out["map_fore_wind_anom_mph"] = out["map_fore_wind_mph"] - out["map_climo_wind_mph"]
    else:
        out["map_fore_wind_anom_mph"] = np.nan

    if "map_climo_pop_pct" in out.columns:
        out["map_fore_pop_anom_pct"] = out["map_fore_pop_pct"] - out["map_climo_pop_pct"]
    else:
        out["map_fore_pop_anom_pct"] = np.nan

    # condition flags
    # thresholds are intentionally modest; tune as you observe validation lift
    windy_th = 12.0
    hot_th = 85.0
    cold_th = 32.0
    wet_th = 50.0  # POP >= 50%

    out["map_outdoor_windy"] = ((out["map_weather_applicable"] == 1) & (out["map_fore_wind_mph"] >= windy_th)).astype(int)
    out["map_outdoor_hot"]   = ((out["map_weather_applicable"] == 1) & (out["map_fore_temp_f"] >= hot_th)).astype(int)
    out["map_outdoor_cold"]  = ((out["map_weather_applicable"] == 1) & (out["map_fore_temp_f"] <= cold_th)).astype(int)
    out["map_precip_risk_high"] = ((out["map_weather_applicable"] == 1) & (out["map_fore_pop_pct"] >= wet_th)).astype(int)

    # Housekeeping: enforce numeric types where appropriate
    num_cols = [
        "map_latitude", "map_longitude", "map_orientation_deg",
        "map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct", "map_fore_pop_pct",
        "map_climo_temp_f", "map_climo_wind_mph", "map_climo_pop_pct",
        "map_fore_temp_anom_f", "map_fore_wind_anom_mph", "map_fore_pop_anom_pct",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    flag_cols = [
        "map_is_dome", "map_is_retractable", "map_is_outdoor",
        "map_surface_grass", "map_surface_turf", "map_surface_unknown",
        "map_weather_applicable", "map_weather_imputed",
        "map_outdoor_windy", "map_outdoor_hot", "map_outdoor_cold", "map_precip_risk_high",
    ]
    for c in flag_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    # Final column order (stable-ish; game_id first)
    keep = ["game_id",
            "map_kickoff_utc", "map_month",
            "map_venue_name", "map_venue_city", "map_venue_city_norm",
            "map_latitude", "map_longitude", "map_orientation_deg",
            "map_is_dome", "map_is_retractable", "map_is_outdoor",
            "map_surface", "map_surface_grass", "map_surface_turf", "map_surface_unknown",
            "map_weather_applicable", "map_weather_imputed",
            "map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct", "map_fore_pop_pct",
            "map_climo_temp_f", "map_climo_wind_mph", "map_climo_pop_pct",
            "map_fore_temp_anom_f", "map_fore_wind_anom_mph", "map_fore_pop_anom_pct",
            "map_outdoor_windy", "map_outdoor_hot", "map_outdoor_cold", "map_precip_risk_high",
           ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    # Enforce one row per game_id
    if out.duplicated("game_id").any():
        dup_ct = int(out.duplicated("game_id").sum())
        logger.warning("MAP: produced %d duplicate game_id rows; keeping last.", dup_ct)
        out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

    if debug:
        _profile(out, "MAP_OUT", debug)
        if "map_weather_applicable" in out.columns:
            logger.debug("MAP: applicability counts=%s", out["map_weather_applicable"].value_counts(dropna=False).to_dict())

    return out
