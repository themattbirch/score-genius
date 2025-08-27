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
    - weather_latest_df (optional): dataframe from
      `public.weather_nfl_latest_forecast_per_game` with columns such as:
        [game_id, fore_temp_f, fore_wind_mph, fore_humidity_pct, fore_pop_pct,
         is_indoor, forecast_captured_at, forecast_valid_at]
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
import re
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

def _norm_token(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _resolve_team_cols(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Return (home_team_norm, away_team_norm) as lower-cased normalized strings.
    Prefer textual names if present; fall back to *_norm.
    """
    def pick(*names: str) -> Optional[pd.Series]:
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    # prefer the most descriptive fields
    h = pick("home_team_name", "home_team", "home_team_norm")
    a = pick("away_team_name", "away_team", "away_team_norm")

    if h is None or a is None:
        h = pd.Series("", index=df.index)
        a = pd.Series("", index=df.index)

    h_norm = h.astype(str).apply(normalize_team_name).str.lower()
    a_norm = a.astype(str).apply(normalize_team_name).str.lower()
    return h_norm, a_norm

def _build_id_to_team_map(stadium_json: Dict) -> Dict[str, str]:
    """
    Heuristic bridge for datasets where normalize_team_name(home_team_id) -> 'id_#'.
    We map 'id_1'..'id_32' onto the NFL keys found in stadium_data.json, sorted
    deterministically by normalized team name. This matches many alphabetical-ID schemas.
    """
    nfl = stadium_json.get("NFL") or stadium_json.get("nfl") or {}
    # deterministically sorted by normalized team token
    teams_sorted = sorted(
        [str(k) for k in nfl.keys()],
        key=lambda s: normalize_team_name(s).lower()
    )
    # 1-based ids: id_1 -> first, id_32 -> last
    return {f"id_{i+1}": normalize_team_name(teams_sorted[i]).lower() for i in range(len(teams_sorted))}


def _team_to_venue_map(stadium_json: Dict) -> Dict[str, Dict]:
    """
    Build a robust mapping: several keys per team (normalized name, nickname, abbr, city)
    -> venue dict. We only index what's available.
    """
    nfl = stadium_json.get("NFL") or stadium_json.get("nfl") or {}
    out: Dict[str, Dict] = {}

    for raw_team, venue in nfl.items():
        v = dict(venue)

        # 1) normalized full team name from the JSON key
        key_norm = normalize_team_name(str(raw_team)).lower()
        out[key_norm] = v

        # 2) nickname (if we can infer it from the normalized name, last token)
        parts = key_norm.split()
        if len(parts) >= 2:
            nick = parts[-1]
            out.setdefault(nick, v)

        # 3) abbreviation fields inside the venue (common patterns)
        for abbr_key in ("abbr", "team_abbr", "team_abbreviation"):
            abbr = str(v.get(abbr_key, "")).strip().lower()
            if abbr:
                out.setdefault(abbr, v)

        # 4) city key for good measure (lets us route via TEAM_TO_CITY)
        city = str(v.get("city", "")).strip().lower()
        if city:
            out.setdefault(city, v)

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
    snapshot_df: Optional[pd.DataFrame] = None,
    stadium_data_path: Optional[str] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Build exogenous, leakage-safe features with `map_` prefix.
    Returns one row per game_id.
    """
    if games_df is None or games_df.empty:
        logger.warning("MAP: received empty games_df; returning empty.")
        return pd.DataFrame(columns=["game_id"])
    
    if debug:
        logger.setLevel(logging.DEBUG)

    # --- spine setup ---
    spine = games_df.copy()
    if "game_id" not in spine.columns:
        logger.warning("MAP: games_df missing game_id; returning empty.")
        return pd.DataFrame(columns=["game_id"])
    
    orig_game_id_dtype = games_df["game_id"].dtype
    spine["game_id"] = spine["game_id"].astype(str).str.strip()

    spine["map_kickoff_utc"] = _build_ts_utc(spine)
    spine["map_month"] = spine["map_kickoff_utc"].dt.month.fillna(
        pd.to_datetime(spine.get("game_date", pd.NaT), errors="coerce").dt.month
    )

    home_norm, away_norm = _resolve_team_cols(spine)
    spine["home_team_norm"] = home_norm
    spine["away_team_norm"] = away_norm

    # --- DEBUG: what do our team tokens look like? ---
    if debug:
        try:
            ht_preview = spine["home_team_norm"].astype(str).head(12).tolist()
            uniq_sample = (
                spine["home_team_norm"].astype(str).str.lower().value_counts().head(15).index.tolist()
            )
            logger.debug("MAP: sample home_team_norm=%s", ht_preview)
            logger.debug("MAP: top15 unique home_team_norm tokens=%s", uniq_sample)
        except Exception as e:
            logger.debug("MAP: failed to preview team tokens: %s", e)





    # --- load static venue data (home team => venue) ---
    try:
        stadium_data = _load_stadium_data(stadium_data_path)
    except Exception as e:
        logger.error("MAP: failed to load stadium_data.json: %s", e)
        return pd.DataFrame(columns=["game_id"])

    team2venue = _team_to_venue_map(stadium_data)

        # --- DEBUG: stadium JSON shape / keys ---
    if debug:
        try:
            nfl_blob = stadium_data.get("NFL") or stadium_data.get("nfl") or {}
            top_keys = list(nfl_blob.keys())[:10]
            logger.debug("MAP: stadium_data NFL keys sample=%s", top_keys)
            if top_keys:
                sample_v = nfl_blob.get(top_keys[0], {})
                logger.debug("MAP: sample venue record keys=%s", list(sample_v.keys()))
        except Exception as e:
            logger.debug("MAP: cannot introspect stadium_data: %s", e)

    id2team = _build_id_to_team_map(stadium_data)
    if debug:
        # show 3 samples to sanity check the mapping
        for probe in ("id_1", "id_16", "id_32"):
            logger.debug("MAP: id-bridge %s -> %s", probe, id2team.get(probe))

    # Canonical city (venue city) for each NFL team.
    # Values are already "normed" (lowercase, no punctuation) to match your climatology join.
    TEAM_TO_CITY = {
        "arizona cardinals":        "glendale",
        "atlanta falcons":          "atlanta",
        "baltimore ravens":         "baltimore",
        "buffalo bills":            "orchard park",
        "carolina panthers":        "charlotte",
        "chicago bears":            "chicago",
        "cincinnati bengals":       "cincinnati",
        "cleveland browns":         "cleveland",
        "dallas cowboys":           "arlington",
        "denver broncos":           "denver",
        "detroit lions":            "detroit",
        "green bay packers":        "green bay",
        "houston texans":           "houston",
        "indianapolis colts":       "indianapolis",
        "jacksonville jaguars":     "jacksonville",
        "kansas city chiefs":       "kansas city",
        "los angeles chargers":     "inglewood",
        "los angeles rams":         "inglewood",
        "las vegas raiders":        "paradise",
        "miami dolphins":           "miami gardens",
        "minnesota vikings":        "minneapolis",
        "new england patriots":     "foxborough",
        "new orleans saints":       "new orleans",
        "new york giants":          "east rutherford",
        "new york jets":            "east rutherford",
        "philadelphia eagles":      "philadelphia",
        "pittsburgh steelers":      "pittsburgh",
        "san francisco 49ers":      "santa clara",
        "seattle seahawks":         "seattle",
        "tampa bay buccaneers":     "tampa",
        "tennessee titans":         "nashville",
        "washington commanders":    "landover",
    }

    # Helpful historical/alias mappings → route to the same city tokens.
    TEAM_TO_CITY_ALIASES = {
        # Legacy franchises/relocations
        "st louis rams":        "inglewood",     # now Los Angeles Rams (SoFi, Inglewood)
        "san diego chargers":   "inglewood",     # now Los Angeles Chargers
        "oakland raiders":      "paradise",      # now Las Vegas Raiders (Allegiant in Paradise, NV)
        "washington football team": "landover",

        # Common short forms that might slip through
        "la rams":              "inglewood",
        "la chargers":          "inglewood",
        "lv raiders":           "paradise",
        "ny giants":            "east rutherford",
        "ny jets":              "east rutherford",
        "sf 49ers":             "santa clara",
        "kc chiefs":            "kansas city",
        "tb buccaneers":        "tampa",
    }

    # Build static venue frame keyed by game_id
    def _lookup_home(row) -> Dict:
        """
        Try several keys; if still not found, route via TEAM_TO_CITY city token
        and match by city inside the stadium JSON.
        """
        team = str(row.get("home_team_norm", "")).strip().lower()
        if not team:
            return {}

        # 0) If it's an id-token, bridge to a normalized team key first
        m = re.match(r"^id_(\d+)$", team)
        if m:
            bridged = id2team.get(team)
            if bridged:
                v = team2venue.get(bridged)
                if v:
                    return v

        # 1) direct hits (full norm, nickname, abbr, or city) from the mapping
        v = team2venue.get(team)
        if v:
            return v

        # 2) try last token (nickname)
        parts = team.split()
        if parts:
            nick = parts[-1]
            v = team2venue.get(nick)
            if v:
                return v

        # 3) route via canonical city map and then find a venue with that city
        alt_city = TEAM_TO_CITY.get(team) or TEAM_TO_CITY_ALIASES.get(team)
        if alt_city:
            for vv in team2venue.values():
                if str(vv.get("city", "")).strip().lower() == alt_city:
                    return vv

        return {}


    static_cols = [
        "stadium", "city", "latitude", "longitude", "orientation", "is_indoor",
        "surface", "is_retractable",
    ]
    static_rows = []

    # collapse common aliases → {"grass", "turf"}
    SURF_ALIAS = {
        "natural": "grass",
        "natural grass": "grass",
        "real grass": "grass",
        "hybrid": "grass",
        "artificial": "turf",
        "artificial turf": "turf",
        "astroturf": "turf",
        "fieldturf": "turf",
        "synthetic": "turf",
    }

    for i, row in spine.iterrows():
        v = _lookup_home(row)
        # read "surface" OR "map_surface" from the JSON; normalize + alias
        surf_raw = v.get("surface") or v.get("map_surface") or ""
        surf_norm = str(surf_raw).strip().lower()
        surf_std = SURF_ALIAS.get(surf_norm, surf_norm if surf_norm in {"grass", "turf"} else "unknown")

        rec = {
            "game_id": row["game_id"],
            "map_venue_name": v.get("stadium", None),
            "map_venue_city": v.get("city", None),
            "map_latitude": v.get("latitude", np.nan),
            "map_longitude": v.get("longitude", np.nan),
            "map_orientation_deg": v.get("orientation", np.nan),
            "map_is_dome": (1 if v.get("is_indoor") else 0) if ("is_indoor" in v) else np.nan,
            "map_is_retractable": (1 if v.get("is_retractable") else 0) if ("is_retractable" in v) else np.nan,
            "map_surface": surf_std,
            "map_surface_grass": int(surf_std == "grass"),
            "map_surface_turf": int(surf_std == "turf"),
            "map_surface_unknown": int(surf_std not in {"grass", "turf"}),
        }
        static_rows.append(rec)

    venue_df = pd.DataFrame(static_rows)
    venue_df["map_is_outdoor"] = ((venue_df["map_is_dome"] == 0) & (venue_df["map_is_retractable"] == 0)).astype(int)

    if debug:
        matched = venue_df["map_venue_name"].notna().sum()
        total = len(venue_df)
        logger.debug("MAP: stadium match coverage %d/%d (%.1f%%)", matched, total, 100.0 * matched / max(total,1))

    if debug:
        try:
            if venue_df["map_venue_name"].isna().any():
                missing_game_ids = spine.loc[
                    spine["game_id"].isin(venue_df.loc[venue_df["map_venue_name"].isna(), "game_id"]),
                    ["game_id", "home_team_norm"]
                ].drop_duplicates()
                miss_sample = missing_game_ids.head(20).to_dict("records")
                logger.debug("MAP: missing venue for %d games; sample=%s",
                             int(venue_df["map_venue_name"].isna().sum()), miss_sample)
        except Exception as e:
            logger.debug("MAP: failed to report missing venue sample: %s", e)


    # Surface one-hots if available; else mark unknown
    # ---- after you create venue_df, replace your current surface one-hot block ----
    if "map_surface" in venue_df.columns:
        s = venue_df["map_surface"].astype(str).str.strip().str.lower()
        s = s.replace(SURF_ALIAS)  # reuse the same alias map as above
        venue_df["map_surface_grass"] = (s == "grass").astype(int)
        venue_df["map_surface_turf"]  = (s == "turf").astype(int)
        venue_df["map_surface_unknown"] = ((venue_df["map_surface_grass"] == 0) & (venue_df["map_surface_turf"] == 0)).astype(int)
    else:
        venue_df["map_surface_grass"] = 0
        venue_df["map_surface_turf"] = 0
        venue_df["map_surface_unknown"] = 1


    # Join static to spine
    out = spine[["game_id", "map_kickoff_utc", "map_month", "home_team_norm", "away_team_norm"]].merge(
        venue_df, on="game_id", how="left"
    )

    # Static venue token derived from stadium JSON (works even when weather join doesn't land)
    out["map_venue_norm_static"] = out["map_venue_name"].apply(_norm_token)
    out["map_venue_city_norm"] = out["map_venue_city"].astype(str).str.strip().str.lower()
    # --- attach latest forecast if provided ---
    if weather_latest_df is not None and not weather_latest_df.empty:
        if "map_weather_match_method" not in out.columns:
            out["map_weather_match_method"] = np.nan
        wf = weather_latest_df.copy()

        # 🔧 same normalization here
        if "game_id" in wf.columns:
            wf["game_id"] = wf["game_id"].astype(str).str.strip()

        # Column adapter: accept either 'fore_*' or your 'temperature_f'/'wind_mph'/... schema.
        rename_map = {}

        # temperature
        if "fore_temp_f" in wf.columns:            rename_map["fore_temp_f"] = "map_fore_temp_f"
        if "temperature_f" in wf.columns:          rename_map["temperature_f"] = "map_fore_temp_f"

        # wind speed
        if "fore_wind_mph" in wf.columns:          rename_map["fore_wind_mph"] = "map_fore_wind_mph"
        if "wind_mph" in wf.columns:               rename_map["wind_mph"] = "map_fore_wind_mph"
        if "wind_speed_mph" in wf.columns:         rename_map["wind_speed_mph"] = "map_fore_wind_mph"

        # humidity
        if "fore_humidity_pct" in wf.columns:      rename_map["fore_humidity_pct"] = "map_fore_humidity_pct"
        if "humidity_pct" in wf.columns:           rename_map["humidity_pct"] = "map_fore_humidity_pct"

        # precip / POP
        if "fore_precip_prob_pct" in wf.columns:   rename_map["fore_precip_prob_pct"] = "map_fore_pop_pct"
        if "fore_pop_pct" in wf.columns:           rename_map["fore_pop_pct"] = "map_fore_pop_pct"
        if "precip_prob_pct" in wf.columns:        rename_map["precip_prob_pct"] = "map_fore_pop_pct"
        if "pop_pct" in wf.columns:                rename_map["pop_pct"] = "map_fore_pop_pct"
        if "prob_precip_pct" in wf.columns:        rename_map["prob_precip_pct"] = "map_fore_pop_pct"
        if "precip_probability" in wf.columns:
            wf["precip_probability"] = pd.to_numeric(wf["precip_probability"], errors="coerce") * 100.0
            rename_map["precip_probability"] = "map_fore_pop_pct"

        # indoor flag (bool/int)
        if "is_indoor" in wf.columns:              rename_map["is_indoor"] = "map_is_dome_from_wf"

        # venue token (if any)
        if "venue_norm" in wf.columns:             rename_map["venue_norm"] = "map_venue_norm"

        # timestamps (for freshness & analytics)
        if "forecast_captured_at" in wf.columns:   rename_map["forecast_captured_at"] = "map_fore_captured_at"
        if "forecast_valid_at" in wf.columns:      rename_map["forecast_valid_at"] = "map_fore_valid_at"

        if debug:
            logger.debug("MAP: weather_latest_df rows=%d cols=%s", len(wf), list(wf.columns))

        wf = wf.rename(columns=rename_map)

        # --- POP normalization: handle %, strings, and fractions uniformly ---
        if "map_fore_pop_pct" in wf.columns:
            s = wf["map_fore_pop_pct"]
            # strip "%" if present
            s = s.astype(str).str.replace("%", "", regex=False)
            s = pd.to_numeric(s, errors="coerce")
            # if many values are on 0..1 scale, scale them to 0..100
            try:
                frac_share = (s.between(0, 1, inclusive="both")).mean()
                if pd.notna(frac_share) and frac_share > 0.6:
                    s = s * 100.0
            except Exception:
                pass
            wf["map_fore_pop_pct"] = s


        cols_keep = ["game_id"] + [
            c for c in (
                "map_fore_temp_f","map_fore_wind_mph","map_fore_humidity_pct","map_fore_pop_pct",
                "map_is_dome_from_wf","map_venue_norm","map_fore_captured_at","map_fore_valid_at"
            ) if c in wf.columns
        ]
        wf = wf[cols_keep].drop_duplicates("game_id", keep="last")

        if debug:
            logger.debug("MAP: dtypes spine.game_id=%s wf.game_id=%s",
                         out["game_id"].dtype, wf["game_id"].dtype)
            logger.debug("MAP: spine ids sample=%s", out["game_id"].head().tolist())
            logger.debug("MAP: wf ids sample=%s", wf["game_id"].head().tolist())

        if debug:
            spine_ids = set(out["game_id"].astype(str))
            wf_ids    = set(weather_latest_df.get("game_id", pd.Series([], dtype=str)).astype(str))
            logger.debug("MAP: missing in weather (sample)=%s", list(spine_ids - wf_ids)[:10])
            logger.debug("MAP: missing in games   (sample)=%s", list(wf_ids - spine_ids)[:10])

        out = out.merge(wf, on="game_id", how="left")
        # track where a forecast landed
        if "map_fore_temp_f" in out.columns:
            out["map_weather_match_method"] = np.where(out["map_fore_temp_f"].notna(),
                                                       "game_id", out.get("map_weather_match_method"))

        if debug:
            try:
                m = pd.merge(out[["game_id"]].drop_duplicates(),
                             wf[["game_id"]].drop_duplicates(),
                             on="game_id", how="inner")
                logger.debug("MAP: weather joinable by game_id = %d/%d", len(m), len(wf))
            except Exception as e:
                logger.debug("MAP: weather joinable inspection failed: %s", e)
        
        # If no game_id matches landed, fall back to timestamp join (minute-level UTC)
        joined_by_id = int(out["map_fore_temp_f"].notna().sum() if "map_fore_temp_f" in out.columns else 0)
        if debug:
            logger.debug("MAP: forecast hits after game_id join=%d", joined_by_id)

        if joined_by_id == 0 and "scheduled_time" in weather_latest_df.columns and "map_kickoff_utc" in out.columns:
            wf_time = weather_latest_df.copy()
            wf_time = wf_time.rename(columns=rename_map)

            # ensure datetime alignment
            wf_time["scheduled_time"] = pd.to_datetime(wf_time["scheduled_time"], errors="coerce", utc=True)
            out["_kick_min_"] = pd.to_datetime(out["map_kickoff_utc"], errors="coerce", utc=True).dt.floor("T")
            wf_time["_sched_min_"] = wf_time["scheduled_time"].dt.floor("T")

            # keep only needed columns for the time-based join
            cols_keep_time = ["_sched_min_"] + [
                c for c in (
                    "map_fore_temp_f","map_fore_wind_mph","map_fore_humidity_pct","map_fore_pop_pct",
                    "map_is_dome_from_wf","map_venue_norm","map_fore_captured_at","map_fore_valid_at"
                ) if c in wf_time.columns
            ]
            wf_time = wf_time[cols_keep_time].drop_duplicates("_sched_min_", keep="last")

            # remove any stale forecast columns from the previous attempt
            out.drop(columns=[
                c for c in ("map_fore_temp_f","map_fore_wind_mph","map_fore_humidity_pct","map_fore_pop_pct",
                            "map_venue_norm","map_fore_captured_at","map_fore_valid_at","map_is_dome_from_wf")
                if c in out.columns
            ], inplace=True, errors="ignore")

            out = out.merge(
                wf_time,
                left_on="_kick_min_",
                right_on="_sched_min_",
                how="left"
            ).drop(columns=["_kick_min_","_sched_min_"], errors="ignore")

            if debug:
                joined_by_ts = int(out["map_fore_temp_f"].notna().sum() if "map_fore_temp_f" in out.columns else 0)
                logger.debug("MAP: forecast hits after time join=%d", joined_by_ts)

            if "map_fore_temp_f" in out.columns:
                out["map_weather_match_method"] = np.where(
                    out["map_fore_temp_f"].notna() & out["map_weather_match_method"].isna(),
                    "time_exact",
                    out.get("map_weather_match_method")
                )

            # --- Fallback #3: team/date nearest-time join (<= 180m) -----------------
            # If we still have no forecast after game_id + exact-time joins,
            # try matching by (home team OR venue) + same calendar date and pick the
            # closest scheduled_time within a tolerance window.
            hits_now = int(out["map_fore_temp_f"].notna().sum() if "map_fore_temp_f" in out.columns else 0)
            if hits_now == 0:
                tol_min = 4320  # 72 hours
                wf2 = weather_latest_df.copy().rename(columns=rename_map)

                # ensure time columns exist & are tz-aware
                if "scheduled_time" in wf2.columns:
                    wf2["scheduled_time"] = pd.to_datetime(wf2["scheduled_time"], errors="coerce", utc=True)
                else:
                    if debug:
                        logger.debug("MAP: team/date bridge skipped (weather DF missing scheduled_time)")
                    wf2 = None  # disable bridge

                def _apply_bridge_by(key_name: str) -> int:
                    """
                    key_name: 'home_team_norm' (preferred) or 'map_venue_norm' (fallback).
                    Returns the number of games that gained non-null forecast values.
                    """
                    nonlocal out, wf2
                    if wf2 is None:
                        return 0

                    # Build normalized join keys on both sides
                    if key_name == "home_team_norm":
                        if "home_team_norm" not in wf2.columns:
                            return 0
                        out["_k_"] = out["home_team_norm"].astype(str).str.strip().str.lower()
                        wf2["_k_"] = wf2["home_team_norm"].astype(str).str.strip().str.lower()
                    else:  # 'map_venue_norm' fallback
                        if "map_venue_norm" not in wf2.columns:
                            return 0
                        out["_k_"] = out["map_venue_norm_static"].astype(str).apply(_norm_token)
                        wf2["_k_"] = wf2["map_venue_norm"].astype(str).apply(_norm_token)

                    # Forecast columns we can bring over (whatever exists)
                    fore_cols = [c for c in (
                        "map_fore_temp_f","map_fore_wind_mph","map_fore_humidity_pct","map_fore_pop_pct",
                        "map_is_dome_from_wf","map_venue_norm","map_fore_captured_at","map_fore_valid_at"
                    ) if c in wf2.columns]

                    if not fore_cols:
                        return 0

                    # Candidate pairs: same key (no same-day prefilter)
                    cand = out[["game_id", "map_kickoff_utc", "_k_"]].merge(
                        wf2[["_k_", "scheduled_time"] + fore_cols],
                        on="_k_", how="left"
                    )

                    if cand.empty:
                        return 0

                    # Find nearest scheduled_time within tolerance
                    cand["delta_min"] = (cand["scheduled_time"] - cand["map_kickoff_utc"]).abs() / np.timedelta64(1, "m")
                    cand = cand.dropna(subset=["delta_min"])
                    if cand.empty:
                        return 0
                    cand = cand[cand["delta_min"] <= tol_min]
                    if cand.empty:
                        return 0

                    idx = cand.groupby("game_id")["delta_min"].idxmin()
                    best = cand.loc[idx, ["game_id"] + fore_cols].copy()

                    # Merge and fill only where still null
                    merged = out.merge(best, on="game_id", how="left", suffixes=("", "_bridge"))
                    gained_total = 0
                    for c in fore_cols:
                        cb = c + "_bridge"
                        if cb in merged.columns:
                            before = merged[c].notna().sum() if c in merged.columns else 0
                            merged[c] = merged[c].where(merged[c].notna(), merged[cb])
                            after = merged[c].notna().sum()
                            gained_total += max(0, after - before)
                            merged.drop(columns=[cb], inplace=True, errors="ignore")
                    out = merged
                    # label rows newly filled by this bridge
                    if "map_fore_temp_f" in out.columns:
                        filled_mask = out["map_fore_temp_f"].notna() & out["map_weather_match_method"].isna()
                        out.loc[filled_mask, "map_weather_match_method"] = (
                            "team_nearest" if key_name == "home_team_norm" else "venue_nearest"
                        )
                    return gained_total

            filled = _apply_bridge_by("home_team_norm")
            if debug:
                logger.debug("MAP: team+date bridge filled %d games (tol=%dm)", int(filled), tol_min)
            if filled == 0:
                filled = _apply_bridge_by("map_venue_norm")
                if debug:
                    logger.debug("MAP: venue+date bridge filled %d games (tol=%dm)", int(filled), tol_min)

            # --- Fallback #4: GLOBAL nearest-time join (<= 360m) -------------------
            # No reliance on team/venue ids; purely time-based nearest within tolerance.
            # This catches ET↔UTC stored offsets and minor schedule drift.
            final_hits_before = int(out["map_fore_temp_f"].notna().sum() if "map_fore_temp_f" in out.columns else 0)
            # --- Fallback #4a: VENUE-scoped nearest-time (<= 72h), if tokens exist ---
            if final_hits_before == 0 and not weather_latest_df.empty:
                if "scheduled_time" in weather_latest_df.columns:
                    wf = weather_latest_df.copy()
                    wf["scheduled_time"] = pd.to_datetime(wf["scheduled_time"], errors="coerce", utc=True)

                    # Try weather-provided venue_norm first (if any non-null), else static token
                    via = None
                    if "map_venue_norm" in wf.columns and wf["map_venue_norm"].notna().any():
                        wf["_map_key_venue_"] = wf["map_venue_norm"].apply(_norm_token)
                        via = "venue_norm_from_weather"

                    if via:
                        left = out[["game_id", "map_kickoff_utc", "map_venue_norm_static"]].copy()
                        left["_map_key_venue_"] = left["map_venue_norm_static"].apply(_norm_token)
                        left = left.dropna(subset=["map_kickoff_utc", "_map_key_venue_"])
                        wf = wf.dropna(subset=["scheduled_time", "_map_key_venue_"])

                        wf = wf[["_map_key_venue_", "scheduled_time"] + [c for c in (
                            "map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct", "map_fore_pop_pct",
                            "map_fore_captured_at", "map_fore_valid_at") if c in wf.columns]].sort_values(["_map_key_venue_", "scheduled_time"])
                        left = left.sort_values(["_map_key_venue_", "map_kickoff_utc"])

                        merged = pd.merge_asof(
                            left, wf,
                            left_on="map_kickoff_utc", right_on="scheduled_time",
                            by="_map_key_venue_", direction="nearest",
                            tolerance=pd.Timedelta(hours=72)
                        )

                        fill_df = merged.drop(columns=["_map_key_venue_", "scheduled_time", "map_kickoff_utc"], errors="ignore")
                        out = out.merge(fill_df, on="game_id", how="left", suffixes=("", "_vnear"))

                        for c in (
                            "map_fore_temp_f", "map_fore_wind_mph", "map_fore_humidity_pct",
                            "map_fore_pop_pct", "map_fore_captured_at", "map_fore_valid_at"):
                            if f"{c}_vnear" in out.columns:
                                out[c] = out[c].where(out[c].notna(), out[f"{c}_vnear"])
                                out.drop(columns=[f"{c}_vnear"], inplace=True, errors="ignore")

                        if "map_fore_temp_f" in out.columns:
                            filled_mask = out["map_fore_temp_f"].notna() & out["map_weather_match_method"].isna()
                            out.loc[filled_mask, "map_weather_match_method"] = via


                    # If we got an indoor flag from weather, reconcile with static
                    if "map_is_dome_from_wf" in out.columns:
                        out["map_is_dome"] = np.where(out["map_is_dome"].isna(), out["map_is_dome_from_wf"], out["map_is_dome"])
                        out.drop(columns=["map_is_dome_from_wf"], inplace=True, errors="ignore")

                    if debug:
                        final_hits_after = int(out["map_fore_temp_f"].notna().sum() if "map_fore_temp_f" in out.columns else 0)
                        logger.debug("MAP: forecast hits after GLOBAL nearest-time join=%d", final_hits_after)
                else:
                    if debug:
                        logger.debug("MAP: global nearest-time bridge skipped (weather DF missing scheduled_time)")

            # cleanup helpers
            out.drop(columns=[c for c in ["_k_","_kick_date_"] if c in out.columns], inplace=True, errors="ignore")
            if "wf2" in locals() and wf2 is not None:
                for c in ["_sched_date_","_k_"]:
                    if c in wf2.columns:
                        wf2.drop(columns=[c], inplace=True, errors="ignore")

        # -------------------------------------------------------------------------

        # Reconcile dome flag if provided in weather feed
        if "map_is_dome_from_wf" in out.columns:
            out["map_is_dome"] = np.where(out["map_is_dome"].isna(), out["map_is_dome_from_wf"], out["map_is_dome"])
            out.drop(columns=["map_is_dome_from_wf"], inplace=True, errors="ignore")

        # --- Fallback: team snapshot (weather_forecast_snapshots) -------------
        # If all forecast columns are still null, try team-level snapshots
        if (
            (out.get("map_fore_temp_f") is None or out["map_fore_temp_f"].isna().all()) and
            snapshot_df is not None and not snapshot_df.empty
        ):
            try:
                snap = snapshot_df.copy()

                # Normalize & coerce
                if "team_name_norm" in snap.columns:
                    snap["team_name_norm"] = (
                        snap["team_name_norm"].astype(str).str.strip().str.lower()
                    )
                for sc, mc in [
                    ("temperature_f", "map_fore_temp_f"),
                    ("wind_speed_mph", "map_fore_wind_mph"),
                    ("humidity_pct", "map_fore_humidity_pct"),
                ]:
                    if sc in snap.columns:
                        snap[sc] = pd.to_numeric(snap[sc], errors="coerce")

                # Keep last non-null per team
                keep_cols = ["team_name_norm"]
                for sc in ("temperature_f", "wind_speed_mph", "humidity_pct"):
                    if sc in snap.columns: keep_cols.append(sc)
                snap = snap[keep_cols].drop_duplicates("team_name_norm", keep="last")

                # Prepare LHS keys (bridge id_# -> normalized team token)
                if "home_team_norm" in out.columns:
                    out["_home_team_norm_"] = (
                        out["home_team_norm"]
                        .astype(str).str.strip().str.lower()
                        .apply(lambda t: id2team.get(t, t))   # <- uses the id2team map built earlier
                    )
                else:
                    out["_home_team_norm_"] = ""


                # Track fills
                merged["map_forecast_fallback_used"] = 0
                fills = 0
                if "temperature_f" in merged.columns:
                    before = merged["map_fore_temp_f"].notna().sum()
                    merged["map_fore_temp_f"] = merged["map_fore_temp_f"].where(
                        merged["map_fore_temp_f"].notna(), merged["temperature_f"]
                    )
                    after = merged["map_fore_temp_f"].notna().sum()
                    fills += max(0, after - before)
                if "wind_speed_mph" in merged.columns:
                    before = merged["map_fore_wind_mph"].notna().sum()
                    merged["map_fore_wind_mph"] = merged["map_fore_wind_mph"].where(
                        merged["map_fore_wind_mph"].notna(), merged["wind_speed_mph"]
                    )
                    after = merged["map_fore_wind_mph"].notna().sum()
                    fills += max(0, after - before)
                if "humidity_pct" in merged.columns:
                    before = merged["map_fore_humidity_pct"].notna().sum()
                    merged["map_fore_humidity_pct"] = merged["map_fore_humidity_pct"].where(
                        merged["map_fore_humidity_pct"].notna(), merged["humidity_pct"]
                    )
                    after = merged["map_fore_humidity_pct"].notna().sum()
                    fills += max(0, after - before)

                # Mark rows where any of the above filled
                any_filled = (
                    (merged.get("temperature_f") .notna() & merged["map_fore_temp_f"].notna())
                    if "temperature_f" in merged.columns else pd.Series(False, index=merged.index)
                )
                if "wind_speed_mph" in merged.columns:
                    any_filled = any_filled | (merged["wind_speed_mph"].notna() & merged["map_fore_wind_mph"].notna())
                if "humidity_pct" in merged.columns:
                    any_filled = any_filled | (merged["humidity_pct"].notna() & merged["map_fore_humidity_pct"].notna())

                merged.loc[any_filled, "map_forecast_fallback_used"] = 1
                if "map_weather_match_method" not in merged.columns:
                    merged["map_weather_match_method"] = np.nan
                merged.loc[any_filled, "map_weather_match_method"] = merged.loc[any_filled, "map_weather_match_method"].fillna("snapshot")

                # Fallback implies imputation for outdoors later; keep consistent with pipeline
                if "map_weather_imputed" in merged.columns:
                    merged.loc[any_filled, "map_weather_imputed"] = 1

                # Cleanup temp columns
                merged.drop(columns=[c for c in ("team_name_norm","temperature_f","wind_speed_mph","humidity_pct","_home_team_norm_") if c in merged.columns],
                            inplace=True, errors="ignore")

                out = merged

                if debug:
                    logger.debug("MAP: snapshot fallback filled %d values (team-level)", int(fills))
            except Exception as e:
                if debug:
                    logger.debug("MAP: snapshot fallback failed: %s", e)


        if debug:
            try:
                nn = {
                    "temp": int(out["map_fore_temp_f"].notna().sum()) if "map_fore_temp_f" in out.columns else 0,
                    "wind": int(out["map_fore_wind_mph"].notna().sum()) if "map_fore_wind_mph" in out.columns else 0,
                    "hum":  int(out["map_fore_humidity_pct"].notna().sum()) if "map_fore_humidity_pct" in out.columns else 0,
                    "pop":  int(out["map_fore_pop_pct"].notna().sum()) if "map_fore_pop_pct" in out.columns else 0,
                }
                logger.debug("MAP: forecast non-null counts %s", nn)
            except Exception as e:
                logger.debug("MAP: forecast non-null inspection failed: %s", e)


    # --- attach climatology if provided ---
    if climatology_df is not None and not climatology_df.empty:
        if debug:
            logger.debug("MAP: climatology_df rows=%d cols=%s",
                        len(climatology_df), list(climatology_df.columns))

        cl = climatology_df.copy()

        # Prefer venue-level join if we actually have tokens (from weather or static)
        has_venue_token = ("map_venue_norm" in out.columns) and out["map_venue_norm"].notna().any()
        has_static_venue_token = out["map_venue_norm_static"].notna().any()
        
        # Accept either median 'med_*' or p50_* names
        rename_cl = {}
        if "med_temp_f" in cl.columns:             rename_cl["med_temp_f"] = "map_climo_temp_f"
        if "med_wind_mph" in cl.columns:           rename_cl["med_wind_mph"] = "map_climo_wind_mph"
        if "med_precip_prob_pct" in cl.columns:    rename_cl["med_precip_prob_pct"] = "map_climo_pop_pct"
        if "med_humidity_pct" in cl.columns:       rename_cl["med_humidity_pct"] = "map_climo_humidity_pct"

        if "p50_temp_f" in cl.columns:             rename_cl["p50_temp_f"] = "map_climo_temp_f"
        if "p50_wind_mph" in cl.columns:           rename_cl["p50_wind_mph"] = "map_climo_wind_mph"
        if "p50_precip_prob_pct" in cl.columns:    rename_cl["p50_precip_prob_pct"] = "map_climo_pop_pct"

        cl = cl.rename(columns=rename_cl)

        # Choose keys
        if ("venue_norm" in cl.columns) and ("month" in cl.columns) and (has_venue_token or has_static_venue_token):
            # venue-level climo
            cl["_cl_key_venue_"] = cl["venue_norm"].apply(_norm_token)
            cl["_cl_key_month_"] = pd.to_numeric(cl["month"], errors="coerce")
            keep = ["_cl_key_venue_","_cl_key_month_"] + [c for c in ("map_climo_temp_f","map_climo_wind_mph","map_climo_pop_pct","map_climo_humidity_pct") if c in cl.columns]
            cl = cl[keep].drop_duplicates(["_cl_key_venue_","_cl_key_month_"], keep="last")

            # Try weather-provided venue_norm first (if any non-null), else static token
            via = None
            if has_venue_token:
                out["_map_key_venue_"] = out["map_venue_norm"].apply(_norm_token)
                via = "venue_norm_from_weather"
            else:
                out["_map_key_venue_"] = out["map_venue_norm_static"]
                via = "venue_norm_from_static"

            before_nonnull = int(out.get("map_climo_temp_f", pd.Series([np.nan]*len(out))).notna().sum())
            out = out.merge(
                cl,
                left_on=["_map_key_venue_", "map_month"],
                right_on=["_cl_key_venue_", "_cl_key_month_"],
                how="left"
            ).drop(columns=["_map_key_venue_","_cl_key_venue_","_cl_key_month_"], errors="ignore")
            # After the join, emit a single summary logger:
            after_nonnull = int(out["map_climo_temp_f"].notna().sum()) if "map_climo_temp_f" in out.columns else 0
            if debug:
                logger.debug("MAP: climo join via=%s → filled %d new rows",
                             via, max(0, after_nonnull - before_nonnull))


        else:
            # Fallback: try city-based climo (older shape)
            city_key_candidates = ["venue_city_norm", "city_norm", "venue_city", "city"]
            cl_city_key = next((k for k in city_key_candidates if k in cl.columns), None)
            month_key_candidates = ["month", "kickoff_month", "mo"]
            cl_month_key = next((k for k in month_key_candidates if k in cl.columns), None)

            if not cl_city_key or not cl_month_key:
                logger.warning("MAP: climatology_df missing required city/month key; skipping climo join (have: city=%s, month=%s).",
                            cl_city_key, cl_month_key)
            else:
                cl["__city_norm__"] = cl[cl_city_key].astype(str).str.strip().str.lower()
                cl["__month__"] = pd.to_numeric(cl[cl_month_key], errors="coerce")
                keep = ["__city_norm__","__month__"] + [c for c in ("map_climo_temp_f","map_climo_wind_mph","map_climo_pop_pct","map_climo_humidity_pct") if c in cl.columns]
                cl = cl[keep].drop_duplicates(["__city_norm__", "__month__"], keep="last")

                out = out.merge(
                    cl,
                    left_on=["map_venue_city_norm", "map_month"],
                    right_on=["__city_norm__", "__month__"],
                    how="left"
                ).drop(columns=["__city_norm__","__month__"], errors="ignore")

        # --- DEBUG: peek at joined climo rows ---
        if debug:
            try:
                cols_to_show = [c for c in ["game_id","map_venue_city_norm","map_month",
                                            "map_climo_temp_f","map_climo_wind_mph","map_climo_pop_pct"] 
                                if c in out.columns]
                climo_preview = out[cols_to_show].head(5).to_dict("records")
                logger.debug("MAP: joined climo sample=%s", climo_preview)
            except Exception as e:
                logger.debug("MAP: failed to preview joined climo rows: %s", e)

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
    has_venue_flags = out["map_is_dome"].notna() | out["map_is_retractable"].notna()
    out["map_weather_applicable"] = 0
    mask_known = has_venue_flags
    out.loc[mask_known, "map_weather_applicable"] = (
        ((out.loc[mask_known, "map_is_dome"] == 0) | (out.loc[mask_known, "map_is_retractable"] == 1))
    ).astype(int)

    # Only zero-out forecasts for *confirmed domes*
    mask_dome = (out["map_is_dome"] == 1)
    for col in ("map_fore_temp_f","map_fore_wind_mph","map_fore_humidity_pct","map_fore_pop_pct"):
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
        "map_forecast_fallback_used", "map_weather_match_method",
        ]

    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    # Ensure outdoor flag is consistent before profiling / casting
    if "map_is_dome" in out.columns:
        out["map_is_outdoor"] = ((out["map_is_dome"] == 0) & (out["map_is_retractable"] == 0)).astype(int)

    if debug:
        _profile(out, "MAP_OUT", debug)
        if "map_weather_applicable" in out.columns:
            logger.debug("MAP: applicability counts=%s", out["map_weather_applicable"].value_counts(dropna=False).to_dict())



    # Enforce one row per game_id
    if out.duplicated("game_id").any():
        dup_ct = int(out.duplicated("game_id").sum())
        logger.warning("MAP: produced %d duplicate game_id rows; keeping last.", dup_ct)
        out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

    if debug:
        _profile(out, "MAP_OUT", debug)
        if "map_weather_applicable" in out.columns:
            logger.debug("MAP: applicability counts=%s", out["map_weather_applicable"].value_counts(dropna=False).to_dict())
    if "map_is_dome" in out.columns:
        out["map_is_outdoor"] = ((out["map_is_dome"] == 0) & (out["map_is_retractable"] == 0)).astype(int)

    if debug:
        try:
            nonnull_static = int(out["map_venue_name"].notna().sum())
            logger.debug("MAP: final static venue coverage %d/%d (%.1f%%)",
                         nonnull_static, len(out), 100.0 * nonnull_static / max(len(out), 1))
        except Exception as e:
            logger.debug("MAP: final coverage log failed: %s", e)
    try:
        if pd.api.types.is_integer_dtype(orig_game_id_dtype):
            out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype(orig_game_id_dtype)
        else:
            out["game_id"] = out["game_id"].astype(orig_game_id_dtype)
    except Exception as e:
        logger.debug("MAP: could not restore original game_id dtype (%s): %s", orig_game_id_dtype, e)


    return out
