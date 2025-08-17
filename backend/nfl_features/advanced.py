# backend/nfl_features/advanced.py
"""
Advanced efficiency features (leakage-safe) for NFL games.

Prefers prior-season aggregates (advanced_stats_df); otherwise computes from
team-game box scores (historical_team_stats_df), aggregates to season, and
optionally blends with current-season rolling (leak-safe; up to kickoff_ts or
game_date-1).

Key behaviors:
- Joins by PRIOR season (season - 1) for priors.
- Optional blend with current-season rolling means (windowed), weighted by the
  number of prior games this season.
- Joins by numeric team_id when available; otherwise falls back to normalized
  team name (team_norm).
- Computes advanced pace/efficiency metrics in fallback:
    pace:    seconds_per_play, plays_per_drive
    eff:     points_per_drive, yards_per_play, rush_yards_per_rush,
             turnover_rate_per_play, points_per_100_yards, pass_rate
    situational: 3rd/4th down %, red zone %
    (optional when present) sack_rate, explosive_play_rate, penalty_yards_per_play
- Emits home_/away_ values, adv_*_diff (computed after fills), total_adv_*,
  and *_imputed flags for observability.

This module avoids label leakage by never using same-day or future information.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, safe_divide

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform", "compute_advanced_metrics"]

# --- module toggles / constants ---
ADV_LOG_SUMMARY = True             # compact coverage log
ALLOW_SAME_SEASON_FALLBACK = True  # allow same-season proxy when prev-season missing
SAME_SEASON_SHRINK = 0.70          # base shrink for same-season priors
WINSORIZE_POST_BLEND = True        # toggle winsorization/clipping after blend

# families used for normalization / engineered totals
ADV_RATE_NAMES = {
    "third_down_pct", "fourth_down_pct", "red_zone_pct", "pythagorean_win_pct", "pass_rate",
    "sack_rate", "explosive_play_rate",
}
ADV_PER_PLAY_NAMES = {
    "yards_per_play", "rush_yards_per_rush", "turnover_rate_per_play", "penalty_yards_per_play",
}
ADV_PER_DRIVE_NAMES = {
    "yards_per_drive", "points_per_drive", "plays_per_drive",
}

# conservative caps (post-blend) to control outliers
WINSOR_CAPS = {
    "seconds_per_play": (20.0, 40.0),
    "pass_rate":        (0.35, 0.75),
    "yards_per_play":   (3.5, 7.5),
    "plays_per_drive":  (4.0, 9.0),
    "points_per_drive": (0.8, 3.8),
    "rush_yards_per_rush": (2.5, 6.5),
    "turnover_rate_per_play": (0.0, 0.08),
    "points_per_100_yards": (3.0, 11.0),
    "third_down_pct":   (0.20, 0.55),
    "fourth_down_pct":  (0.20, 0.80),
    "red_zone_pct":     (0.35, 0.75),
    "yards_per_drive":  (15.0, 55.0),
    # new caps for optional metrics
    "sack_rate":        (0.0, 0.15),
    "explosive_play_rate": (0.05, 0.20),
    "penalty_yards_per_play": (0.0, 1.5),
    "time_of_possession_seconds": (1200.0, 2400.0),  # 20–40 minutes
    "turnovers_per_game": (0.0, 4.0),
    "_default":         (-1e6, 1e6),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_merge_asof_sort(df: pd.DataFrame, on: str, by: List[str]) -> bool:
    """
    True if within every group defined by `by`, column `on` is monotonic non-decreasing.
    """
    if df is None or df.empty:
        return True
    try:
        return bool(df.groupby(by, dropna=False)[on].apply(lambda s: s.is_monotonic_increasing).all())
    except Exception:
        return False


def _first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _to_seconds_maybe(val) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    try:
        s = str(val)
        if ":" in s:
            # supports M:SS or MM:SS
            m, ss = s.split(":")
            return int(m) * 60 + int(ss)
        return float(s)
    except Exception:
        return np.nan


def _nfl_season_from_date(dt: pd.Timestamp) -> Optional[int]:
    """NFL season labeling: months 9–12 → that year; months 1–8 → previous year."""
    if pd.isna(dt):
        return None
    y = int(dt.year)
    return y if int(dt.month) >= 9 else (y - 1)


def _normalize_team_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_team_name).astype(str).str.lower()


def _normalize_adv_map(season_map: pd.DataFrame) -> pd.DataFrame:
    """Normalize/clip adv_* metrics: ensure rates in [0,1] (and convert 0–100 to 0–1 if needed)."""
    if season_map is None or season_map.empty:
        return season_map
    adv_cols = [c for c in season_map.columns if c.startswith("adv_")]
    for c in adv_cols:
        lc = c.lower()
        v = pd.to_numeric(season_map[c], errors="coerce")
        med = float(v.median()) if not np.isnan(v.median()) else np.nan
        if ("_pct" in lc or any(name in lc for name in ADV_RATE_NAMES)) and isinstance(med, float) and not np.isnan(med) and med > 1.1:
            v = v / 100.0
        if ("_pct" in lc or any(name in lc for name in ADV_RATE_NAMES)):
            season_map[c] = v.clip(0.0, 1.0).astype("float32")
        else:
            season_map[c] = v.astype("float32")
    return season_map


def _cap_for(col: str) -> Tuple[float, float]:
    for k, rng in WINSOR_CAPS.items():
        if k != "_default" and col.endswith(k):
            return rng
    return WINSOR_CAPS["_default"]


def _dedupe_by(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Deterministic de-duplication: keep the last row per key combination.
    """
    if not keys or any(k not in df.columns for k in keys):
        return df
    if not df.duplicated(subset=keys).any():
        return df
    before = len(df)
    out = df.sort_values(keys, kind="mergesort").drop_duplicates(subset=keys, keep="last")
    logger.warning("ADVANCED_FIX: De-duplicated at %s: %d -> %d rows", keys, before, len(out))
    return out


def _mk_kickoff_ts(df: pd.DataFrame, *, date_col="game_date", time_col="game_time", kickoff_col="kickoff_ts") -> pd.Series:
    """
    Build a robust, UTC-aware kickoff timestamp per row.
    Prefer kickoff_col if present; else compose from date+time (fallback midday).
    """
    # kickoff_ts if present
    if kickoff_col in df.columns:
        ko = pd.to_datetime(df[kickoff_col], errors="coerce", utc=True)
    else:
        ko = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # compose from date+time
    gd = pd.to_datetime(df.get(date_col, pd.NaT), errors="coerce")
    if time_col in df.columns:
        tt = df[time_col].astype(str).fillna("00:00:00")
        combo = pd.to_datetime(gd.astype(str) + " " + tt, errors="coerce")
        combo = combo.fillna(gd + pd.Timedelta(hours=12))
    else:
        combo = gd + pd.Timedelta(hours=12)

    try:
        combo = combo.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    except Exception:
        combo = pd.to_datetime(combo, errors="coerce", utc=True)

    ts = ko.where(ko.notna(), combo)
    return ts


# ---------------------------------------------------------------------------
# Per-game advanced metrics (used by fallback/rolling paths)
# ---------------------------------------------------------------------------

def compute_advanced_metrics(
    box_scores: pd.DataFrame,
    *,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Compute team-game advanced metrics with broad column synonym support."""
    if box_scores is None or box_scores.empty:
        return pd.DataFrame()

    df = box_scores.copy()

    # Identity columns
    if "team_id" in df.columns:
        df["team_id"] = _coerce_int(df["team_id"])

    team_norm_src = _first(df, ["team_norm", "team_name", "team", "team_abbr"])
    if team_norm_src:
        df["team_norm"] = _normalize_team_series(df[team_norm_src])

    # Attach game_date if present (downstream rolling uses it)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", utc=False)

    id_cols = [c for c in ["game_id", "team_id", "team_norm", "game_date"] if c in df.columns]
    out = df[id_cols].copy()

    # Synonyms
    plays_col       = _first(df, ["plays_total", "total_plays", "offensive_plays", "plays"])
    yards_col       = _first(df, ["yards_total", "total_yards", "net_yards", "yards"])
    points_col      = _first(df, ["team_score", "points", "pts", "score"])
    drives_col      = _first(df, ["total_drives", "drives", "offensive_drives", "num_drives"])
    pen_yards_col   = _first(df, ["penalty_yards", "penalty_yds", "pen_yds"])
    penalties_col   = _first(df, ["penalties", "penalties_total", "penalties_count"])
    rush_yards_col  = _first(df, ["rushings_total", "rushing_yards", "rush_yards"])
    rush_att_col    = _first(df, ["rushings_attempts", "rush_attempts", "rushes", "rushing_att"])
    third_made      = _first(df, ["third_down_made", "third_down_conversions", "third_downs_made", "third_downs_conv"])
    third_att       = _first(df, ["third_down_attempts", "third_downs", "third_downs_att"])
    fourth_made     = _first(df, ["fourth_down_made", "fourth_down_conversions"])
    fourth_att      = _first(df, ["fourth_down_attempts", "fourth_downs", "fourth_downs_att"])
    rz_made         = _first(df, ["red_zone_made", "red_zone_td", "redzone_td", "rzt_td"])
    rz_att          = _first(df, ["red_zone_att", "red_zone_trips", "redzone_att", "rzt_att"])
    turnovers_col   = _first(df, ["turnovers_total", "turnovers", "to_total"])
    # For pass rate
    pass_att_col    = _first(df, ["passing_attempts", "pass_att", "attempts_passing"])
    # possession time
    top_col         = _first(df, ["possession_time", "time_of_possession", "top"])
    # extras
    sacks_col       = _first(df, ["sacks_allowed", "sacks"])
    dropbacks_col   = _first(df, ["pass_dropbacks", "dropbacks"])
    explosive_col   = _first(df, ["explosive_plays", "explosive_total"])
    pass_sacks_alt  = sacks_col  # capture even if named "sacks"

    def _add(col_out: str, numer: pd.Series, denom: pd.Series, default_key: Optional[str] = None, *, scale: float = 1.0):
        default_val = float(DEFAULTS.get(default_key, 0.0)) if default_key else 0.0
        rate = safe_divide(pd.to_numeric(numer, errors="coerce"),
                           pd.to_numeric(denom, errors="coerce"),
                           default_val=default_val) * scale
        out[col_out] = pd.to_numeric(rate, errors="coerce").astype("float32")
        if flag_imputations:
            out[f"{col_out}_imputed"] = ((pd.to_numeric(denom, errors="coerce").isna()) | (pd.to_numeric(denom, errors="coerce") == 0) |
                                         (pd.to_numeric(numer, errors="coerce").isna())).astype("int8")

    # Core rates/ratios
    if third_made and third_att:
        _add("adv_third_down_pct", df[third_made], df[third_att], None)
    if fourth_made and fourth_att:
        _add("adv_fourth_down_pct", df[fourth_made], df[fourth_att], None)
    if rz_made and rz_att:
        _add("adv_red_zone_pct", df[rz_made], df[rz_att], None)

    if yards_col and plays_col:
        _add("adv_yards_per_play", df[yards_col], df[plays_col], "yards_per_play_avg")
    if rush_yards_col and rush_att_col:
        _add("adv_rush_yards_per_rush", df[rush_yards_col], df[rush_att_col], None)

    if turnovers_col and plays_col:
        _add("adv_turnover_rate_per_play", df[turnovers_col], df[plays_col], None)

    if yards_col and drives_col:
        _add("adv_yards_per_drive", df[yards_col], df[drives_col], None)
    if points_col and drives_col:
        _add("adv_points_per_drive", df[points_col], df[drives_col], None)
    if plays_col and top_col:
        top_seconds = df[top_col].apply(_to_seconds_maybe)
        _add("adv_seconds_per_play", top_seconds, df[plays_col], None)
    if plays_col and drives_col:
        _add("adv_plays_per_drive", df[plays_col], df[drives_col], None)
    if points_col and yards_col:
        _add("adv_points_per_100_yards", df[points_col], df[yards_col], "points_for_avg", scale=100.0)

    # Pass rate if we can approximate it (total attempts = pass + rush)
    if pass_att_col and rush_att_col:
        total_att = pd.to_numeric(df[pass_att_col], errors="coerce") + pd.to_numeric(df[rush_att_col], errors="coerce")
        _add("adv_pass_rate", pd.to_numeric(df[pass_att_col], errors="coerce"), total_att, None)

    # Optional: sack rate = sacks / dropbacks (or pass_att + sacks)
    if pass_sacks_alt and (dropbacks_col or pass_att_col):
        denom = (pd.to_numeric(df[dropbacks_col], errors="coerce")
                 if dropbacks_col else (pd.to_numeric(df[pass_att_col], errors="coerce") + pd.to_numeric(df[pass_sacks_alt], errors="coerce")))
        _add("adv_sack_rate", pd.to_numeric(df[pass_sacks_alt], errors="coerce"), denom, None)

    # Optional: explosive play rate = explosive plays / total plays
    if explosive_col and plays_col:
        _add("adv_explosive_play_rate", pd.to_numeric(df[explosive_col], errors="coerce"), pd.to_numeric(df[plays_col], errors="coerce"), None)

    # Optional: penalty yards per play
    if pen_yards_col and plays_col:
        _add("adv_penalty_yards_per_play", pd.to_numeric(df[pen_yards_col], errors="coerce"), pd.to_numeric(df[plays_col], errors="coerce"), None)

    # Keep only advanced columns + ids
    adv_cols = [c for c in out.columns if c.startswith("adv_") and not c.endswith("_imputed")]
    if not adv_cols:
        logger.warning("advanced.compute: no metrics computed. available_cols=%s", list(df.columns))
        return pd.DataFrame()

    keep_cols = [c for c in ["game_id", "team_id", "team_norm", "game_date"] if c in out.columns]
    impute_cols = [c for c in out.columns if c.endswith("_imputed")]
    return out[keep_cols + adv_cols + impute_cols].copy()


# ---------------------------------------------------------------------------
# Internal: build prior-season map (preferred source or fallback)
# ---------------------------------------------------------------------------

def _build_prior_season_map(
    *,
    advanced_stats_df: Optional[pd.DataFrame],
    historical_team_stats_df: Optional[pd.DataFrame],
    flag_imputations: bool,
) -> Optional[pd.DataFrame]:
    """
    Returns season_map with columns:
      keys: ["season"] + one of ["team_id"] or ["team_norm"]
      values: adv_* (float32)
    """
    season_map: Optional[pd.DataFrame] = None

    if advanced_stats_df is not None and not advanced_stats_df.empty:
        adv = advanced_stats_df.copy()

        # Keys
        if "team_id" in adv.columns:
            adv["team_id"] = _coerce_int(adv["team_id"])
        if "team_name" in adv.columns and "team_norm" not in adv.columns:
            adv["team_norm"] = _normalize_team_series(adv["team_name"])
        if "team_norm" in adv.columns:
            adv["team_norm"] = adv["team_norm"].astype(str).str.lower()
        if "season" in adv.columns:
            adv["season"] = _coerce_int(adv["season"])

        # Rename known aggregates to adv_*
        rename_map: Dict[str, str] = {}
        if "avg_third_down_pct" in adv.columns:
            rename_map["avg_third_down_pct"] = "adv_third_down_pct"
        if "avg_red_zone_pct" in adv.columns:
            rename_map["avg_red_zone_pct"] = "adv_red_zone_pct"
        if "avg_yards_per_drive" in adv.columns:
            rename_map["avg_yards_per_drive"] = "adv_yards_per_drive"
        if "avg_turnovers_per_game" in adv.columns:
            rename_map["avg_turnovers_per_game"] = "adv_turnovers_per_game"
        if "pythagorean_win_pct" in adv.columns:
            rename_map["pythagorean_win_pct"] = "adv_pythagorean_win_pct"
        if "avg_time_of_possession" in adv.columns:
            adv["adv_time_of_possession_seconds"] = adv["avg_time_of_possession"].apply(_to_seconds_maybe).astype("float32")

        adv = adv.rename(columns=rename_map)
        adv_cols = [c for c in adv.columns if c.startswith("adv_")]

        if "season" in adv.columns and adv_cols:
            keep_keys = [k for k in ["team_id", "team_norm"] if k in adv.columns]
            season_map = adv[["season"] + keep_keys + adv_cols].copy()
            season_map["__adv_source__"] = "preferred"

    if (season_map is None or season_map.empty) and historical_team_stats_df is not None and not historical_team_stats_df.empty:
        # Fallback: compute per-game then average by season/team (leak-safe since season priors)
        per_game = compute_advanced_metrics(historical_team_stats_df, flag_imputations=flag_imputations)
        if not per_game.empty:
            key_cols = [k for k in ["team_id", "team_norm"] if k in per_game.columns]

            # derive season from game_date if not provided
            if "season" not in historical_team_stats_df.columns:
                if "game_date" in historical_team_stats_df.columns:
                    s = pd.to_datetime(historical_team_stats_df["game_date"], errors="coerce", utc=False).apply(_nfl_season_from_date)
                    season_src = pd.DataFrame({"game_id": historical_team_stats_df["game_id"], "season": s})
                else:
                    season_src = None
            else:
                season_src = historical_team_stats_df[["game_id", "season"]].drop_duplicates()

            if "game_id" in per_game.columns and season_src is not None and not season_src.empty:
                per_game = per_game.merge(season_src, on="game_id", how="left")

            if key_cols and "season" in per_game.columns:
                adv_cols = [c for c in per_game.columns if c.startswith("adv_") and not c.endswith("_imputed")]
                grp = per_game.groupby(key_cols + ["season"], dropna=False)[adv_cols].mean().reset_index()
                grp["season"] = _coerce_int(grp["season"])
                if "team_id" in grp.columns:
                    grp["team_id"] = _coerce_int(grp["team_id"])
                season_map = grp
                season_map["__adv_source__"] = "fallback"

    if season_map is None or season_map.empty:
        return None

    # normalize & clip rates
    season_map = _normalize_adv_map(season_map)

    # dedupe by merge keys
    if "team_id" in season_map.columns:
        season_map = _dedupe_by(season_map, ["team_id", "season"])
    if "team_norm" in season_map.columns:
        season_map = _dedupe_by(season_map, ["team_norm", "season"])

    return season_map


# ---------------------------------------------------------------------------
# Internal: build current-season rolling map (leak-safe, optional blend)
# ---------------------------------------------------------------------------

def _build_current_season_rolling(
    *,
    historical_team_stats_df: Optional[pd.DataFrame],
    window: int = 5,
    min_periods: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Returns per-game rolling means (shifted) by (team_id or team_norm, season_current),
    ready for merge_asof. Columns:
      ["team_id"/"team_norm", "season_current", "__ts__", "game_date"] + adv_*_roll + ["roll_count"]
    """
    if historical_team_stats_df is None or historical_team_stats_df.empty:
        return None

    per_game = compute_advanced_metrics(historical_team_stats_df, flag_imputations=True)
    if per_game.empty:
        return None

    per_game = per_game.copy()

    # --- Defensive de-duplication on the RIGHT (rolling source) ---
    key_cols_for_dedupe = [c for c in ["team_id", "team_norm", "game_date"] if c in per_game.columns]
    if len(key_cols_for_dedupe) > 1 and per_game.duplicated(subset=key_cols_for_dedupe).any():
        per_game = _dedupe_by(per_game, key_cols_for_dedupe)

    per_game["game_date"] = pd.to_datetime(per_game.get("game_date", pd.NaT), errors="coerce", utc=False)
    per_game["__ts__"] = _mk_kickoff_ts(per_game, date_col="game_date", time_col="game_time" if "game_time" in per_game.columns else "game_time", kickoff_col="kickoff_ts" if "kickoff_ts" in per_game.columns else "kickoff_ts")
    per_game["season_current"] = per_game["game_date"].apply(_nfl_season_from_date).astype("Int64")

    key = "team_id" if ("team_id" in per_game.columns and per_game["team_id"].notna().any()) else ("team_norm" if "team_norm" in per_game.columns else None)
    if key is None:
        return None

    adv_cols = [c for c in per_game.columns if c.startswith("adv_") and not c.endswith("_imputed")]

    # Sort for group-wise rolling and for asof later
    per_game = per_game.sort_values(["season_current", key, "__ts__"], kind="mergesort").reset_index(drop=True)

    # Rolling means (leak-safe): shift(1) to exclude current game
    grp = per_game.groupby([key, "season_current"], dropna=False, sort=False)
    for c in adv_cols:
        per_game[f"{c}_roll"] = grp[c].transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()).astype("float32")
    per_game["roll_count"] = grp.cumcount().astype("Int64")  # number of prior games in-season

    # Normalize *_roll if they are rates/percentages
    roll_cols = [f"{c}_roll" for c in adv_cols]
    for rc in roll_cols:
        lc = rc.lower()
        if ("_pct" in lc or any(name in lc for name in ADV_RATE_NAMES)):
            v = pd.to_numeric(per_game[rc], errors="coerce")
            med = float(v.median()) if not np.isnan(v.median()) else np.nan
            if isinstance(med, float) and not np.isnan(med) and med > 1.1:
                v = v / 100.0
            per_game[rc] = v.clip(0.0, 1.0).astype("float32")

    cols = [key, "season_current", "__ts__", "game_date"] + roll_cols + ["roll_count"]
    return per_game[cols].copy()


# ---------------------------------------------------------------------------
# Same-season fallback applied after we know roll_counts (smarter shrink)
# ---------------------------------------------------------------------------

def _apply_same_season_fallback_after_roll(
    *,
    wide: pd.DataFrame,
    side: str,
    season_map: pd.DataFrame,
    key_kind: Optional[str],
    g_games: pd.DataFrame,
    roll_count_series: pd.Series,
    adv_cols_base: List[str],
) -> pd.DataFrame:
    """
    For rows where prev-season priors are entirely missing for a side, try same-season
    values from season_map and shrink based on prior games played this season.

    roll_count_series must be indexed to wide (aligned via game_id before passing).
    """
    key_col = f"{side}_{key_kind}" if key_kind in ("team_id", "team_norm") else None
    if key_col is None or key_col not in g_games.columns or "season" not in g_games.columns:
        return wide

    # Build probe with (key, season)
    probe = g_games[["game_id", key_col, "season"]].copy()
    # Map game_id alignment to wide index
    probe = wide[["game_id"]].merge(probe, on="game_id", how="left")

    # same-season slice from season_map
    if key_kind == "team_id" and "team_id" in season_map.columns:
        same_map = season_map[["team_id", "season"] + adv_cols_base].rename(columns={"team_id": key_col})
    elif key_kind == "team_norm" and "team_norm" in season_map.columns:
        same_map = season_map[["team_norm", "season"] + adv_cols_base].rename(columns={"team_norm": key_col})
    else:
        return wide

    miss_mask = wide[[f"{side}_prior_{c}" for c in adv_cols_base if f"{side}_prior_{c}" in wide.columns]].isna().all(axis=1)
    if not miss_mask.any():
        return wide

    ss = probe.merge(same_map, on=[key_col, "season"], how="left")
    # shrink = min(1, games_played/8) * SAME_SEASON_SHRINK
    ng = pd.to_numeric(roll_count_series, errors="coerce").fillna(0.0).astype(float)
    shrink = np.clip(ng / 8.0, 0.0, 1.0) * float(SAME_SEASON_SHRINK)

    for c in adv_cols_base:
        pcol = f"{side}_prior_{c}"
        if pcol not in wide.columns or c not in ss.columns:
            continue
        vals = pd.to_numeric(ss[c], errors="coerce")
        wide.loc[miss_mask, pcol] = (vals.loc[miss_mask] * shrink.loc[miss_mask]).astype("float32")

    return wide


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,                # used to backfill team IDs when missing
    historical_team_stats_df: Optional[pd.DataFrame] = None,     # per-game team stats (fallback + rolling)
    advanced_stats_df: Optional[pd.DataFrame] = None,            # season aggregates (preferred for priors)
    flag_imputations: bool = True,
    # rolling blend controls:
    enable_current_season_blend: bool = True,
    rolling_window: int = 5,
    blend_k_games: float = 5.0,  # trust fully after ~k prior games this season
    # winsorization toggle:
    winsorize_post_blend: bool = WINSORIZE_POST_BLEND,
) -> pd.DataFrame:
    """
    Attach leakage-free advanced efficiency features to `games`.

    Preference order for priors:
      1) advanced_stats_df (season aggregates)
      2) compute from historical_team_stats_df -> aggregate to season

    Optionally blends current-season rolling (up to kickoff_ts or game_date-1) with priors.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    g = games.copy()

    # Require season & (preferably) kickoff_ts or game_date for leak-safe rolling
    if "season" not in g.columns:
        logger.warning("advanced.transform: games missing 'season'; cannot attach advanced.")
        return pd.DataFrame(columns=["game_id"])
    g["season"] = _coerce_int(g["season"])

    if "game_date" in g.columns:
        g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce", utc=False)

    # Build robust kickoff timestamp for upcoming games
    g["__ts__"] = _mk_kickoff_ts(g, date_col="game_date", time_col="game_time" if "game_time" in g.columns else "game_time", kickoff_col="kickoff_ts" if "kickoff_ts" in g.columns else "kickoff_ts")

    # Ensure team keys on games
    have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))
    have_norms = all(c in g.columns for c in ("home_team_norm", "away_team_norm"))

    if not have_ids and historical_df is not None and not historical_df.empty:
        if all(c in historical_df.columns for c in ["game_id", "home_team_id", "away_team_id"]):
            id_map = historical_df[["game_id", "home_team_id", "away_team_id"]].drop_duplicates("game_id")
            g = g.merge(id_map, on="game_id", how="left")
            have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))

    if have_ids:
        g["home_team_id"] = _coerce_int(g["home_team_id"])
        g["away_team_id"] = _coerce_int(g["away_team_id"])

    if not have_ids and not have_norms:
        # derive normalized names on games
        for hn, an in [("home_team_name", "away_team_name"), ("home_team", "away_team")]:
            if hn in g.columns and an in g.columns:
                g["home_team_norm"] = _normalize_team_series(g[hn])
                g["away_team_norm"] = _normalize_team_series(g[an])
                have_norms = True
                break
    if have_norms:
        g["home_team_norm"] = _normalize_team_series(g["home_team_norm"])
        g["away_team_norm"] = _normalize_team_series(g["away_team_norm"])

    # ------------------------------------------------------------------
    # 1) Build prior-seasons map
    # ------------------------------------------------------------------
    season_map = _build_prior_season_map(
        advanced_stats_df=advanced_stats_df,
        historical_team_stats_df=historical_team_stats_df,
        flag_imputations=flag_imputations,
    )

    if season_map is None or season_map.empty:
        logger.warning("advanced: no usable prior-season source (checked advanced_stats_df, historical_team_stats_df).")
        return pd.DataFrame(columns=["game_id"])

    adv_cols_base = [c for c in season_map.columns if c.startswith("adv_")]
    can_join_by_id = have_ids and "team_id" in season_map.columns and season_map["team_id"].notna().any()
    can_join_by_norm = ("home_team_norm" in g.columns and "away_team_norm" in g.columns and "team_norm" in season_map.columns)

    # Compute per-season league averages for priors (metric-wise fallback)
    league_avgs = None
    try:
        if "season" in season_map.columns:
            adv_cols = [c for c in season_map.columns if c.startswith("adv_")]
            league_avgs = (season_map.groupby("season", dropna=False)[adv_cols].mean().reset_index()
                           .rename(columns={c: f"league_{c}" for c in adv_cols}))
    except Exception:
        league_avgs = None

    # Prepare prev-season frames
    g_home = g[["game_id", "season"]].copy()
    g_home["prev_season"] = (g_home["season"] - 1).astype("Int64")
    g_away = g_home.copy()

    if can_join_by_id:
        logger.debug("advanced: joining priors by team_id & prev_season")
        g_home = g_home.join(g[["game_id", "home_team_id"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_id"]].set_index("game_id"), on="game_id")

        map_home = _dedupe_by(season_map.rename(columns={"team_id": "home_team_id"}), ["home_team_id", "season"])
        map_away = _dedupe_by(season_map.rename(columns={"team_id": "away_team_id"}), ["away_team_id", "season"])

        home_join = g_home.merge(
            map_home,
            left_on=["home_team_id", "prev_season"],
            right_on=["home_team_id", "season"],
            how="left",
        )
        away_join = g_away.merge(
            map_away,
            left_on=["away_team_id", "prev_season"],
            right_on=["away_team_id", "season"],
            how="left",
        )

    elif can_join_by_norm:
        logger.debug("advanced: joining priors by team_norm & prev_season")
        g_home = g_home.join(g[["game_id", "home_team_norm"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_norm"]].set_index("game_id"), on="game_id")

        map_home = _dedupe_by(season_map.rename(columns={"team_norm": "home_team_norm"}), ["home_team_norm", "season"])
        map_away = _dedupe_by(season_map.rename(columns={"team_norm": "away_team_norm"}), ["away_team_norm", "season"])

        home_join = g_home.merge(
            map_home,
            left_on=["home_team_norm", "prev_season"],
            right_on=["home_team_norm", "season"],
            how="left",
        )
        away_join = g_away.merge(
            map_away,
            left_on=["away_team_norm", "prev_season"],
            right_on=["away_team_norm", "season"],
            how="left",
        )
    else:
        logger.warning(
            "advanced: cannot attach priors – missing join keys. games_cols=%s season_map_keys=%s",
            list(g.columns), [c for c in ["team_id", "team_norm"] if c in season_map.columns]
        )
        return pd.DataFrame(columns=["game_id"])

    # Build prior columns with explicit prefixes
    home_prior = home_join[["game_id"] + adv_cols_base].copy()
    home_prior.columns = ["game_id"] + [f"home_prior_{c}" for c in adv_cols_base]

    away_prior = away_join[["game_id"] + adv_cols_base].copy()
    away_prior.columns = ["game_id"] + [f"away_prior_{c}" for c in adv_cols_base]

    wide = home_prior.merge(away_prior, on="game_id", how="outer")

    # League-average fallback per metric for priors
    if league_avgs is not None and not league_avgs.empty:
        tmp = g[["game_id", "season"]].copy()
        tmp["prev_season"] = (tmp["season"] - 1).astype("Int64")
        tmp = tmp.merge(league_avgs.rename(columns={"season": "prev_season"}), on="prev_season", how="left")
        for c in adv_cols_base:
            league_c = f"league_{c}"
            for side in ("home", "away"):
                pcol = f"{side}_prior_{c}"
                if pcol in wide.columns and league_c in tmp.columns:
                    wide[pcol] = wide[pcol].fillna(pd.to_numeric(tmp[league_c], errors="coerce"))

    # If still NaN, fill with sensible global defaults (or 0.0)
    for c in adv_cols_base:
        dflt = float(DEFAULTS.get(c.replace("adv_", ""), 0.0))
        for side in ("home", "away"):
            pcol = f"{side}_prior_{c}"
            if pcol in wide.columns:
                wide[pcol] = pd.to_numeric(wide[pcol], errors="coerce").fillna(dflt).astype("float32")

    # ------------------------------------------------------------------
    # 2) Current-season rolling (optional) and blend
    # ------------------------------------------------------------------
    roll_map = None
    key_kind = None
    if enable_current_season_blend:
        roll_map = _build_current_season_rolling(
            historical_team_stats_df=historical_team_stats_df,
            window=rolling_window,
            min_periods=1,
        )
        if roll_map is not None:
            if "team_id" in roll_map.columns and have_ids:
                key_kind = "team_id"
            elif "team_norm" in roll_map.columns and have_norms:
                key_kind = "team_norm"

    def _merge_roll_per_group(side_key_col: str) -> Optional[pd.DataFrame]:
        """
        Per-(season_current, team) as-of join for rolling features (no global `by=`).
        This avoids the 'left keys must be sorted' pitfall from cross-group ordering.
        """
        if roll_map is None or key_kind is None:
            return None

        # Build LEFT slice
        tmp = g[["game_id", "__ts__", "season", side_key_col]].copy()
        tmp = tmp.rename(columns={side_key_col: key_kind})
        tmp["season_current"] = tmp["season"].astype("Int64")

        # Normalize dtypes and strings
        r = roll_map.copy()
        if key_kind == "team_id":
            tmp[key_kind] = _coerce_int(tmp[key_kind])
            r[key_kind] = _coerce_int(r[key_kind])
        else:
            tmp[key_kind] = _normalize_team_series(tmp[key_kind].astype(str))
            r[key_kind] = r[key_kind].astype(str).str.lower()

        # Ensure UTC tz-awareness on both sides
        tmp["__ts__"] = pd.to_datetime(tmp["__ts__"], errors="coerce", utc=True)
        r["__ts__"] = pd.to_datetime(r["__ts__"], errors="coerce", utc=True)

        # Filter null keys/timestamps
        tmp = tmp[tmp["season_current"].notna() & tmp[key_kind].notna() & tmp["__ts__"].notna()].copy()
        r   = r[  r["season_current"].notna() &   r[key_kind].notna() &   r["__ts__"].notna()].copy()

        if tmp.empty or r.empty:
            return pd.DataFrame(columns=["game_id"])

        # De-duplicate precisely at merge grain
        key_tuple = ["season_current", key_kind, "__ts__"]
        tmp = _dedupe_by(tmp, key_tuple)
        r   = _dedupe_by(r,   key_tuple)

        # Group keys set
        grp_keys = ["season_current", key_kind]

        out_parts: List[pd.DataFrame] = []

        # Iterate per group to avoid global ordering constraints
        for (season_cur, key_val), lsub in tmp.groupby(grp_keys, sort=False):
            rsub = r[(r["season_current"] == season_cur) & (r[key_kind] == key_val)]
            if rsub.empty:
                # No history for this group → return only game_id with NaNs later filled upstream
                out_parts.append(lsub[["game_id"]].copy())
                continue

            # Stable sort (tie-breaker = game_id on left)
            lsub = lsub.sort_values(["__ts__", "game_id"], kind="mergesort")
            rsub = rsub.sort_values(["__ts__"], kind="mergesort")

            joined = pd.merge_asof(
                left=lsub,
                right=rsub,
                left_on="__ts__",
                right_on="__ts__",
                direction="backward",
                allow_exact_matches=True,
            )

            roll_cols = [c for c in joined.columns if c.endswith("_roll")]
            if "roll_count" in joined.columns:
                roll_cols.append("roll_count")

            keep = ["game_id"] + roll_cols
            out_parts.append(joined[keep].copy())

        if not out_parts:
            return pd.DataFrame(columns=["game_id"])

        out = pd.concat(out_parts, ignore_index=True)
        # Final dedupe on game_id in case of any overlap (paranoia)
        if out["game_id"].duplicated().any():
            out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

        return out

    # Select keys for sides
    left_key  = "home_team_id" if (key_kind == "team_id") else ("home_team_norm" if key_kind == "team_norm" else None)
    right_key = "away_team_id" if (key_kind == "team_id") else ("away_team_norm" if key_kind == "team_norm" else None)

    home_roll = _merge_roll_per_group(left_key)  if left_key  else None
    away_roll = _merge_roll_per_group(right_key) if right_key else None

    # Same-season fallback AFTER roll counts (smarter shrink)
    if ALLOW_SAME_SEASON_FALLBACK and adv_cols_base:
        def _roll_n(df_roll: Optional[pd.DataFrame]) -> pd.Series:
            if df_roll is None or "roll_count" not in df_roll.columns:
                return pd.Series(0, index=wide.index, dtype="int64")
            j = wide[["game_id"]].merge(df_roll[["game_id", "roll_count"]], on="game_id", how="left")
            return pd.to_numeric(j["roll_count"], errors="coerce").fillna(0).astype(int)

        home_roll_n = _roll_n(home_roll)
        away_roll_n = _roll_n(away_roll)

        # temporarily attach keys for fallback function
        g_keys = g[["game_id", "season"]].copy()
        if key_kind == "team_id":
            g_keys["home_team_id"] = g.get("home_team_id")
            g_keys["away_team_id"] = g.get("away_team_id")
        elif key_kind == "team_norm":
            g_keys["home_team_norm"] = g.get("home_team_norm")
            g_keys["away_team_norm"] = g.get("away_team_norm")

        # ensure prior columns exist
        for c in adv_cols_base:
            for side in ("home", "away"):
                pcol = f"{side}_prior_{c}"
                if pcol not in wide.columns:
                    wide[pcol] = np.nan

        wide = _apply_same_season_fallback_after_roll(
            wide=wide, side="home", season_map=season_map, key_kind=key_kind,
            g_games=g.merge(g_keys, on="game_id", how="left"), roll_count_series=home_roll_n, adv_cols_base=adv_cols_base
        )
        wide = _apply_same_season_fallback_after_roll(
            wide=wide, side="away", season_map=season_map, key_kind=key_kind,
            g_games=g.merge(g_keys, on="game_id", how="left"), roll_count_series=away_roll_n, adv_cols_base=adv_cols_base
        )

    # ------------------------------------------------------------------
    # 3) Produce blended home_/away_ values (no fill yet)
    # ------------------------------------------------------------------
    base_metrics = [c.replace("adv_", "", 1) for c in adv_cols_base]

    def _get_prior(side: str, metric: str) -> pd.Series:
        col = f"{side}_prior_adv_{metric}"
        return wide[col] if col in wide.columns else pd.Series(np.nan, index=wide.index)

    def _get_roll(df_roll: Optional[pd.DataFrame], metric: str) -> pd.Series:
        if df_roll is None:
            return pd.Series(np.nan, index=wide.index)
        mcol = f"adv_{metric}_roll"
        joined = wide[["game_id"]].merge(df_roll, on="game_id", how="left")
        return joined[mcol] if mcol in joined.columns else pd.Series(np.nan, index=wide.index)

    def _get_roll_count(df_roll: Optional[pd.DataFrame]) -> pd.Series:
        if df_roll is None or "roll_count" not in df_roll.columns:
            return pd.Series(0, index=wide.index, dtype="int64")
        joined = wide[["game_id"]].merge(df_roll[["game_id", "roll_count"]], on="game_id", how="left")
        return pd.to_numeric(joined["roll_count"], errors="coerce").fillna(0).astype(int)

    # Precompute counts once
    home_roll_n = _get_roll_count(home_roll)
    away_roll_n = _get_roll_count(away_roll)

    def _blend_series(prior: pd.Series, roll: pd.Series, n_games: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Return blended series and imputed flag series."""
        ng = pd.to_numeric(n_games, errors="coerce").fillna(0.0).astype(float)
        w = np.clip(ng / float(blend_k_games), 0.0, 1.0)

        both = prior.notna() & roll.notna()
        only_prior = prior.notna() & ~roll.notna()
        only_roll = ~prior.notna() & roll.notna()

        blended = pd.Series(np.nan, index=prior.index, dtype="float32")
        blended.loc[both] = (1.0 - w[both]) * prior[both].astype(float) + w[both] * roll[both].astype(float)
        blended.loc[only_prior] = prior[only_prior].astype(float)
        blended.loc[only_roll] = roll[only_roll].astype(float)

        # Imputed if either source is missing (observability)
        imputed = (prior.isna() | roll.isna()).astype("int8")
        return blended.astype("float32"), imputed

    # ---- per-side blend & write-out ------------------------------------
    for side in ("home", "away"):
        df_roll = home_roll if side == "home" else away_roll
        n_games = home_roll_n if side == "home" else away_roll_n

        for metric in base_metrics:
            prior = _get_prior(side, metric)
            roll  = _get_roll(df_roll, metric) if enable_current_season_blend else pd.Series(np.nan, index=prior.index)

            if enable_current_season_blend:
                blended, imp = _blend_series(prior, roll, n_games)
            else:
                blended = pd.to_numeric(prior, errors="coerce").astype("float32")
                imp = (prior.isna()).astype("int8")

            wide[f"{side}_adv_{metric}"] = blended
            if flag_imputations:
                wide[f"{side}_adv_{metric}_imputed"] = imp

    # --- NO-NaN guard for side values (belt & suspenders) ---
    for metric in base_metrics:
        h = f"home_adv_{metric}"
        a = f"away_adv_{metric}"
        if h not in wide.columns:
            wide[h] = np.nan
        if a not in wide.columns:
            wide[a] = np.nan

    value_cols = [c for c in wide.columns if c.startswith(("home_adv_", "away_adv_")) and not c.endswith("_imputed")]

    # Winsorize then fill (stabilize outliers, ensure numeric & non-null for diffs/totals)
    for c in value_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
        if winsorize_post_blend:
            lo, hi = _cap_for(c)
            wide[c] = wide[c].clip(lower=lo, upper=hi)
        wide[c] = wide[c].astype("float32").fillna(0.0)

    out_cols: List[str] = ["game_id"]

    # Compute diffs/totals AFTER fill (stable; avoids NaN propagation)
    base_names = sorted({c.replace("home_adv_", "") for c in value_cols if c.startswith("home_adv_")})
    for metric in base_names:
        h = f"home_adv_{metric}"
        a = f"away_adv_{metric}"
        if h in wide.columns and a in wide.columns:
            diff_col = f"adv_{metric}_diff"
            tot_col  = f"total_adv_{metric}"

            wide[diff_col] = (wide[h] - wide[a]).astype("float32")
            wide[tot_col]  = (wide[h] + wide[a]).astype("float32")

            out_cols.extend([h, a, diff_col, tot_col])

            # diff_imputed if either side was imputed
            if flag_imputations:
                hi = f"{h}_imputed"
                ai = f"{a}_imputed"
                if hi in wide.columns and ai in wide.columns:
                    wide[f"{diff_col}_imputed"] = ((wide[hi] == 1) | (wide[ai] == 1)).astype("int8")
                    out_cols.append(f"{diff_col}_imputed")

    # Minimal observability
    if ADV_LOG_SUMMARY:
        present_cols = [c for c in out_cols if c.startswith(("home_adv_", "away_adv_"))]
        present_ratio = (wide[present_cols].notna().mean().mean() * 100.0) if present_cols else 0.0

        try:
            families = {
                "per_play":  [n for n in base_names if any(n.endswith(k) for k in ADV_PER_PLAY_NAMES)],
                "per_drive": [n for n in base_names if any(n.endswith(k) for k in ADV_PER_DRIVE_NAMES)],
                "rates":     [n for n in base_names if any(n.endswith(k) for k in ADV_RATE_NAMES) or n.endswith("_pct")],
            }
        except Exception:
            families = {}

        src = "unknown"
        if "__adv_source__" in season_map.columns:
            src_counts = season_map["__adv_source__"].value_counts(dropna=False).to_dict()
            src = " / ".join(f"{k}:{v}" for k, v in src_counts.items())

        join_mode = "team_id" if can_join_by_id else ("team_norm" if can_join_by_norm else "none")
        logger.info(
            "ADVANCED[summary] games=%d | source=%s | join=%s | features=%d | present≈%.1f%% | fam=%s",
            len(wide), src, join_mode, len(base_names), present_ratio, {k: len(v) for k, v in families.items()}
        )

    return wide[["game_id"] + sorted(set(out_cols) - {"game_id"})].copy()
