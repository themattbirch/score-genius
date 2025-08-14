# backend/nfl_features/advanced.py
"""
Advanced efficiency features (leakage-safe) for NFL games.

Prefers prior-season aggregates (advanced_stats_df); otherwise computes from
team-game box scores (historical_team_stats_df), aggregates to season, and
optionally blends with current-season rolling (leak-safe; up to game_date-1).

Key behaviors:
- Joins by PRIOR season (season - 1) for priors.
- Optional blend with current-season rolling means (windowed), weighted by
  number of prior games this season.
- Joins by numeric team_id when available; otherwise falls back to normalized
  team name (team_norm).
- Computes advanced pace/efficiency metrics in fallback:
    pace:    seconds_per_play, plays_per_drive
    eff:     points_per_drive, yards_per_play, rush_yards_per_rush,
             turnover_rate_per_play, points_per_100_yards, pass_rate
    situational: 3rd/4th down %, red zone %
- Surfaces home_/away_ values, adv_*_diff (computed AFTER fills), total_adv_*,
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
ADV_LOG_SUMMARY = True            # print one compact coverage line per transform()
ALLOW_SAME_SEASON_FALLBACK = True # if prev season missing, optionally use same-season with shrink
SAME_SEASON_SHRINK = 0.70         # shrink factor when same-season is used as proxy for prev-season

# families used for normalization / engineered totals
ADV_RATE_NAMES = {
    "third_down_pct", "fourth_down_pct", "red_zone_pct", "pythagorean_win_pct", "pass_rate",
}
ADV_PER_PLAY_NAMES = {
    "yards_per_play", "turnover_rate_per_play",
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


def _ensure_numeric(s: pd.Series) -> pd.Series:
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
        # if column looks like a rate (or has '_pct') and median > 1, assume percentage and scale down
        med = float(v.median()) if not np.isnan(v.median()) else np.nan
        if ("_pct" in lc or any(name in lc for name in ADV_RATE_NAMES)) and isinstance(med, float) and not np.isnan(med) and med > 1.1:
            v = v / 100.0
        season_map[c] = v.clip(0.0, 1.0).astype("float32") if ("_pct" in lc or any(name in lc for name in ADV_RATE_NAMES)) else v.astype("float32")
    return season_map


def _cap_for(col: str) -> Tuple[float, float]:
    for k, rng in WINSOR_CAPS.items():
        if k != "_default" and col.endswith(k):
            return rng
    return WINSOR_CAPS["_default"]


def _dedupe_by(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Deterministic de-duplication: keep the last row per key combination.
    This is used to enforce a single row per (season_current, key, game_date).
    """
    if not keys or any(k not in df.columns for k in keys):
        return df
    if not df.duplicated(subset=keys).any():
        return df
    before = len(df)
    out = df.sort_values(keys).drop_duplicates(subset=keys, keep="last")
    logger.warning("ADVANCED_FIX: De-duplicated left/right at %s: %d -> %d rows", keys, before, len(out))
    return out


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
        df["team_id"] = _ensure_numeric(df["team_id"])

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

    def _add(col_out: str, numer: pd.Series, denom: pd.Series, default_key: Optional[str] = None, *, scale: float = 1.0):
        default_val = float(DEFAULTS.get(default_key, 0.0)) if default_key else 0.0
        rate = safe_divide(numer, denom, default_val=default_val) * scale
        out[col_out] = rate.astype("float32")
        if flag_imputations:
            out[f"{col_out}_imputed"] = (denom.isna()) | (denom == 0) | numer.isna()

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
            adv["team_id"] = _ensure_numeric(adv["team_id"])
        if "team_name" in adv.columns:
            adv["team_norm"] = _normalize_team_series(adv["team_name"])
        if "team_norm" in adv.columns:
            adv["team_norm"] = adv["team_norm"].astype(str).str.lower()
        if "season" in adv.columns:
            adv["season"] = _ensure_numeric(adv["season"])

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
                grp["season"] = _ensure_numeric(grp["season"])
                if "team_id" in grp.columns:
                    grp["team_id"] = _ensure_numeric(grp["team_id"])
                season_map = grp
                season_map["__adv_source__"] = "fallback"

    if season_map is None or season_map.empty:
        return None

    # normalize & clip rates
    season_map = _normalize_adv_map(season_map)
    return season_map


# ---------------------------------------------------------------------------
# Internal: build current-seaon rolling map (leak-safe, optional blend)
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
      ["team_id"/"team_norm", "season_current", "game_date"] + adv_*_roll + ["roll_count"]
    """
    if historical_team_stats_df is None or historical_team_stats_df.empty:
        return None

    per_game = compute_advanced_metrics(historical_team_stats_df, flag_imputations=True)
    if per_game.empty or "game_date" not in per_game.columns:
        return None

    per_game = per_game.copy()

    # --- Defensive de-duplication on the RIGHT (rolling source) ---
    key_cols_for_dedupe = [c for c in ["team_id", "team_norm", "game_date"] if c in per_game.columns]
    if len(key_cols_for_dedupe) > 1 and per_game.duplicated(subset=key_cols_for_dedupe).any():
        logger.warning("ADVANCED_FIX: Found and removed duplicate team-game date entries in RIGHT source data.")
        per_game = per_game.sort_values(key_cols_for_dedupe).drop_duplicates(subset=key_cols_for_dedupe, keep="last")

    per_game["game_date"] = pd.to_datetime(per_game["game_date"], errors="coerce", utc=False)
    per_game["season_current"] = per_game["game_date"].apply(_nfl_season_from_date).astype("Int64")

    key = "team_id" if "team_id" in per_game.columns and per_game["team_id"].notna().any() else "team_norm"
    if key not in per_game.columns:
        return None

    adv_cols = [c for c in per_game.columns if c.startswith("adv_") and not c.endswith("_imputed")]

    per_game = per_game.sort_values(["season_current", key, "game_date"]).reset_index(drop=True)

    # Rolling means (leak-safe): shift(1) to exclude current game
    grp = per_game.groupby([key, "season_current"], dropna=False, sort=False)
    for c in adv_cols:
        per_game[f"{c}_roll"] = grp[c].transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()).astype("float32")
    per_game["roll_count"] = grp.cumcount().astype("Int64")  # number of prior games in-season

    cols = [key, "season_current", "game_date"] + [f"{c}_roll" for c in adv_cols] + ["roll_count"]
    return per_game[cols].copy()


# ---------------------------------------------------------------------------
# Same-season fallback (shrunken) when prev-season missing
# ---------------------------------------------------------------------------

def _same_season_fallback(join_df: pd.DataFrame, key_col: str, adv_cols: List[str], season_map: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where prev-season join missed, try same-season, then shrink.
    `join_df` must contain [key_col, "season"] for target season and adv_cols (joined prev-season columns).
    """
    if season_map is None or season_map.empty or key_col not in join_df.columns or "season" not in join_df.columns:
        return join_df

    key = key_col
    if key.endswith("_team_id") and "team_id" in season_map.columns:
        same_map = season_map[["team_id", "season"] + adv_cols].rename(columns={"team_id": key})
    elif key.endswith("_team_norm") and "team_norm" in season_map.columns:
        same_map = season_map[["team_norm", "season"] + adv_cols].rename(columns={"team_norm": key})
    else:
        return join_df

    miss_mask = join_df[adv_cols].isna().all(axis=1)
    if not miss_mask.any():
        return join_df

    probe = join_df.loc[miss_mask, [key, "season"]]
    probe = probe.merge(
        same_map.rename(columns={"season": "__same_season__"}),
        left_on=[key, "season"], right_on=[key, "__same_season__"], how="left"
    )

    if not probe.empty:
        for c in adv_cols:
            vals = pd.to_numeric(probe[c], errors="coerce")
            join_df.loc[miss_mask, c] = (vals * SAME_SEASON_SHRINK).values

    return join_df


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
) -> pd.DataFrame:
    """
    Attach leakage-free advanced efficiency features to `games`.

    Preference order for priors:
      1) advanced_stats_df (season aggregates)
      2) compute from historical_team_stats_df -> aggregate to season

    Optionally blends current-season rolling (up to game_date-1) with priors.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    g = games.copy()

    # Require season & game_date to be sane
    if "season" not in g.columns:
        logger.warning("advanced.transform: games missing 'season'; cannot attach advanced.")
        return pd.DataFrame(columns=["game_id"])
    g["season"] = _ensure_numeric(g["season"])

    if "game_date" in g.columns:
        g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce", utc=False)
    else:
        # Without dates we cannot do leak-safe rolling; we can still attach priors
        enable_current_season_blend = False

    # Ensure team keys on games
    have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))
    have_norms = all(c in g.columns for c in ("home_team_norm", "away_team_norm"))

    if not have_ids and historical_df is not None and not historical_df.empty:
        if all(c in historical_df.columns for c in ["game_id", "home_team_id", "away_team_id"]):
            id_map = historical_df[["game_id", "home_team_id", "away_team_id"]].drop_duplicates("game_id")
            g = g.merge(id_map, on="game_id", how="left")
            have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))

    if have_ids:
        g["home_team_id"] = _ensure_numeric(g["home_team_id"])
        g["away_team_id"] = _ensure_numeric(g["away_team_id"])

    if not have_ids and not have_norms:
        # derive normalized names on games
        for hn, an in [("home_team_name", "away_team_name"), ("home_team", "away_team")]:
            if hn in g.columns and an in g.columns:
                g["home_team_norm"] = _normalize_team_series(g[hn])
                g["away_team_norm"] = _normalize_team_series(g[an])
                have_norms = True
                break

    # ------------------------------------------------------------------
    # 1) Build prior-season map
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

    # Prepare prev-season frames
    g_home = g[["game_id", "season"]].copy()
    g_home["prev_season"] = (g_home["season"] - 1).astype("Int64")
    g_away = g_home.copy()

    if can_join_by_id:
        logger.debug("advanced: joining priors by team_id & prev_season")
        g_home = g_home.join(g[["game_id", "home_team_id"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_id"]].set_index("game_id"), on="game_id")

        map_home = season_map.rename(columns={"team_id": "home_team_id"})
        map_away = season_map.rename(columns={"team_id": "away_team_id"})

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
        g["home_team_norm"] = _normalize_team_series(g["home_team_norm"])
        g["away_team_norm"] = _normalize_team_series(g["away_team_norm"])

        g_home = g_home.join(g[["game_id", "home_team_norm"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_norm"]].set_index("game_id"), on="game_id")

        map_home = season_map.rename(columns={"team_norm": "home_team_norm"})
        map_away = season_map.rename(columns={"team_norm": "away_team_norm"})

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

    # If allowed, try same-season fallback for rows where prev-season was missing
    if ALLOW_SAME_SEASON_FALLBACK and adv_cols_base:
        if can_join_by_id:
            home_join = _same_season_fallback(home_join, "home_team_id", adv_cols_base, season_map)
            away_join = _same_season_fallback(away_join, "away_team_id", adv_cols_base, season_map)
        elif can_join_by_norm:
            home_join = _same_season_fallback(home_join, "home_team_norm", adv_cols_base, season_map)
            away_join = _same_season_fallback(away_join, "away_team_norm", adv_cols_base, season_map)

    # Build prior columns with explicit prefixes
    home_prior = home_join[["game_id"] + adv_cols_base].copy()
    home_prior.columns = ["game_id"] + [f"home_prior_{c}" for c in adv_cols_base]

    away_prior = away_join[["game_id"] + adv_cols_base].copy()
    away_prior.columns = ["game_id"] + [f"away_prior_{c}" for c in adv_cols_base]

    wide = home_prior.merge(away_prior, on="game_id", how="outer")

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
            elif "team_norm" in roll_map.columns and "home_team_norm" in g.columns and "away_team_norm" in g.columns:
                key_kind = "team_norm"

    def _merge_roll(side: str) -> Optional[pd.DataFrame]:
        """
        Merge last known rolling values up to the game_date for given side.
        Ensures:
        - dtype parity for key
        - no duplicates at ["season_current", key_kind, "game_date"]
        - global/group-wise sort invariants required by merge_asof
        - rows with null join keys/dates are excluded from the rolling merge
        """
        if roll_map is None or key_kind is None:
            return None

        tmp = g[["game_id", "game_date", side, "season"]].copy()
        tmp = tmp.rename(columns={side: key_kind})
        tmp["season_current"] = tmp["season"].astype("Int64")

        # Key dtype parity
        if key_kind == "team_id":
            tmp[key_kind] = _ensure_numeric(tmp[key_kind])
            r = roll_map.copy()
            r[key_kind] = _ensure_numeric(r[key_kind])
        else:  # key_kind == "team_norm"
            tmp[key_kind] = _normalize_team_series(tmp[key_kind].astype(str))
            r = roll_map.copy()
            r[key_kind] = r[key_kind].astype(str).str.lower()

        # Dates tz-naive
        tmp["game_date"] = pd.to_datetime(tmp["game_date"], errors="coerce", utc=False)
        r["game_date"]   = pd.to_datetime(r["game_date"], errors="coerce", utc=False)

        # -----------------------------
        # NULL-KEY / NULL-DATE FILTERS
        # -----------------------------
        left_nulls = {
            "season_current": int(tmp["season_current"].isna().sum()),
            key_kind:         int(tmp[key_kind].isna().sum()),
            "game_date":      int(tmp["game_date"].isna().sum()),
        }
        right_nulls = {
            "season_current": int(r["season_current"].isna().sum()) if "season_current" in r.columns else 0,
            key_kind:         int(r[key_kind].isna().sum()),
            "game_date":      int(r["game_date"].isna().sum()),
        }
        logger.debug("ADVANCED_DIAG: LEFT nulls %s | RIGHT nulls %s", left_nulls, right_nulls)

        left_len_before  = len(tmp)
        right_len_before = len(r)

        # Drop rows that cannot participate in merge_asof groups
        tmp = tmp[tmp["season_current"].notna() & tmp[key_kind].notna() & tmp["game_date"].notna()].copy()
        r   = r[  r["season_current"].notna() &   r[key_kind].notna() &   r["game_date"].notna()].copy()

        logger.debug(
            "ADVANCED_DIAG: Filtered null keys/dates | LEFT %d→%d | RIGHT %d→%d",
            left_len_before, len(tmp), right_len_before, len(r)
        )

        # ---------------------------------
        # DE-DUPLICATE AT MERGE GRANULARITY
        # ---------------------------------
        key_tuple = ["season_current", key_kind, "game_date"]
        dup_l = int(tmp.duplicated(key_tuple).sum())
        dup_r = int(r.duplicated(key_tuple).sum())
        if dup_l or dup_r:
            logger.warning("ADVANCED_FIX: pre-dedupe duplicates | LEFT=%d RIGHT=%d at %s", dup_l, dup_r, key_tuple)
        tmp = _dedupe_by(tmp, key_tuple)
        r   = _dedupe_by(r,   key_tuple)

        # --------------
        # GLOBAL SORTING
        # --------------
        sort_keys = ["game_date", "season_current", key_kind]
        tmp = tmp.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        r   = r.sort_values(sort_keys,   kind="mergesort").reset_index(drop=True)

        # explicit global monotonic check on the 'on' key to mirror pandas' expectation
        if not tmp["game_date"].is_monotonic_increasing:
            # Defensive re-sort in case upstream reordering intervened
            tmp = tmp.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        if not r["game_date"].is_monotonic_increasing:
            r = r.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


        # ---------------------------------------------
        # PER-GROUP MONOTONICITY (HELPFUL ERROR DUMPS)
        # ---------------------------------------------
        def _first_bad_group(df: pd.DataFrame) -> Optional[tuple]:
            try:
                monotone = df.groupby(["season_current", key_kind], dropna=False)["game_date"] \
                            .apply(lambda s: s.is_monotonic_increasing)
                bad = monotone[~monotone]
                return tuple(bad.index[0]) if len(bad) else None
            except Exception:
                return None

        logger.info("ADVANCED_DIAG: Preparing for merge_asof for side '%s' | key=%s", side, key_kind)
        logger.debug("ADVANCED_DIAG: dtypes tmp[%s]=%s r[%s]=%s", key_kind, tmp[key_kind].dtype, key_kind, r[key_kind].dtype)

        if not _verify_merge_asof_sort(tmp, on="game_date", by=["season_current", key_kind]):
            bad = _first_bad_group(tmp)
            if bad:
                s_bad, k_bad = bad
                sample = tmp[(tmp["season_current"] == s_bad) & (tmp[key_kind] == k_bad)] \
                        [["game_id", "season_current", key_kind, "game_date"]].head(20)
                logger.error(
                    "ADVANCED_ERROR: LEFT not monotonic for season=%s key=%s; sample:\n%s",
                    s_bad, k_bad, sample.to_string(index=False)
                )
            raise ValueError("left keys must be sorted")

        if not _verify_merge_asof_sort(r, on="game_date", by=["season_current", key_kind]):
            bad = _first_bad_group(r)
            if bad:
                s_bad, k_bad = bad
                sample = r[(r["season_current"] == s_bad) & (r[key_kind] == k_bad)] \
                        [["game_date", "season_current", key_kind]].head(20)
                logger.error(
                    "ADVANCED_ERROR: RIGHT not monotonic for season=%s key=%s; sample:\n%s",
                    s_bad, k_bad, sample.to_string(index=False)
                )
            raise ValueError("right keys must be sorted")

        out = pd.merge_asof(
            left=tmp,
            right=r,
            by=["season_current", key_kind],
            left_on="game_date",
            right_on="game_date",
            direction="backward",
            allow_exact_matches=True,
        )

        roll_cols = [c for c in out.columns if c.endswith("_roll")]
        if "roll_count" in out.columns:
            roll_cols.append("roll_count")
        keep = ["game_id"] + roll_cols
        return out[keep].copy()

    # Select keys for sides
    left_key  = "home_team_id" if (key_kind == "team_id") else ("home_team_norm" if key_kind == "team_norm" else None)
    right_key = "away_team_id" if (key_kind == "team_id") else ("away_team_norm" if key_kind == "team_norm" else None)

    home_roll = _merge_roll(left_key)  if left_key  else None
    away_roll = _merge_roll(right_key) if right_key else None

    # ------------------------------------------------------------------
    # 3) Produce blended home_/away_ values
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
        w = np.minimum(np.maximum(n_games.astype(float) / float(blend_k_games), 0.0), 1.0)
        both = prior.notna() & roll.notna()
        only_prior = prior.notna() & ~roll.notna()
        only_roll = ~prior.notna() & roll.notna()

        blended = pd.Series(np.nan, index=prior.index, dtype="float32")
        blended.loc[both] = (1.0 - w[both]) * prior[both].astype(float) + w[both] * roll[both].astype(float)
        blended.loc[only_prior] = prior[only_prior].astype(float)
        blended.loc[only_roll] = roll[only_roll].astype(float)

        imputed = (~prior.notna()) | (~roll.notna())  # if either source missing, flag as imputed
        return blended.astype("float32"), imputed.astype("int8")

    for side in ("home", "away"):
        df_roll = home_roll if side == "home" else away_roll
        n_games = home_roll_n if side == "home" else away_roll_n

        for metric in base_metrics:
            prior = _get_prior(side, metric)
            roll = _get_roll(df_roll, metric) if enable_current_season_blend else pd.Series(np.nan, index=prior.index)
            if enable_current_season_blend:
                blended, imp = _blend_series(prior, roll, n_games)
            else:
                blended = prior.astype("float32")
                imp = (~prior.notna()).astype("int8")

            wide[f"{side}_adv_{metric}"] = blended
            if flag_imputations:
                wide[f"{side}_adv_{metric}_imputed"] = imp

    # ------------------------------------------------------------------
    # 4) Finalize, fill, then compute diffs/totals (diffs AFTER fill!)
    # ------------------------------------------------------------------
    value_cols = [c for c in wide.columns if c.startswith(("home_adv_", "away_adv_")) and not c.endswith("_imputed")]

    # Winsorize to control outliers
    for c in value_cols:
        lo, hi = _cap_for(c)
        wide[c] = pd.to_numeric(wide[c], errors="coerce").clip(lower=lo, upper=hi)

    # Fill NaNs in side values (post-cap) to 0.0 for stable diffs/totals
    for c in value_cols:
        wide[c] = wide[c].astype("float32").fillna(0.0)

    # Now compute diffs and totals AFTER fills; attach diff_imputed if either side was imputed
    out_cols: List[str] = ["game_id"]
    for metric in base_metrics:
        h = f"home_adv_{metric}"
        a = f"away_adv_{metric}"
        if h in wide.columns and a in wide.columns:
            diff_col = f"adv_{metric}_diff"
            tot_col  = f"total_adv_{metric}"

            wide[diff_col] = (wide[h] - wide[a]).astype("float32")
            wide[tot_col]  = (wide[h] + wide[a]).astype("float32")

            out_cols.extend([h, a, diff_col, tot_col])

            if flag_imputations:
                hi = f"{h}_imputed"
                ai = f"{a}_imputed"
                if hi in wide.columns and ai in wide.columns:
                    wide[f"{diff_col}_imputed"] = ((wide[hi] == 1) | (wide[ai] == 1)).astype("int8")
                    out_cols.append(f"{diff_col}_imputed")

    # Minimal observability
    present_cols = [c for c in out_cols if c.startswith(("home_adv_", "away_adv_"))]
    present_ratio = (wide[present_cols].notna().mean().mean() * 100.0) if present_cols else 0.0
    logger.debug(
        "advanced: features attached | games=%d | metrics=%d | side_value_cols=%d | present≈%.1f%%",
        len(wide), len(set(base_metrics)), len(value_cols), present_ratio
    )

    # Compact coverage summary (first ~8 metrics) + imputation %
    if ADV_LOG_SUMMARY:
        base_names = sorted({c.replace("home_adv_", "").replace("away_adv_", "") for c in value_cols})
        nonnull = {}
        for b in base_names[:8]:
            h, a = f"home_adv_{b}", f"away_adv_{b}"
            cols = [c for c in (h, a) if c in wide.columns]
            cnt = (wide[cols].notna().any(axis=1)).sum() if cols else 0
            nonnull[b] = int(cnt)

        imp_cols = [f"{c}_imputed" for c in value_cols if f"{c}_imputed" in wide.columns]
        imp_ratio = (float(wide[imp_cols].mean().mean()) * 100.0) if imp_cols else None

        src = "unknown"
        if "__adv_source__" in season_map.columns:
            src_counts = season_map["__adv_source__"].value_counts(dropna=False).to_dict()
            src = " / ".join(f"{k}:{v}" for k, v in src_counts.items())

        join_mode = "team_id" if can_join_by_id else ("team_norm" if can_join_by_norm else "none")
        logger.info(
            "ADVANCED[summary] rows=%d | source=%s | join=%s | nonnull_by_metric=%s%s",
            len(wide), src, join_mode, nonnull,
            (f" | imputed≈{imp_ratio:.1f}%" if imp_ratio is not None else "")
        )

    return wide[out_cols].copy()
