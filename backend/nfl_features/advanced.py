# backend/nfl_features/advanced.py
"""
Advanced efficiency features (leakage-safe) for NFL games.

Prefers season-level aggregates (advanced_stats_df), otherwise computes from team-game
box scores (historical_team_stats_df), aggregates to season, then attaches to games.

Key behaviors:
- Joins by PRIOR season (season - 1).
- Uses numeric team_id when available; otherwise falls back to normalized team name.
- If games slice lacks team IDs, backfill from historical_df via game_id.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, safe_divide

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform", "compute_advanced_metrics"]


# ---------- helpers ----------

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
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val)
        if ":" in s:
            m, ss = s.split(":")
            return int(m) * 60 + int(ss)
        return float(s)
    except Exception:
        return np.nan


# ---------- per-game advanced metrics (used by fallback path) ----------

def compute_advanced_metrics(
    box_scores: pd.DataFrame,
    *,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    if box_scores is None or box_scores.empty:
        return pd.DataFrame()

    df = box_scores.copy()

    # identity
    if "team_id" in df.columns:
        df["team_id"] = _ensure_numeric(df["team_id"])
    team_norm_src = _first(df, ["team_norm", "team_name", "team", "team_abbr"])
    if team_norm_src:
        df["team_norm"] = df[team_norm_src].apply(normalize_team_name).astype(str).str.lower()

    id_cols = [c for c in ["game_id", "team_id", "team_norm"] if c in df.columns]
    out = df[id_cols].copy()

    def _add(col_out: str, numer: pd.Series, denom: pd.Series, default_key: Optional[str] = None, *, scale: float = 1.0):
        default_val = float(DEFAULTS.get(default_key, 0.0)) if default_key else 0.0
        rate = safe_divide(numer, denom, default_val=default_val) * scale
        out[col_out] = rate.astype("float32")
        if flag_imputations:
            out[f"{col_out}_imputed"] = (denom.isna()) | (denom == 0) | numer.isna()

    # synonyms
    plays_col = _first(df, ["plays_total", "total_plays", "offensive_plays", "plays"])
    yards_col = _first(df, ["yards_total", "total_yards", "net_yards", "yards"])
    points_col = _first(df, ["team_score", "points", "pts", "score"])
    drives_col = _first(df, ["total_drives", "drives", "offensive_drives", "num_drives"])
    pen_yards_col = _first(df, ["penalty_yards", "penalty_yds", "pen_yds"])
    penalties_col = _first(df, ["penalties", "penalties_total", "penalties_count"])
    rush_yards_col = _first(df, ["rushings_total", "rushing_yards", "rush_yards"])
    rush_att_col = _first(df, ["rushings_attempts", "rush_attempts", "rushes", "rushing_att"])
    third_made = _first(df, ["third_down_made", "third_down_conversions", "third_downs_made", "third_downs_conv"])
    third_att = _first(df, ["third_down_attempts", "third_downs", "third_downs_att"])
    fourth_made = _first(df, ["fourth_down_made", "fourth_down_conversions"])
    fourth_att = _first(df, ["fourth_down_attempts", "fourth_downs", "fourth_downs_att"])
    rz_made = _first(df, ["red_zone_made", "red_zone_td", "redzone_td", "rzt_td"])
    rz_att = _first(df, ["red_zone_att", "red_zone_trips", "redzone_att", "rzt_att"])
    turnovers_col = _first(df, ["turnovers_total", "turnovers", "to_total"])

    # metrics
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

    if pen_yards_col and penalties_col:
        _add("adv_yards_per_penalty", df[pen_yards_col], df[penalties_col], None)

    if points_col and yards_col:
        _add("adv_points_per_100_yards", df[points_col], df[yards_col], "points_for_avg", scale=100.0)

    adv_cols = [c for c in out.columns if c.startswith("adv_")]
    if not adv_cols:
        logger.warning("advanced.compute: no metrics computed. available_cols=%s", list(df.columns))
        return pd.DataFrame()

    keep_cols = [c for c in ["game_id", "team_id", "team_norm"] if c in out.columns]
    impute_cols = [c for c in out.columns if c.endswith("_imputed")]
    return out[keep_cols + adv_cols + impute_cols].copy()


# ---------- public API ----------

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,                # used to backfill team IDs when missing
    historical_team_stats_df: Optional[pd.DataFrame] = None,     # per-game team stats (fallback compute)
    advanced_stats_df: Optional[pd.DataFrame] = None,            # season aggregates (preferred)
    flag_imputations: bool = True,
    min_games_prior: int = 0,
) -> pd.DataFrame:
    """
    Attach leakage-free advanced efficiency features to `games`.

    Preference order:
      1) advanced_stats_df (season aggregates)
      2) compute from historical_team_stats_df -> aggregate to season
    """
    if games is None or games.empty:
        return pd.DataFrame()

    # Base columns and season dtype
    g = games.copy()
    if "season" not in g.columns:
        logger.warning("advanced.transform: games missing 'season'; cannot attach advanced.")
        return pd.DataFrame(columns=["game_id"])
    g["season"] = _ensure_numeric(g["season"])

    # Ensure we have team keys on games: try IDs, else norms; backfill from historical_df if needed
    have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))
    have_norms = all(c in g.columns for c in ("home_team_norm", "away_team_norm"))

    if not have_ids:
        # backfill from historical_df by game_id
        if historical_df is not None and not historical_df.empty:
            id_candidates = ["home_team_id", "away_team_id"]
            if all(c in historical_df.columns for c in id_candidates):
                id_map = historical_df[["game_id", "home_team_id", "away_team_id"]].drop_duplicates("game_id")
                g = g.merge(id_map, on="game_id", how="left", suffixes=("", ""))
                have_ids = all(c in g.columns for c in ("home_team_id", "away_team_id"))
        # normalize dtype if we got them
    if have_ids:
        g["home_team_id"] = _ensure_numeric(g["home_team_id"])
        g["away_team_id"] = _ensure_numeric(g["away_team_id"])

    # If still no IDs, try to get/construct team norms on games
    if not have_ids and not have_norms:
        name_pairs = [("home_team_name", "away_team_name"), ("home_team", "away_team")]
        for hn, an in name_pairs:
            if hn in g.columns and an in g.columns:
                g["home_team_norm"] = g[hn].apply(normalize_team_name).astype(str).str.lower()
                g["away_team_norm"] = g[an].apply(normalize_team_name).astype(str).str.lower()
                have_norms = True
                break

    # Build season_map from preferred source (advanced_stats_df), else from compute()
    season_map = None
    if advanced_stats_df is not None and not advanced_stats_df.empty:
        adv = advanced_stats_df.copy()

        # Coerce keys and build BOTH id + norm where possible
        if "team_id" in adv.columns:
            adv["team_id"] = _ensure_numeric(adv["team_id"])
        if "team_name" in adv.columns:
            adv["team_norm"] = adv["team_name"].apply(normalize_team_name).astype(str).str.lower()
        if "team_norm" in adv.columns and adv["team_norm"].isna().any():
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
        adv = adv.rename(columns=rename_map)

        if "avg_time_of_possession" in adv.columns:
            adv["adv_time_of_possession_seconds"] = adv["avg_time_of_possession"].apply(_to_seconds_maybe).astype("float32")

        adv_cols = [c for c in adv.columns if c.startswith("adv_")]
        if "season" in adv.columns and adv_cols:
            keep_keys = [k for k in ["team_id", "team_norm"] if k in adv.columns]
            season_map = adv[["season"] + keep_keys + adv_cols].copy()

    if season_map is None:
        # Fallback: compute per-game, then aggregate to season by team
        if historical_team_stats_df is not None and not historical_team_stats_df.empty:
            per_game = compute_advanced_metrics(historical_team_stats_df, flag_imputations=flag_imputations)
            if not per_game.empty:
                key_cols = [k for k in ["team_id", "team_norm"] if k in per_game.columns]
                # get season for per_game rows
                season_src = None
                if "season" in historical_team_stats_df.columns:
                    season_src = historical_team_stats_df[["game_id", "season"]].drop_duplicates()
                elif "game_date" in historical_team_stats_df.columns:
                    tmp = historical_team_stats_df[["game_id", "game_date"]].drop_duplicates()
                    tmp["season"] = pd.to_datetime(tmp["game_date"], errors="coerce").dt.year
                    season_src = tmp[["game_id", "season"]]

                if "game_id" in per_game.columns and season_src is not None and not season_src.empty:
                    per_game = per_game.merge(season_src, on="game_id", how="left")

                if key_cols and "season" in per_game.columns:
                    adv_cols = [c for c in per_game.columns if c.startswith("adv_") and not c.endswith("_imputed")]
                    grp = per_game.groupby(key_cols + ["season"], dropna=False)[adv_cols].mean().reset_index()
                    grp["season"] = _ensure_numeric(grp["season"])
                    if "team_id" in grp.columns:
                        grp["team_id"] = _ensure_numeric(grp["team_id"])
                    season_map = grp

    if season_map is None or season_map.empty:
        logger.warning("advanced: no usable source after compute (checked advanced_stats_df, historical_team_stats_df).")
        return pd.DataFrame(columns=["game_id"])

    # Decide join mode
    adv_cols = [c for c in season_map.columns if c.startswith("adv_")]
    can_join_by_id = have_ids and "team_id" in season_map.columns and season_map["team_id"].notna().any()
    can_join_by_norm = ("home_team_norm" in g.columns and "away_team_norm" in g.columns and "team_norm" in season_map.columns)

    # Prepare game-side frames with prev_season
    g_home = g[["game_id", "season"]].copy()
    g_home["prev_season"] = g_home["season"] - 1

    g_away = g[["game_id", "season"]].copy()
    g_away["prev_season"] = g_away["season"] - 1

    if can_join_by_id:
        logger.debug("advanced: joining by team_id & prev_season")
        g_home = g_home.join(g[["game_id", "home_team_id"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_id"]].set_index("game_id"), on="game_id")

        map_home = season_map.rename(columns={"team_id": "home_team_id"})
        map_away = season_map.rename(columns={"team_id": "away_team_id"})

        home_join = g_home.merge(map_home, left_on=["home_team_id", "prev_season"], right_on=["home_team_id", "season"], how="left")
        away_join = g_away.merge(map_away, left_on=["away_team_id", "prev_season"], right_on=["away_team_id", "season"], how="left")

    elif can_join_by_norm:
        logger.debug("advanced: joining by team_norm & prev_season")
        # ensure norms are normalized strings
        g["home_team_norm"] = g["home_team_norm"].apply(normalize_team_name).astype(str).str.lower()
        g["away_team_norm"] = g["away_team_norm"].apply(normalize_team_name).astype(str).str.lower()

        g_home = g_home.join(g[["game_id", "home_team_norm"]].set_index("game_id"), on="game_id")
        g_away = g_away.join(g[["game_id", "away_team_norm"]].set_index("game_id"), on="game_id")

        map_home = season_map.rename(columns={"team_norm": "home_team_norm"})
        map_away = season_map.rename(columns={"team_norm": "away_team_norm"})

        home_join = g_home.merge(map_home, left_on=["home_team_norm", "prev_season"], right_on=["home_team_norm", "season"], how="left")
        away_join = g_away.merge(map_away, left_on=["away_team_norm", "prev_season"], right_on=["away_team_norm", "season"], how="left")

    else:
        logger.warning("advanced: cannot attach – missing join keys. games_cols=%s season_map_keys=%s",
                       list(g.columns), [c for c in ["team_id", "team_norm"] if c in season_map.columns])
        return pd.DataFrame(columns=["game_id"])

    # Build output
    home_feats = home_join[["game_id"] + adv_cols].copy()
    home_feats.columns = ["game_id"] + [f"home_{c}" for c in adv_cols]

    away_feats = away_join[["game_id"] + adv_cols].copy()
    away_feats.columns = ["game_id"] + [f"away_{c}" for c in adv_cols]

    wide = home_feats.merge(away_feats, on="game_id", how="outer")

    # diffs
    for base in adv_cols:
        h, a = f"home_{base}", f"away_{base}"
        if h in wide.columns and a in wide.columns:
            wide[f"{base}_diff"] = (wide[h] - wide[a]).astype("float32")

    # imputation flags + fills
    value_cols = [c for c in wide.columns if c.startswith(("home_adv_", "away_adv_"))]
    if flag_imputations:
        for c in value_cols:
            wide[f"{c}_imputed"] = wide[c].isna().astype("int8")
    for c in value_cols:
        wide[c] = wide[c].astype("float32").fillna(0.0)

    return wide[["game_id"] + [c for c in wide.columns if c != "game_id"]].copy()
