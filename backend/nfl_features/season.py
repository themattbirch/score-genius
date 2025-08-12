# backend/nfl_features/season.py
from __future__ import annotations
from typing import Optional, Mapping, List, Dict
import logging
import time

import pandas as pd
import numpy as np

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()

def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _ensure_team_norm(df: pd.DataFrame, *, in_place: bool = True) -> pd.DataFrame:
    tgt = df if in_place else df.copy()
    if "team_norm" in tgt.columns:
        tgt["team_norm"] = tgt["team_norm"].astype(str).str.lower()
        return tgt
    for cand in ("team_name", "team", "abbr", "abbreviation"):
        if cand in tgt.columns:
            tgt["team_norm"] = tgt[cand].apply(normalize_team_name).astype(str).str.lower()
            return tgt
    if "team_id" in tgt.columns:
        # As a last resort, normalize the id (stringify) — still stable per team
        tgt["team_norm"] = tgt["team_id"].apply(normalize_team_name).astype(str).str.lower()
        return tgt
    tgt["team_norm"] = "unknown_team"
    return tgt

def _ensure_side_norms(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    for side in ("home", "away"):
        col = f"{side}_team_norm"
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower()
            continue
        # Try a few common upstream names
        for cand in (f"{side}_team_name", f"{side}_team", f"{side}_abbr", f"{side}_abbreviation"):
            if cand in out.columns:
                out[col] = out[cand].apply(normalize_team_name).astype(str).str.lower()
                break
        if col not in out.columns:
            out[col] = "unknown_team"
    return out

def _pick_season_stats(kwargs: Mapping[str, object]) -> Optional[pd.DataFrame]:
    candidates = (
        "season_stats_df", "season_stats", "seasonal_stats_df", "prev_season_stats_df",
        "nfl_season_stats_df", "all_season_stats", "nfl_all_season_stats",
        "mv_nfl_season_stats", "rpc_get_nfl_all_season_stats", "season_df",
    )
    for key in candidates:
        obj = kwargs.get(key) if isinstance(kwargs, dict) else None
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            logger.info("season: using season_stats source '%s' with shape %s", key, obj.shape)
            return obj
    if isinstance(kwargs, dict) and kwargs:
        snapshot = {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in kwargs.items()}
        logger.warning("season: no season_stats_df found. Available kwargs keys: %s", snapshot)
    return None

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map upstream season stats columns to internal names."""
    rename_map: Dict[str, str] = {
        # win pct
        "wins_all_percentage": "prev_season_win_pct",
        "win_pct": "prev_season_win_pct",
        "wins_percentage": "prev_season_win_pct",
        "wins_pct": "prev_season_win_pct",
        # srs-lite
        "srs_lite": "prev_season_srs_lite",
        "srs": "prev_season_srs_lite",
        "simple_rating": "prev_season_srs_lite",
        # (optionals not currently emitted but kept for future)
        "points_for_avg_all": "prev_points_for_avg_all",
        "points_against_avg_all": "prev_points_against_avg_all",
    }
    out = df.rename(columns=rename_map)
    return out

def _append_differentials(df: pd.DataFrame) -> pd.DataFrame:
    if {"home_prev_season_win_pct", "away_prev_season_win_pct"}.issubset(df.columns):
        df["prev_season_win_pct_diff"] = (
            df["home_prev_season_win_pct"] - df["away_prev_season_win_pct"]
        ).astype("float32")
    if {"home_prev_season_srs_lite", "away_prev_season_srs_lite"}.issubset(df.columns):
        df["prev_season_srs_lite_diff"] = (
            df["home_prev_season_srs_lite"] - df["away_prev_season_srs_lite"]
        ).astype("float32")
    return df

def _emit_defaults(g: pd.DataFrame, flag_imputations: bool) -> pd.DataFrame:
    out = g.copy()
    defaults = {
        "prev_season_win_pct": float(DEFAULTS.get("win_pct", 0.5)),
        "prev_season_srs_lite": float(DEFAULTS.get("srs_lite", 0.0)),
    }
    for side in ("home", "away"):
        for feat, dval in defaults.items():
            col = f"{side}_{feat}"
            out[col] = dval
            if flag_imputations:
                out[f"{col}_imputed"] = np.int8(1)
    out = _append_differentials(out)
    return out

def _league_avgs_per_season(df: pd.DataFrame, seasons: pd.Series) -> pd.DataFrame:
    """Compute per-season league averages for features used, covering requested seasons."""
    need = ["prev_season_win_pct", "prev_season_srs_lite"]
    have = [c for c in need if c in df.columns]
    if not have or "season" not in df.columns:
        # If not available, return empty → caller will use global defaults
        return pd.DataFrame(columns=["season"] + need)

    grp = (
        df.groupby("season", dropna=False)[have]
        .mean()
        .reset_index()
        .rename(columns={c: f"league_{c}" for c in have})
    )
    # Ensure all seasons requested are present
    uniq = pd.DataFrame({"season": pd.unique(seasons.dropna())})
    filled = uniq.merge(grp, on="season", how="left")
    return filled


# ----------------------------
# Public API
# ----------------------------

def transform(
    games: pd.DataFrame,
    *,
    season_stats_df: Optional[pd.DataFrame] = None,
    # Hint from engine: True when we *know* the previous season is absent (e.g., 2020 not in DB for 2021 games)
    known_missing_prev: bool = False,
    # optional/compatible knobs
    flag_imputations: bool = True,
    debug: bool = False,
    strict_inputs: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Attach prior-season team baselines to games (leakage-safe).

    Preferred key: (team_id, season-1). Fallback: (team_norm, season-1).
    Fallback values: per-season league averages, then global defaults from DEFAULTS.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    t0 = _now()
    if games is None or games.empty:
        logger.info("season: start – empty input")
        return pd.DataFrame()

    logger.info("season: start – input shape %s", games.shape)
    g = games.copy()

    # Ensure season exists and is numeric
    if "season" not in g.columns:
        msg = "season: 'season' column missing – applying league defaults"
        if strict_inputs:
            raise ValueError(msg)
        # downgrade to INFO if the engine hinted this is an expected gap
        (logger.info if known_missing_prev else logger.warning)(msg)
        out = _emit_defaults(g, flag_imputations)
        keep = _final_cols(out)
        logger.info("season: complete (defaults) in %.2f s – output shape %s", _now() - t0, out[keep].shape)
        return out[keep]

    g["season"] = _coerce_int(g["season"])
    g = _ensure_side_norms(g)

    # Keys we can use from games
    have_ids = all(k in g.columns for k in ("home_team_id", "away_team_id"))
    if have_ids:
        g["home_team_id"] = _coerce_int(g["home_team_id"])
        g["away_team_id"] = _coerce_int(g["away_team_id"])

    # Acquire priors table, accept aliases via kwargs if not passed directly
    if season_stats_df is None or (isinstance(season_stats_df, pd.DataFrame) and season_stats_df.empty):
        season_stats_df = _pick_season_stats(kwargs)

    # If still none/empty → defaults (quiet if known_missing_prev)
    if season_stats_df is None or season_stats_df.empty:
        msg = "season: no usable season_stats_df provided – applying league defaults"
        if strict_inputs:
            raise ValueError(msg)
        (logger.info if known_missing_prev else logger.warning)(msg)
        out = _emit_defaults(g, flag_imputations)
        keep = _final_cols(out)
        logger.info("season: complete (defaults) in %.2f s – output shape %s", _now() - t0, out[keep].shape)
        return out[keep]

    # ----------------------------
    # Prepare lookup (priors)
    # ----------------------------
    look = season_stats_df.copy()
    if "season" not in look.columns:
        msg = "season: season_stats_df missing 'season' column – applying league defaults"
        if strict_inputs:
            raise ValueError(msg)
        (logger.info if known_missing_prev else logger.warning)(msg)
        out = _emit_defaults(g, flag_imputations)
        keep = _final_cols(out)
        logger.info("season: complete (lookup missing 'season') in %.2f s – output shape %s", _now() - t0, out[keep].shape)
        return out[keep]

    # Coerce keys, normalize names, and map columns
    if "team_id" in look.columns:
        look["team_id"] = _coerce_int(look["team_id"])
    look["season"] = _coerce_int(look["season"])
    look = _ensure_team_norm(look, in_place=True)
    look = _map_columns(look)

    # Keep freshest per (team, season) if timestamp exists
    if "updated_at" in look.columns:
        look = look.sort_values("updated_at", ascending=False, kind="mergesort")
    look = look.drop_duplicates(subset=["team_norm", "season"], keep="first")
    if "team_id" in look.columns:
        look = look.sort_values(by=["team_id", "season"], kind="mergesort") \
                   .drop_duplicates(subset=["team_id", "season"], keep="last")

    # Feature columns we expect to emit
    feats = ["prev_season_win_pct", "prev_season_srs_lite"]

    # Coverage diagnostics (helpful for auditing)
    required_prev = sorted(pd.Series(g["season"] - 1).dropna().astype(int).unique().tolist())
    have_prev = sorted(pd.Series(look["season"]).dropna().astype(int).unique().tolist())
    missing_prev = [s for s in required_prev if s not in have_prev]
    if missing_prev:
        msg = f"season: missing previous seasons {missing_prev} in priors – league averages/defaults will be used"
        (logger.info if known_missing_prev else logger.warning)(msg)

    # Compute league averages per season for smarter fallback
    league_avgs = _league_avgs_per_season(look, pd.Series(required_prev))

    # ----------------------------
    # Attach priors: prefer team_id, else team_norm
    # ----------------------------
    g["lookup_season"] = g["season"] - 1

    def _attach(side: str) -> pd.DataFrame:
        if have_ids and "team_id" in look.columns:
            logger.debug("season: joining by team_id for %s", side)
            key_col = f"{side}_team_id"
            join_left = g[["game_id", "lookup_season", key_col]].rename(columns={key_col: "team_id"})
            join = join_left.merge(
                look[["team_id", "season"] + feats],
                left_on=["team_id", "lookup_season"],
                right_on=["team_id", "season"],
                how="left",
            )
        else:
            logger.debug("season: joining by team_norm for %s", side)
            key_col = f"{side}_team_norm"
            join_left = g[["game_id", "lookup_season", key_col]].rename(columns={key_col: "team_norm"})
            join = join_left.merge(
                look[["team_norm", "season"] + feats],
                left_on=["team_norm", "lookup_season"],
                right_on=["team_norm", "season"],
                how="left",
            )

        use = ["game_id"] + feats
        out = join[use].copy()
        out.columns = ["game_id"] + [f"{side}_{c}" for c in feats]
        return out

    home = _attach("home")
    away = _attach("away")

    out = g.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # ----------------------------
    # Imputation & fallback fills
    # ----------------------------
    # Merge league averages for missing team-season priors
    if not league_avgs.empty:
        out = out.merge(
            league_avgs.rename(columns={
                "season": "lookup_season",
                "league_prev_season_win_pct": "league_prev_season_win_pct",
                "league_prev_season_srs_lite": "league_prev_season_srs_lite",
            }),
            on="lookup_season",
            how="left",
        )

    for side in ("home", "away"):
        for feat, default_key in (("prev_season_win_pct", "win_pct"),
                                  ("prev_season_srs_lite", "srs_lite")):
            col = f"{side}_{feat}"
            # Imputed flag BEFORE filling
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna().astype("int8")
            # Fill: per-season league avg → global DEFAULT
            league_col = f"league_{feat}"
            if league_col in out.columns:
                out[col] = out[col].fillna(out[league_col])
            out[col] = out[col].fillna(float(DEFAULTS.get(default_key, 0.0))).astype("float32")

    # Diffs & cleanup
    out = _append_differentials(out)
    out.drop(columns=["lookup_season"] + [c for c in out.columns if c.startswith("league_prev_season_")],
             errors="ignore", inplace=True)

    # Lightweight stats for quick eyeballing
    try:
        for k in ("home_prev_season_win_pct", "away_prev_season_win_pct"):
            if k in out.columns:
                s = out[k]
                logger.debug("season: %s stats – min=%.3f mean=%.3f max=%.3f (n=%d)",
                             k, float(s.min()), float(s.mean()), float(s.max()), int(s.notna().sum()))
    except Exception:
        pass

    # Return only engine-expected columns
    keep = _final_cols(out)
    logger.info("season: complete in %.2f s – output shape %s", _now() - t0, out[keep].shape)
    return out[keep]

# ----------------------------
# Output column selector
# ----------------------------

def _final_cols(out: pd.DataFrame) -> List[str]:
    passthru = [c for c in (
        "game_id",
        "home_score", "away_score",
        "game_date", "game_time",
        "season",
        "home_team_norm", "away_team_norm",
        "kickoff_ts",
    ) if c in out.columns]

    new_cols = [
        "home_prev_season_win_pct", "away_prev_season_win_pct",
        "home_prev_season_srs_lite", "away_prev_season_srs_lite",
        "prev_season_win_pct_diff", "prev_season_srs_lite_diff",
    ]

    imputed_cols = [c for c in (
        "home_prev_season_win_pct_imputed", "away_prev_season_win_pct_imputed",
        "home_prev_season_srs_lite_imputed", "away_prev_season_srs_lite_imputed",
    ) if c in out.columns]

    cols = list(dict.fromkeys(passthru + new_cols + imputed_cols))  # preserve order, dedupe
    return [c for c in cols if c in out.columns]
