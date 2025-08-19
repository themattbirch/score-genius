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
    # Strict nullable integer for all numeric join keys
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _as_obj_str(s: pd.Series) -> pd.Series:
    # Lowercased Python str (object dtype), matching the known-good script
    return s.astype(str).str.lower()

def _ensure_team_norm(df: pd.DataFrame, *, in_place: bool = True) -> pd.DataFrame:
    tgt = df if in_place else df.copy()
    if "team_norm" in tgt.columns:
        tgt["team_norm"] = _as_obj_str(tgt["team_norm"])
        return tgt
    for cand in ("team_name", "team", "abbr", "abbreviation"):
        if cand in tgt.columns:
            tgt["team_norm"] = _as_obj_str(tgt[cand].apply(normalize_team_name))
            return tgt
    if "team_id" in tgt.columns:
        tgt["team_norm"] = _as_obj_str(tgt["team_id"].apply(normalize_team_name))
        return tgt
    tgt["team_norm"] = "unknown_team"
    return tgt

def _ensure_side_norms(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    for side in ("home", "away"):
        col = f"{side}_team_norm"
        if col in out.columns:
            out[col] = _as_obj_str(out[col])
            continue
        for cand in (f"{side}_team_name", f"{side}_team", f"{side}_abbr", f"{side}_abbreviation"):
            if cand in out.columns:
                out[col] = _as_obj_str(out[cand].apply(normalize_team_name))
                break
        if col not in out.columns:
            out[col] = "unknown_team"
    return out

def _pick_season_stats(kwargs: Mapping[str, object]) -> Optional[pd.DataFrame]:
    candidates = (
        "season_stats_df", "season_stats", "seasonal_stats_df", "prev_season_stats_df",
        "nfl_season_stats_df", "all_season_stats", "nfl_all_season_stats",
        "mv_nfl_season_stats", "rpc_get_nfl_all_season_stats", "season_df",
        # NEW: accept test-provided historical team stats as the priors source
        "historical_team_stats_df", "historical_team_stats",
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
    """Map upstream season stats columns to internal names (minimal, working set)."""
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
    }
    return df.rename(columns=rename_map)

def _append_differentials(df: pd.DataFrame) -> pd.DataFrame:
    # Always create the two diffs (trainer expects them)
    if {"home_prev_season_win_pct", "away_prev_season_win_pct"}.issubset(df.columns):
        df["prev_season_win_pct_diff"] = (
            df["home_prev_season_win_pct"] - df["away_prev_season_win_pct"]
        ).astype("float32")
    else:
        df["prev_season_win_pct_diff"] = pd.Series(pd.NA, index=df.index, dtype="float32")

    if {"home_prev_season_srs_lite", "away_prev_season_srs_lite"}.issubset(df.columns):
        df["prev_season_srs_lite_diff"] = (
            df["home_prev_season_srs_lite"] - df["away_prev_season_srs_lite"]
        ).astype("float32")
    else:
        df["prev_season_srs_lite_diff"] = pd.Series(pd.NA, index=df.index, dtype="float32")

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
            out[col] = np.float32(dval)
            if flag_imputations:
                out[f"{col}_imputed"] = np.int8(1)
    return _append_differentials(out)

def _league_avgs_per_season(df: pd.DataFrame, seasons: pd.Series) -> pd.DataFrame:
    """
    Compute per-season league averages for features used, covering requested seasons.
    Keep 'season' as Int64 to avoid float/object mismatches.
    """
    need = ["prev_season_win_pct", "prev_season_srs_lite"]
    have = [c for c in need if c in df.columns]
    if not have or "season" not in df.columns:
        return pd.DataFrame(columns=["season"] + need)

    tmp = df.copy()
    tmp["season"] = _coerce_int(tmp["season"])
    grp = (
        tmp.groupby("season", dropna=False)[have]
        .mean()
        .reset_index()
        .rename(columns={c: f"league_{c}" for c in have})
    )

    uniq = pd.DataFrame({"season": pd.unique(_coerce_int(seasons).dropna())})
    uniq["season"] = _coerce_int(uniq["season"])
    filled = uniq.merge(grp, on="season", how="left")
    return filled


# ----------------------------
# Public API
# ----------------------------

def transform(
    games: pd.DataFrame,
    *,
    season_stats_df: Optional[pd.DataFrame] = None,
    known_missing_prev: bool = False,
    flag_imputations: bool = True,
    debug: bool = False,
    strict_inputs: bool = False,
    prefer_league_avg_if_missing: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Attach prior-season team baselines to games (leakage-safe).

    Preferred key: (team_id, season-1). Fallback: (team_norm, season-1).
    Fallback values: per-season league averages, then global defaults.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    t0 = _now()
    if games is None or games.empty:
        logger.info("season: start – empty input")
        return pd.DataFrame()

    logger.info("season: start – input shape %s", games.shape)
    g = games.copy()

    # Ensure season exists and is **Int64**
    if "season" not in g.columns:
        msg = "season: 'season' column missing – applying league defaults"
        if strict_inputs:
            raise ValueError(msg)
        (logger.info if known_missing_prev else logger.warning)(msg)
        out = _emit_defaults(g, flag_imputations)
        keep = _final_cols(out)
        logger.info("season: complete (defaults) in %.2f s – output shape %s", _now() - t0, out[keep].shape)
        return out[keep]

    g["season"] = _coerce_int(g["season"])
    g = _ensure_side_norms(g)

    # Team IDs (nullable Int64 if present)
    have_ids = all(k in g.columns for k in ("home_team_id", "away_team_id"))
    if have_ids:
        g["home_team_id"] = _coerce_int(g["home_team_id"])
        g["away_team_id"] = _coerce_int(g["away_team_id"])

    # Acquire priors (and accept aliases)
    if season_stats_df is None or (isinstance(season_stats_df, pd.DataFrame) and season_stats_df.empty):
        season_stats_df = _pick_season_stats(kwargs)

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

    # Match working version: Int64 keys, object strings for team_norm
    if "team_id" in look.columns:
        look["team_id"] = _coerce_int(look["team_id"])
    look["season"] = _coerce_int(look["season"])
    look = _ensure_team_norm(look, in_place=True)
    look = _map_columns(look)
    
    if "prev_season_win_pct" not in look.columns or look["prev_season_win_pct"].isna().all():
        # locate columns for wins / losses / ties / games played
        wins_col   = next((c for c in ("wins", "won") if c in look.columns), None)
        losses_col = next((c for c in ("losses", "lost") if c in look.columns), None)
        ties_col   = next((c for c in ("ties", "draws", "tied") if c in look.columns), None)
        gp_col     = next((c for c in ("games", "games_played", "played", "gp", "g") if c in look.columns), None)

        wins = pd.to_numeric(look[wins_col], errors="coerce") if wins_col else None

        if gp_col:
            games = pd.to_numeric(look[gp_col], errors="coerce")
        else:
            # compute GP from components if no explicit games column
            losses = pd.to_numeric(look[losses_col], errors="coerce") if losses_col else 0
            ties   = pd.to_numeric(look[ties_col],   errors="coerce") if ties_col   else 0
            # if wins is None, we can't compute a ratio anyway; produce NaN GP
            if wins is not None:
                games = wins.fillna(0) + (losses if isinstance(losses, pd.Series) else 0) + (ties if isinstance(ties, pd.Series) else 0)
            else:
                games = pd.Series(np.nan, index=look.index, dtype="float32")

        if wins is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = wins / games.replace(0, np.nan)
            look["prev_season_win_pct"] = ratio.astype("float32")


    # Freshest per (team, season) if timestamp exists
    if "updated_at" in look.columns:
        look = look.sort_values("updated_at", ascending=False, kind="mergesort")
    look = look.drop_duplicates(subset=["team_norm", "season"], keep="first")
    if "team_id" in look.columns:
        look = look.sort_values(by=["team_id", "season"], kind="mergesort") \
                   .drop_duplicates(subset=["team_id", "season"], keep="last")

    feats = ["prev_season_win_pct", "prev_season_srs_lite"]

    # Coverage diagnostics
    required_prev = sorted(pd.Series(g["season"] - 1).dropna().astype(int).unique().tolist())
    have_prev = sorted(pd.Series(look["season"]).dropna().astype(int).unique().tolist())
    missing_prev = [s for s in required_prev if s not in have_prev]
    if missing_prev:
        msg = f"season: missing previous seasons {missing_prev} in priors – league averages/defaults will be used"
        (logger.info if known_missing_prev else logger.warning)(msg)

    # League averages for requested seasons (kept Int64)
    league_avgs = _league_avgs_per_season(look, pd.Series(required_prev, dtype="Int64"))

    # ----------------------------
    # Attach priors: prefer team_id (Int64), else team_norm (object str)
    # ----------------------------
    g["lookup_season"] = _coerce_int(g["season"] - 1)

    def _attach(side: str) -> pd.DataFrame:
        if have_ids and "team_id" in look.columns:
            key_col = f"{side}_team_id"
            left = g[["game_id", "lookup_season", key_col]].rename(columns={key_col: "team_id"}).copy()
            left["team_id"] = _coerce_int(left["team_id"])
            left["lookup_season"] = _coerce_int(left["lookup_season"])

            right = look[["team_id", "season"] + feats].copy()
            right["team_id"] = _coerce_int(right["team_id"])
            right["season"] = _coerce_int(right["season"])

            join = left.merge(
                right,
                left_on=["team_id", "lookup_season"],
                right_on=["team_id", "season"],
                how="left",
            )
        else:
            key_col = f"{side}_team_norm"
            left = g[["game_id", "lookup_season", key_col]].rename(columns={key_col: "team_norm"}).copy()
            left["team_norm"] = _as_obj_str(left["team_norm"])
            left["lookup_season"] = _coerce_int(left["lookup_season"])

            right = look[["team_norm", "season"] + feats].copy()
            right["team_norm"] = _as_obj_str(right["team_norm"])
            right["season"] = _coerce_int(right["season"])

            join = left.merge(
                right,
                left_on=["team_norm", "lookup_season"],
                right_on=["team_norm", "season"],
                how="left",
            )

        use = ["game_id"] + feats
        out_side = join[use].copy()
        out_side.columns = ["game_id"] + [f"{side}_{c}" for c in feats]
        return out_side

    home = _attach("home")
    away = _attach("away")

    out = g.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # ----------------------------
    # Imputation & fallback fills
    # ----------------------------
    # Merge league averages (Int64 join key) if available
    if not league_avgs.empty:
        league_avgs = league_avgs.rename(columns={"season": "lookup_season"})
        league_avgs["lookup_season"] = _coerce_int(league_avgs["lookup_season"])
        out["lookup_season"] = _coerce_int(out["lookup_season"])
        out = out.merge(league_avgs, on="lookup_season", how="left")

    for side in ("home", "away"):
        for feat, default_key in (
            ("prev_season_win_pct", "win_pct"),
            ("prev_season_srs_lite", "srs_lite"),
        ):
            col = f"{side}_{feat}"

            # Track imputation before any fills (was the team prior missing?)
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna().astype("int8")

            # Fill strategy:
            # - If prefer_league_avg_if_missing and league averages are present,
            #   prefer league-season average before falling back to global default.
            # - Otherwise, go straight to global default.
            if prefer_league_avg_if_missing:
                league_col = f"league_{feat}"
                if league_col in out.columns:
                    out[col] = out[col].fillna(out[league_col])

            out[col] = out[col].fillna(float(DEFAULTS.get(default_key, 0.0))).astype("float32")

    # Diffs & cleanup
    out = _append_differentials(out)
    out.drop(
        columns=["lookup_season", "league_prev_season_win_pct", "league_prev_season_srs_lite"],
        errors="ignore",
        inplace=True,
    )


    # Quick stats (optional)
    try:
        for k in ("home_prev_season_win_pct", "away_prev_season_win_pct"):
            if k in out.columns and out[k].notna().any():
                s = out[k]
                logger.debug(
                    "season: %s stats – min=%.3f mean=%.3f max=%.3f (n=%d)",
                    k, float(s.min()), float(s.mean()), float(s.max()), int(s.notna().sum())
                )
    except Exception:
        pass

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

    cols = list(dict.fromkeys(passthru + new_cols + imputed_cols))
    return [c for c in cols if c in out.columns]
