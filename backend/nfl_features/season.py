# backend/nfl_features/season.py
from __future__ import annotations
from typing import Optional, Mapping
import logging
import time

import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


def _append_differentials(df: pd.DataFrame) -> pd.DataFrame:
    # Compute diffs only if the source columns exist to avoid KeyError
    if {"home_prev_season_win_pct", "away_prev_season_win_pct"}.issubset(df.columns):
        df["prev_season_win_pct_diff"] = (
            df["home_prev_season_win_pct"] - df["away_prev_season_win_pct"]
        )
    if {"home_prev_season_srs_lite", "away_prev_season_srs_lite"}.issubset(df.columns):
        df["prev_season_srs_lite_diff"] = (
            df["home_prev_season_srs_lite"] - df["away_prev_season_srs_lite"]
        )
    return df


def _pick_season_stats(kwargs: Mapping[str, object]) -> Optional[pd.DataFrame]:
    candidates = (
        "season_stats_df", "season_stats", "seasonal_stats_df", "prev_season_stats_df",
        "nfl_season_stats_df", "all_season_stats", "nfl_all_season_stats",
        "mv_nfl_season_stats", "rpc_get_nfl_all_season_stats",
        "season_df",
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


def _ensure_team_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "team_norm" in df.columns:
        df["team_norm"] = df["team_norm"].astype(str).str.lower()
        return df
    # try textual fields first
    for cand in ("team", "team_name", "abbr", "abbreviation"):
        if cand in df.columns:
            df["team_norm"] = df[cand].apply(normalize_team_name).astype(str).str.lower()
            return df
    # fall back to id
    if "team_id" in df.columns:
        df["team_norm"] = df["team_id"].apply(normalize_team_name).astype(str).str.lower()
        return df
    df["team_norm"] = "unknown_team"
    return df


def _ensure_side_team_norm(out: pd.DataFrame, side: str) -> pd.DataFrame:
    col = f"{side}_team_norm"
    if col in out.columns:
        out[col] = out[col].astype(str).str.lower()
        return out
    # try a few common upstream names
    for cand in (f"{side}_team", f"{side}_team_name", f"{side}_abbr", f"{side}_abbreviation", f"{side}_team_id"):
        if cand in out.columns:
            out[col] = out[cand].apply(normalize_team_name).astype(str).str.lower()
            return out
    # last resort
    out[col] = "unknown_team"
    return out


def transform(
    games: pd.DataFrame,
    *,
    season_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
    strict_inputs: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    start_ts = time.time()
    logger.info("season: start – input shape %s", games.shape)

    if games.empty:
        return games.copy()

    out = games.copy()

    # Ensure side team_norms exist & normalized
    out = _ensure_side_team_norm(out, "home")
    out = _ensure_side_team_norm(out, "away")

    # If not passed directly, accept aliases
    if season_stats_df is None or (isinstance(season_stats_df, pd.DataFrame) and season_stats_df.empty):
        season_stats_df = _pick_season_stats(kwargs)

    # Column mapping from your season stats source -> internal feature names
    feature_map = {
        "wins_all_percentage": "prev_season_win_pct",
        "srs_lite": "prev_season_srs_lite",
    }
    default_vals = {
        "prev_season_win_pct": DEFAULTS["win_pct"],
        "prev_season_srs_lite": DEFAULTS["srs_lite"],
    }

    # If season is missing, we can't join—fall back to defaults
    missing_season = "season" not in out.columns

    # === FAST PATH: no season stats or can't join -> fill defaults and return
    if season_stats_df is None or (isinstance(season_stats_df, pd.DataFrame) and season_stats_df.empty) or missing_season:
        msg = (
            "season: no usable season_stats_df provided – applying league defaults"
            if not missing_season else
            "season: 'season' column missing – applying league defaults"
        )
        if strict_inputs:
            raise ValueError(msg)
        logger.warning(msg)

        for side in ("home", "away"):
            for feat, dval in default_vals.items():
                col = f"{side}_{feat}"
                out[col] = dval
                if flag_imputations:
                    out[f"{col}_imputed"] = True

        out = _append_differentials(out)

        # Pass-through base columns so later modules have what they need
        passthru = [c for c in (
            "home_score", "away_score",     # labels
            "game_date", "game_time",
            "season",
            "home_team_norm", "away_team_norm",
            "kickoff_ts"                    # if your schedule carries a timestamp
        ) if c in out.columns]

        new_cols = [
            "home_prev_season_win_pct", "away_prev_season_win_pct",
            "home_prev_season_srs_lite", "away_prev_season_srs_lite",
            "prev_season_win_pct_diff", "prev_season_srs_lite_diff",
        ]
        if flag_imputations:
            new_cols += [
                "home_prev_season_win_pct_imputed", "away_prev_season_win_pct_imputed",
                "home_prev_season_srs_lite_imputed", "away_prev_season_srs_lite_imputed",
            ]

        final_cols = ["game_id"] + passthru + [c for c in new_cols if c in out.columns]
        final_cols = [c for c in final_cols if c in out.columns]  # guard
        logger.info("season: complete (defaults) in %.2f s – output shape %s", time.time() - start_ts, out[final_cols].shape)
        return out[final_cols]

    # === NORMAL PATH: join against prior season stats
    lookup_df = season_stats_df.copy()

    # 1) Ensure stable team key regardless of the RPC payload
    lookup_df = _ensure_team_norm(lookup_df)

    # Must have 'season' in lookup
    if "season" not in lookup_df.columns:
        warn = "season: season_stats_df missing 'season' column – falling back to defaults"
        if strict_inputs:
            raise ValueError(warn)
        logger.warning(warn)
        # Re-enter the defaults route
        for side in ("home", "away"):
            for feat, dval in default_vals.items():
                col = f"{side}_{feat}"
                out[col] = dval
                if flag_imputations:
                    out[f"{col}_imputed"] = True
        out = _append_differentials(out)
        passthru = [c for c in (
            "home_score", "away_score",
            "game_date", "game_time",
            "season",
            "home_team_norm", "away_team_norm",
            "kickoff_ts"
        ) if c in out.columns]
        new_cols = [
            "home_prev_season_win_pct", "away_prev_season_win_pct",
            "home_prev_season_srs_lite", "away_prev_season_srs_lite",
            "prev_season_win_pct_diff", "prev_season_srs_lite_diff",
        ]
        if flag_imputations:
            new_cols += [
                "home_prev_season_win_pct_imputed", "away_prev_season_win_pct_imputed",
                "home_prev_season_srs_lite_imputed", "away_prev_season_srs_lite_imputed",
            ]
        final_cols = ["game_id"] + passthru + [c for c in new_cols if c in out.columns]
        final_cols = [c for c in final_cols if c in out.columns]
        logger.info("season: complete (lookup missing 'season') in %.2f s – output shape %s", time.time() - start_ts, out[final_cols].shape)
        return out[final_cols]

    # 2) Keep freshest per (team_norm, season)
    if "updated_at" in lookup_df.columns:
        lookup_df = lookup_df.sort_values("updated_at", ascending=False, kind="mergesort")
    lookup_df = lookup_df.drop_duplicates(subset=["team_norm", "season"], keep="first")

    # 3) Rename payload columns to internal names
    lookup_df = lookup_df.rename(columns=feature_map)

    # 4) Verify expected columns (warn if RPC changed)
    feature_cols = list(DEFAULTS.keys())
    # but only the ones we actually use here
    feature_cols = ["win_pct", "srs_lite"]
    feature_cols = [  # mapped internal names
        "prev_season_win_pct",
        "prev_season_srs_lite",
    ]
    missing = [c for c in feature_cols if c not in lookup_df.columns]
    if missing:
        logger.warning("season: missing expected cols in season_stats_df: %s", missing)

    # 5) Build the lookup index and attach
    keep_cols = [c for c in feature_cols if c in lookup_df.columns]
    lookup = lookup_df.set_index(["team_norm", "season"])[keep_cols]

    out["lookup_season"] = out["season"] - 1
    for side in ("home", "away"):
        team_series = out[f"{side}_team_norm"].astype(str).str.lower()
        season_series = out["lookup_season"]
        idx = pd.MultiIndex.from_arrays([team_series, season_series])
        joined = lookup.reindex(idx)
        joined = joined.add_prefix(f"{side}_").reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)

    # Fill NAs with defaults + optional imputation flags
    for side in ("home", "away"):
        for feat, dval in {"prev_season_win_pct": DEFAULTS["win_pct"],
                           "prev_season_srs_lite": DEFAULTS["srs_lite"]}.items():
            col = f"{side}_{feat}"
            if flag_imputations:
                out[f"{col}_imputed"] = out[col].isna()
            out[col] = out[col].fillna(dval)

    out = _append_differentials(out)
    out.drop(columns=["lookup_season"], errors="ignore", inplace=True)

    # Pass-through base columns so later modules have what they need
    passthru = [c for c in (
        "home_score", "away_score",     # labels
        "game_date", "game_time",
        "season",
        "home_team_norm", "away_team_norm",
        "kickoff_ts"                    # if your schedule carries a timestamp
    ) if c in out.columns]

    new_cols = [
        "home_prev_season_win_pct", "away_prev_season_win_pct",
        "home_prev_season_srs_lite", "away_prev_season_srs_lite",
        "prev_season_win_pct_diff", "prev_season_srs_lite_diff",
    ]
    if flag_imputations:
        new_cols += [
            "home_prev_season_win_pct_imputed", "away_prev_season_win_pct_imputed",
            "home_prev_season_srs_lite_imputed", "away_prev_season_srs_lite_imputed",
        ]

    final_cols = ["game_id"] + passthru + [c for c in new_cols if c in out.columns]
    final_cols = [c for c in final_cols if c in out.columns]  # guard against missing passthru fields
    logger.info("season: complete in %.2f s – output shape %s", time.time() - start_ts, out[final_cols].shape)
    return out[final_cols]
