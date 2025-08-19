# backend/nfl_features/form.py
"""Win-loss *form* and current streak features for NFL teams.

Leakage-free + memory-lean.
Key updates:
  • Optional per-team merge_asof tolerance (days) to avoid dragging in very old form.
  • Clamp lookback_window to ≥1 and use the effective window consistently.
  • Pre-sort long frames once; avoid re-sorting inside per-team asof.
  • Unify default helper; add OR-imputation flags for quick downstream visibility.
"""

from __future__ import annotations

import logging
from typing import Optional, Mapping, List

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# ---------- Helpers ----------

def _mk_ts(df: pd.DataFrame, date_col: str = "game_date", time_col: str = "game_time") -> pd.Series:
    date_s = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if time_col in df.columns:
        t = df[time_col].astype(str).fillna("00:00:00")
        ts = pd.to_datetime(date_s.dt.strftime("%Y-%m-%d") + " " + t, errors="coerce", utc=True)
        ts = ts.fillna(date_s + pd.Timedelta(hours=12))
    else:
        ts = date_s + pd.Timedelta(hours=12)
    return ts


def _pick_historical(kwargs: Mapping[str, object]) -> Optional[pd.DataFrame]:
    candidates = (
        "historical_df", "historical", "hist_df", "history_df",
        "historical_games", "historical_games_df",
        "historical_game_stats", "historical_team_game_stats",
        "historical_team_games", "historical_team_games_df",
        "recent_form_df", "recent_form", "nfl_recent_form",
        "rpc_recent_form_df", "rpc_get_nfl_all_recent_form",
        "team_game_stats_df", "team_game_stats",
    )
    for key in candidates:
        obj = kwargs.get(key) if isinstance(kwargs, dict) else None
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            logger.info("form: using historical source '%s' with shape %s", key, obj.shape)
            return obj
    if isinstance(kwargs, dict) and kwargs:
        snapshot = {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in kwargs.items()}
        logger.warning("form: no historical_df found. Available kwargs keys: %s", snapshot)
    return None


def _prepare_long_format(historical: pd.DataFrame) -> pd.DataFrame:
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce")

    # Treat *_norm as already-canonical: lowercase only (do NOT re-normalize).
    for c in ("home_team_norm", "away_team_norm"):
        if c in hist.columns:
            hist[c] = hist[c].astype(str).str.lower()

    # Keep only the columns we actually need if present
    base_cols = ["game_id", "game_date", "home_team_norm", "away_team_norm", "home_score", "away_score"]
    if "game_time" in hist.columns:
        base_cols.append("game_time")
    hist = hist[[c for c in base_cols if c in hist.columns]]

    # Build long format (home and away perspectives)
    home_games = hist.rename(columns={
        "home_team_norm": "team",
        "away_team_norm": "opponent",
        "home_score": "team_score",
        "away_score": "opp_score",
    })
    away_games = hist.rename(columns={
        "away_team_norm": "team",
        "home_team_norm": "opponent",
        "away_score": "team_score",
        "home_score": "opp_score",
    })

    long_df = pd.concat([home_games, away_games], ignore_index=True)
    long_df["team"] = long_df["team"].astype(str).str.lower()
    long_df["opponent"] = long_df["opponent"].astype(str).str.lower()

    # Sort using whatever keys exist; game_id may be absent in history
    sort_keys = [c for c in ["team", "game_date", "game_id"] if c in long_df.columns]
    if not sort_keys:
        sort_keys = ["team", "game_date"]
    long_df = long_df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    for c in ("team_score", "opp_score"):
        if c in long_df.columns:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce", downcast="integer")
    return long_df


def _compute_outcomes(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()

    # +1 win, -1 loss, 0 tie
    df["outcome"] = np.select(
        [df["team_score"] > df["opp_score"], df["team_score"] < df["opp_score"]],
        [1, -1],
        default=0,
    ).astype("int8")

    # Build the *after this game* running streak (so the asof join picks the
    # last completed game's streak value as the "current" entering the next game).
    def _streak_signed_after(outcome: np.ndarray) -> np.ndarray:
        n = outcome.shape[0]
        after = np.zeros(n, dtype=np.int16)
        cur = 0
        for i in range(n):
            v = outcome[i]
            if v == 0:
                cur = 0
            elif v > 0:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
            after[i] = cur
        return after

    df["current_streak"] = 0
    for _, idx in df.groupby("team", sort=False).indices.items():
        s = df.loc[idx, "outcome"].to_numpy()
        df.loc[idx, "current_streak"] = _streak_signed_after(s)

    # <-- This was missing and causes the KeyError: 'win'
    df["win"] = (df["outcome"] > 0).astype("int8")

    return df

def _dedup_form_rhs(rhs: pd.DataFrame) -> pd.DataFrame:
    rhs = rhs.copy()
    rhs["kickoff_ts"] = _mk_ts(rhs, date_col="game_date", time_col="game_time" if "game_time" in rhs.columns else "game_time")

    rhs = rhs.sort_values(["team", "kickoff_ts"], kind="mergesort")
    rhs = rhs.drop_duplicates(["team", "kickoff_ts"], keep="last")

    rhs["game_day"] = rhs["kickoff_ts"].dt.date
    rhs = rhs.sort_values(["team", "game_day", "kickoff_ts"], kind="mergesort")
    rhs = rhs.drop_duplicates(["team", "game_day"], keep="last")
    return rhs


def _pivot_wide(merged_long: pd.DataFrame, win_k: int) -> pd.DataFrame:
    value_cols = [f"form_win_pct_{win_k}", "current_streak"]

    if "game_id" not in merged_long.columns:
        if "game_id_x" in merged_long.columns:
            merged_long = merged_long.rename(columns={"game_id_x": "game_id"})
        if "game_id_y" in merged_long.columns:
            merged_long = merged_long.drop(columns=["game_id_y"])

    merged_long = merged_long.sort_values(["game_id", "side", "kickoff_ts"], kind="mergesort")
    merged_long = merged_long.drop_duplicates(["game_id", "side"], keep="last")

    pivot_df = merged_long.pivot_table(index="game_id", columns="side", values=value_cols, aggfunc="last")
    pivot_df.columns = [f"{side}_{metric}" for metric, side in pivot_df.columns.to_flat_index()]
    return pivot_df.reset_index()


def _asof_join_per_team(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    on: str,
    tolerance: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for team, left_idx in left_df.groupby("team", sort=False).indices.items():
        left_g = left_df.loc[left_idx]
        rg = right_df[right_df["team"] == team]
        if rg.empty:
            parts.append(left_g)
            continue
        merged = pd.merge_asof(
            left_g,
            rg,
            left_on=on,
            right_on=on,
            direction="backward",
            allow_exact_matches=False,
            tolerance=tolerance,
        )
        parts.append(merged)
    if not parts:
        return left_df
    return pd.concat(parts, ignore_index=True)


# ---------- Main entrypoint ----------

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    lookback_window: int = 5,
    flag_imputations: bool = True,
    strict_inputs: bool = False,
    tolerance_days: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    win_k = max(1, int(lookback_window))

    def _return_default_form(base_games: pd.DataFrame) -> pd.DataFrame:
        base = base_games[["game_id"]].copy()
        for side in ("home", "away"):
            base[f"{side}_form_win_pct_{win_k}"] = DEFAULTS["form_win_pct"]
            base[f"{side}_current_streak"] = DEFAULTS["current_streak"]
            if flag_imputations:
                base[f"{side}_form_imputed"] = 1
                base[f"{side}_streak_imputed"] = 1
        base[f"form_win_pct_{win_k}_diff"] = 0.0
        base["current_streak_diff"] = 0.0

        passthru = [c for c in (
            "game_date", "game_time", "season",
            "home_team_norm", "away_team_norm",
            "home_score", "away_score", "kickoff_ts"
        ) if c in base_games.columns]
        if passthru:
            base = base.merge(base_games[["game_id"] + passthru], on="game_id", how="left")

        if flag_imputations:
            base[f"form_any_imputed_{win_k}"] = np.int8(1)
            base["streak_any_imputed"] = np.int8(1)
        return base

    if historical_df is None or (isinstance(historical_df, pd.DataFrame) and historical_df.empty):
        historical_df = _pick_historical(kwargs)

    if historical_df is None or historical_df.empty:
        msg = "form: No historical data – defaulting all form features."
        if strict_inputs:
            raise ValueError(msg)
        logger.warning(msg)
        return _return_default_form(games)

    long_df = _prepare_long_format(historical_df)
    long_df = _compute_outcomes(long_df)

    long_df[f"form_win_pct_{win_k}"] = (
    long_df.groupby("team", sort=False)["win"]
    .rolling(win_k, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype("float32")
)

    rhs_cols = ["team", "game_date", "current_streak", f"form_win_pct_{win_k}"]
    if "game_time" in long_df.columns:
        rhs_cols.append("game_time")
    features_long = _dedup_form_rhs(long_df[rhs_cols])

    upcoming = games.copy()
    if "game_date" not in upcoming.columns or upcoming["game_date"].isna().all():
        if "kickoff_ts" in upcoming.columns:
            ts = pd.to_datetime(upcoming["kickoff_ts"], errors="coerce", utc=True)
            if ts.notna().any():
                upcoming["game_date"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                logger.warning("form: kickoff_ts present but unusable – defaulting form features.")
                return _return_default_form(games)
        else:
            logger.warning("form: missing game_date and kickoff_ts – defaulting form features.")
            return _return_default_form(games)

    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"], errors="coerce")
    # normalize upcoming ONLY if we do NOT already have *_team_norm
    for c in ("home_team_norm", "away_team_norm"):
        if c in upcoming.columns:
            # trust that *_team_norm is already canonical
            upcoming[c] = upcoming[c].astype(str).str.lower()
        elif c.replace("_team_norm", "_team") in upcoming.columns:
            # only normalize if it's a raw name/id
            raw = c.replace("_team_norm", "_team")
            upcoming[c] = upcoming[raw].apply(normalize_team_name).astype(str).str.lower()



    lhs_cols = ["game_id", "game_date", "home_team_norm", "away_team_norm"]
    if "game_time" in upcoming.columns:
        lhs_cols.append("game_time")
    if "kickoff_ts" in upcoming.columns:
        lhs_cols.append("kickoff_ts")
    upcoming = upcoming[[c for c in lhs_cols if c in upcoming.columns]]

    id_vars = [c for c in ["game_id", "game_date", "game_time", "kickoff_ts"] if c in upcoming.columns]
    upcoming_long = pd.melt(
        upcoming,
        id_vars=id_vars,
        value_vars=["home_team_norm", "away_team_norm"],
        var_name="side",
        value_name="team",
    )
    upcoming_long["side"] = upcoming_long["side"].str.replace("_team_norm", "", regex=False)
    upcoming_long["team"] = upcoming_long["team"].astype(str).str.lower()

    if "kickoff_ts" in upcoming_long.columns:
        upcoming_long["kickoff_ts"] = pd.to_datetime(upcoming_long["kickoff_ts"], errors="coerce", utc=True)
        if upcoming_long["kickoff_ts"].isna().all():
            upcoming_long["kickoff_ts"] = _mk_ts(upcoming_long, date_col="game_date", time_col="game_time" if "game_time" in upcoming_long.columns else "game_time")
    else:
        upcoming_long["kickoff_ts"] = _mk_ts(upcoming_long, date_col="game_date", time_col="game_time" if "game_time" in upcoming_long.columns else "game_time")

    upcoming_long = upcoming_long.dropna(subset=["team", "kickoff_ts"])
    features_long = features_long.dropna(subset=["team", "kickoff_ts"])

    upcoming_long["team"] = upcoming_long["team"].astype("category")
    features_long["team"] = features_long["team"].astype("category")
    upcoming_long["side"] = upcoming_long["side"].astype("category")

    upcoming_long = upcoming_long.sort_values(["team", "kickoff_ts"], kind="mergesort").reset_index(drop=True)
    features_long = features_long.sort_values(["team", "kickoff_ts"], kind="mergesort").reset_index(drop=True)

    tol = pd.Timedelta(days=int(tolerance_days)) if tolerance_days is not None else None
    merged_long = _asof_join_per_team(upcoming_long, features_long, on="kickoff_ts", tolerance=tol)

    pivot_df = _pivot_wide(merged_long, win_k)

    result = games[["game_id"]].merge(pivot_df, on="game_id", how="left")

    fills = {
        f"home_form_win_pct_{win_k}": DEFAULTS["form_win_pct"],
        f"away_form_win_pct_{win_k}": DEFAULTS["form_win_pct"],
        "home_current_streak": DEFAULTS["current_streak"],
        "away_current_streak": DEFAULTS["current_streak"],
    }
    for col, dval in fills.items():
        if flag_imputations:
            result[f"{col}_imputed"] = result[col].isna().astype("int8")
        result[col] = result[col].fillna(dval)

    if flag_imputations:
        result[f"form_any_imputed_{win_k}"] = (
            result[f"home_form_win_pct_{win_k}_imputed"].astype("int8") |
            result[f"away_form_win_pct_{win_k}_imputed"].astype("int8")
        ).astype("int8")
        result["streak_any_imputed"] = (
            result["home_current_streak_imputed"].astype("int8") |
            result["away_current_streak_imputed"].astype("int8")
        ).astype("int8")

    result[f"form_win_pct_{win_k}_diff"] = (
        result[f"home_form_win_pct_{win_k}"] - result[f"away_form_win_pct_{win_k}"]
    )
    result["current_streak_diff"] = result["home_current_streak"] - result["away_current_streak"]

    for c in (f"home_form_win_pct_{win_k}", f"away_form_win_pct_{win_k}", f"form_win_pct_{win_k}_diff"):
        result[c] = result[c].astype("float32")
    for c in ("home_current_streak", "away_current_streak", "current_streak_diff"):
        result[c] = result[c].astype("int16")

    return result
