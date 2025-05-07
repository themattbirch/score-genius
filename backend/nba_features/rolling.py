# ---------------------------------------------------------------------
# backend/nba_features/rolling.py - Using robust helper functions
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

# Assuming utils.py contains DEFAULTS dict and normalize_team_name function
from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]


# ------------------------------------------------------------------ #
# Helper Functions for Rolling Calculations
# ------------------------------------------------------------------ #

def _lagged_rolling_stat(
    s: pd.Series,
    w: int,
    min_p: int,
    stat_func: str
) -> pd.Series:
    """
    Leakage‑free rolling statistic (mean or std).

    • Excludes *all* games played on the same calendar date as the current row.  
    • Falls back to the mean/std of the available history (>=1 game) when the
      group has fewer than min_periods rows – this covers the common
      “second‑game” edge case without touching min_periods itself.
    """
    if s.empty:
        return s

    ordered = s.sort_index()                       # index == game_date
    dates   = pd.Series(ordered.index, index=ordered.index)

    # --- 1. shift(1) to drop the current game ------------------------------
    shifted_vals = ordered.shift(1)

    # --- 2. wipe out any same‑day rows (idx == previous idx) ---------------
    same_day_mask = dates == dates.shift(1)
    shifted_vals[same_day_mask] = np.nan

    # --- 3. rolling calc ---------------------------------------------------
    _roller = shifted_vals.rolling(window=w, min_periods=min_p)
    if stat_func == "mean":
        rolled = _roller.mean()
        fallback = shifted_vals.rolling(window=w, min_periods=1).mean()
    elif stat_func == "std":
        rolled = _roller.std()
        fallback = shifted_vals.rolling(window=w, min_periods=1).std()
    else:
        raise ValueError(f"Unsupported stat_func: {stat_func}")

    # If we failed the min_periods test but still have ≥1 prior game,
    # use the fallback value (covers the “one‑game history” scenario).
    rolled = rolled.where(~rolled.isna(), fallback)

    # Re‑align to the original slice order expected by .transform
    return rolled.reindex(s.index)

# ------------------------------------------------------------------ #
# Main Transform Function
# ------------------------------------------------------------------ #

def _mk(side: str, base: str, kind: str, w: int) -> str:
    return f"{side}_rolling_{base}_{kind}_{w}"


def transform(
    df: pd.DataFrame,
    *,
    historical_df: pd.DataFrame,
    window_sizes: List[int] = (5, 10, 20),
    debug: bool = False,
) -> pd.DataFrame:
    if debug:
        logger.debug(f"[rolling] input df shape: {df.shape}")

    if df.empty:
        return df.copy()

    # --- Prepare history + upcoming as one DataFrame ---
    hist = historical_df.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"], utc=True).dt.tz_localize(None)
    hist["home_norm"] = hist["home_team"].map(normalize_team_name)
    hist["away_norm"] = hist["away_team"].map(normalize_team_name)
    hist["_is_hist"] = True

    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.tz_localize(None)
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)
    out["_is_hist"] = False

    full = pd.concat([hist, out], ignore_index=True)

    # --- Build long format via melt ---
    # 1) slice home
    home = (
        full
        .rename(columns={
            "home_norm": "team_norm",
            "home_score": "val_score_for",
            "home_offensive_rating": "val_off_rating",
            "home_defensive_rating": "val_def_rating",
            "home_net_rating": "val_net_rating",
        })
        [["game_id","game_date","team_norm","val_score_for","val_off_rating","val_def_rating","val_net_rating"]]
    )
    # 2) slice away
    away = (
        full
        .rename(columns={
            "away_norm": "team_norm",
            "away_score": "val_score_for",
            "away_offensive_rating": "val_off_rating",
            "away_defensive_rating": "val_def_rating",
            "away_net_rating": "val_net_rating",
        })
        [["game_id","game_date","team_norm","val_score_for","val_off_rating","val_def_rating","val_net_rating"]]
    )
    # 3) stack + melt
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.melt(
        id_vars=["game_id","team_norm","game_date"],
        value_vars=["val_score_for","val_off_rating","val_def_rating","val_net_rating"],
        var_name="stat",
        value_name="val"
    )
    long_df["stat"] = long_df["stat"].str.replace("val_", "", regex=False)
    long_df = long_df.set_index("game_date").sort_index()

    # --- rolling calculations ---
    pieces = []
    for w in window_sizes:
        min_p = max(1, w // 2)
        grp = long_df.groupby(["team_norm","stat"], group_keys=False, observed=True)
        mean_col = f"mean_{w}"
        std_col  = f"std_{w}"

        long_df[mean_col] = grp["val"].transform(lambda s: _lagged_rolling_stat(s, w, min_p, "mean"))
        long_df[std_col]  = grp["val"].transform(lambda s: _lagged_rolling_stat(s, w, min_p, "std"))

        # fill←DEFAULTS, clip std
        defaults = long_df["stat"].map(DEFAULTS)
        default_stds = long_df["stat"].map(lambda s: DEFAULTS.get(f"{s}_std", 0.0))
        long_df[mean_col].fillna(defaults, inplace=True)
        long_df[std_col].fillna(default_stds).clip(lower=0, inplace=True)

        # pivot back to wide
        reset = long_df.reset_index()
        wm = reset.pivot_table(
            index=["game_id","team_norm"], 
            columns="stat", 
            values=mean_col, 
            aggfunc="first"
        )
        wm.columns = [f"rolling_{stat}_mean_{w}" for stat in wm.columns]
        ws = reset.pivot_table(
            index=["game_id","team_norm"], 
            columns="stat", 
            values=std_col, 
            aggfunc="first"
        )
        ws.columns = [f"rolling_{stat}_std_{w}" for stat in ws.columns]
        pieces.append(pd.concat([wm, ws], axis=1))

    if not pieces:
        logger.warning("No rolling features generated; returning original")
        return out.drop(columns=["home_norm","away_norm","_is_hist"], errors="ignore")

    tidy = pd.concat(pieces, axis=1).reset_index()
    tidy["merge_key"] = tidy["game_id"].astype(str) + "_" + tidy["team_norm"]
    out["merge_key_home"] = out["game_id"].astype(str) + "_" + out["home_norm"]
    out["merge_key_away"] = out["game_id"].astype(str) + "_" + out["away_norm"]

    roll_cols = [c for c in tidy.columns if c.startswith("rolling_")]
    home_map = {c: _mk("home", "_".join(c.split("_")[1:-2]), c.split("_")[-2], int(c.split("_")[-1])) for c in roll_cols}
    away_map = {c: _mk("away", "_".join(c.split("_")[1:-2]), c.split("_")[-2], int(c.split("_")[-1])) for c in roll_cols}

    out = out.merge(
        tidy[["merge_key"] + roll_cols].rename(columns=home_map),
        left_on="merge_key_home", right_on="merge_key", how="left"
    ).merge(
        tidy[["merge_key"] + roll_cols].rename(columns=away_map),
        left_on="merge_key_away", right_on="merge_key", how="left",
        suffixes=("", "_drop")
    )

    # clean up
    drops = {
        "home_norm","away_norm","_is_hist",
        "merge_key","merge_key_home","merge_key_away"
    }
    drops |= {c for c in out if c.endswith("_drop")}
    return out.drop(columns=list(drops), errors="ignore")
