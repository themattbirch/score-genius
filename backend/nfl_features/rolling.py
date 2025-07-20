# backend/mlb_features/rolling.py
from __future__ import annotations

"""
Calculates leakage-free rolling mean features for MLB games.
This version has been simplified for robustness and clarity.
"""
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,  # alias for df
    window_sizes: List[int] = [5, 10, 30],
    debug: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    stat_map = {
        "runs_scored": ("home_score", "away_score"),
        "runs_allowed": ("away_score", "home_score"),
        "hits_for": ("home_hits", "away_hits"),
    }

    # Ensure columns exist
    for home_col, away_col in stat_map.values():
        if home_col not in out.columns:
            out[home_col] = pd.NA
        if away_col not in out.columns:
            out[away_col] = pd.NA

    # Build long format
    home_log = out[["game_id","game_date_et","home_team_norm","home_score","away_score","home_hits","away_hits"]].rename(columns={"home_team_norm":"team_norm"})
    home_log["is_home"] = True
    away_log = out[["game_id","game_date_et","away_team_norm","home_score","away_score","home_hits","away_hits"]].rename(columns={"away_team_norm":"team_norm"})
    away_log["is_home"] = False
    long_df = pd.concat([home_log, away_log], ignore_index=True)

    # Map stats
    for stat, (h_col, a_col) in stat_map.items():
        long_df[stat] = pd.to_numeric(
            np.where(long_df["is_home"], long_df[h_col], long_df[a_col]),
            errors="coerce",
        )

    long_df.sort_values(by=["team_norm","game_date_et","game_id"], inplace=True)

    # Rolling means
    all_roll = []
    for stat in stat_map:
        for w in window_sizes:
            col = f"rolling_{stat}_mean_{w}"
            all_roll.append(col)
            long_df[col] = (
                long_df.groupby("team_norm")[stat]
                .shift(1)
                .rolling(window=w, min_periods=1)
                .mean()
            )

    # Pivot back
    pivoted = long_df.pivot_table(index="game_id", columns="team_norm", values=all_roll)
    pivoted.columns = ["_".join(c) for c in pivoted.columns.values]
    result = pd.merge(out, pivoted, on="game_id", how="left")

    # Home/away prefix
    for idx, row in result.iterrows():
        home, away = row["home_team_norm"], row["away_team_norm"]
        for col in pivoted.columns:
            if f"_{home}" in col:
                new = col.replace(f"_{home}", "")
                result.loc[idx, f"home_{new}"] = row[col]
            if f"_{away}" in col:
                new = col.replace(f"_{away}", "")
                result.loc[idx, f"away_{new}"] = row[col]

    result.drop(columns=pivoted.columns, inplace=True, errors="ignore")
    return result
