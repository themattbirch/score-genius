# backend/nba_features/form.py

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS  # make sure this points at the same DEFAULTS you’ve been using

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

__all__ = ["transform"]

def _extract_form_metrics_single(form_string: Optional[str]) -> Dict[str, float]:
    defaults: Dict[str, float] = {
        "form_win_pct": DEFAULTS.get("form_win_pct", 0.5),
        "current_streak": DEFAULTS.get("current_streak", 0),
        "momentum_direction": DEFAULTS.get("momentum_direction", 0.0),
    }

    if not form_string or pd.isna(form_string) or not isinstance(form_string, str):
        return defaults

    s = form_string.upper().strip().replace("-", "").replace("?", "")
    if not s or s == "N/A":
        return defaults

    length = len(s)
    wins = s.count("W")
    form_win_pct = float(wins / length) if length > 0 else defaults["form_win_pct"]

    # current streak
    current_streak = 0.0
    last_char = s[-1]
    streak_count = 0
    for ch in reversed(s):
        if ch == last_char:
            streak_count += 1
        else:
            break
    current_streak = float(streak_count) if last_char == "W" else float(-streak_count)

    # momentum direction
    momentum_direction = 0.0
    if length >= 4:
        split_point = length // 2
        recent_half = s[-split_point:]
        older_half = s[: length - split_point]
        if recent_half and older_half:
            pct_r = float(recent_half.count("W") / len(recent_half))
            pct_o = float(older_half.count("W") / len(older_half))
            if pct_r > pct_o:
                momentum_direction = 1.0
            elif pct_r < pct_o:
                momentum_direction = -1.0

    return {
        "form_win_pct": form_win_pct,
        "current_streak": current_streak,
        "momentum_direction": momentum_direction,
    }

def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # pull raw form strings (or NaN if column missing)
    home_raw = df.get("home_current_form", pd.Series([np.nan] * n, index=df.index))
    away_raw = df.get("away_current_form", pd.Series([np.nan] * n, index=df.index))

    # extract per-row metrics
    home_metrics = [_extract_form_metrics_single(x) for x in home_raw]
    away_metrics = [_extract_form_metrics_single(x) for x in away_raw]

    # build DataFrames and rename
    df_home = (
        pd.DataFrame(home_metrics, index=df.index)
        .rename(columns={
            "form_win_pct":        "home_form_win_pct",
            "current_streak":      "home_current_streak",
            "momentum_direction":  "home_momentum_direction"
        })
    )
    df_away = (
        pd.DataFrame(away_metrics, index=df.index)
        .rename(columns={
            "form_win_pct":        "away_form_win_pct",
            "current_streak":      "away_current_streak",
            "momentum_direction":  "away_momentum_direction"
        })
    )

    # stitch back on
    df = pd.concat([df, df_home, df_away], axis=1)

    # compute the three diff columns
    df["form_win_pct_diff"] = df["home_form_win_pct"] - df["away_form_win_pct"]
    df["streak_advantage"]   = df["home_current_streak"] - df["away_current_streak"]
    df["momentum_diff"]      = df["home_momentum_direction"] - df["away_momentum_direction"]

    # fill any stragglers with defaults / zeros
    df["home_form_win_pct"].fillna(DEFAULTS["form_win_pct"], inplace=True)
    df["away_form_win_pct"].fillna(DEFAULTS["form_win_pct"], inplace=True)
    df["home_current_streak"].fillna(DEFAULTS["current_streak"], inplace=True)
    df["away_current_streak"].fillna(DEFAULTS["current_streak"], inplace=True)
    df["home_momentum_direction"].fillna(DEFAULTS["momentum_direction"], inplace=True)
    df["away_momentum_direction"].fillna(DEFAULTS["momentum_direction"], inplace=True)
    df["form_win_pct_diff"].fillna(0.0, inplace=True)
    df["streak_advantage"].fillna(0.0, inplace=True)
    df["momentum_diff"].fillna(0.0, inplace=True)

    return df
