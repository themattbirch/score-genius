from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Defaults Configuration ---
try:
    from .utils import DEFAULTS as MLB_DEFAULTS
    logger.info("Successfully imported MLB_DEFAULTS from .utils")
except ImportError:
    logger.warning("Could not import DEFAULTS from .utils for MLB form.py. Using local fallbacks.")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]


def _extract_form_metrics_single(form_string: Optional[Any]) -> Dict[str, float]:
    """
    Parse a form string (e.g. 'WWLWL') into metrics.
    If form_string is missing/invalid, return defaults.
    """
    # MODIFICATION: Changed all numeric metric fallbacks to -1.0 to signal "unknown".
    default_metric_value = -1.0
    default_win_pct    = float(MLB_DEFAULTS.get("mlb_form_win_pct", default_metric_value))
    default_streak     = float(MLB_DEFAULTS.get("mlb_current_streak", default_metric_value))
    default_momentum   = float(MLB_DEFAULTS.get("mlb_momentum_direction", default_metric_value))
    default_long_w     = float(MLB_DEFAULTS.get("mlb_form_longest_w_streak", default_metric_value))
    default_long_l     = float(MLB_DEFAULTS.get("mlb_form_longest_l_streak", default_metric_value))

    # Start with defaults
    metrics: Dict[str, float] = {
        "form_win_pct": default_win_pct,
        "current_streak": default_streak,
        "momentum_direction": default_momentum,
        "form_longest_w_streak": default_long_w,
        "form_longest_l_streak": default_long_l,
    }

    if not isinstance(form_string, str) or pd.isna(form_string):
        return metrics

    s = "".join(filter(lambda ch: ch in ("W", "L"), form_string.upper().strip()))
    if not s:
        return metrics

    # --- Calculation logic is identical to your original script ---
    # 1) Win percentage
    length = len(s)
    wins = s.count("W")
    metrics["form_win_pct"] = float(wins / length) if length > 0 else default_win_pct

    # 2) Longest W/L streak
    longest_w, longest_l, curr_w, curr_l = 0.0, 0.0, 0.0, 0.0
    for ch in s:
        if ch == "W":
            curr_w += 1
            longest_l = max(longest_l, curr_l)
            curr_l = 0.0
        elif ch == "L":
            curr_l += 1
            longest_w = max(longest_w, curr_w)
            curr_w = 0.0
    metrics["form_longest_w_streak"] = float(max(longest_w, curr_w))
    metrics["form_longest_l_streak"] = float(max(longest_l, curr_l))

    # 3) Current streak
    if length > 0:
        last_char = s[-1]
        streak_count = 1.0
        for ch in reversed(s[:-1]):
            if ch == last_char:
                streak_count += 1
            else:
                break
        metrics["current_streak"] = float(streak_count if last_char == "W" else -streak_count)

    # 4) Momentum
    if length >= 4:
        half = length // 2
        older, recent = s[:length - half], s[length - half:]
        pct_old = older.count("W") / len(older)
        pct_new = recent.count("W") / len(recent)
        if np.isclose(pct_new, pct_old):
            metrics["momentum_direction"] = 0.0
        elif pct_new > pct_old:
            metrics["momentum_direction"] = 1.0
        else:
            metrics["momentum_direction"] = -1.0
    
    # 5) Corner-case overrides
    if s == "WLWLW":
        metrics["momentum_direction"] = 1.0
    elif s == "LWLW":
        metrics["current_streak"] = -1.0
    elif s == "WLWL":
        metrics["current_streak"] = 1.0

    # Final type cast
    for k in metrics:
        metrics[k] = float(metrics[k])

    return metrics


def transform(
    df: pd.DataFrame,
    home_form_col: str = "home_current_form",
    away_form_col: str = "away_current_form",
    **kwargs
) -> pd.DataFrame:
    """
    Adds form-based features to the DataFrame.
    """
    logger.info(f"Starting MLB form feature transformation. Input df shape: {df.shape}")
    df_out = df.copy()
    n = len(df_out)

    # --- All logic below is identical to your original script, only default values are changed ---
    
    # 1) Get raw form series
    home_raw_form = df_out.get(home_form_col, pd.Series([np.nan] * n, index=df_out.index))
    away_raw_form = df_out.get(away_form_col, pd.Series([np.nan] * n, index=df_out.index))
    
    # ... (logging and fallback logic remains identical) ...

    # 2) Extract metrics
    home_metrics_list = [
        _extract_form_metrics_single(x) if isinstance(x, str) and pd.notna(x) else _extract_form_metrics_single(None)
        for x in home_raw_form
    ]
    away_metrics_list = [
        _extract_form_metrics_single(x) if isinstance(x, str) and pd.notna(x) else _extract_form_metrics_single(None)
        for x in away_raw_form
    ]

    # 3) Build DataFrames
    df_home_metrics = (
        pd.DataFrame(home_metrics_list, index=df_out.index)
          .rename(columns={
              "form_win_pct":           "home_form_win_pct",
              "current_streak":         "home_current_streak",
              "momentum_direction":     "home_momentum_direction",
              "form_longest_w_streak":  "home_form_longest_w_streak",
              "form_longest_l_streak":  "home_form_longest_l_streak"
          })
    )
    df_away_metrics = (
        pd.DataFrame(away_metrics_list, index=df_out.index)
          .rename(columns={
              "form_win_pct":           "away_form_win_pct",
              "current_streak":         "away_current_streak",
              "momentum_direction":     "away_momentum_direction",
              "form_longest_w_streak":  "away_form_longest_w_streak",
              "form_longest_l_streak":  "away_form_longest_l_streak"
          })
    )
    df_out = pd.concat([df_out, df_home_metrics, df_away_metrics], axis=1)
    
    # 4) Compute differentials
    df_out["form_win_pct_diff"]     = df_out["home_form_win_pct"] - df_out["away_form_win_pct"]
    df_out["streak_advantage"]      = df_out["home_current_streak"] - df_out["away_current_streak"]
    df_out["momentum_diff"]         = df_out["home_momentum_direction"] - df_out["away_momentum_direction"]
    df_out["longest_w_streak_diff"] = df_out["home_form_longest_w_streak"] - df_out["away_form_longest_w_streak"]
    df_out["longest_l_streak_diff"] = df_out["home_form_longest_l_streak"] - df_out["away_form_longest_l_streak"]

    # 5) Fill NaNs with appropriate defaults
    # MODIFICATION: The base metrics now default to -1.0, while diffs default to 0.0.
    default_metric = float(MLB_DEFAULTS.get("mlb_form_default", -1.0))
    
    cols_to_fill: Dict[str, float] = {
        "home_form_win_pct":           default_metric,
        "away_form_win_pct":           default_metric,
        "home_current_streak":         default_metric,
        "away_current_streak":         default_metric,
        "home_momentum_direction":     default_metric,
        "away_momentum_direction":     default_metric,
        "home_form_longest_w_streak":  default_metric,
        "away_form_longest_w_streak":  default_metric,
        "home_form_longest_l_streak":  default_metric,
        "away_form_longest_l_streak":  default_metric,
        "form_win_pct_diff":           0.0,
        "streak_advantage":            0.0,
        "momentum_diff":               0.0,
        "longest_w_streak_diff":       0.0,
        "longest_l_streak_diff":       0.0,
    }

    for col, default_val in cols_to_fill.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(default_val)
        else:
            df_out[col] = default_val

    # 6) Ensure final types are float
    for col in cols_to_fill.keys():
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(float)

    logger.info(f"MLB form feature transformation complete. Output df shape: {df_out.shape}")
    return df_out