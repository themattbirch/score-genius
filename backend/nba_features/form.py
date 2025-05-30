from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import DEFAULTS

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Default thresholds and fallback values
_FORM_DEFAULTS: Dict[str, float] = {
    "form_win_pct": DEFAULTS.get("form_win_pct", 0.5),
    "current_streak": DEFAULTS.get("current_streak", 0.0),
    "momentum_direction": DEFAULTS.get("momentum_direction", 0.0),
    "form_longest_w_streak": DEFAULTS.get("form_longest_w_streak", 0.0),
    "form_longest_l_streak": DEFAULTS.get("form_longest_l_streak", 0.0),
}


def _extract_form_metrics_single(form_string: Optional[str]) -> Dict[str, float]:
    """
    Parse a form string (e.g. 'WWLWL') into metrics:
      - form_win_pct: win percentage
      - current_streak: positive for Ws, negative for Ls
      - momentum_direction: +1 if recent half win_pct > older half, -1 if lower
      - form_longest_w_streak, form_longest_l_streak
    """
    metrics = _FORM_DEFAULTS.copy()
    if not isinstance(form_string, str) or pd.isna(form_string):
        return metrics

    s = form_string.upper().strip().replace("-", "").replace("?", "")
    if not s or s == "N/A":
        return metrics

    length = len(s)
    wins = s.count("W")
    metrics["form_win_pct"] = wins / length if length > 0 else metrics["form_win_pct"]

    longest_w = longest_l = 0.0 # Ensure float from start
    curr_w = curr_l = 0.0     # Ensure float from start
    for ch in s:
        if ch == 'W':
            curr_w += 1
            longest_l = max(longest_l, curr_l)
            curr_l = 0.0
        elif ch == 'L':
            curr_l += 1
            longest_w = max(longest_w, curr_w)
            curr_w = 0.0
        else:
            longest_w = max(longest_w, curr_w)
            longest_l = max(longest_l, curr_l)
            curr_w = curr_l = 0.0
    longest_w = max(longest_w, curr_w)
    longest_l = max(longest_l, curr_l)
    metrics["form_longest_w_streak"] = longest_w
    metrics["form_longest_l_streak"] = longest_l

    if length > 0:
        last = s[-1]
        streak_count = 1.0 # Ensure float
        for ch in reversed(s[:-1]):
            if ch == last:
                streak_count += 1
            else:
                break
        metrics["current_streak"] = streak_count if last == 'W' else -streak_count

    if length >= 4:
        half = length // 2
        older, recent = s[:length-half], s[length-half:]
        # len(older) and len(recent) are guaranteed > 0 if length >= 4
        pct_old = older.count('W') / len(older)
        pct_new = recent.count('W') / len(recent)
        if pct_new > pct_old:
            metrics["momentum_direction"] = 1.0
        elif pct_new < pct_old:
            metrics["momentum_direction"] = -1.0
    
    # Ensure all metrics returned are explicitly float if they come from calculations not directly from _FORM_DEFAULTS
    for key in metrics:
        if key not in _FORM_DEFAULTS: # Only for keys that might have been calculated as int/other
             metrics[key] = float(metrics[key])
        elif isinstance(metrics[key], (int, np.integer)): # Also ensure defaults are float if they were int
             metrics[key] = float(metrics[key])


    return metrics


def transform(
    df: pd.DataFrame, 
    *, 
    team_stats_df: Optional[pd.DataFrame] = None, # It already expected this implicitly via kwargs
    debug: bool = False  # Add the debug parameter
    ) -> pd.DataFrame:    
    """
    Expand home_current_form and away_current_form into individual metrics,
    then compute differences.
    """
    df = df.copy()
    n = len(df)

    home_series = df.get("home_current_form", pd.Series([np.nan]*n, index=df.index))
    away_series = df.get("away_current_form", pd.Series([np.nan]*n, index=df.index))

    home_metrics = pd.DataFrame(
        [_extract_form_metrics_single(x) for x in home_series],
        index=df.index
    ).add_prefix('home_')

    away_metrics = pd.DataFrame(
        [_extract_form_metrics_single(x) for x in away_series],
        index=df.index
    ).add_prefix('away_')

    df = pd.concat([df, home_metrics, away_metrics], axis=1)

    df['form_win_pct_diff'] = df['home_form_win_pct'] - df['away_form_win_pct']
    df['streak_advantage'] = df['home_current_streak'] - df['away_current_streak']
    df['momentum_diff'] = df['home_momentum_direction'] - df['away_momentum_direction']
    df['longest_w_streak_diff'] = (
        df['home_form_longest_w_streak'] - df['away_form_longest_w_streak']
    )
    df['longest_l_streak_diff'] = (
        df['home_form_longest_l_streak'] - df['away_form_longest_l_streak']
    )

    fill_vals = {
        **{f'home_{k}': v for k, v in _FORM_DEFAULTS.items()},
        **{f'away_{k}': v for k, v in _FORM_DEFAULTS.items()},
        'form_win_pct_diff': 0.0,
        'streak_advantage': 0.0,
        'momentum_diff': 0.0,
        'longest_w_streak_diff': 0.0,
        'longest_l_streak_diff': 0.0,
    }
    df = df.fillna(fill_vals)
    
    # Ensure numeric types for all generated/filled columns
    columns_to_cast = [col for col in fill_vals.keys() if col in df.columns]
    if columns_to_cast: # Check if list is not empty
        df[columns_to_cast] = df[columns_to_cast].astype(float)

    return df