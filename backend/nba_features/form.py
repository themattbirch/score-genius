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

def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    team_stats_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    original_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled for form.transform")

    if df is None or df.empty:
        logger.warning("form.transform: empty input, returning empty DataFrame")
        if debug:
            logger.setLevel(original_level)
        return pd.DataFrame()

    result_df = df.copy()
    form_col = "current_form"
    metric_keys = list(_extract_form_metrics_single("").keys())  # ['form_win_pct','current_streak','momentum_direction']

    # 1) If there's no current_form on the games DF, try to pull it from team_stats_df
    if form_col not in result_df.columns:
        if (
            team_stats_df is not None
            and "team_name" in team_stats_df.columns
            and form_col in team_stats_df.columns
        ):
            logger.info("Pulling 'current_form' from team_stats_df")
            uniq = (
                team_stats_df
                .loc[:, ["team_name", "current_form"]]
                .drop_duplicates(subset="team_name", keep="last")
            )
            mapping = (
                uniq
                .set_index("team_name")["current_form"]
                .fillna("")
            )

            result_df[form_col] = (
                result_df["home_team"]
                .map(mapping)
                .fillna("")   # fallback if a team wasn’t in team_stats_df
            )
        else:
            logger.warning(f"Missing '{form_col}' and no team_stats_df available; filling with defaults")
            # fill metrics with defaults & skip extraction
            for key in metric_keys:
                result_df[key] = DEFAULTS.get(key, 0.0)
            if debug:
                logger.setLevel(original_level)
            return result_df  # we’re done

    # 2) At this point, we have a 'current_form' column (maybe injected, maybe original)
    #    Clean it and then apply the extractor:
    result_df[form_col] = (
        result_df[form_col]
        .fillna("")
        .astype(str)
        .replace("N/A", "")
    )
    metrics_series = result_df[form_col].apply(_extract_form_metrics_single)
    metrics_df = pd.DataFrame(metrics_series.tolist(), index=result_df.index)
    result_df = result_df.join(metrics_df)

    # 3) Enforce types & fill any remaining NaNs (shouldn’t be any):
    for key in metric_keys:
        default_val = DEFAULTS.get(key, 0.0 if key != "current_streak" else 0)
        result_df[key] = pd.to_numeric(
            result_df.get(key, default_val), errors="coerce"
        ).fillna(default_val)
        if key == "current_streak":
            result_df[key] = result_df[key].round().astype(int)

    if debug:
        logger.debug("form.transform: done, output shape=%s", result_df.shape)
        logger.setLevel(original_level)

    return result_df
