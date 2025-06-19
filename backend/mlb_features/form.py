# backend/mlb_features/form.py

from __future__ import annotations
import logging
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("Could not import DEFAULTS; using local fallbacks in form.py")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]

def _calculate_form_metrics(
    team_games: pd.DataFrame, team_norm: str, num_form_games: int
) -> dict:
    # Drop rows missing scores
    team_games = team_games.dropna(subset=["home_score", "away_score"])
    if team_games.empty:
        return {
            "current_form": "N/A",
            "form_win_pct": MLB_DEFAULTS.get("form_win_pct", 0.5),
            "current_streak": MLB_DEFAULTS.get("current_streak", 0.0),
        }

    recent = (
        team_games
        .sort_values("game_date_et", ascending=False)
        .head(num_form_games)
        .copy()
    )

    # Build W/L/T vectorised
    is_home = recent["home_team_norm"] == team_norm
    recent["outcome"] = np.select(
        [
            (is_home) & (recent["home_score"] > recent["away_score"]),
            (~is_home) & (recent["away_score"] > recent["home_score"]),
            recent["home_score"] == recent["away_score"],
        ],
        ["W", "W", "T"],
        default="L",
    )

    # Chronological form string
    form_string = "".join(reversed(recent["outcome"].tolist()))

    # Compute win % on decisive games
    win_sample = recent[recent["outcome"].isin(["W", "L"])]
    win_pct = 0.5 if win_sample.empty else (win_sample["outcome"] == "W").mean()

    # Compute streak
    streak_base = win_sample["outcome"].iloc[::-1]  # newest → oldest
    if streak_base.empty:
        streak_val = 0.0
    else:
        last_res = streak_base.iloc[0]
        streak_len = (streak_base == last_res).cummin().sum()
        streak_val = streak_len if last_res == "W" else -streak_len

    return {
        "current_form": form_string,
        "form_win_pct": float(win_pct),
        "current_streak": float(streak_val),
    }

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    num_form_games: int = 5,
    debug: bool = False,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.info(f"form: starting transform – input shape {df.shape}")

    result = df.copy()
    if historical_df is None or historical_df.empty:
        logger.warning("form: No historical data provided. Skipping form calculations.")
        # ... (unchanged fallback) ...
        return result

    # --- Make a safe working copy of history & clean it ---------------
    hist = historical_df.copy()
    # Ensure numeric scores
    for col in ['home_score', 'away_score']:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors='coerce')
    # Drop any rows missing date or scores
    hist = hist.dropna(subset=['game_date_et', 'home_score', 'away_score'])
    # Reset index so boolean masks align
    hist = hist.reset_index(drop=True)

    all_form_metrics = []
    for _, game_row in result.iterrows():
        game_date = game_row["game_date_et"]
        home_team = game_row["home_team_norm"]
        away_team = game_row["away_team_norm"]

        # Use .loc and our cleaned `hist` copy
        home_hist = hist.loc[
            ((hist["home_team_norm"] == home_team) |
             (hist["away_team_norm"] == home_team))
            & (hist["game_date_et"] < game_date)
        ]
        away_hist = hist.loc[
            ((hist["home_team_norm"] == away_team) |
             (hist["away_team_norm"] == away_team))
            & (hist["game_date_et"] < game_date)
        ]

        home_metrics = _calculate_form_metrics(home_hist, home_team, num_form_games)
        away_metrics = _calculate_form_metrics(away_hist, away_team, num_form_games)

        all_form_metrics.append({
            "game_id": game_row["game_id"],
            "home_current_form": home_metrics["current_form"],
            "home_form_win_pct": home_metrics["form_win_pct"],
            "home_current_streak": home_metrics["current_streak"],
            "away_current_form": away_metrics["current_form"],
            "away_form_win_pct": away_metrics["form_win_pct"],
            "away_current_streak": away_metrics["current_streak"],
        })

    form_df = pd.DataFrame(all_form_metrics)
    result = result.merge(form_df, on="game_id", how="left")
    result["form_win_pct_diff"] = result["home_form_win_pct"] - result["away_form_win_pct"]
    result["current_streak_diff"] = result["home_current_streak"] - result["away_current_streak"]

    logger.info(f"form: transform complete – output shape {result.shape}")
    return result
