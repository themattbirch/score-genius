# backend/mlb_features/h2h.py
"""
Calculates historical head-to-head (H2H) features for MLB games.
This version uses a vectorized approach for performance and relies on the
feature engine for all data preparation (e.g., column normalization).
Computes stats using shift+rolling to consider only past `max_games` per matchup.
"""
from __future__ import annotations
import logging
import time
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    logger.warning("Could not import MLB_DEFAULTS; using local fallbacks in h2h.py")
    MLB_DEFAULTS: Dict[str, Any] = {}

__all__ = ["transform"]

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 10,
    debug: bool = False,
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)
    start = time.time()
    result = df.copy()

    # Fast-exit if no history
    if historical_df is None or historical_df.empty:
        for c, v in {
            "matchup_num_games": 0,
            "matchup_avg_run_diff": 0.0,
            "matchup_home_win_pct": 0.5,
            "matchup_avg_total_runs": 9.0,
        }.items():
            result[c] = v
        logger.info("h2h: no history → filled defaults in %.3fs", time.time() - start)
        return result

    # Prepare data ----------------------------------------------------------
    # 1) slice & tag history
    hist = historical_df[[
        "game_date_et", "home_team_norm", "away_team_norm",
        "home_score", "away_score"
    ]].copy()
    hist["is_future"] = False

    # 2) slice & tag future rows
    future = result[[
        "game_id", "game_date_et", "home_team_norm", "away_team_norm",
        "home_score", "away_score"
    ]].copy()
    future["is_future"] = True

    # 3) combine and sort
    combined = pd.concat([hist, future], ignore_index=True, sort=False)
    combined.sort_values("game_date_et", inplace=True)

    combined["matchup_key"] = combined.apply(
        lambda r: "_vs_".join(sorted([r["home_team_norm"], r["away_team_norm"]])),
        axis=1,
    )
    combined["home_win"] = (combined["home_score"] > combined["away_score"]).astype("float").fillna(0.0)
    combined["run_diff"] = (combined["home_score"] - combined["away_score"]).fillna(0.0)
    combined["total_runs"] = (combined["home_score"] + combined["away_score"]).fillna(0.0)

    # Streaming H2H computation --------------------------------------------
    deques: dict[str, deque] = {}
    stats = {
        "matchup_num_games": [],
        "matchup_avg_run_diff": [],
        "matchup_home_win_pct": [],
        "matchup_avg_total_runs": [],
    }

    for _, row in combined.iterrows():
        key = row["matchup_key"]
        dq = deques.setdefault(key, deque())

        # Current features BEFORE adding this game
        n = len(dq)
        if n:
            rd_sum  = sum(x["run_diff"] for x in dq)
            win_sum = sum(x["home_win"] for x in dq)
            tr_sum  = sum(x["total_runs"] for x in dq)
            stats["matchup_num_games"].append(n)
            stats["matchup_avg_run_diff"].append(rd_sum / n)
            stats["matchup_home_win_pct"].append(win_sum / n)
            stats["matchup_avg_total_runs"].append(tr_sum / n)
        else:
            stats["matchup_num_games"].append(0)
            stats["matchup_avg_run_diff"].append(0.0)
            stats["matchup_home_win_pct"].append(0.5)
            stats["matchup_avg_total_runs"].append(9.0)

        # Push current game into deque for future rows
        dq.append({
            "run_diff": row["run_diff"],
            "home_win": row["home_win"],
            "total_runs": row["total_runs"],
        })
        if len(dq) > max_games:
            dq.popleft()

    combined[list(stats)] = pd.DataFrame(stats)

    # Merge features back
    features = (
        combined[combined["is_future"]]                # ← future rows only
        .loc[:, [
            "game_id",
            "matchup_num_games",
            "matchup_avg_run_diff",
            "matchup_home_win_pct",
            "matchup_avg_total_runs",
        ]]
        .drop_duplicates(subset="game_id")             # safety net
        .set_index("game_id")
    )
    result = result.join(features, on="game_id")

    logger.info("h2h: completed in %.3fs – output %s",
                time.time() - start, result.shape)
    return result
