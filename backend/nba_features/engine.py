# backend/nba_features/engine.py

from __future__ import annotations
import logging
import time
from typing import Any, List, Optional, Dict

import pandas as pd

import os
import sys
import pathlib

# 1) make sure your project root is on PYTHONPATH so imports work
ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# 1b) load your .env
from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

import pandas as pd
from pathlib import Path

# now this will see SUPABASE_URL and SUPABASE_SERVICE_KEY
from backend.caching.supabase_client import supabase

from backend.data_pipeline.history_loaders import (
    load_recent_box_scores,
    load_team_season_stats,
    load_schedule_next_14_days,
)

# bring in your transforms
from backend.nba_features.rolling   import transform as rolling_transform
from backend.nba_features.rest      import transform as rest_transform
from backend.nba_features.h2h       import transform as h2h_transform
from backend.nba_features.season    import transform as season_transform
from backend.nba_features.form      import transform as form_transform

logger = logging.getLogger(__name__)

# default order
DEFAULT_EXECUTION_ORDER = [
    "rolling","rest","h2h","season","form"
]

TRANSFORM_MAP: Dict[str, Any] = {
    "rolling":  rolling_transform,
    "rest":     rest_transform,
    "h2h":      h2h_transform,
    "season":   season_transform,
    "form":     form_transform,
}

def run_feature_pipeline(
    df: pd.DataFrame,
    *,
    historical_games_df: Optional[pd.DataFrame] = None,
    team_stats_df:       Optional[pd.DataFrame] = None,
    rolling_windows:     List[int]         = [5, 10, 20],
    h2h_window:          int               = 7,
    execution_order:     List[str]         = DEFAULT_EXECUTION_ORDER,
    debug:               bool              = False,
) -> pd.DataFrame:
    """
    Builds a fully‐enriched BOX_SCORES for history (including advanced metrics),
    then runs only the non‐historical steps on `df` (no momentum/advanced there).
    """

    # ——————————————————————————————————————————————————————————————————————
    # 1) prepare the HISTORY with advanced metrics baked in
    # ——————————————————————————————————————————————————————————————————————

    raw_hist = (
        historical_games_df.copy()
        if historical_games_df is not None
        else load_recent_box_scores()
    )

    # pul lout quarter totals if you haven’t done it in the loader
    home_qs = ["home_q1","home_q2","home_q3","home_q4","home_ot"]
    away_qs = ["away_q1","away_q2","away_q3","away_q4","away_ot"]
    if "home_score" not in raw_hist:
        raw_hist["home_score"] = raw_hist[home_qs].sum(axis=1)
        raw_hist["away_score"] = raw_hist[away_qs].sum(axis=1)

    from .advanced import transform as advanced_transform
    BOX_SCORES = advanced_transform(raw_hist, debug=debug)
    # grab team stats & schedule for the other modules
    TEAM_STATS = team_stats_df if team_stats_df is not None else load_team_season_stats()

    # ——————————————————————————————————————————————————————————————————————
    # 2) trim out momentum+advanced from the upcoming‐games pipeline
    # ——————————————————————————————————————————————————————————————————————

    # if user didn’t explicitly drop them, do it here:
    exec_order = [
        m for m in execution_order
        if m not in ("momentum", "advanced")
    ]

    MODULE_KWARGS = {
        "rolling": {
        "historical_df": BOX_SCORES,
        "window_sizes": rolling_windows,
         },        
        "rest":    {},
        "h2h":     {"historical_df": BOX_SCORES, "max_games": h2h_window},
        "season":  {"team_stats_df": TEAM_STATS},
        "form":    {},
    }

    processed_df = df.copy()
    for module_name in exec_order:
        if module_name not in MODULE_KWARGS:
            logger.warning(f"Skipping unknown module '{module_name}'")
            continue

        logger.info(f"Running module: {module_name}…")
        fn = TRANSFORM_MAP[module_name]
        kwargs = {"debug": debug, **MODULE_KWARGS[module_name]}

        try:
            processed_df = fn(processed_df, **kwargs)
            if processed_df is None or processed_df.empty:
                logger.error(f"Module '{module_name}' returned no data; aborting.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in '{module_name}': {e}", exc_info=debug)
            # on debug you want an empty DF so you see the blow-up; otherwise keep what you have
            return pd.DataFrame() if debug else processed_df

    logger.info(f"Feature pipeline done — final shape: {processed_df.shape}")
    return processed_df

