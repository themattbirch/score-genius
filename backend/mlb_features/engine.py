# backend/mlb_features/engine.py

import logging
import time
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rest import transform as rest_transform
from .season import transform as season_transform
from .rolling import transform as rolling_transform
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .advanced import transform as advanced_transform
from .utils import DEFAULTS as MLB_DEFAULTS

logger = logging.getLogger(__name__)

DEFAULT_ORDER: List[str] = ["rest", "season", "rolling", "form", "h2h", "advanced"]
TRANSFORMS: Dict[str, Any] = {
    "rest": rest_transform,
    "season": season_transform,
    "rolling": rolling_transform,
    "form": form_transform,
    "h2h": h2h_transform,
    "advanced": advanced_transform,
}

def normalize_team_name(team_id: Any) -> str:
    """
    Normalize any team identifier (int, float, str, e.g. '10.0', 10, 'LAD') into the
    canonical key used in MLB_DEFAULTS. Falls back to 'unknown_team'.
    """
    # 1) Null or NaN → unknown
    if team_id is None or (isinstance(team_id, float) and pd.isna(team_id)):
        return "unknown_team"

    # 2) Stringify and strip whitespace
    team_str = str(team_id).strip()

    # 3) Remove trailing '.0' from floats represented as strings
    if team_str.lower().endswith(".0"):
        team_str = team_str[:-2]

    # 4) Try key variants: as-is, lowercase, uppercase
    for key in (team_str, team_str.lower(), team_str.upper()):
        if key in MLB_DEFAULTS:
            return MLB_DEFAULTS[key]

    # 5) Nothing matched → unknown
    return "unknown_team"

def _supports_kwarg(func: Any, kw: str) -> bool:
    from inspect import signature
    try:
        return kw in signature(func).parameters
    except (ValueError, TypeError):
        return False

def run_mlb_feature_pipeline(
    df: pd.DataFrame,
    *,
    mlb_historical_games_df: Optional[pd.DataFrame] = None,
    mlb_historical_team_stats_df: Optional[pd.DataFrame] = None,
    rolling_window_sizes: List[int] = [15, 30, 60, 100],
    h2h_max_games: int = 10,
    execution_order: List[str] = DEFAULT_ORDER,
    flag_imputations: bool = True,
    debug: bool = False,
    **extra_kwargs,
) -> pd.DataFrame:
    """
    Orchestrates all MLB feature transforms in the given execution_order,
    merging new columns season by season.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    if df is None or df.empty:
        logger.error("Input DataFrame is empty")
        return pd.DataFrame()

    # Prepare working copies
    hist_games = mlb_historical_games_df.copy() if mlb_historical_games_df is not None else pd.DataFrame()
    team_stats = mlb_historical_team_stats_df.copy() if mlb_historical_team_stats_df is not None else pd.DataFrame()
    working_df = df.copy()

    # Ensure date parsing
    for d in (hist_games, working_df):
        if not d.empty:
            date_col = "game_date_et" if "game_date_et" in d else "game_date"
            d["game_date_et"] = pd.to_datetime(d[date_col], errors="coerce")
    if "season" not in working_df.columns:
        working_df["season"] = working_df["game_date_et"].dt.year

    all_seasons = sorted(working_df["season"].dropna().unique().astype(int))
    processed_chunks: List[pd.DataFrame] = []
    start_time = time.time()

    for season in all_seasons:
        logger.info(f"ENGINE: Processing season {season}...")
        season_df = working_df[working_df["season"] == season].copy()
        cutoff = season_df["game_date_et"].min()
        hist_context = hist_games[hist_games["game_date_et"] < cutoff] if pd.notna(cutoff) else pd.DataFrame()

        for module_name in execution_order:
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                continue

            logger.debug(f"ENGINE: Running '{module_name}' for season {season}")
            kwargs: Dict[str, Any] = {}

            # Common flags
            if _supports_kwarg(fn, "debug"):
                kwargs["debug"] = debug
            if flag_imputations and _supports_kwarg(fn, "flag_imputations"):
                kwargs["flag_imputations"] = flag_imputations

            # Module‐specific args
            if module_name == "rolling":
                if _supports_kwarg(fn, "window_sizes"):
                    kwargs["window_sizes"] = rolling_window_sizes
                data_in = pd.concat([hist_context, season_df], ignore_index=True)
                try:
                    out = fn(data_in, **kwargs)
                    # restrict back to this season
                    ids = season_df["game_id"].astype(str)
                    season_df = out[out["game_id"].astype(str).isin(ids)].copy()
                except Exception as e:
                    logger.error(f"ENGINE: rolling error: {e}", exc_info=True)
                continue

            if module_name == "h2h":
                if _supports_kwarg(fn, "historical_df"):
                    kwargs["historical_df"] = hist_context
                if _supports_kwarg(fn, "max_games"):
                    kwargs["max_games"] = h2h_max_games

            if module_name in ("season", "advanced"):
                if _supports_kwarg(fn, "team_stats_df"):
                    kwargs["team_stats_df"] = team_stats
                if _supports_kwarg(fn, "historical_team_stats_df"):
                    kwargs["historical_team_stats_df"] = team_stats
                if _supports_kwarg(fn, "season_to_lookup"):
                    kwargs["season_to_lookup"] = season - 1

                if module_name == "advanced":
                    home_col = ("home_starter_pitcher_handedness"
                                if "home_starter_pitcher_handedness" in season_df.columns
                                else "home_probable_pitcher_handedness")
                    away_col = ("away_starter_pitcher_handedness"
                                if "away_starter_pitcher_handedness" in season_df.columns
                                else "away_probable_pitcher_handedness")
                    if _supports_kwarg(fn, "home_hand_col_param"):
                        kwargs["home_hand_col_param"] = home_col
                    if _supports_kwarg(fn, "away_hand_col_param"):
                        kwargs["away_hand_col_param"] = away_col
                    if _supports_kwarg(fn, "home_team_col_param"):
                        kwargs["home_team_col_param"] = "home_team_id"
                    if _supports_kwarg(fn, "away_team_col_param"):
                        kwargs["away_team_col_param"] = "away_team_id"

            # Execute transform
            try:
                out = fn(season_df, **kwargs)
                if out is None or out.empty:
                    continue
            except Exception as e:
                logger.error(f"ENGINE: {module_name} error: {e}", exc_info=True)
                continue

            # Merge only brand-new columns
            new_cols = [c for c in out.columns if c not in season_df.columns]
            if new_cols:
                season_df = pd.concat(
                    [season_df.reset_index(drop=True),
                     out[new_cols].reset_index(drop=True)],
                    axis=1
                )

        processed_chunks.append(season_df)

    if not processed_chunks:
        logger.error("ENGINE: No chunks processed")
        return pd.DataFrame()

    final_df = pd.concat(processed_chunks, ignore_index=True)
    logger.info(f"ENGINE: Finished in {time.time() - start_time:.2f}s; final shape {final_df.shape}")
    return final_df
