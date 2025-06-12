# backend/mlb_features/engine.py

import logging
import time
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from inspect import signature

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
    if debug:
        logger.setLevel(logging.DEBUG)

    if df is None or df.empty:
        logger.error("Input DataFrame is empty")
        return pd.DataFrame()

    # Prepare working copies
    hist_games = mlb_historical_games_df.copy() if mlb_historical_games_df is not None else pd.DataFrame()
    team_stats = mlb_historical_team_stats_df.copy() if mlb_historical_team_stats_df is not None else pd.DataFrame()
    working_df = df.copy()

    for df_key, df_obj_ref in {"hist_games": hist_games, "working_df": working_df}.items():
        # Make a modifiable copy if it's not already one (important for loop iterators)
        current_df_obj = df_obj_ref.copy() if not df_obj_ref.empty else pd.DataFrame() 

        if not current_df_obj.empty:
            target_date_col_name = "game_date_et"
            source_date_col_name = None

            # Prioritize existing 'game_date_et' if it's already the standardized one
            if target_date_col_name in current_df_obj.columns:
                source_date_col_name = target_date_col_name
            # Then check for 'game_date_time_utc' which is common in mlb_historical_game_stats
            elif "game_date_time_utc" in current_df_obj.columns:
                source_date_col_name = "game_date_time_utc"
            # Fallback to a generic 'game_date' if it exists (less common for MLB raw data)
            elif "game_date" in current_df_obj.columns:
                source_date_col_name = "game_date"
            
            if source_date_col_name:
                # perform robust datetime parsing, handling timezone awareness
                dt_series = pd.to_datetime(current_df_obj[source_date_col_name], errors='coerce')

                # If it's naive, localize entire series to UTC, then convert to ET
                if dt_series.dt.tz is None:
                    dt_series = dt_series.dt.tz_localize('UTC')
                dt_series = dt_series.dt.tz_convert('America/New_York')

                # extract just the date part
                current_df_obj[target_date_col_name] = dt_series.dt.normalize()


                # drop the original source if it isn’t the same
                if source_date_col_name != target_date_col_name and source_date_col_name in current_df_obj.columns:
                    current_df_obj = current_df_obj.drop(columns=[source_date_col_name])
            else:
                logger.warning(f"ENGINE: No suitable date column found for DF {df_key}. Date parsing skipped. Ensure '{target_date_col_name}' is available downstream.")
                # Ensure the target column exists as NaT if no source was found
                current_df_obj[target_date_col_name] = pd.NaT 
        else:
            # If the DataFrame is empty, ensure the target_date_col_name is present as a column
            # so subsequent feature modules don't hit KeyError
            if target_date_col_name not in current_df_obj.columns:
                 current_df_obj[target_date_col_name] = pd.Series(dtype='object') # Create empty column

        # Update the original DataFrame reference outside the loop, as DataFrames are passed by reference
        # when working_df or hist_games is mutable from outside.
        # This explicit assignment ensures changes are reflected back.
        if df_key == "hist_games":
            hist_games = current_df_obj
        elif df_key == "working_df":
            working_df = current_df_obj
    if "season" not in working_df.columns:
        working_df["season"] = working_df["game_date_et"].dt.year

    all_seasons = sorted(working_df["season"].dropna().unique().astype(int))
    processed_chunks: List[pd.DataFrame] = []
    start_time = time.time()

    for season in all_seasons:
        logger.info(f"ENGINE: Processing season {season}...")
        season_df = working_df[working_df["season"] == season].copy()
        cutoff = season_df["game_date_et"].min()
        hist_context = (
            hist_games[hist_games["game_date_et"] < cutoff]
            if pd.notna(cutoff)
            else pd.DataFrame()
        )
        # DEBUG: list out modules
        logger.debug(f"Modules in execution_order: {execution_order}")

        # ----- per‐module dispatch -----
        for module_name in execution_order:
            logger.debug(f"→ about to run module: {module_name}")
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                logger.debug(f"   ✗ no transform found for {module_name}")
                continue

            # ✨ DEBUG: see exactly what params this transform expects
            logger.debug(
                f"{module_name} signature → "
                f"{list(signature(fn).parameters.keys())}"
            )

            logger.debug(f"ENGINE: Running '{module_name}' for season {season}")
            kwargs: Dict[str, Any] = {}

            # common flags
            if _supports_kwarg(fn, "debug"):
                kwargs["debug"] = debug
            if flag_imputations and _supports_kwarg(fn, "flag_imputations"):
                kwargs["flag_imputations"] = flag_imputations

            # ── rolling ──
            if module_name == "rolling":
                if _supports_kwarg(fn, "window_sizes"):
                    kwargs["window_sizes"] = rolling_window_sizes
                data_in = pd.concat([hist_context, season_df], ignore_index=True)
                out = fn(data_in, **kwargs)
                season_df = out[
                    out["game_id"].astype(str).isin(season_df["game_id"].astype(str))
                ].copy()
                continue

            # ── h2h ──
            if module_name == "h2h":
                if _supports_kwarg(fn, "historical_df"):
                    kwargs["historical_df"] = hist_context
                if _supports_kwarg(fn, "max_games"):
                    kwargs["max_games"] = h2h_max_games

                out = fn(season_df, **kwargs)
                if out is not None and not out.empty:
                    season_df = out
                continue

            # ── season & advanced ──
            if module_name in ("season", "advanced"):
                lookup = season
                prev_stats = team_stats[team_stats["season"] == lookup]
                hist_stats = team_stats[team_stats["season"] <= lookup]
                sig = signature(fn).parameters

                if module_name == "season":
                    if "team_stats_df" in sig:
                        kwargs["team_stats_df"] = prev_stats
                    logger.debug(f"season uses stats = {kwargs['team_stats_df']['season'].unique()}")

                if module_name == "advanced":
                    if "historical_team_stats_df" in sig:
                        kwargs["historical_team_stats_df"] = prev_stats
                    if "season_to_lookup" in sig:
                        kwargs["season_to_lookup"] = lookup
                    # handedness & team‐col params
                    hc = (
                        "home_starter_pitcher_handedness"
                        if "home_starter_pitcher_handedness" in season_df.columns
                        else "home_probable_pitcher_handedness"
                    )
                    ac = (
                        "away_starter_pitcher_handedness"
                        if "away_starter_pitcher_handedness" in season_df.columns
                        else "away_probable_pitcher_handedness"
                    )
                    if "home_hand_col_param" in sig:
                        kwargs["home_hand_col_param"] = hc
                    if "away_hand_col_param" in sig:
                        kwargs["away_hand_col_param"] = ac
                    if "home_team_col_param" in sig:
                        kwargs["home_team_col_param"] = "home_team_id"
                    if "away_team_col_param" in sig:
                        kwargs["away_team_col_param"] = "away_team_id"

                    logger.debug(
                        f"advanced will run with historical_team_stats_df seasons = "
                        f"{kwargs['historical_team_stats_df']['season'].unique()}"
                    )

                out = fn(season_df, **kwargs)
                if out is None or out.empty:
                    continue

                new = [c for c in out.columns if c not in season_df.columns]
                if new:
                    season_df = pd.concat(
                        [
                            season_df.reset_index(drop=True),
                            out[new].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                continue

        # after all modules, keep this season’s chunk
        processed_chunks.append(season_df)


    if not processed_chunks:
        logger.error("ENGINE: No chunks processed")
        return pd.DataFrame()

    final_df = pd.concat(processed_chunks, ignore_index=True)
    logger.info(f"ENGINE: Finished in {time.time() - start_time:.2f}s; final shape {final_df.shape}")
    return final_df