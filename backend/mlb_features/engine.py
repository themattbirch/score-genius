# backend/mlb_features/engine.py

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np # Import numpy for calculations
from inspect import signature

# Import all transform functions
from .rest import transform as rest_transform
from .season import transform as season_transform
from .rolling import transform as rolling_transform
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .advanced import transform as advanced_transform, _precompute_two_season_aggregates
from .handedness_for_display import transform as handedness_transform

# Import shared utilities
from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)

DEFAULT_ORDER: List[str] = [
    "rest", "season", "rolling", "form", "h2h",
    "advanced", "handedness_for_display",
]

TRANSFORMS: Dict[str, Any] = {
    "rest": rest_transform, "season": season_transform, "rolling": rolling_transform,
    "form": form_transform, "h2h": h2h_transform, "advanced": advanced_transform,
    "handedness_for_display": handedness_transform
}

def _supports_kwarg(func: Any, kw: str) -> bool:
    try:
        return kw in signature(func).parameters
    except (ValueError, TypeError):
        return False

def run_mlb_feature_pipeline(
    df: pd.DataFrame,
    *,
    seasonal_splits_data: Optional[pd.DataFrame] = None,
    precomputed_rolling_features_df: Optional[pd.DataFrame] = None,
    mlb_historical_games_df: Optional[pd.DataFrame] = None,
    mlb_historical_team_stats_df: Optional[pd.DataFrame] = None,
    rolling_window_sizes: List[int] = [15, 30, 60, 100],
    h2h_max_games: int = 10,
    num_form_games: int = 5,
    execution_order: List[str] = DEFAULT_ORDER,
    flag_imputations: bool = True,
    debug: bool = False,
    keep_display_only_features: bool = False
) -> pd.DataFrame:
    
    if df is None or df.empty:
        logger.warning("engine.run_mlb_feature_pipeline: Input DataFrame is empty. Aborting.")
        return pd.DataFrame()
            
    required_base_cols = {'game_id', 'home_team_id', 'away_team_id'}
    date_cols = {'game_date_et', 'game_date', 'game_date_time_utc'}

    missing_base = required_base_cols - set(df.columns)
    if missing_base:
        logger.error(f"engine.run_mlb_feature_pipeline: Input 'df' missing required base columns: {list(missing_base)}. Aborting.")
        return df

    if not any(col in df.columns for col in date_cols):
        logger.error(f"engine.run_mlb_feature_pipeline: Input 'df' missing a required date column (e.g., game_date_et or game_date). Aborting.")
        return df

    if debug:
        logger.setLevel(logging.DEBUG)
    if df is None or df.empty:
        return pd.DataFrame()

    hist_games = mlb_historical_games_df.copy() if mlb_historical_games_df is not None else pd.DataFrame()
    team_stats = mlb_historical_team_stats_df.copy() if mlb_historical_team_stats_df is not None else pd.DataFrame()
    working_df = df.copy()


    logger.info("ENGINE: Normalizing team identifiers and dates in all dataframes...")
    for df_ref in [hist_games, team_stats, working_df]:
        if df_ref.empty:
            continue
        if 'home_team_id' in df_ref.columns:
            df_ref['home_team_norm'] = df_ref['home_team_id'].apply(normalize_team_name)
        if 'away_team_id' in df_ref.columns:
            df_ref['away_team_norm'] = df_ref['away_team_id'].apply(normalize_team_name)
        if 'team_id' in df_ref.columns:
            df_ref['team_norm'] = df_ref['team_id'].apply(normalize_team_name)
        source_date_col = next((col for col in ['game_date_et', 'game_date_time_utc', 'game_date'] if col in df_ref.columns), None)
        if source_date_col:
            dt_series = pd.to_datetime(df_ref[source_date_col], errors='coerce')
            if dt_series.dt.tz is None:
                dt_series = dt_series.dt.tz_localize('UTC')
            df_ref['game_date_et'] = dt_series.dt.tz_convert('America/New_York').dt.normalize()

    logger.info("ENGINE: Standardizing game_id to string to prevent dtype conflicts.")
    if not working_df.empty and 'game_id' in working_df.columns:
        working_df['game_id'] = working_df['game_id'].astype(str)
    if not hist_games.empty and 'game_id' in hist_games.columns:
        hist_games['game_id'] = hist_games['game_id'].astype(str)

    if "season" not in working_df.columns and "game_date_et" in working_df.columns:
        working_df["season"] = working_df["game_date_et"].dt.year
    elif "season" not in working_df.columns:
        working_df["season"] = pd.Timestamp.now().year

    logger.info("ENGINE: Pre-computing 2-season aggregate stats for all available data...")
    precomputed_adv_stats = _precompute_two_season_aggregates(team_stats)
    for col in ("team_norm", "season"):
        if col in precomputed_adv_stats:
            precomputed_adv_stats[col] = precomputed_adv_stats[col].astype("category")

    start_time = time.time()
    processed_chunks: List[pd.DataFrame] = []

    for season in sorted(working_df["season"].dropna().unique().astype(int)):
        logger.info(f"ENGINE: Processing season {season}…")
        # 1) slice out this season’s games
        season_df = working_df[working_df["season"] == season].copy()
        season_start = season_df["game_date_et"].min()
        raw_hist_games = (
            hist_games[hist_games["game_date_et"] < season_start]
            if not hist_games.empty else pd.DataFrame()
        )

        # 2) run each module in execution_order exactly once
        for module_name in execution_order:
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                continue

            # build kwargs
            kwargs: Dict[str, Any] = {}
            if _supports_kwarg(fn, "debug"):
                kwargs["debug"] = debug
            if _supports_kwarg(fn, "flag_imputations"):
                kwargs["flag_imputations"] = flag_imputations

            # prepare input_df
            input_df = season_df
            if module_name in ("rest", "rolling"):
                input_df = pd.concat([raw_hist_games, season_df], ignore_index=True)
                if module_name == "rolling":
                    kwargs["window_sizes"] = rolling_window_sizes
                    if precomputed_rolling_features_df is not None:
                        kwargs["precomputed_rolling_features_df"] = precomputed_rolling_features_df
                if module_name == "rest":
                    kwargs["historical_df"] = raw_hist_games
            elif module_name in ("h2h", "form"):
                kwargs["historical_df"] = raw_hist_games
                if module_name == "h2h":
                    kwargs["max_games"] = h2h_max_games
                if module_name == "form":
                    kwargs["num_form_games"] = num_form_games
            elif module_name == "season":
                # Prioritize using the new RPC data for seasonal stats if it was provided
                if seasonal_splits_data is not None and not seasonal_splits_data.empty:
                    kwargs["historical_team_stats_df"] = seasonal_splits_data[seasonal_splits_data["season"] == (season - 1)].copy()
                # Fallback to the old method if RPC data is not available
                else:
                    kwargs["historical_team_stats_df"] = (
                        team_stats[team_stats["season"] == (season - 1)].copy()
                        if not team_stats.empty else pd.DataFrame()
                    )
            elif module_name == "advanced":
                kwargs["precomputed_stats"] = precomputed_adv_stats
            elif module_name == "handedness_for_display":
                kwargs["historical_team_stats_df"] = (
                    team_stats[team_stats["season"] == season]
                    if not team_stats.empty else pd.DataFrame()
                )

            # call the transform
            output_df = fn(input_df, **kwargs)

            # HALT EARLY if empty
            if isinstance(output_df, pd.DataFrame) and output_df.empty:
                logger.error(
                    f"ENGINE: DataFrame became empty after module '{module_name}'. Halting pipeline for this season."
                )
                break

            # merge output_df back into season_df
            if module_name in ("rest", "rolling"):
                season_df = output_df[output_df["game_id"].isin(season_df["game_id"])].copy()
            else:
                produced = [
                    c for c in output_df.columns
                    if c not in season_df.columns and c != "game_id"
                ]
                if produced:
                    season_df = season_df.merge(
                        output_df[["game_id", *produced]].drop_duplicates(subset="game_id"),
                        on="game_id",
                        how="left"
                    )

            # drop display-only hand-split metric after advanced
            if module_name == "advanced" and not keep_display_only_features:
                season_df.drop(
                    columns=["h_team_off_avg_runs_vs_opp_hand"],
                    inplace=True,
                    errors="ignore"
                )

            # optional debug logging…
            if debug:
                logger.debug(f"--- ENGINE Debug for module {module_name} ---")
                # … your existing debug-report code …

        # 3) dedupe columns & collect this season’s result
        season_df = season_df.loc[:, ~season_df.columns.duplicated()].reset_index(drop=True)
        processed_chunks.append(season_df)

    if not processed_chunks:
        return pd.DataFrame()

    final_df = pd.concat(processed_chunks, ignore_index=True)
    # drop legacy display-only columns
    if not keep_display_only_features:
        logger.info("ENGINE: Dropping display-only columns from final dataframe.")
        final_df.drop(
            columns=[
                "h_team_off_avg_runs_vs_opp_hand",
                "a_team_off_avg_runs_vs_opp_hand"
            ],
            inplace=True,
            errors="ignore"
        )

    logger.info(f"ENGINE: Finished in {time.time() - start_time:.2f}s; final shape {final_df.shape}")
    return final_df