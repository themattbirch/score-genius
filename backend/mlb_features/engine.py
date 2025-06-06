import logging
import time
from inspect import signature
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rest import transform as rest_transform
from .season import transform as season_transform
from .rolling import transform as rolling_transform
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .advanced import transform as advanced_transform

logger = logging.getLogger(__name__)

DEFAULT_ORDER: List[str] = [
    "rest",
    "season",
    "rolling",
    "form",
    "h2h",
    "advanced",
]

TRANSFORMS: Dict[str, Any] = {
    "rest": rest_transform,
    "season": season_transform,
    "rolling": rolling_transform,
    "form": form_transform,
    "h2h": h2h_transform,
    "advanced": advanced_transform,
}


def _supports_kwarg(func: Any, kw: str) -> bool:
    try:
        return kw in signature(func).parameters
    except (TypeError, ValueError):
        return True


def run_mlb_feature_pipeline(
    df: pd.DataFrame,
    *,
    mlb_historical_games_df: Optional[pd.DataFrame] = None,
    mlb_historical_team_stats_df: Optional[pd.DataFrame] = None,
    rolling_window_sizes: List[int] = [15, 30, 60, 100],
    form_home_col: str = "home_current_form",
    form_away_col: str = "away_current_form",
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

    required_cols = ["game_date_et", "home_team_id", "away_team_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return df

    original_df_full_scope = df.copy()

    if "game_date_et" not in original_df_full_scope.columns:
        if "game_date" in original_df_full_scope.columns:
            original_df_full_scope["game_date_et"] = pd.to_datetime(
                original_df_full_scope["game_date"], errors="coerce"
            )
        else:
            logger.error("ENGINE: Neither 'game_date_et' nor 'game_date' found. Cannot proceed.")
            return original_df_full_scope
    else:
        original_df_full_scope["game_date_et"] = pd.to_datetime(
            original_df_full_scope["game_date_et"], errors="coerce"
        )

    if "season" not in original_df_full_scope.columns and "game_date_et" in original_df_full_scope.columns:
        original_df_full_scope["season"] = original_df_full_scope["game_date_et"].dt.year

    original_df_full_scope["season"] = pd.to_numeric(
        original_df_full_scope["season"], errors="coerce"
    )
    original_df_full_scope.dropna(
        subset=["season", "game_date_et", "home_team_id", "away_team_id"], inplace=True
    )
    if original_df_full_scope.empty:
        logger.error("Input DataFrame empty after initial NaN drop for critical columns.")
        return pd.DataFrame()
    original_df_full_scope["season"] = original_df_full_scope["season"].astype(int)

    if "_row_id_pipeline" not in original_df_full_scope.columns:
        original_df_full_scope["_row_id_pipeline"] = np.arange(len(original_df_full_scope))

    if "orig_game_id" not in original_df_full_scope.columns and "game_id" in original_df_full_scope.columns:
        original_df_full_scope["orig_game_id"] = original_df_full_scope["game_id"]
    elif "orig_game_id" not in original_df_full_scope.columns:
        logger.warning(
            "ENGINE: 'orig_game_id' and 'game_id' missing. Creating placeholder orig_game_id."
        )
        original_df_full_scope["orig_game_id"] = "gid_" + original_df_full_scope["_row_id_pipeline"].astype(str)

    for id_col in ["game_id", "orig_game_id", "home_team_id", "away_team_id"]:
        if id_col in original_df_full_scope.columns:
            original_df_full_scope[id_col] = original_df_full_scope[id_col].astype(str)

    KEEP_COLS_ALWAYS = [
        "orig_game_id",
        "home_team_id",
        "away_team_id",
        "_row_id_pipeline",
        "home_starter_pitcher_handedness",
        "away_starter_pitcher_handedness",
    ]
    for col in ["home_starter_pitcher_handedness", "away_starter_pitcher_handedness"]:
        if col not in original_df_full_scope.columns:
            logger.warning(
                f"ENGINE: Expected column '{col}' for KEEP_COLS_ALWAYS not found on initial df. "
                "Will not be restored if dropped."
            )

    historical_games_df_context = None
    if mlb_historical_games_df is not None and not mlb_historical_games_df.empty:
        historical_games_df_context = mlb_historical_games_df
        if "game_date_et" not in historical_games_df_context.columns and "game_date" in historical_games_df_context.columns:
            historical_games_df_context["game_date_et"] = pd.to_datetime(
                historical_games_df_context["game_date"], errors="coerce"
            )
        elif "game_date_et" in historical_games_df_context.columns:
            historical_games_df_context["game_date_et"] = pd.to_datetime(
                historical_games_df_context["game_date_et"], errors="coerce"
            )
        else:
            historical_games_df_context = None

        if historical_games_df_context is not None:
            outcome_cols_hist = [
                "home_score", "away_score", "home_hits", "away_hits", "home_errors", "away_errors"
            ]
            for col in outcome_cols_hist:
                if col not in historical_games_df_context.columns:
                    historical_games_df_context[col] = np.nan
            if "game_id" in historical_games_df_context.columns:
                historical_games_df_context["game_id"] = historical_games_df_context["game_id"].astype(str)

    team_stats_df_context = None
    if mlb_historical_team_stats_df is not None and not mlb_historical_team_stats_df.empty:
        team_stats_df_context = mlb_historical_team_stats_df
        if "season" in team_stats_df_context.columns:
            team_stats_df_context["season"] = pd.to_numeric(
                team_stats_df_context["season"], errors="coerce"
            ).astype("Int64")
            team_stats_df_context.dropna(subset=["season"], inplace=True)
            team_stats_df_context["season"] = team_stats_df_context["season"].astype(int)
        else:
            logger.warning("ENGINE: 'season' column missing in mlb_historical_team_stats_df.")

    all_processed_season_chunks: List[pd.DataFrame] = []
    pipeline_start_time = time.time()

    for game_season in sorted(original_df_full_scope["season"].unique()):
        logger.info(f"ENGINE: —— Processing games for input df season {game_season} ——")
        current_season_chunk = original_df_full_scope[
            original_df_full_scope["season"] == game_season
        ].copy()
        if current_season_chunk.empty:
            continue

        loop_chunk_for_season = current_season_chunk.copy()

        for module_name in execution_order:
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                logger.warning(f"ENGINE: Unknown module '{module_name}' in execution_order. Skipping.")
                continue

            input_to_transform_fn = loop_chunk_for_season
            kwargs_for_module: Dict[str, Any] = {"debug": debug}

            if module_name == "season":
                if _supports_kwarg(fn, "team_stats_df"):
                    kwargs_for_module["team_stats_df"] = team_stats_df_context
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            ## FIX: Simplified rolling block
            elif module_name == "rolling":
                # Prepare a temporary DataFrame with the upcoming games for this season
                # ensuring the outcome columns exist but are NaN.
                temp_chunk_with_nan_outcomes = loop_chunk_for_season.copy()
                outcome_cols = ["home_score", "away_score", "home_hits", "away_hits", "home_errors", "away_errors"]
                for col in outcome_cols:
                    if col not in temp_chunk_with_nan_outcomes.columns:
                        temp_chunk_with_nan_outcomes[col] = np.nan

                # Prepare the historical context for the rolling calculation
                input_df_for_rolling = temp_chunk_with_nan_outcomes
                if (
                    historical_games_df_context is not None
                    and not historical_games_df_context.empty
                    and "game_date_et" in historical_games_df_context.columns
                ):
                    # Find the earliest game date in the current chunk to select only past games
                    cutoff = loop_chunk_for_season["game_date_et"].min()
                    if pd.notna(cutoff):
                        historical_games_df_context["game_date_et"] = pd.to_datetime(
                            historical_games_df_context["game_date_et"], errors="coerce"
                        )
                        relevant_hist_for_rolling = historical_games_df_context[
                            historical_games_df_context["game_date_et"] < cutoff
                        ].copy()

                        # Combine historical context with the upcoming games chunk
                        if not relevant_hist_for_rolling.empty:
                            all_concat_cols = sorted(
                                set(relevant_hist_for_rolling.columns) | set(temp_chunk_with_nan_outcomes.columns)
                            )
                            input_df_for_rolling = pd.concat(
                                [
                                    relevant_hist_for_rolling.reindex(columns=all_concat_cols),
                                    temp_chunk_with_nan_outcomes.reindex(columns=all_concat_cols),
                                ],
                                ignore_index=True,
                                sort=False,
                            )
                
                # Call the rolling transform on the combined dataframe
                kwargs_for_module.update({
                    "window_sizes": rolling_window_sizes,
                    "flag_imputations": flag_imputations,
                })
                processed_data_from_module = fn(input_df_for_rolling, **kwargs_for_module)

                if processed_data_from_module is None or processed_data_from_module.empty:
                    logger.warning(f"ENGINE (ROLLING): Rolling module returned empty DataFrame for season {game_season}.")
                    continue 

                # The `processed_data_from_module` now contains both historical and upcoming games with rolling stats.
                # We only need the rows that correspond to our original upcoming games.
                
                # Ensure game_id types match for filtering
                processed_data_from_module["game_id"] = processed_data_from_module["game_id"].astype(str)
                original_game_ids_for_season = loop_chunk_for_season["game_id"].astype(str).unique()

                # Filter the results to get just the upcoming games we care about
                upcoming_games_with_rolling_stats = processed_data_from_module[
                    processed_data_from_module["game_id"].isin(original_game_ids_for_season)
                ].copy()
                
                # Identify only the new columns created by the rolling module
                new_rolling_cols = [
                    col for col in upcoming_games_with_rolling_stats.columns
                    if col not in loop_chunk_for_season.columns and col not in KEEP_COLS_ALWAYS
                ]

                # Merge the new rolling features back into the main chunk for this season
                if not upcoming_games_with_rolling_stats.empty and new_rolling_cols:
                    loop_chunk_for_season = loop_chunk_for_season.merge(
                        upcoming_games_with_rolling_stats[['game_id'] + new_rolling_cols],
                        on="game_id",
                        how="left"
                    )
                    logger.debug(f"ENGINE (ROLLING): Successfully merged {len(new_rolling_cols)} rolling features for season {game_season}.")
                
                # Since we've handled the merge, continue to the next module in the execution order
                continue

            elif module_name == "h2h":
                kwargs_for_module["historical_df"] = historical_games_df_context
                kwargs_for_module["max_games"] = h2h_max_games
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            elif module_name == "advanced":
                if team_stats_df_context is None or team_stats_df_context.empty:
                    logger.warning(
                        f"Skipping 'advanced' module for season {game_season} (no team stats)."
                    )
                    continue

                ## FIX: Add this debug line to inspect the dataframe columns
                logger.debug(f"ENGINE_ADV_DEBUG: Columns in team_stats_df_context before passing to advanced.transform: {team_stats_df_context.columns.tolist()}")

                kwargs_for_module.update({
                    "home_team_col_param": "home_team_id",
                    "away_team_col_param": "away_team_id",
                    "home_hand_col_param": "home_starter_pitcher_handedness",
                    "away_hand_col_param": "away_starter_pitcher_handedness",
                    "flag_imputations": flag_imputations,
                    "historical_team_stats_df": team_stats_df_context,
                })

                if _supports_kwarg(fn, "season_to_lookup"):
                    lookup_season_for_advanced = game_season - 1
                    kwargs_for_module["season_to_lookup"] = lookup_season_for_advanced
                
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            else:
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            if processed_data_from_module is None or processed_data_from_module.empty:
                continue

            if "game_id" not in processed_data_from_module.columns:
                continue

            processed_data_from_module["game_id"] = processed_data_from_module["game_id"].astype(str)
            upcoming_game_ids_for_season = current_season_chunk["game_id"].astype(str).unique()

            processed_upcoming = processed_data_from_module[
                processed_data_from_module["game_id"].isin(upcoming_game_ids_for_season)
            ].copy()

            existing_cols = set(loop_chunk_for_season.columns)
            new_cols = [
                col for col in processed_upcoming.columns
                if col not in existing_cols and col not in KEEP_COLS_ALWAYS
            ]

            if new_cols:
                to_merge = processed_upcoming[["game_id"] + new_cols]
                loop_chunk_for_season = loop_chunk_for_season.merge(
                    to_merge, on="game_id", how="left"
                )

        if not loop_chunk_for_season.empty:
            cols_to_restore = [
                col
                for col in KEEP_COLS_ALWAYS
                if col in original_df_full_scope.columns and col not in loop_chunk_for_season.columns
            ]
            if cols_to_restore and "_row_id_pipeline" in loop_chunk_for_season.columns:
                restore_data = original_df_full_scope[
                    ["_row_id_pipeline"] + cols_to_restore
                ].drop_duplicates(subset=["_row_id_pipeline"])
                loop_chunk_for_season = loop_chunk_for_season.merge(
                    restore_data, on="_row_id_pipeline", how="left"
                )

            all_processed_season_chunks.append(loop_chunk_for_season)
        else:
            logger.warning(f"ENGINE: Season {game_season} resulted in an empty chunk. Excluding.")

    if not all_processed_season_chunks:
        logger.error("No data chunks processed. Returning empty DataFrame.")
        return pd.DataFrame()

    final_df = pd.concat(all_processed_season_chunks, ignore_index=True, sort=False)
    
    if "_row_id_pipeline" in final_df.columns and "_row_id_pipeline" in original_df_full_scope.columns:
        merge_cols_from_original = [
            col
            for col in KEEP_COLS_ALWAYS
            if col != "_row_id_pipeline" and col in original_df_full_scope.columns
        ]

        final_df_for_merge_final = final_df.drop(
            columns=[col for col in merge_cols_from_original if col in final_df.columns],
            errors="ignore",
        )
        final_df = final_df_for_merge_final.merge(
            original_df_full_scope[
                ["_row_id_pipeline"] + merge_cols_from_original
            ].drop_duplicates(subset=["_row_id_pipeline"]),
            on="_row_id_pipeline",
            how="left",
        )
    else:
        logger.warning(
            "ENGINE: _row_id_pipeline missing for final KEEP_COLS_ALWAYS merge. Skipping this merge."
        )

    if "_row_id_pipeline" in final_df.columns:
        final_df = final_df.drop(columns=["_row_id_pipeline"])

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"ENGINE: Feature pipeline complete in {pipeline_duration:.2f}s — final shape {final_df.shape}")

    if debug:
        logger.setLevel(logging.INFO)
    return final_df