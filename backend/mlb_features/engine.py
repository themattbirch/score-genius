# backend/mlb_features/engine.py

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
from .momentum import transform as momentum_transform
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
    momentum_num_innings: int = 9,
    execution_order: List[str] = DEFAULT_ORDER,
    flag_imputations: bool = True,
    debug: bool = False,
    **extra_kwargs,
) -> pd.DataFrame:

    if debug:
        logger.setLevel(logging.DEBUG)

    # --- EARLY GUARD: empty or None input ---
    if df is None or df.empty:
        logger.error("Input DataFrame is empty")
        return pd.DataFrame()

    # --- CHECK FOR MISSING ESSENTIAL COLUMNS BEFORE ANY DROPNA ---
    required_cols = ["game_date_et", "home_team_id", "away_team_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return df

    original_df_full_scope = df.copy()

    # --- ENSURE 'game_date_et' exists, else try to derive it ---
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

    # --- DERIVE 'season' if missing ---
    if "season" not in original_df_full_scope.columns and "game_date_et" in original_df_full_scope.columns:
        original_df_full_scope["season"] = original_df_full_scope["game_date_et"].dt.year

    original_df_full_scope["season"] = pd.to_numeric(
        original_df_full_scope["season"], errors="coerce"
    )
    # Now drop rows missing any of these critical fields
    original_df_full_scope.dropna(
        subset=["season", "game_date_et", "home_team_id", "away_team_id"], inplace=True
    )
    if original_df_full_scope.empty:
        logger.error("Input DataFrame empty after initial NaN drop for critical columns.")
        return pd.DataFrame()
    original_df_full_scope["season"] = original_df_full_scope["season"].astype(int)

    # --- Add the pipeline‐internal row ID if it doesn't already exist ---
    if "_row_id_pipeline" not in original_df_full_scope.columns:
        original_df_full_scope["_row_id_pipeline"] = np.arange(len(original_df_full_scope))

    # --- Ensure 'orig_game_id' exists so we can restore at the end ---
    if "orig_game_id" not in original_df_full_scope.columns and "game_id" in original_df_full_scope.columns:
        original_df_full_scope["orig_game_id"] = original_df_full_scope["game_id"]
    elif "orig_game_id" not in original_df_full_scope.columns:
        logger.warning(
            "ENGINE: 'orig_game_id' and 'game_id' missing. Creating placeholder orig_game_id."
        )
        original_df_full_scope["orig_game_id"] = (
            "gid_" + original_df_full_scope["_row_id_pipeline"].astype(str)
        )

    # --- Force ID columns to string for consistent merging afterwards ---
    for id_col in ["game_id", "orig_game_id", "home_team_id", "away_team_id"]:
        if id_col in original_df_full_scope.columns:
            original_df_full_scope[id_col] = original_df_full_scope[id_col].astype(str)

    # ─── KEEP_COLS_ALWAYS definitions ─────────────────────
    KEEP_COLS_ALWAYS = [
        "orig_game_id",
        "home_team_id",
        "away_team_id",
        "_row_id_pipeline",
        "home_starter_pitcher_handedness",
        "away_starter_pitcher_handedness",
    ]
    # Warn if starter_handedness columns are missing (prediction.py is supposed to add them)
    for col in ["home_starter_pitcher_handedness", "away_starter_pitcher_handedness"]:
        if col not in original_df_full_scope.columns:
            logger.warning(
                f"ENGINE: Expected column '{col}' for KEEP_COLS_ALWAYS not found on initial df. "
                "Will not be restored if dropped."
            )

    # ─── PREPARE historical_games_df_context ─────────────────
    historical_games_df_context = None
    if mlb_historical_games_df is not None and not mlb_historical_games_df.empty:
        # **DO NOT COPY** here; tests check object identity
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
            # Ensure those six “outcome” columns exist for rolling
            outcome_cols_hist = [
                "home_score",
                "away_score",
                "home_hits",
                "away_hits",
                "home_errors",
                "away_errors",
            ]
            for col in outcome_cols_hist:
                if col not in historical_games_df_context.columns:
                    historical_games_df_context[col] = np.nan
            if "game_id" in historical_games_df_context.columns:
                historical_games_df_context["game_id"] = (
                    historical_games_df_context["game_id"].astype(str)
                )

    # ─── PREPARE team_stats_df_context ────────────────────
    team_stats_df_context = None
    if mlb_historical_team_stats_df is not None and not mlb_historical_team_stats_df.empty:
        # **DO NOT COPY** here; tests check object identity
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
    season_hint: Optional[int] = extra_kwargs.get("season_to_lookup")

    for game_season in sorted(original_df_full_scope["season"].unique()):
        logger.info(f"ENGINE: —— Processing games for input df season {game_season} ——")
        current_season_chunk = original_df_full_scope[
            original_df_full_scope["season"] == game_season
        ].copy()
        if current_season_chunk.empty:
            continue

        current_season_chunk_original_game_ids = current_season_chunk[
            "orig_game_id"
        ].unique()
        current_season_chunk["game_id"] = current_season_chunk["game_id"].astype(str)

        loop_chunk_for_season = current_season_chunk.copy()

        for module_name in execution_order:
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                logger.warning(
                    f"ENGINE: Unknown module '{module_name}' in execution_order. Skipping."
                )
                continue

            # Base kwargs for every module
            kwargs_for_module: Dict[str, Any] = {"debug": debug}
            input_to_transform_fn = loop_chunk_for_season

            # ─── Module‐specific preparations ───────────────────
            if module_name == "rolling":
                temp_chunk_with_nan_outcomes = loop_chunk_for_season.copy()
                outcome_cols = [
                    "home_score",
                    "away_score",
                    "home_hits",
                    "away_hits",
                    "home_errors",
                    "away_errors",
                ]
                for col in outcome_cols:
                    if col not in temp_chunk_with_nan_outcomes.columns:
                        temp_chunk_with_nan_outcomes[col] = np.nan

                if (
                    historical_games_df_context is not None
                    and not historical_games_df_context.empty
                    and "game_date_et" in historical_games_df_context.columns
                ):
                    relevant_hist_for_rolling = historical_games_df_context[
                        pd.to_datetime(historical_games_df_context["game_date_et"]).dt.year
                        <= game_season
                    ].copy()
                    if not relevant_hist_for_rolling.empty:
                        all_concat_cols = sorted(
                            list(
                                set(relevant_hist_for_rolling.columns)
                                | set(temp_chunk_with_nan_outcomes.columns)
                            )
                        )
                        input_to_transform_fn = pd.concat(
                            [
                                relevant_hist_for_rolling.reindex(
                                    columns=all_concat_cols
                                ),
                                temp_chunk_with_nan_outcomes.reindex(
                                    columns=all_concat_cols
                                ),
                            ],
                            ignore_index=True,
                            sort=False,
                        )
                    else:
                        input_to_transform_fn = temp_chunk_with_nan_outcomes
                else:
                    input_to_transform_fn = temp_chunk_with_nan_outcomes

                kwargs_for_module.update(
                    {
                        "window_sizes": rolling_window_sizes,
                        "flag_imputations": flag_imputations,
                    }
                )

            elif module_name == "momentum":
                kwargs_for_module["num_innings"] = momentum_num_innings
                kwargs_for_module["historical_df"] = historical_games_df_context

            elif module_name == "advanced":
                if team_stats_df_context is None or team_stats_df_context.empty:
                    logger.warning(
                        f"Skipping 'advanced' module for season {game_season} (no team stats)."
                    )
                    continue
                kwargs_for_module["historical_team_stats_df"] = team_stats_df_context
                                
                # --- START: EXPLICITLY SET season_to_lookup for advanced.transform ---
                if _supports_kwarg(fn, "season_to_lookup"):
                    # For upcoming games in 'game_season', use historical stats from 'game_season - 1'.
                    # This assumes 'game_season' is the current season of the games being processed.
                    lookup_season_for_advanced = game_season - 1 
                    kwargs_for_module["season_to_lookup"] = lookup_season_for_advanced
                    logger.info(f"ENGINE_ADV_CONFIG: Setting 'season_to_lookup' for advanced.transform to: {lookup_season_for_advanced}")
                else:
                    # This case should ideally not happen if advanced.transform has the param.
                    logger.warning(f"'season_to_lookup' not supported by {module_name}.transform, though expected. Advanced features might use default season logic.")
                    # If not supported, ensure 'season_to_lookup' is not in kwargs if fn can't handle it.
                    if "season_to_lookup" in kwargs_for_module:
                        del kwargs_for_module["season_to_lookup"]
                # --- END: EXPLICITLY SET season_to_lookup for advanced.transform ---

                # The original extra_kwargs.get("season_to_lookup") logic for 'advanced' module's 
                # 'season_to_lookup' is now replaced by the block above to prioritize game_season - 1.
                # If you need an override mechanism via extra_kwargs, it would need to be structured differently,
                # e.g., check extra_kwargs first and only use game_season - 1 if not provided there.
                # For now, this ensures game_season - 1 is the primary driver.

                kwargs_for_module.update(
                    {
                        "home_team_col_param": "home_team_id",
                        "away_team_col_param": "away_team_id",
                        "home_hand_col_param": "home_starter_pitcher_handedness",
                        "away_hand_col_param": "away_starter_pitcher_handedness",
                        "flag_imputations": flag_imputations,
                        # 'debug' is already in base_kwargs
                    }
                )
                # Log presence/absence of handedness columns (existing good logic)
                #logger.debug(
                   # f"ENGINE_ADV_PRE_CALL: Columns in input_to_transform_fn for 'advanced': {input_to_transform_fn.columns.tolist()}"
               # )
                for hand_col_check in [
                    "home_starter_pitcher_handedness",
                    "away_starter_pitcher_handedness",
                ]:
                    if hand_col_check not in input_to_transform_fn.columns:
                        logger.warning(
                            f"ENGINE_ADV_PRE_CALL: '{hand_col_check}' is MISSING from input to advanced.transform!"
                        )
                    else:
                        logger.debug(
                            f"ENGINE_ADV_PRE_CALL: '{hand_col_check}' IS PRESENT. Sample: \n{input_to_transform_fn[[hand_col_check]].head().to_string()}"
                        )
                
                # Log the final kwargs being passed
                logger.debug(f"ENGINE_ADV_PRE_CALL: Final Kwargs being passed to advanced.transform: {kwargs_for_module}")

            elif module_name == "season": # For the 'season' module
                kwargs_for_module["team_stats_df"] = team_stats_df_context
                # Apply a similar explicit logic for 'season' module if it also benefits from game_season - 1
                # or keep its original extra_kwargs.get("season_to_lookup") logic if that's preferred for it.
                # For consistency, let's try game_season - 1 for it too, if it supports season_to_lookup.
                if _supports_kwarg(fn, "season_to_lookup"):
                    lookup_season_for_season_mod = game_season - 1 # Or just game_season, depending on what 'season' module needs
                    kwargs_for_module["season_to_lookup"] = lookup_season_for_season_mod
                    logger.info(f"ENGINE_SEASON_CONFIG: Setting 'season_to_lookup' for {module_name}.transform to: {lookup_season_for_season_mod}")
                else:
                    # Fallback to original extra_kwargs logic if you want to maintain that flexibility for the 'season' module
                    season_hint_season_mod = extra_kwargs.get("season_to_lookup")
                    if season_hint_season_mod is not None:
                        if _supports_kwarg(fn, "season_to_lookup"): # Redundant check if outer if handles this
                           kwargs_for_module["season_to_lookup"] = season_hint_season_mod
                        # else: already logged if not supported by outer if block for advanced
                
                kwargs_for_module["flag_imputations"] = flag_imputations


            elif module_name == "h2h":
                kwargs_for_module["historical_df"] = historical_games_df_context
                kwargs_for_module["max_games"] = h2h_max_games

            # ─── Invoke and merge module ────────────────────────
            try:
                logger.debug(
                    f"ENGINE: Invoking module '{module_name}' for season {game_season} "
                    f"with input shape {input_to_transform_fn.shape}."
                )
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

                if processed_data_from_module is None or processed_data_from_module.empty:
                    # Immediately abort entire pipeline
                    logger.error(f"Pipeline aborted: '{module_name}' returned empty DataFrame.")
                    return pd.DataFrame()


                if module_name in ["rolling", "momentum"]:
                    if "game_id" in processed_data_from_module.columns:
                        processed_data_from_module["game_id"] = processed_data_from_module[
                            "game_id"
                        ].astype(str)
                        features_for_this_season_chunk = processed_data_from_module[
                            processed_data_from_module["game_id"]
                            .isin(current_season_chunk["game_id"].astype(str).unique())
                        ].copy()

                        new_cols = [
                            col
                            for col in features_for_this_season_chunk.columns
                            if col not in current_season_chunk.columns
                            and col not in KEEP_COLS_ALWAYS + outcome_cols
                        ]

                        if new_cols:
                            logger.debug(
                                f"ENGINE ({module_name.upper()}): Merging {len(new_cols)} new features for season {game_season}."
                            )
                            loop_chunk_for_season = current_season_chunk.merge(
                                features_for_this_season_chunk[["game_id"] + new_cols],
                                on="game_id",
                                how="left",
                            )
                        else:
                            loop_chunk_for_season = current_season_chunk
                            logger.warning(
                                f"ENGINE ({module_name.upper()}): No new specific features identified to merge for season {game_season}."
                            )
                    else:
                        logger.error(
                            f"ENGINE ({module_name.upper()}): 'game_id' missing in output. "
                            "Cannot merge features. Reverting to original chunk."
                        )
                        loop_chunk_for_season = current_season_chunk
                else:
                    # “Standard” modules simply replace the chunk
                    loop_chunk_for_season = processed_data_from_module

            except Exception as exc:
                logger.error(
                    f"Error in module '{module_name}' for season {game_season}: {exc}",
                    exc_info=debug,
                )
                # Stop further modules but keep the already‐merged state
                break

        # ─── After all modules (or early break) for one season ───
        if not loop_chunk_for_season.empty:
            # Restore any missing KEEP_COLS_ALWAYS before saving this season’s chunk
            cols_to_restore = [
                col
                for col in KEEP_COLS_ALWAYS
                if col in original_df_full_scope.columns and col not in loop_chunk_for_season.columns
            ]
            if cols_to_restore and "_row_id_pipeline" in loop_chunk_for_season.columns:
                logger.debug(
                    f"ENGINE: Restoring {cols_to_restore} to loop_chunk for season {game_season} using _row_id_pipeline."
                )
                restore_data = original_df_full_scope[
                    ["_row_id_pipeline"] + cols_to_restore
                ].drop_duplicates(subset=["_row_id_pipeline"])
                loop_chunk_for_season = loop_chunk_for_season.merge(
                    restore_data, on="_row_id_pipeline", how="left"
                )

            all_processed_season_chunks.append(loop_chunk_for_season)
        else:
            logger.warning(f"ENGINE: Season {game_season} resulted in an empty chunk. Excluding.")

    # ─── COMBINE ALL SEASONS ──────────────────────────────────
    if not all_processed_season_chunks:
        logger.error("No data chunks processed. Returning empty DataFrame.")
        return pd.DataFrame()

    final_df = pd.concat(all_processed_season_chunks, ignore_index=True, sort=False)
    logger.info(
        f"ENGINE: Pipeline complete. Shape before final KEEP_COLS_ALWAYS merge: {final_df.shape}"
    )

    # ─── FINAL MERGE OF KEEP_COLS_ALWAYS USING _row_id_pipeline ──
    if "_row_id_pipeline" in final_df.columns and "_row_id_pipeline" in original_df_full_scope.columns:
        logger.debug("ENGINE: Performing final merge of KEEP_COLS_ALWAYS onto final_df.")
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
        logger.debug(f"ENGINE: Final df shape after KEEP_COLS_ALWAYS merge: {final_df.shape}")
    else:
        logger.warning(
            "ENGINE: _row_id_pipeline missing for final KEEP_COLS_ALWAYS merge. Skipping this merge."
        )

    if "_row_id_pipeline" in final_df.columns:
        final_df = final_df.drop(columns=["_row_id_pipeline"])

    if debug:
        logger.setLevel(logging.INFO)
    return final_df
