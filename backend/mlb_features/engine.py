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
        original_df_full_scope["orig_game_id"] = "gid_" + original_df_full_scope["_row_id_pipeline"].astype(str)

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
            # Ensure outcome columns exist for rolling
            outcome_cols_hist = [
                "home_score", "away_score", "home_hits", "away_hits", "home_errors", "away_errors"
            ]
            for col in outcome_cols_hist:
                if col not in historical_games_df_context.columns:
                    historical_games_df_context[col] = np.nan
            if "game_id" in historical_games_df_context.columns:
                historical_games_df_context["game_id"] = historical_games_df_context["game_id"].astype(str)

    # ─── PREPARE team_stats_df_context ────────────────────
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

        current_season_chunk_original_game_ids = current_season_chunk["orig_game_id"].unique()
        current_season_chunk["game_id"] = current_season_chunk["game_id"].astype(str)
        loop_chunk_for_season = current_season_chunk.copy()

        for module_name in execution_order:
            fn = TRANSFORMS.get(module_name)
            if fn is None:
                logger.warning(f"ENGINE: Unknown module '{module_name}' in execution_order. Skipping.")
                continue

            input_to_transform_fn = loop_chunk_for_season
            kwargs_for_module: Dict[str, Any] = {"debug": debug}

            if module_name == "rolling":
                temp_chunk_with_nan_outcomes = loop_chunk_for_season.copy()
                outcome_cols = ["home_score", "away_score", "home_hits", "away_hits", "home_errors", "away_errors"]
                for col in outcome_cols:
                    if col not in temp_chunk_with_nan_outcomes.columns:
                        temp_chunk_with_nan_outcomes[col] = np.nan

                input_df_for_rolling = temp_chunk_with_nan_outcomes
                if (
                    historical_games_df_context is not None
                    and not historical_games_df_context.empty
                    and "game_date_et" in historical_games_df_context.columns
                ):
                    cutoff = loop_chunk_for_season["game_date_et"].min()
                    if pd.isna(cutoff):
                        relevant_hist_for_rolling = pd.DataFrame()
                    else:
                        historical_games_df_context["game_date_et"] = pd.to_datetime(
                            historical_games_df_context["game_date_et"], errors="coerce"
                        )
                        relevant_hist_for_rolling = historical_games_df_context[
                            historical_games_df_context["game_date_et"] < cutoff
                        ].copy()

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

                kwargs_for_module.update({
                    "window_sizes": rolling_window_sizes,
                    "flag_imputations": flag_imputations,
                })

                processed_data_from_module = fn(input_df_for_rolling, **kwargs_for_module)

                if processed_data_from_module is None or processed_data_from_module.empty:
                    return pd.DataFrame()

                processed_data_from_module["game_id"] = processed_data_from_module["game_id"].astype(str)
                upcoming_game_ids_for_season = current_season_chunk["game_id"].astype(str).unique()

                upcoming_games_component = processed_data_from_module[
                    processed_data_from_module["game_id"].isin(upcoming_game_ids_for_season)
                ].copy()
                hist_component_with_stats = processed_data_from_module[
                    ~processed_data_from_module["game_id"].isin(upcoming_game_ids_for_season)
                ].copy()

                if not hist_component_with_stats.empty:
                    for col_id in ["home_team_id", "away_team_id"]:
                        if col_id in hist_component_with_stats.columns:
                            hist_component_with_stats[col_id] = hist_component_with_stats[col_id].astype(str)
                    if "game_date_et" in hist_component_with_stats.columns:
                        hist_component_with_stats["game_date_et"] = pd.to_datetime(
                            hist_component_with_stats["game_date_et"]
                        )
                        hist_component_with_stats = hist_component_with_stats.sort_values(
                            by="game_date_et", ascending=False
                        )
                    else:
                        hist_component_with_stats = pd.DataFrame()

                rolling_cols_added_by_module = [
                    col for col in processed_data_from_module.columns
                    if col not in loop_chunk_for_season.columns and col not in KEEP_COLS_ALWAYS
                ]
                base_rolling_stat_cols_to_carry = [
                    rc for rc in rolling_cols_added_by_module if not rc.endswith("_imputed")
                ]

                if (
                    not hist_component_with_stats.empty
                    and not upcoming_games_component.empty
                    and base_rolling_stat_cols_to_carry
                ):
                    for col_id in ["home_team_id", "away_team_id"]:
                        if col_id in upcoming_games_component.columns:
                            upcoming_games_component[col_id] = upcoming_games_component[col_id].astype(str)
                    if "game_date_et" in upcoming_games_component.columns:
                        upcoming_games_component["game_date_et"] = pd.to_datetime(
                            upcoming_games_component["game_date_et"]
                        )

                    for idx, upcoming_row in upcoming_games_component.iterrows():
                        upcoming_date = upcoming_row.get("game_date_et")
                        home_team_id = str(upcoming_row.get("home_team_id", ""))
                        away_team_id = str(upcoming_row.get("away_team_id", ""))

                        if pd.isna(upcoming_date) or not home_team_id or not away_team_id:
                            logger.warning(
                                f"ENGINE_ROLLING_CARRY_FORWARD: Skipping game {upcoming_row.get('game_id')} due to missing date/team_ids."
                            )
                            continue

                        # --- Home Team Carry-Forward Logic ---
                        last_game_home_was_home = hist_component_with_stats[
                            (hist_component_with_stats["home_team_id"] == home_team_id)
                            & (hist_component_with_stats["game_date_et"] < upcoming_date)
                        ].head(1)
                        last_game_home_was_away = hist_component_with_stats[
                            (hist_component_with_stats["away_team_id"] == home_team_id)
                            & (hist_component_with_stats["game_date_et"] < upcoming_date)
                        ].head(1)

                        source_game_for_home = pd.DataFrame()
                        if not last_game_home_was_home.empty and not last_game_home_was_away.empty:
                            source_game_for_home = (
                                last_game_home_was_home
                                if last_game_home_was_home.iloc[0]["game_date_et"]
                                >= last_game_home_was_away.iloc[0]["game_date_et"]
                                else last_game_home_was_away
                            )
                        elif not last_game_home_was_home.empty:
                            source_game_for_home = last_game_home_was_home
                        elif not last_game_home_was_away.empty:
                            source_game_for_home = last_game_home_was_away

                        if not source_game_for_home.empty:
                            source_row_home = source_game_for_home.iloc[0]
                            home_team_was_home_in_source = (
                                source_row_home["home_team_id"] == home_team_id
                            )

                            for target_col in base_rolling_stat_cols_to_carry:
                                if target_col.startswith("home_rolling_"):
                                    stat_suffix = target_col.replace("home_rolling_", "")
                                    source_col = (
                                        f"home_rolling_{stat_suffix}"
                                        if home_team_was_home_in_source
                                        else f"away_rolling_{stat_suffix}"
                                    )
                                    if (
                                        source_col in source_row_home
                                        and pd.notna(source_row_home[source_col])
                                    ):
                                        value_to_carry = source_row_home[source_col]
                                        upcoming_games_component.loc[idx, target_col] = value_to_carry
                                        imputed_flag = f"{target_col}_imputed"
                                        if imputed_flag in upcoming_games_component.columns:
                                            upcoming_games_component.loc[idx, imputed_flag] = 0

                        # --- Away Team Carry-Forward Logic ---
                        last_game_away_was_home = hist_component_with_stats[
                            (hist_component_with_stats["home_team_id"] == away_team_id)
                            & (hist_component_with_stats["game_date_et"] < upcoming_date)
                        ].head(1)
                        last_game_away_was_away = hist_component_with_stats[
                            (hist_component_with_stats["away_team_id"] == away_team_id)
                            & (hist_component_with_stats["game_date_et"] < upcoming_date)
                        ].head(1)

                        source_game_for_away = pd.DataFrame()
                        if not last_game_away_was_home.empty and not last_game_away_was_away.empty:
                            source_game_for_away = (
                                last_game_away_was_home
                                if last_game_away_was_home.iloc[0]["game_date_et"]
                                >= last_game_away_was_away.iloc[0]["game_date_et"]
                                else last_game_away_was_away
                            )
                        elif not last_game_away_was_home.empty:
                            source_game_for_away = last_game_away_was_home
                        elif not last_game_away_was_away.empty:
                            source_game_for_away = last_game_away_was_away

                        if not source_game_for_away.empty:
                            source_row_away = source_game_for_away.iloc[0]
                            away_team_was_home_in_source = (
                                source_row_away["home_team_id"] == away_team_id
                            )

                            for target_col in base_rolling_stat_cols_to_carry:
                                if target_col.startswith("away_rolling_"):
                                    stat_suffix = target_col.replace("away_rolling_", "")
                                    source_col = (
                                        f"home_rolling_{stat_suffix}"
                                        if away_team_was_home_in_source
                                        else f"away_rolling_{stat_suffix}"
                                    )
                                    if (
                                        source_col in source_row_away
                                        and pd.notna(source_row_away[source_col])
                                    ):
                                        value_to_carry = source_row_away[source_col]
                                        upcoming_games_component.loc[idx, target_col] = value_to_carry
                                        imputed_flag = f"{target_col}_imputed"
                                        if imputed_flag in upcoming_games_component.columns:
                                            upcoming_games_component.loc[idx, imputed_flag] = 0

                    # --- Merge carried-forward rolling features back into loop_chunk_for_season ---
                    if rolling_cols_added_by_module:
                        if "game_id" not in loop_chunk_for_season.columns:
                            logger.error(
                                "ENGINE_ROLLING_MERGE: 'game_id' is missing from loop_chunk_for_season. Cannot merge rolling features."
                            )
                            # Skip to next module if critical key is missing
                            continue

                        loop_chunk_for_season["game_id"] = loop_chunk_for_season["game_id"].astype(str)

                        # Drop existing columns if they exist
                        cols_to_drop_from_target = [
                            col for col in rolling_cols_added_by_module
                            if col in loop_chunk_for_season.columns
                        ]
                        if cols_to_drop_from_target:
                            logger.debug(
                                f"ENGINE_ROLLING_MERGE: Preemptively dropping existing columns from loop_chunk_for_season: {cols_to_drop_from_target}"
                            )
                            loop_chunk_for_season = loop_chunk_for_season.drop(columns=cols_to_drop_from_target)

                        # Merge the rolling features
                        loop_chunk_for_season = loop_chunk_for_season.merge(
                            upcoming_games_component[["game_id"] + rolling_cols_added_by_module],
                            on="game_id",
                            how="left"
                        )
                        logger.debug(
                            f"ENGINE (ROLLING): Merged {len(rolling_cols_added_by_module)} features "
                            f"(after carry-forward) into loop_chunk_for_season for season {game_season}."
                        )
                    else:
                        logger.warning(
                            f"ENGINE (ROLLING): No new columns identified from rolling module output to merge for season {game_season}."
                        )

                # Move to next module after rolling
                continue

            elif module_name == "h2h":
                kwargs_for_module["historical_df"] = historical_games_df_context
                kwargs_for_module["max_games"] = h2h_max_games
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)
                if processed_data_from_module is None or processed_data_from_module.empty:
                    continue

            elif module_name == "advanced":
                if team_stats_df_context is None or team_stats_df_context.empty:
                    logger.warning(
                        f"Skipping 'advanced' module for season {game_season} (no team stats)."
                    )
                    continue

                # Populate handedness and team ID params, plus historical_team_stats_df
                kwargs_for_module.update({
                    "home_team_col_param": "home_team_id",
                    "away_team_col_param": "away_team_id",
                    "home_hand_col_param": "home_starter_pitcher_handedness",
                    "away_hand_col_param": "away_starter_pitcher_handedness",
                    "flag_imputations": flag_imputations,
                    "historical_team_stats_df": team_stats_df_context,
                })

                # Explicitly set season_to_lookup for advanced.transform
                if _supports_kwarg(fn, "season_to_lookup"):
                    lookup_season_for_advanced = game_season - 1
                    kwargs_for_module["season_to_lookup"] = lookup_season_for_advanced
                    logger.info(
                        f"ENGINE_ADV_CONFIG: Setting 'season_to_lookup' for advanced.transform to: {lookup_season_for_advanced}"
                    )
                else:
                    # If not supported, remove any leftover
                    if "season_to_lookup" in kwargs_for_module:
                        del kwargs_for_module["season_to_lookup"]

                # Log presence/absence of handedness columns
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
                            f"ENGINE_ADV_PRE_CALL: '{hand_col_check}' IS PRESENT. Sample: \n"
                            f"{input_to_transform_fn[[hand_col_check]].head().to_string()}"
                        )

                # Log final kwargs for advanced.transform
                logger.debug(f"ENGINE_ADV_PRE_CALL: Final Kwargs being passed to advanced.transform: {kwargs_for_module}")
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            else:
                # Default handling for any other module
                processed_data_from_module = fn(input_to_transform_fn, **kwargs_for_module)

            # --- Common post-processing for non-rolling modules ---
            if processed_data_from_module is None or processed_data_from_module.empty:
                return pd.DataFrame()

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
    logger.info(f"ENGINE: Pipeline complete. Shape before final KEEP_COLS_ALWAYS merge: {final_df.shape}")

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

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"ENGINE: Feature pipeline complete in {pipeline_duration:.2f}s — final shape {final_df.shape}")

    if debug:
        logger.setLevel(logging.INFO)
    return final_df
