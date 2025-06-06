from __future__ import annotations
import logging
from typing import Any, Dict as TypingDict, Optional, Tuple

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS_IMPORT, normalize_team_name
    MLB_DEFAULTS: TypingDict[str, float] = MLB_DEFAULTS_IMPORT
    # logger.info("Imported MLB_DEFAULTS and normalize_team_name from .utils in advanced.py")
except ImportError:
    logger.warning("advanced.py: Could not import DEFAULTS or normalize_team_name; using fallbacks.")
    MLB_DEFAULTS: TypingDict[str, float] = {}
    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip().lower() if pd.notna(team_id) else "unknown"

# ────── NORMALIZATION WRAPPER ──────
def _safe_norm(team_val: Any) -> str:
    norm = normalize_team_name(team_val)
    if not isinstance(norm, str):
        norm = str(norm)
    norm = norm.strip().lower()
    if norm in {"unknown", "unknown_team"} or not norm: # Check if norm is empty string
        norm = str(team_val).strip().lower()
    return norm.replace(" ", "")

# --- Module‐level constants for default column names (keep as is) ---
game_id_col       = "game_id"
home_team_col     = "home_team_id"
away_team_col     = "away_team_id"
home_hand_col     = "home_probable_pitcher_handedness"
away_hand_col     = "away_probable_pitcher_handedness"
hist_team_col_default   = "team_id"
hist_season_col_default = "season"
hist_w_home_pct_default = "wins_home_percentage"
hist_rf_home_default    = "runs_for_avg_home"
hist_ra_home_default    = "runs_against_avg_home"
hist_w_away_pct_default = "wins_away_percentage"
hist_rf_away_default    = "runs_for_avg_away"
hist_ra_away_default    = "runs_against_avg_away"
hist_vs_lhp_default     = "season_avg_runs_vs_lhp"
hist_vs_rhp_default     = "season_avg_runs_vs_rhp"


def _attach_split(
    df_to_modify: pd.DataFrame,
    team_col_name_in_df: str,
    hist_lookup_df: pd.DataFrame,
    col_mappings: TypingDict[str, Tuple[str, str, float]],
    flag_imputations: bool
) -> pd.DataFrame:

    if team_col_name_in_df not in df_to_modify.columns:
        logger.error(
            f"_attach_split: Essential team column '{team_col_name_in_df}' not found in df_to_modify. "
            "All features from this split will be populated with defaults."
        )
        for new_col_name, (_, defaults_key, fallback_value) in col_mappings.items():
            default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
            df_to_modify[new_col_name] = default_to_use
            df_to_modify[new_col_name] = pd.to_numeric(df_to_modify[new_col_name], errors='coerce').fillna(default_to_use).astype(float)
            if flag_imputations:
                df_to_modify[f"{new_col_name}_imputed"] = True
                df_to_modify[f"{new_col_name}_imputed"] = df_to_modify[f"{new_col_name}_imputed"].astype(bool).astype(int)
        return df_to_modify

    normalized_team_id_series = df_to_modify[team_col_name_in_df].apply(_safe_norm)

    for new_col_name, (hist_col_name, defaults_key, fallback_value) in col_mappings.items():
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        imputed_flags_for_col = pd.Series(True, index=df_to_modify.index, dtype=bool)

        if not hist_lookup_df.empty and hist_lookup_df.index.name == "team_norm" and \
           hist_col_name in hist_lookup_df.columns:

            hist_series_map = hist_lookup_df[hist_col_name]
            mapped_values = normalized_team_id_series.map(hist_series_map)

            imputed_flags_for_col = mapped_values.isna()
            df_to_modify[new_col_name] = mapped_values.fillna(default_to_use)
        else:
            if hist_lookup_df.empty:
                logger.debug(f"_attach_split: Hist_lookup_df is empty for '{new_col_name}'. Using default: {default_to_use}")
            elif hist_lookup_df.index.name != "team_norm":
                logger.warning(f"_attach_split: Hist_lookup_df not indexed by 'team_norm' for '{new_col_name}'. Using default: {default_to_use}")
            else:
                logger.warning(f"_attach_split: Column '{hist_col_name}' for feature '{new_col_name}' not in hist_lookup_df. Using default: {default_to_use}")
            df_to_modify[new_col_name] = default_to_use

        if flag_imputations:
            df_to_modify[f"{new_col_name}_imputed"] = imputed_flags_for_col.values
            df_to_modify[f"{new_col_name}_imputed"] = df_to_modify[f"{new_col_name}_imputed"].astype(bool).astype(int)

        df_to_modify[new_col_name] = pd.to_numeric(df_to_modify[new_col_name], errors="coerce").fillna(default_to_use).astype(float)

    return df_to_modify


def _attach_vs_hand(
    df_to_modify: pd.DataFrame,
    team_col_name_in_df: str,
    opp_hand_col_name_in_df: str,
    hist_lookup_df: pd.DataFrame,
    new_col_name_base: str,
    hist_stat_vs_lhp: str,
    hist_stat_vs_rhp: str,
    defaults_key: str,
    fallback_value: float,
    flag_imputations: bool,
    season_to_lookup: Optional[int] = None
) -> pd.DataFrame:
    """
    Attaches offense‐vs‐pitcher‐hand feature. For each row, look at that team's historical
    stat vs LHP or vs RHP based on the opponent's hand. If missing → default.
    """
    # ... (initial setup and checks remain the same) ...
    
    # ... (this part remains the same)
    normalized_team_ids = df_to_modify[team_col_name_in_df].apply(_safe_norm)
    opponent_hands = (
        df_to_modify[opp_hand_col_name_in_df]
        .astype(str)
        .str.upper()
        .replace("", "UNKNOWN_HAND")
        .replace("NONE", "UNKNOWN_HAND")
        .replace("NULL", "UNKNOWN_HAND")
        .fillna("UNKNOWN_HAND")
    )
    default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
    calculated_values_arr = np.full(len(df_to_modify), default_to_use, dtype=float)
    imputation_flags_arr = np.full(len(df_to_modify), True, dtype=bool)

    # Loop for each game to calculate the feature
    for i, (team_norm, opp_hand_val) in enumerate(zip(normalized_team_ids, opponent_hands)):
        val_for_row = np.nan
        imputed_this_row = True

        if not hist_lookup_df.empty and team_norm in hist_lookup_df.index:
            
            ## FINAL FIX: Use a safer .get() to access data instead of an 'in' check.
            ## This avoids issues with subtle column name mismatches.
            row_data = hist_lookup_df.loc[team_norm]
            looked_up_val = np.nan # Default to NaN

            if opp_hand_val == 'L':
                looked_up_val = row_data.get(hist_stat_vs_lhp) # Safely get LHP data
            elif opp_hand_val == 'R':
                looked_up_val = row_data.get(hist_stat_vs_rhp) # Safely get RHP data

            # Now, process the looked_up_val
            if pd.notna(looked_up_val):
                val_for_row = looked_up_val
                imputed_this_row = False
            else:
                # This path is taken if the .get() returned None or if the value was already NaN
                logger.debug(f"Looked up value for {team_norm} vs {opp_hand_val} is missing. Will use default: {default_to_use}")

        # ... (rest of the assignment logic remains the same)
        calculated_values_arr[i] = val_for_row if pd.notna(val_for_row) else default_to_use
        if flag_imputations:
            imputation_flags_arr[i] = imputed_this_row

    # ... (final dataframe assignment logic remains the same) ...
    df_to_modify[new_col_name_base] = pd.to_numeric(
        pd.Series(calculated_values_arr, index=df_to_modify.index, name=new_col_name_base),
        errors='coerce'
    ).fillna(default_to_use).astype(float)

    if flag_imputations:
        df_to_modify[f"{new_col_name_base}_imputed"] = pd.Series(
            imputation_flags_arr, index=df_to_modify.index, name=f"{new_col_name_base}_imputed", dtype=bool
        )

    return df_to_modify

def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
    *,
    season_to_lookup: Optional[int] = None,
    flag_imputations: bool = True,
    debug: bool = False,
    game_id_col_param: str = game_id_col,
    home_team_col_param: str = home_team_col,
    away_team_col_param: str = away_team_col,
    home_hand_col_param: str = home_hand_col,
    away_hand_col_param: str = away_hand_col,
    hist_team_col_param: str = hist_team_col_default,
    hist_season_col_param: str = hist_season_col_default,
    hist_w_home_pct_param: str = hist_w_home_pct_default,
    hist_rf_home_param: str = hist_rf_home_default,
    hist_ra_home_param: str = hist_ra_home_default,
    hist_w_away_pct_param: str = hist_w_away_pct_default,
    hist_rf_away_param: str = hist_rf_away_default,
    hist_ra_away_param: str = hist_ra_away_default,
    hist_vs_lhp_param: str = hist_vs_lhp_default,
    hist_vs_rhp_param: str = hist_vs_rhp_default,
    drop_input_team_cols: bool = True
) -> pd.DataFrame:
    logger.debug(f"advanced.transform ENTRY: Received season_to_lookup = {season_to_lookup!r}")

    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    if df is None or df.empty:
        logger.warning("ADVANCED.TRANSFORM: Empty input df; returning copy.")
        if debug: logger.setLevel(orig_level)
        return df.copy() if df is not None else pd.DataFrame()

    result = df.copy()

    essential_df_cols = [home_team_col_param, away_team_col_param, home_hand_col_param, away_hand_col_param]
    missing_essential_on_df = [c for c in essential_df_cols if c not in result.columns]
    if missing_essential_on_df:
        logger.error(
            f"ADVANCED.TRANSFORM: Input DataFrame 'df' is missing essential columns: {missing_essential_on_df}. "
            "Returning copy of original df."
        )
        if debug: logger.setLevel(orig_level)
        return result

    hist_filtered_indexed = pd.DataFrame()
    if historical_team_stats_df is not None and not historical_team_stats_df.empty:
                ## FIX: ADD DEBUG LINE 1 HERE
        logger.debug(f"ADVANCED_TRANSFORM_DEBUG (1): Columns received in historical_team_stats_df: {historical_team_stats_df.columns.tolist()}")

        hist_copy = historical_team_stats_df.copy()
        required_hist_cols_for_lookup = [hist_season_col_param, hist_team_col_param]

        if season_to_lookup is not None and isinstance(season_to_lookup, (int, np.integer)) and \
        hist_season_col_param in hist_copy.columns:
            hist_copy[hist_season_col_param] = pd.to_numeric(hist_copy[hist_season_col_param], errors="coerce")
            hist_copy = hist_copy[hist_copy[hist_season_col_param] == season_to_lookup].copy()
        elif season_to_lookup is not None:
            logger.warning(f"ADVANCED.TRANSFORM: season_to_lookup ({season_to_lookup}) provided, but "
                           f"'{hist_season_col_param}' not in historical_team_stats_df or season_to_lookup not int.")

        if not all(col in hist_copy.columns for col in required_hist_cols_for_lookup):
            logger.error(
                f"ADVANCED.TRANSFORM: historical_team_stats_df is missing essential columns: {required_hist_cols_for_lookup}. "
                "Historical lookups will use defaults."
            )
        elif hist_copy.empty:
            logger.warning(
                 "ADVANCED.TRANSFORM: historical_team_stats_df became empty after filtering for "
                f"season_to_lookup={season_to_lookup if season_to_lookup is not None else 'all'}. "
                 "Historical lookups will use defaults."
            )
        else:
            if "team_norm" not in hist_copy.columns:
                 hist_copy["team_norm"] = hist_copy[hist_team_col_param].apply(_safe_norm)
            unique_team_stats = hist_copy.drop_duplicates(subset=["team_norm"], keep="first")
            if not unique_team_stats.empty:
                try:
                    hist_filtered_indexed = unique_team_stats.set_index("team_norm")
                                        
                    ## FIX: ADD DEBUG LINE 2 HERE
                    logger.debug(f"ADVANCED_TRANSFORM_DEBUG (2): Columns in final hist_filtered_indexed: {hist_filtered_indexed.columns.tolist()}")

                    logger.info(
                        f"ADVANCED.TRANSFORM: Built hist_filtered_indexed (shape {hist_filtered_indexed.shape}) "
                        f"for lookup (using season context: {season_to_lookup if season_to_lookup is not None else 'first_available_per_team'})."
                    )
                except KeyError:
                    logger.error("ADVANCED.TRANSFORM: Failed to set 'team_norm' as index.")
            else:
                logger.warning("ADVANCED.TRANSFORM: No unique team stats after drop_duplicates.")
    else:
        logger.warning(
            "ADVANCED.TRANSFORM: historical_team_stats_df is None or empty. All historical lookups will use defaults."
        )

    split_map_defs = {
        "h_team_hist_HA_win_pct":           (hist_w_home_pct_param, "mlb_hist_win_pct_home",  0.5),
        "h_team_hist_HA_runs_for_avg":      (hist_rf_home_param,    "mlb_hist_runs_for_home", 4.5),
        "h_team_hist_HA_runs_against_avg":  (hist_ra_home_param,    "mlb_hist_runs_against_home", 4.5),
        "a_team_hist_HA_win_pct":           (hist_w_away_pct_param, "mlb_hist_win_pct_away",  0.5),
        "a_team_hist_HA_runs_for_avg":      (hist_rf_away_param,    "mlb_hist_runs_for_away", 4.5),
        "a_team_hist_HA_runs_against_avg":  (hist_ra_away_param,    "mlb_hist_runs_against_away", 4.5),
    }
    home_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("h_team_")}
    result = _attach_split(result, home_team_col_param, hist_filtered_indexed, home_split_defs, flag_imputations)
    away_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("a_team_")}
    result = _attach_split(result, away_team_col_param, hist_filtered_indexed, away_split_defs, flag_imputations)


    result = _attach_vs_hand(
        result,
        home_team_col_param,
        away_hand_col_param,
        hist_filtered_indexed,
        "h_team_off_avg_runs_vs_opp_hand",
        hist_vs_lhp_param,
        hist_vs_rhp_param,
        "mlb_avg_runs_vs_hand",
        4.5,
        flag_imputations,
        season_to_lookup=season_to_lookup
    )
    result = _attach_vs_hand(
        result,
        away_team_col_param,
        home_hand_col_param, # away team vs home pitcher
        hist_filtered_indexed,
        "a_team_off_avg_runs_vs_opp_hand",
        hist_vs_lhp_param,
        hist_vs_rhp_param,
        "mlb_avg_runs_vs_hand",
        4.5,
        flag_imputations,
        season_to_lookup=season_to_lookup
    )

    if not flag_imputations:
        cols_to_drop = [c for c in result.columns if c.endswith("_imputed")]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop, errors="ignore")

    if drop_input_team_cols:
        cols_to_drop_ids = [col for col in [home_team_col_param, away_team_col_param] if col in result.columns]
        if cols_to_drop_ids:
            result = result.drop(columns=cols_to_drop_ids, errors="ignore")

    if flag_imputations:
        for col_name in result.columns:
            if col_name.endswith("_imputed"):
                if result[col_name].dtype == 'bool' or result[col_name].isin([True, False, 0, 1, 0.0, 1.0]).all():
                    result[col_name] = result[col_name].astype(bool).astype(int)
                else:
                    logger.warning(f"ADVANCED.TRANSFORM: Imputed flag column '{col_name}' has unexpected dtype {result[col_name].dtype}. Setting to default int 1 (True).")
                    result[col_name] = 1


    if debug:
        logger.setLevel(orig_level)
    logger.debug(f"ADVANCED.TRANSFORM: Complete. output_shape={result.shape}")
    return result