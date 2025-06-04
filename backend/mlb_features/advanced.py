# backend/mlb_features/advanced.py

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
# ... (game_id_col, home_team_col, etc.) ...
game_id_col       = "game_id"
home_team_col     = "home_team_id"
away_team_col     = "away_team_id"
home_hand_col     = "home_probable_pitcher_handedness" # Default, will be overridden by param
away_hand_col     = "away_probable_pitcher_handedness" # Default, will be overridden by param

hist_team_col_default   = "team_id"
hist_season_col_default = "season"
hist_w_home_pct_default = "wins_home_percentage"
# ... (other hist_ defaults) ...
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
    hist_lookup_df: pd.DataFrame,  # Indexed by unique 'team_norm'
    col_mappings: TypingDict[str, Tuple[str, str, float]], # e.g. {"h_HA_win": ("hist_win", "def_key", 0.5)}
    flag_imputations: bool
) -> pd.DataFrame:
    
    if team_col_name_in_df not in df_to_modify.columns:
        logger.error(
            f"_attach_split: Essential team column '{team_col_name_in_df}' not found in df_to_modify. "
            "All features from this split will be populated with defaults."
        )
        # If the team column is missing, we still need to create all the output columns
        # defined in col_mappings, but fill them with their defaults.
        for new_col_name, (_, defaults_key, fallback_value) in col_mappings.items():
            default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
            df_to_modify[new_col_name] = default_to_use
            df_to_modify[new_col_name] = pd.to_numeric(df_to_modify[new_col_name], errors='coerce').fillna(default_to_use).astype(float)
            if flag_imputations:
                df_to_modify[f"{new_col_name}_imputed"] = True # All imputed
                df_to_modify[f"{new_col_name}_imputed"] = df_to_modify[f"{new_col_name}_imputed"].astype(bool).astype(int)
        return df_to_modify # Return after setting all defaults

    # Proceed if team_col_name_in_df exists
    normalized_team_id_series = df_to_modify[team_col_name_in_df].apply(_safe_norm)

    for new_col_name, (hist_col_name, defaults_key, fallback_value) in col_mappings.items():
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        imputed_flags_for_col = pd.Series(True, index=df_to_modify.index, dtype=bool) # Default to all imputed

        if not hist_lookup_df.empty and hist_lookup_df.index.name == "team_norm" and \
           hist_col_name in hist_lookup_df.columns:
            
            hist_series_map = hist_lookup_df[hist_col_name]
            mapped_values = normalized_team_id_series.map(hist_series_map)
            
            imputed_flags_for_col = mapped_values.isna() # True where lookup failed (NaN)
            df_to_modify[new_col_name] = mapped_values.fillna(default_to_use)
        else:
            # Conditions for using default: hist_lookup_df is empty OR not indexed by team_norm OR hist_col_name is missing
            if hist_lookup_df.empty:
                logger.debug(f"_attach_split: Hist_lookup_df is empty for '{new_col_name}'. Using default: {default_to_use}")
            elif hist_lookup_df.index.name != "team_norm":
                logger.warning(f"_attach_split: Hist_lookup_df not indexed by 'team_norm' for '{new_col_name}'. Using default: {default_to_use}")
            else: # hist_col_name not in hist_lookup_df.columns
                logger.warning(f"_attach_split: Column '{hist_col_name}' for feature '{new_col_name}' not in hist_lookup_df. Using default: {default_to_use}")
            df_to_modify[new_col_name] = default_to_use
            # All values are from default, so imputed_flags_for_col remains all True

        if flag_imputations:
            df_to_modify[f"{new_col_name}_imputed"] = imputed_flags_for_col.values # Assign the boolean Series/array
            df_to_modify[f"{new_col_name}_imputed"] = df_to_modify[f"{new_col_name}_imputed"].astype(bool).astype(int) # Convert to int
        
        df_to_modify[new_col_name] = pd.to_numeric(df_to_modify[new_col_name], errors="coerce").fillna(default_to_use).astype(float)
        
    return df_to_modify


def _attach_vs_hand(
    df_to_modify: pd.DataFrame,
    team_col_name_in_df: str,
    opp_hand_col_name_in_df: str,
    hist_lookup_df: pd.DataFrame,
    new_col_name_base: str, # <<<< THIS IS THE PARAMETER
    hist_stat_vs_lhp: str,
    hist_stat_vs_rhp: str,
    defaults_key: str,
    fallback_value: float,
    flag_imputations: bool
) -> pd.DataFrame:
    logger.debug(f"_attach_vs_hand: Creating '{new_col_name_base}' for team col '{team_col_name_in_df}' vs opp_hand_col '{opp_hand_col_name_in_df}'.")

    # Guard: If the primary team column itself is missing, we can't do lookups.
    # new_col_name_base IS defined here as it's a function parameter.
    if team_col_name_in_df not in df_to_modify.columns:
        logger.error(
            f"_attach_vs_hand: Essential team column '{team_col_name_in_df}' not found in df_to_modify for creating '{new_col_name_base}'. " # Using param here is fine
            "This feature will be populated with defaults."
        )
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        
        # Assign default to the column named by new_col_name_base
        df_to_modify[new_col_name_base] = default_to_use # Use new_col_name_base
        df_to_modify[new_col_name_base] = pd.to_numeric(df_to_modify[new_col_name_base], errors='coerce').fillna(default_to_use).astype(float)

        if flag_imputations:
            # Assign True to the imputation flag column named based on new_col_name_base
            df_to_modify[f"{new_col_name_base}_imputed"] = True 
            df_to_modify[f"{new_col_name_base}_imputed"] = df_to_modify[f"{new_col_name_base}_imputed"].astype(bool).astype(int)
        return df_to_modify # Correctly return after setting defaults
    # Ensure opposing hand column exists; if not, create it and fill with a placeholder that becomes "UNKNOWN_HAND"
    if opp_hand_col_name_in_df not in df_to_modify.columns:
        logger.warning(
            f"_attach_vs_hand: Opposing-hand column '{opp_hand_col_name_in_df}' was missing for '{new_col_name_base}'. "
            "Creating it and defaulting its values to UNKNOWN_HAND for processing."
        )
        df_to_modify[opp_hand_col_name_in_df] = "" # This will become "UNKNOWN_HAND" below

    normalized_team_ids = df_to_modify[team_col_name_in_df].apply(_safe_norm)
    
    # Process opponent_hands: convert to string, upper, replace "" with "UNKNOWN_HAND", then fill actual NaNs
    opponent_hands = (
        df_to_modify[opp_hand_col_name_in_df] # Column is now guaranteed to exist
          .astype(str)
          .str.upper()
          .replace("", "UNKNOWN_HAND") # Ensure empty strings become UNKNOWN_HAND
          .fillna("UNKNOWN_HAND")      # Ensure NaNs become UNKNOWN_HAND
    )
    unknown_hand_count = (opponent_hands == "UNKNOWN_HAND").sum()
    if unknown_hand_count > 0 and unknown_hand_count < len(df_to_modify) : # Log only if some, but not all, are unknown
        logger.debug(f"_attach_vs_hand for '{new_col_name_base}': {unknown_hand_count}/{len(df_to_modify)} games will use UNKNOWN opponent hand logic.")
    elif unknown_hand_count == len(df_to_modify):
        logger.warning(f"_attach_vs_hand for '{new_col_name_base}': ALL {len(df_to_modify)} games have UNKNOWN opponent hand. Feature will rely heavily on defaults.")


    default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
    # Initialize as NumPy arrays for direct indexed assignment
    calculated_values_arr = np.full(len(df_to_modify), default_to_use, dtype=float) 
    imputation_flags_arr = np.full(len(df_to_modify), True, dtype=bool)     

    if not hist_lookup_df.empty and hist_lookup_df.index.name == "team_norm":
        for i, (team_norm, opp_hand_val) in enumerate(zip(normalized_team_ids, opponent_hands)):
            val_for_row_iteration = np.nan # Default for this iteration if no match

            if team_norm in hist_lookup_df.index:
                stat_col_to_use: Optional[str] = None
                if opp_hand_val == "L" and hist_stat_vs_lhp in hist_lookup_df.columns:
                    stat_col_to_use = hist_stat_vs_lhp
                elif opp_hand_val == "R" and hist_stat_vs_rhp in hist_lookup_df.columns:
                    stat_col_to_use = hist_stat_vs_rhp
                # If opp_hand_val is "UNKNOWN_HAND", stat_col_to_use remains None, 
                # leading to default_to_use for val_for_row_iteration.

                if stat_col_to_use:
                    looked_up_val = hist_lookup_df.at[team_norm, stat_col_to_use]
                    
                    # Your ValueError FIX (handles if .at returns a Series due to non-unique index)
                    if isinstance(looked_up_val, pd.Series):
                        logger.debug(f"_attach_vs_hand: Duplicate index '{team_norm}' found in hist_lookup_df for col '{stat_col_to_use}'. Taking first non-NaN value.")
                        valid_series_values = looked_up_val.dropna()
                        looked_up_val = valid_series_values.iloc[0] if not valid_series_values.empty else np.nan
                    
                    if pd.notna(looked_up_val):
                        val_for_row_iteration = float(looked_up_val) # Ensure float
                        imputation_flags_arr[i] = False # Successfully looked up
            
            calculated_values_arr[i] = val_for_row_iteration if pd.notna(val_for_row_iteration) else default_to_use
    else:
        logger.warning(
            f"_attach_vs_hand for '{new_col_name_base}': hist_lookup_df is empty or not indexed by 'team_norm'. "
            "All values for this feature will be the default."
        )
        # calculated_values_arr and imputation_flags_arr remain initialized with defaults/True

    # Assign the calculated values (now as a Series) to the DataFrame
    # This correctly handles the conversion from NumPy array to Pandas Series with index.
    calculated_series = pd.Series(calculated_values_arr, index=df_to_modify.index, name=new_col_name_base)
    df_to_modify[new_col_name_base] = pd.to_numeric(calculated_series, errors='coerce').fillna(default_to_use).astype(float)

    if flag_imputations:
        df_to_modify[f"{new_col_name_base}_imputed"] = pd.Series(
            imputation_flags_arr, index=df_to_modify.index, dtype=bool
        ).astype(int) # Convert final boolean flag to int

    return df_to_modify



def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
    *,
    season_to_lookup: Optional[int] = None, 
    flag_imputations: bool = True,
    debug: bool = False,
    game_id_col_param: str = game_id_col, # Using module-level defaults
    home_team_col_param: str = home_team_col,
    away_team_col_param: str = away_team_col,
    home_hand_col_param: str = home_hand_col, # Default "home_probable_pitcher_handedness"
    away_hand_col_param: str = away_hand_col, # Default "away_probable_pitcher_handedness"
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

    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
    
    logger.debug(f"ADVANCED.TRANSFORM: Start. df_shape={df.shape if df is not None else 'None'}. "
                 f"home_hand_col_param='{home_hand_col_param}', away_hand_col_param='{away_hand_col_param}'")

    if df is None or df.empty:
        logger.warning("ADVANCED.TRANSFORM: Empty input df; returning copy.")
        if debug: logger.setLevel(orig_level)
        return df.copy() if df is not None else pd.DataFrame()

    result = df.copy()

    # Ensure essential team ID columns are present on the input 'result' (which is df.copy())
    essential_df_cols = [home_team_col_param, away_team_col_param, home_hand_col_param, away_hand_col_param]
    # Note: home_hand_col_param and away_hand_col_param are used by _attach_vs_hand directly on 'result'
    # engine.py passes "home_starter_pitcher_handedness" for these params.
    # prediction.py ensures these "starter" columns are on the df passed to engine.py.
    # engine.py's KEEP_COLS_ALWAYS (if updated) should preserve them on loop_chunk_for_season.
    
    missing_essential_on_df = [c for c in essential_df_cols if c not in result.columns]
    if missing_essential_on_df:
        logger.error(
            f"ADVANCED.TRANSFORM: Input DataFrame 'df' is missing essential columns: {missing_essential_on_df}. "
            "These are needed for splits/vs_hand features. Returning copy of original df."
        )
        # Add placeholder columns for what would have been created to maintain schema if needed downstream,
        # though the values will be pure defaults.
        # This path means features won't be calculated correctly.
        # It's better to ensure these columns arrive correctly.
        # For now, returning result as is, subsequent calls might fail or use defaults.
        if debug: logger.setLevel(orig_level)
        return result


    # Build historical lookup (hist_filtered_indexed)
    hist_filtered_indexed = pd.DataFrame() # Default to empty
    if historical_team_stats_df is not None and not historical_team_stats_df.empty:
        hist_copy = historical_team_stats_df.copy()
        required_hist_cols_for_lookup = [hist_season_col_param, hist_team_col_param] # Base requirements

        # Filter to specific season if 'season_to_lookup' is provided and valid
        if season_to_lookup is not None and isinstance(season_to_lookup, int) and \
           hist_season_col_param in hist_copy.columns:
            logger.debug(f"ADVANCED.TRANSFORM: Filtering historical_team_stats_df to season_to_lookup = {season_to_lookup}")
            # Ensure season column is numeric before comparison
            hist_copy[hist_season_col_param] = pd.to_numeric(hist_copy[hist_season_col_param], errors='coerce')
            hist_copy = hist_copy[hist_copy[hist_season_col_param] == season_to_lookup]
        elif season_to_lookup is not None:
            logger.warning(f"ADVANCED.TRANSFORM: season_to_lookup ({season_to_lookup}) provided, but "
                           f"'{hist_season_col_param}' not in historical_team_stats_df or season_to_lookup not int. "
                           "Using all available seasons from historical_team_stats_df for lookup if not filtered otherwise.")

        if not all(col in hist_copy.columns for col in required_hist_cols_for_lookup):
            logger.error(
                f"ADVANCED.TRANSFORM: historical_team_stats_df (after potential season filter) "
                f"is missing one or more essential columns for lookup: {required_hist_cols_for_lookup}. "
                "Historical lookups will use defaults."
            )
        elif hist_copy.empty:
            logger.warning(
                 "ADVANCED.TRANSFORM: historical_team_stats_df became empty after filtering for "
                f"season_to_lookup={season_to_lookup if season_to_lookup is not None else 'all'}. "
                 "Historical lookups will use defaults."
            )
        else:
            if "team_norm" not in hist_copy.columns: # Create 'team_norm' if not present
                 hist_copy["team_norm"] = hist_copy[hist_team_col_param].apply(_safe_norm)
            
            # If multiple seasons are still in hist_copy (e.g., season_to_lookup was None),
            # drop_duplicates on "team_norm" will keep the first encountered season's stats for each team.
            # This might be okay if sorted by season descending, or if only one season is expected.
            # For now, assume season_to_lookup ensures we have one season's context.
            unique_team_stats = hist_copy.drop_duplicates(subset=["team_norm"], keep="first")
            if not unique_team_stats.empty:
                try:
                    hist_filtered_indexed = unique_team_stats.set_index("team_norm")
                    logger.info(
                        f"ADVANCED.TRANSFORM: Built hist_filtered_indexed (shape {hist_filtered_indexed.shape}) "
                        f"for lookup (using season context: {season_to_lookup if season_to_lookup is not None else 'first_available_per_team'})."
                    )
                except KeyError:
                    logger.error("ADVANCED.TRANSFORM: Failed to set 'team_norm' as index for hist_filtered_indexed. It might be missing.")
            else:
                logger.warning("ADVANCED.TRANSFORM: No unique team stats after filtering and drop_duplicates. Historical lookups will use defaults.")
    else: # historical_team_stats_df was None or empty
        logger.warning(
            "ADVANCED.TRANSFORM: historical_team_stats_df is None or empty. All historical lookups will use defaults."
        )

    # Home/Away split definitions (using parameters for hist_ stat names)
    split_map_defs = {
        "h_team_hist_HA_win_pct":           (hist_w_home_pct_param, "mlb_hist_win_pct_home",  0.5), # Default key more specific
        "h_team_hist_HA_runs_for_avg":      (hist_rf_home_param,    "mlb_hist_runs_for_home", 4.5),
        "h_team_hist_HA_runs_against_avg":  (hist_ra_home_param,    "mlb_hist_runs_against_home", 4.5),
        "a_team_hist_HA_win_pct":           (hist_w_away_pct_param, "mlb_hist_win_pct_away",  0.5),
        "a_team_hist_HA_runs_for_avg":      (hist_rf_away_param,    "mlb_hist_runs_for_away", 4.5),
        "a_team_hist_HA_runs_against_avg":  (hist_ra_away_param,    "mlb_hist_runs_against_away", 4.5),
    }
    # ... (calls to _attach_split as before) ...
    home_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("h_team_")}
    result = _attach_split(result, home_team_col_param, hist_filtered_indexed, home_split_defs, flag_imputations)
    away_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("a_team_")}
    result = _attach_split(result, away_team_col_param, hist_filtered_indexed, away_split_defs, flag_imputations)


    # Offense vs Pitcher Hand (using parameters for hand col names and hist_ stat names)
    result = _attach_vs_hand(
        result, home_team_col_param, away_hand_col_param, hist_filtered_indexed,
        "h_team_off_avg_runs_vs_opp_hand", hist_vs_lhp_param, hist_vs_rhp_param,
        "mlb_avg_runs_vs_hand_offense", 4.5, flag_imputations, # More specific default key
    )
    result = _attach_vs_hand(
        result, away_team_col_param, home_hand_col_param, hist_filtered_indexed,
        "a_team_off_avg_runs_vs_opp_hand", hist_vs_lhp_param, hist_vs_rhp_param,
        "mlb_avg_runs_vs_hand_offense", 4.5, flag_imputations, # More specific default key
    )

    # ... (Cleanup: drop_input_team_cols, convert _imputed flags to int, as before) ...
    if not flag_imputations:
        cols_to_drop = [c for c in result.columns if c.endswith("_imputed")]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop, errors="ignore")

    if drop_input_team_cols:
        cols_to_drop_ids = [col for col in [home_team_col_param, away_team_col_param] if col in result.columns]
        if cols_to_drop_ids:
            result = result.drop(columns=cols_to_drop_ids, errors="ignore")
            
    if flag_imputations: # Ensure this runs after all imputed columns are created
        for col_name in result.columns:
            if col_name.endswith("_imputed"):
                # Ensure the column is boolean before converting to int for safety
                if result[col_name].dtype == 'bool' or result[col_name].isin([True, False, 0, 1, 0.0, 1.0]).all():
                    result[col_name] = result[col_name].astype(bool).astype(int)
                else: # If it's not clearly boolean (e.g. object with other strings), default it
                    logger.warning(f"ADVANCED.TRANSFORM: Imputed flag column '{col_name}' has unexpected dtype {result[col_name].dtype}. Setting to default int 1 (True).")
                    result[col_name] = 1


    if debug:
        logger.setLevel(orig_level)
    logger.debug(f"ADVANCED.TRANSFORM: Complete. output_shape={result.shape}")
    return result