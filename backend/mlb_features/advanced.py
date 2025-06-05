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
    hist_lookup_df: pd.DataFrame,  # Indexed by unique 'team_norm'
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
    # --- START: New Debug Log ---
    logger.debug(
        #f"_attach_vs_hand ENTRY: new_col_name_base='{new_col_name_base}', "
        f"team_col_name_in_df='{team_col_name_in_df}', opp_hand_col_name_in_df='{opp_hand_col_name_in_df}'. "
        f"Columns in df_to_modify: {df_to_modify.columns.tolist()}"
    )
    if opp_hand_col_name_in_df in df_to_modify.columns:
        logger.debug(f"Sample of input opp_hand_col ('{opp_hand_col_name_in_df}') before processing:\n{df_to_modify[[opp_hand_col_name_in_df]].head().to_string()}")
    else:
        logger.debug(f"Input opp_hand_col ('{opp_hand_col_name_in_df}') is NOT in df_to_modify.columns upon entry.")
    # --- END: New Debug Log ---

    if team_col_name_in_df not in df_to_modify.columns:
        logger.error(
            f"Missing team column '{team_col_name_in_df}' for _attach_vs_hand '{new_col_name_base}'."
        )
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        logger.debug(f"_attach_vs_hand: Using default_to_use = {default_to_use} for key '{defaults_key}' (fallback {fallback_value})")
        df_to_modify[new_col_name_base] = default_to_use
        if flag_imputations:
            df_to_modify[f"{new_col_name_base}_imputed"] = True
        df_to_modify[new_col_name_base] = (
            pd.to_numeric(df_to_modify[new_col_name_base], errors="coerce")
              .fillna(default_to_use)
              .astype(float)
        )
        return df_to_modify

    if opp_hand_col_name_in_df not in df_to_modify.columns:
        logger.warning(
            f"Opposing‐hand column '{opp_hand_col_name_in_df}' missing for _attach_vs_hand '{new_col_name_base}'. "
            "Defaulting to UNKNOWN_HAND for all rows by creating the column." # Clarified log
        )
        df_to_modify[opp_hand_col_name_in_df] = "UNKNOWN_HAND" # Ensure column exists before astype

    normalized_team_ids = df_to_modify[team_col_name_in_df].apply(_safe_norm)
    
    # --- START: Modified opponent_hands processing ---
    opponent_hands = (
        df_to_modify[opp_hand_col_name_in_df]
        .astype(str)
        .str.upper() # Converts 'l', 'r' to 'L', 'R'; 'none' to 'NONE'
        .replace("", "UNKNOWN_HAND")
        .replace("NONE", "UNKNOWN_HAND") # ADD THIS
        .replace("NULL", "UNKNOWN_HAND") # ADD THIS (if 'NULL' string might appear)
        .fillna("UNKNOWN_HAND") 
    )
    logger.debug(f"Sample of processed 'opponent_hands' Series (first 5 values): {opponent_hands.head().tolist()}")
    # --- END: Modified opponent_hands processing ---

    default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
    # Initialize as NumPy arrays for direct indexed assignment
    calculated_values_arr = np.full(len(df_to_modify), default_to_use, dtype=float)
    imputation_flags_arr = np.full(len(df_to_modify), True, dtype=bool)

    # --- START: Modified loop for indexed assignment ---
    for i, (team_norm, opp_hand_val) in enumerate(zip(normalized_team_ids, opponent_hands)):
        logger.debug(f"Loop Iteration {i}: team_norm='{team_norm}', opp_hand_val='{opp_hand_val}' (type: {type(opp_hand_val)})")
        val_for_row = np.nan  # Reset for each row
        imputed_this_row = True # Assume imputed until a value is found

        if not hist_lookup_df.empty and team_norm in hist_lookup_df.index:
            stat_col_to_use: Optional[str] = None
            if opp_hand_val == "L" and hist_stat_vs_lhp in hist_lookup_df.columns:
                stat_col_to_use = hist_stat_vs_lhp
            elif opp_hand_val == "R" and hist_stat_vs_rhp in hist_lookup_df.columns:
                stat_col_to_use = hist_stat_vs_rhp
            elif opp_hand_val == "UNKNOWN_HAND": # Optional: Handle UNKNOWN_HAND explicitly if you have a general stat
                # Example: if you had a 'season_avg_runs_overall' column to use as a fallback here
                # This part is conceptual; current logic correctly falls to default_to_use if L/R specific not found
                logger.debug(f"Opponent hand is UNKNOWN_HAND for team_norm {team_norm}. Specific L/R lookup won't occur.")
                pass # Will use default_to_use unless a specific UNKNOWN_HAND stat column exists

            if stat_col_to_use:
                try:
                    looked_up_val = hist_lookup_df.at[team_norm, stat_col_to_use]
                    
                    if isinstance(looked_up_val, pd.Series):
                        # ... (handle series)
                        non_na_values = looked_up_val.dropna()
                        if not non_na_values.empty:
                            looked_up_val = non_na_values.iloc[0]
                        else:
                            looked_up_val = np.nan
                    
                    # NOW, 'season_to_lookup' (the function parameter) IS IN SCOPE HERE
                    if pd.notna(looked_up_val):
                        is_historical_team_vs_hand_stat = (stat_col_to_use == hist_stat_vs_lhp or stat_col_to_use == hist_stat_vs_rhp)
                        
                        # Use the season_to_lookup parameter from the function signature
                        if is_historical_team_vs_hand_stat and \
                           looked_up_val == 0.0 and \
                           season_to_lookup is not None and \
                           season_to_lookup < 2025: # Make sure 2025 is your actual cutoff year
                            
                            logger.debug(f"For {team_norm}, stat {stat_col_to_use} in season {season_to_lookup}, found 0.0. "
                                         "Treating as 'data not tracked' for pre-2025. Will use default.")
                            val_for_row = np.nan 
                            imputed_this_row = True
                        else:
                            val_for_row = looked_up_val
                            imputed_this_row = False
                    else:
                        logger.debug(f"Looked up value for {team_norm}, {stat_col_to_use if stat_col_to_use else 'N/A'} is NaN. Will use default: {default_to_use}")


                        # val_for_row remains np.nan (its initial state in the loop for this iteration)
                        # imputed_this_row remains True (its in
                except KeyError:
                    logger.warning(
                        f"KeyError accessing hist_lookup_df.at['{team_norm}', '{stat_col_to_use}']."
                        "This might indicate team_norm not in index or column not present after checks. Using default."
                    )
                    # val_for_row remains np.nan, will use default_to_use
                except Exception as e:
                    logger.error(f"Unexpected error during hist_lookup_df.at access for {team_norm}, {stat_col_to_use}: {e}. Using default.")
                    # val_for_row remains np.nan, will use default_to_use
            else:
                 # This path is taken if opp_hand_val is not 'L' or 'R' with corresponding valid columns,
                 # or if hist_stat_vs_lhp/rhp columns themselves are missing from hist_lookup_df.
                 logger.debug(f"No specific stat column to use for team_norm {team_norm} and opp_hand_val {opp_hand_val} (L/R cols: {hist_stat_vs_lhp in hist_lookup_df.columns}/{hist_stat_vs_rhp in hist_lookup_df.columns}). Will use default.")


        # Assign to the pre-allocated arrays
        calculated_values_arr[i] = val_for_row if pd.notna(val_for_row) else default_to_use
        if flag_imputations:
            imputation_flags_arr[i] = imputed_this_row
    # --- END: Modified loop for indexed assignment ---

    # Assign the NumPy arrays (converted to Series) to the DataFrame
    # Ensure pd.to_numeric is still applied to handle any non-float types that might sneak in, though dtype=float in np.full helps
    df_to_modify[new_col_name_base] = pd.to_numeric(
        pd.Series(calculated_values_arr, index=df_to_modify.index, name=new_col_name_base),
        errors='coerce'
    ).fillna(default_to_use).astype(float)

    if flag_imputations:
        df_to_modify[f"{new_col_name_base}_imputed"] = pd.Series(
            imputation_flags_arr, index=df_to_modify.index, name=f"{new_col_name_base}_imputed", dtype=bool
        ) # Ensure boolean type for imputation flags

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
    # --- START: New Debug Log for received season_to_lookup ---
    logger.debug(f"advanced.transform ENTRY: Received season_to_lookup = {season_to_lookup!r}, flag_imputations = {flag_imputations}, debug = {debug}")

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

        logger.debug(f"ADVANCED.TRANSFORM: hist_copy columns before season_to_lookup filter: {hist_copy.columns.tolist()}")
        logger.debug(f"ADVANCED.TRANSFORM: hist_season_col_param = '{hist_season_col_param}'")

        logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: season_to_lookup = {season_to_lookup!r} (type: {type(season_to_lookup)})")
        logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: hist_season_col_param = '{hist_season_col_param}' (type: {type(hist_season_col_param)})")
        if 'hist_copy' in locals() and hist_copy is not None and not hist_copy.empty: # Check if hist_copy is defined and not empty
            logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: hist_copy columns: {hist_copy.columns.tolist()}")
            logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: Does '{hist_season_col_param}' exist in hist_copy.columns? {(hist_season_col_param in hist_copy.columns)}")
            if hist_season_col_param in hist_copy.columns:
                logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: Sample of hist_copy['{hist_season_col_param}'] before to_numeric:\n{hist_copy[hist_season_col_param].head().to_string()}")
        elif 'hist_copy' in locals() and (hist_copy is None or hist_copy.empty):
             logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: hist_copy is defined but is None or empty.")
        else:
            logger.debug(f"ADVANCED.TRANSFORM: PRE-FILTER CHECK: hist_copy is not yet defined or encountered an issue.")


        # Filter to specific season if 'season_to_lookup' is provided and valid
        if season_to_lookup is not None and isinstance(season_to_lookup, (int, np.integer)) and \
        hist_season_col_param in hist_copy.columns:
            hist_copy[hist_season_col_param] = pd.to_numeric(hist_copy[hist_season_col_param], errors="coerce")
            hist_copy = hist_copy[hist_copy[hist_season_col_param] == season_to_lookup].copy()
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
    result,
    home_team_col_param,
    away_hand_col_param,
    hist_filtered_indexed, # This is now correctly 2024 data
    "h_team_off_avg_runs_vs_opp_hand",
    hist_vs_lhp_param,
    hist_vs_rhp_param,
    "mlb_avg_runs_vs_hand",
    4.5,
    flag_imputations,
    season_to_lookup=season_to_lookup # Pass it down
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