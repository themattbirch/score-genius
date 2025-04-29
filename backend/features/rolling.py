# backend/features/rolling.py

import pandas as pd
import logging
from .utils import DEFAULTS # Assuming DEFAULTS dict is in utils.py
from .base_windows import fetch_rolling # Assumes this function fetches from SQL view
# Import legacy class
from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Instantiate legacy engine once at module level for fallback
try:
    _legacy_engine_instance = LegacyFeatureEngine()
    logger.info("Legacy FeatureEngine instance created for fallback.")
except Exception as init_err:
    logger.error(f"Failed to instantiate LegacyFeatureEngine: {init_err}", exc_info=True)
    _legacy_engine_instance = None # Ensure fallback fails gracefully if init fails

def add_rolling_features(df: pd.DataFrame,
                         conn=None,
                         window_sizes: list[int] = [5, 10, 20] # Allow passing windows
                        ) -> pd.DataFrame:
    """
    Adds rolling features. Tries fetching from Supabase materialized view first.
    Falls back to legacy Python calculation if connection fails or returns empty.
    """
    # --- Fallback Path (Legacy Python Calculation) ---
    def run_legacy_fallback(input_df, reason=""):
        logger.warning(f"Falling back to legacy Python rolling feature calculation. Reason: {reason}")
        if _legacy_engine_instance is None:
            logger.error("Cannot run legacy fallback: Legacy FeatureEngine failed to initialize.")
            return input_df # Return original df without features
        try:
            # Call the METHOD on the instance
            return _legacy_engine_instance.add_rolling_features(input_df, window_sizes=window_sizes)
        except Exception as legacy_err:
            logger.error(f"Error during legacy fallback calculation: {legacy_err}", exc_info=True)
            # Return original df or empty df on complete failure? Return original for now.
            return input_df

    # --- Primary Path (SQL Materialized View) ---
    if conn is None: # No DB connection provided
        logger.info("No database connection provided. Using legacy rolling features.")
        return run_legacy_fallback(df, "No DB connection provided")

    try:
        logger.debug("Attempting to fetch rolling features from materialized view...")
        # Ensure game_id column exists before trying to access it
        if 'game_id' not in df.columns:
            logger.error("Input DataFrame missing 'game_id' column. Cannot fetch rolling features from SQL.")
            return run_legacy_fallback(df, "Missing game_id column in input")

        # Fetch data using unique game IDs
        game_ids = df['game_id'].astype(str).unique().tolist()
        if not game_ids:
            logger.warning("No unique game IDs found in input DataFrame. Skipping SQL fetch.")
            return run_legacy_fallback(df, "No game_ids in input")

        rolled = fetch_rolling(conn, game_ids) # Assumes fetch_rolling exists and works

        if rolled is None or rolled.empty:
            logger.warning("Fetching from materialized view returned empty or None.")
            return run_legacy_fallback(df, "SQL fetch returned empty/None")

        logger.info(f"Successfully fetched {len(rolled)} rows from materialized view for {len(game_ids)} unique games.")
        logger.debug(f"Sample of fetched rolling data:\n{rolled.head().to_string()}")


        # --- Merge SQL results ---
        logger.debug("Merging fetched rolling features into main DataFrame...")
        # Ensure necessary columns exist in the input 'df' for merging
        required_merge_cols = ['game_id', 'home_team_norm', 'away_team_norm'] # Assumes _norm cols added earlier
        if not all(col in df.columns for col in required_merge_cols):
             missing_cols_str = ", ".join(set(required_merge_cols) - set(df.columns))
             logger.error(f"Input df missing required columns for merging SQL results: {missing_cols_str}")
             return run_legacy_fallback(df, f"Missing merge columns: {missing_cols_str}")

        # Ensure types match for merge keys
        df_merged = df.copy() # Work on a copy
        df_merged['game_id'] = df_merged['game_id'].astype(str)
        rolled['game_id'] = rolled['game_id'].astype(str)

        # Check if 'team_norm' exists in rolled df - needed for join
        if 'team_norm' not in rolled.columns:
             logger.error("Fetched rolling data missing 'team_norm' column required for merge.")
             return run_legacy_fallback(df, "Missing 'team_norm' in SQL results")
        rolled['team_norm'] = rolled['team_norm'].astype(str)

        # Assuming _norm columns were added in a previous step correctly
        df_merged['home_team_norm'] = df_merged['home_team_norm'].astype(str)
        df_merged['away_team_norm'] = df_merged['away_team_norm'].astype(str)


        # Prepare the rolled data for merging by renaming columns for home/away context
        # Assumes rolled columns are like 'rolling_score_for_mean_20' etc.
        home_rename_map = {col: f"home_{col}" for col in rolled.columns if col not in ['game_id', 'team_norm', 'game_date']}
        away_rename_map = {col: f"away_{col}" for col in rolled.columns if col not in ['game_id', 'team_norm', 'game_date']}

        # Check if rename maps are empty (might happen if rolled only has key columns)
        if not home_rename_map or not away_rename_map:
             logger.warning("No rolling stat columns found in fetched data to rename/merge.")
             # Decide if fallback is needed or just return df_merged
             # If fetch_rolling succeeded but returned no stats, fallback might not be right.
             # Let's return the df for now, effectively adding no new rolling stats.
             return df_merged
             # Alternatively: return run_legacy_fallback(df, "No rolling stat columns in SQL results")


        rolled_home = rolled.rename(columns=home_rename_map)
        rolled_away = rolled.rename(columns=away_rename_map)
        logger.debug(f"Renamed home columns example: {list(rolled_home.columns[:5])}")
        logger.debug(f"Renamed away columns example: {list(rolled_away.columns[:5])}")


        # Merge for Home Team
        df_merged = pd.merge(df_merged, rolled_home.drop(columns=['game_date'], errors='ignore'), # Avoid date collision
                             left_on=['game_id', 'home_team_norm'],
                             right_on=['game_id', 'team_norm'],
                             how='left',
                             suffixes=('', '_h_dup'))
        # Merge for Away Team
        df_merged = pd.merge(df_merged, rolled_away.drop(columns=['game_date'], errors='ignore'), # Avoid date collision
                             left_on=['game_id', 'away_team_norm'],
                             right_on=['game_id', 'team_norm'],
                             how='left',
                             suffixes=('', '_a_dup'))

        # Drop temporary/duplicate columns generated by merge
        cols_to_drop = [col for col in df_merged.columns if '_dup' in col or col.startswith('team_norm')]
        df_merged = df_merged.drop(columns=cols_to_drop, errors='ignore')


        # --- Fill NaNs in newly merged columns ---
        # Identify columns added by the merge (those present in rename maps)
        new_home_cols = list(home_rename_map.values())
        new_away_cols = list(away_rename_map.values())
        all_new_rolling_cols = new_home_cols + new_away_cols

        logger.debug(f"Attempting to fill NaNs for {len(all_new_rolling_cols)} potentially new rolling columns...")

        nan_counts_before = df_merged[all_new_rolling_cols].isna().sum()
        cols_with_nans_before = nan_counts_before[nan_counts_before > 0].index.tolist()
        if cols_with_nans_before:
             logger.debug(f"Columns with NaNs before fill: {cols_with_nans_before}")

        for col in all_new_rolling_cols:
             if col in df_merged.columns:
                 # Try to determine base stat for default lookup
                 try:
                      parts = col.split('_') # e.g., ['home', 'rolling', 'score', 'for', 'mean', '20']
                      stat_type = parts[-2] # e.g., 'mean'
                      # Handle multi-word base stats like 'score_for'
                      base_stat_parts = []
                      for i in range(2, len(parts) - 2): # Iterate between 'rolling' and stat_type
                          if parts[i].isdigit(): # Stop if we hit window size prematurely
                              break
                          base_stat_parts.append(parts[i])
                      base_stat_name = "_".join(base_stat_parts)

                      default_key = base_stat_name
                      if stat_type == 'std':
                           default_key += '_std' # Try to get specific std default if available

                      default_val = DEFAULTS.get(default_key, 0.0)
                      df_merged[col] = df_merged[col].fillna(default_val)
                      # Ensure std dev is not negative
                      if stat_type == 'std':
                            df_merged[col] = df_merged[col].apply(lambda x: max(0, x) if pd.notna(x) else 0.0)

                 except Exception as parse_err:
                      logger.warning(f"Could not parse column '{col}' to find default value during NaN fill. Filling with 0.0. Error: {parse_err}")
                      df_merged[col] = df_merged[col].fillna(0.0)
             # else: Column wasn't even created by merge, something is wrong upstream.

        nan_counts_after = df_merged[all_new_rolling_cols].isna().sum().sum()
        logger.debug(f"NaN fill completed. Total NaNs remaining in new rolling columns: {nan_counts_after}")

        logger.info("Successfully merged rolling features from SQL view.")
        return df_merged.reset_index(drop=True) # Return with clean index

    except Exception as e:
        logger.error(f"Error during SQL rolling feature fetch/merge: {e}. Falling back to legacy.", exc_info=True)
        return run_legacy_fallback(df, f"Error in SQL path: {e}")