# backend/nba_features/data_processing_utils.py

import logging
from typing import Dict, Any, Sequence # Ensure Sequence is imported

import numpy as np
import pandas as pd
from supabase import Client # Ensure Client is imported for type hinting

# Assuming normalize_team_name and DEFAULTS are correctly imported from .utils
from .utils import normalize_team_name, DEFAULTS

logger = logging.getLogger(__name__)

# Exact stat columns from the RPC
RPC_OUTPUT_STAT_COLUMNS = [
    "games_played", "pace_avg", "off_rtg", "def_rtg", "net_rtg",
    "efg_pct", "tov_pct", "oreb_pct", "ft_rate"
]

# Special naming overrides for RPC stats
RPC_TARGET_NAMES: Dict[str, tuple[str, str]] = {
    "games_played": ("games_played_home", "games_played_away"),
    "pace_avg":     ("pace_home",           "pace_away"),
}

# --- THIS FUNCTION SHOULD BE PRESENT ---
def upsert_seasonal_splits_to_supabase(
    pivoted_df: pd.DataFrame,
    supabase_client: Client, 
    table_name: str = "nba_team_seasonal_advanced_splits"
) -> bool:
    """
    Upserts the pivoted seasonal advanced stats DataFrame into a Supabase table.
    """
    if pivoted_df is None or pivoted_df.empty:
        logger.info("Upsert: No data in pivoted_df to upsert. Skipping.")
        return True 

    records_to_upsert = pivoted_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')

    if not records_to_upsert: 
        logger.info("Upsert: Record list is empty after NA conversion. Skipping.")
        return True

    try:
        logger.info(f"Upserting {len(records_to_upsert)} records into '{table_name}'...") 
        
        on_conflict_cols_string = "team_norm,season" # Correct comma-separated string
        
        response = (
            supabase_client
            .from_(table_name)
            .upsert(
                records_to_upsert, 
                on_conflict=on_conflict_cols_string
            )
            .execute()
        )

        if hasattr(response, 'error') and response.error:
            logger.error(f"Supabase upsert error: {response.error}")
            return False
        
        logger.info(f"Successfully upserted data to '{table_name}'. Response data count: {len(response.data) if hasattr(response, 'data') else 'N/A'}")
        return True

    except Exception as e:
        logger.error(f"Upsert: Exception during Supabase upsert to '{table_name}': {e}", exc_info=True)
        return False


def pivot_rpc_output_to_seasonal_splits_df(
    rpc_df: pd.DataFrame,
    season_year: int
) -> pd.DataFrame:
    if rpc_df is None or rpc_df.empty:
        logger.warning("pivot_rpc_output: Received empty or None rpc_df. Returning empty DataFrame.")
        return pd.DataFrame()

    required_input_cols = ['team_name', 'is_home'] + RPC_OUTPUT_STAT_COLUMNS
    missing_cols = [c for c in required_input_cols if c not in rpc_df.columns]
    if missing_cols:
        logger.error(f"pivot_rpc_output: rpc_df is missing required columns: {missing_cols}. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Pivoting RPC output for season {season_year}...")
    df = rpc_df.copy()

    df['team_norm'] = df['team_name'].astype(str).map(normalize_team_name)
    df = df.dropna(subset=['team_norm'])
    if df.empty:
        logger.warning("pivot_rpc_output: DataFrame empty after team name normalization. Returning empty DataFrame.")
        return pd.DataFrame()
        
    df['is_home_flag'] = df['is_home'].astype(bool)

    try:
        pivot_df = (
            df.pivot_table(
                index=['team_norm', 'team_name'],
                columns='is_home_flag',
                values=RPC_OUTPUT_STAT_COLUMNS,
                aggfunc='first'
            )
        )
    except Exception as e:
        logger.error(f"pivot_rpc_output: Error during pivot_table operation: {e}", exc_info=True)
        return pd.DataFrame()

    if pivot_df.empty:
        logger.warning("pivot_rpc_output: Pivot table is empty.")
        return pd.DataFrame()

    new_column_names = []
    for stat_name_from_rpc, is_home_value in pivot_df.columns:
        is_home_bool = bool(is_home_value)
        home_target, away_target = RPC_TARGET_NAMES.get(
            stat_name_from_rpc,
            (f'{stat_name_from_rpc}_home', f'{stat_name_from_rpc}_away')
        )
        if is_home_bool:
            new_column_names.append(home_target)
        else:
            new_column_names.append(away_target)
    
    pivot_df.columns = new_column_names
    pivot_df = pivot_df.reset_index()
    pivot_df['season'] = season_year

    # Critical step: Ensure uniqueness on (team_norm, season) before final processing
    pivot_df = pivot_df.sort_values(by=['team_norm', 'season', 'team_name']) # Sort for deterministic keep='first'
    pivot_df = pivot_df.drop_duplicates(subset=['team_norm', 'season'], keep='first')
    logger.debug(f"Shape of pivot_df after dropping duplicates on (team_norm, season): {pivot_df.shape}")

    # Define all expected columns for the final DataFrame to ensure consistent schema
    expected_final_columns = ['team_norm', 'team_name', 'season']
    for rpc_stat_name in RPC_OUTPUT_STAT_COLUMNS:
        home_col, away_col = RPC_TARGET_NAMES.get(
            rpc_stat_name,
            (f'{rpc_stat_name}_home', f'{rpc_stat_name}_away')
        )
        expected_final_columns.append(home_col)
        expected_final_columns.append(away_col)
    
    # Reindex to ensure all expected columns exist and are in order, adding missing ones as NaN
    seasonal_splits_df = pivot_df.reindex(columns=expected_final_columns)

    # Fill defaults for any NaNs
    for col_name in seasonal_splits_df.columns:
        if col_name in ('team_norm', 'team_name', 'season'):
            continue

        base_stat_name = col_name.replace('_home', '').replace('_away', '')
        # Adjust for special cases like 'pace_avg' -> 'pace' or 'games_played'
        if col_name.startswith("pace_"): 
             base_stat_name = "pace_avg" 
        elif col_name.startswith("games_played_"):
             base_stat_name = "games_played"
        
        default_value = DEFAULTS.get(base_stat_name, 0.0)
        seasonal_splits_df[col_name] = seasonal_splits_df[col_name].fillna(default_value)

    # Final type casting
    for col_name in seasonal_splits_df.columns:
        if col_name in ('team_norm', 'team_name'):
            seasonal_splits_df[col_name] = seasonal_splits_df[col_name].astype(str)
        elif col_name == 'season':
            seasonal_splits_df[col_name] = pd.to_numeric(seasonal_splits_df[col_name], errors='coerce').fillna(0).astype(int)
        elif col_name.startswith("games_played_"):
            seasonal_splits_df[col_name] = pd.to_numeric(seasonal_splits_df[col_name], errors='coerce').fillna(0).astype(int)
        else: # All other stat columns (assumed numeric/float)
            seasonal_splits_df[col_name] = pd.to_numeric(seasonal_splits_df[col_name], errors='coerce').fillna(0.0).astype(float)

    logger.info(f"pivot_rpc_output: Created seasonal_splits_df. Shape={seasonal_splits_df.shape}.")
    return seasonal_splits_df