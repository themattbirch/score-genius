# backend/mlb_features/advanced.py

from __future__ import annotations
import logging
from typing import Any, Dict as TypingDict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name
except ImportError:
    MLB_DEFAULTS = {}
    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip().lower() if pd.notna(team_id) else "unknown"

def _safe_norm(team_val: Any) -> str:
    norm = normalize_team_name(team_val)
    return str(norm).strip().lower().replace(" ", "") if pd.notna(norm) else "unknown"

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
    # season_to_lookup is passed by the engine
    **kwargs
) -> pd.DataFrame:
    """
    Vectorized and robust version to attach offense-vs-pitcher-hand stats.
    Initializes columns first to prevent KeyErrors, then updates with looked-up data.
    """
    df = df_to_modify.copy()

    # 1. Initialize the output columns with default values first. This prevents KeyErrors.
    default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
    df[new_col_name_base] = default_to_use
    if flag_imputations:
        df[f"{new_col_name_base}_imputed"] = 1

    # 2. If there's no historical data to look up, return the defaults.
    if hist_lookup_df.empty or not (hist_stat_vs_lhp in hist_lookup_df.columns and hist_stat_vs_rhp in hist_lookup_df.columns):
        logger.warning(f"_attach_vs_hand: historical lookup unavailable for '{new_col_name_base}'. Using defaults.")
        return df

    # 3. Prepare for merging
    df['_norm_team_for_merge'] = df[team_col_name_in_df].apply(_safe_norm)
    df['_opp_hand'] = df[opp_hand_col_name_in_df].astype(str).str.upper().fillna('U').replace(["", "NONE", "NULL"], "U")
    
    # 4. Perform a single merge to get all potential stats
    lookup_cols = [hist_stat_vs_lhp, hist_stat_vs_rhp]
    merged_df = df.merge(
        hist_lookup_df[lookup_cols],
        left_on='_norm_team_for_merge',
        right_index=True,
        how='left'
    )

    # 5. Use vectorized `np.where` to select the correct stat and update the columns
    # This is much faster and safer than iterating.
    
    # Condition for using LHP stats
    is_vs_lhp = (merged_df['_opp_hand'] == 'L')
    # Condition for using RHP stats
    is_vs_rhp = (merged_df['_opp_hand'] == 'R')

    # Get the series of looked-up values
    lhp_values = merged_df[hist_stat_vs_lhp]
    rhp_values = merged_df[hist_stat_vs_rhp]

    # Update the main feature column based on conditions
    df[new_col_name_base] = np.where(
        is_vs_lhp, lhp_values, np.where(is_vs_rhp, rhp_values, np.nan)
    )

    # Update the imputation flag based on where the chosen lookup was successful
    if flag_imputations:
        df[f"{new_col_name_base}_imputed"] = df[new_col_name_base].isna().astype(int)

    # Fill any remaining NaNs (for Unknown hands or failed lookups) with the default
    df[new_col_name_base].fillna(default_to_use, inplace=True)

    # Clean up helper columns
    return df.drop(columns=['_norm_team_for_merge', '_opp_hand'], errors='ignore')

def _attach_split(df_to_modify: pd.DataFrame, team_col: str, lookup_df: pd.DataFrame, mappings: TypingDict[str, Tuple[str, str, float]], flag_imputations: bool) -> pd.DataFrame:
    df = df_to_modify.copy()
    if team_col not in df.columns:
        logger.error(f"_attach_split: Essential team column '{team_col}' not found.")
        # Create all columns with defaults if the team column is missing
        for new_name, (_, default_key, default_val) in mappings.items():
            df[new_name] = MLB_DEFAULTS.get(default_key, default_val)
            if flag_imputations:
                df[f"{new_name}_imputed"] = 1
        return df

    df["_merge_key"] = df[team_col].apply(_safe_norm)
    
    hist_cols_to_get = {details[0] for details in mappings.values()}
    valid_hist_cols = [col for col in hist_cols_to_get if col in lookup_df.columns]

    if valid_hist_cols and not lookup_df.empty:
        df = df.merge(lookup_df[valid_hist_cols], left_on='_merge_key', right_index=True, how='left')

    for new_name, (hist_name, default_key, default_val) in mappings.items():
        default_to_use = MLB_DEFAULTS.get(default_key, default_val)
        
        if hist_name in df.columns:
            if flag_imputations:
                df[f"{new_name}_imputed"] = df[hist_name].isna().astype(int)
            df[new_name] = df.pop(hist_name).fillna(default_to_use)
        else:
            df[new_name] = default_to_use
            if flag_imputations:
                df[f"{new_name}_imputed"] = 1
    
    return df.drop(columns=['_merge_key'], errors='ignore')


def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Lean transform function focused on Home/Away splits.
    The complex vs-Hand logic has been removed for performance and stability.
    """
    result = df.copy()
    
    # Use passed params or fall back to defaults
    flag_imputations = kwargs.get("flag_imputations", True)
    home_team_col_param = kwargs.get("home_team_col_param", "home_team_id")
    away_team_col_param = kwargs.get("away_team_id", "away_team_id")
    season_to_lookup = kwargs.get("season_to_lookup")
    
    if historical_team_stats_df is None or historical_team_stats_df.empty:
        return result

    hist_lookup = historical_team_stats_df.copy()
    if "team_norm" not in hist_lookup.columns:
        hist_lookup['team_norm'] = hist_lookup['team_id'].apply(_safe_norm)
    
    if season_to_lookup is not None and 'season' in hist_lookup.columns:
        hist_lookup = hist_lookup[hist_lookup['season'] == season_to_lookup]

    hist_lookup = hist_lookup.drop_duplicates('team_norm').set_index('team_norm')

    split_map_defs = {
        "h_team_hist_HA_win_pct": ("wins_home_percentage", "mlb_win_pct", 0.5),
        "h_team_hist_HA_runs_for_avg": ("runs_for_avg_home", "mlb_avg_runs", 4.5),
        "h_team_hist_HA_runs_against_avg": ("runs_against_avg_home", "mlb_avg_runs", 4.5),
        "a_team_hist_HA_win_pct": ("wins_away_percentage", "mlb_win_pct", 0.5),
        "a_team_hist_HA_runs_for_avg": ("runs_for_avg_away", "mlb_avg_runs", 4.5),
        "a_team_hist_HA_runs_against_avg": ("runs_against_avg_away", "mlb_avg_runs", 4.5),
    }
    
    home_defs = {k: v for k, v in split_map_defs.items() if k.startswith("h_")}
    away_defs = {k: v for k, v in split_map_defs.items() if k.startswith("a_")}

    result = _attach_split(result, home_team_col_param, hist_lookup, home_defs, flag_imputations)
    result = _attach_split(result, away_team_col_param, hist_lookup, away_defs, flag_imputations)
    
    return result