# backend/mlb_features/advanced.py
"""
Attaches advanced MLB features: home/away splits and offensive vs pitcher handedness.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name
    logger.info("Imported MLB_DEFAULTS and normalize_team_name")
except ImportError:
    logger.warning("Could not import DEFAULTS or normalize_team_name; using fallbacks")
    MLB_DEFAULTS: Dict[str, float] = {}
    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip() if pd.notna(team_id) else "unknown"


def _attach_split(
    df: pd.DataFrame,
    team_col_key: str, # e.g., 'home_team_id' or 'away_team_id' (from main df)
    hist_lookup_df: pd.DataFrame, # This is the 'hist' df, indexed by 'team_norm'
    col_mappings: Dict[str, Tuple[str, str, float]], # new_col -> (hist_col_name, defaults_key, fallback_val)
    flag_imputations: bool
) -> pd.DataFrame:
    """Helper to attach home/away split stats."""
    # Get the series of normalized team IDs for mapping
    # Assumes team_col_key refers to a column in df with the original team IDs
    normalized_team_id_series = df[team_col_key].apply(normalize_team_name)

    for new_col_name, (hist_col_name, defaults_key, fallback_value) in col_mappings.items():
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)

        if hist_col_name in hist_lookup_df.columns:
            # Map normalized team IDs to the historical stat column
            mapped_values = normalized_team_id_series.map(hist_lookup_df[hist_col_name])
            
            if flag_imputations:
                df[f"{new_col_name}_imputed"] = mapped_values.isna()
            
            df[new_col_name] = mapped_values.fillna(default_to_use)
        else:
            # If the historical column doesn't even exist in hist_lookup_df
            logger.warning(f"Column '{hist_col_name}' for feature '{new_col_name}' not found in historical_team_stats_df. Using default.")
            df[new_col_name] = default_to_use
            if flag_imputations:
                df[f"{new_col_name}_imputed"] = True # Definitely imputed

        df[new_col_name] = pd.to_numeric(df[new_col_name], errors='coerce').fillna(default_to_use).astype(float)
    return df


def _attach_vs_hand(
    df: pd.DataFrame,
    team_col: str,
    opp_hand_col: str,
    hist: pd.DataFrame,
    def_cols: Dict[str, Tuple[str, str, float]]
) -> pd.DataFrame:
    """
    Attach offensive stats vs LHP/RHP.
    def_cols: new_col -> (vs_L_col, vs_R_col, defaults_key, fallback)
    """
    norm = df[team_col].map(normalize_team_name)
    for new_col, (vs_L, vs_R, def_key, fallback) in def_cols.items():
        default = MLB_DEFAULTS.get(def_key, fallback)
        values = []
        imputed = []
        for team, hand in zip(norm, df[opp_hand_col].str.upper().fillna('')):
            if team in hist.index and hand in ('L', 'R') and vs_L in hist.columns and vs_R in hist.columns:
                val = hist.at[team, vs_L if hand == 'L' else vs_R]
            else:
                val = np.nan
            if pd.isna(val):
                values.append(default)
                imputed.append(True)
            else:
                values.append(val)
                imputed.append(False)
        df[new_col] = pd.to_numeric(values, errors='coerce').astype(float)
        df[f"{new_col}_imputed"] = imputed
    return df


def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
    season_to_lookup: int,
    flag_imputations: bool = True,
    debug: bool = False,
    # schedule cols
    game_id_col: str = "game_id",
    home_team_col: str = "home_team_id",
    away_team_col: str = "away_team_id",
    home_hand_col: str = "home_probable_pitcher_handedness",
    away_hand_col: str = "away_probable_pitcher_handedness",
    # hist splits
    hist_team_col: str = "team_id",
    hist_season_col: str = "season",
    hist_w_home_pct: str = "wins_home_percentage",
    hist_rf_home: str = "runs_for_avg_home",
    hist_ra_home: str = "runs_against_avg_home",
    hist_w_away_pct: str = "wins_away_percentage",
    hist_rf_away: str = "runs_for_avg_away",
    hist_ra_away: str = "runs_against_avg_away",
    # vs hand
    hist_vs_lhp: str = "season_avg_runs_vs_lhp",
    hist_vs_rhp: str = "season_avg_runs_vs_rhp",
) -> pd.DataFrame:
    """
    Attach advanced MLB features: HA splits + offense vs pitcher hand.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"advanced.transform start: df_shape={df.shape}")

    if df is None or df.empty:
        logger.warning("Empty input df; returning copy.")
        return df.copy()

    # Prepare hist
    hist = historical_team_stats_df.copy()
    hist = hist[hist[hist_season_col] == season_to_lookup]
    hist['team_norm'] = hist[hist_team_col].apply(normalize_team_name)
    hist = hist.set_index('team_norm')

    result = df.copy()
    # Home/Away splits
    split_map = {
        'h_team_hist_HA_win_pct': (hist_w_home_pct, 'mlb_hist_win_pct', 0.5),
        'h_team_hist_HA_runs_for_avg': (hist_rf_home, 'mlb_hist_runs_avg', 4.5),
        'h_team_hist_HA_runs_against_avg': (hist_ra_home, 'mlb_hist_runs_avg', 4.5),
        'a_team_hist_HA_win_pct': (hist_w_away_pct, 'mlb_hist_win_pct', 0.5),
        'a_team_hist_HA_runs_for_avg': (hist_rf_away, 'mlb_hist_runs_avg', 4.5),
        'a_team_hist_HA_runs_against_avg': (hist_ra_away, 'mlb_hist_runs_avg', 4.5),
    }
    home_split_definitions = {k: v for k, v in split_map.items() if k.startswith('h_team_')}
    result = _attach_split(result, home_team_col, hist, home_split_definitions, flag_imputations)

    away_split_definitions = {k: v for k, v in split_map.items() if k.startswith('a_team_')}
    result = _attach_split(result, away_team_col, hist, away_split_definitions, flag_imputations)
    
    # Offense vs hand
    hand_map = {
        'h_team_off_avg_runs_vs_opp_hand': (hist_vs_lhp, hist_vs_rhp, 'mlb_avg_runs_vs_hand', 4.5),
        'a_team_off_avg_runs_vs_opp_hand': (hist_vs_lhp, hist_vs_rhp, 'mlb_avg_runs_vs_hand', 4.5),
    }
    result = _attach_vs_hand(result, home_team_col, away_hand_col, hist, {'h_team_off_avg_runs_vs_opp_hand': hand_map['h_team_off_avg_runs_vs_opp_hand']})
    result = _attach_vs_hand(result, away_team_col, home_hand_col, hist, {'a_team_off_avg_runs_vs_opp_hand': hand_map['a_team_off_avg_runs_vs_opp_hand']})

    if not flag_imputations:
        # drop *_imputed columns
        result = result.drop(columns=[c for c in result.columns if c.endswith('_imputed')], errors='ignore')

    # Cleanup
    result = result.drop(columns=[home_team_col, away_team_col], errors='ignore')

    if debug:
        logger.debug(f"advanced.transform complete: output_shape={result.shape}")
    return result
