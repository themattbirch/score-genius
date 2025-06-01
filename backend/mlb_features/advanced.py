# backend/mlb_features/advanced.py

"""
Attaches advanced MLB features: home/way splits and offensive vs pitcher handedness.
"""

from __future__ import annotations
import logging
from typing import Any, Dict as TypingDict, Optional, Tuple

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
    from .utils import DEFAULTS as MLB_DEFAULTS_IMPORT, normalize_team_name
    MLB_DEFAULTS: TypingDict[str, float] = MLB_DEFAULTS_IMPORT
    logger.info("Imported MLB_DEFAULTS and normalize_team_name")
except ImportError:
    logger.warning("Could not import DEFAULTS or normalize_team_name; using fallbacks")
    MLB_DEFAULTS: TypingDict[str, float] = {}
    def normalize_team_name(team_id: Any) -> str:  # Fallback normalize_team_name
        return str(team_id).strip().lower() if pd.notna(team_id) else "unknown"

# ────── NORMALIZATION WRAPPER ──────
def _safe_norm(team_val: Any) -> str:
    """
    Guarantees a deterministic key for every team name.

    • First calls `normalize_team_name`.
    • If that returns one of the generic placeholders
      (“unknown”, “unknown_team”, or blank), fall back to the
      raw string, lower-cased and stripped.
    • All spaces are then removed to avoid accidental duplicates.
    """
    norm = normalize_team_name(team_val)
    if not isinstance(norm, str):
        norm = str(norm)
    if norm.lower() in {"unknown", "unknown_team"} or not norm.strip():
        norm = str(team_val).strip().lower()
    return norm.replace(" ", "")

# --- Module-level constants for default column names ---
game_id_col = "game_id"
home_team_col = "home_team_id"
away_team_col = "away_team_id"
home_hand_col = "home_probable_pitcher_handedness"
away_hand_col = "away_probable_pitcher_handedness"

# Historical‐stats columns
hist_team_col_default = "team_id"
hist_season_col_default = "season"
hist_w_home_pct_default = "wins_home_percentage"
hist_rf_home_default = "runs_for_avg_home"
hist_ra_home_default = "runs_against_avg_home"
hist_w_away_pct_default = "wins_away_percentage"
hist_rf_away_default = "runs_for_avg_away"
hist_ra_away_default = "runs_against_avg_away"
hist_vs_lhp_default = "season_avg_runs_vs_lhp"
hist_vs_rhp_default = "season_avg_runs_vs_rhp"


def _attach_split(
    df_to_modify: pd.DataFrame,
    team_col_name_in_df: str,
    hist_lookup_df: pd.DataFrame,  # Indexed by unique 'team_norm'
    col_mappings: TypingDict[str, Tuple[str, str, float]],
    flag_imputations: bool
) -> pd.DataFrame:
    """
    For each new_col_name in col_mappings, look up historical value by normalized team name.
    If missing or NaN, fallback to MLB_DEFAULTS[defaults_key] or fallback_value.
    """
    # If the input DataFrame lacks the required team column, fill with defaults immediately.
    if team_col_name_in_df not in df_to_modify.columns:
        logger.error(
            f"Team column '{team_col_name_in_df}' not found for _attach_split. Populating with defaults."
        )
        for new_col_name, (_, defaults_key, fallback_value) in col_mappings.items():
            default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
            df_to_modify[new_col_name] = default_to_use
            if flag_imputations:
                df_to_modify[f"{new_col_name}_imputed"] = True

            # Coerce to numeric (float) and fill
            df_to_modify[new_col_name] = (
                pd.to_numeric(df_to_modify[new_col_name], errors="coerce")
                .fillna(default_to_use)
                .astype(float)
            )
        return df_to_modify

    # Normalize team names (remove spaces after normalization)
    normalized_team_id_series = df_to_modify[team_col_name_in_df].apply(_safe_norm)


    for new_col_name, (hist_col_name, defaults_key, fallback_value) in col_mappings.items():
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        # Start with all imputed (True) by default
        imputed_flags_for_col = pd.Series([True] * len(df_to_modify), index=df_to_modify.index)

        if hist_col_name in hist_lookup_df.columns and not hist_lookup_df.empty:
            # Map normalized team names to the historical column
            hist_series = hist_lookup_df.get(hist_col_name, pd.Series(dtype=float))
            mapped_values = normalized_team_id_series.map(hist_series.to_dict())
            # Any row where mapped_values is NaN → imputed
            imputed_flags_for_col = mapped_values.isna()

            # Fill NaNs with default
            df_to_modify[new_col_name] = mapped_values.fillna(default_to_use)
        else:
            if hist_lookup_df.empty:
                logger.debug(f"Hist_lookup_df is empty for '{new_col_name}'. Using default.")
            else:
                logger.warning(
                    f"Column '{hist_col_name}' for '{new_col_name}' not in hist_lookup_df. Using default."
                )
            df_to_modify[new_col_name] = default_to_use
            # imputed_flags remain all True

        # If user wants imputation flags, attach them
        if flag_imputations:
            df_to_modify[f"{new_col_name}_imputed"] = imputed_flags_for_col.astype(bool)

        # Ensure numeric, fill any extra NaNs, cast to float
        df_to_modify[new_col_name] = (
            pd.to_numeric(df_to_modify[new_col_name], errors="coerce")
            .fillna(default_to_use)
            .astype(float)
        )

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
    flag_imputations: bool
) -> pd.DataFrame:
    """
    Attaches offense‐vs‐pitcher‐hand feature. For each row, look at that team's historical
    stat vs LHP or vs RHP based on the opponent's hand. If anything missing → default.
    """
    # If required columns missing, fill entire new_col_name_base with default
    if not all(c in df_to_modify.columns for c in [team_col_name_in_df, opp_hand_col_name_in_df]):
        logger.error(
            f"Missing input columns for _attach_vs_hand for {new_col_name_base}. Populating with defaults."
        )
        default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
        df_to_modify[new_col_name_base] = default_to_use
        if flag_imputations:
            df_to_modify[f"{new_col_name_base}_imputed"] = True

        df_to_modify[new_col_name_base] = (
            pd.to_numeric(df_to_modify[new_col_name_base], errors="coerce")
            .fillna(default_to_use)
            .astype(float)
        )
        return df_to_modify

    # Normalize team names (remove spaces)
    normalized_team_ids = df_to_modify[team_col_name_in_df].apply(_safe_norm)

    # Uppercase pitcher hand; any missing/invalid will default to "UNKNOWN_HAND"
    opponent_hands = (
        df_to_modify[opp_hand_col_name_in_df]
        .astype(str)
        .str.upper()
        .fillna("UNKNOWN_HAND")
    )

    default_to_use = MLB_DEFAULTS.get(defaults_key, fallback_value)
    calculated_values = []
    imputation_flags_list = []

    for team_norm, opp_hand_val in zip(normalized_team_ids, opponent_hands):
        val_for_row = np.nan
        imputed_this_row = True

        # Only attempt a lookup if hist_lookup_df is non-empty and team_norm is in index
        if not hist_lookup_df.empty and team_norm in hist_lookup_df.index:
            stat_col_to_use: Optional[str] = None
            if opp_hand_val == "L" and hist_stat_vs_lhp in hist_lookup_df.columns:
                stat_col_to_use = hist_stat_vs_lhp
            elif opp_hand_val == "R" and hist_stat_vs_rhp in hist_lookup_df.columns:
                stat_col_to_use = hist_stat_vs_rhp

            if stat_col_to_use:
                looked_up_val = hist_lookup_df.at[team_norm, stat_col_to_use]
                if pd.notna(looked_up_val):
                    val_for_row = looked_up_val
                    imputed_this_row = False

        # Append either the looked‐up number or the default
        calculated_values.append(val_for_row if pd.notna(val_for_row) else default_to_use)
        if flag_imputations:
            imputation_flags_list.append(imputed_this_row)

    df_to_modify[new_col_name_base] = (
        pd.Series(pd.to_numeric(calculated_values, errors="coerce"), index=df_to_modify.index)
        .fillna(default_to_use)
        .astype(float)
    )
    if flag_imputations:
        df_to_modify[f"{new_col_name_base}_imputed"] = pd.Series(
            imputation_flags_list, index=df_to_modify.index, dtype=bool
        )

    return df_to_modify


def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
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
) -> pd.DataFrame:
    """
    Main entry point for attaching all advanced MLB features:
      1) Home/Away splits (_attach_split)
      2) Offense vs Opponent Pitcher's Hand (_attach_vs_hand)
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"advanced.transform start: df_shape={df.shape}")

    # 1) If input is None or empty, return a (possibly empty) copy immediately
    if df is None or df.empty:
        logger.warning("Empty input df; returning copy.")
        return df.copy() if df is not None else pd.DataFrame()

    result = df.copy()

    # 2) Verify essential columns exist
    essential_cols_in_df = [
        game_id_col_param,
        home_team_col_param,
        away_team_col_param,
        home_hand_col_param,
        away_hand_col_param,
    ]
    missing_essential = [col for col in essential_cols_in_df if col not in result.columns]
    if missing_essential:
        logger.error(
            f"Input DataFrame 'df' is missing essential columns for advanced stats: "
            f"{missing_essential}. Returning copy of original df."
        )
        if debug:
            logger.setLevel(orig_level)
        return df.copy()

    # 3) Build a filtered, indexed historical DataFrame (by normalized team name sans spaces)
    hist_filtered_indexed: pd.DataFrame = pd.DataFrame()
    if historical_team_stats_df is not None and not historical_team_stats_df.empty:
        hist_copy = historical_team_stats_df.copy()
        if (hist_season_col_param in hist_copy.columns) and (hist_team_col_param in hist_copy.columns):
            # Normalize and remove spaces
            hist_copy["team_norm"] = hist_copy[hist_team_col_param].apply(_safe_norm)
            hist_for_season = hist_copy[hist_copy[hist_season_col_param] == season_to_lookup]

            # If duplicates, drop them (keep first)
            if not hist_for_season.empty:
                hist_for_season_unique = hist_for_season.drop_duplicates(subset=["team_norm"], keep="first")
                hist_filtered_indexed = hist_for_season_unique.set_index("team_norm")

            if hist_filtered_indexed.empty:
                logger.warning(
                    f"No historical data found for season {season_to_lookup} after filtering/indexing. "
                    f"All historical features will use defaults."
                )
        else:
            logger.error(
                f"Historical stats DataFrame missing '{hist_season_col_param}' or '{hist_team_col_param}'. "
                f"Cannot process historical stats."
            )
    else:
        logger.warning(
            "Historical team stats DataFrame is empty or None. All historical features will use defaults."
        )

    # 4) Home/Away split definitions
    split_map_defs = {
        "h_team_hist_HA_win_pct": (hist_w_home_pct_param, "mlb_hist_win_pct", 0.5),
        "h_team_hist_HA_runs_for_avg": (hist_rf_home_param, "mlb_hist_runs_avg", 4.5),
        "h_team_hist_HA_runs_against_avg": (hist_ra_home_param, "mlb_hist_runs_avg", 4.5),
        "a_team_hist_HA_win_pct": (hist_w_away_pct_param, "mlb_hist_win_pct", 0.5),
        "a_team_hist_HA_runs_for_avg": (hist_rf_away_param, "mlb_hist_runs_avg", 4.5),
        "a_team_hist_HA_runs_against_avg": (hist_ra_away_param, "mlb_hist_runs_avg", 4.5),
    }
    home_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("h_team_")}
    result = _attach_split(
        result,
        home_team_col_param,
        hist_filtered_indexed,
        home_split_defs,
        flag_imputations,
    )

    away_split_defs = {k: v for k, v in split_map_defs.items() if k.startswith("a_team_")}
    result = _attach_split(
        result,
        away_team_col_param,
        hist_filtered_indexed,
        away_split_defs,
        flag_imputations,
    )

    # 5) Offense vs Pitcher Hand
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
    )
    result = _attach_vs_hand(
        result,
        away_team_col_param,
        home_hand_col_param,
        hist_filtered_indexed,
        "a_team_off_avg_runs_vs_opp_hand",
        hist_vs_lhp_param,
        hist_vs_rhp_param,
        "mlb_avg_runs_vs_hand",
        4.5,
        flag_imputations,
    )

    # 6) If no imputation flags requested, drop any *_imputed columns
    if not flag_imputations:
        result = result.drop(columns=[c for c in result.columns if c.endswith("_imputed")], errors="ignore")

    # 7) Drop original home/away team columns
    cols_to_drop = []
    if home_team_col_param in result.columns:
        cols_to_drop.append(home_team_col_param)
    if away_team_col_param in result.columns:
        cols_to_drop.append(away_team_col_param)
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop, errors="ignore")

    # 8) Convert all *_imputed columns to standard Python bools (object dtype) 
    #    so tests that check `is True`/`is False` pass.
    if flag_imputations:
        for col_name in result.columns:
            if col_name.endswith("_imputed") and col_name in result.columns:
                python_bool_list = [bool(x) for x in result[col_name].tolist()]
                result[col_name] = pd.Series(python_bool_list, index=result.index, name=col_name, dtype="object")

    if debug:
        logger.setLevel(orig_level)
        logger.debug(f"advanced.transform complete: output_shape={result.shape}")

    return result
