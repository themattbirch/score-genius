# backend/mlb_features/rest.py
"""
Calculates rest-based features for MLB games.

Focus
-----
1.  Strict “no-look-ahead” guarantee when multiple games share a date.
2.  Correct rest-day math, B2B flags, 7-/14-day density, and advantages.
3.  Idempotency + NaN-free outputs.

MLB Specifics:
-   Uses 'game_date_et' for game dates.
-   Uses 'home_team_id' and 'away_team_id' for team identification.
-   Rest and schedule density metrics are generally applicable, though MLB game
    frequency is high. B2B (rest_days == 1) is a common scenario.
"""

from __future__ import annotations
import logging
from typing import List, Dict as TypingDict # Renamed to avoid conflict

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS_IMPORT # Use a different alias
    MLB_DEFAULTS: TypingDict[str, float] = MLB_DEFAULTS_IMPORT # Ensure it's a dict
except ImportError:
    logger.warning("Could not import DEFAULTS from .utils. Using local fallbacks.")
    MLB_DEFAULTS: TypingDict[str, float] = {}


DEF_REST_MLB = float(MLB_DEFAULTS.get("mlb_rest_days", 3.0))

PLACEHOLDER_COLS: List[str] = [
    "rest_days_home", "rest_days_away",
    "games_last_7_days_home", "games_last_14_days_home",
    "games_last_7_days_away", "games_last_14_days_away",
    "is_back_to_back_home", "is_back_to_back_away",
    "rest_advantage", "schedule_advantage",
]


def _fill_defaults_mlb(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure placeholder columns exist and contain no NaNs with correct dtypes."""
    df_filled = df.copy() # Work on a copy to avoid SettingWithCopyWarning if df is a slice
    for col in PLACEHOLDER_COLS:
        if col.startswith("is_back_to_back"):
            default_val = 0
            dtype = int
        elif col.startswith("games_last_"):
            default_val = 0
            dtype = int
        elif col.startswith("rest_days"):
            default_val = DEF_REST_MLB
            dtype = float
        else:  # advantages
            default_val = 0.0 # Ensure float for advantage columns
            dtype = float

        if col not in df_filled.columns:
            df_filled[col] = default_val
        else:
            # Ensure column is numeric before fillna, then fill, then astype
            df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce').fillna(default_val)
        
        # Apply astype(dtype) after creation or fillna
        df_filled[col] = df_filled[col].astype(dtype)
    return df_filled


def transform(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Calculates rest and schedule-based features for MLB games.

    Args:
        df: DataFrame with columns:
            - 'game_id'
            - 'game_date_et'
            - 'home_team_id', 'away_team_id'
        debug: enable DEBUG logging for this run.

    Returns:
        df with all PLACEHOLDER_COLS added and original key columns preserved.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Starting mlb_features.rest.transform in DEBUG mode")

    if df is None: # df.empty check comes after input column validation
        logger.warning("Null input DataFrame to rest.transform; returning empty DataFrame with placeholders.")
        return _fill_defaults_mlb(pd.DataFrame(columns=PLACEHOLDER_COLS))

    # --- Fix 1 (Part A): Define column mapping for internal use and for renaming back ---
    # Original input names to internal processing names
    col_map_to_internal = {
        "game_id": "game_id", # No change, but good to list
        "game_date": "game_date",
        "home_team_id": "home_team",
        "away_team_id": "away_team",
    }
    # Internal processing names to original input names (for final output)
    col_map_to_original = {v: k for k, v in col_map_to_internal.items()}


    # Check for required original columns before any operations
    missing_original_cols = [orig_col for orig_col in col_map_to_internal if orig_col not in df.columns]
    if missing_original_cols:
        logger.error(f"Missing required columns for rest.py: {missing_original_cols}. Input DataFrame returned with default placeholders.")
        # If original df is returned, it won't have placeholders. Test expects placeholders.
        # So, take original df, add/fill placeholders, then return.
        # df_with_placeholders = df.copy() # Avoid modifying input df if it's a slice
        # for placeholder_col in PLACEHOLDER_COLS: # Ensure placeholder columns are added before fill_defaults
        #     if placeholder_col not in df_with_placeholders.columns:
        #         df_with_placeholders[placeholder_col] = np.nan # Add as NaN so _fill_defaults handles it
        # return _fill_defaults_mlb(df_with_placeholders)
        
        # Simpler: if required cols are missing, return original df + default placeholder columns.
        # The test test_missing_required_cols_mlb expects this behavior.
        df_to_fill = df.copy()
        return _fill_defaults_mlb(df_to_fill)


    # Idempotency: Clean up any prior runs of placeholder/intermediate columns
    # Use internal names for prev_game_date as they are purely intermediate
    intermediate_cols_to_drop = ["prev_home_game_date", "prev_away_game_date"]
    cols_to_drop_for_idempotency = PLACEHOLDER_COLS + intermediate_cols_to_drop
    
    # Operate on a copy
    current_df = df.copy()
    
    # Drop existing derived/intermediate columns
    cols_actually_in_df_to_drop = [c for c in cols_to_drop_for_idempotency if c in current_df.columns]
    if cols_actually_in_df_to_drop:
        current_df = current_df.drop(columns=cols_actually_in_df_to_drop, errors="ignore")
    
    # Rename to internal column names for processing
    current_df = current_df.rename(columns=col_map_to_internal)
    
    # Process game_date
    current_df["game_date"] = pd.to_datetime(current_df["game_date"], errors="coerce").dt.tz_localize(None)
    
    # Drop rows with invalid game_date *after* rename and conversion
    if current_df["game_date"].isna().any():
        n_bad = int(current_df["game_date"].isna().sum())
        logger.warning(f"Dropping {n_bad} rows with invalid game_date")
        current_df = current_df.dropna(subset=["game_date"])
        if current_df.empty:
            logger.warning("DataFrame became empty after dropping NaNs in game_date.")
            # Return empty frame with placeholders and original key column names
            empty_with_placeholders = _fill_defaults_mlb(pd.DataFrame(columns=list(col_map_to_original.values()) + PLACEHOLDER_COLS))
            # Ensure only original names from col_map_to_original keys if they existed in input df
            original_keys_present = [col_map_to_original[k] for k in col_map_to_original if col_map_to_original[k] in df.columns]
            return _fill_defaults_mlb(pd.DataFrame(columns=original_keys_present + PLACEHOLDER_COLS))


    if current_df.empty: # Handles case where input df was empty but had required columns
        logger.warning("Empty input to rest.transform (or became empty after date processing); returning placeholders only.")
        # Ensure original column names are preserved if possible, plus placeholders
        original_keys_present = [k for k in df.columns if k in col_map_to_internal.keys()] # Original names that were in input df
        return _fill_defaults_mlb(pd.DataFrame(columns=original_keys_present + PLACEHOLDER_COLS))

    # Ensure team columns are string type for reliable grouping/mapping
    current_df["home_team"] = current_df["home_team"].astype(str)
    current_df["away_team"] = current_df["away_team"].astype(str)
    current_df = current_df.reset_index(drop=True) # Ensure unique default index after potential drops

    # Build team–game log using internal names
    home_log = current_df[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away_log = current_df[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    full_log = pd.concat([home_log, away_log], ignore_index=True)

    # Compute previous game date per team
    prev_log = (
        full_log
        .sort_values(["team", "game_date", "game_id"], ignore_index=True)
        .drop_duplicates(["team", "game_date"], keep="first") # One entry per team per day
    )
    prev_log["prev_game_date"] = prev_log.groupby("team")["game_date"].shift(1)
    prev_map = prev_log.set_index(["team", "game_date"])["prev_game_date"]

    # Merge prev_game_date for both sides
    for side in ("home", "away"):
        current_df = current_df.merge(
            prev_map.rename("prev_game_date_temp"), # Use temp name to avoid conflict if column exists
            left_on=(f"{side}_team", "game_date"),
            right_index=True,
            how="left",
        ).rename(columns={"prev_game_date_temp": f"prev_{side}_game_date"})


    # Calculate raw rest days
    current_df["rest_days_home"] = (current_df["game_date"] - current_df["prev_home_game_date"]).dt.days
    current_df["rest_days_away"] = (current_df["game_date"] - current_df["prev_away_game_date"]).dt.days

    # Identify each team’s first game_id in the processed dataset
    first_game_for_team_map = ( 
        full_log # Based on current_df's valid games
        .sort_values(["team", "game_date", "game_id"])
        .drop_duplicates("team", keep="first")
        .set_index("team")["game_id"]
    )

    home_team_first_game_ids = current_df["home_team"].map(first_game_for_team_map)
    away_team_first_game_ids = current_df["away_team"].map(first_game_for_team_map)

    current_df.loc[current_df["game_id"].isin(home_team_first_game_ids[current_df["game_id"] == home_team_first_game_ids]), "rest_days_home"] = DEF_REST_MLB
    current_df.loc[current_df["game_id"].isin(away_team_first_game_ids[current_df["game_id"] == away_team_first_game_ids]), "rest_days_away"] = DEF_REST_MLB
    
    # General fillna for any other NaNs in rest_days (e.g. team not in map, or prev_game_date was NaT for other reasons)
    current_df["rest_days_home"] = current_df["rest_days_home"].fillna(DEF_REST_MLB)
    current_df["rest_days_away"] = current_df["rest_days_away"].fillna(DEF_REST_MLB)


    # Schedule density: count games in the past 7/14 days (exclusive of current game_date)
    # Ensure sched_log is based on the current_df's processed game_dates
    sched_log = (
        full_log # Derived from current_df which has valid, processed game_date
        .drop_duplicates(["team", "game_date"], keep="first") # One game per team per day
        .set_index("game_date") # Index must be DatetimeIndex for rolling
        .sort_index()
    )
    if not sched_log.empty:
        # --- Fix 2: Change closed="neither" to closed="left" ---
        cnt7 = (
            sched_log.groupby("team")["game_id"] # Use any non-null column for count
            .rolling("7D", closed="left") 
            .count()
            .rename("cnt7")
        )
        cnt14 = (
            sched_log.groupby("team")["game_id"]
            .rolling("14D", closed="left")
            .count()
            .rename("cnt14")
        )
        # Align multi-index from rolling back to columns for merge
        sched_counts = pd.concat([cnt7, cnt14], axis=1).reset_index()
        
        for side in ("home", "away"):
            current_df = current_df.merge(
                sched_counts, # Contains 'team', 'game_date', 'cnt7', 'cnt14'
                left_on=(f"{side}_team", "game_date"),
                right_on=("team", "game_date"), # Match on team and the game's date
                how="left",
            ).rename(columns={
                "cnt7": f"games_last_7_days_{side}",
                "cnt14": f"games_last_14_days_{side}",
            }).drop(columns=["team"], errors="ignore") # Drop the 'team' column from sched_counts after merge
    else: # If sched_log is empty (e.g. current_df was empty)
        for side in ("home", "away"):
            current_df[f"games_last_7_days_{side}"] = 0
            current_df[f"games_last_14_days_{side}"] = 0

    # Fill missing density and ensure types (redundant if _fill_defaults_mlb is comprehensive)
    # for side in ("home", "away"):
    #     current_df[f"games_last_7_days_{side}"] = current_df[f"games_last_7_days_{side}"].fillna(0).astype(int)
    #     current_df[f"games_last_14_days_{side}"] = current_df[f"games_last_14_days_{side}"].fillna(0).astype(int)

    # B2B flags and advantage metrics
    current_df["is_back_to_back_home"] = (current_df["rest_days_home"] == 1.0).astype(int) # Compare float to float
    current_df["is_back_to_back_away"] = (current_df["rest_days_away"] == 1.0).astype(int) # Compare float to float
    current_df["rest_advantage"] = current_df["rest_days_home"] - current_df["rest_days_away"]
    # schedule_advantage: positive if away team played more games (i.e., home team has advantage)
    current_df["schedule_advantage"] = current_df.get(f"games_last_7_days_away", 0) - current_df.get(f"games_last_7_days_home", 0)


    # Cleanup intermediate columns
    current_df.drop(columns=intermediate_cols_to_drop, errors="ignore", inplace=True)

    # --- Fix 3 (Applied within _fill_defaults_mlb) ---
    # Final default-filling and type enforcement for all placeholder columns
    out_df = _fill_defaults_mlb(current_df)

    # --- Fix 1 (Part B): Rename columns back to original schema before returning ---
    out_df = out_df.rename(columns=col_map_to_original, errors="ignore")


    logger.debug("Finished rest.transform; output shape %s", out_df.shape)
    if debug:
        logger.setLevel(orig_level)
    return out_df