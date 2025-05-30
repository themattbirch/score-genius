# backend/nba_features/rest.py

"""
Wide‑coverage tests for backend.nba_features.rest.transform

Focus
-----
1.  Strict “no‑look‑ahead” guarantee when multiple games share a date.
2.  Correct rest‑day math, B2B flags, 7‑/14‑day density, and advantages.
3.  Idempotency + NaN‑free outputs.
"""

from __future__ import annotations
import logging
from typing import Mapping, Sequence # Use Sequence from typing for lists

import numpy as np
import pandas as pd

from .utils import DEFAULTS # Assuming convert_and_fill is not used here, but DEFAULTS is

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Output columns for rest and schedule features
PLACEHOLDER_COLS: Sequence[str] = [
    "rest_days_home", "rest_days_away",
    "games_last_7_days_home", "games_last_14_days_home",
    "games_last_7_days_away", "games_last_14_days_away",
    "is_back_to_back_home", "is_back_to_back_away",
    "rest_advantage", "schedule_advantage", "schedule_advantage_14d",
]

# Default rest if no prior game
DEF_REST: float = float(DEFAULTS.get("rest_days", 3.0))


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all placeholder columns exist and fill NaNs with sensible defaults.
    Also ensures correct data types for output columns.
    """
    df_out = df.copy()
    for col in PLACEHOLDER_COLS:
        # Determine default value
        if col.startswith("is_back_to_back"):
            default_value = 0
            final_dtype = int
        elif "games_last_" in col:
            default_value = 0
            final_dtype = int
        elif "advantage" in col:
            default_value = 0.0
            final_dtype = float
        else: # rest_days_home, rest_days_away
            default_value = DEF_REST
            final_dtype = float

        # Apply default and type
        if col not in df_out.columns:
            df_out[col] = default_value
        else:
            # Try to convert to numeric if it's not, before fillna.
            # This handles cases where a column might exist as 'object' type.
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                 try:
                    df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
                 except Exception: # Broad exception if to_numeric itself fails
                    pass 
            df_out[col] = df_out[col].fillna(default_value)
        
        df_out[col] = df_out[col].astype(final_dtype)
            
    return df_out


def transform(
    df: pd.DataFrame,
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Compute rest days, back-to-back flags, and schedule density for home/away teams.

    Args:
        df: DataFrame, must include 'game_id','game_date','home_team','away_team'.
        debug: If True, enable DEBUG logging for this function call.

    Returns:
        DataFrame with placeholder columns added, populated, and correctly typed.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("rest.transform: DEBUG on")

    try:
        if df is None or df.empty:
            logger.warning("rest.transform: empty input, returning schema only")
            return _fill_defaults(pd.DataFrame(columns=PLACEHOLDER_COLS))

        temp_cols_to_drop_later = [] # Keep track of temporary columns to drop at the end

        df_work = df.drop(columns=[c for c in PLACEHOLDER_COLS if c in df.columns], errors="ignore").copy()
        df_work = df_work.reset_index(drop=True)

        req = {"game_id", "game_date", "home_team", "away_team"}
        if not req.issubset(df_work.columns):
            missing = req - set(df_work.columns)
            logger.error(f"rest.transform: missing required columns {missing}")
            return _fill_defaults(df_work)

        df_work["game_date"] = pd.to_datetime(df_work["game_date"], errors="coerce").dt.tz_localize(None)
        if df_work["game_date"].isnull().any():
            logger.warning("rest.transform: NaT values present in 'game_date' after conversion. Results may be affected.")
        
        # Ensure team columns are string for reliable mapping/merging
        df_work["home_team"] = df_work["home_team"].astype(str)
        df_work["away_team"] = df_work["away_team"].astype(str)


        # Build long form log of games per team
        home_log = df_work[["home_team", "game_date", "game_id"]].rename(columns={"home_team": "team"})
        away_log = df_work[["away_team", "game_date", "game_id"]].rename(columns={"away_team": "team"})
        game_log = pd.concat([home_log, away_log], ignore_index=True)
        game_log = game_log.dropna(subset=["team", "game_date"])
        # Sort by game_id as tie-breaker for same-day games for consistent shift() and drop_duplicates()
        game_log = game_log.sort_values(["team", "game_date", "game_id"], ignore_index=True)
        
        # Create a DataFrame with unique team-date entries for prev_game_date calculation
        # This handles cases where a team might have multiple game_ids on the same date in the raw log due to data issues
        # We only want one "previous game date" per actual game day for a team.
        unique_team_dates = game_log.drop_duplicates(subset=["team", "game_date"], keep="first").copy()
        unique_team_dates["prev_game_date_calc"] = unique_team_dates.groupby("team")["game_date"].shift(1)
        
        # Prepare this prev_dates_df for merging
        prev_dates_df = unique_team_dates[['team', 'game_date', 'prev_game_date_calc']]

        for side in ("home", "away"):
            # Define temporary column name for previous game date
            prev_game_date_col = f"prev_{side}_game_date"
            temp_cols_to_drop_later.append(prev_game_date_col)

            df_work = df_work.merge(
                prev_dates_df,
                left_on=[f"{side}_team", "game_date"],
                right_on=["team", "game_date"],
                how="left"
            ).rename(columns={"prev_game_date_calc": prev_game_date_col}).drop(columns=["team"], errors="ignore")

            # Rest days
            rd_col = f"rest_days_{side}"
            df_work[rd_col] = (df_work["game_date"] - df_work[prev_game_date_col]).dt.days
            
            # Assign DEF_REST for first game per team (identified by game_id)
            # game_log still has all game_ids, first() after sort gives the earliest game_id
            first_game_id_map = game_log.groupby("team")["game_id"].first()
            is_first_game_mask = (df_work["game_id"] == df_work[f"{side}_team"].map(first_game_id_map))
            
            # Apply DEF_REST to NaNs (from true first games where prev_game_date_col is NaT)
            # AND explicitly to games identified as first_game by game_id_map.
            df_work.loc[is_first_game_mask, rd_col] = DEF_REST
            df_work[rd_col] = df_work[rd_col].fillna(DEF_REST) # Fill any other NaNs (e.g. if map missed)


        # Schedule density: Calculate counts for all teams once
        # Use unique_team_dates as it represents one activity per team per day
        log_indexed = unique_team_dates.set_index("game_date").sort_index() # Ensure index is sorted for rolling

        counts7_all = (
            log_indexed.groupby("team")["game_id"] # Count any column, e.g., 'team' itself after reset_index or game_id
            .rolling(window="7D", closed="neither") 
            .count()
            .rename("cnt7")
            .reset_index() # Converts 'game_date' from index back to column
        )
        counts14_all = (
            log_indexed.groupby("team")["game_id"]
            .rolling(window="14D", closed="neither")
            .count()
            .rename("cnt14")
            .reset_index()
        )
        
        sched_counts_all_teams = counts7_all.merge(
            counts14_all, on=["team", "game_date"], how="outer" 
        )
        # After merge, NaNs in cnt7/cnt14 mean 0 games in that window. These will be filled by _fill_defaults.

        for side in ("home", "away"):
            games_7d_col = f"games_last_7_days_{side}"
            games_14d_col = f"games_last_14_days_{side}"
            df_work = df_work.merge(
                sched_counts_all_teams, # Contains 'team', 'game_date', 'cnt7', 'cnt14'
                left_on=[f"{side}_team", "game_date"],
                right_on=["team", "game_date"],
                how="left"
            ).rename(columns={
                "cnt7": games_7d_col,
                "cnt14": games_14d_col,
            }).drop(columns=["team"], errors="ignore") # Drop 'team' column from sched_counts_all_teams

        # Back-to-back and advantages
        # These calculations might produce NaN if underlying rest_days or games_last_... are NaN
        # _fill_defaults will handle these NaNs.
        df_work["is_back_to_back_home"] = (df_work["rest_days_home"] == 1)
        df_work["is_back_to_back_away"] = (df_work["rest_days_away"] == 1)
        df_work["rest_advantage"] = df_work["rest_days_home"] - df_work["rest_days_away"]
        df_work["schedule_advantage"] = (
            df_work.get(f"games_last_7_days_away", pd.Series(0, index=df_work.index)) - 
            df_work.get(f"games_last_7_days_home", pd.Series(0, index=df_work.index))
        )
        df_work["schedule_advantage_14d"] = (
            df_work.get(f"games_last_14_days_away", pd.Series(0, index=df_work.index)) - 
            df_work.get(f"games_last_14_days_home", pd.Series(0, index=df_work.index))
        )
        
        df_work = df_work.drop(columns=temp_cols_to_drop_later, errors="ignore")

        final_df = _fill_defaults(df_work)
        return final_df

    finally:
        if debug:
            logger.setLevel(orig_level)