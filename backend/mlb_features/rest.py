# backend/mlb_features/rest.py
from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS
except ImportError:
    MLB_DEFAULTS: Dict[str, float] = {}

DEF_REST_DAYS = float(MLB_DEFAULTS.get("mlb_rest_days", 7.0))

def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
    **kwargs, # Absorb other kwargs from engine
) -> pd.DataFrame:
    if debug:
        logger.setLevel(logging.DEBUG)

    if df.empty:
        return df.copy()
    
    # The engine now provides df and historical_df with normalized columns,
    # including 'game_date_et', 'home_team_norm', and 'away_team_norm'.
    
    # --- Create a unified log of all team activities (past and present) ---
    def _rows(frame, side):
        return frame[[f"{side}_team_norm", "game_date_et"]].rename(
            columns={f"{side}_team_norm": "team"}
        )

    current_rows = pd.concat([_rows(df, "home"), _rows(df, "away")], ignore_index=True)
    
    if historical_df is not None and not historical_df.empty:
        historical_rows = pd.concat([_rows(historical_df, "home"), _rows(historical_df, "away")], ignore_index=True)
        long_log = pd.concat([historical_rows, current_rows], ignore_index=True)
    else:
        long_log = current_rows

    long_log.dropna(subset=["team", "game_date_et"], inplace=True)
    long_log.sort_values(["team", "game_date_et"], inplace=True)
    long_log.drop_duplicates(subset=["team", "game_date_et"], keep="first", inplace=True)

    # --- Calculate Previous Game Date and Rest Days ---
    long_log["prev_date"] = long_log.groupby("team")["game_date_et"].shift(1)

    # Merge previous dates back to the main DataFrame
    result = df.copy()
    for side in ("home", "away"):
        side_log = long_log.rename(columns={"team": f"{side}_team_norm", "prev_date": f"prev_{side}_date"})
        result = pd.merge(result, side_log, on=[f"{side}_team_norm", "game_date_et"], how="left")

    result["rest_days_home"] = (result["game_date_et"] - result["prev_home_date"]).dt.days
    result["rest_days_away"] = (result["game_date_et"] - result["prev_away_date"]).dt.days

    # --- Final Calculations and Cleanup ---
    result["is_back_to_back_home"] = (result["rest_days_home"] <= 1).astype(int)
    result["is_back_to_back_away"] = (result["rest_days_away"] <= 1).astype(int)
    result["rest_advantage"] = result["rest_days_home"] - result["rest_days_away"]
    
    # Fill NaNs that occur from a team's first game of the season
    result['rest_days_home'].fillna(DEF_REST_DAYS, inplace=True)
    result['rest_days_away'].fillna(DEF_REST_DAYS, inplace=True)
    result['rest_advantage'].fillna(0, inplace=True)
    
    # Select final columns to return
    final_cols = [
        "rest_days_home", "rest_days_away", "is_back_to_back_home",
        "is_back_to_back_away", "rest_advantage"
    ]
    
    # Merge the new feature columns back to the original df
    output = df.merge(result[['game_id'] + final_cols], on='game_id', how='left')
    
    return output