# backend/mlb_score_prediction/utils.py

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)
EPSILON = 1e-6

# --- DataFrame Operations ---

def convert_and_fill(df: pd.DataFrame, columns: List[str], default: float = 0.0) -> pd.DataFrame:
    """Convert specified columns to numeric and fill NaNs with a default value."""
    if df is None:
        logger.error("Input DataFrame is None in convert_and_fill.")
        return pd.DataFrame()

    df_copy = df.copy()
    for col in columns:
        if col not in df_copy.columns:
            df_copy[col] = default
            logger.warning(f"Column '{col}' not found in convert_and_fill. Added with default value {default}.")
        else:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(default)
    return df_copy

def generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """
    Generate standardized rolling feature column name (prefix can be 'home', 'away', or '').

    Args:
        prefix: 'home', 'away', or '' for team perspective.
        base: The base statistic name (e.g., 'woba', 'era', 'runs_scored').
        stat_type: 'mean' or 'std'.
        window: The rolling window size.

    Returns:
        The standardized column name string.
    """
    prefix_part = f"{prefix}_" if prefix else ""
    return f"{prefix_part}rolling_{base}_{stat_type}_{window}"

def slice_dataframe_by_features(df: pd.DataFrame, feature_list: List[str], fill_value: float = 0.0) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame containing only the columns specified in feature_list,
    ordered correctly. Adds missing columns filled with fill_value.
    """
    if df is None:
        logger.error("Input DataFrame is None in slice_dataframe_by_features.")
        return None
    if not feature_list:
        logger.warning("Empty feature_list provided to slice_dataframe_by_features. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    df_copy = df.copy()
    current_cols = set(df_copy.columns)
    missing_features = set(feature_list) - current_cols
    if missing_features:
        logger.warning(f"slice_dataframe: Missing features {missing_features}. Filling with {fill_value}.")
        for col in missing_features:
            df_copy[col] = fill_value

    extra_features = current_cols - set(feature_list)
    if extra_features:
        logger.debug(f"slice_dataframe: Input df has extra columns not in feature_list: {extra_features}")

    try:
        return df_copy.reindex(columns=feature_list, fill_value=fill_value)
    except Exception as e:
        logger.error(f"Error reindexing DataFrame in slice_dataframe_by_features: {e}", exc_info=True)
        logger.error(f"Feature list requested: {feature_list}")
        logger.error(f"DataFrame columns available: {list(df_copy.columns)}")
        return None

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate columns from a DataFrame, keeping only the first occurrence."""
    if df is None:
        return pd.DataFrame()
    if df.columns.duplicated().any():
        logger.warning("Duplicate column names found. Removing duplicates, keeping first occurrence.")
        return df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> List[str]:
    """Returns a list of unique column names from the DataFrame."""
    if df is None:
        return []
    return list(pd.Index(df.columns))

def fill_missing_numeric(df: pd.DataFrame, default_value: float = 0.0) -> pd.DataFrame:
    """Fills missing values (NaN) in all numeric columns with the specified default_value."""
    if df is None:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        # Use .loc to ensure modification on the original DataFrame copy
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(default_value)
    return df

# --- Calculation Helpers ---

def safe_divide(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros in denominator and NaNs."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce')
    
    # Replace 0 in denominator with NaN to avoid division by zero error, then handle it with fillna
    result = num / den.replace(0, np.nan)
    
    # Replace inf/-inf that might result from num/0 if 0 wasn't caught by .replace(0, np.nan)
    # (though .replace(0, np.nan) should prevent this for true zeros)
    # This also handles cases where num is inf.
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(default_val)

# --- String/Date Helpers ---

@lru_cache(maxsize=128) # Reduced cache size slightly, MLB has fewer team name variations typically
def normalize_team_name(team_name: Optional[str]) -> str:
    """Normalize MLB team names using a predefined mapping."""
    if not isinstance(team_name, str) or not team_name.strip():
        return "Unknown" # Return "Unknown" for None, empty, or non-string inputs

    team_lower = team_name.lower().strip()

    # MLB Team Name Mapping
    # Standardized name is typically the mascot or common short name.
    mapping = {
        # AL East
        "baltimore orioles": "orioles", "baltimore": "orioles", "bal": "orioles",
        "boston red sox": "redsox", "boston": "redsox", "bos": "redsox", "red sox": "redsox",
        "new york yankees": "yankees", "ny yankees": "yankees", "nyy": "yankees",
        "tampa bay rays": "rays", "tampa bay": "rays", "tam": "rays", "tb": "rays",
        "toronto blue jays": "bluejays", "toronto": "bluejays", "tor": "bluejays", "blue jays": "bluejays",
        # AL Central
        "chicago white sox": "whitesox", "chi white sox": "whitesox", "cws": "whitesox", "chisox": "whitesox",
        "cleveland guardians": "guardians", "cleveland": "guardians", "cle": "guardians", "cleveland indians": "guardians", # Previous name
        "detroit tigers": "tigers", "detroit": "tigers", "det": "tigers",
        "kansas city royals": "royals", "kansas city": "royals", "kc": "royals", "kcr": "royals",
        "minnesota twins": "twins", "minnesota": "twins", "min": "twins",
        # AL West
        "houston astros": "astros", "houston": "astros", "hou": "astros",
        "los angeles angels": "angels", "la angels": "angels", "laa": "angels", "anaheim angels": "angels", "california angels": "angels",
        "oakland athletics": "athletics", "oakland as": "athletics", "oak": "athletics", "as": "athletics", "oakland a's": "athletics",
        "seattle mariners": "mariners", "seattle": "mariners", "sea": "mariners",
        "texas rangers": "rangers", "texas": "rangers", "tex": "rangers", "tx": "rangers",
        # NL East
        "atlanta braves": "braves", "atlanta": "braves", "atl": "braves",
        "miami marlins": "marlins", "miami": "marlins", "mia": "marlins", "florida marlins": "marlins",
        "new york mets": "mets", "ny mets": "mets", "nym": "mets",
        "philadelphia phillies": "phillies", "philadelphia": "phillies", "phi": "phillies",
        "washington nationals": "nationals", "washington": "nationals", "was": "nationals", "wsn": "nationals", "montreal expos": "nationals",
        # NL Central
        "chicago cubs": "cubs", "chi cubs": "cubs", "chc": "cubs",
        "cincinnati reds": "reds", "cincinnati": "reds", "cin": "reds",
        "milwaukee brewers": "brewers", "milwaukee": "brewers", "mil": "brewers",
        "pittsburgh pirates": "pirates", "pittsburgh": "pirates", "pit": "pirates",
        "st. louis cardinals": "cardinals", "st louis cardinals": "cardinals", "stl": "cardinals", "st. louis": "cardinals",
        # NL West
        "arizona diamondbacks": "dbacks", "arizona": "dbacks", "ari": "dbacks", "diamondbacks": "dbacks", "d-backs": "dbacks",
        "colorado rockies": "rockies", "colorado": "rockies", "col": "rockies",
        "los angeles dodgers": "dodgers", "la dodgers": "dodgers", "lad": "dodgers", "brooklyn dodgers": "dodgers",
        "san diego padres": "padres", "san diego": "padres", "sd": "padres", "sdp": "padres",
        "san francisco giants": "giants", "san francisco": "giants", "sf": "giants", "sfg": "giants", "new york giants": "giants",
        # Common variations / abbreviations often used as keys themselves
        "dbacks": "dbacks", "redsox": "redsox", "yankees": "yankees", "rays": "rays", "bluejays": "bluejays",
        "whitesox": "whitesox", "guardians": "guardians", "tigers": "tigers", "royals": "royals", "twins": "twins",
        "astros": "astros", "angels": "angels", "athletics": "athletics", "mariners": "mariners", "rangers": "rangers",
        "braves": "braves", "marlins": "marlins", "mets": "mets", "phillies": "phillies", "nationals": "nationals",
        "cubs": "cubs", "reds": "reds", "brewers": "brewers", "pirates": "pirates", "cardinals": "cardinals",
        "rockies": "rockies", "dodgers": "dodgers", "padres": "padres", "giants": "giants"
    }
    # Direct match
    if team_lower in mapping:
        return mapping[team_lower]

    # Fallback: check if any key is a substring of team_lower (e.g. "ny yankees" contains "yankees")
    # More specific keys should be listed first in mapping for this to be more effective.
    # This part is less critical if the mapping above is comprehensive.
    for map_key, norm_name in mapping.items():
        if map_key in team_lower: # "yankees" in "new york yankees"
            # This might be too greedy if not careful, e.g. "st louis" in "st louis cardinals"
            # The comprehensive mapping should reduce need for this.
            # Consider adding a length check or more sophisticated fuzzy matching if this becomes an issue.
            # For now, relying on the comprehensive direct map.
            pass # Keeping the original NBA fallback logic commented out as comprehensive map is preferred.

    # Fallback substring matching from original NBA code (less reliable, use comprehensive map above)
    # for name, norm in mapping.items():
    #     if len(team_lower) > 3 and team_lower in name: return norm
    #     if len(name) > 3 and name in team_lower: return norm

    logger.warning(f"MLB Team name '{team_name}' (normalized to '{team_lower}') not found in mapping. Returning as is.")
    return team_lower # Return the cleaned lowercase name if no mapping found

def to_datetime_naive(series: pd.Series) -> pd.Series:
    """Converts a Series to datetime objects, coercing errors and making timezone naive."""
    try:
        # Attempt to convert to datetime
        dt_series = pd.to_datetime(series, errors='coerce')
        # If successful and has timezone info, localize to None (make naive)
        if hasattr(dt_series.dt, 'tz') and dt_series.dt.tz is not None:
            return dt_series.dt.tz_localize(None)
        return dt_series # Already naive or all NaT
    except Exception as e:
        logger.error(f"Error in to_datetime_naive for series. Input type: {type(series)}. Error: {e}", exc_info=True)
        # Return a Series of NaT with the original index if conversion fails broadly
        return pd.Series([pd.NaT] * len(series), index=series.index if hasattr(series, 'index') else None)

def get_mlb_team_form_string(
    team_id_to_check: str, # Ensure it's string for consistent comparison
    current_game_date: pd.Timestamp, # This should be a datetime object (date part)
    all_historical_games_df: pd.DataFrame, # Pass the full history
    game_date_col: str = 'game_date', # The standardized game_date column
    home_id_col: str = 'home_team_id',
    away_id_col: str = 'away_team_id',
    home_score_col: str = 'home_score',
    away_score_col: str = 'away_score',
    status_col_short: str = 'status_short', # Assuming you have a reliable status column
    status_col_long: str = 'status_long',
    num_form_games: int = 5
) -> str:
    if pd.isna(current_game_date) or all_historical_games_df.empty:
        return "N/A"

    # Ensure date columns are in the right format for comparison
    # The all_historical_games_df[game_date_col] should already be datetime objects
    
    team_games = all_historical_games_df[
        ((all_historical_games_df[home_id_col].astype(str) == team_id_to_check) | \
         (all_historical_games_df[away_id_col].astype(str) == team_id_to_check)) & \
        (all_historical_games_df[game_date_col].dt.date < current_game_date.date()) & \
        (all_historical_games_df[status_col_short].isin(['FT', 'F']) | all_historical_games_df[status_col_long] == 'Finished') # Example finished statuses
    ]

    if team_games.empty:
        return "N/A"

    # Sort by date to get the most recent games
    recent_games = team_games.sort_values(by=game_date_col, ascending=False).head(num_form_games)

    if recent_games.empty:
        return "N/A"

    form_results = []
    # Iterate from oldest to newest among the recent games
    for _, game_row in recent_games.sort_values(by=game_date_col, ascending=True).iterrows():
        is_home = str(game_row[home_id_col]) == team_id_to_check
        team_score = pd.to_numeric(game_row[home_score_col if is_home else away_score_col], errors='coerce')
        opponent_score = pd.to_numeric(game_row[away_score_col if is_home else home_score_col], errors='coerce')
        
        if pd.isna(team_score) or pd.isna(opponent_score):
            form_results.append("?") # Or U for Unknown
        elif team_score > opponent_score:
            form_results.append("W")
        elif team_score < opponent_score:
            form_results.append("L")
        else: # Tie (rare in MLB but possible if called before full game or if data includes ties)
            form_results.append("T") 

    return "".join(form_results) if form_results else "N/A"
