# backend/nba_score_prediction/utils.py

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
            # If column is entirely missing, add it with the default value
            df_copy[col] = default
            logger.warning(f"Column '{col}' not found in convert_and_fill. Added with default value {default}.")
        else:
            # Convert existing column, coerce errors, fill NaNs resulting from coercion or already present
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(default)
    return df_copy

def generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """
    Generate standardized rolling feature column name (prefix can be 'home', 'away', or '').

    Args:
        prefix: 'home', 'away', or '' for team perspective.
        base: The base statistic name (e.g., 'net_rating', 'pace').
        stat_type: 'mean' or 'std'.
        window: The rolling window size.

    Returns:
        The standardized column name string.
    """
    # Ensure prefix is followed by underscore only if it exists
    prefix_part = f"{prefix}_" if prefix else ""
    # Using base name as passed directly
    return f"{prefix_part}rolling_{base}_{stat_type}_{window}"

def slice_dataframe_by_features(df: pd.DataFrame, feature_list: List[str], fill_value: float = 0.0) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame containing only the columns specified in feature_list,
    ordered correctly. Adds missing columns filled with fill_value.

    Args:
        df: Input DataFrame.
        feature_list: List of features to select, in desired order.
        fill_value: Value to fill in for missing features.

    Returns:
        A new DataFrame with columns ordered as in feature_list, or None if input df is None.
    """
    if df is None:
        logger.error("Input DataFrame is None in slice_dataframe_by_features.")
        return None
    if not feature_list:
        logger.warning("Empty feature_list provided to slice_dataframe_by_features. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    df_copy = df.copy()
    # Identify missing features
    current_cols = set(df_copy.columns)
    missing_features = set(feature_list) - current_cols
    if missing_features:
        logger.warning(f"slice_dataframe: Missing features {missing_features}. Filling with {fill_value}.")
        for col in missing_features:
            df_copy[col] = fill_value

    # Identify extra features not in the list (optional logging)
    extra_features = current_cols - set(feature_list)
    if extra_features:
        logger.debug(f"slice_dataframe: Input df has extra columns not in feature_list: {extra_features}")

    try:
        # Use reindex for selecting AND ordering. It handles missing columns by default if fill_value is not None
        # But we explicitly add missing ones above for clarity and logging.
        return df_copy.reindex(columns=feature_list, fill_value=fill_value)
    except Exception as e:
        logger.error(f"Error reindexing DataFrame in slice_dataframe_by_features: {e}", exc_info=True)
        logger.error(f"Feature list requested: {feature_list}")
        logger.error(f"DataFrame columns available: {list(df_copy.columns)}")
        return None 

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate columns from a DataFrame, keeping only the first occurrence.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicate columns removed, or original if no duplicates.
    """
    if df is None: return pd.DataFrame()
    if df.columns.duplicated().any():
        logger.warning("Duplicate column names found. Removing duplicates, keeping first occurrence.")
        return df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of unique column names from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of unique column names, or empty list if df is None.
    """
    if df is None: return []
    # pd.Index automatically handles uniqueness, converting to list gives unique names
    return list(pd.Index(df.columns)) 

def fill_missing_numeric(df: pd.DataFrame, default_value: float = 0.0) -> pd.DataFrame:
    """
    Fills missing values (NaN) in all numeric columns with the specified default_value.

    Args:
        df: Input DataFrame.
        default_value: The value to use for filling missing numeric data (default: 0.0).

    Returns:
        DataFrame with missing numeric values filled.
    """
    if df is None: return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].fillna(default_value)
    return df

# --- Calculation Helpers (Moved from Feature Engine) ---

def safe_divide(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan) 
    result = num / den
    result.replace([np.inf, -np.inf], np.nan, inplace=True) 
    return result.fillna(default_val) #

# --- String/Date Helpers (Moved from Feature Engine) ---

@lru_cache(maxsize=512) 
def normalize_team_name(team_name: Optional[str]) -> str:
    """Normalize team names using a predefined mapping."""
    if not isinstance(team_name, str): return "Unknown"
    team_lower = team_name.lower().strip()
    mapping = {
        "atlanta hawks": "hawks", "atlanta": "hawks", "atl": "hawks", "hawks": "hawks", "atlanta h": "hawks",
        "boston celtics": "celtics", "boston": "celtics", "bos": "celtics", "celtics": "celtics",
        "brooklyn nets": "nets", "brooklyn": "nets", "bkn": "nets", "nets": "nets", "new jersey nets": "nets",
        "charlotte hornets": "hornets", "charlotte": "hornets", "cha": "hornets", "hornets": "hornets", "charlotte bobcats": "hornets",
        "chicago bulls": "bulls", "chicago": "bulls", "chi": "bulls", "bulls": "bulls", "chicago b": "bulls",
        "cleveland cavaliers": "cavaliers", "cleveland": "cavaliers", "cle": "cavaliers", "cavaliers": "cavaliers", "cavs": "cavaliers",
        "dallas mavericks": "mavericks", "dallas": "mavericks", "dal": "mavericks", "mavericks": "mavericks", "mavs": "mavericks",
        "denver nuggets": "nuggets", "denver": "nuggets", "den": "nuggets", "nuggets": "nuggets", "denver n": "nuggets",
        "detroit pistons": "pistons", "detroit": "pistons", "det": "pistons", "pistons": "pistons",
        "golden state warriors": "warriors", "golden state": "warriors", "gsw": "warriors", "warriors": "warriors",
        "houston rockets": "rockets", "houston": "rockets", "hou": "rockets", "rockets": "rockets",
        "indiana pacers": "pacers", "indiana": "pacers", "ind": "pacers", "pacers": "pacers",
        "los angeles clippers": "clippers", "la clippers": "clippers", "lac": "clippers", "clippers": "clippers",
        "los angeles lakers": "lakers", "la lakers": "lakers", "lal": "lakers", "lakers": "lakers",
        "memphis grizzlies": "grizzlies", "memphis": "grizzlies", "mem": "grizzlies", "grizzlies": "grizzlies", "memphis gri": "grizzlies",
        "miami heat": "heat", "miami": "heat", "mia": "heat", "heat": "heat",
        "milwaukee bucks": "bucks", "milwaukee": "bucks", "mil": "bucks", "bucks": "bucks", "milwaukee b": "bucks",
        "minnesota timberwolves": "timberwolves", "minnesota": "timberwolves", "min": "timberwolves", "timberwolves": "timberwolves", "twolves": "timberwolves", "minnesota t": "timberwolves",
        "new orleans pelicans": "pelicans", "new orleans": "pelicans", "nop": "pelicans", "pelicans": "pelicans", "new orleans/oklahoma city hornets": "pelicans",
        "new york knicks": "knicks", "new york": "knicks", "nyk": "knicks", "knicks": "knicks", "new york knick": "knicks",
        "oklahoma city thunder": "thunder", "oklahoma city": "thunder", "okc": "thunder", "thunder": "thunder", "seattle supersonics": "thunder",
        "orlando magic": "magic", "orlando": "magic", "orl": "magic", "magic": "magic", "orlando mag": "magic",
        "philadelphia 76ers": "76ers", "philadelphia": "76ers", "phi": "76ers", "76ers": "76ers", "sixers": "76ers",
        "phoenix suns": "suns", "phoenix": "suns", "phx": "suns", "suns": "suns", "phoenix s": "suns",
        "portland trail blazers": "blazers", "portland": "blazers", "por": "blazers", "blazers": "blazers", "trail blazers": "blazers", "portland trail": "blazers",
        "sacramento kings": "kings", "sacramento": "kings", "sac": "kings", "kings": "kings",
        "san antonio spurs": "spurs", "san antonio": "spurs", "sas": "spurs", "spurs": "spurs", "san antonio s": "spurs",
        "toronto raptors": "raptors", "toronto": "raptors", "tor": "raptors", "raptors": "raptors", "toronto rap": "raptors",
        "utah jazz": "jazz", "utah": "jazz", "uta": "jazz", "jazz": "jazz",
        "washington wizards": "wizards", "washington": "wizards", "was": "wizards", "wizards": "wizards", "wiz": "wizards",
        "la": "lakers",
        # Placeholders for non-NBA teams
        "east": "east", "west": "west", "team lebron": "other_team", "team durant":"other_team",
        "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
        "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
    }
    if team_lower in mapping: return mapping[team_lower]
    # Fallback substring matching (less reliable)
    for name, norm in mapping.items():
        if len(team_lower) > 3 and team_lower in name: return norm
        if len(name) > 3 and name in team_lower: return norm
    logger.warning(f"Team name '{team_name}' normalized to '{team_lower}' - no mapping found!")
    return team_lower 

def to_datetime_naive(series: pd.Series) -> pd.Series:
    """Converts a Series to datetime objects, coercing errors and making timezone naive."""
    try:
        return pd.to_datetime(series, errors='coerce').dt.tz_localize(None)
    except AttributeError: 
        return pd.to_datetime(series, errors='coerce')
    except Exception as e:
        logger.error(f"Error in to_datetime_naive: {e}")
        return pd.Series([pd.NaT] * len(series), index=series.index) 