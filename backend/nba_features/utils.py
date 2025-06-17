# backend/nba_features/utils.py

from __future__ import annotations
import logging
import time
import re
from functools import wraps, lru_cache
from typing import List, Optional, Dict, Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Default fallback values for various statistical features ---
DEFAULTS: Mapping[str, Any] = {
    'win_pct': 0.5, 'avg_pts_for': 115.0, 'avg_pts_against': 115.0,
    'home_advantage': 3.0, 'score_for': 115.0, 'score_against': 115.0,
    'offensive_rating': 115.0, 'defensive_rating': 115.0, 'off_rating': 115.0,
    'def_rating': 115.0, 'net_rating': 0.0, 'pace': 100.0,
    'estimated_possessions': 95.0, 'efg_pct': 0.54, 'ft_rate': 0.20,
    'tov_rate': 13.0, 'oreb_pct': 0.23, 'dreb_pct': 0.77, 'trb_pct': 0.50,
    'score_for_std': 10.0, 'score_against_std': 10.0, 'off_rating_std': 10.0,
    'def_rating_std': 10.0, 'net_rating_std': 10.0, 'pace_std': 5.0,
    'efg_pct_std': 0.05, 'tov_rate_std': 3.0, 'oreb_pct_std': 0.05,
    'dreb_pct_std': 0.05, 'trb_pct_std': 0.05, 'ft_rate_std': 0.05,
    'momentum_ewma': 0.0, 'momentum_ewma_std': 5.0, 'form_win_pct': 0.5,
    'current_streak': 0, 'momentum_direction': 0.0, 'matchup_num_games': 0,
    'matchup_avg_point_diff': 0.0, 'matchup_home_win_pct': 0.5,
    'matchup_avg_total_score': 230.0, 'matchup_avg_home_score': 115.0,
    'matchup_avg_away_score': 115.0, 'matchup_streak': 0, 'rest_days': 3.0,
    'games_last_7_days_home': 2, 'games_last_14_days_home': 4,
    'games_last_7_days_away': 2, 'games_last_14_days_away': 4,
}


def profile_time(func=None, *, enabled: bool | None = None):
    """Decorator to log execution time of a function."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            dbg = enabled if enabled is not None else kwargs.get('debug', False)
            start = time.time()
            result = f(*args, **kwargs)
            elapsed = time.time() - start
            if dbg:
                logger.debug(f"[PROFILE] {f.__name__} executed in {elapsed:.4f}s")
            return result
        return wrapper
    return decorator(func) if func else decorator


# --- DataFrame Operations ---

def convert_and_fill(df: pd.DataFrame, columns: List[str], default: float = 0.0) -> pd.DataFrame:
    """Convert specified columns to numeric and fill NaNs with a default value."""
    df_copy = df.copy()
    for col in columns:
        if col not in df_copy.columns:
            df_copy[col] = default
            logger.warning(f"Column '{col}' not found in convert_and_fill. Added with default value {default}.")
        else:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(default)
    return df_copy

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate columns from a DataFrame, keeping only the first occurrence."""
    if df.columns.duplicated().any():
        logger.warning("Duplicate column names found. Removing duplicates, keeping first occurrence.")
        return df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


# --- String/Date/Name Helpers ---

@lru_cache(maxsize=512)
def normalize_team_name(team_name: Optional[str]) -> str:
    """Normalize team names using a predefined mapping."""
    if not isinstance(team_name, str) or not team_name.strip():
        return "Unknown"
        
    team_lower = team_name.lower().strip()
    
    mapping = {
        "atlanta hawks": "hawks", "atlanta": "hawks", "atl": "hawks",
        "boston celtics": "celtics", "boston": "celtics", "bos": "celtics",
        "brooklyn nets": "nets", "brooklyn": "nets", "bkn": "nets", "new jersey nets": "nets",
        "charlotte hornets": "hornets", "charlotte": "hornets", "cha": "hornets", "charlotte bobcats": "hornets",
        "chicago bulls": "bulls", "chicago": "bulls", "chi": "bulls",
        "cleveland cavaliers": "cavaliers", "cleveland": "cavaliers", "cle": "cavaliers", "cavs": "cavaliers",
        "dallas mavericks": "mavericks", "dallas": "mavericks", "dal": "mavericks", "mavs": "mavericks",
        "denver nuggets": "nuggets", "denver": "nuggets", "den": "nuggets",
        "detroit pistons": "pistons", "detroit": "pistons", "det": "pistons",
        "golden state warriors": "warriors", "golden state": "warriors", "gsw": "warriors",
        "houston rockets": "rockets", "houston": "rockets", "hou": "rockets",
        "indiana pacers": "pacers", "indiana": "pacers", "ind": "pacers",
        "los angeles clippers": "clippers", "la clippers": "clippers", "lac": "clippers",
        "los angeles lakers": "lakers", "la lakers": "lakers", "lal": "lakers",
        "memphis grizzlies": "grizzlies", "memphis": "grizzlies", "mem": "grizzlies",
        "miami heat": "heat", "miami": "heat", "mia": "heat",
        "milwaukee bucks": "bucks", "milwaukee": "bucks", "mil": "bucks",
        "minnesota timberwolves": "timberwolves", "minnesota": "timberwolves", "min": "timberwolves", "twolves": "timberwolves",
        "new orleans pelicans": "pelicans", "new orleans": "pelicans", "nop": "pelicans", "new orleans/oklahoma city hornets": "pelicans",
        "new york knicks": "knicks", "new york": "knicks", "nyk": "knicks",
        "oklahoma city thunder": "thunder", "oklahoma city": "thunder", "okc": "thunder", "seattle supersonics": "thunder",
        "orlando magic": "magic", "orlando": "magic", "orl": "magic",
        "philadelphia 76ers": "76ers", "philadelphia": "76ers", "phi": "76ers", "sixers": "76ers",
        "phoenix suns": "suns", "phoenix": "suns", "phx": "suns",
        "portland trail blazers": "blazers", "portland": "blazers", "por": "blazers", "trail blazers": "blazers",
        "sacramento kings": "kings", "sacramento": "kings", "sac": "kings",
        "san antonio spurs": "spurs", "san antonio": "spurs", "sas": "spurs",
        "toronto raptors": "raptors", "toronto": "raptors", "tor": "raptors",
        "utah jazz": "jazz", "utah": "jazz", "uta": "jazz",
        "washington wizards": "wizards", "washington": "wizards", "was": "wizards", "wiz": "wizards",
    }
    # Add self-references for the canonical names
    for team in list(mapping.values()): mapping[team] = team
    
    if team_lower in mapping:
        return mapping[team_lower]
            
    logger.warning(f"Team name '{team_name}' has no explicit mapping; returning lowercased and stripped version.")
    return team_lower

def determine_season(game_date: pd.Timestamp) -> str:
    """Given a date, return NBA season string like '2023-24'."""
    if pd.isna(game_date):
        return 'Unknown_Season'
    year, month = game_date.year, game_date.month
    start = year if month >= 8 else year - 1
    end_short = (start + 1) % 100
    return f"{start}-{end_short:02d}"

def generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """Generate standardized rolling feature column name."""
    prefix_part = f"{prefix}_" if prefix else ""
    return f"{prefix_part}rolling_{base}_{stat_type}_{window}"

def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """
    Safely divide two pandas Series, avoiding divide-by-zero and NaNs.
    Replaces infinite results with NaN and fills NaNs with `default`.
    """
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
    result = num / den
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(default)