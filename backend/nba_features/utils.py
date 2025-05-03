# backend/features/utils.py

from __future__ import annotations
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import traceback
import os
from datetime import datetime, timedelta
import time
import functools
from functools import wraps, lru_cache
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# bring in your Supabase client and FeatureEngine class
from caching.supabase_client import supabase as supabase_client

# --- Logger Configuration  ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Constants --
EPSILON = 1e-6 
# -- Module-Level Helper Functions --


# -------- Defaults --------
DEFAULTS: dict[str, Any] = {
    # Basic/Season Stats
    'win_pct': 0.5,
    'avg_pts_for': 115.0,
    'avg_pts_against': 115.0,
    'home_advantage': 3.0, # Often used conceptually, good to have?
    'score_for': 115.0,     # Alias often used in rolling stats
    'score_against': 115.0, # Alias often used in rolling stats
    'offensive_rating': 115.0,
    'defensive_rating': 115.0,
    'off_rating': 115.0,    # Explicit alias for rolling stats
    'def_rating': 115.0,    # Explicit alias for rolling stats
    'net_rating': 0.0,
    'pace': 100.0,
    'estimated_possessions': 95.0,

    # Advanced Box Score Stats
    'efg_pct': 0.54,
    'ft_rate': 0.20,
    'tov_rate': 13.0,
    'oreb_pct': 0.23,
    'dreb_pct': 0.77,
    'trb_pct': 0.50,

    # Standard Deviations (if needed by models/features directly)
    'score_for_std': 10.0,
    'score_against_std': 10.0,
    'off_rating_std': 10.0,
    'def_rating_std': 10.0,
    'net_rating_std': 10.0,
    'pace_std': 5.0,
    'efg_pct_std': 0.05,
    'tov_rate_std': 3.0,
    'oreb_pct_std': 0.05,
    'dreb_pct_std': 0.05,
    'trb_pct_std': 0.05,
    'ft_rate_std': 0.05,

    # Momentum/Form Stats
    'momentum_ewma': 0.0, # Might be calculated, but good default
    'momentum_ewma_std': 5.0,
    'form_win_pct': 0.5,
    'current_streak': 0,
    'momentum_direction': 0.0,

    # Head-to-Head Matchup Stats
    'matchup_num_games': 0,
    'matchup_avg_point_diff': 0.0,
    'matchup_home_win_pct': 0.5,
    'matchup_avg_total_score': 230.0,
    'matchup_avg_home_score': 115.0,
    'matchup_avg_away_score': 115.0,
    'matchup_streak': 0,

    # Rest/Schedule Stats
    'rest_days': 3.0,
    'games_last_7_days_home': 2,
    'games_last_14_days_home': 4,
    'games_last_7_days_away': 2,
    'games_last_14_days_away': 4,
}



TEAMS_TO_WATCH = {"pistons", "grizzlies", "lakers", "clippers", "nets", "knicks"}

LEAGUE_AVERAGES: dict[str, Any] = {
    'score': DEFAULTS['avg_pts_for'],
    'quarter_scores': {1: 28.5, 2: 28.5, 3: 28.0, 4: 29.0}
}



# -------- Helpers --------
def safe_divide(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
    result = num / den
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(default_val)

def profile_time(func=None, debug_mode: Optional[bool] = None):
    """Decorator to log the execution time of functions."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time
            message = f"{f.__name__} executed in {execution_time:.4f} seconds"
            # Check if the instance has a debug flag
            is_debug_instance = False
            if args and hasattr(args[0], 'debug'):
                is_debug_instance = args[0].debug
            final_debug_mode = debug_mode if debug_mode is not None else is_debug_instance
            if final_debug_mode:
                logger.debug(f"[Profiler] {message}")
            return result
        return wrapper
    return decorator if func is None else decorator(func)

@lru_cache(maxsize=512) # Keep cache for performance
def normalize_team_name(team_name: Optional[Any]) -> str: # Accept Any initially for robustness
    """
    Normalize team names using a predefined mapping.
    Handles potential non-string inputs gracefully.
    """
    # *** ADDED: Explicit type check and handling for NaN/None ***
    if pd.isna(team_name):
        # logger.debug("normalize_team_name received NaN/None input. Returning 'Unknown'.") # Optional debug log
        return "Unknown"
    if not isinstance(team_name, str):
        # logger.debug(f"normalize_team_name received non-string input: {team_name} (type: {type(team_name)}). Converting to string.") # Optional debug log
        try:
            # Attempt to convert other types to string
            team_name = str(team_name)
        except Exception:
            # logger.warning(f"Could not convert input {team_name} to string. Returning 'Unknown'.") # Optional warning
             return "Unknown"


    # Proceed with existing logic now that we are sure team_name is a string
    team_lower = team_name.lower().strip()

    if not team_lower:
        # logger.debug("normalize_team_name received empty string input. Returning 'Unknown'.") # Optional debug log
        return "Unknown"

    # --- Mapping Dictionary (ensure this is complete) ---
    mapping = {
        "atlanta hawks": "hawks", "atlanta": "hawks", "atl": "hawks", "hawks": "hawks", "atlanta h": "hawks",
        "boston celtics": "celtics", "boston": "celtics", "bos": "celtics", "celtics": "celtics",
        "brooklyn nets": "nets", "brooklyn": "nets", "bkn": "nets", "nets": "nets", "new jersey nets": "nets",
        "charlotte hornets": "hornets", "charlotte": "hornets", "cha": "hornets", "hornets": "hornets", "charlotte bobcats": "hornets",
        "chicago bulls": "bulls", "chicago": "bulls", "chi": "bulls", "bulls": "bulls", "chicago b": "bulls",
        "cleveland cavaliers": "cavaliers", "cleveland": "cavaliers", "cle": "cavaliers", "cavaliers": "cavaliers", "cavs": "cavaliers",
        "dallas mavericks": "mavericks", "dallas": "mavericks", "dal": "mavericks", "mavericks": "mavericks", "mavs": "mavericks",
        "denver nuggets": "nuggets", "denver": "nuggets", "den": "nuggets", "nuggets": "nuggets", "denver n": "nuggets",
        "detroit pistons": "pistons", "detroit": "pistons", "det": "pistons", "pistons": "pistons", "detroit p": "pistons",
        "golden state warriors": "warriors", "golden state": "warriors", "gsw": "warriors", "warriors": "warriors", "gs warriors": "warriors",
        "houston rockets": "rockets", "houston": "rockets", "hou": "rockets", "rockets": "rockets",
        "indiana pacers": "pacers", "indiana": "pacers", "ind": "pacers", "pacers": "pacers",
        "los angeles clippers": "clippers", "la clippers": "clippers", "lac": "clippers", "clippers": "clippers",
        "los angeles lakers": "lakers", "la lakers": "lakers", "lal": "lakers", "lakers": "lakers",
        "la": "lakers", # Potential ambiguity, maps 'la' to lakers
        "memphis grizzlies": "grizzlies", "memphis": "grizzlies", "mem": "grizzlies", "grizzlies": "grizzlies", "memphis gri": "grizzlies", "mem grizzlies": "grizzlies", "grizz": "grizzlies",
        "miami heat": "heat", "miami": "heat", "mia": "heat", "heat": "heat",
        "milwaukee bucks": "bucks", "milwaukee": "bucks", "mil": "bucks", "bucks": "bucks", "milwaukee b": "bucks",
        "minnesota timberwolves": "timberwolves", "minnesota": "timberwolves", "min": "timberwolves", "timberwolves": "timberwolves", "twolves": "timberwolves", "minnesota t": "timberwolves",
        "new orleans pelicans": "pelicans", "new orleans": "pelicans", "nop": "pelicans", "pelicans": "pelicans", "nola": "pelicans", "new orleans/oklahoma city hornets": "pelicans", # Added NOLA
        "new york knicks": "knicks", "new york": "knicks", "nyk": "knicks", "knicks": "knicks", "new york knick": "knicks",
        "oklahoma city thunder": "thunder", "oklahoma city": "thunder", "okc": "thunder", "thunder": "thunder", "seattle supersonics": "thunder", "oklahoma city t": "thunder",
        "orlando magic": "magic", "orlando": "magic", "orl": "magic", "magic": "magic", "orlando mag": "magic",
        "philadelphia 76ers": "76ers", "philadelphia": "76ers", "phi": "76ers", "76ers": "76ers", "sixers": "76ers", "phila": "76ers", # Added Phila
        "phoenix suns": "suns", "phoenix": "suns", "phx": "suns", "suns": "suns", "phoenix s": "suns",
        "portland trail blazers": "blazers", "portland": "blazers", "por": "blazers", "blazers": "blazers", "trail blazers": "blazers", "portland trail": "blazers",
        "sacramento kings": "kings", "sacramento": "kings", "sac": "kings", "kings": "kings",
        "san antonio spurs": "spurs", "san antonio": "spurs", "sas": "spurs", "spurs": "spurs", "san antonio s": "spurs",
        "toronto raptors": "raptors", "toronto": "raptors", "tor": "raptors", "raptors": "raptors", "toronto rap": "raptors",
        "utah jazz": "jazz", "utah": "jazz", "uta": "jazz", "jazz": "jazz",
        "washington wizards": "wizards", "washington": "wizards", "was": "wizards", "wizards": "wizards", "wiz": "wizards", "wash wizards": "wizards",
        # Special Cases / All-Star Teams
        "east": "east", "west": "west",
        "team lebron": "allstar", "team durant": "allstar", "team giannis": "allstar", "team stephen": "allstar",
        "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
        "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
    }

    # Check for exact match in mapping
    if team_lower in mapping:
        return mapping[team_lower]

    # Check for substring containment (be careful with short inputs)
    for name, norm in mapping.items():
        # Added check for len > 3 to avoid spurious matches like 'la' in 'atlanta'
        if len(team_lower) > 3 and team_lower in name:
            # logger.debug(f"Normalized '{team_name}' to '{norm}' via substring containment (team_lower in name)") # Optional
            return norm

    # If no match found after explicit and substring checks
    logger.warning(f"Team name '{team_name}' (normalized to '{team_lower}') did not match any mapping. Returning raw normalized input: '{team_lower}'")
    return team_lower # Return the cleaned input if no match

def determine_season(game_date: pd.Timestamp) -> str:
    """Determines the NBA season string (e.g., '2023-2024') for a given game date."""
    if pd.isna(game_date):
        logger.warning("Missing game_date for season determination.")
        return "Unknown_Season"
    year = game_date.year
    month = game_date.month
    start_year = year if month >= 9 else year - 1
    return f"{start_year}-{start_year + 1}"


def generate_rolling_column_name(
    prefix: str,
    base: str,
    stat_type: str,
    window: int
) -> str:
    """Generate standardized rolling feature column names."""
    pre = f'{prefix}_' if prefix else ''
    return f'{pre}rolling_{base}_{stat_type}_{window}'

def convert_and_fill(
    df: pd.DataFrame,
    cols: list[str],
    default: float = 0.0
) -> pd.DataFrame:
    """
    Ensure specified columns exist, coerce them to numeric, 
    and fill any NaNs with the given default.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(default)
    return df
