# backend/mlb_features/utils.py

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
    if team_lower.endswith(".0"):
       team_lower = team_lower[:-2]
        
    if not team_lower:
        # logger.debug("normalize_team_name received empty string input. Returning 'Unknown'.") # Optional debug log
        return "Unknown"
    
    # --- Mapping Dictionary (ensure this is complete) ---
    # In backend/mlb_features/utils.py, inside normalize_team_name function

    # MLB Team Name Mapping
    # Standardized name is typically the mascot or common short name.
    # !!! IMPORTANT: Replace placeholder IDs like "101", "102" with your actual stringified numerical team IDs !!!
    mapping = {
        # --- Arizona Diamondbacks ---
        "arizona diamondbacks": "dbacks", "arizona": "dbacks", "ari": "dbacks", "diamondbacks": "dbacks", "d-backs": "dbacks",
        "2": "dbacks", # Actual ID for Arizona Diamondbacks

        # --- Atlanta Braves ---
        "atlanta braves": "braves", "atlanta": "braves", "atl": "braves",
        "3": "braves", # Actual ID for Atlanta Braves

        # --- Baltimore Orioles ---
        "baltimore orioles": "orioles", "baltimore": "orioles", "bal": "orioles",
        "4": "orioles", # Actual ID for Baltimore Orioles

        # --- Boston Red Sox ---
        "boston red sox": "redsox", "boston": "redsox", "bos": "redsox", "red sox": "redsox",
        "5": "redsox", # Actual ID for Boston Red Sox

        # --- Chicago Cubs ---
        "chicago cubs": "cubs", "chi cubs": "cubs", "chc": "cubs",
        "6": "cubs", # Actual ID for Chicago Cubs

        # --- Chicago White Sox ---
        "chicago white sox": "whitesox", "chi white sox": "whitesox", "cws": "whitesox", "chisox": "whitesox",
        "7": "whitesox", # Actual ID for Chicago White Sox

        # --- Cincinnati Reds ---
        "cincinnati reds": "reds", "cincinnati": "reds", "cin": "reds",
        "8": "reds", # Actual ID for Cincinnati Reds

        # --- Cleveland Guardians ---
        "cleveland guardians": "guardians", "cleveland": "guardians", "cle": "guardians", "cleveland indians": "guardians", "indians": "guardians",
        "9": "guardians", # Actual ID for Cleveland Guardians
        "625": "guardians", # Actual ID for Cleveland Guardians

        # --- Colorado Rockies ---
        "colorado rockies": "rockies", "colorado": "rockies", "col": "rockies", "rox": "rockies",
        "10": "rockies", # Actual ID for Colorado Rockies

        # --- Detroit Tigers ---
        "detroit tigers": "tigers", "detroit": "tigers", "det": "tigers",
        "12": "tigers", # Actual ID for Detroit Tigers (Note: ID 11 was skipped in your list)

        # --- Houston Astros ---
        "houston astros": "astros", "houston": "astros", "hou": "astros",
        "15": "astros", # Actual ID for Houston Astros (Note: IDs 13, 14 skipped)

        # --- Kansas City Royals ---
        "kansas city royals": "royals", "kansas city": "royals", "kc": "royals", "kcr": "royals",
        "16": "royals", # Actual ID for Kansas City Royals

        # --- Los Angeles Angels ---
        "los angeles angels": "angels", "la angels": "angels", "laa": "angels", "anaheim angels": "angels", "california angels": "angels", "los angeles angels of anaheim": "angels",
        "17": "angels", # Actual ID for Los Angeles Angels

        # --- Los Angeles Dodgers ---
        "los angeles dodgers": "dodgers", "la dodgers": "dodgers", "lad": "dodgers", "brooklyn dodgers": "dodgers",
        "18": "dodgers", # Actual ID for Los Angeles Dodgers

        # --- Miami Marlins ---
        "miami marlins": "marlins", "miami": "marlins", "mia": "marlins", "florida marlins": "marlins",
        "19": "marlins", # Actual ID for Miami Marlins

        # --- Milwaukee Brewers ---
        "milwaukee brewers": "brewers", "milwaukee": "brewers", "mil": "brewers", "brew crew": "brewers",
        "20": "brewers", # Actual ID for Milwaukee Brewers (Note: ID 21 skipped)

        # --- Minnesota Twins ---
        "minnesota twins": "twins", "minnesota": "twins", "min": "twins",
        "22": "twins", # Actual ID for Minnesota Twins (Note: ID 23 skipped)

        # --- New York Mets ---
        "new york mets": "mets", "ny mets": "mets", "nym": "mets",
        "24": "mets", # Actual ID for New York Mets

        # --- New York Yankees ---
        "new york yankees": "yankees", "ny yankees": "yankees", "nyy": "yankees",
        "25": "yankees", # Actual ID for New York Yankees

        # --- Oakland Athletics ---
        "oakland athletics": "athletics", "oakland as": "athletics", "oak": "athletics", "as": "athletics", "oakland a's": "athletics", "a's": "athletics", "athletics": "athletics", # Added "athletics" itself
        "26": "athletics", # Actual ID for Oakland Athletics
        "963": "athletics",# Actual ID for Athletics (from your list, likely Oakland)

        # --- Philadelphia Phillies ---
        "philadelphia phillies": "phillies", "philadelphia": "phillies", "phi": "phillies", "phils": "phillies",
        "27": "phillies", # Actual ID for Philadelphia Phillies

        # --- Pittsburgh Pirates ---
        "pittsburgh pirates": "pirates", "pittsburgh": "pirates", "pit": "pirates", "bucn": "pirates", "bucs": "pirates",
        "28": "pirates", # Actual ID for Pittsburgh Pirates (Note: ID 29 skipped)

        # --- San Diego Padres ---
        "san diego padres": "padres", "san diego": "padres", "sd": "padres", "sdp": "padres", "friars": "padres",
        "30": "padres", # Actual ID for San Diego Padres

        # --- San Francisco Giants ---
        "san francisco giants": "giants", "san francisco": "giants", "sf": "giants", "sfg": "giants", "new york giants": "giants",
        "31": "giants", # Actual ID for San Francisco Giants

        # --- Seattle Mariners ---
        "seattle mariners": "mariners", "seattle": "mariners", "sea": "mariners",
        "32": "mariners", # Actual ID for Seattle Mariners

        # --- St. Louis Cardinals ---
        "st. louis cardinals": "cardinals", "st louis cardinals": "cardinals", "stl": "cardinals", "st. louis": "cardinals", "cards": "cardinals",
        "33": "cardinals", # Actual ID for St. Louis Cardinals

        # --- Tampa Bay Rays ---
        "tampa bay rays": "rays", "tampa bay": "rays", "tam": "rays", "tb": "rays", "devil rays": "rays",
        "34": "rays", # Actual ID for Tampa Bay Rays

        # --- Texas Rangers ---
        "texas rangers": "rangers", "texas": "rangers", "tex": "rangers", "tx": "rangers",
        "35": "rangers", # Actual ID for Texas Rangers

        # --- Toronto Blue Jays ---
        "toronto blue jays": "bluejays", "toronto": "bluejays", "tor": "bluejays", "blue jays": "bluejays",
        "36": "bluejays", # Actual ID for Toronto Blue Jays

        # --- Washington Nationals ---
        "washington nationals": "nationals", "washington": "nationals", "was": "nationals", "wsn": "nationals", "nats": "nationals", "montreal expos": "nationals", "expos": "nationals",
        "37": "nationals", # Actual ID for Washington Nationals

        # Add the standardized short names themselves as keys mapping to themselves for robustness
        "orioles": "orioles", "redsox": "redsox", "yankees": "yankees", "rays": "rays", "bluejays": "bluejays",
        "whitesox": "whitesox", "guardians": "guardians", "tigers": "tigers", "royals": "royals", "twins": "twins",
        "astros": "astros", "angels": "angels", "athletics": "athletics", "mariners": "mariners", "rangers": "rangers",
        "braves": "braves", "marlins": "marlins", "mets": "mets", "phillies": "phillies", "nationals": "nationals",
        "cubs": "cubs", "reds": "reds", "brewers": "brewers", "pirates": "pirates", "cardinals": "cardinals",
        "dbacks": "dbacks", "rockies": "rockies", "dodgers": "dodgers", "padres": "padres", "giants": "giants"
    }


    # Check for exact match in mapping
    if team_lower in mapping:
        return mapping[team_lower]
    
        # Fallback for anything not in the map
    logger.warning(
        f"Team identifier '{team_name}' (type: {type(team_name)}, normalized to '{team_lower}') "
        f"not found in mapping. Returning 'unknown_team' as standardized name."
    )
    return "unknown_team" # Return a generic unknown identifier


def get_supabase_client():
    """
    Lazily import and return the shared Supabase client.
    Doing it here (inside a function) prevents config.py from validating
    too early when this module is merely imported.
    """
    from caching.supabase_client import supabase as supabase_client
    return supabase_client


def determine_season(game_date: pd.Timestamp) -> int: # Changed return type annotation to int
    """Determines the MLB season as an integer year (e.g., 2023) for a given game date."""
    if pd.isna(game_date):
        logger.warning("Missing game_date for season determination. Returning current year as a fallback.")
        # Fallback strategy: using the current year. You might prefer to return 0 or raise an error
        # depending on how strictly you want to handle missing/invalid dates.
        return datetime.now().year
        
    # For MLB, the season is typically just the calendar year of the game.
    # Regular season games fall within a single calendar year.
    # Post-season games (e.g., October) also belong to that same calendar year's season.
    return game_date.year

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