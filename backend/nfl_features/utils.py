# backend/mlb_features/utils.py

from __future__ import annotations

import logging
import time
import re
from functools import wraps, lru_cache
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# -- Logger Configuration  --
logger = logging.getLogger(__name__)

# -- Constants & Dictionaries --

DEFAULTS: Mapping[str, Any] = {
    # Basic & Season Stats
    'win_pct': 0.5, 'avg_runs_for': 4.6, 'avg_runs_against': 4.6,
    'runs_for_avg': 4.6, 'runs_against_avg': 4.6, 'net_runs_avg': 0.0,
    # Common Rate Stats
    'batting_avg': 0.250, 'on_base_pct': 0.320, 'slugging_pct': 0.410,
    'on_base_plus_slugging': 0.730, 'era': 4.00, 'whip': 1.30,
    'k_per_9': 8.5, 'bb_per_9': 3.0,
    # Standard Deviations
    'runs_scored_std': 3.0, 'hits_for_std': 3.5,
    # Form/Streak Stats
    'form_win_pct': 0.5, 'current_streak': 0.0, 'momentum_direction': 0.0,
    # Head-to-Head Matchup Stats
    'matchup_num_games': 0.0, 'matchup_avg_run_diff': 0.0, 'matchup_home_win_pct': 0.5,
    'matchup_avg_total_runs': 9.2, 'matchup_avg_home_team_runs': 4.6,
    'matchup_avg_away_team_runs': 4.6, 'matchup_home_team_streak': 0.0,
    # Rest/Schedule Stats
    'games_last_7_days': 3.0, 'games_last_14_days': 6.0,
}

# ==============================================================================
# === NEW ID MAP - YOU MUST POPULATE THIS DICTIONARY ===
# ==============================================================================
# This map translates the numerical team IDs from your database to the
# canonical team name used throughout the feature engineering pipeline.
MLB_ID_MAP: Mapping[str, str] = {
    "2": "dbacks",              # Arizona Diamondbacks
    "3": "braves",              # Atlanta Braves
    "4": "orioles",             # Baltimore Orioles
    "5": "redsox",              # Boston Red Sox
    "6": "cubs",                # Chicago Cubs
    "7": "whitesox",            # Chicago White Sox
    "8": "reds",                # Cincinnati Reds
    "9": "guardians",           # Cleveland Guardians
    "10": "rockies",             # Colorado Rockies
    "12": "tigers",              # Detroit Tigers
    "15": "astros",              # Houston Astros
    "16": "royals",              # Kansas City Royals
    "17": "angels",              # Los Angeles Angels
    "18": "dodgers",             # Los Angeles Dodgers
    "19": "marlins",             # Miami Marlins
    "20": "brewers",             # Milwaukee Brewers
    "22": "twins",               # Minnesota Twins
    "24": "mets",                # New York Mets
    "25": "yankees",             # New York Yankees
    "26": "athletics",           # Oakland Athletics
    "27": "phillies",            # Philadelphia Phillies
    "28": "pirates",             # Pittsburgh Pirates
    "30": "padres",              # San Diego Padres
    "31": "giants",              # San Francisco Giants
    "32": "mariners",            # Seattle Mariners
    "33": "cardinals",           # St.Louis Cardinals
    "34": "rays",                # Tampa Bay Rays
    "35": "rangers",             # Texas Rangers
    "36": "bluejays",            # Toronto Blue Jays
    "37": "nationals",           # Washington Nationals
    # Handling alternate IDs from your data
    "963": "athletics",          # Athletics (Alternate ID)
    "625": "guardians",          # Cleveland Guardians (Alternate ID)
}
# ==============================================================================


# Definitive mapping for all 30 MLB teams to a single, canonical key.
MLB_TEAM_MAP: Mapping[str, str] = {
    # Full Name, City, Abbreviation, Mascot -> Canonical Name
    "arizona diamondbacks": "dbacks", "arizona": "dbacks", "ari": "dbacks", "diamondbacks": "dbacks",
    "atlanta braves": "braves", "atlanta": "braves", "atl": "braves",
    "baltimore orioles": "orioles", "baltimore": "orioles", "bal": "orioles",
    "boston red sox": "redsox", "boston": "redsox", "bos": "redsox", "red sox": "redsox",
    "chicago cubs": "cubs", "chi cubs": "cubs", "chc": "cubs",
    "chicago white sox": "whitesox", "chi white sox": "whitesox", "cws": "whitesox", "chisox": "whitesox",
    "cincinnati reds": "reds", "cincinnati": "reds", "cin": "reds",
    "cleveland guardians": "guardians", "cleveland": "guardians", "cle": "guardians", "cleveland indians": "guardians",
    "colorado rockies": "rockies", "colorado": "rockies", "col": "rockies",
    "detroit tigers": "tigers", "detroit": "tigers", "det": "tigers",
    "houston astros": "astros", "houston": "astros", "hou": "astros",
    "kansas city royals": "royals", "kansas city": "royals", "kc": "royals",
    "los angeles angels": "angels", "la angels": "angels", "laa": "angels",
    "los angeles dodgers": "dodgers", "la dodgers": "dodgers", "lad": "dodgers",
    "miami marlins": "marlins", "miami": "marlins", "mia": "marlins", "florida marlins": "marlins",
    "milwaukee brewers": "brewers", "milwaukee": "brewers", "mil": "brewers",
    "minnesota twins": "twins", "minnesota": "twins", "min": "twins",
    "new york mets": "mets", "ny mets": "mets", "nym": "mets",
    "new york yankees": "yankees", "ny yankees": "yankees", "nyy": "yankees",
    "oakland athletics": "athletics", "oakland as": "athletics", "oak": "athletics", "oakland a's": "athletics",
    "philadelphia phillies": "phillies", "philadelphia": "phillies", "phi": "phillies",
    "pittsburgh pirates": "pirates", "pittsburgh": "pirates", "pit": "pirates",
    "san diego padres": "padres", "san diego": "padres", "sd": "padres",
    "san francisco giants": "giants", "san francisco": "giants", "sf": "giants",
    "seattle mariners": "mariners", "seattle": "mariners", "sea": "mariners",
    "st. louis cardinals": "cardinals", "st louis cardinals": "cardinals", "stl": "cardinals",
    "tampa bay rays": "rays", "tampa bay": "rays", "tb": "rays", "devil rays": "rays",
    "texas rangers": "rangers", "texas": "rangers", "tex": "rangers",
    "toronto blue jays": "bluejays", "toronto": "bluejays", "tor": "bluejays", "blue jays": "bluejays",
    "washington nationals": "nationals", "washington": "nationals", "was": "nationals", "montreal expos": "nationals",
    # Add canonical names themselves to the map for full consistency
    "dbacks": "dbacks", "braves": "braves", "orioles": "orioles", "redsox": "redsox", "cubs": "cubs",
    "whitesox": "whitesox", "reds": "reds", "guardians": "guardians", "rockies": "rockies", "tigers": "tigers",
    "astros": "astros", "royals": "royals", "angels": "angels", "dodgers": "dodgers", "marlins": "marlins",
    "brewers": "brewers", "twins": "twins", "mets": "mets", "yankees": "yankees", "athletics": "athletics",
    "phillies": "phillies", "pirates": "pirates", "padres": "padres", "giants": "giants", "mariners": "mariners",
    "cardinals": "cardinals", "rays": "rays", "rangers": "rangers", "bluejays": "bluejays", "nationals": "nationals"
}

# -- Helper Functions --

# ==============================================================================
# === REWRITTEN NORMALIZATION FUNCTION ===
# ==============================================================================
@lru_cache(maxsize=512)
def normalize_team_name(team_identifier: Optional[Any]) -> str:
    """
    Normalizes a team identifier (ID or name) to its canonical form.
    It prioritizes matching by ID from MLB_ID_MAP, then falls back to matching
    by name from MLB_TEAM_MAP.
    """
    if pd.isna(team_identifier):
        return "unknown_team"
    
    try:
        # Convert to a clean string, but don't lowercase yet to preserve IDs
        id_str = str(team_identifier).strip()
    except Exception:
        return "unknown_team"

    if not id_str:
        return "unknown_team"

    # --- Main Logic ---
    # 1. First, check if the input is a known ID from our new ID map.
    if id_str in MLB_ID_MAP:
        return MLB_ID_MAP[id_str]
    
    # 2. If it's not a known ID, assume it's a name. Lowercase and check the name map.
    name_lower = id_str.lower()
    if name_lower in MLB_TEAM_MAP:
        return MLB_TEAM_MAP[name_lower]
    
    # 3. If it's not in either map, log a more informative warning and return unknown.
    logger.warning(
        f"Team identifier '{team_identifier}' could not be resolved. "
        f"Not found in MLB_ID_MAP or MLB_TEAM_MAP. Returning 'unknown_team'."
    )
    return "unknown_team"
# ==============================================================================


def determine_season(game_date: pd.Timestamp) -> int:
    """Determines the MLB season as an integer year for a given game date."""
    if pd.isna(game_date):
        logger.warning("Missing game_date for season determination. Returning current year as fallback.")
        return pd.Timestamp.now().year
    # For MLB, the season year is the same as the calendar year of the game.
    return game_date.year

def safe_divide(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
    result = num / den
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(default_val)

def profile_time(func=None, *, enabled: bool = True):
    """Decorator to log the execution time of a function."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            is_debug = enabled and kwargs.get('debug', False)
            start_time = time.time()
            result = f(*args, **kwargs)
            if is_debug:
                execution_time = time.time() - start_time
                logger.debug(f"[Profiler] {f.__name__} executed in {execution_time:.4f}s")
            return result
        return wrapper
    return decorator(func) if func else decorator