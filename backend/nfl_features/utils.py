# backend/nfl_features/utils.py

from __future__ import annotations

import logging
import time
from functools import wraps, lru_cache
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# -- Logger Configuration --
logger = logging.getLogger(__name__)

# -- NFL Constants & Dictionaries --

DEFAULTS: Mapping[str, Any] = {
    # Basic & Season Stats
    'win_pct': 0.5, 'points_for_avg': 22.0, 'points_against_avg': 22.0,
    'point_differential_avg': 0.0, 'srs_lite': 0.0, 'elo': 1500.0,
    # Offensive Rate Stats
    'yards_per_play_avg': 5.4, 'passing_yards_per_pass_avg': 6.8, 'rushings_yards_per_rush_avg': 4.2,
    'turnovers_per_game_avg': 1.5, 'turnover_differential_avg': 0.0, 'third_down_pct': 0.40, 'fourth_down_pct': 0.50, 'red_zone_pct': 0.55,
    # Defensive Rate Stats
    'sacks_per_game_avg': 2.5, 'takeaways_per_game_avg': 1.5,
    # Standard Deviations
    'points_scored_std': 10.0, 'yards_total_std': 90.0,
    # Form/Streak Stats
    'form_win_pct': 0.5, 'current_streak': 0.0, 'momentum_direction': 0.0,
    # Head-to-Head Matchup Stats
    'matchup_num_games': 0.0, 'matchup_avg_point_diff': 0.0, 'matchup_home_win_pct': 0.5,
    'matchup_avg_total_points': 44.0, 'matchup_avg_home_team_points': 22.0,
    'matchup_avg_away_team_points': 22.0,
    # Rest/Schedule Stats
    'days_since_last_game': 7.0,
}

# ==============================================================================
# === IMPORTANT: POPULATE THIS DICTIONARY WITH YOUR DATABASE IDs ===
# ==============================================================================
# This map translates the numerical team IDs from your Supabase tables to the
# canonical team name used throughout the feature engineering pipeline.
NFL_ID_MAP: Mapping[str, str] = {
    # Example: "1": "cardinals", # Arizona Cardinals
    # You must fill this section with the actual IDs for all 32 teams.
}
# ==============================================================================

# Definitive mapping for all 32 NFL teams to a single, canonical key.
NFL_TEAM_MAP: Mapping[str, str] = {
    # Full Name, City, Abbreviation, Mascot -> Canonical Name
    "arizona cardinals": "cardinals", "arizona": "cardinals", "ari": "cardinals",
    "atlanta falcons": "falcons", "atlanta": "falcons", "atl": "falcons",
    "baltimore ravens": "ravens", "baltimore": "ravens", "bal": "ravens",
    "buffalo bills": "bills", "buffalo": "bills", "buf": "bills",
    "carolina panthers": "panthers", "carolina": "panthers", "car": "panthers",
    "chicago bears": "bears", "chicago": "bears", "chi": "bears",
    "cincinnati bengals": "bengals", "cincinnati": "bengals", "cin": "bengals",
    "cleveland browns": "browns", "cleveland": "browns", "cle": "browns",
    "dallas cowboys": "cowboys", "dallas": "cowboys", "dal": "cowboys",
    "denver broncos": "broncos", "denver": "broncos", "den": "broncos",
    "detroit lions": "lions", "detroit": "lions", "det": "lions",
    "green bay packers": "packers", "green bay": "packers", "gb": "packers",
    "houston texans": "texans", "houston": "texans", "hou": "texans",
    "indianapolis colts": "colts", "indianapolis": "colts", "ind": "colts",
    "jacksonville jaguars": "jaguars", "jacksonville": "jaguars", "jax": "jaguars", "jac": "jaguars",
    "kansas city chiefs": "chiefs", "kansas city": "chiefs", "kc": "chiefs",
    "las vegas raiders": "raiders", "las vegas": "raiders", "lv": "raiders", "oakland raiders": "raiders",
    "los angeles chargers": "chargers", "la chargers": "chargers", "lac": "chargers", "san diego chargers": "chargers",
    "los angeles rams": "rams", "la rams": "rams", "lar": "rams", "st. louis rams": "rams",
    "miami dolphins": "dolphins", "miami": "dolphins", "mia": "dolphins",
    "minnesota vikings": "vikings", "minnesota": "vikings", "min": "vikings",
    "new england patriots": "patriots", "new england": "patriots", "ne": "patriots",
    "new orleans saints": "saints", "new orleans": "saints", "no": "saints",
    "new york giants": "giants", "ny giants": "giants", "nyg": "giants",
    "new york jets": "jets", "ny jets": "jets", "nyj": "jets",
    "philadelphia eagles": "eagles", "philadelphia": "eagles", "phi": "eagles",
    "pittsburgh steelers": "steelers", "pittsburgh": "steelers", "pit": "steelers",
    "san francisco 49ers": "49ers", "san francisco": "49ers", "sf": "49ers",
    "seattle seahawks": "seahawks", "seattle": "seahawks", "sea": "seahawks",
    "tampa bay buccaneers": "buccaneers", "tampa bay": "buccaneers", "tb": "buccaneers",
    "tennessee titans": "titans", "tennessee": "titans", "ten": "titans",
    "washington commanders": "commanders", "washington": "commanders", "was": "commanders", "washington football team": "commanders", "washington redskins": "commanders",
    # Add canonical names themselves to the map for full consistency
    "cardinals": "cardinals", "falcons": "falcons", "ravens": "ravens", "bills": "bills", "panthers": "panthers",
    "bears": "bears", "bengals": "bengals", "browns": "browns", "cowboys": "cowboys", "broncos": "broncos",
    "lions": "lions", "packers": "packers", "texans": "texans", "colts": "colts", "jaguars": "jaguars",
    "chiefs": "chiefs", "raiders": "raiders", "chargers": "chargers", "rams": "rams", "dolphins": "dolphins",
    "vikings": "vikings", "patriots": "patriots", "saints": "saints", "giants": "giants", "jets": "jets",
    "eagles": "eagles", "steelers": "steelers", "49ers": "49ers", "seahawks": "seahawks",
    "buccaneers": "buccaneers", "titans": "titans", "commanders": "commanders",
}

# -- Helper Functions --

@lru_cache(maxsize=512)
def normalize_team_name(team_identifier: Optional[Any]) -> str:
    """
    Normalizes a team identifier (ID or name) to its canonical NFL form.
    It prioritizes matching by ID from NFL_ID_MAP, then falls back to matching
    by name from NFL_TEAM_MAP.
    """
    if pd.isna(team_identifier):
        return "unknown_team"
    
    try:
        id_str = str(team_identifier).strip()
    except Exception:
        return "unknown_team"

    if not id_str:
        return "unknown_team"

    # 1. First, check if the input is a known ID from our ID map.
    if id_str in NFL_ID_MAP:
        return NFL_ID_MAP[id_str]
    
    # 2. If not a known ID, assume it's a name. Lowercase and check the name map.
    name_lower = id_str.lower()
    if name_lower in NFL_TEAM_MAP:
        return NFL_TEAM_MAP[name_lower]
    
    # 3. If it's not in either map, log a warning and return unknown.
    logger.warning(
        f"Team identifier '{team_identifier}' could not be resolved. "
        f"Not found in NFL_ID_MAP or NFL_TEAM_MAP. Returning 'unknown_team'."
    )
    return "unknown_team"


def determine_season(game_date: pd.Timestamp) -> int:
    """
    Determines the NFL season as an integer year for a given game date.
    The NFL season crosses calendar years (e.g., games in Jan/Feb 2024
    belong to the 2023 season). We use March as the cutoff.
    """
    if pd.isna(game_date):
        logger.warning("Missing game_date for season determination. Returning current year as fallback.")
        now = pd.Timestamp.now()
        return now.year if now.month >= 3 else now.year - 1
    
    # If a game is played before March, it belongs to the previous calendar year's season.
    return game_date.year if game_date.month >= 3 else game_date.year - 1

def prefix_columns(
    df: pd.DataFrame,
    prefix: str,
    exclude: list[str] | None = None
) -> pd.DataFrame:
    """
    Rename all columns in `df` by prepending `prefix + '_'`,
    except for any columns listed in `exclude`.
    """
    exclude_set = set(exclude) if exclude is not None else set()
    mapping = {
        col: (col if col in exclude_set else f"{prefix}_{col}")
        for col in df.columns
    }
    return df.rename(columns=mapping)


def safe_divide(
    numerators: pd.Series,
    denominators: pd.Series,
    fill: float = 0.0,
    default_val: float | None = None
) -> pd.Series:
    """
    Safely divide two Series element‑wise.
    - ±inf → NaN
    - NaN → fill value
    - 'default_val' is an alias for 'fill' (tests call default_val).
    """
    # if default_val provided, it wins
    fill_value = default_val if default_val is not None else fill

    result = numerators / denominators
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill_value)

# keep the alias for rate computations
compute_rate = safe_divide

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