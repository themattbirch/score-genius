# backend/nfl_features/utils.py

from __future__ import annotations

import logging
import time
from functools import wraps, lru_cache
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS: Mapping[str, Any] = {
    'win_pct': 0.5, 'points_for_avg': 22.0, 'points_against_avg': 22.0,
    'point_differential_avg': 0.0, 'srs_lite': 0.0, 'elo': 1500.0,
    'yards_per_play_avg': 5.4, 'passing_yards_per_pass_avg': 6.8, 'rushings_yards_per_rush_avg': 4.2,
    'turnovers_per_game_avg': 1.5, 'turnover_differential_avg': 0.0, 'third_down_pct': 0.40, 'fourth_down_pct': 0.50, 'red_zone_pct': 0.55,
    'sacks_per_game_avg': 2.5, 'takeaways_per_game_avg': 1.5,
    'points_scored_std': 10.0, 'yards_total_std': 90.0,
    'form_win_pct': 0.5, 'current_streak': 0.0, 'momentum_direction': 0.0,
    'matchup_num_games': 0.0, 'matchup_avg_point_diff': 0.0, 'matchup_home_win_pct': 0.5,
    'matchup_avg_total_points': 44.0, 'matchup_avg_home_team_points': 22.0,
    'matchup_avg_away_team_points': 22.0,
    'days_since_last_game': 7.0,
}

NFL_ID_MAP: Mapping[str, str] = {
    # Fill with your real IDs → canonical names when ready
}

NFL_TEAM_MAP: Mapping[str, str] = {
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
    "cardinals": "cardinals", "falcons": "falcons", "ravens": "ravens", "bills": "bills", "panthers": "panthers",
    "bears": "bears", "bengals": "bengals", "browns": "browns", "cowboys": "cowboys", "broncos": "broncos",
    "lions": "lions", "packers": "packers", "texans": "texans", "colts": "colts", "jaguars": "jaguars",
    "chiefs": "chiefs", "raiders": "raiders", "chargers": "chargers", "rams": "rams", "dolphins": "dolphins",
    "vikings": "vikings", "patriots": "patriots", "saints": "saints", "giants": "giants", "jets": "jets",
    "eagles": "eagles", "steelers": "steelers", "49ers": "49ers", "seahawks": "seahawks",
    "buccaneers": "buccaneers", "titans": "titans", "commanders": "commanders",
}

_UNKNOWN_SEEN: set[str] = set()

@lru_cache(maxsize=2048)
def normalize_team_name(team_identifier: Optional[Any]) -> str:
    if pd.isna(team_identifier):
        return "unknown_team"

    try:
        token = str(team_identifier).strip()
    except Exception:
        return "unknown_team"

    if not token:
        return "unknown_team"

    # 1) Exact ID mapping
    if token in NFL_ID_MAP:
        return NFL_ID_MAP[token]

    # 2) Pre-tokenized numeric ID like "id_24"
    tlow = token.lower()
    if tlow.startswith("id_") and tlow[3:].isdigit():
        return tlow  # keep distinct, no warning

    # 3) Bare numeric ID "24" → stable token if map not provided
    if token.isdigit():
        return f"id_{token}"

    # 4) Name mapping
    if tlow in NFL_TEAM_MAP:
        return NFL_TEAM_MAP[tlow]

        # 5) Warn once per unknown string
    if tlow not in _UNKNOWN_SEEN:
        _UNKNOWN_SEEN.add(tlow)
        logger.warning(
            "Team identifier '%s' could not be resolved. Not in NFL_ID_MAP or NFL_TEAM_MAP. Returning 'unknown_team'.",
            team_identifier,
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