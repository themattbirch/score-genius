# backend/features/utils.py

from __future__ import annotations
import logging
import time
import re
from functools import wraps, lru_cache
from typing import Any, Mapping, Sequence, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Default fallback values for various statistical features ---
DEFAULTS: Mapping[str, Any] = {
    # Basic/Season Stats
    'win_pct': 0.5,
    'avg_pts_for': 115.0,
    'avg_pts_against': 115.0,
    'home_advantage': 3.0,
    'score_for': 115.0,
    'score_against': 115.0,
    'offensive_rating': 115.0,
    'defensive_rating': 115.0,
    'off_rating': 115.0,
    'def_rating': 115.0,
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

    # Standard Deviations
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

    # Form/Momentum Stats
    'momentum_ewma': 0.0,
    'momentum_ewma_std': 5.0,
    'form_win_pct': 0.5,
    'current_streak': 0.0,
    'momentum_direction': 0.0,
    'form_longest_w_streak': 0.0,
    'form_longest_l_streak': 0.0,

    # Head-to-Head
    'matchup_num_games': 0,
    'matchup_avg_point_diff': 0.0,
    'matchup_home_win_pct': 0.5,
    'matchup_avg_total_score': 230.0,
    'matchup_avg_home_score': 115.0,
    'matchup_avg_away_score': 115.0,
    'matchup_std_point_diff': 0.0,

    # Rest/Schedule
    'rest_days': 3.0,
    'games_last_7_days': 0.0,
    'games_last_14_days': 0.0,
    'schedule_advantage': 0.0,
    'schedule_advantage_14d': 0.0,

    # Momentum per quarter
    'momentum_score_ewma_q3': 0.0,
    'momentum_score_ewma_q4': 0.0,
    'std_dev_q_margins': 0.0,
}

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

# 1) Canonical list of NBA teams (use whatever canon keys you want downstream)
CANON = {
    "hawks":        ["atlanta hawks", "atlanta", "atl", "hawks"],
    "celtics":      ["boston celtics", "boston", "bos", "celtics"],
    "nets":         ["brooklyn nets", "brooklyn", "bkn", "nets", "new jersey nets"],
    "hornets":      ["charlotte hornets", "charlotte", "cha", "hornets", "charlotte bobcats"],
    "bulls":        ["chicago bulls", "chicago", "chi", "bulls"],
    "cavaliers":    ["cleveland cavaliers", "cleveland", "cle", "cavaliers", "cavs"],
    "mavericks":    ["dallas mavericks", "dallas", "dal", "mavericks", "mavs"],
    "nuggets":      ["denver nuggets", "denver", "den", "nuggets"],
    "pistons":      ["detroit pistons", "detroit", "det", "pistons"],
    "warriors":     ["golden state warriors", "golden state", "gsw", "warriors"],
    "rockets":      ["houston rockets", "houston", "hou", "rockets"],
    "pacers":       ["indiana pacers", "indiana", "ind", "pacers"],
    "clippers":     ["los angeles clippers", "la clippers", "lac", "clippers"],
    "lakers":       ["los angeles lakers", "la lakers", "lal", "lakers"],
    "grizzlies":    ["memphis grizzlies", "memphis", "mem", "grizzlies", "grizz"],
    "heat":         ["miami heat", "miami", "mia", "heat"],
    "bucks":        ["milwaukee bucks", "milwaukee", "mil", "bucks"],
    "timberwolves": ["minnesota timberwolves", "minnesota", "min", "timberwolves", "twolves"],
    "pelicans":     ["new orleans pelicans", "new orleans", "nop", "pelicans", "nola"],
    "knicks":       ["new york knicks", "new york", "nyk", "knicks"],
    "thunder":      ["oklahoma city thunder", "oklahoma city", "okc", "thunder", "seattle supersonics"],
    "magic":        ["orlando magic", "orlando", "orl", "magic"],
    "76ers":        ["philadelphia 76ers", "philadelphia", "phi", "sixers", "phila"],
    "suns":         ["phoenix suns", "phoenix", "phx", "suns"],
    "blazers":      ["portland trail blazers", "portland", "por", "blazers", "trail blazers"],
    "kings":        ["sacramento kings", "sacramento", "sac", "kings"],
    "spurs":        ["san antonio spurs", "san antonio", "sas", "spurs"],
    "raptors":      ["toronto raptors", "toronto", "tor", "raptors"],
    "jazz":         ["utah jazz", "utah", "uta", "jazz"],
    "wizards":      ["washington wizards", "washington", "was", "wizards", "wiz"],
    # add special/all-star groupings if needed
}

# ---------------------------------------------------------------------------
# 2) Build *two* fast-lookup tables once:
SPACED2CANON = {alias: canon for canon, aliases in CANON.items() for alias in aliases}
COMPACT2CANON = {
    re.sub(r"[^a-z0-9]", "", alias): canon
    for alias, canon in SPACED2CANON.items()
}



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


def profile_time(func=None, *, enabled: bool | None = None):
    """
    Decorator to log execution time of a function when `enabled` is True,
    or when the decorated function is called with `debug=True`.
    """
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

@lru_cache(maxsize=512)
def normalize_team_name(name) -> str:
    """Return canonical team key, or 'Unknown'."""
    if pd.isna(name):
        return "Unknown"
    try:
        raw = str(name).lower().strip()
    except Exception:
        return "Unknown"
    if not raw:
        return "Unknown"

    compact = re.sub(r"[^a-z0-9]", "", raw)

    # 1. exact spaced alias
    if raw in SPACED2CANON:
        return SPACED2CANON[raw]
    # 2. exact compact alias
    if compact in COMPACT2CANON:
        return COMPACT2CANON[compact]
    # 3. substring fallback on compact (avoid 3-char junk)
    if len(compact) > 3:
        for key, canon in COMPACT2CANON.items():
            if compact in key:
                return canon

    logger.warning(
        "normalize_team_name: '%s' → no mapping, falling back to '%s'",
        name, compact
    )
    return compact

def determine_season(game_date: pd.Timestamp) -> str:
    """
    Given a date, return NBA season string like '2023-24'.
    """
    if pd.isna(game_date):
        return 'Unknown_Season'
    year, month = game_date.year, game_date.month
    start = year if month >= 8 else year - 1
    end_short = (start + 1) % 100
    return f"{start}-{end_short:02d}"


def generate_rolling_column_name(
    prefix: str, base_stat: str, stat_type: str, window: int
) -> str:
    """
    Construct rolling feature names: e.g., 'home_rolling_score_for_mean_5'.
    """
    pref = f"{prefix}_" if prefix else ""
    return f"{pref}rolling_{base_stat}_{stat_type}_{window}"


def convert_and_fill(
    df: pd.DataFrame, cols: Sequence[str], default: Any = 0.0
) -> pd.DataFrame:
    """
    Ensure `cols` exist in df, coerce to numeric, and fill NaNs with default.
    Returns the modified DataFrame.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(default)
    return df
