# backend/nba_score_prediction/feature_engineering.py - Unified & Integrated Feature Engineering

"""
NBAFeatureEngine - Unified module for NBA score prediction feature engineering.
Handles the creation of various features for NBA game prediction models.
Integrated version combining best practices from previous iterations.
"""

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import functools
from functools import wraps, lru_cache
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# -- Optional Plotting (conditional import) --
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# -- Logger Configuration --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Constants --
EPSILON = 1e-6  # Small value to avoid division by zero if needed elsewhere

# -- Define Output Directory (Consistent) --
try:
    SCRIPT_DIR_FE = Path(__file__).resolve().parent
    # Adjust path relative to this file's location
    REPORTS_DIR_FE = SCRIPT_DIR_FE.parent.parent / 'reports'
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environment)
    REPORTS_DIR_FE = Path('./reports')
FEATURE_DEBUG_DIR = REPORTS_DIR_FE / "feature_debug"


# -- Module-Level Helper Functions --

def safe_divide(numerator: pd.Series, denominator: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
    result = num / den
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(default_val)


def convert_and_fill(df: pd.DataFrame, columns: List[str], default: float = 0.0) -> pd.DataFrame:
    """Convert specified columns to numeric and fill NaNs with a default value."""
    for col in columns:
        if col not in df.columns:
            df[col] = default
            logger.warning(f"Column '{col}' not found. Added with default value {default}.")
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    return df


def generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """Generate standardized rolling feature column name (prefix can be 'home', 'away', or '')."""
    prefix_part = f"{prefix}_" if prefix else ""
    return f"{prefix_part}rolling_{base}_{stat_type}_{window}"

def _is_debug_enabled(logger_instance):
    return logger_instance.isEnabledFor(logging.DEBUG)

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


# -- NBAFeatureEngine Class --

class NBAFeatureEngine:
    """Core class for NBA feature engineering."""
    
    def __init__(self, supabase_client: Optional[Any] = None, debug: bool = False):
        """Initialize the feature engine."""
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
            # FEATURE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("DEBUG mode enabled for detailed logging.")
        else:
            logger.setLevel(logging.INFO)
        self.supabase_client = supabase_client
        logger.debug("NBAFeatureEngine Initialized.")

                # <<< INSERT THIS LINE HERE >>>
        self._teams_to_watch = {"pistons", "grizzlies", "lakers," "clippers", "nets", "knicks"}
        # <<< END INSERT >>>


        # --- Default Values ---
        self.defaults = {
            'win_pct': 0.5, 'avg_pts_for': 115.0, 'avg_pts_against': 115.0,
            'home_advantage': 3.0, 'score_for': 115.0, 'score_against': 115.0,
            'score_for_std': 10.0, 'score_against_std': 10.0,
            'offensive_rating': 115.0, 'defensive_rating': 115.0, 'net_rating': 0.0,
            'pace': 100.0, 'estimated_possessions': 95.0,
            'efg_pct': 0.54, 'ft_rate': 0.20, 'tov_rate': 13.0,
            'oreb_pct': 0.23, 'dreb_pct': 0.77, 'trb_pct': 0.50,
            'off_rating_std': 10.0, 'def_rating_std': 10.0, 'net_rating_std': 10.0,
            'pace_std': 5.0, 'efg_pct_std': 0.05, 'tov_rate_std': 3.0,
            'oreb_pct_std': 0.05, 'dreb_pct_std': 0.05, 'trb_pct_std': 0.05,
            'ft_rate_std': 0.05,
            'momentum_ewma': 0.0, 'momentum_ewma_std': 5.0,
            'form_win_pct': 0.5, 'current_streak': 0, 'momentum_direction': 0.0,
            'matchup_num_games': 0, 'matchup_avg_point_diff': 0.0, 'matchup_home_win_pct': 0.5,
            'matchup_avg_total_score': 230.0, 'matchup_avg_home_score': 115.0,
            'matchup_avg_away_score': 115.0, 'matchup_streak': 0,
            'rest_days': 3.0,
            'games_last_7_days_home': 2, 'games_last_14_days_home': 4,
            'games_last_7_days_away': 2, 'games_last_14_days_away': 4,
        }
        # Add alias defaults for rolling features
        self.defaults['off_rating'] = self.defaults['offensive_rating']
        self.defaults['def_rating'] = self.defaults['defensive_rating']

        # League averages (can be updated externally if needed)
        self.league_averages = {
            'score': self.defaults['avg_pts_for'],
            'quarter_scores': {1: 28.5, 2: 28.5, 3: 28.0, 4: 29.0}
        }
    
    @lru_cache(maxsize=512) # Keep the cache decorator
    def normalize_team_name(self, team_name: Optional[str]) -> str:
        """Normalize team names using a predefined mapping."""
        if not isinstance(team_name, str):
            # Handle non-string inputs gracefully
            logger.debug(f"normalize_team_name received non-string input: {team_name}. Returning 'Unknown'.")
            return "Unknown"

        team_lower = team_name.lower().strip()

        # Handle empty strings after stripping
        if not team_lower:
             logger.debug("normalize_team_name received empty string input. Returning 'Unknown'.")
             return "Unknown"

        # --- Updated Mapping Dictionary ---
        mapping = {
            "atlanta hawks": "hawks", "atlanta": "hawks", "atl": "hawks", "hawks": "hawks", "atlanta h": "hawks",
            "boston celtics": "celtics", "boston": "celtics", "bos": "celtics", "celtics": "celtics",
            "brooklyn nets": "nets", "brooklyn": "nets", "bkn": "nets", "nets": "nets", "new jersey nets": "nets",
            "charlotte hornets": "hornets", "charlotte": "hornets", "cha": "hornets", "hornets": "hornets", "charlotte bobcats": "hornets",
            "chicago bulls": "bulls", "chicago": "bulls", "chi": "bulls", "bulls": "bulls", "chicago b": "bulls",
            "cleveland cavaliers": "cavaliers", "cleveland": "cavaliers", "cle": "cavaliers", "cavaliers": "cavaliers", "cavs": "cavaliers",
            "dallas mavericks": "mavericks", "dallas": "mavericks", "dal": "mavericks", "mavericks": "mavericks", "mavs": "mavericks",
            "denver nuggets": "nuggets", "denver": "nuggets", "den": "nuggets", "nuggets": "nuggets", "denver n": "nuggets",

            # --- DETROIT PISTONS (Added 'detroit p') ---
            "detroit pistons": "pistons", "detroit": "pistons", "det": "pistons", "pistons": "pistons", "detroit p": "pistons",

            "golden state warriors": "warriors", "golden state": "warriors", "gsw": "warriors", "warriors": "warriors", "gs warriors": "warriors",
            "houston rockets": "rockets", "houston": "rockets", "hou": "rockets", "rockets": "rockets",
            "indiana pacers": "pacers", "indiana": "pacers", "ind": "pacers", "pacers": "pacers",
            "los angeles clippers": "clippers", "la clippers": "clippers", "lac": "clippers", "clippers": "clippers",
            "los angeles lakers": "lakers", "la lakers": "lakers", "lal": "lakers", "lakers": "lakers",
            "la": "lakers", # Common abbreviation explicitly mapped

            # --- MEMPHIS GRIZZLIES (Added 'mem grizzlies', 'grizz') ---
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

            # --- Special Cases / All-Star Teams ---
            "east": "east", "west": "west", # Keep if needed for specific data sources
            # You might want to map All-Star teams to a generic 'allstar' or 'other_team'
            "team lebron": "allstar", "team durant": "allstar", "team giannis": "allstar", "team stephen": "allstar",
            "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
            "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
        }

        # 1. Check for exact match in mapping (most common case)
        if team_lower in mapping:
            return mapping[team_lower]

        # 2. Fallback: Check if input contains a known full name/city (e.g., "Utah Jazz" in "Utah Jazz Team") - less common
        # Be careful with short keys here
        # for name, norm in mapping.items():
        #    if len(name) > 5 and name in team_lower: # Check if known name is substring of input
        #        logger.debug(f"Normalized '{team_name}' to '{norm}' via substring containment (name in team_lower)")
        #        return norm

        # 3. Fallback: Check if input is contained within a known name (e.g., "Portland" in "Portland Trail Blazers")
        # More likely scenario for partial inputs
        for name, norm in mapping.items():
             if len(team_lower) > 3 and team_lower in name: # Check if input is substring of known name
                 logger.debug(f"Normalized '{team_name}' to '{norm}' via substring containment (team_lower in name)")
                 return norm


        # 4. If no match found after explicit and substring checks
        logger.warning(f"Team name '{team_name}' (normalized to '{team_lower}') did not match any explicit or substring mapping. Returning raw normalized input.")

    def _determine_season(self, game_date: pd.Timestamp) -> str:
        """Determines the NBA season string (e.g., '2023-2024') for a given game date."""
        if pd.isna(game_date):
            logger.warning("Missing game_date for season determination.")
            return "Unknown_Season"
        year = game_date.year
        month = game_date.month
        start_year = year if month >= 9 else year - 1
        return f"{start_year}-{start_year + 1}"
    
    @profile_time
    def add_intra_game_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds features related to quarter-by-quarter score margin changes."""
        logger.debug("Adding intra-game momentum features...")
        if df is None or df.empty:
            return df
        result_df = df.copy()
        qtr_cols_base = [f'q{i}' for i in range(1, 5)]
        qtr_cols_home = [f'home_{base}' for base in qtr_cols_base]
        qtr_cols_away = [f'away_{base}' for base in qtr_cols_base]
        all_qtr_cols = qtr_cols_home + qtr_cols_away
        result_df = convert_and_fill(result_df, all_qtr_cols, default=0.0)
        for i in range(1, 5):
            result_df[f'q{i}_margin'] = result_df[f'home_q{i}'] - result_df[f'away_q{i}']
        result_df['end_q1_diff'] = result_df['q1_margin']
        result_df['end_q2_diff'] = result_df['end_q1_diff'] + result_df['q2_margin']
        result_df['end_q3_diff'] = result_df['end_q2_diff'] + result_df['q3_margin']
        result_df['end_q4_reg_diff'] = result_df['end_q3_diff'] + result_df['q4_margin']
        result_df['q2_margin_change'] = result_df['q2_margin'] - result_df['q1_margin']
        result_df['q3_margin_change'] = result_df['q3_margin'] - result_df['q2_margin']
        result_df['q4_margin_change'] = result_df['q4_margin'] - result_df['q3_margin']
        valid_q_margin_cols = [f'q{i}_margin' for i in range(1, 5) if f'q{i}_margin' in result_df.columns]
        if len(valid_q_margin_cols) == 4:
            quarter_margins = result_df[valid_q_margin_cols]
            result_df['momentum_score_ewma_q4'] = quarter_margins.ewm(span=3, axis=1, adjust=False).mean().iloc[:, -1].fillna(0)
            result_df['momentum_score_ewma_q3'] = result_df[valid_q_margin_cols[:3]].ewm(span=2, axis=1, adjust=False).mean().iloc[:, -1].fillna(0)
        else:
            logger.warning("Could not calculate EWMA momentum scores due to missing quarter margin columns.")
            result_df['momentum_score_ewma_q4'] = self.defaults['momentum_ewma']
            result_df['momentum_score_ewma_q3'] = self.defaults['momentum_ewma']
        logger.debug("Finished adding intra-game momentum features.")
        return result_df
    
    @profile_time
    def integrate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates advanced metrics: eFG%, FT%, Reb%, Possessions, Pace, Ratings, TOV%.
        Uses distinct raw possessions for Ratings and TOV% calculations.
        Calculates game pace based on average possessions.
        """
        logger.debug("Integrating advanced metrics (Revised Pace/Poss/Ratings)...")
        result_df = df.copy()

        # --- 1. Ensure Required Base Stats Exist & Are Numeric ---
        stat_cols = [
            'home_score', 'away_score', 'home_fg_made', 'home_fg_attempted',
            'away_fg_made', 'away_fg_attempted','home_3pm', 'home_3pa',
            'away_3pm', 'away_3pa', 'home_ft_made', 'home_ft_attempted',
            'away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb',
            'home_total_reb','away_off_reb', 'away_def_reb', 'away_total_reb',
            'home_turnovers', 'away_turnovers','home_ot', 'away_ot'
        ]
        # Add missing columns as 0 and ensure numeric type, filling NaNs with 0
        for col in stat_cols:
            if col not in result_df.columns:
                logger.warning(f"Advanced Metrics: Column '{col}' not found. Adding with default value 0.")
                result_df[col] = 0
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # Define nested safe_divide helper function
        def safe_divide(numerator, denominator, default_val):
            num = pd.to_numeric(numerator, errors='coerce')
            den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan) # Avoid division by zero
            result = num / den
            result.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinity
            return result.fillna(default_val)

        # --- 2. Calculate Basic Shooting / Rebounding / FT Rates ---
        logger.debug("Calculating basic rates (eFG%, FT%, Reb%)...")
        result_df['home_efg_pct'] = safe_divide(result_df['home_fg_made'] + 0.5 * result_df['home_3pm'], result_df['home_fg_attempted'], self.defaults['efg_pct'])
        result_df['away_efg_pct'] = safe_divide(result_df['away_fg_made'] + 0.5 * result_df['away_3pm'], result_df['away_fg_attempted'], self.defaults['efg_pct'])
        result_df['home_ft_rate'] = safe_divide(result_df['home_ft_attempted'], result_df['home_fg_attempted'], self.defaults['ft_rate'])
        result_df['away_ft_rate'] = safe_divide(result_df['away_ft_attempted'], result_df['away_fg_attempted'], self.defaults['ft_rate'])
        result_df['home_oreb_pct'] = safe_divide(result_df['home_off_reb'], result_df['home_off_reb'] + result_df['away_def_reb'], self.defaults['oreb_pct'])
        result_df['away_dreb_pct'] = safe_divide(result_df['away_def_reb'], result_df['away_def_reb'] + result_df['home_off_reb'], self.defaults['dreb_pct']) # Away team's DREB% = Away DREB / (Away DREB + Opponent OREB)
        result_df['away_oreb_pct'] = safe_divide(result_df['away_off_reb'], result_df['away_off_reb'] + result_df['home_def_reb'], self.defaults['oreb_pct'])
        result_df['home_dreb_pct'] = safe_divide(result_df['home_def_reb'], result_df['home_def_reb'] + result_df['away_off_reb'], self.defaults['dreb_pct']) # Home team's DREB% = Home DREB / (Home DREB + Opponent OREB)
        result_df['home_trb_pct'] = safe_divide(result_df['home_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct'])
        result_df['away_trb_pct'] = safe_divide(result_df['away_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct']) # Make sure denominator is consistent


        # --- 3. Calculate Pace and Possessions ---
        logger.debug("Calculating raw possessions, average possessions, and Pace...")
        # Calculate Raw Possessions per team (using standard 0.44 factor)
        home_poss_raw = (
            result_df['home_fg_attempted']
            + 0.44 * result_df['home_ft_attempted']
            - result_df['home_off_reb']
            + result_df['home_turnovers']
        )
        away_poss_raw = (
            result_df['away_fg_attempted']
            + 0.44 * result_df['away_ft_attempted']
            - result_df['away_off_reb']
            + result_df['away_turnovers']
        )
        # Store raw possessions (handle 0s before using as denominator)
        result_df['home_poss_raw'] = home_poss_raw
        result_df['away_poss_raw'] = away_poss_raw
        home_poss_safe_denom = home_poss_raw.replace(0, np.nan) # Use NaN for safe division
        away_poss_safe_denom = away_poss_raw.replace(0, np.nan) # Use NaN for safe division

        # Calculate Average Possessions per Team (for Pace calculation)
        avg_poss_per_team = 0.5 * (home_poss_raw + away_poss_raw)

        # Clip and Store Average Possessions Estimate
        possessions_est = avg_poss_per_team.clip(lower=50, upper=120) # Apply clipping
        result_df['possessions_est'] = possessions_est.fillna(self.defaults['estimated_possessions'])

        # Calculate Game Minutes Played
        num_ot = np.maximum(result_df.get('home_ot', 0), result_df.get('away_ot', 0)).clip(lower=0)
        game_minutes_calc = 48.0 + num_ot * 5.0
        result_df['game_minutes_played'] = np.maximum(48.0, game_minutes_calc) # Ensure minimum 48 mins

        # Calculate Game Pace (Pace per 48 minutes, using average possessions estimate)
        result_df['game_pace'] = safe_divide(
            result_df['possessions_est'] * 48.0,
            result_df['game_minutes_played'],
            self.defaults['pace']
        )
        # Assign the same game pace to both teams
        result_df['home_pace'] = result_df['game_pace']
        result_df['away_pace'] = result_df['game_pace']


        # --- 4. Calculate Efficiency (Ratings) and TOV% using RAW possessions ---
        logger.debug("Calculating ratings and TOV% using raw possessions...")
        result_df['home_offensive_rating'] = safe_divide(result_df['home_score'] * 100, home_poss_safe_denom, self.defaults['offensive_rating'])
        result_df['away_offensive_rating'] = safe_divide(result_df['away_score'] * 100, away_poss_safe_denom, self.defaults['offensive_rating'])
        result_df['home_defensive_rating'] = result_df['away_offensive_rating'] # Opponent's offense
        result_df['away_defensive_rating'] = result_df['home_offensive_rating'] # Opponent's offense

        # Apply clipping to ratings
        rating_cols = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']
        for col in rating_cols:
            default_key = col.replace('home_', '').replace('away_', '')
            result_df[col] = result_df[col].clip(lower=70, upper=150).fillna(self.defaults.get(default_key, 115.0))

        result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
        result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']

        # Calculate TOV% using raw possessions
        result_df['home_tov_rate'] = safe_divide(result_df['home_turnovers'] * 100, home_poss_safe_denom, self.defaults['tov_rate'])
        result_df['away_tov_rate'] = safe_divide(result_df['away_turnovers'] * 100, away_poss_safe_denom, self.defaults['tov_rate'])

        # Apply clipping to TOV%
        result_df['home_tov_rate'] = result_df['home_tov_rate'].clip(lower=5, upper=25).fillna(self.defaults['tov_rate'])
        result_df['away_tov_rate'] = result_df['away_tov_rate'].clip(lower=5, upper=25).fillna(self.defaults['tov_rate'])


        # --- 5. Calculate Differentials & Convenience Columns ---
        logger.debug("Calculating differential features...")
        result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
        result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
        result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct'] # Corrected from previous potential copy-paste
        result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']
        # 'pace_differential' is intentionally omitted as home_pace == away_pace
        result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate'] # Note the order (Away - Home for TOV%)

        if 'total_score' not in result_df.columns:
            result_df['total_score'] = result_df['home_score'] + result_df['away_score']
        if 'point_diff' not in result_df.columns:
            result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

        # --- 6. Clean Up ---
        # Remove intermediate columns if they exist and are no longer needed
        # The old 'home_possessions'/'away_possessions' stored the average, we used raw now.
        result_df = result_df.drop(columns=['home_possessions', 'away_possessions'], errors='ignore')    

        logger.debug("Finished integrating advanced features with revised Pace/Poss/Ratings.")
        return result_df

    @profile_time
    def add_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Adds rolling mean and standard deviation features, preventing data leakage."""
        logger.debug(f"Adding rolling features for windows: {window_sizes}...")
        if df is None or df.empty:
            logger.warning("Rolling: Input df empty.")
            return df

        local_df = df.copy()
        # Define generic columns we want rolling stats for
        cols_to_roll_generic = [
            'score_for', 'score_against', 'point_diff', 'momentum_ewma',
            'off_rating', 'def_rating', 'net_rating', 'pace', 'efg_pct',
            'tov_rate', 'trb_pct', 'oreb_pct', 'dreb_pct', 'ft_rate'
        ]
        # Map generic names to actual source columns in the DataFrame for home/away perspectives
        source_mapping_home = { # How to get 'generic_col' from the perspective of the home team
            'home_score': 'score_for',
            'away_score': 'score_against',
            'point_diff': 'point_diff',
            'momentum_score_ewma_q4': 'momentum_ewma', # Make sure this source col exists after momentum step
            'home_offensive_rating': 'off_rating',
            'home_defensive_rating': 'def_rating',
            'home_net_rating': 'net_rating',
            'game_pace': 'pace',
            'home_efg_pct': 'efg_pct',
            'home_tov_rate': 'tov_rate',
            'home_trb_pct': 'trb_pct',
            'home_oreb_pct': 'oreb_pct',
            'home_dreb_pct': 'dreb_pct',
            'home_ft_rate': 'ft_rate'
        }
        source_mapping_away = { # How to get 'generic_col' from the perspective of the away team
            'away_score': 'score_for',
            'home_score': 'score_against',
            'point_diff': 'point_diff', # Sign flip applied later
            'momentum_score_ewma_q4': 'momentum_ewma',
            'away_offensive_rating': 'off_rating',
            'away_defensive_rating': 'def_rating',
            'away_net_rating': 'net_rating',
            'game_pace': 'pace',
            'away_efg_pct': 'efg_pct',
            'away_tov_rate': 'tov_rate',
            'away_trb_pct': 'trb_pct',
            'away_oreb_pct': 'oreb_pct',
            'away_dreb_pct': 'dreb_pct',
            'away_ft_rate': 'ft_rate'
        }

        # --- Check and Prepare Source Columns ---
        if 'point_diff' not in local_df.columns and 'home_score' in local_df.columns and 'away_score' in local_df.columns:
             local_df['point_diff'] = pd.to_numeric(local_df['home_score'], errors='coerce') - pd.to_numeric(local_df['away_score'], errors='coerce')
        if 'game_pace' not in local_df.columns:
             logger.warning("Rolling: 'game_pace' column missing. Pace features rely on defaults.")
             local_df['game_pace'] = self.defaults.get('pace', 100.0) # Assign default pace

        required_sources = set(source_mapping_home.keys()) | set(source_mapping_away.keys()) | {'game_id', 'game_date', 'home_team', 'away_team'}
        missing_sources = [col for col in required_sources if col not in local_df.columns]
        if missing_sources:
             logger.warning(f"Rolling: Missing essential source columns: {missing_sources}. Features might use defaults.")
             # Add missing cols as NaN before proceeding, they will be filled with defaults later
             for col in missing_sources:
                  if col not in local_df.columns: local_df[col] = np.nan

        # Determine which generic columns can actually be calculated
        cols_to_roll_final = []
        for generic_col in cols_to_roll_generic:
            # Find corresponding source columns
            home_source = next((k for k, v in source_mapping_home.items() if v == generic_col), None)
            away_source = next((k for k, v in source_mapping_away.items() if v == generic_col), None)
            # Check if sources exist (even if added as NaN above)
            home_source_exists = home_source in local_df.columns if home_source else False
            away_source_exists = away_source in local_df.columns if away_source else False

            # Special handling for point_diff and game_pace
            if generic_col == 'point_diff' and 'point_diff' in local_df.columns: home_source_exists = away_source_exists = True; home_source = away_source = 'point_diff'
            if generic_col == 'pace' and 'game_pace' in local_df.columns: home_source_exists = away_source_exists = True; home_source = away_source = 'game_pace'

            if home_source_exists and away_source_exists:
                cols_to_roll_final.append(generic_col)
                # Convert sources to numeric safely *now*, applying defaults
                default_val = self.defaults.get(generic_col, 0.0)
                if home_source: local_df[home_source] = pd.to_numeric(local_df[home_source], errors='coerce').fillna(default_val)
                if away_source and away_source != home_source: local_df[away_source] = pd.to_numeric(local_df[away_source], errors='coerce').fillna(default_val)
            else:
                logger.warning(f"Rolling: Cannot calculate for '{generic_col}'. Missing sources: Home='{home_source}', Away='{away_source}'.")

        if not cols_to_roll_final:
            logger.error("Rolling: No valid columns found to calculate rolling features after checking sources.")
            return df # Return original df if no features can be made

        # --- Create Team-Centric View ---
        logger.debug(f"Rolling: Creating team-centric view for columns: {cols_to_roll_final}")
        try:
            local_df['home_team_norm'] = local_df['home_team'].astype(str).apply(self.normalize_team_name)
            local_df['away_team_norm'] = local_df['away_team'].astype(str).apply(self.normalize_team_name)

            # <<< START INSERT (Team View Logging) -- COMMENTED OUT ATM >>>
            #if self.debug:
                #watched_home_rows = local_df[local_df['home_team_norm'].isin(self._teams_to_watch)]
                #if not watched_home_rows.empty:
                     #logger.debug(f"[WATCH_TEAM] Rolling (Pre-TeamView): Found {len(watched_home_rows)} watched home teams. Sample:\n{watched_home_rows[['game_id', 'game_date', 'home_team_norm', 'away_team_norm']].head()}")
                #watched_away_rows = local_df[local_df['away_team_norm'].isin(self._teams_to_watch)]
                #if not watched_away_rows.empty:
                     #logger.debug(f"[WATCH_TEAM] Rolling (Pre-TeamView): Found {len(watched_away_rows)} watched away teams. Sample:\n{watched_away_rows[['game_id', 'game_date', 'home_team_norm', 'away_team_norm']].head()}")
            # <<< END INSERT >>>

            home_data_list = []
            away_data_list = []
            base_cols = ['game_id', 'game_date'] # Ensure game_id exists
            home_view_base = local_df[base_cols + ['home_team_norm']].rename(columns={'home_team_norm': 'team_norm'})
            away_view_base = local_df[base_cols + ['away_team_norm']].rename(columns={'away_team_norm': 'team_norm'})

            for generic_col in cols_to_roll_final:
                home_source = next((k for k, v in source_mapping_home.items() if v == generic_col), None)
                away_source = next((k for k, v in source_mapping_away.items() if v == generic_col), None)

                # Get the correct series for home perspective
                if home_source and home_source in local_df.columns:
                    home_series = local_df[home_source].rename(generic_col)
                    # point_diff perspective is already correct (home - away)
                    home_data_list.append(home_series)

                # Get the correct series for away perspective
                if away_source and away_source in local_df.columns:
                    away_series = local_df[away_source].rename(generic_col)
                    # Flip sign for point_diff perspective (needs to be away - home = -point_diff)
                    if generic_col == 'point_diff' and away_source == 'point_diff':
                        away_series = -away_series
                    # score_against for away team comes from home_score - already handled by mapping
                    away_data_list.append(away_series)

            home_view = pd.concat([home_view_base] + home_data_list, axis=1)
            away_view = pd.concat([away_view_base] + away_data_list, axis=1)

            team_view = (
                pd.concat([home_view, away_view], ignore_index=True)
                .assign(game_date=lambda x: pd.to_datetime(x['game_date'])) # Ensure date is datetime
                .sort_values(['team_norm', 'game_date', 'game_id'], kind='mergesort') # Stable sort needed for rolling
                .reset_index(drop=True)
            )
            # Ensure game_id is string in team_view for merge key consistency
            team_view['game_id'] = team_view['game_id'].astype(str)

            # <<< START INSERT (Team View Logging - After Creation -- COMMENTED OUT ATM) >>>
            #if self.debug:
               #watched_team_view_rows = team_view[team_view['team_norm'].isin(self._teams_to_watch)]
                #if not watched_team_view_rows.empty:
                    #logger.debug(f"[WATCH_TEAM] Rolling: Team view created. Watched teams sample rows:\n{watched_team_view_rows.head()}")
                #else:
                     #logger.debug(f"[WATCH_TEAM] Rolling: No watched teams found in the combined team_view.")
            # <<< END INSERT >>>

        except Exception as e:
            logger.error(f"Rolling: Error creating team view: {e}", exc_info=True)
            # If team view fails, we cannot calculate rolling stats, return original df
            return df

        # --- Calculate Rolling Features on Team View ---
        logger.debug(f"Rolling: Calculating shifted mean/std for windows {window_sizes}...")
        rolling_cols_generated = [] # Keep track of columns successfully generated
        for window in window_sizes:
            min_p = max(1, window // 2) # Ensure min_periods is at least 1
            for col in cols_to_roll_final:
                # Check column exists in team_view (should based on cols_to_roll_final)
                if col not in team_view.columns:
                     logger.warning(f"Rolling: Column '{col}' missing from team_view. Skipping.")
                     continue

                # Define output column names
                roll_mean_col = generate_rolling_column_name('', col, 'mean', window)
                roll_std_col = generate_rolling_column_name('', col, 'std', window)

                try:
                    grouped_col = team_view.groupby('team_norm', observed=True)[col]
                    # Shift by 1 to prevent data leakage from the current game
                    shifted_data = grouped_col.shift(1)
                    # Get rolling object on shifted data
                    rolling_op = shifted_data.rolling(window=window, min_periods=min_p)

                    # Calculate mean and std, fill NaNs resulting *from the rolling operation*
                    team_view[roll_mean_col] = rolling_op.mean()
                    std_dev = rolling_op.std()
                    team_view[roll_std_col] = np.maximum(0, std_dev) # Ensure std is non-negative

                    # Fill remaining NaNs (from shift/min_periods) with appropriate defaults
                    default_mean = self.defaults.get(col, 0.0)
                    # Check for specific std default (e.g., 'pace_std') or use generic 0.0
                    default_std = self.defaults.get(f'{col}_std', 0.0)

                    team_view[roll_mean_col] = team_view[roll_mean_col].fillna(default_mean)
                    team_view[roll_std_col] = team_view[roll_std_col].fillna(default_std)

                    # Add successfully generated columns to list
                    rolling_cols_generated.extend([roll_mean_col, roll_std_col])

                except Exception as calc_err:
                     logger.error(f"Rolling: Error calculating for col='{col}', window={window}: {calc_err}", exc_info=True)
                     # Add columns with defaults if calculation failed
                     team_view[roll_mean_col] = self.defaults.get(col, 0.0)
                     team_view[roll_std_col] = self.defaults.get(f'{col}_std', 0.0)
                     # Do not add to rolling_cols_generated if failed

        # Remove duplicates just in case? Unlikely needed here if sort is correct.
        rolling_cols_generated = sorted(list(set(rolling_cols_generated))) # Get unique list


        # --- Merge Rolling Stats Back to Original DataFrame ---
        logger.debug("Rolling: Merging rolling stats back...")

        # <<< FIX: Initialize parsed_bases BEFORE the try block >>>
        parsed_bases = set() # Ensures variable exists for the except block

        try: # Main try block for merging and renaming
             merge_data = None # Initialize merge_data to None

             # --- Sub-Try block for key/merge_data creation ---
             try:
                 logger.debug("Rolling: Preparing merge keys and data...")
                 # Build merge keys robustly (ensure game_id is string)
                 local_df['game_id'] = local_df['game_id'].astype(str)
                 team_view['game_id'] = team_view['game_id'].astype(str) # Redundant? Belt & suspenders.

                 team_view['merge_key_rolling'] = team_view['game_id'] + "_" + team_view['team_norm'].astype(str)
                 local_df['merge_key_home'] = local_df['game_id'] + "_" + local_df['home_team_norm'].astype(str)
                 local_df['merge_key_away'] = local_df['game_id'] + "_" + local_df['away_team_norm'].astype(str)

                 # Prepare data to merge - ensure only valid generated columns are selected
                 cols_to_merge = ['merge_key_rolling'] + [c for c in rolling_cols_generated if c in team_view.columns]
                 if len(cols_to_merge) <= 1 : # Only contains the key
                     logger.warning("Rolling: No rolling columns were successfully generated to merge.")
                     # If no columns to merge, skip merge logic, but maybe add defaults later?
                     # For now, let it proceed, merge will likely be empty/ineffective.
                     merge_data = pd.DataFrame(columns=['merge_key_rolling']) # Create empty DF with key
                 else:
                     logger.debug(f"Rolling: Columns selected for merge: {cols_to_merge}")
                     merge_data = (
                         team_view[cols_to_merge]
                         .drop_duplicates(subset=['merge_key_rolling'], keep='last')
                     )
                     logger.debug(f"Rolling: merge_data shape after selection/deduplication: {merge_data.shape}")

             except Exception as prep_err:
                 logger.error(f"Rolling: Error PREPARING merge keys or data: {prep_err}", exc_info=True)
                 # If prep fails, set merge_data to None or empty to prevent later errors
                 merge_data = None # Signal that merge cannot proceed


             # --- Proceed with Merge only if merge_data is valid ---
             if merge_data is not None and not merge_data.empty:
                 # <<< START INSERT - Before Home Merge-- COMMENTED OUT ATM>>>
                 #if self.debug:
                     #logger.debug(f"Rolling Merge (Home): Left DF shape: {local_df.shape}, Right DF (merge_data) shape: {merge_data.shape}")
                     #watched_left_home = local_df[local_df['home_team_norm'].isin(self._teams_to_watch)]
                     #if not watched_left_home.empty:
                         #logger.debug(f"[WATCH_TEAM] Rolling Merge (Home) - Left DF sample (watched teams):\n{watched_left_home[['game_id', 'game_date', 'home_team_norm', 'merge_key_home']].head()}")
                         #watched_keys_home = watched_left_home['merge_key_home'].unique()
                         #watched_right_home = merge_data[merge_data['merge_key_rolling'].isin(watched_keys_home)]
                         #if not watched_right_home.empty: logger.debug(f"[WATCH_TEAM] Rolling Merge (Home) - Right DF sample (matching watched keys):\n{watched_right_home.head()}")
                         #else: logger.debug(f"[WATCH_TEAM] Rolling Merge (Home) - Right DF: No matching keys found for watched teams.")
                     #else: logger.debug(f"[WATCH_TEAM] Rolling Merge (Home): No watched teams found in Left DF.")
                 # <<< END INSERT >>>

                 # --- Dictionary Comprehensions for Renaming ---
                 home_rename_dict = {}
                 away_rename_dict = {}
                 # parsed_bases = set() # Moved initialization outside try block

                 for col in merge_data.columns:
                     if col == 'merge_key_rolling': continue
                     try:
                         parts = col.split('_')
                         if len(parts) >= 4 and parts[0] == 'rolling':
                             window = int(parts[-1])
                             stat_type = parts[-2]
                             base = "_".join(parts[1:-2])
                             parsed_bases.add(base) # Track the base column name

                             home_rename_dict[col] = generate_rolling_column_name('home', base, stat_type, window)
                             away_rename_dict[col] = generate_rolling_column_name('away', base, stat_type, window)
                         else: logger.warning(f"Rolling: Could not parse column '{col}' during rename.")
                     except (ValueError, IndexError) as parse_err: logger.warning(f"Rolling: Error parsing column '{col}' for rename: {parse_err}.")

                 # --- Merge Logic ---
                 local_df = pd.merge(local_df, merge_data, how='left',
                                     left_on='merge_key_home', right_on='merge_key_rolling'
                                    ).rename(columns=home_rename_dict)
                 if 'merge_key_rolling' in local_df.columns: local_df = local_df.drop(columns=['merge_key_rolling'])

                 # <<< START INSERT - Before Away Merge-- COMMENTED OUT ATM>>>
                 #if self.debug:
                     # Log shapes and watched teams again before away merge
                     #logger.debug(f"Rolling Merge (Away): Left DF shape: {local_df.shape}, Right DF (merge_data) shape: {merge_data.shape}")
                     #watched_left_away = local_df[local_df['away_team_norm'].isin(self._teams_to_watch)]
                     #if not watched_left_away.empty:
                         #logger.debug(f"[WATCH_TEAM] Rolling Merge (Away) - Left DF sample (watched teams):\n{watched_left_away[['game_id', 'game_date', 'away_team_norm', 'merge_key_away']].head()}")
                         #watched_keys_away = watched_left_away['merge_key_away'].unique()
                         #watched_right_away = merge_data[merge_data['merge_key_rolling'].isin(watched_keys_away)]
                         #if not watched_right_away.empty: logger.debug(f"[WATCH_TEAM] Rolling Merge (Away) - Right DF sample (matching watched keys):\n{watched_right_away.head()}")
                         #else: logger.debug(f"[WATCH_TEAM] Rolling Merge (Away) - Right DF: No matching keys found.")
                     #else: logger.debug(f"[WATCH_TEAM] Rolling Merge (Away): No watched teams found in Left DF.")
                 # <<< END INSERT >>>

                 local_df = pd.merge(local_df, merge_data, how='left', # Use same merge_data
                                     left_on='merge_key_away', right_on='merge_key_rolling'
                                    ).rename(columns=away_rename_dict)
                 if 'merge_key_rolling' in local_df.columns: local_df = local_df.drop(columns=['merge_key_rolling'])
                 # --- End Merge Logic ---

             else: # Handle case where merge_data prep failed or was empty
                  logger.warning("Rolling: Skipping merge operations as merge_data was not prepared successfully or was empty.")
                  # Consider adding default columns here if merge skipped entirely
                  # This logic is now primarily handled by the main except block below

        # --- Main Exception Handler for Merging/Renaming ---
        except Exception as e:
            logger.error(f"Rolling: Error during merge/rename stats back: {e}", exc_info=True)
            # Fallback: Add columns with defaults if merge failed
            missing_cols_defaults = {}
            # Use cols_to_roll_final as fallback if parsed_bases is empty (e.g., error before rename)
            bases_for_fallback = parsed_bases if parsed_bases else cols_to_roll_final

            logger.warning(f"Rolling: Attempting to add defaults for bases: {bases_for_fallback}")
            for prefix in ['home', 'away']:
                for base in bases_for_fallback:
                    for stat in ['mean', 'std']:
                        for w in window_sizes:
                            col_name = generate_rolling_column_name(prefix, base, stat, w)
                            if col_name not in local_df.columns:
                                default_key = f'{base}_std' if stat == 'std' else base
                                # Use .get() with another fallback for safety
                                missing_cols_defaults[col_name] = self.defaults.get(default_key, 0.0)

            if missing_cols_defaults:
                 logger.warning(f"Adding default values for {len(missing_cols_defaults)} missing rolling columns due to error.")
                 local_df = local_df.assign(**missing_cols_defaults)


        # --- Final NaN Filling and Differential Calculation ---
        logger.debug("Rolling: Finalizing (filling NaNs, calculating diffs)...")
        primary_window = max(window_sizes) if window_sizes else 10
        new_diff_cols = {}

        for stat_type in ['mean', 'std']:
            for base_col in cols_to_roll_final: # Iterate through cols attempted
                # Fill NaNs for existing rolling columns first
                for w in window_sizes:
                    for prefix in ['home', 'away']:
                        col_name = generate_rolling_column_name(prefix, base_col, stat_type, w)
                        default_key = f'{base_col}_std' if stat_type == 'std' else base_col
                        default = self.defaults.get(default_key, 0.0)
                        if col_name not in local_df.columns:
                            # Add column if missing entirely (e.g., due to error)
                            logger.warning(f"Rolling column '{col_name}' missing before final fill/diff. Adding default.")
                            local_df[col_name] = default
                        else:
                            # Fill NaNs that might still exist
                            local_df[col_name] = local_df[col_name].fillna(default)
                        # Ensure std dev is non-negative after filling NaNs
                        if stat_type == 'std': local_df[col_name] = np.maximum(0, local_df[col_name])

                # --- Calculate differential using primary window ---
                w = primary_window
                home_col = generate_rolling_column_name('home', base_col, stat_type, w)
                away_col = generate_rolling_column_name('away', base_col, stat_type, w)

                # Map generic base_col name for the differential column
                expected_base_name = base_col
                if base_col == 'point_diff': expected_base_name = 'margin'
                elif base_col == 'net_rating': expected_base_name = 'eff'
                elif base_col.endswith('_pct'): expected_base_name = base_col[:-4] # Remove _pct
                elif base_col == 'momentum_ewma': expected_base_name = 'momentum'

                diff_col_name = f'rolling_{expected_base_name}_diff_{stat_type}'

                # Check if source columns for diff exist after merging/filling
                if home_col in local_df.columns and away_col in local_df.columns:
                    home_vals = local_df[home_col]
                    away_vals = local_df[away_col]
                    # Use subtraction for differentials instead of ratio for simplicity/stability
                    # Positive diff means home advantage (except for 'lower is better' stats)
                    if base_col in ['tov_rate', 'def_rating', 'score_against']:
                        # Lower is better: away - home -> positive means home is better
                        diff = away_vals - home_vals
                    else:
                        # Higher is better: home - away -> positive means home is better
                        diff = home_vals - away_vals

                    new_diff_cols[diff_col_name] = diff.fillna(0.0) # Fill NaN diffs with 0.0
                else:
                    logger.warning(f"Could not calculate differential for {diff_col_name}, missing sources: '{home_col}' or '{away_col}'. Assigning default 0.0.")
                    new_diff_cols[diff_col_name] = 0.0

        # Assign all new differential columns at once
        if new_diff_cols: local_df = local_df.assign(**new_diff_cols)

        # --- Clean up intermediate columns ---
        logger.debug("Rolling: Cleaning up intermediate columns...")
        cols_to_drop = ['merge_key_home', 'merge_key_away', 'home_team_norm', 'away_team_norm']
        local_df = local_df.drop(columns=cols_to_drop, errors='ignore')

        logger.debug("Finished adding rolling features.")
        return local_df

    @profile_time
    def add_rest_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds rest days, games in last N days, back-to-back flags, and schedule advantages."""
        logger.debug("Adding rest features (vectorized)...")
        essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
        placeholder_cols = [
            'rest_days_home', 'games_last_7_days_home', 'games_last_14_days_home',
            'rest_days_away', 'games_last_7_days_away', 'games_last_14_days_away',
            'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage'
        ]
        expected_schedule_cols = [
             'games_last_7_days_home', 'games_last_14_days_home',
             'games_last_7_days_away', 'games_last_14_days_away'
        ]
        if df is None or df.empty or not all(col in df.columns for col in essential_cols):
            logger.warning("Input df empty or missing essential columns for rest features. Returning defaults.")
            output_df = df.copy() if df is not None else pd.DataFrame()
            for col in placeholder_cols:
                if col not in output_df.columns:
                    default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '')
                    output_df[col] = self.defaults.get(default_key, 0.0)
            return output_df

        df_copy = df.copy().reset_index(drop=True)
        try:
            df_copy['game_date'] = pd.to_datetime(df_copy['game_date'], errors='coerce').dt.tz_localize(None)
            df_copy['home_team'] = df_copy['home_team'].astype(str)
            df_copy['away_team'] = df_copy['away_team'].astype(str)
            df_copy['game_id'] = df_copy['game_id'].astype(str)
            if df_copy['game_date'].isnull().all():
                logger.warning("All game_date values invalid/missing after conversion. Cannot calculate rest features accurately.")
                raise ValueError("Invalid game dates")
        except Exception as e:
            logger.error(f"Error processing essential columns for rest features: {e}. Returning defaults.", exc_info=True)
            for col in placeholder_cols:
                if col not in df_copy.columns:
                    default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '')
                    df_copy[col] = self.defaults.get(default_key, 0.0)
            return df_copy

        logger.debug("Calculating rest days...")
        # Ensure team_log is unique by (team, game_date)
        team_log = (
            pd.concat([
                df_copy[['game_date', 'home_team']].rename(columns={'home_team': 'team'}),
                df_copy[['game_date', 'away_team']].rename(columns={'away_team': 'team'})
            ], ignore_index=True)
            .sort_values(['team', 'game_date'])
            .drop_duplicates(subset=['team', 'game_date'], keep='first')
        )
        team_log['prev_game_date'] = team_log.groupby('team', observed=True)['game_date'].shift(1)

        # Use a deduplicated temporary DataFrame for merging rest days
        temp_rest = team_log[['team', 'game_date', 'prev_game_date']].drop_duplicates(subset=['team', 'game_date'])
        # Merge for home team with merge validation to catch unexpected duplicates
        
        # <<< START INSERT - Before Home Rest Merge -- COMMENTED OUT ATM >>>
        #if self.debug:
            #logger.debug(f"Rest Merge (Home Prev Date): Left DF shape: {df_copy.shape}, Right DF (temp_rest) shape: {temp_rest.shape}")
            #watched_left_rest_h = df_copy[df_copy['home_team'].apply(self.normalize_team_name).isin(self._teams_to_watch)]
            #if not watched_left_rest_h.empty:
                 #logger.debug(f"[WATCH_TEAM] Rest Merge (Home Prev Date) - Left DF sample (watched teams):\n{watched_left_rest_h[['game_id', 'game_date', 'home_team']].head()}")
                 #watched_teams_rest = self._teams_to_watch # Re-evaluate normalization if needed here
                # watched_right_rest_h = temp_rest[temp_rest['team'].apply(self.normalize_team_name).isin(watched_teams_rest)]
                 #if not watched_right_rest_h.empty:
                     #logger.debug(f"[WATCH_TEAM] Rest Merge (Home Prev Date) - Right DF sample (watched teams):\n{watched_right_rest_h.head()}")
                 #else:
                     #logger.debug(f"[WATCH_TEAM] Rest Merge (Home Prev Date) - Right DF: No watched teams found.")
            #else:
                 #logger.debug(f"[WATCH_TEAM] Rest Merge (Home Prev Date): No watched teams found in Left DF.")
        # <<< END INSERT >>>
        
        df_copy = pd.merge(
            df_copy,
            temp_rest,
            how='left',
            left_on=['home_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'  # Expect each home_team/game_date to match at most one row
        ).rename(columns={'prev_game_date': 'prev_home_game_date'}).drop(columns='team', errors='ignore')

        # <<< START INSERT - Before Away Rest Merg -- COMMENTED OUT ATMe >>>
        #if self.debug:
            #logger.debug(f"Rest Merge (Away Prev Date): Left DF shape: {df_copy.shape}, Right DF (temp_rest) shape: {temp_rest.shape}")
            #watched_left_rest_a = df_copy[df_copy['away_team'].apply(self.normalize_team_name).isin(self._teams_to_watch)]
            #if not watched_left_rest_a.empty:
                 #logger.debug(f"[WATCH_TEAM] Rest Merge (Away Prev Date) - Left DF sample (watched teams):\n{watched_left_rest_a[['game_id', 'game_date', 'away_team']].head()}")
                 #watched_teams_rest = self._teams_to_watch # Re-evaluate normalization if needed here
                 #watched_right_rest_a = temp_rest[temp_rest['team'].apply(self.normalize_team_name).isin(watched_teams_rest)]
                 #if not watched_right_rest_a.empty:
                     #logger.debug(f"[WATCH_TEAM] Rest Merge (Away Prev Date) - Right DF sample (watched teams):\n{watched_right_rest_a.head()}")
                 #else:
                     #logger.debug(f"[WATCH_TEAM] Rest Merge (Away Prev Date) - Right DF: No watched teams found.")
            #else:
                 #logger.debug(f"[WATCH_TEAM] Rest Merge (Away Prev Date): No watched teams found in Left DF.")
        # <<< END INSERT >>>

        # Merge for away team with similar safeguards
        df_copy = pd.merge(
            df_copy,
            temp_rest,
            how='left',
            left_on=['away_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'
        ).rename(columns={'prev_game_date': 'prev_away_game_date'}).drop(columns='team', errors='ignore')

        df_copy['rest_days_home'] = (df_copy['game_date'] - df_copy['prev_home_game_date']).dt.days.fillna(self.defaults['rest_days'])
        df_copy['rest_days_away'] = (df_copy['game_date'] - df_copy['prev_away_game_date']).dt.days.fillna(self.defaults['rest_days'])

        logger.debug("Calculating games in last 7/14 days...")

        # <<< START INSERT - After Rest Days Calc -- COMMENTED OUT ATM >>>
        #if self.debug:
            #watched_rows_after_rest = df_copy[
                #df_copy['home_team'].apply(self.normalize_team_name).isin(self._teams_to_watch) |
                #df_copy['away_team'].apply(self.normalize_team_name).isin(self._teams_to_watch)
            #]
            #if not watched_rows_after_rest.empty:
                #logger.debug(f"[WATCH_TEAM] Calculated Rest Days (watched teams sample): \n{watched_rows_after_rest[['game_id', 'game_date', 'home_team', 'away_team', 'prev_home_game_date', 'prev_away_game_date', 'rest_days_home', 'rest_days_away']].head()}")
        # <<< END INSERT >>>

        try:
            # Build a lookup for game_ids per team and game_date
            home_log_ids = df_copy[['game_date', 'home_team', 'game_id']].rename(columns={'home_team': 'team'})
            away_log_ids = df_copy[['game_date', 'away_team', 'game_id']].rename(columns={'away_team': 'team'})
            game_ids_log = pd.concat([home_log_ids, away_log_ids], ignore_index=True)
            # Ensure uniqueness on (team, game_date)
            game_ids_log = game_ids_log.drop_duplicates(subset=['team', 'game_date'], keep='first')

            # Merge the unique game_ids back to team_log
            team_log_with_id = pd.merge(
                team_log,
                game_ids_log,
                on=['team', 'game_date'],
                how='left',
                validate='one_to_one'
            )
            if team_log_with_id['game_id'].isnull().any():
                logger.warning("Could not associate game_id with all team-date entries in team_log. Rolling counts might be affected.")

            team_log_indexed = team_log_with_id.set_index('game_date')
            counts_7d_series = team_log_indexed.groupby('team', observed=True)['game_id'].rolling('7D', closed='left').count()
            counts_14d_series = team_log_indexed.groupby('team', observed=True)['game_id'].rolling('14D', closed='left').count()
            counts_7d_series.name = 'count_7d'
            counts_14d_series.name = 'count_14d'
            counts_7d_df = counts_7d_series.reset_index()
            counts_14d_df = counts_14d_series.reset_index()

            # Merge counts back to team_log ensuring uniqueness
            team_log_with_counts = pd.merge(
                team_log, counts_7d_df, on=['team', 'game_date'], how='left', validate='one_to_one'
            )
            team_log_with_counts = pd.merge(
                team_log_with_counts, counts_14d_df, on=['team', 'game_date'], how='left', validate='one_to_one'
            )
            team_log_with_counts['count_7d'] = team_log_with_counts['count_7d'].fillna(0).astype(int)
            team_log_with_counts['count_14d'] = team_log_with_counts['count_14d'].fillna(0).astype(int)
            team_log_with_counts = team_log_with_counts.drop_duplicates(subset=['team', 'game_date'])


             # <<< START INSERT - Before Home Schedule Merge -- COMMENTED OUT ATM >>>
            #if self.debug:
                 #logger.debug(f"Schedule Merge (Home): Left DF shape: {df_copy.shape}, Right DF (temp_sched) shape: {temp_sched.shape}")
                 #watched_left_sched_h = df_copy[df_copy['home_team'].apply(self.normalize_team_name).isin(self._teams_to_watch)]
                 #if not watched_left_sched_h.empty:
                     #logger.debug(f"[WATCH_TEAM] Schedule Merge (Home) - Left DF sample (watched teams):\n{watched_left_sched_h[['game_id', 'game_date', 'home_team']].head()}")
                     #watched_teams_sched = self._teams_to_watch # Re-evaluate normalization if needed
                     #watched_right_sched_h = temp_sched[temp_sched['team'].apply(self.normalize_team_name).isin(watched_teams_sched)]
                     #if not watched_right_sched_h.empty:
                         #logger.debug(f"[WATCH_TEAM] Schedule Merge (Home) - Right DF sample (watched teams):\n{watched_right_sched_h.head()}")
                     #else:
                         #logger.debug(f"[WATCH_TEAM] Schedule Merge (Home) - Right DF: No watched teams found.")
                 #else:
                      #logger.debug(f"[WATCH_TEAM] Schedule Merge (Home): No watched teams found in Left DF.")
             # <<< END INSERT >>>

            # Prepare a deduplicated temp for schedule counts merges
            temp_sched = team_log_with_counts[['team', 'game_date', 'count_7d', 'count_14d']].drop_duplicates(subset=['team', 'game_date'])
            df_copy = pd.merge(
                df_copy,
                temp_sched,
                how='left',
                left_on=['home_team', 'game_date'],
                right_on=['team', 'game_date'],
                validate='many_to_one'
            ).rename(columns={'count_7d': 'games_last_7_days_home', 'count_14d': 'games_last_14_days_home'}).drop(columns='team', errors='ignore')
            
            # <<< START INSERT - Before Away Schedule Merge -- COMMENTED OUT ATM >>>
            #if self.debug:
                 #logger.debug(f"Schedule Merge (Away): Left DF shape: {df_copy.shape}, Right DF (temp_sched) shape: {temp_sched.shape}")
                 #watched_left_sched_a = df_copy[df_copy['away_team'].apply(self.normalize_team_name).isin(self._teams_to_watch)]
                 #if not watched_left_sched_a.empty:
                     #logger.debug(f"[WATCH_TEAM] Schedule Merge (Away) - Left DF sample (watched teams):\n{watched_left_sched_a[['game_id', 'game_date', 'away_team']].head()}")
                     #watched_teams_sched = self._teams_to_watch # Re-evaluate normalization if needed
                     #watched_right_sched_a = temp_sched[temp_sched['team'].apply(self.normalize_team_name).isin(watched_teams_sched)]
                     #if not watched_right_sched_a.empty:
                         #logger.debug(f"[WATCH_TEAM] Schedule Merge (Away) - Right DF sample (watched teams):\n{watched_right_sched_a.head()}")
                     #else:
                         #logger.debug(f"[WATCH_TEAM] Schedule Merge (Away) - Right DF: No watched teams found.")
                 #else:
                      #logger.debug(f"[WATCH_TEAM] Schedule Merge (Away): No watched teams found in Left DF.")
             # <<< END INSERT >>>

            df_copy = pd.merge(
                df_copy,
                temp_sched,
                how='left',
                left_on=['away_team', 'game_date'],
                right_on=['team', 'game_date'],
                validate='many_to_one'
            ).rename(columns={'count_7d': 'games_last_7_days_away', 'count_14d': 'games_last_14_days_away'}).drop(columns='team', errors='ignore')
            
            for col in expected_schedule_cols:
                df_copy[col] = df_copy[col].fillna(self.defaults.get(col, 0)).astype(int)
        except Exception as rolling_e:
            logger.error(f"Error during time-based rolling count for schedule density: {rolling_e}", exc_info=True)
            logger.warning("Filling schedule density columns with defaults due to calculation error.")
            for col in expected_schedule_cols:
                if col not in df_copy.columns:
                    df_copy[col] = self.defaults.get(col, 0)
                else:
                    df_copy[col] = df_copy[col].fillna(self.defaults.get(col, 0))
                df_copy[col] = df_copy[col].astype(int)
        df_copy['is_back_to_back_home'] = (df_copy['rest_days_home'] == 1).astype(int)
        df_copy['is_back_to_back_away'] = (df_copy['rest_days_away'] == 1).astype(int)
        df_copy['rest_advantage'] = df_copy['rest_days_home'] - df_copy['rest_days_away']
        df_copy['schedule_advantage'] = df_copy['games_last_7_days_away'] - df_copy['games_last_7_days_home']
        df_copy = df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date'], errors='ignore')
        logger.debug("Finished adding rest features.")
        final_cols = list(df.columns) + [col for col in placeholder_cols if col not in df.columns]
        for col in placeholder_cols:
            if col not in df_copy.columns:
                default_key = col.replace('_home', '').replace('_away', '').replace('_advantage', '')
                df_copy[col] = self.defaults.get(default_key, 0.0)
        return df_copy[final_cols]

    @profile_time
    def add_matchup_history_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame], max_games: int = 7) -> pd.DataFrame: # Default max_games set here for clarity, but uses value passed from generate_all_features
        """Adds head-to-head matchup statistics based on recent historical games."""
        logger.debug(f"Adding head-to-head matchup features (last {max_games} games)...")
        if df is None or df.empty:
            logger.warning("H2H: Input df is empty.")
            return df
        result_df = df.copy()

        # Define placeholder columns based on the helper function's expected output keys
        # Ensure _get_matchup_history_single returns expected structure even with empty input for this setup
        placeholder_cols = [
            'matchup_num_games', 'matchup_avg_point_diff', 'matchup_home_win_pct',
            'matchup_avg_total_score', 'matchup_avg_home_score', 'matchup_avg_away_score',
            'matchup_last_date', 'matchup_streak'
        ]
        if historical_df is None or historical_df.empty:
            logger.warning("H2H: Historical DataFrame empty or None. Adding H2H placeholders with defaults.")
            for col in placeholder_cols:
                if col not in result_df.columns:
                    default_key = col.replace('matchup_', '')
                    default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                    result_df[col] = default_val
            return result_df # Return early if no historical data

        try:
            # --- Prepare Historical Data ---
            hist_df = historical_df.copy()
            hist_required = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
            if not all(col in hist_df.columns for col in hist_required):
                missing_hist_cols = set(hist_required) - set(hist_df.columns)
                raise ValueError(f"H2H: Historical DF missing required columns: {missing_hist_cols}")

            # Convert types and clean historical data
            hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
            hist_df['home_score'] = pd.to_numeric(hist_df['home_score'], errors='coerce')
            hist_df['away_score'] = pd.to_numeric(hist_df['away_score'], errors='coerce')
            hist_df = hist_df.dropna(subset=['game_date', 'home_score', 'away_score'])

            if hist_df.empty:
                 logger.warning("H2H: Historical DataFrame has no valid rows after cleaning. Adding H2H placeholders.")
                 # Add placeholders similar to the check above if needed, though it should fall through correctly
                 # (Returning result_df here might be safer if downstream relies on these columns existing)

            # Normalize team names and create matchup keys in historical data
            hist_df['home_team_norm'] = hist_df['home_team'].astype(str).apply(self.normalize_team_name)
            hist_df['away_team_norm'] = hist_df['away_team'].astype(str).apply(self.normalize_team_name)
            hist_df['matchup_key'] = hist_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            hist_df = hist_df.sort_values('game_date') # Sort for correct history lookup

            # Create a lookup dictionary for faster access
            #logger.debug("H2H: Grouping historical data by matchup key...")
            hist_lookup = {key: group for key, group in hist_df.groupby('matchup_key', observed=True)}
            #logger.debug(f"H2H: Created lookup for {len(hist_lookup)} unique matchup keys.")

            # --- Prepare Target Data ---
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date']) # Drop rows where target date is invalid

            if result_df.empty:
                logger.warning("H2H: No valid input rows remaining in target df after date processing.")
                # Ensure placeholder columns exist if returning empty or partially processed df
                for col in placeholder_cols:
                    if col not in result_df.columns:
                        default_key = col.replace('matchup_', '')
                        default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                        result_df[col] = default_val
                return result_df # Return early if no valid target rows

            # Normalize team names and create matchup keys in target data
            result_df['home_team_norm'] = result_df['home_team'].astype(str).apply(self.normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].astype(str).apply(self.normalize_team_name)
            result_df['matchup_key'] = result_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)

            # --- Calculate H2H Features Row-by-Row ---
            logger.debug("H2H: Calculating features row by row...")
            h2h_results_list = []

                        # Loop through each target game row
            for index, row in result_df.iterrows():
                # --- Define variables for this specific row ---
                home_norm = row.get('home_team_norm', 'Unknown_Home') # Get normalized home team
                away_norm = row.get('away_team_norm', 'Unknown_Away') # Get normalized away team
                matchup_key = row.get('matchup_key', 'Unknown_Key')   # Get the pre-calculated key
                game_id = row.get('game_id', 'Unknown_ID')           # Get game ID
                # Get the date for THIS game, used for filtering history
                current_game_date = row.get('game_date', pd.NaT)

                # --- Check if this row involves a watched team ---
                is_watched = home_norm in self._teams_to_watch or away_norm in self._teams_to_watch

            # --- Logging Block 1: Log basic info for watched teams (COMMENTED OUT) ---
            # if self.debug and is_watched:
            #    logger.debug(f"[WATCH_TEAM] H2H Processing Row: GameID={game_id}, Date={current_game_date}, Home={home_norm}, Away={away_norm}, Key={matchup_key}")

            # --- Find historical games for this specific matchup key ---
            matchup_hist_subset = hist_lookup.get(matchup_key, pd.DataFrame())

            # --- Logging Block 2: Log the historical data found for watched teams (COMMENTED OUT) ---
            # if self.debug and is_watched:  # <<< Comment out this entire conditional block
            #      if matchup_hist_subset.empty:
            #          logger.debug(f"[WATCH_TEAM] H2H History Lookup: No historical data found in lookup for key '{matchup_key}'")
            #      else:
            #          # Filter history to games strictly BEFORE the current game date
            #          if pd.notna(current_game_date): # Check if current_game_date is valid
            #             relevant_hist = matchup_hist_subset[matchup_hist_subset['game_date'] < current_game_date]
            #             logger.debug(f"[WATCH_TEAM] H2H History Found: Found {len(relevant_hist)} past games for key '{matchup_key}' before {current_game_date}. Sample of most recent {max_games}:\n{relevant_hist.sort_values('game_date', ascending=False).head(max_games)[['game_date', 'home_team_norm', 'away_team_norm', 'home_score', 'away_score']]}")
            #          else:
            #             # Log if we cannot filter due to invalid date
            #             logger.debug(f"[WATCH_TEAM] H2H History Found: Cannot filter history for GameID={game_id} because current_game_date is invalid ({current_game_date}).")

            # --- Calculate H2H stats using the helper function ---
            # This part remains active and is essential
            single_h2h_stats = self._get_matchup_history_single(
                home_team_norm=home_norm,                 # Pass correct home team
                away_team_norm=away_norm,                 # Pass correct away team
                historical_subset=matchup_hist_subset,    # Pass the specific history found
                max_games=max_games,                      # Pass the function's parameter
                current_game_date=current_game_date     # Pass the date of the current game
            )


            # +++ ADD THIS PRINT STATEMENT (maybe limit iterations) +++
            if index < 20: # Print for first 20 games only
                 print(f"DEBUG LOOP - Game {game_id} - Index {index}: {single_h2h_stats}")
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


            # --- Logging Block 3: Log the result for watched teams (COMMENTED OUT) ---
            # if self.debug and is_watched:
            #      logger.debug(f"[WATCH_TEAM] H2H Calculated Result (GameID={game_id}): {single_h2h_stats}")

            # --- Append result to list ---
            # This part remains active and is essential
            h2h_results_list.append(single_h2h_stats)

            # --- End of loop ---
            # --- Combine Results ---
            if h2h_results_list:
                h2h_results_df = pd.DataFrame(h2h_results_list, index=result_df.index)
                if self.debug:
                    logger.debug("H2H: h2h_results_df Info:")
                    h2h_results_df.info(verbose=True, show_counts=True)
                    logger.debug("H2H: h2h_results_df Describe:\n" + h2h_results_df.describe().to_string())
                result_df = result_df.join(h2h_results_df, how='left')
        #logger.debug(f"H2H: Joined {len(h2h_results_df)} rows of H2H results.")
            else:
                logger.warning("H2H: No head-to-head results were generated (list was empty).")
                # Ensure placeholder columns exist even if no results generated
                for col in placeholder_cols:
                     if col not in result_df.columns: result_df[col] = np.nan # Add as NaN first



            # --- Finalize and Fill Defaults ---
            logger.debug("H2H: Finalizing features (filling defaults)...")
            for col in placeholder_cols:
                default_key = col.replace('matchup_', '')
                # Determine the correct default value based on column name
                default_val = self.defaults.get(default_key, 0.0) # Default to 0.0
                if col == 'matchup_last_date':
                    default_val = pd.NaT
                elif col == 'matchup_home_win_pct':
                     default_val = self.defaults.get('matchup_home_win_pct', 0.5) # Specific default

                # Fill NaNs or add column if missing entirely
                if col not in result_df.columns:
                    result_df[col] = default_val
                else:
                    # Fill NaNs that might exist from failed calculations or joins
                    result_df[col] = result_df[col].fillna(default_val)

                # Ensure correct data types after filling
                if col == 'matchup_last_date':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                elif col in ['matchup_num_games', 'matchup_streak']:
                    # Ensure these are numeric before rounding/casting
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).round().astype(int)
                else:
                    # Ensure other stats are numeric
                     result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)


        except Exception as e:
            logger.error(f"H2H: Error adding features: {e}", exc_info=True)
            # Attempt to add default placeholders if an error occurred mid-processing
            logger.warning("H2H: Adding placeholders due to error during processing.")
            for col in placeholder_cols:
                if col not in result_df.columns:
                    default_key = col.replace('matchup_', '')
                    default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                    if col == 'matchup_home_win_pct': default_val = self.defaults.get('matchup_home_win_pct', 0.5)
                    result_df[col] = default_val
                    # Apply types again just in case
                    if col == 'matchup_last_date': result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                    elif col in ['matchup_num_games', 'matchup_streak']: result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).round().astype(int)
                    else: result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)


        if self.debug and not result_df.empty:
                matchup_cols_in_final = [col for col in placeholder_cols if col in result_df.columns and col != 'matchup_last_date']
                if matchup_cols_in_final:
                    variances_final = result_df[matchup_cols_in_final].var()
                    logger.debug(f"H2H: Final Variances before return:\n{variances_final}")

        logger.debug("H2H: Finished adding head-to-head features.")
        return result_df

    @profile_time
    def add_season_context_features(self, df: pd.DataFrame, team_stats_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merges seasonal team statistics onto the game data."""
        logger.info("Adding season context features...")
        result_df = df.copy()
        placeholder_cols = [
            'home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for', 'away_season_avg_pts_for',
            'home_season_avg_pts_against', 'away_season_avg_pts_against', 'home_current_form', 'away_current_form',
            'season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff',
            'home_season_net_rating', 'away_season_net_rating', 'season_net_rating_diff'
        ]
        req_ts_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
        team_stats_available = False
        ts_df = None
        if team_stats_df is not None and not team_stats_df.empty:
            ts_df = team_stats_df.copy()
            missing_ts_cols = [col for col in req_ts_cols if col not in ts_df.columns]
            if missing_ts_cols:
                logger.warning(f"Team stats DF missing required columns: {missing_ts_cols}. Filling with defaults.")
                for col in missing_ts_cols:
                    if col == 'current_form':
                        ts_df[col] = 'N/A'
                    elif 'percentage' in col:
                        ts_df[col] = 0.5
                    else:
                        ts_df[col] = 0.0
            team_stats_available = True
        else:
            logger.warning("`team_stats_df` is empty or None. Season context features will use defaults.")
        if 'game_date' not in result_df.columns or result_df['game_date'].isna().all():
            logger.error("`game_date` missing or all NaN in input df. Cannot determine season for merging.")
            team_stats_available = False
        if not team_stats_available:
            logger.warning("Adding season context placeholders with defaults.")
            for col in placeholder_cols:
                if col not in result_df.columns:
                    default_key = col.replace('home_', '').replace('away_', '').replace('season_', '').replace('_diff', '').replace('_net_rating','')
                    default_val = self.defaults.get(default_key, 0.0 if 'form' not in col else 'N/A')
                    if 'win_pct' in col:
                        default_val = self.defaults.get('win_pct', 0.5)
                    if 'diff' in col:
                        default_val = 0.0
                    result_df[col] = default_val
                if 'form' in col:
                    result_df[col] = result_df[col].fillna('N/A').astype(str)
                else:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
            return result_df

        try:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date'])
            if result_df.empty:
                raise ValueError("No valid game dates in target df after cleaning.")
            result_df['season'] = result_df['game_date'].apply(self._determine_season)
            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            ts_df['team_name_norm'] = ts_df['team_name'].apply(self.normalize_team_name)
            ts_merge = ts_df[['team_name_norm', 'season'] + [c for c in req_ts_cols if c not in ['team_name', 'season']]].copy()
            for col in ['wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all']:
                ts_merge[col] = pd.to_numeric(ts_merge.get(col), errors='coerce')
            ts_merge['season'] = ts_merge['season'].astype(str)
            ts_merge['current_form'] = ts_merge.get('current_form','N/A').astype(str).fillna('N/A')
            ts_merge['merge_key'] = ts_merge['team_name_norm'] + "_" + ts_merge['season']
            ts_merge = ts_merge.drop(columns=['team_name_norm', 'season']).drop_duplicates(subset=['merge_key'], keep='last')
            result_df['merge_key_home'] = result_df['home_team_norm'] + "_" + result_df['season']
            result_df['merge_key_away'] = result_df['away_team_norm'] + "_" + result_df['season']
            home_rename = {'wins_all_percentage': 'home_season_win_pct', 'points_for_avg_all': 'home_season_avg_pts_for',
                           'points_against_avg_all': 'home_season_avg_pts_against', 'current_form': 'home_current_form'}
            away_rename = {'wins_all_percentage': 'away_season_win_pct', 'points_for_avg_all': 'away_season_avg_pts_for',
                           'points_against_avg_all': 'away_season_avg_pts_against', 'current_form': 'away_current_form'}
            logger.debug("Attempting to merge season stats using keys...")
            
            # <<< START INSERT - Before Home Season Merge -- COMMENTED OUT ATM>>>
            #if self.debug:
                #logger.debug(f"Season Merge (Home): Left DF shape: {result_df.shape}, Right DF (ts_merge) shape: {ts_merge.shape}")
                #watched_left_season_h = result_df[result_df['home_team_norm'].isin(self._teams_to_watch)]
                #if not watched_left_season_h.empty:
                    #logger.debug(f"[WATCH_TEAM] Season Merge (Home) - Left DF sample (watched teams):\n{watched_left_season_h[['game_id', 'game_date', 'home_team_norm', 'season', 'merge_key_home']].head()}")
                    #watched_keys_season_h = watched_left_season_h['merge_key_home'].unique()
                    # Need to recreate keys for ts_merge if not present or join based on components
                    #ts_merge_watched_h = ts_merge[ts_merge['merge_key'].isin(watched_keys_season_h)]
                    #if not ts_merge_watched_h.empty:
                         #logger.debug(f"[WATCH_TEAM] Season Merge (Home) - Right DF sample (matching watched keys):\n{ts_merge_watched_h.head()}")
                    #else:
                         #logger.debug(f"[WATCH_TEAM] Season Merge (Home) - Right DF: No matching keys found for watched teams.")
                #else:
                     #logger.debug(f"[WATCH_TEAM] Season Merge (Home): No watched teams found in Left DF.")
            # <<< END INSERT >>>
            
            result_df = pd.merge(result_df, ts_merge.rename(columns=home_rename), how='left',
                                 left_on='merge_key_home', right_on='merge_key', indicator='_merge_home')
            result_df = result_df.drop(columns=['merge_key'], errors='ignore')
            
            # <<< START INSERT - Before Away Season Merge -- COMMENTED OUT ATM >>>
            #if self.debug:
                #logger.debug(f"Season Merge (Away): Left DF shape: {result_df.shape}, Right DF (ts_merge) shape: {ts_merge.shape}") # Shape might have changed
                #watched_left_season_a = result_df[result_df['away_team_norm'].isin(self._teams_to_watch)]
                #if not watched_left_season_a.empty:
                    #logger.debug(f"[WATCH_TEAM] Season Merge (Away) - Left DF sample (watched teams):\n{watched_left_season_a[['game_id', 'game_date', 'away_team_norm', 'season', 'merge_key_away']].head()}")
                    #watched_keys_season_a = watched_left_season_a['merge_key_away'].unique()
                    #ts_merge_watched_a = ts_merge[ts_merge['merge_key'].isin(watched_keys_season_a)]
                    #if not ts_merge_watched_a.empty:
                         #logger.debug(f"[WATCH_TEAM] Season Merge (Away) - Right DF sample (matching watched keys):\n{ts_merge_watched_a.head()}")
                    #else:
                         #logger.debug(f"[WATCH_TEAM] Season Merge (Away) - Right DF: No matching keys found for watched teams.")
                #else:
                     #logger.debug(f"[WATCH_TEAM] Season Merge (Away): No watched teams found in Left DF.")
            # <<< END INSERT >>>
            
            result_df = pd.merge(result_df, ts_merge.rename(columns=away_rename), how='left',
                                 left_on='merge_key_away', right_on='merge_key', suffixes=('', '_away_dup'), indicator='_merge_away')
            result_df = result_df.drop(columns=['merge_key'], errors='ignore')
            if '_merge_home' in result_df.columns and '_merge_away' in result_df.columns:
                home_success = (result_df['_merge_home'] == 'both').mean()
                away_success = (result_df['_merge_away'] == 'both').mean()
                logger.info(f"Season stats merge success rate: Home={home_success:.1%}, Away={away_success:.1%}")
                if home_success < 0.9 or away_success < 0.9:
                    logger.warning("Low merge success rate for season stats. Check team name normalization and season formats.")
                result_df = result_df.drop(columns=['_merge_home', '_merge_away'], errors='ignore')
            else:
                logger.warning("Merge indicators not found after merging season stats.")
            result_df = result_df.drop(columns=[c for c in result_df.columns if '_away_dup' in c], errors='ignore')
        except Exception as merge_e:
            logger.error(f"Error during season stats merge process: {merge_e}", exc_info=True)
            for col in placeholder_cols:
                if col not in result_df.columns:
                    result_df[col] = np.nan
        logger.debug("Finalizing season context features (filling NaNs, calculating diffs)...")
        for col in placeholder_cols:
            default_key = col.replace('home_', '').replace('away_', '').replace('season_', '').replace('_diff', '').replace('_net_rating','')
            if 'win_pct' in default_key:
                default_val = self.defaults['win_pct']
            elif 'form' in default_key:
                default_val = 'N/A'
            elif default_key == 'avg_pts_for':
                default_val = self.defaults['avg_pts_for']
            elif default_key == 'avg_pts_against':
                default_val = self.defaults['avg_pts_against']
            elif 'diff' in col:
                default_val = 0.0
            else:
                default_val = self.defaults.get(default_key, 0.0)
            if col not in result_df.columns:
                result_df[col] = default_val
            else:
                result_df[col] = result_df[col].fillna(default_val)
            if 'form' in col:
                result_df[col] = result_df[col].astype(str)
            else:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)
        result_df['season_win_pct_diff'] = result_df['home_season_win_pct'] - result_df['away_season_win_pct']
        result_df['season_pts_for_diff'] = result_df['home_season_avg_pts_for'] - result_df['away_season_avg_pts_for']
        result_df['season_pts_against_diff'] = result_df['home_season_avg_pts_against'] - result_df['away_season_avg_pts_against']
        result_df['home_season_net_rating'] = result_df['home_season_avg_pts_for'] - result_df['home_season_avg_pts_against']
        result_df['away_season_net_rating'] = result_df['away_season_avg_pts_for'] - result_df['away_season_avg_pts_against']
        result_df['season_net_rating_diff'] = result_df['home_season_net_rating'] - result_df['away_season_net_rating']
        result_df = result_df.drop(columns=['season', 'home_team_norm', 'away_team_norm', 'merge_key_home', 'merge_key_away'], errors='ignore')
        
        # <<< FINAL LOGGING BLOCK (Using original 'df' input -- COMMENTED OUT ATM) >>>
        if self.debug:
            try:
                """  # Start block comment
                # Define the final seasonal columns created/populated by this function
                final_season_cols_to_log = ['game_id', 'game_date'] + [c for c in placeholder_cols if c in result_df.columns]
                # Check if necessary columns exist for linking and filtering
                if ('game_id' in result_df.columns and
                    'game_id' in df.columns and
                    'home_team' in df.columns and
                    'away_team' in df.columns):
                    # Merge final results with original team names for filtering watched teams
                    # Select only essential columns to minimize merge overhead
                    log_check_df = pd.merge(
                        result_df[final_season_cols_to_log], # Get final calculated season cols
                        df[['game_id', 'home_team', 'away_team']], # Get original team names from input df
                        on='game_id',
                        how='inner' # Keep only rows present in both (should be all target rows)
                    )
                    # Filter for watched teams using original team names
                    watched_rows_after_season = log_check_df[
                        log_check_df['home_team'].astype(str).apply(self.normalize_team_name).isin(self._teams_to_watch) |
                        log_check_df['away_team'].astype(str).apply(self.normalize_team_name).isin(self._teams_to_watch)
                    ]
                    if not watched_rows_after_season.empty:
                        # Log the final seasonal feature values for the watched teams
                        logger.debug(f"[WATCH_TEAM] Final Season Context Values (watched teams sample): \n{watched_rows_after_season[final_season_cols_to_log].head()}")
                    else:
                        logger.debug("[WATCH_TEAM] No watched teams found in final result_df for season context logging.")
                else:
                    logger.warning("[WATCH_TEAM] Could not perform final season context logging: Key columns missing in result_df or original df (game_id, home_team, away_team).")
                """ # End block comment
            except Exception as log_err:
                 logger.warning(f"[WATCH_TEAM] Error during final season context logging: {log_err}", exc_info=True) # Keep this active maybe?
        # <<< END OF CORRECTED FINAL LOGGING BLOCK >>>

        logger.info("Finished adding season context features.")
        return result_df

    @profile_time
    def add_form_string_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates features based on team form strings (e.g., 'WWLWL')."""
        logger.debug("Adding form string derived features...")
        result_df = df.copy()
        form_metric_keys = list(self._extract_form_metrics_single("").keys())
        home_cols = [f'home_{k}' for k in form_metric_keys]
        away_cols = [f'away_{k}' for k in form_metric_keys]
        diff_cols = ['form_win_pct_diff', 'streak_advantage', 'momentum_diff']
        placeholder_cols = home_cols + away_cols + diff_cols
        home_form_col = 'home_current_form'
        away_form_col = 'away_current_form'
        if home_form_col not in result_df.columns or away_form_col not in result_df.columns:
            logger.warning(f"Missing one or both form columns ('{home_form_col}', '{away_form_col}'). Adding placeholders.")
        else:
            try:
                result_df[home_form_col] = result_df[home_form_col].fillna('').astype(str).replace('N/A', '')
                result_df[away_form_col] = result_df[away_form_col].fillna('').astype(str).replace('N/A', '')
                home_metrics = result_df[home_form_col].apply(self._extract_form_metrics_single)
                away_metrics = result_df[away_form_col].apply(self._extract_form_metrics_single)
                result_df = result_df.join(pd.DataFrame(home_metrics.tolist(), index=result_df.index).add_prefix('home_'))
                result_df = result_df.join(pd.DataFrame(away_metrics.tolist(), index=result_df.index).add_prefix('away_'))
                result_df['form_win_pct_diff'] = (result_df.get('home_form_win_pct', self.defaults['form_win_pct']) -
                                                  result_df.get('away_form_win_pct', self.defaults['form_win_pct']))
                result_df['streak_advantage'] = (result_df.get('home_current_streak', self.defaults['current_streak']) -
                                                 result_df.get('away_current_streak', self.defaults['current_streak']))
                result_df['momentum_diff'] = (result_df.get('home_momentum_direction', self.defaults['momentum_direction']) -
                                              result_df.get('away_momentum_direction', self.defaults['momentum_direction']))
                logger.debug("Successfully calculated form metrics and diffs.")
            except Exception as e:
                logger.error(f"Error processing form strings: {e}", exc_info=True)

         # <<< START INSERT - After Form Calc -- COMMENTED OUT ATM >>>
        if self.debug:
            # Check original team names before they might be dropped if needed
            # Assuming 'home_team', 'away_team' still exist or use normalized names if available
            home_team_col_check = 'home_team' if 'home_team' in result_df.columns else 'home_team_norm' # Adjust if names change
            away_team_col_check = 'away_team' if 'away_team' in result_df.columns else 'away_team_norm'

            #watched_rows_after_form = result_df[
                # result_df[home_team_col_check].apply(self.normalize_team_name).isin(self._teams_to_watch) |
                 #result_df[away_team_col_check].apply(self.normalize_team_name).isin(self._teams_to_watch)
            # ]
            #form_cols_to_log = ['game_id', 'game_date'] + [c for c in placeholder_cols if c in result_df.columns]
            #if not watched_rows_after_form.empty:
                 # logger.debug(f"[WATCH_TEAM] Final Form String Values (watched teams sample): \n{watched_rows_after_form[form_cols_to_log].head()}")
        # <<< END INSERT >>>
        logger.debug("Finalizing form string features (filling defaults/types)...")
        for col in placeholder_cols:
            default_key = col.replace('home_', '').replace('away_', '').replace('_diff', '').replace('_advantage', '')
            default_val = self.defaults.get(default_key, 0.0)
            if 'win_pct' in default_key:
                default_val = self.defaults['form_win_pct']
            if col not in result_df.columns:
                result_df[col] = default_val
            else:
                result_df[col] = result_df[col].fillna(default_val)
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)
            if 'streak' in col and 'advantage' not in col:
                result_df[col] = result_df[col].round().astype(int)
        logger.debug("Finished adding form string features.")
        return result_df

    def _get_matchup_history_single(self, home_team_norm: str, away_team_norm: str, historical_subset: pd.DataFrame,
                                max_games: int = 5, current_game_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """Calculates H2H stats for a single matchup, perspective of home_team_norm."""
        # <<< Add Check for Debug Level >>>
        is_debug = _is_debug_enabled(logger)

        #if is_debug COMMENTED:
            #logger.debug(f"H2H Helper: Called for Home='{home_team_norm}', Away='{away_team_norm}', Date='{current_game_date}', MaxGames={max_games}. History subset shape: {historical_subset.shape}")

        default_result = {
            'matchup_num_games': self.defaults['matchup_num_games'],
            'matchup_avg_point_diff': self.defaults['matchup_avg_point_diff'],
            'matchup_home_win_pct': self.defaults.get('matchup_home_win_pct', 0.5),
            'matchup_avg_total_score': self.defaults['matchup_avg_total_score'],
            'matchup_avg_home_score': self.defaults['matchup_avg_home_score'],
            'matchup_avg_away_score': self.defaults['matchup_avg_away_score'],
            'matchup_last_date': pd.NaT,
            'matchup_streak': self.defaults['matchup_streak']
        }

        if historical_subset.empty or max_games <= 0 or pd.isna(current_game_date):
            #if is_debug: logger.debug(f"H2H Helper: Returning default due to empty history ({historical_subset.empty}), max_games<=0 ({max_games<=0}), or invalid date ({pd.isna(current_game_date)}).")
            return default_result

        # Filter for games strictly before the current date
        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        #if is_debug: logger.debug(f"H2H Helper: Filtered to {len(past_games_df)} games before {current_game_date}.")

        if past_games_df.empty:
            #if is_debug: logger.debug("H2H Helper: Returning default as no games found before current date.")
            return default_result

        try:
            # Get the N most recent games from the filtered past games
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            #if is_debug: logger.debug(f"H2H Helper: Selected {len(recent_matchups)} most recent matchups for calculation.")

            if recent_matchups.empty: # Should not happen if past_games_df wasn't empty, but check
                #if is_debug: logger.debug("H2H Helper: Returning default as recent_matchups is unexpectedly empty.")
                return default_result

            # Ensure scores are numeric (should be pre-processed, but double-check)
            recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce')
            recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
            recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score'])

            if recent_matchups.empty:
                #if is_debug: logger.debug("H2H Helper: Returning default as no valid scores found in recent matchups.")
                return default_result

            # Calculate stats - loop through the selected recent games *chronologically* for streak
            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []
            home_persp_wins = 0
            current_streak = 0
            last_winner_norm = None # Tracks winner for streak calc (home_team_norm or away_team_norm)

            # Iterate from oldest to newest among the selected recent games
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
                h_score = game['home_score']
                a_score = game['away_score']
                g_home_norm = game.get('home_team_norm', 'Hist_Unknown_Home')
                g_away_norm = game.get('away_team_norm', 'Hist_Unknown_Away')

                #if is_debug: logger.debug(f"H2H Helper Iter: Date={game['game_date'].date()}, GameHome={g_home_norm}, GameAway={g_away_norm}, HScore={h_score}, AScore={a_score}")

                # Determine diff and winner from perspective of the *target* home_team_norm
                if g_home_norm == home_team_norm: # Target home team played at home in this historical game
                    diff = h_score - a_score
                    won = h_score > a_score
                    h_persp_score = h_score
                    a_persp_score = a_score
                elif g_away_norm == home_team_norm: # Target home team played away
                    diff = a_score - h_score
                    won = a_score > h_score
                    h_persp_score = a_score # Score FOR the target team
                    a_persp_score = h_score # Score AGAINST the target team
                else:
                    # This case should ideally not happen if historical_subset was correctly filtered by matchup_key
                    logger.warning(f"H2H Helper: Mismatch in game teams vs target teams! Game: {g_home_norm} vs {g_away_norm}, Target Home: {home_team_norm}. Skipping game.")
                    continue

                diffs.append(diff)
                total_scores.append(h_score + a_score)
                home_persp_scores.append(h_persp_score)
                away_persp_scores.append(a_persp_score)

                game_winner_norm = home_team_norm if won else away_team_norm
                #if is_debug: logger.debug(f"H2H Helper Iter: Perspective={home_team_norm}, Diff={diff:.1f}, Won={won}, Winner={game_winner_norm}")

                if won: home_persp_wins += 1

                # Calculate streak (relative to the target home team)
                if last_winner_norm is None: # First game in streak calculation
                    current_streak = 1 if won else -1
                elif game_winner_norm == last_winner_norm: # Streak continues
                    current_streak += (1 if won else -1)
                else: # Streak broken
                    current_streak = 1 if won else -1
                last_winner_norm = game_winner_norm # Update for next iteration

                #if is_debug: logger.debug(f"H2H Helper Iter: Streak after game: {current_streak}")


            num_games = len(diffs)
            if num_games == 0:
                #if is_debug: logger.debug("H2H Helper: Returning default as no valid games processed in loop.")
                return default_result

            # Assemble final results
            final_stats = {
                'matchup_num_games': num_games,
                'matchup_avg_point_diff': np.mean(diffs) if diffs else 0.0,
                'matchup_home_win_pct': home_persp_wins / num_games if num_games > 0 else 0.5,
                'matchup_avg_total_score': np.mean(total_scores) if total_scores else 0.0,
                'matchup_avg_home_score': np.mean(home_persp_scores) if home_persp_scores else 0.0,
                'matchup_avg_away_score': np.mean(away_persp_scores) if away_persp_scores else 0.0,
                'matchup_last_date': recent_matchups['game_date'].max(), # Date of most recent game in H2H history
                'matchup_streak': int(current_streak) # Final streak value
            }
            #if is_debug: logger.debug(f"H2H Helper: Calculated final stats: {final_stats}")

            # Ensure all keys from default_result are present
            for k, v in default_result.items():
                final_stats.setdefault(k, v)
                if pd.isna(final_stats[k]): final_stats[k] = v # Fill NaNs just in case

            
            if is_debug and num_games > 0: # Only log if we actually calculated something
                logger.debug(f"H2H Helper: Calculated final stats for {home_team_norm} vs {away_team_norm} on {current_game_date}: {final_stats}")

            return final_stats

        except Exception as e:
            logger.error(f"H2H Helper: Error calculating for {home_team_norm} vs {away_team_norm} on {current_game_date}: {e}", exc_info=True)
            return default_result # Return defaults on error

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        """Extracts metrics (win %, streak, momentum) from a form string like 'WWLWL'."""
        # <<< Add Check for Debug Level >>>
        is_debug = _is_debug_enabled(logger)

        #if is_debug: logger.debug(f"Form Helper: Called with string='{form_string}'")

        defaults = {
            'form_win_pct': self.defaults['form_win_pct'],
            'current_streak': self.defaults['current_streak'],
            'momentum_direction': self.defaults['momentum_direction']
        }

        if not form_string or pd.isna(form_string) or not isinstance(form_string, str):
            #if is_debug: logger.debug("Form Helper: Returning default due to invalid/empty input string.")
            return defaults

        # Clean the string
        form_string = form_string.upper().strip().replace('-', '').replace('?', '')
        form_len = len(form_string)

        if form_len == 0 or form_string == 'N/A':
            #if is_debug: logger.debug("Form Helper: Returning default due to empty/NA string after cleaning.")
            return defaults

        # Calculate Win Percentage
        wins = form_string.count('W')
        form_win_pct = wins / form_len
        #if is_debug: logger.debug(f"Form Helper: Wins={wins}, Len={form_len}, WinPct={form_win_pct:.3f}")

        # Calculate Current Streak
        current_streak = 0
        if form_len > 0:
            streak_char = form_string[-1] # Last game result determines streak type (W or L)
            streak_count = 0
            for char in reversed(form_string): # Count consecutive same results from the end
                if char == streak_char:
                    streak_count += 1
                else:
                    break
            current_streak = streak_count if streak_char == 'W' else -streak_count
        #if is_debug: logger.debug(f"Form Helper: Calculated Streak={current_streak}")


        # Calculate Momentum Direction (comparing recent half vs older half)
        momentum_direction = 0.0
        if form_len >= 4: # Need at least 4 games for a meaningful split
            split_point = form_len // 2 # Integer division
            # Ensure correct slicing
            recent_half_str = form_string[-split_point:]
            older_half_str = form_string[:form_len - split_point] # Handles odd lengths correctly
            len_r, len_o = len(recent_half_str), len(older_half_str)

            #if is_debug: logger.debug(f"Form Helper: Momentum check - Older='{older_half_str}' (len={len_o}), Recent='{recent_half_str}' (len={len_r})")

            if len_r > 0 and len_o > 0: # Ensure both halves exist
                wins_r = recent_half_str.count('W')
                wins_o = older_half_str.count('W')
                pct_r = wins_r / len_r
                pct_o = wins_o / len_o
                #if is_debug: logger.debug(f"Form Helper: Momentum check - OlderPct={pct_o:.3f}, RecentPct={pct_r:.3f}")

                if pct_r > pct_o: momentum_direction = 1.0 # Improving form
                elif pct_r < pct_o: momentum_direction = -1.0 # Declining form
        #if is_debug: logger.debug(f"Form Helper: Calculated MomentumDir={momentum_direction}")


        final_metrics = {
            'form_win_pct': form_win_pct,
            'current_streak': int(current_streak),
            'momentum_direction': momentum_direction
        }
        #if is_debug: logger.debug(f"Form Helper: Returning metrics: {final_metrics}")

        return final_metrics

    @profile_time
    def generate_all_features(self,
                              df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None,
                              rolling_windows: List[int] = [5, 10, 20],
                              h2h_window: int = 7
                              ) -> pd.DataFrame:
        """Applies all feature engineering steps in sequence."""
        logger.info("Starting comprehensive feature generation pipeline...")
        start_time_total = time.time()

        # --- Input Validation & Prep ---
        if df is None or df.empty:
            logger.error("Input 'df' (target games) is empty. Cannot generate features.")
            return pd.DataFrame()
        essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
        if not all(c in df.columns for c in essential_cols):
            missing_essentials = set(essential_cols) - set(df.columns)
            logger.error(f"Input 'df' missing essential columns: {missing_essentials}. Cannot generate features.")
            return pd.DataFrame()

        target_games_df = df.copy()
        try:
            target_games_df['game_date'] = pd.to_datetime(target_games_df['game_date'], errors='coerce').dt.tz_localize(None)
            target_games_df = target_games_df.dropna(subset=['game_date'])
            # Ensure game_id is string early
            if 'game_id' in target_games_df.columns:
                 target_games_df['game_id'] = target_games_df['game_id'].astype(str)
        except Exception as e:
            logger.error(f"Error processing 'game_date' or 'game_id' in target df: {e}")
            return pd.DataFrame()
        if target_games_df.empty:
            logger.error("No valid target games remaining after date processing.")
            return pd.DataFrame()
        logger.info(f"Target games date range: {target_games_df['game_date'].min().date()} to {target_games_df['game_date'].max().date()}")

        hist_df_processed = None
        if historical_games_df is not None and not historical_games_df.empty:
            logger.debug(f"Processing {len(historical_games_df)} historical games...")
            try:
                hist_df_processed = historical_games_df.copy()
                hist_df_processed['game_date'] = pd.to_datetime(hist_df_processed['game_date'], errors='coerce').dt.tz_localize(None)
                hist_df_processed = hist_df_processed.dropna(subset=['game_date'])
                # Ensure game_id is string early
                if 'game_id' in hist_df_processed.columns:
                    hist_df_processed['game_id'] = hist_df_processed['game_id'].astype(str)
                logger.debug(f"{len(hist_df_processed)} historical games remaining after date processing.")
            except Exception as e:
                logger.error(f"Error processing 'game_date' or 'game_id' in historical_games_df: {e}. Proceeding without historical.")
                hist_df_processed = None
        # --- Determine base_calc_df START ---
        base_calc_df = None
        if hist_df_processed is not None and not hist_df_processed.empty:
            # Check if inputs are effectively the same dataset
            df_ids = set(target_games_df['game_id'].unique())
            hist_ids = set(hist_df_processed['game_id'].unique())

            if df_ids == hist_ids: # Scenario 1: Inputs are identical
                logger.info("Using the provided historical data directly as the base (inputs appear identical).")
                base_calc_df = hist_df_processed.copy().sort_values(['game_date', 'game_id'], kind='mergesort').reset_index(drop=True)
                # game_id already string
                logger.info(f"Created base calculation DataFrame with {len(base_calc_df)} unique games.")
            else: # Scenario 2: History and Target are distinct (and History exists)
                try:
                    logger.info(f"Combining {len(hist_df_processed)} processed historical and {len(target_games_df)} target games...")
                    # Align columns before concat
                    hist_cols = set(hist_df_processed.columns); target_cols = set(target_games_df.columns); all_cols_union = list(hist_cols.union(target_cols))
                    for col in all_cols_union:
                        if col not in hist_df_processed.columns: hist_df_processed[col] = np.nan
                        if col not in target_games_df.columns: target_games_df[col] = np.nan

                    base_calc_df = pd.concat([hist_df_processed[all_cols_union], target_games_df[all_cols_union]], ignore_index=True)\
                                    .sort_values(['game_date','game_id'], kind='mergesort')
                    initial_rows = len(base_calc_df)
                    # game_id already string
                    base_calc_df = base_calc_df.drop_duplicates('game_id', keep='last') # Keep dedup here
                    rows_dropped = initial_rows - len(base_calc_df)
                    if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} duplicate game_id rows during combination.")
                    logger.info(f"Created base calculation DataFrame with {len(base_calc_df)} unique games.")
                except Exception as e:
                    logger.error(f"Error combining distinct historical and target data: {e}", exc_info=True)
                    logger.warning("Falling back to using only target games data."); base_calc_df = target_games_df.copy().sort_values(['game_date','game_id']).reset_index(drop=True)
                    # game_id already string in target_games_df
        else: # Scenario 3: Only Target exists (no History)
            logger.warning("No historical data provided or processed. Using only target games data.")
            base_calc_df = target_games_df.copy().sort_values(['game_date','game_id']).reset_index(drop=True)
            # game_id already string

        if base_calc_df is None or base_calc_df.empty:
            logger.error("Base DataFrame for feature calculation is empty. Cannot proceed.")
            return pd.DataFrame()
        # --- Determine base_calc_df END ---


        # --- Feature Generation Sequence ---
        try:
            # --- Add Checks After Each Step ---
            logger.info("Step 1/8: Adding intra-game momentum features...")
            base_calc_df = self.add_intra_game_momentum(base_calc_df)
            logger.debug(f"Shape after momentum: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 2/8: Integrating advanced metrics...")
            base_calc_df = self.integrate_advanced_features(base_calc_df)
            logger.debug(f"Shape after advanced: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 3/8: Adding rolling features...")
            base_calc_df = self.add_rolling_features(base_calc_df, window_sizes=rolling_windows)
            logger.debug(f"Shape after rolling: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 4/8: Adding rest & schedule density features...")
            base_calc_df = self.add_rest_features_vectorized(base_calc_df)
            logger.debug(f"Shape after rest: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 5/8: Adding head-to-head matchup features...")
            # Pass the *original processed* historical df for lookup if it exists
            hist_lookup_df = hist_df_processed if hist_df_processed is not None else pd.DataFrame()
            base_calc_df = self.add_matchup_history_features(base_calc_df, hist_lookup_df, max_games=h2h_window)
            logger.debug(f"Shape after H2H: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 6/8: Adding season context features...")
            base_calc_df = self.add_season_context_features(base_calc_df, team_stats_df)
            logger.debug(f"Shape after season: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")

            logger.info("Step 7/8: Adding form string features...")
            base_calc_df = self.add_form_string_features(base_calc_df)
            logger.debug(f"Shape after form: {base_calc_df.shape}, Unique game_ids: {base_calc_df['game_id'].nunique()}")
        
            # --- End Checks ---

        except Exception as e:
            logger.error(f"Error during feature generation pipeline step: {e}", exc_info=True)
            return pd.DataFrame()

        # --- Final Filtering ---
        logger.info("Filtering results back to target games...")
        try:
            original_game_ids = df['game_id'].astype(str).unique() # Use the original input df game_ids
            num_unique_original_ids = len(original_game_ids)
            logger.debug(f"Filtering for {num_unique_original_ids} unique target game IDs...")
            if 'game_id' not in base_calc_df.columns:
                raise ValueError("'game_id' column missing after feature generation.")
            # base_calc_df['game_id'] is already string
            final_df = base_calc_df[base_calc_df['game_id'].isin(original_game_ids)].copy()
            logger.info(f"Shape of final DataFrame after filtering: {final_df.shape}")
            if len(final_df) != num_unique_original_ids:
                 # This could happen if some target games had issues during feature gen (e.g., missing critical data)
                logger.warning(f"Final Filtering Count Mismatch! Expected: {num_unique_original_ids}, Got: {len(final_df)}. Some target games might have been dropped due to processing errors.")
                # Decide if this is critical - returning empty might be too harsh if some rows are okay
                # For now, let's return the rows we have, but log the warning strongly.
                # return pd.DataFrame() # Option: return empty if mismatch is unacceptable
            else:
                logger.info("Final filtering successful, row count matches target game count.")
        except Exception as filter_e:
            logger.error(f"Error during final filtering step: {filter_e}", exc_info=True)
            return pd.DataFrame()

        total_time = time.time() - start_time_total
        logger.info(f"Feature generation pipeline complete for {len(final_df)} games in {total_time:.2f}s.")
        return final_df.reset_index(drop=True) # Return the filtered df


# -- Example Usage (Keep empty if run externally) --
if __name__ == '__main__':
    logger.info("NBAFeatureEngine script executed directly (usually imported).")
    # Example usage:
    engine = NBAFeatureEngine(debug=True)
    # dummy_df = pd.DataFrame(...) # Create dummy data
    # features = engine.generate_all_features(dummy_df)
    # print(features.info())
    pass
