# feature_engineering.py - Unified & Integrated Feature Engineering

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
    
    @lru_cache(maxsize=512)
    def normalize_team_name(self, team_name: Optional[str]) -> str:
        """Normalize team names using a predefined mapping."""
        if not isinstance(team_name, str):
            return "Unknown"
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
            "la": "lakers",  # Common abbreviation
            # Special cases
            "east": "east", "west": "west", "team lebron": "other_team", "team durant": "other_team",
            "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
            "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
        }
        if team_lower in mapping:
            return mapping[team_lower]
        for name, norm in mapping.items():
            if len(team_lower) > 3 and team_lower in name:
                return norm
            if len(name) > 3 and name in team_lower:
                return norm
        logger.warning(f"Team name '{team_name}' normalized to '{team_lower}' - no mapping found!")
        return team_lower

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
            logger.warning("Input df empty for rolling features.")
            return df
        local_df = df.copy()
        cols_to_roll_generic = [
            'score_for', 'score_against', 'point_diff',
            'momentum_ewma',
            'off_rating', 'def_rating', 'net_rating',
            'pace', 'efg_pct', 'tov_rate', 'trb_pct', 'oreb_pct', 'dreb_pct', 'ft_rate'
        ]
        source_mapping_home = {
            'home_score': 'score_for',
            'away_score': 'score_against',
            'point_diff': 'point_diff', # Assumes point_diff = home_score - away_score
            'momentum_score_ewma_q4': 'momentum_ewma', # Example source name
            'home_offensive_rating': 'off_rating',
            'home_defensive_rating': 'def_rating',
            'home_net_rating': 'net_rating',
            'game_pace': 'pace', # Assuming pace is game-level, used for both
            'home_efg_pct': 'efg_pct',
            'home_tov_rate': 'tov_rate',
            'home_trb_pct': 'trb_pct',
            'home_oreb_pct': 'oreb_pct',
            'home_dreb_pct': 'dreb_pct',
            'home_ft_rate': 'ft_rate'
        }
        source_mapping_away = {
            'away_score': 'score_for',
            'home_score': 'score_against',
            'point_diff': 'point_diff', # Uses same point_diff, but sign flips perspective
            'momentum_score_ewma_q4': 'momentum_ewma', # Example source name
            'away_offensive_rating': 'off_rating',
            'away_defensive_rating': 'def_rating',
            'away_net_rating': 'net_rating',
            'game_pace': 'pace', # Assuming pace is game-level, used for both
            'away_efg_pct': 'efg_pct',
            'away_tov_rate': 'tov_rate',
            'away_trb_pct': 'trb_pct',
            'away_oreb_pct': 'oreb_pct',
            'away_dreb_pct': 'dreb_pct',
            'away_ft_rate': 'ft_rate'
        }
        # Special handling for columns calculated from others or game-level
        if 'point_diff' not in local_df.columns and 'home_score' in local_df.columns and 'away_score' in local_df.columns:
             local_df['point_diff'] = pd.to_numeric(local_df['home_score'], errors='coerce') - pd.to_numeric(local_df['away_score'], errors='coerce')
        if 'game_pace' not in local_df.columns:
            # Add logic to calculate or fetch pace if missing
            logger.warning("Rolling: 'game_pace' column missing. Pace features may be inaccurate.")
            local_df['game_pace'] = np.nan # Or a default value


        required_sources = set(source_mapping_home.keys()) | set(source_mapping_away.keys()) | {'game_id', 'game_date', 'home_team', 'away_team'}
        missing_sources = [col for col in required_sources if col not in local_df.columns]

        if missing_sources:
            logger.warning(f"Rolling: Missing source columns needed for rolling calculations: {missing_sources}. Rolling features may be inaccurate or defaulted.")
            # Handle potentially missing calculated cols explicitly if needed
            for col in missing_sources:
                if col not in local_df.columns:
                    local_df[col] = np.nan # Assign NaN if truly missing

        cols_to_roll_final = []
        for generic_col in cols_to_roll_generic:
            home_source = next((k for k, v in source_mapping_home.items() if v == generic_col), None)
            away_source = next((k for k, v in source_mapping_away.items() if v == generic_col), None)

            # Check if sources exist (even if they were initially missing and now NaN)
            home_source_exists = home_source in local_df.columns if home_source else False
            away_source_exists = away_source in local_df.columns if away_source else False

            # Special handling for point_diff and game_pace (may exist but not be in both mappings directly)
            if generic_col == 'point_diff' and 'point_diff' in local_df.columns:
                 home_source_exists = True
                 away_source_exists = True # Requires careful handling later if using raw scores
                 home_source = 'point_diff' # Use the calculated/existing column
                 away_source = 'point_diff' # Use the calculated/existing column, perspective handled later
            if generic_col == 'pace' and 'game_pace' in local_df.columns:
                 home_source_exists = True
                 away_source_exists = True
                 home_source = 'game_pace'
                 away_source = 'game_pace'

            if home_source_exists and away_source_exists:
                cols_to_roll_final.append(generic_col)
                # Convert to numeric safely, applying defaults from self.defaults
                if home_source:
                   local_df[home_source] = pd.to_numeric(local_df[home_source], errors='coerce').fillna(self.defaults.get(generic_col, 0.0))
                if away_source and away_source != home_source: # Avoid converting same column twice
                   local_df[away_source] = pd.to_numeric(local_df[away_source], errors='coerce').fillna(self.defaults.get(generic_col, 0.0))
            else:
                logger.warning(f"Cannot calculate rolling features for '{generic_col}' due to missing source columns: Home needed='{home_source}', Away needed='{away_source}'.")

        if not cols_to_roll_final:
            logger.error("Rolling: No valid columns found to calculate rolling features.")
            return df

        logger.debug(f"Rolling: Creating team-centric view for columns: {cols_to_roll_final}")
        try:
            local_df['home_team_norm'] = local_df['home_team'].apply(self.normalize_team_name)
            local_df['away_team_norm'] = local_df['away_team'].apply(self.normalize_team_name)

            home_data_list = []
            away_data_list = []

            base_cols = ['game_id', 'game_date']
            home_view_base = local_df[base_cols + ['home_team_norm']].rename(columns={'home_team_norm': 'team_norm'})
            away_view_base = local_df[base_cols + ['away_team_norm']].rename(columns={'away_team_norm': 'team_norm'})

            for generic_col in cols_to_roll_final:
                home_source = next((k for k, v in source_mapping_home.items() if v == generic_col), None)
                away_source = next((k for k, v in source_mapping_away.items() if v == generic_col), None)

                # Get the correct series for home and away perspectives
                if home_source and home_source in local_df.columns:
                    home_series = local_df[home_source].rename(generic_col)
                    # Handle point_diff perspective for home team (home - away)
                    if generic_col == 'point_diff' and home_source == 'point_diff':
                         pass # Already correct perspective
                    home_data_list.append(home_series)

                if away_source and away_source in local_df.columns:
                    away_series = local_df[away_source].rename(generic_col)
                    # Handle point_diff perspective for away team (needs sign flip: away - home = -point_diff)
                    if generic_col == 'point_diff' and away_source == 'point_diff':
                        away_series = -away_series # Flip sign for away team perspective
                    # Handle score_against perspective for away team (which uses home_score)
                    elif generic_col == 'score_against' and away_source == 'home_score':
                         pass # Already correct source mapped
                    # Handle other cases if necessary
                    away_data_list.append(away_series)

            home_view = pd.concat([home_view_base] + home_data_list, axis=1)
            away_view = pd.concat([away_view_base] + away_data_list, axis=1)

            team_view = (
                pd.concat([home_view, away_view], ignore_index=True)
                .assign(game_date=lambda x: pd.to_datetime(x['game_date']))
                .sort_values(['team_norm', 'game_date', 'game_id'], kind='mergesort')
                .reset_index(drop=True)
            )
        except Exception as e:
            logger.error(f"Rolling: Error creating team view: {e}", exc_info=True)
            return df

        logger.debug(f"Rolling: Calculating shifted mean/std for windows {window_sizes}...")
        rolling_cols_generated = []
        for window in window_sizes:
            min_p = max(1, window // 2) # Ensure min_periods is at least 1
            for col in cols_to_roll_final:
                # Check if column exists in team_view (it should, based on previous logic)
                if col not in team_view.columns:
                    logger.warning(f"Rolling: Column '{col}' unexpectedly missing from team_view. Skipping rolling calculation for this column.")
                    continue

                roll_mean_col = generate_rolling_column_name('', col, 'mean', window)
                roll_std_col = generate_rolling_column_name('', col, 'std', window)
                rolling_cols_generated.extend([roll_mean_col, roll_std_col])

                grouped_col = team_view.groupby('team_norm', observed=True)[col]
                # Shift by 1 to prevent data leakage from the current game
                shifted_data = grouped_col.shift(1)
                # Calculate rolling features on the shifted data
                rolling_op = shifted_data.rolling(window=window, min_periods=min_p)

                # Use defaults specific to the column
                default_mean = self.defaults.get(col, 0.0)
                default_std = self.defaults.get(f'{col}_std', 0.0) # Check for specific std default first

                # Calculate mean and std, fill NaNs resulting *from the rolling operation*
                team_view[roll_mean_col] = rolling_op.mean()
                std_dev = rolling_op.std()
                team_view[roll_std_col] = np.maximum(0, std_dev) # Ensure std is non-negative

                # Fill NaNs that existed *before* rolling (due to shift/min_periods) or *after* if default needed
                team_view[roll_mean_col] = team_view[roll_mean_col].fillna(default_mean)
                team_view[roll_std_col] = team_view[roll_std_col].fillna(default_std)


        logger.debug("Rolling: Merging rolling stats back to the original DataFrame...")
        try:
            # Build merge keys more robustly
            team_view['merge_key_rolling'] = (
                team_view['game_id'].astype(str) + "_" + team_view['team_norm'].astype(str)
            )
            local_df['merge_key_home'] = (
                local_df['game_id'].astype(str) + "_" + local_df['home_team_norm'].astype(str)
            )
            local_df['merge_key_away'] = (
                local_df['game_id'].astype(str) + "_" + local_df['away_team_norm'].astype(str)
            )

            # Prepare data to merge - ensure only valid generated columns are selected
            cols_to_merge = ['merge_key_rolling'] + [
                c for c in rolling_cols_generated if c in team_view.columns
            ]
            # Use drop_duplicates AFTER selecting columns to avoid keeping unnecessary ones
            merge_data = (
                team_view[cols_to_merge]
                .drop_duplicates(subset=['merge_key_rolling'], keep='last')
            )

            # --- Dictionary Comprehensions for Renaming ---
            home_rename_dict = {}
            away_rename_dict = {}
            parsed_bases = set() # Keep track of base columns parsed

            for col in merge_data.columns:
                if col == 'merge_key_rolling':
                    continue
                try:
                    # Assuming generate_rolling_column_name uses underscores consistently
                    parts = col.split('_') # e.g., ['rolling', 'score', 'for', 'mean', '5']
                    if len(parts) >= 4 and parts[0] == 'rolling':
                        window = int(parts[-1])
                        stat_type = parts[-2]
                        base = "_".join(parts[1:-2]) # Reconstruct base (e.g., 'score_for')
                        parsed_bases.add(base) # Track the base column name

                        home_rename_dict[col] = generate_rolling_column_name(
                            'home', base, stat_type, window
                        )
                        away_rename_dict[col] = generate_rolling_column_name(
                            'away', base, stat_type, window
                        )
                    else:
                        logger.warning(
                            f"Rolling: Could not parse rolling column '{col}' during rename. Skipping."
                        )
                except (ValueError, IndexError) as parse_err:
                    logger.warning(
                        f"Rolling: Error parsing column '{col}' for rename: {parse_err}. Skipping."
                    )

            # --- Merge Logic ---
            # Merge for Home team
            local_df = pd.merge(
                local_df,
                merge_data,
                how='left',
                left_on='merge_key_home',
                right_on='merge_key_rolling'
            ).rename(columns=home_rename_dict)
             # Drop the key from the right dataframe immediately after merge
            if 'merge_key_rolling' in local_df.columns:
                 local_df = local_df.drop(columns=['merge_key_rolling'])

            # Merge for Away team
            local_df = pd.merge(
                local_df,
                merge_data, # Use the same merge_data prepared earlier
                how='left',
                left_on='merge_key_away',
                right_on='merge_key_rolling'
            ).rename(columns=away_rename_dict)
            # Drop the key from the right dataframe
            if 'merge_key_rolling' in local_df.columns:
                 local_df = local_df.drop(columns=['merge_key_rolling'])
            # --- End Merge Logic ---

        except Exception as e:
            logger.error(f"Rolling: Error merging rolling stats back: {e}", exc_info=True)
            # Fallback: Add columns with defaults if merge failed
            missing_cols_defaults = {}
            # Ensure we use the base columns derived *during* the rename attempt
            # or fall back to cols_to_roll_final if rename dicts are empty
            bases_for_fallback = parsed_bases if parsed_bases else cols_to_roll_final

            for prefix in ['home', 'away']:
                for base in bases_for_fallback:
                    for stat in ['mean', 'std']:
                        for w in window_sizes:
                            col_name = generate_rolling_column_name(prefix, base, stat, w)
                            if col_name not in local_df.columns:
                                default_key = f'{base}_std' if stat == 'std' else base
                                missing_cols_defaults[col_name] = self.defaults.get(default_key, 0.0)

            if missing_cols_defaults:
                logger.warning(f"Adding default values for {len(missing_cols_defaults)} missing rolling columns due to merge error.")
                local_df = local_df.assign(**missing_cols_defaults)


        logger.debug("Rolling: Filling final NaNs and calculating rolling differentials...")
        primary_window = max(window_sizes) if window_sizes else 10 # Define primary window
        new_diff_cols = {} # Dictionary to collect new differential columns

        for stat_type in ['mean', 'std']:
            # Use cols_to_roll_final as the definitive list of stats processed
            for base_col in cols_to_roll_final:
                # Fill NaNs for existing rolling columns first
                for w in window_sizes:
                    for prefix in ['home', 'away']:
                        col_name = generate_rolling_column_name(prefix, base_col, stat_type, w)
                        default_key = f'{base_col}_std' if stat_type == 'std' else base_col
                        default = self.defaults.get(default_key, 0.0)
                        if col_name not in local_df.columns:
                             # Should be rare after merge/fallback, but handle defensively
                            logger.warning(f"Rolling column '{col_name}' missing before diff calc. Adding default.")
                            local_df[col_name] = default
                        else:
                            # Fill NaNs that might still exist (e.g., from original data issues)
                            local_df[col_name] = local_df[col_name].fillna(default)

                        # Ensure std dev is non-negative after filling NaNs
                        if stat_type == 'std':
                            local_df[col_name] = np.maximum(0, local_df[col_name])

                # --- Calculate differential using primary window ---
                w = primary_window
                home_col = generate_rolling_column_name('home', base_col, stat_type, w)
                away_col = generate_rolling_column_name('away', base_col, stat_type, w)

                # *** START INSERTED CODE ***
                # Map generic base_col name to the expected name for the differential column
                # *** START FIXED CODE ***
                # Map generic base_col name to the expected name for the differential column
                expected_base_name = base_col  # Default mapping
                if base_col == 'point_diff':
                    expected_base_name = 'margin'
                elif base_col == 'net_rating':
                    expected_base_name = 'eff'  # Short for efficiency difference
                elif base_col == 'dreb_pct':
                    expected_base_name = 'dreb'  # Remove pct
                elif base_col == 'efg_pct':
                    expected_base_name = 'efg'   # Remove pct
                elif base_col == 'trb_pct':
                    expected_base_name = 'trb'   # Remove pct
                elif base_col == 'momentum_ewma':
                    expected_base_name = 'momentum'  # Remove ewma

                # Always include the '_diff' suffix for consistency with base model expectations
                diff_col_name = f'rolling_{expected_base_name}_diff_{stat_type}'
                # *** END FIXED CODE ***



                # Check if necessary source columns for diff exist after merging/filling
                if home_col in local_df.columns and away_col in local_df.columns:
                    home_vals = local_df[home_col]
                    away_vals = local_df[away_col]

                    # Apply division logic: For 'lower is better' stats (tov, def_rating, score_against),
                    # use away/home ratio. Otherwise use home/away ratio. Use 1.0 as default for ratios.
                    if base_col in ['tov_rate', 'def_rating', 'score_against']:
                        # Lower is better: Favorable ratio is < 1 (e.g., away_tov/home_tov)
                        diff = safe_divide(away_vals, home_vals, default_val=1.0)
                    else:
                        # Higher is better: Favorable ratio is > 1 (e.g., home_off_rating/away_off_rating)
                        diff = safe_divide(home_vals, away_vals, default_val=1.0)

                    new_diff_cols[diff_col_name] = diff.fillna(1.0) # Store in dict, ensure no NaNs
                else:
                    logger.warning(f"Could not calculate differential for {diff_col_name}, missing source columns: '{home_col}' or '{away_col}'. Assigning default 1.0.")
                    new_diff_cols[diff_col_name] = 1.0 # Default value for missing differential ratio

        # Assign all new differential columns at once for efficiency
        if new_diff_cols:
            local_df = local_df.assign(**new_diff_cols)

        logger.debug("Rolling: Cleaning up intermediate rolling feature columns...")
        # Drop keys used for merging and normalized team names
        local_df = local_df.drop(
            columns=['merge_key_home', 'merge_key_away', 'home_team_norm', 'away_team_norm'],
            errors='ignore' # Ignore errors if columns were already dropped or not created (e.g., merge error)
        )
        logger.debug("Finished adding rolling features.")
        return local_df

# Example Usage (requires a DataFrame 'your_df' with necessary columns)
# feature_engineer = FeatureEngineerExample()
# df_with_rolling = feature_engineer.add_rolling_features(your_df.copy())
# print(df_with_rolling.head())


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
        df_copy = pd.merge(
            df_copy,
            temp_rest,
            how='left',
            left_on=['home_team', 'game_date'],
            right_on=['team', 'game_date'],
            validate='many_to_one'  # Expect each home_team/game_date to match at most one row
        ).rename(columns={'prev_game_date': 'prev_home_game_date'}).drop(columns='team', errors='ignore')

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
    def add_matchup_history_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame], max_games: int = 5) -> pd.DataFrame:
        """Adds head-to-head matchup statistics based on recent historical games."""
        logger.debug(f"Adding head-to-head matchup features (last {max_games} games)...")
        if df is None or df.empty:
            return df
        result_df = df.copy()
        placeholder_cols = list(self._get_matchup_history_single("", "", pd.DataFrame(), 0).keys())
        if historical_df is None or historical_df.empty:
            logger.warning("Historical DataFrame empty or None. Adding H2H placeholders with defaults.")
            for col in placeholder_cols:
                if col not in result_df.columns:
                    default_key = col.replace('matchup_', '')
                    default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                    result_df[col] = default_val
            return result_df
        try:
            hist_df = historical_df.copy()
            hist_required = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
            if not all(col in hist_df.columns for col in hist_required):
                raise ValueError(f"H2H: Historical DF missing required columns: {set(hist_required) - set(hist_df.columns)}")
            hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
            hist_df['home_score'] = pd.to_numeric(hist_df['home_score'], errors='coerce')
            hist_df['away_score'] = pd.to_numeric(hist_df['away_score'], errors='coerce')
            hist_df = hist_df.dropna(subset=['game_date', 'home_score', 'away_score'])
            hist_df['home_team_norm'] = hist_df['home_team'].apply(self.normalize_team_name)
            hist_df['away_team_norm'] = hist_df['away_team'].apply(self.normalize_team_name)
            hist_df['matchup_key'] = hist_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            hist_df = hist_df.sort_values('game_date')
            logger.debug("Grouping H2H history by matchup key...")
            hist_lookup = {key: group for key, group in hist_df.groupby('matchup_key', observed=True)}
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date'])
            if result_df.empty:
                logger.warning("H2H: No valid input rows after date processing.")
                return df
            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            result_df['matchup_key'] = result_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            logger.debug("Calculating H2H features row by row...")
            h2h_results_list = []
            for index, row in result_df.iterrows():
                matchup_hist = hist_lookup.get(row['matchup_key'], pd.DataFrame())
                single_h2h = self._get_matchup_history_single(
                    home_team_norm=row['home_team_norm'],
                    away_team_norm=row['away_team_norm'],
                    historical_subset=matchup_hist,
                    max_games=max_games,
                    current_game_date=row['game_date']
                )
                h2h_results_list.append(single_h2h)
            if h2h_results_list:
                h2h_results_df = pd.DataFrame(h2h_results_list, index=result_df.index)
                result_df = result_df.join(h2h_results_df, how='left')
            else:
                logger.warning("No head-to-head results were generated.")
            logger.debug("Finalizing H2H features (filling defaults)...")
            for col in placeholder_cols:
                default_key = col.replace('matchup_', '')
                default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                if col not in result_df.columns:
                    result_df[col] = default_val
                else:
                    result_df[col] = result_df[col].fillna(default_val)
                if col == 'matchup_last_date':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                else:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)
                if col in ['matchup_num_games', 'matchup_streak']:
                    result_df[col] = result_df[col].fillna(0).round().astype(int)
        except Exception as e:
            logger.error(f"Error adding H2H features: {e}", exc_info=True)
            for col in placeholder_cols:
                if col not in result_df.columns:
                    default_key = col.replace('matchup_', '')
                    default_val = self.defaults.get(default_key, 0.0 if col != 'matchup_last_date' else pd.NaT)
                    result_df[col] = default_val
        result_df = result_df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')
        logger.debug("Finished adding head-to-head features.")
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
            result_df = pd.merge(result_df, ts_merge.rename(columns=home_rename), how='left',
                                 left_on='merge_key_home', right_on='merge_key', indicator='_merge_home')
            result_df = result_df.drop(columns=['merge_key'], errors='ignore')
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
            return default_result
        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        if past_games_df.empty:
            return default_result
        try:
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            if recent_matchups.empty:
                return default_result
            recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce')
            recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
            recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score'])
            if recent_matchups.empty:
                return default_result
            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []
            home_persp_wins = 0
            current_streak = 0
            last_winner_norm = None
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
                h_score = game['home_score']
                a_score = game['away_score']
                g_home_norm = game.get('home_team_norm')
                g_away_norm = game.get('away_team_norm')
                if g_home_norm == home_team_norm:
                    diff = h_score - a_score
                    won = h_score > a_score
                    h_persp_score = h_score
                    a_persp_score = a_score
                elif g_away_norm == home_team_norm:
                    diff = a_score - h_score
                    won = a_score > h_score
                    h_persp_score = a_score
                    a_persp_score = h_score
                else:
                    continue
                diffs.append(diff)
                total_scores.append(h_score + a_score)
                home_persp_scores.append(h_persp_score)
                away_persp_scores.append(a_persp_score)
                if won:
                    home_persp_wins += 1
                winner_norm = home_team_norm if won else away_team_norm
                if last_winner_norm is None:
                    last_winner_norm = winner_norm
                    current_streak = 1 if won else -1
                elif winner_norm == last_winner_norm:
                    current_streak += (1 if won else -1)
                else:
                    last_winner_norm = winner_norm
                    current_streak = 1 if won else -1
            num_games = len(diffs)
            if num_games == 0:
                return default_result
            final_stats = {
                'matchup_num_games': num_games,
                'matchup_avg_point_diff': np.mean(diffs) if diffs else 0.0,
                'matchup_home_win_pct': home_persp_wins / num_games if num_games > 0 else 0.5,
                'matchup_avg_total_score': np.mean(total_scores) if total_scores else 0.0,
                'matchup_avg_home_score': np.mean(home_persp_scores) if home_persp_scores else 0.0,
                'matchup_avg_away_score': np.mean(away_persp_scores) if away_persp_scores else 0.0,
                'matchup_last_date': recent_matchups['game_date'].max(),
                'matchup_streak': int(current_streak)
            }
            for k, v in default_result.items():
                final_stats.setdefault(k, v)
                if pd.isna(final_stats[k]):
                    final_stats[k] = v
            return final_stats
        except Exception as e:
            logger.error(f"Error in _get_matchup_history_single ({home_team_norm} vs {away_team_norm}): {e}", exc_info=True)
            return default_result

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        """Extracts metrics (win %, streak, momentum) from a form string like 'WWLWL'."""
        defaults = {
            'form_win_pct': self.defaults['form_win_pct'],
            'current_streak': self.defaults['current_streak'],
            'momentum_direction': self.defaults['momentum_direction']
        }
        if not form_string or pd.isna(form_string) or not isinstance(form_string, str):
            return defaults
        form_string = form_string.upper().strip().replace('-', '').replace('?', '')
        form_len = len(form_string)
        if form_len == 0 or form_string == 'N/A':
            return defaults
        wins = form_string.count('W')
        form_win_pct = wins / form_len
        current_streak = 0
        if form_len > 0:
            streak_char = form_string[-1]
            streak_count = 0
            for char in reversed(form_string):
                if char == streak_char:
                    streak_count += 1
                else:
                    break
            current_streak = streak_count if streak_char == 'W' else -streak_count
        momentum_direction = 0.0
        if form_len >= 4:
            split_point = form_len // 2
            recent_half = form_string[-split_point:]
            older_half = form_string[:form_len - split_point]
            len_r, len_o = len(recent_half), len(older_half)
            if len_r > 0 and len_o > 0:
                pct_r = recent_half.count('W') / len_r
                pct_o = older_half.count('W') / len_o
                if pct_r > pct_o:
                    momentum_direction = 1.0
                elif pct_r < pct_o:
                    momentum_direction = -1.0
        return {'form_win_pct': form_win_pct,
                'current_streak': int(current_streak),
                'momentum_direction': momentum_direction}


    @profile_time
    def generate_all_features(self,
                              df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None,
                              betting_odds_data: Optional[Union[pd.DataFrame, Dict[Any, Dict]]] = None,
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
    # engine = NBAFeatureEngine(debug=True)
    # dummy_df = pd.DataFrame(...) # Create dummy data
    # features = engine.generate_all_features(dummy_df)
    # print(features.info())
    pass
