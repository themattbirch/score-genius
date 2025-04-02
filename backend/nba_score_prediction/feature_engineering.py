# feature_engineering.py - Unified Feature Engineering for NBA Score Prediction
"""
NBAFeatureEngine - Unified module for NBA score prediction feature engineering.
Handles the creation of various features for NBA game prediction models.

Includes debug capabilities to save summary statistics and plots for intermediate features.
"""
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
from functools import wraps, lru_cache
import functools  # Keep for standalone wraps usage
from typing import Dict, List, Tuple, Optional, Union, Any
import logging  # Use standard logging
from pathlib import Path

# --- Matplotlib and Seaborn for Debug Plots ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib or seaborn not found. Debug plots will not be generated.")
# --------------------------------------------

# Configure logger for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)  # Get logger for this module

# Small constant to avoid division by zero or near-zero
EPSILON = 1e-6

# --- Define Output Directory for Debugging ---
try:
    SCRIPT_DIR_FE = Path(__file__).resolve().parent
    REPORTS_DIR_FE = SCRIPT_DIR_FE.parent.parent / 'reports'
except NameError: # Fallback if __file__ is not defined (e.g., interactive)
    REPORTS_DIR_FE = Path('./reports') # Use relative path

FEATURE_DEBUG_DIR = REPORTS_DIR_FE / "feature_debug"
# --- End Directory Definition ---


# -------------------- NBAFeatureEngine Class --------------------
class NBAFeatureEngine:
    """Core class for NBA feature engineering."""

    def __init__(self, supabase_client: Optional[Any] = None, debug: bool = False):
        """
        Initialize the feature engine.
        Args:
            supabase_client: Optional configured Supabase client instance.
            debug: If True, enable debug logging AND saving of debug stats/plots.
        """
        self.debug = debug # This flag controls debug logging AND saving diagnostic files
        if self.debug:
            logger.setLevel(logging.DEBUG)
            FEATURE_DEBUG_DIR.mkdir(parents=True, exist_ok=True) # Create dir if debugging
            logger.info(f"DEBUG mode enabled. Diagnostic outputs will be saved to: {FEATURE_DEBUG_DIR}")
        else:
             logger.setLevel(logging.INFO)

        self.supabase_client = supabase_client
        logger.debug("NBAFeatureEngine Initialized.")

        # --- Default Values for Fallbacks ---
        self.defaults = {
            'win_pct': 0.5, 'avg_pts_for': 115.0, 'avg_pts_against': 115.0,
            'home_advantage': 3.0, 'offensive_rating': 115.0, 'defensive_rating': 115.0,
            'pace': 100.0, 'estimated_possessions': 100.0, 'oreb_pct': 0.23,
            'dreb_pct': 0.77, 'trb_pct': 0.50, 'tov_rate': 13.0, 'efg_pct': 0.54,
            'ft_rate': 0.20, 'score_for': 115.0, 'score_against': 115.0,
            'off_rating': 115.0, 'def_rating': 115.0, 'momentum_ewma': 0.0,
            'matchup_avg_point_diff': 0.0, 'matchup_streak': 0, 'form_win_pct': 0.5,
            'current_streak': 0, 'momentum_direction': 0.0, 'rest_days': 3.0,
            'score_for_std': 10.0, 'score_against_std': 10.0, 'off_rating_std': 10.0,
            'def_rating_std': 10.0, 'pace_std': 5.0, 'efg_pct_std': 0.05,
            'tov_rate_std': 3.0, 'trb_pct_std': 0.05, 'oreb_pct_std': 0.05,
            'dreb_pct_std': 0.05, 'momentum_ewma_std': 5.0, 'matchup_num_games': 0,
            'matchup_avg_total_score': 230.0, 'matchup_avg_home_score': 115.0,
            'matchup_avg_away_score': 115.0, 'matchup_home_win_pct': 0.5,
            'vegas_home_spread': 0.0, 'vegas_over_under': 230.0,
            # Defaults for games_last_x_days (used if rest calculation fails)
            'games_last_7_days_home': 2, 'games_last_14_days_home': 4,
            'games_last_7_days_away': 2, 'games_last_14_days_away': 4,
        }
        self.league_averages = {
            'score': self.defaults['avg_pts_for'],
            'quarter_scores': {1: 28.5, 2: 28.5, 3: 28.0, 4: 29.0}
        }

    # --------------------------------------------------------------------------
    # Logging and Utility Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def profile_time(func=None, debug_mode=None):
        """Decorator to profile the execution time of a function."""
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = f(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                message = f"{f.__name__} executed in {execution_time:.4f} seconds"
                # Check instance's debug flag if 'self' is the first arg
                is_debug_instance = False
                if args and isinstance(args[0], NBAFeatureEngine):
                    is_debug_instance = args[0].debug

                final_debug_mode = debug_mode if debug_mode is not None else is_debug_instance
                if final_debug_mode:
                    logger.debug(f"[Profiler] {message}")
                return result
            return wrapper
        return decorator if func is None else decorator(func)

    @lru_cache(maxsize=512)
    def normalize_team_name(self, team_name: Optional[str]) -> str:
        """Normalizes team name for consistent lookup."""
        if not isinstance(team_name, str):
            return "Unknown"
        team_lower = team_name.lower().strip()
        mapping = {
            "atlanta hawks": "hawks", "atlanta": "hawks", "atl": "hawks", "hawks": "hawks", "atlanta h": "hawks",
            "boston celtics": "celtics", "boston": "celtics", "bos": "celtics", "celtics": "celtics",
            "brooklyn nets": "nets", "brooklyn": "nets", "bkn": "nets", "nets": "nets",
            "charlotte hornets": "hornets", "charlotte": "hornets", "cha": "hornets", "hornets": "hornets",
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
            "new orleans pelicans": "pelicans", "new orleans": "pelicans", "nop": "pelicans", "pelicans": "pelicans",
            "new york knicks": "knicks", "new york": "knicks", "nyk": "knicks", "knicks": "knicks",
            "oklahoma city thunder": "thunder", "oklahoma city": "thunder", "okc": "thunder", "thunder": "thunder",
            "orlando magic": "magic", "orlando": "magic", "orl": "magic", "magic": "magic", "orlando mag": "magic",
            "philadelphia 76ers": "76ers", "philadelphia": "76ers", "phi": "76ers", "76ers": "76ers", "sixers": "76ers",
            "phoenix suns": "suns", "phoenix": "suns", "phx": "suns", "suns": "suns", "phoenix s": "suns",
            "portland trail blazers": "blazers", "portland": "blazers", "por": "blazers", "blazers": "blazers", "trail blazers": "blazers", "portland trail": "blazers",
            "sacramento kings": "kings", "sacramento": "kings", "sac": "kings", "kings": "kings",
            "san antonio spurs": "spurs", "san antonio": "spurs", "sas": "spurs", "spurs": "spurs", "san antonio s": "spurs",
            "toronto raptors": "raptors", "toronto": "raptors", "tor": "raptors", "raptors": "raptors", "toronto rap": "raptors",
            "utah jazz": "jazz", "utah": "jazz", "uta": "jazz", "jazz": "jazz",
            "washington wizards": "wizards", "washington": "wizards", "was": "wizards", "wizards": "wizards", "wiz": "wizards",
            # Add potentially missing ones if needed
            "indiana":"pacers", "la":"lakers", # Ambiguous, check data source preference
            "charlotte bobcats": "hornets", # Historical name change
            "new jersey nets": "nets", # Historical name change
            "new orleans/oklahoma city hornets": "pelicans", # Historical name change
            "seattle supersonics": "thunder", # Historical name change
            # Common non-NBA team names to map to 'other' or handle specifically if needed
            "east": "east", "west": "west", "team lebron": "team_lebron", "team durant":"team_durant",
            "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
            "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
        }
        if team_lower in mapping:
            return mapping[team_lower]
        # Keep the partial matching logic as a fallback
        for name, norm in mapping.items():
            if len(team_lower) > 3 and team_lower in name: return norm
            if len(name) > 3 and name in team_lower: return norm

        logger.warning(f"Team name '{team_name}' normalized to '{team_lower}' - no specific mapping found!")
        return team_lower # Return potentially unmapped name

    def _determine_season(self, game_date: pd.Timestamp) -> str:
        """ REVISED: Determines the NBA season string in YYYY-YYYY format. """
        if pd.isna(game_date):
            logger.warning("Missing game_date in _determine_season, cannot determine season.")
            return "Unknown_Season"
        year = game_date.year
        month = game_date.month
        if month >= 9: # Assuming season start month is around September/October
            start_year = year
            end_year = year + 1
        else:
            start_year = year - 1
            end_year = year
        return f"{start_year}-{end_year}" # Use full end_year

    # --------------------------------------------------------------------------
    # Feature Calculation Methods
    # --------------------------------------------------------------------------

    # --- add_intra_game_momentum ---
    @profile_time
    def add_intra_game_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """[Vectorized] Adds intra-game momentum features based on quarter scores."""
        logger.debug("Adding intra-game momentum features...")
        if df is None or df.empty: return df
        result_df = df.copy()
        qtr_cols = [f'{loc}_q{i}' for loc in ['home', 'away'] for i in range(1, 5)]
        missing_cols = [col for col in qtr_cols if col not in result_df.columns]
        if missing_cols: logger.warning(f"Momentum: Missing quarter cols: {missing_cols}. Filling with 0.")
        for col in qtr_cols: result_df[col] = pd.to_numeric(result_df.get(col), errors='coerce').fillna(0) # Use .get()
        for i in range(1, 5): result_df[f'q{i}_margin'] = result_df.get(f'home_q{i}', 0) - result_df.get(f'away_q{i}', 0)
        result_df['end_q1_diff'] = result_df.get('q1_margin', 0)
        result_df['end_q2_diff'] = result_df.get('end_q1_diff', 0) + result_df.get('q2_margin', 0)
        result_df['end_q3_diff'] = result_df.get('end_q2_diff', 0) + result_df.get('q3_margin', 0)
        result_df['end_q4_reg_diff'] = result_df.get('end_q3_diff', 0) + result_df.get('q4_margin', 0)
        result_df['q2_margin_change'] = result_df.get('q2_margin', 0) - result_df.get('q1_margin', 0)
        result_df['q3_margin_change'] = result_df.get('q3_margin', 0) - result_df.get('q2_margin', 0)
        result_df['q4_margin_change'] = result_df.get('q4_margin', 0) - result_df.get('q3_margin', 0)
        valid_q_margin_cols = [f'q{i}_margin' for i in range(1, 5) if f'q{i}_margin' in result_df.columns]
        if len(valid_q_margin_cols) == 4:
            quarter_margins = result_df[valid_q_margin_cols]
            result_df['momentum_score_ewma_q4'] = quarter_margins.ewm(span=3, axis=1, adjust=False).mean().iloc[:, -1].fillna(0)
            q3_cols = valid_q_margin_cols[:3]
            result_df['momentum_score_ewma_q3'] = result_df[q3_cols].ewm(span=2, axis=1, adjust=False).mean().iloc[:, -1].fillna(0)
        else:
            logger.warning("Momentum: Not enough quarter margin columns for EWMA.")
            result_df['momentum_score_ewma_q4'] = 0.0; result_df['momentum_score_ewma_q3'] = 0.0
        logger.debug("Finished adding intra-game momentum features.")
        return result_df

    @profile_time
    def integrate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """REVISED: Calculates advanced metrics. Includes Pace/Poss debugging."""
        logger.debug("Integrating advanced metrics (v4 - Debugging Pace/Poss)...")
        result_df = df.copy()
        # Ensure all required input columns exist and are numeric
        stat_cols = ['home_score', 'away_score', 'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted','home_3pm', 'home_3pa', 'away_3pm', 'away_3pa', 'home_ft_made', 'home_ft_attempted','away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb', 'home_total_reb','away_off_reb', 'away_def_reb', 'away_total_reb', 'home_turnovers', 'away_turnovers','home_ot', 'away_ot']
        for col in stat_cols: result_df[col] = pd.to_numeric(result_df.get(col), errors='coerce').fillna(0)

        def safe_divide(numerator, denominator, default_val):
            num = pd.to_numeric(numerator, errors='coerce'); den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
            return (num / den).fillna(default_val)

        # --- Basic Shooting Stats ---
        result_df['home_efg_pct'] = safe_divide(result_df['home_fg_made'] + 0.5 * result_df['home_3pm'], result_df['home_fg_attempted'], self.defaults['efg_pct'])
        result_df['away_efg_pct'] = safe_divide(result_df['away_fg_made'] + 0.5 * result_df['away_3pm'], result_df['away_fg_attempted'], self.defaults['efg_pct'])
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
        result_df['home_ft_rate'] = safe_divide(result_df['home_ft_attempted'], result_df['home_fg_attempted'], self.defaults['ft_rate'])
        result_df['away_ft_rate'] = safe_divide(result_df['away_ft_attempted'], result_df['away_fg_attempted'], self.defaults['ft_rate'])
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']

        # --- Rebounding Percentages ---
        result_df['home_oreb_pct'] = safe_divide(result_df['home_off_reb'], result_df['home_off_reb'] + result_df['away_def_reb'], self.defaults['oreb_pct'])
        result_df['away_dreb_pct'] = safe_divide(result_df['away_def_reb'], result_df['away_def_reb'] + result_df['home_off_reb'], self.defaults['dreb_pct']) # Team perspective DReb%
        result_df['away_oreb_pct'] = safe_divide(result_df['away_off_reb'], result_df['away_off_reb'] + result_df['home_def_reb'], self.defaults['oreb_pct'])
        result_df['home_dreb_pct'] = safe_divide(result_df['home_def_reb'], result_df['home_def_reb'] + result_df['away_off_reb'], self.defaults['dreb_pct']) # Team perspective DReb%
        result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
        result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct']
        result_df['home_trb_pct'] = safe_divide(result_df['home_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct'])
        result_df['away_trb_pct'] = safe_divide(result_df['away_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct'])
        result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']

        # --- Possessions and Pace (CORRECTED FORMULA + DEBUGGING) ---
        # ** Add Debug Logging for Inputs **
            # --- Add Detailed Input Logging for Possession ---
        if self.debug and len(result_df) > 0:
            logger.debug("--- Possession Input Debug ---")
            inputs_to_check = ['home_fg_attempted', 'home_ft_attempted', 'home_off_reb', 'home_turnovers',
                               'away_fg_attempted', 'away_ft_attempted', 'away_off_reb', 'away_turnovers']
            # Log stats for required inputs
            if all(c in result_df.columns for c in inputs_to_check):
                logger.debug(f"Input Stats (Describe):\n{result_df[inputs_to_check].describe().to_string()}")
                # Log first few rows
                logger.debug(f"Input Stats (Head):\n{result_df[inputs_to_check].head().to_string()}")
            else:
                logger.warning(f"Missing some inputs needed for possession debug log: {[c for c in inputs_to_check if c not in result_df]}")
        # --- End Input Logging ---

        # Standard approximation: Poss ≈ FGA + 0.44*FTA - OReb + TOV
        home_poss = result_df['home_fg_attempted'] + 0.44 * result_df['home_ft_attempted'] - result_df['home_off_reb'] + result_df['home_turnovers']
        away_poss = result_df['away_fg_attempted'] + 0.44 * result_df['away_ft_attempted'] - result_df['away_off_reb'] + result_df['away_turnovers']

        # Average team possessions for game estimate
        result_df['possessions_est'] = (0.5 * (home_poss + away_poss)).fillna(self.defaults['estimated_possessions'])
        # REMOVE CLIP lower=70 temporarily for debugging
        # result_df['possessions_est'] = result_df['possessions_est'].clip(lower=70.0)

        if self.debug:
             logger.debug(f"RAW Possession Estimates (Describe):\n{result_df['possessions_est'].describe().to_string()}")
             logger.debug(f"RAW Possession Estimates (Sample):\n{result_df['possessions_est'].head().to_string()}")
        result_df['home_possessions'] = result_df['possessions_est'].replace(0, np.nan)
        result_df['away_possessions'] = result_df['possessions_est'].replace(0, np.nan)

        # Pace: Possessions per 48 minutes
        num_ot = np.where((result_df['home_ot'] > 0) | (result_df['away_ot'] > 0), 1, 0)
        game_minutes_calc = 48.0 + num_ot * 5.0
        result_df['game_minutes_played'] = np.where(game_minutes_calc <= 0, 48.0, game_minutes_calc) # Use corrected non-replace version

        # Calculate Pace using the NEW raw possessions_est
        result_df['game_pace'] = safe_divide(result_df['possessions_est'] * 48.0, result_df['game_minutes_played'], self.defaults['pace'])
        result_df['home_pace'] = result_df['game_pace']; result_df['away_pace'] = result_df['game_pace']
        result_df['pace_differential'] = 0.0 # Pace is game-level

        # --- Efficiency (Ratings) - Points per 100 Possessions ---
        result_df['home_offensive_rating'] = safe_divide(result_df['home_score'] * 100, result_df['home_possessions'], self.defaults['offensive_rating'])
        result_df['away_offensive_rating'] = safe_divide(result_df['away_score'] * 100, result_df['away_possessions'], self.defaults['offensive_rating'])
        result_df['home_defensive_rating'] = result_df['away_offensive_rating']; result_df['away_defensive_rating'] = result_df['home_offensive_rating']
        # ** CLIPPING REMOVED **
        rating_cols = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']
        for col in rating_cols: result_df[col] = result_df[col].fillna(self.defaults.get(col.split('_')[1]+'_rating', 115.0)) # Keep fillna
        result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
        result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']
        result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']

        # --- Turnover Rate (TOV per 100 Poss) ---
        result_df['home_tov_rate'] = safe_divide(result_df['home_turnovers'] * 100, result_df['home_possessions'], self.defaults['tov_rate'])
        result_df['away_tov_rate'] = safe_divide(result_df['away_turnovers'] * 100, result_df['away_possessions'], self.defaults['tov_rate'])
        # ** CLIPPING REMOVED **
        result_df['home_tov_rate'] = result_df['home_tov_rate'].fillna(self.defaults['tov_rate']) # Keep fillna
        result_df['away_tov_rate'] = result_df['away_tov_rate'].fillna(self.defaults['tov_rate']) # Keep fillna
        result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate']

        # --- Convenience Score Columns ---
        if 'total_score' not in result_df.columns: result_df['total_score'] = result_df['home_score'] + result_df['away_score']
        if 'point_diff' not in result_df.columns: result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

        # Final fillna for potentially NaN possession columns used as divisors
        result_df['home_possessions'] = result_df['home_possessions'].fillna(self.defaults['estimated_possessions'])
        result_df['away_possessions'] = result_df['away_possessions'].fillna(self.defaults['estimated_possessions'])

        # --- Debug Stats Generation (Keep this block) ---
        if self.debug:
            try:
                FEATURE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("Generating debug stats for integrate_advanced_features...")
                key_metrics = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating', 'game_pace', 'home_efg_pct', 'away_efg_pct', 'home_tov_rate', 'away_tov_rate', 'home_trb_pct', 'away_trb_pct', 'possessions_est', 'home_net_rating', 'away_net_rating', 'efficiency_differential', 'home_ft_rate', 'away_ft_rate'] # Added FT Rate
                metrics_to_describe = [m for m in key_metrics if m in result_df.columns]
                if metrics_to_describe:
                    summary = result_df[metrics_to_describe].describe()
                    summary_path = FEATURE_DEBUG_DIR / "advanced_metrics_summary.txt"
                    with open(summary_path, 'w') as f: f.write(f"Summary Stats for Advanced Metrics (n={len(result_df)}):\n{summary.to_string()}")
                    logger.info(f"Saved advanced metrics summary to {summary_path}")
                    if PLOTTING_AVAILABLE:
                        plot_cols = ['home_offensive_rating', 'game_pace', 'home_efg_pct', 'home_tov_rate', 'possessions_est', 'home_ft_rate'] # Added Poss and FTr
                        for col in plot_cols:
                             if col in result_df.columns:
                                 plt.figure(figsize=(8, 5)); sns.histplot(result_df[col].dropna(), kde=True, bins=30); plt.title(f'Distribution: {col}')
                                 plot_path = FEATURE_DEBUG_DIR / f"advanced_{col}_dist.png"; plt.savefig(plot_path); plt.close()
                                 logger.info(f"Saved distribution plot to {plot_path}")
                    else: logger.warning("Plotting libraries not available.")
                else: logger.warning("No key advanced metrics found to describe.")
            except Exception as e: logger.error(f"Error generating debug stats for advanced features: {e}", exc_info=True)

        logger.debug("Finished integrating advanced features (v4 - Debugging Pace/Poss, No Clipping).")
        return result_df

    # --- add_rolling_features ---
    @profile_time
    def add_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10]) -> pd.DataFrame:
        """REVISED (v2): Adds rolling mean AND std dev features using corrected base metrics."""
        logger.debug(f"Adding rolling features (v2) for windows: {window_sizes}...")
        if df is None or df.empty: logger.warning("Input df empty for rolling."); return df
        local_df = df.copy()
        cols_to_roll_generic = ['score_for', 'score_against', 'momentum_ewma', 'off_rating', 'def_rating', 'net_rating', 'pace', 'efg_pct', 'tov_rate', 'trb_pct', 'oreb_pct', 'dreb_pct', 'ft_rate']
        source_mapping = {'home_score': 'score_for', 'away_score': 'score_against', 'momentum_score_ewma_q4': 'momentum_ewma', 'home_offensive_rating': 'off_rating', 'home_defensive_rating': 'def_rating', 'home_net_rating': 'net_rating', 'home_pace': 'pace', 'home_efg_pct': 'efg_pct', 'home_tov_rate': 'tov_rate', 'home_trb_pct': 'trb_pct', 'home_oreb_pct': 'oreb_pct', 'home_dreb_pct': 'dreb_pct', 'home_ft_rate': 'ft_rate'}
        away_source_mapping_override = {'away_score': 'score_for', 'home_score': 'score_against', 'momentum_score_ewma_q4': 'momentum_ewma', 'away_offensive_rating': 'off_rating', 'away_defensive_rating': 'def_rating', 'away_net_rating': 'net_rating', 'away_pace': 'pace', 'away_efg_pct': 'efg_pct', 'away_tov_rate': 'tov_rate', 'away_trb_pct': 'trb_pct', 'away_oreb_pct': 'oreb_pct', 'away_dreb_pct': 'dreb_pct', 'away_ft_rate': 'ft_rate'}
        required_sources = set(source_mapping.keys()) | set(away_source_mapping_override.keys()) | {'game_id', 'game_date', 'home_team', 'away_team'}
        missing_sources = [col for col in required_sources if col not in local_df.columns];
        if missing_sources: logger.warning(f"Rolling: Missing source cols: {missing_sources}. Defaults used.")
        cols_to_roll_final = [];
        for source_col_dict in [source_mapping, away_source_mapping_override]:
            for source_col, generic_col in source_col_dict.items():
                if generic_col in cols_to_roll_generic and source_col in local_df.columns:
                    if generic_col not in cols_to_roll_final: cols_to_roll_final.append(generic_col)
                    default = self.defaults.get(generic_col, 0.0); local_df[source_col] = pd.to_numeric(local_df[source_col], errors='coerce').fillna(default)
        if not cols_to_roll_final: logger.error("Rolling: No valid columns to calculate rolling features."); return df
        logger.debug("Rolling: Creating team-centric view...")
        try:
            local_df['home_team_norm'] = local_df['home_team'].apply(self.normalize_team_name); local_df['away_team_norm'] = local_df['away_team'].apply(self.normalize_team_name)
            home_map = {s: g for s, g in source_mapping.items() if g in cols_to_roll_final and s in local_df.columns}; away_map = {s: g for s, g in away_source_mapping_override.items() if g in cols_to_roll_final and s in local_df.columns}
            home_view = local_df[['game_id', 'game_date', 'home_team_norm'] + list(home_map.keys())].rename(columns={'home_team_norm': 'team_norm', **home_map})
            away_view = local_df[['game_id', 'game_date', 'away_team_norm'] + list(away_map.keys())].rename(columns={'away_team_norm': 'team_norm', **away_map})
            home_view['game_date'] = pd.to_datetime(home_view['game_date']); away_view['game_date'] = pd.to_datetime(away_view['game_date'])
            team_view = pd.concat([home_view, away_view], ignore_index=True).sort_values(['team_norm', 'game_date', 'game_id'], kind='mergesort').reset_index(drop=True)
        except Exception as e: logger.error(f"Rolling: Error creating team view: {e}", exc_info=True); return df
        logger.debug(f"Rolling: Calculating stats for {cols_to_roll_final}...")
        rolling_cols_generated = [];
        for window in window_sizes:
            min_p = max(1, window // 2);
            for col in cols_to_roll_final:
                roll_mean_col = f'rolling_{col}_mean_{window}'; roll_std_col = f'rolling_{col}_std_{window}'
                default_mean = self.defaults.get(col, 0.0); default_std = self.defaults.get(f'{col}_std', 0.0)
                grouped_shifted = team_view.groupby('team_norm', observed=True)[col].shift(1)
                rolling_mean = grouped_shifted.rolling(window=window, min_periods=min_p).mean()
                rolling_std = grouped_shifted.rolling(window=window, min_periods=min_p).std()
                team_view[roll_mean_col] = rolling_mean.fillna(default_mean)
                team_view[roll_std_col] = np.maximum(0, rolling_std.fillna(default_std)) # Ensure non-negative std
                rolling_cols_generated.extend([roll_mean_col, roll_std_col])
        logger.debug("Rolling: Merging stats back...")
        try:
            team_view['merge_key_rolling'] = team_view['game_id'].astype(str) + "_" + team_view['team_norm']
            local_df['merge_key_home'] = local_df['game_id'].astype(str) + "_" + local_df['home_team_norm']
            local_df['merge_key_away'] = local_df['game_id'].astype(str) + "_" + local_df['away_team_norm']
            cols_to_merge = ['merge_key_rolling'] + rolling_cols_generated
            merge_data = team_view[cols_to_merge].drop_duplicates(subset=['merge_key_rolling'], keep='last')
            def get_rename_dict(prefix, base_cols, windows): rd = {}; [rd.update({f'rolling_{b}_{s}_{w}': f'{prefix}_rolling_{b}_{s}_{w}' for s in ['mean','std'] for w in windows}) for b in base_cols]; return rd
            local_df = pd.merge(local_df, merge_data, how='left', left_on='merge_key_home', right_on='merge_key_rolling').rename(columns=get_rename_dict('home', cols_to_roll_final, window_sizes)).drop(columns=['merge_key_rolling'], errors='ignore')
            local_df = pd.merge(local_df, merge_data, how='left', left_on='merge_key_away', right_on='merge_key_rolling').rename(columns=get_rename_dict('away', cols_to_roll_final, window_sizes)).drop(columns=['merge_key_rolling'], errors='ignore')
        except Exception as e:
            logger.error(f"Rolling: Error merging rolling stats back: {e}", exc_info=True)
            # Add placeholders if merge fails... (same fallback as before)
            for prefix in ['home', 'away']:
                 for col_base in cols_to_roll_generic:
                     for stat_type in ['mean', 'std']:
                         for w in window_sizes:
                              col_name = f'{prefix}_rolling_{col_base}_{stat_type}_{w}'
                              if col_name not in local_df.columns:
                                  default_key = f'{col_base}_std' if stat_type == 'std' else col_base
                                  local_df[col_name] = self.defaults.get(default_key, 0.0)

        logger.debug("Rolling: Filling NaNs and calculating diffs...")
        primary_window = max(window_sizes) if window_sizes else 10
        for stat_type in ['mean', 'std']:
             for col_base in cols_to_roll_final: # Ensure base columns used for diff exist
                 for w in window_sizes: # Fill NaNs for all generated rolling columns first
                      for prefix in ['home', 'away']:
                           col_name = f'{prefix}_rolling_{col_base}_{stat_type}_{w}'
                           default_key = f'{col_base}_std' if stat_type == 'std' else col_base
                           default = self.defaults.get(default_key, 0.0)
                           if col_name not in local_df.columns: local_df[col_name] = default
                           else: local_df[col_name] = local_df[col_name].fillna(default)
                           if stat_type == 'std': local_df[col_name] = np.maximum(0, local_df[col_name]) # Re-apply after fillna

             # Now calculate diffs using the primary window
             w = primary_window
             hm = self._calc_diff_safe(local_df, 'home_rolling_score_for_{stat_type}_{w}', 'home_rolling_score_against_{stat_type}_{w}', stat_type, w)
             am = self._calc_diff_safe(local_df, 'away_rolling_score_for_{stat_type}_{w}', 'away_rolling_score_against_{stat_type}_{w}', stat_type, w)
             local_df[f'rolling_margin_diff_{stat_type}'] = (hm - am).fillna(0.0)
             hne = self._calc_diff_safe(local_df, 'home_rolling_off_rating_{stat_type}_{w}', 'home_rolling_def_rating_{stat_type}_{w}', stat_type, w)
             ane = self._calc_diff_safe(local_df, 'away_rolling_off_rating_{stat_type}_{w}', 'away_rolling_def_rating_{stat_type}_{w}', stat_type, w)
             local_df[f'rolling_eff_diff_{stat_type}'] = (hne - ane).fillna(0.0)
             # Calculate other diffs similarly, ensuring base cols exist
             if 'pace' in cols_to_roll_final: local_df[f'rolling_pace_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_pace_{stat_type}_{w}', 'away_rolling_pace_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'efg_pct' in cols_to_roll_final: local_df[f'rolling_efg_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_efg_pct_{stat_type}_{w}', 'away_rolling_efg_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'tov_rate' in cols_to_roll_final: local_df[f'rolling_tov_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'away_rolling_tov_rate_{stat_type}_{w}', 'home_rolling_tov_rate_{stat_type}_{w}', stat_type, w).fillna(0.0) # Note: away-home
             if 'trb_pct' in cols_to_roll_final: local_df[f'rolling_trb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_trb_pct_{stat_type}_{w}', 'away_rolling_trb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'oreb_pct' in cols_to_roll_final: local_df[f'rolling_oreb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_oreb_pct_{stat_type}_{w}', 'away_rolling_oreb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'dreb_pct' in cols_to_roll_final: local_df[f'rolling_dreb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_dreb_pct_{stat_type}_{w}', 'away_rolling_dreb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'ft_rate' in cols_to_roll_final: local_df[f'rolling_ft_rate_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_ft_rate_{stat_type}_{w}', 'away_rolling_ft_rate_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'momentum_ewma' in cols_to_roll_final: local_df[f'rolling_momentum_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_momentum_ewma_{stat_type}_{w}', 'away_rolling_momentum_ewma_{stat_type}_{w}', stat_type, w).fillna(0.0)


        # --- Debug Stats for Rolling Features ---
        if self.debug: # Check the debug flag
            try:
                FEATURE_DEBUG_DIR.mkdir(parents=True, exist_ok=True) # Ensure exists
                logger.info("Generating debug stats for add_rolling_features...")
                # Select representative rolling columns (e.g., for the middle window size)
                if not window_sizes: window_sizes=[5,10] # Use default if empty
                debug_window = window_sizes[len(window_sizes) // 2] # Pick middle window
                key_rolling_cols = []
                rolled_bases = cols_to_roll_final if 'cols_to_roll_final' in locals() and isinstance(cols_to_roll_final, list) else []

                for prefix in ['home', 'away']:
                    for base in rolled_bases:
                         for stat_type in ['mean', 'std']:
                             col_name = f'{prefix}_rolling_{base}_{stat_type}_{debug_window}'
                             if col_name in local_df.columns:
                                 key_rolling_cols.append(col_name)
                # Add some diff columns too
                diff_cols_to_check = [f'rolling_margin_diff_mean', f'rolling_eff_diff_mean', f'rolling_pace_diff_mean', f'rolling_margin_diff_std', f'rolling_eff_diff_std']
                key_rolling_cols.extend([c for c in diff_cols_to_check if c in local_df.columns])

                if key_rolling_cols:
                    # Save Summary Stats
                    summary = local_df[key_rolling_cols].describe()
                    summary_path = FEATURE_DEBUG_DIR / f"rolling_features_w{debug_window}_summary.txt"
                    with open(summary_path, 'w') as f:
                        f.write(f"Summary Statistics for Rolling Features (Window={debug_window}, n={len(local_df)}):\n")
                        f.write(summary.to_string(max_rows=100))
                    logger.info(f"Saved rolling features summary to {summary_path}")

                    # Check Std Dev Columns
                    std_cols = [c for c in key_rolling_cols if '_std' in c and 'diff' not in c] # Check base std devs
                    if std_cols:
                        # Calculate % of values close to zero
                        std_check_summary = local_df[std_cols].agg(lambda x: (np.abs(x) < EPSILON).mean()).reset_index()
                        std_check_summary.columns = ['Feature', '% Near Zero']
                        # Get min/max too
                        std_min_max = local_df[std_cols].agg(['min', 'max']).T.reset_index()
                        std_min_max.columns = ['Feature', 'Min', 'Max']
                        # Merge checks
                        std_final_summary = pd.merge(std_check_summary, std_min_max, on='Feature')

                        zero_std_path = FEATURE_DEBUG_DIR / f"rolling_std_dev_w{debug_window}_check.txt"
                        with open(zero_std_path, 'w') as f:
                            f.write(f"Rolling Std Dev Check (Window={debug_window}, n={len(local_df)}):\n")
                            f.write(std_final_summary.to_string(float_format="%.4f", index=False))
                        logger.info(f"Saved rolling std dev check to {zero_std_path}")
                        # Log warning if many std devs are zero
                        if not std_final_summary.empty and (std_final_summary['% Near Zero'] > 0.1).any():
                            logger.warning(f"High percentage of near-zero values found in some rolling std dev features (w={debug_window}). Check base metric variance.")

                    # Save Distribution Plots (Optional)
                    if PLOTTING_AVAILABLE:
                        plot_cols = [f'home_rolling_off_rating_mean_{debug_window}', f'home_rolling_off_rating_std_{debug_window}', f'rolling_eff_diff_mean']
                        for col in plot_cols:
                            if col in local_df.columns:
                                plt.figure(figsize=(8, 5))
                                sns.histplot(local_df[col].dropna(), kde=True, bins=30)
                                plt.title(f'Distribution: {col}')
                                plot_path = FEATURE_DEBUG_DIR / f"rolling_{col}_dist.png"
                                plt.savefig(plot_path)
                                plt.close() # Close plot
                                logger.info(f"Saved distribution plot to {plot_path}")
                    else:
                         logger.warning("Plotting libraries not available, skipping debug plots.")
                else:
                    logger.warning("No key rolling features found to describe.")
            except Exception as e:
                 logger.error(f"Error generating debug stats for rolling features: {e}", exc_info=True)


        logger.debug("Rolling: Cleaning up intermediate columns...")
        local_df = local_df.drop(columns=['merge_key_home', 'merge_key_away', 'home_team_norm', 'away_team_norm'], errors='ignore')
        logger.debug("Finished adding rolling features (v2).")
        return local_df

    # --- _calc_diff_safe ---
    def _calc_diff_safe(self, df: pd.DataFrame, home_template: str, away_template: str, stat_type: str, window: int) -> pd.Series:
        """Helper to compute the difference between two rolling columns."""
        home_col = home_template.format(stat_type=stat_type, w=window)
        away_col = away_template.format(stat_type=stat_type, w=window)
        # Check if columns exist before attempting access
        if home_col in df.columns and away_col in df.columns:
            # Ensure they are numeric before subtracting
            home_vals = pd.to_numeric(df[home_col], errors='coerce')
            away_vals = pd.to_numeric(df[away_col], errors='coerce')
            return (home_vals - away_vals) # Let downstream handle NaNs if needed
        else:
            logger.warning(f"Columns {home_col} or {away_col} not found for diff calculation.")
            return pd.Series(np.nan, index=df.index) # Return NaNs if columns missing

    # --- add_rest_features_vectorized (v5.1 - Time-Based w/ Merge) ---
    @profile_time
    def add_rest_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rest days and schedule density features using time-based rolling.
        REVISED (v5.1 - Corrected Key Matching Merge) for robustness to duplicate dates.
        """
        logger.debug("Adding rest features (vectorized, v5.1 - Time-Based w/ Merge)...")
        # --- Input Validation and Setup ---
        if df is None or df.empty or not all(c in df.columns for c in ['game_id', 'game_date', 'home_team', 'away_team']):
            logger.warning("Input df empty or missing essential columns for rest features.")
            placeholder_cols = ['rest_days_home', 'games_last_7_days_home', 'games_last_14_days_home', 'rest_days_away', 'games_last_7_days_away', 'games_last_14_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage']
            temp_df = df.copy() if df is not None else pd.DataFrame(); [temp_df.setdefault(col, self.defaults.get(col.replace('_home','').replace('_away','').replace('_advantage',''), 0.0)) for col in placeholder_cols if col not in temp_df]; return temp_df

        df_copy = df.copy().reset_index(drop=True) # Work on a copy with clean index
        try:
            df_copy['game_date'] = pd.to_datetime(df_copy['game_date'], errors='coerce')
            if df_copy['game_date'].isnull().all(): raise ValueError("All game_date values invalid after conversion.")
        except Exception as e:
            logger.warning(f"Error processing game_date: {e}. Cannot calculate rest features accurately.")
            placeholder_cols = ['rest_days_home', 'prev_home_game_date', 'rest_days_away', 'prev_away_game_date', 'games_last_7_days_home', 'games_last_14_days_home', 'games_last_7_days_away', 'games_last_14_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage']
            [df_copy.setdefault(col, self.defaults.get(col.replace('_home','').replace('_away','').replace('_advantage','').replace('prev_','').replace('_date',''), 0.0) if 'days' in col or 'games' in col or 'advantage' in col else pd.NaT) for col in placeholder_cols if col not in df_copy]; return df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date'], errors='ignore')

        # --- Calculate Rest Days (Shift method) ---
        if not df_copy['game_date'].isnull().all():
             df_copy_sorted_home = df_copy.sort_values(['home_team', 'game_date'])
             df_copy['prev_home_game_date'] = df_copy_sorted_home.groupby('home_team')['game_date'].shift(1).reindex(df_copy.index)
             df_copy['rest_days_home'] = (df_copy['game_date'] - df_copy['prev_home_game_date']).dt.days.fillna(self.defaults['rest_days'])
             df_copy_sorted_away = df_copy.sort_values(['away_team', 'game_date'])
             df_copy['prev_away_game_date'] = df_copy_sorted_away.groupby('away_team')['game_date'].shift(1).reindex(df_copy.index)
             df_copy['rest_days_away'] = (df_copy['game_date'] - df_copy['prev_away_game_date']).dt.days.fillna(self.defaults['rest_days'])
        else:
             df_copy['rest_days_home'] = self.defaults['rest_days']; df_copy['rest_days_away'] = self.defaults['rest_days']
             df_copy['prev_home_game_date'] = pd.NaT; df_copy['prev_away_game_date'] = pd.NaT

        # --- Calculate Rolling Counts using Time Window and Key Merge ---
        try:
            df_copy = df_copy.sort_values(['game_date', 'game_id']).reset_index(drop=True)
            df_copy['temp_calc_id'] = df_copy.index # Unique ID for each row
                        # Create team-centric views with the temp_calc_id AND game_id
            home_view = df_copy[['temp_calc_id', 'game_date', 'game_id', 'home_team']].rename(columns={'home_team': 'team'}) # ADDED 'game_id'
            away_view = df_copy[['temp_calc_id', 'game_date', 'game_id', 'away_team']].rename(columns={'away_team': 'team'}) # ADDED 'game_id'
            team_game_log = pd.concat([home_view, away_view], ignore_index=True)
            team_game_log = team_game_log.sort_values(['team', 'game_date', 'temp_calc_id']).reset_index(drop=True)

            # Calculate rolling counts on the log, indexed by game_date
            team_game_log_indexed = team_game_log.set_index('game_date')
            # Use 'game_id' for count (this should work now)
            counts_7d_series = team_game_log_indexed.groupby('team')['game_id'].rolling('7D', closed='left').count()
            counts_14d_series = team_game_log_indexed.groupby('team')['game_id'].rolling('14D', closed='left').count()
            counts_7d_df = counts_7d_series.reset_index().rename(columns={'game_id': 'games_last_7d'})
            counts_14d_df = counts_14d_series.reset_index().rename(columns={'game_id': 'games_last_14d'})
            team_counts_merged = pd.merge(counts_7d_df, counts_14d_df, on=['team', 'game_date'], how='outer')
            # Merge counts back using left join to preserve all rows from team_game_log
            team_game_log_with_counts = pd.merge(team_game_log, team_counts_merged, on=['team', 'game_date'], how='left')
            # Keep only the entry corresponding to the specific game instance (temp_calc_id)
            # Drop duplicates based on temp_calc_id, keeping the latest entry which corresponds to the correct calculation for that game row
            final_counts = team_game_log_with_counts.sort_values(['team', 'game_date', 'temp_calc_id']).drop_duplicates(subset=['temp_calc_id'], keep='last')
            # --- Merge Counts Back to df_copy using temp_calc_id ---
            home_counts_to_merge = final_counts[final_counts['team'].isin(df_copy['home_team'].unique())][['temp_calc_id', 'games_last_7d', 'games_last_14d']] \
                                     .rename(columns={'games_last_7d': 'games_last_7_days_home', 'games_last_14d': 'games_last_14_days_home'})
            away_counts_to_merge = final_counts[final_counts['team'].isin(df_copy['away_team'].unique())][['temp_calc_id', 'games_last_7d', 'games_last_14d']] \
                                     .rename(columns={'games_last_7d': 'games_last_7_days_away', 'games_last_14d': 'games_last_14_days_away'})
            df_copy = pd.merge(df_copy, home_counts_to_merge, on='temp_calc_id', how='left')
            df_copy = pd.merge(df_copy, away_counts_to_merge, on='temp_calc_id', how='left')
        except Exception as rolling_e:
            logger.error(f"Error during time-based rolling count calculation/merge: {rolling_e}", exc_info=True)
            logger.warning("Rolling counts will be filled with 0 due to error.")
            df_copy['games_last_7_days_home'] = df_copy.get('games_last_7_days_home', 0)
            df_copy['games_last_14_days_home'] = df_copy.get('games_last_14_days_home', 0)
            df_copy['games_last_7_days_away'] = df_copy.get('games_last_7_days_away', 0)
            df_copy['games_last_14_days_away'] = df_copy.get('games_last_14_days_away', 0)

        # --- Fill NaNs & Finalize ---
        fill_cols = ['games_last_7_days_home', 'games_last_14_days_home', 'games_last_7_days_away', 'games_last_14_days_away']
        for col in fill_cols: df_copy[col] = df_copy[col].fillna(0).astype(int)
        df_copy['is_back_to_back_home'] = (df_copy['rest_days_home'] == 1).astype(int) if 'rest_days_home' in df_copy else 0
        df_copy['is_back_to_back_away'] = (df_copy['rest_days_away'] == 1).astype(int) if 'rest_days_away' in df_copy else 0
        df_copy['rest_advantage'] = df_copy.get('rest_days_home', self.defaults['rest_days']) - df_copy.get('rest_days_away', self.defaults['rest_days'])
        df_copy['schedule_advantage'] = df_copy['games_last_7_days_away'] - df_copy['games_last_7_days_home']
        df_copy = df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date', 'temp_calc_id'], errors='ignore')
        logger.debug("Finished adding rest features (vectorized, v5.1 - Time-Based w/ Merge).")
        return df_copy

    # --- add_matchup_history_features ---
    @profile_time
    def add_matchup_history_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame], max_games: int = 5) -> pd.DataFrame:
        """Adds historical head-to-head (H2H) matchup features."""
        logger.debug(f"Adding H2H features (last {max_games} games)...")
        if df is None or df.empty: return df
        result_df = df.copy()
        placeholder_cols = list(self._get_matchup_history_single("", "", pd.DataFrame(), 0).keys())
        if historical_df is None or historical_df.empty:
            logger.warning("Historical DataFrame is empty. Adding H2H placeholders.")
            for col in placeholder_cols: [result_df.setdefault(col, self.defaults.get(col.replace('matchup_',''), 0.0 if col != 'matchup_last_date' else pd.NaT))]
            return result_df # Return df with placeholders
        try:
            hist_df = historical_df.copy()
            # Ensure necessary columns exist and have correct types
            for col in ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']:
                 if col not in hist_df.columns: raise ValueError(f"H2H: Historical DF missing required column: {col}")
                 if col == 'game_date': hist_df[col] = pd.to_datetime(hist_df[col], errors='coerce').dt.tz_localize(None)
                 elif 'score' in col: hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
            hist_df = hist_df.dropna(subset=['game_date', 'home_score', 'away_score'])

            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date'])
            if result_df.empty: logger.warning("H2H: No valid rows in input df after date cleaning."); return df

            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            hist_df['home_team_norm'] = hist_df['home_team'].apply(self.normalize_team_name)
            hist_df['away_team_norm'] = hist_df['away_team'].apply(self.normalize_team_name)

            # Create matchup keys efficiently
            result_df['matchup_key'] = result_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            hist_df['matchup_key'] = hist_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)

            logger.debug("Grouping historical data for H2H lookup...")
            # Optimize lookup: Create a dictionary of matchup DataFrames
            hist_lookup = {key: group.copy() for key, group in hist_df.sort_values('game_date').groupby('matchup_key')}

            logger.debug("Applying H2H calculation per game...")
            # Apply function row-wise (can be slow for very large df)
            h2h_results = result_df.apply(
                 lambda game_to_predict: self._get_matchup_history_single(
                     game_to_predict['home_team_norm'],
                     game_to_predict['away_team_norm'],
                     hist_lookup.get(game_to_predict['matchup_key'], pd.DataFrame()), # Pass relevant history subset
                     max_games,
                     current_game_date=game_to_predict['game_date'] # Pass current date for filtering
                 ), axis=1
             )

            if not h2h_results.empty:
                results_df_h2h = pd.DataFrame(h2h_results.tolist(), index=result_df.index)
                result_df = result_df.join(results_df_h2h, how='left')
            else:
                logger.warning("No H2H matchup results generated.")

            # Fill NaNs and ensure types AFTER join
            logger.debug("Filling NaNs and ensuring types for H2H features...")
            for col in placeholder_cols:
                 if col == 'matchup_last_date':
                     result_df[col] = pd.to_datetime(result_df.get(col), errors='coerce') # Use get for safety
                 else:
                     default_key = col.replace('matchup_', '')
                     default = self.defaults.get(default_key, 0.0)
                     result_df[col] = pd.to_numeric(result_df.get(col), errors='coerce').fillna(default)
                     dtype = int if col in ['matchup_num_games', 'matchup_streak'] else float
                     try: result_df[col] = result_df[col].astype(dtype)
                     except (ValueError, TypeError): result_df[col] = default # Fallback on type conversion error

            result_df = result_df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')
        except Exception as e:
            logger.error(f"Error adding H2H features: {e}", exc_info=True)
            # Add placeholders if error
            for col in placeholder_cols: [result_df.setdefault(col, self.defaults.get(col.replace('matchup_',''), 0.0 if col != 'matchup_last_date' else pd.NaT))]

        logger.debug("Finished adding H2H features.")
        return result_df

    # --- add_season_context_features (Includes V2 robustness + Logging Fix) ---
    def add_season_context_features(self, df: pd.DataFrame, team_stats_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """REVISED (v2+): Merges team season statistics with diagnostic logging."""
        logger.info("Adding season context features (v2+)...")
        result_df = df.copy()
        placeholder_cols = [
            'home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for', 'away_season_avg_pts_for',
            'home_season_avg_pts_against', 'away_season_avg_pts_against', 'home_current_form', 'away_current_form',
            'season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff',
            'home_season_net_rating', 'away_season_net_rating', 'season_net_rating_diff'
        ]
        req_ts_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
        team_stats_available = False
        if team_stats_df is not None and not team_stats_df.empty:
            ts_df = team_stats_df.copy()
            # Validate required columns in team_stats_df
            missing_ts_cols = [col for col in req_ts_cols if col not in ts_df.columns]
            if missing_ts_cols:
                logger.warning(f"Team stats DF missing columns: {missing_ts_cols}. Filling with defaults before merge.")
                for col in missing_ts_cols:
                    default = 'N/A' if col == 'current_form' else (0.5 if 'percentage' in col else 0.0)
                    ts_df[col] = default  # Use standard assignment
            team_stats_available = True
        else:
            logger.warning("`team_stats_df` is empty/None. Adding placeholders.")
        if 'game_date' not in result_df.columns or pd.isna(result_df['game_date']).all():
            logger.error("`game_date` missing/all NaN. Cannot merge.")
            team_stats_available = False

        if not team_stats_available:
            for col in placeholder_cols:
                default_key = col.replace('home_', '').replace('away_', '').replace('season_', '')
                default = self.defaults.get(default_key, 0.0 if 'form' not in col else 'N/A')
                if col not in result_df.columns:
                    result_df[col] = default
                else:
                    result_df[col] = result_df[col].fillna(default)
            for col in placeholder_cols:
                if 'form' not in col:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
                else:
                    result_df[col] = result_df[col].astype(str)
            return result_df

        # --- Main Merge Logic ---
        try:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date'])
            if result_df.empty:
                raise ValueError("No valid game dates.")
            result_df['season'] = result_df['game_date'].apply(self._determine_season)
            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name)
            result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            ts_df['team_name_norm'] = ts_df['team_name'].apply(self.normalize_team_name)
            ts_merge = ts_df[['team_name_norm', 'season'] + [c for c in req_ts_cols if c not in ['team_name', 'season']]].copy()
            for col in ['wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all']:
                ts_merge[col] = pd.to_numeric(ts_merge[col], errors='coerce')
            ts_merge['season'] = ts_merge['season'].astype(str)
            ts_merge['merge_key'] = ts_merge['team_name_norm'] + "_" + ts_merge['season']
            ts_merge = ts_merge.drop(columns=['team_name_norm', 'season']).drop_duplicates(subset=['merge_key'], keep='last')
            result_df['merge_key_home'] = result_df['home_team_norm'] + "_" + result_df['season']
            result_df['merge_key_away'] = result_df['away_team_norm'] + "_" + result_df['season']
            home_rename = {
                'wins_all_percentage': 'home_season_win_pct',
                'points_for_avg_all': 'home_season_avg_pts_for',
                'points_against_avg_all': 'home_season_avg_pts_against',
                'current_form': 'home_current_form'
            }
            away_rename = {
                'wins_all_percentage': 'away_season_win_pct',
                'points_for_avg_all': 'away_season_avg_pts_for',
                'points_against_avg_all': 'away_season_avg_pts_against',
                'current_form': 'away_current_form'
            }
            logger.debug("Attempting to merge season stats...")
            result_df = pd.merge(
                result_df,
                ts_merge.rename(columns=home_rename),
                how='left',
                left_on='merge_key_home',
                right_on='merge_key',
                indicator='_merge_home'
            )
            result_df = pd.merge(
                result_df,
                ts_merge.rename(columns=away_rename),
                how='left',
                left_on='merge_key_away',
                right_on='merge_key',
                suffixes=('', '_away_dup'),
                indicator='_merge_away'
            )
            # --- Diagnostic Logging ---
            if '_merge_home' in result_df.columns and '_merge_away' in result_df.columns:
                home_merge_success = (result_df['_merge_home'] == 'both').mean()
                away_merge_success = (result_df['_merge_away'] == 'both').mean()
                logger.info(f"Season stats merge success rate: Home={home_merge_success:.1%}, Away={away_merge_success:.1%}")
                if home_merge_success < 0.9 or away_merge_success < 0.9:
                    logger.warning("Low merge success rate for season stats. Check team name normalization and season keys.")
                    failed_home_keys = result_df.loc[result_df['_merge_home'] == 'left_only', 'merge_key_home'].unique()
                    failed_away_keys = result_df.loc[result_df['_merge_away'] == 'left_only', 'merge_key_away'].unique()
                    if len(failed_home_keys) > 0:
                        logger.warning(f" Example failed HOME merge keys: {failed_home_keys[:5]}")
                    if len(failed_away_keys) > 0:
                        logger.warning(f" Example failed AWAY merge keys: {failed_away_keys[:5]}")
                result_df = result_df.drop(columns=['_merge_home', '_merge_away'], errors='ignore')
            else:
                logger.warning("Merge indicator columns not found. Cannot report exact merge success rate.")
            # Drop intermediate columns
            result_df = result_df.drop(columns=[col for col in result_df.columns if '_away_dup' in col or col == 'merge_key'], errors='ignore')
            result_df = result_df.drop(columns=['season', 'home_team_norm', 'away_team_norm', 'merge_key_home', 'merge_key_away'], errors='ignore')
        except Exception as merge_e:
            logger.error(f"Error during season context merge/processing: {merge_e}", exc_info=True)

        # --- Fill NaNs & Calculate Differentials ---
        logger.debug("Filling NaNs and calculating season diffs...")
        for col in placeholder_cols:
            default_key = col.replace('home_', '').replace('away_', '').replace('season_', '')
            default = self.defaults.get(default_key, 0.0 if 'form' not in col else 'N/A')
            if 'diff' in col:
                default = 0.0
            if col not in result_df.columns:
                result_df[col] = default
            else:
                result_df[col] = result_df[col].fillna(default)
            if 'form' not in col:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
            else:
                result_df[col] = result_df[col].astype(str)
        result_df['season_win_pct_diff'] = result_df['home_season_win_pct'] - result_df['away_season_win_pct']
        result_df['season_pts_for_diff'] = result_df['home_season_avg_pts_for'] - result_df['away_season_avg_pts_for']
        result_df['season_pts_against_diff'] = result_df['home_season_avg_pts_against'] - result_df['away_season_avg_pts_against']
        result_df['home_season_net_rating'] = result_df['home_season_avg_pts_for'] - result_df['home_season_avg_pts_against']
        result_df['away_season_net_rating'] = result_df['away_season_avg_pts_for'] - result_df['away_season_avg_pts_against']
        result_df['season_net_rating_diff'] = result_df['home_season_net_rating'] - result_df['away_season_net_rating']
        logger.info("Finished adding season context features.")
        return result_df

    # --- add_form_string_features ---
    @profile_time
    def add_form_string_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """REVISED (v2): Adds features derived from team form strings with improved logging."""
        logger.debug("Adding features derived from form strings (v2)...")
        result_df = df.copy()
        form_metric_keys = list(self._extract_form_metrics_single("").keys())
        home_cols = [f'home_{key}' for key in form_metric_keys]; away_cols = [f'away_{key}' for key in form_metric_keys]
        diff_cols = ['form_win_pct_diff', 'streak_advantage', 'momentum_diff']; placeholder_cols = home_cols + away_cols + diff_cols
        home_form_col = 'home_current_form'; away_form_col = 'away_current_form'
        if home_form_col not in result_df.columns or away_form_col not in result_df.columns:
            logger.warning(f"Missing input form columns. Adding default placeholders.")
        else:
            try:
                valid_home_forms = result_df[home_form_col].fillna('').astype(str).replace('N/A', '').str.len() > 0
                valid_away_forms = result_df[away_form_col].fillna('').astype(str).replace('N/A', '').str.len() > 0
                logger.info(f"Processing form strings: Found valid Home forms in {valid_home_forms.mean():.1%} of rows, Away forms in {valid_away_forms.mean():.1%} of rows.")
                result_df[home_form_col] = result_df[home_form_col].fillna('').astype(str); result_df[away_form_col] = result_df[away_form_col].fillna('').astype(str)
                home_metrics = result_df[home_form_col].apply(self._extract_form_metrics_single)
                away_metrics = result_df[away_form_col].apply(self._extract_form_metrics_single)
                home_form_df = pd.DataFrame(home_metrics.tolist(), index=result_df.index).add_prefix('home_')
                away_form_df = pd.DataFrame(away_metrics.tolist(), index=result_df.index).add_prefix('away_')
                result_df = result_df.join(home_form_df); result_df = result_df.join(away_form_df)
                result_df['form_win_pct_diff'] = result_df.get('home_form_win_pct', self.defaults['form_win_pct']) - result_df.get('away_form_win_pct', self.defaults['form_win_pct'])
                result_df['streak_advantage'] = result_df.get('home_current_streak', self.defaults['current_streak']) - result_df.get('away_current_streak', self.defaults['current_streak'])
                result_df['momentum_diff'] = result_df.get('home_momentum_direction', self.defaults['momentum_direction']) - result_df.get('away_momentum_direction', self.defaults['momentum_direction'])
            except Exception as e: logger.error(f"Error processing form string features: {e}", exc_info=True)
        logger.debug("Finalizing form feature types...")
        for col in placeholder_cols:
            default_key = col.replace('home_','').replace('away_','').replace('_diff','').replace('_advantage','')
            default = self.defaults.get(default_key, 0.0)
            if col not in result_df.columns:
                result_df[col] = default
            else:
                result_df[col] = result_df[col].fillna(default)
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
            if 'streak' in col and 'advantage' not in col: result_df[col] = result_df[col].round().astype(int)
        logger.debug("Finished adding form string features.")
        return result_df

    # --- add_betting_odds ---
    @profile_time
    def add_betting_odds(self, df: pd.DataFrame, betting_odds_data: Optional[Union[pd.DataFrame, Dict[Any, Dict]]] = None) -> pd.DataFrame:
        """Merges betting odds (spread, over/under) onto the features DataFrame."""
        logger.debug("Adding betting odds features (if provided)...")
        result_df = df.copy()
        odds_cols = ['vegas_home_spread', 'vegas_over_under']
        default_spread = self.defaults['vegas_home_spread']
        default_ou = self.defaults['vegas_over_under']
        
        # Ensure the odds columns exist and fill any NaNs with defaults.
        for col in odds_cols:
            default = default_spread if col == 'vegas_home_spread' else default_ou
            if col not in result_df.columns:
                result_df[col] = default
            else:
                result_df[col] = result_df[col].fillna(default)
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
        
        # If no betting odds data is provided, return with defaults.
        if betting_odds_data is None:
            logger.debug("No betting odds data provided. Filling odds with defaults.")
            return result_df
        
        try:
            if isinstance(betting_odds_data, pd.DataFrame):
                if 'game_id' not in betting_odds_data.columns:
                    raise ValueError("Odds DF missing 'game_id'.")
                odds_to_merge = [col for col in odds_cols if col in betting_odds_data.columns]
                if not odds_to_merge:
                    raise ValueError("Odds DF missing relevant columns.")
                logger.debug(f"Merging odds columns: {odds_to_merge} from DataFrame.")
                odds_df = betting_odds_data[['game_id'] + odds_to_merge].copy()
                result_df['game_id_merge'] = result_df['game_id'].astype(str)
                odds_df['game_id_merge'] = odds_df['game_id'].astype(str)
                result_df = pd.merge(
                    result_df, 
                    odds_df.drop(columns=['game_id']), 
                    on='game_id_merge', 
                    how='left', 
                    suffixes=('', '_odds')
                )
                for col in odds_to_merge:
                    merged_col = f"{col}_odds"
                    if merged_col in result_df.columns:
                        result_df[col] = result_df[merged_col].combine_first(result_df[col])
                        result_df = result_df.drop(columns=[merged_col])
                result_df = result_df.drop(columns=['game_id_merge'], errors='ignore')
            elif isinstance(betting_odds_data, dict):
                logger.debug("Extracting odds from dictionary.")
                def get_odds(game_id, odds_dict):
                    odds = odds_dict.get(game_id, odds_dict.get(str(game_id)))
                    if isinstance(odds, dict):
                        return pd.Series({
                            'vegas_home_spread': odds.get('spread', {}).get('home_line', np.nan),
                            'vegas_over_under': odds.get('total', {}).get('line', np.nan)
                        })
                    return pd.Series({'vegas_home_spread': np.nan, 'vegas_over_under': np.nan})
                odds_extracted = result_df['game_id'].apply(lambda gid: get_odds(gid, betting_odds_data))
                result_df['vegas_home_spread'] = result_df['vegas_home_spread'].fillna(odds_extracted['vegas_home_spread'])
                result_df['vegas_over_under'] = result_df['vegas_over_under'].fillna(odds_extracted['vegas_over_under'])
            else:
                raise TypeError("Unsupported format for betting_odds_data.")
            
            # Final fillna and type conversion for odds columns.
            for col in odds_cols:
                default = default_spread if col == 'vegas_home_spread' else default_ou
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
            logger.debug("Finished adding betting odds features.")
        except Exception as e:
            logger.error(f"Error adding betting odds: {e}. Filling with defaults.", exc_info=True)
            for col in odds_cols:
                default = default_spread if col == 'vegas_home_spread' else default_ou
                result_df[col] = pd.to_numeric(result_df.get(col), errors='coerce').fillna(default)
        return result_df


    # --------------------------------------------------------------------------
    # Private Helper Methods
    # --------------------------------------------------------------------------
    def _get_matchup_history_single(self, home_team_norm: str, away_team_norm: str, historical_subset: pd.DataFrame, max_games: int = 5, current_game_date: Optional[pd.Timestamp] = None, **kwargs) -> Dict[str, Any]:
        """(Private Helper) Calculates H2H stats for a single matchup, excluding current game."""
        # ... (defaults definition - no change) ...
        default_result = {'matchup_num_games': 0, 'matchup_avg_point_diff': 0.0, 'matchup_home_win_pct': 0.5, 'matchup_avg_total_score': 230.0, 'matchup_avg_home_score': 115.0, 'matchup_avg_away_score': 115.0, 'matchup_last_date': pd.NaT, 'matchup_streak': 0}
        if historical_subset.empty or max_games <= 0 or current_game_date is None: return default_result
        # Filter to games strictly BEFORE the current game date
        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        if past_games_df.empty: return default_result
        try:
            # Sort by date desc to get recent games easily
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            if recent_matchups.empty: return default_result
            # --- Calculate Stats (same logic as before, applied to 'recent_matchups') ---
            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []
            home_persp_wins, current_streak, last_winner_norm = 0, 0, None
            # Iterate over the *date-sorted* recent matchups
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows(): # Iterate chronologically for streak
                home_score = game.get('home_score'); away_score = game.get('away_score') # Already numeric
                if pd.isna(home_score) or pd.isna(away_score): continue # Skip if scores missing
                game_home_norm = game.get('home_team_norm'); game_away_norm = game.get('away_team_norm')
                if game_home_norm == home_team_norm: diff, won, h_persp, a_persp = home_score - away_score, (home_score > away_score), home_score, away_score
                elif game_away_norm == home_team_norm: diff, won, h_persp, a_persp = away_score - home_score, (away_score > home_score), away_score, home_score
                else: continue # Should not happen if matchup_key filter worked
                diffs.append(diff); total_scores.append(home_score + away_score); home_persp_scores.append(h_persp); away_persp_scores.append(a_persp)
                if won: home_persp_wins += 1
                winner = home_team_norm if won else away_team_norm
                if last_winner_norm is None: last_winner_norm = winner; current_streak = 1 if won else -1
                elif winner == last_winner_norm: current_streak += (1 if won else -1)
                else: last_winner_norm = winner; current_streak = 1 if won else -1 # Reset streak on change
            num_games = len(diffs)
            if num_games == 0: return default_result
            return {'matchup_num_games': num_games, 'matchup_avg_point_diff': np.mean(diffs), 'matchup_home_win_pct': home_persp_wins / num_games, 'matchup_avg_total_score': np.mean(total_scores), 'matchup_avg_home_score': np.mean(home_persp_scores), 'matchup_avg_away_score': np.mean(away_persp_scores), 'matchup_last_date': recent_matchups['game_date'].max(), 'matchup_streak': int(current_streak)}
        except Exception as e: logger.error(f"Error in _get_matchup_history_single ({home_team_norm} vs {away_team_norm}): {e}", exc_info=True); return default_result

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        """(Private Helper) Extract metrics from a single form string."""
        # ... (no changes needed, seems robust) ...
        defaults = {'form_win_pct': self.defaults['form_win_pct'], 'current_streak': self.defaults['current_streak'], 'momentum_direction': self.defaults['momentum_direction']}
        if not form_string or pd.isna(form_string) or not isinstance(form_string, str): return defaults
        form_string = form_string.upper().strip().replace('-','').replace('?',''); form_len = len(form_string)
        if form_len == 0 or form_string == 'N/A': return defaults
        wins = form_string.count('W'); form_win_pct = wins / form_len; current_streak = 0
        if form_len > 0:
            streak_char = form_string[-1] # Check streak from MOST RECENT game
            streak_count = 0
            for char in reversed(form_string):
                 if char == streak_char: streak_count += 1
                 else: break
            current_streak = streak_count if streak_char == 'W' else -streak_count
        momentum_direction = 0.0
        if form_len >= 4:
             split = form_len // 2; recent_half, older_half = form_string[-split:], form_string[:-split] # Correct slicing for recent/older
             len_r, len_o = len(recent_half), len(older_half)
             if len_r > 0 and len_o > 0:
                 pct_r = recent_half.count('W') / len_r; pct_o = older_half.count('W') / len_o
                 if pct_r > pct_o: momentum_direction = 1.0
                 elif pct_r < pct_o: momentum_direction = -1.0
        return {'form_win_pct': form_win_pct, 'current_streak': int(current_streak), 'momentum_direction': momentum_direction}

    # --------------------------------------------------------------------------
    # Main Orchestration Method
    # --------------------------------------------------------------------------
    # @profile_time # Optional profiling for the whole pipeline
    def generate_all_features(self, df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None,
                              betting_odds_data: Optional[Union[pd.DataFrame, Dict[Any, Dict]]] = None,
                              rolling_windows: List[int] = [5, 10],
                              h2h_window: int = 5) -> pd.DataFrame:
        """
        Applies all feature engineering steps in sequence. Uses REVISED methods.
        Includes debug output generation if self.debug is True.
        """
        logger.info("Starting comprehensive feature generation pipeline...")
        start_time_total = time.time()
        # --- Input Validation ---
        if df is None or df.empty: logger.error("Input DataFrame `df` is empty."); return pd.DataFrame()
        essential_cols = ['game_id', 'game_date', 'home_team', 'away_team']
        if not all(col in df.columns for col in essential_cols): logger.error(f"Input `df` missing essential columns."); return pd.DataFrame()

        # --- Prepare DataFrames ---
        target_games_df = df.copy() # This is the df for which we want final features
        try:
            target_games_df['game_date'] = pd.to_datetime(target_games_df['game_date'], errors='coerce').dt.tz_localize(None)
            target_games_df = target_games_df.dropna(subset=['game_date'])
            if target_games_df.empty: raise ValueError("All target game dates invalid.")
            logger.info(f"Target games date range: {target_games_df['game_date'].min().date()} to {target_games_df['game_date'].max().date()}")
        except Exception as e: logger.error(f"Error processing game_date in target df: {e}"); return pd.DataFrame()

        # Create base_calc_df: Combine historical context with target games for calculations
        base_calc_df = None
        hist_df_processed = None # Keep track if history was processed
        if historical_games_df is not None and not historical_games_df.empty:
            logger.debug(f"Processing {len(historical_games_df)} historical games for context...")
            try:
                hist_df_processed = historical_games_df.copy()
                hist_df_processed['game_date'] = pd.to_datetime(hist_df_processed['game_date'], errors='coerce').dt.tz_localize(None)
                hist_df_processed = hist_df_processed.dropna(subset=['game_date'])

                # Align columns before concat - ensure all needed columns exist in both
                hist_cols = set(hist_df_processed.columns); target_cols = set(target_games_df.columns)
                all_cols_union = list(hist_cols.union(target_cols))
                for col in all_cols_union:
                    if col not in hist_df_processed.columns:
                        hist_df_processed[col] = np.nan
                    if col not in target_games_df.columns:
                        target_games_df[col] = np.nan

                logger.info(f"Combining {len(hist_df_processed)} historical and {len(target_games_df)} target games for calculation base...")
                base_calc_df = pd.concat(
                    [hist_df_processed[all_cols_union], target_games_df[all_cols_union]], # Use consistent columns
                    ignore_index=True
                ).sort_values(by=['game_date', 'game_id'], kind='mergesort').reset_index(drop=True)

                # *** Deduplication Step (Crucial Fix) ***
                initial_rows = len(base_calc_df)
                if 'game_id' in base_calc_df.columns:
                    base_calc_df['game_id'] = base_calc_df['game_id'].astype(str)
                    base_calc_df = base_calc_df.drop_duplicates(subset=['game_id'], keep='last')
                    rows_dropped = initial_rows - len(base_calc_df)
                    if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} duplicate game_id rows from base_calc_df, keeping last occurrence.")
                else: raise ValueError("game_id column missing after concat, cannot deduplicate.")
                # *** End Deduplication Step ***

                logger.info(f"Created DEDUPLICATED base_calc_df with {len(base_calc_df)} rows total.")

            except Exception as e:
                logger.error(f"Error combining historical/target data: {e}. Proceeding with target data only.", exc_info=True)
                base_calc_df = target_games_df.copy().sort_values(['game_date', 'game_id'], kind='mergesort').reset_index(drop=True)
        else:
            logger.warning("No historical_games_df provided. Rolling/H2H/Rest features will use defaults or be limited.")
            base_calc_df = target_games_df.copy().sort_values(['game_date', 'game_id'], kind='mergesort').reset_index(drop=True) # Ensure base_calc_df is defined

        # Final checks on base_calc_df
        if base_calc_df is None or base_calc_df.empty: logger.error("base_calc_df is empty before feature steps."); return pd.DataFrame()
        if 'game_date' not in base_calc_df.columns: logger.error("Critical: 'game_date' missing from base_calc_df."); return pd.DataFrame()

        # --- Feature Generation Sequence (Applied to base_calc_df) ---
        try:
            # Note: Order matters if features depend on previously calculated ones
            logger.info("Step 1/9: Adding intra-game momentum features...") # Mostly for context/post-game
            base_calc_df = self.add_intra_game_momentum(base_calc_df)

            logger.info("Step 2/9: Integrating advanced metrics (v2)...") # Game-level stats
            base_calc_df = self.integrate_advanced_features(base_calc_df)

            logger.info("Step 3/9: Adding rolling features (v2)...") # Requires prior games' advanced stats
            base_calc_df = self.add_rolling_features(base_calc_df, window_sizes=rolling_windows)

            logger.info("Step 4/9: Adding rest & schedule density features (v5.1)...") # Requires game_date sequence
            base_calc_df = self.add_rest_features_vectorized(base_calc_df) # Use the time-based robust version

            logger.info("Step 5/9: Adding H2H matchup features...") # Requires historical context
            # Pass the *original* processed history, not base_calc_df, for H2H lookup
            base_calc_df = self.add_matchup_history_features(base_calc_df, hist_df_processed, max_games=h2h_window)

            logger.info("Step 6/9: Adding season context features (v2+)...") # Requires team_stats_df
            base_calc_df = self.add_season_context_features(base_calc_df, team_stats_df)

            logger.info("Step 7/9: Adding form string features (v2)...") # Requires season context features
            base_calc_df = self.add_form_string_features(base_calc_df)

            logger.info("Step 8/9: Adding betting odds features...") # Requires betting_odds_data
            base_calc_df = self.add_betting_odds(base_calc_df, betting_odds_data)

            # --- Final Filtering Step (Corrected Logic) ---
            logger.info("Step 9/9: Filtering back to target games...")
            original_game_ids = df['game_id'].astype(str).unique() # Use original input df IDs
            num_unique_original_ids = len(original_game_ids)
            logger.debug(f"Original Target Game IDs (Unique): {original_game_ids[:10]}... ({num_unique_original_ids} total)")

            if 'game_id' not in base_calc_df.columns: raise ValueError("'game_id' column lost during feature generation.")
            base_calc_df['game_id'] = base_calc_df['game_id'].astype(str)

            # Filter base_calc_df (which now has all features) to only the original target game_ids
            final_df = base_calc_df[base_calc_df['game_id'].isin(original_game_ids)].copy()
            logger.info(f"Shape of final_df after filtering: {final_df.shape}")

            # Validate final row count against *unique* original IDs
            if len(final_df) != num_unique_original_ids:
                 logger.error(f"Row count mismatch after final filtering! Expected unique IDs: {num_unique_original_ids}, Final Output Rows: {len(final_df)}.")
                 # Log details for debugging
                 unique_final_ids = final_df['game_id'].nunique()
                 logger.error(f"Unique game_ids in final_df: {unique_final_ids}")
                 if len(final_df) != unique_final_ids:
                      logger.error(">>> CRITICAL: Duplicate game_ids found in FINAL output! Check base_calc_df generation/deduplication. <<<")
                 # Decide whether to return potentially faulty data or empty
                 logger.error("Returning empty DataFrame due to final filtering mismatch.")
                 return pd.DataFrame() # Return empty on mismatch
            else:
                 logger.info("Final filtering successful. Row count matches unique original target games.")

        except Exception as e:
            logger.error(f"Critical error during feature generation pipeline steps: {e}", exc_info=True)
            return pd.DataFrame() # Return empty on any pipeline error

        total_time = time.time() - start_time_total
        logger.info(f"Feature generation pipeline complete for {len(final_df)} games in {total_time:.2f}s.")
        return final_df.reset_index(drop=True) # Return the final features for target games

    # --------------------------------------------------------------------------
    # Private Helper Methods
    # --------------------------------------------------------------------------
    def _get_matchup_history_single(self, home_team_norm: str, away_team_norm: str, historical_subset: pd.DataFrame, max_games: int = 5, current_game_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """(Private Helper) Calculates H2H stats for a single matchup, excluding current game."""
        # Define defaults within the function
        default_result = {
            'matchup_num_games': self.defaults['matchup_num_games'],
            'matchup_avg_point_diff': self.defaults['matchup_avg_point_diff'],
            'matchup_home_win_pct': self.defaults['matchup_home_win_pct'],
            'matchup_avg_total_score': self.defaults['matchup_avg_total_score'],
            'matchup_avg_home_score': self.defaults['matchup_avg_home_score'],
            'matchup_avg_away_score': self.defaults['matchup_avg_away_score'],
            'matchup_last_date': pd.NaT,
            'matchup_streak': self.defaults['matchup_streak']
        }
        # Filter to games strictly BEFORE the current game date
        # Ensure current_game_date is provided and valid
        if historical_subset.empty or max_games <= 0 or pd.isna(current_game_date):
            return default_result

        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        if past_games_df.empty: return default_result

        try:
            # Sort by date desc to get recent games easily, then take top N
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            if recent_matchups.empty: return default_result

            # Ensure scores are numeric
            recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce')
            recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
            recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score'])
            if recent_matchups.empty: return default_result

            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []
            home_persp_wins, current_streak, last_winner_norm = 0, 0, None

            # Iterate chronologically to calculate streak correctly
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
                home_score = game['home_score']; away_score = game['away_score']
                game_home_norm = game['home_team_norm']; game_away_norm = game['away_team_norm']

                # Determine point diff and winner from the perspective of 'home_team_norm'
                if game_home_norm == home_team_norm:
                    diff, won = home_score - away_score, (home_score > away_score)
                    h_persp, a_persp = home_score, away_score
                elif game_away_norm == home_team_norm: # If the target 'home' team was away in this historical game
                    diff, won = away_score - home_score, (away_score > home_score)
                    h_persp, a_persp = away_score, home_score
                else: continue # Should not happen due to matchup_key grouping

                diffs.append(diff); total_scores.append(home_score + away_score)
                home_persp_scores.append(h_persp); away_persp_scores.append(a_persp)
                if won: home_persp_wins += 1

                # Calculate streak
                winner_norm = home_team_norm if won else away_team_norm
                if last_winner_norm is None: # First game in the series (chronologically)
                    last_winner_norm = winner_norm
                    current_streak = 1 if won else -1
                elif winner_norm == last_winner_norm: # Streak continues
                    current_streak += (1 if won else -1)
                else: # Streak broken
                    last_winner_norm = winner_norm
                    current_streak = 1 if won else -1

            num_games = len(diffs)
            if num_games == 0: return default_result

            # Calculate final stats
            final_stats = {
                'matchup_num_games': num_games,
                'matchup_avg_point_diff': np.mean(diffs) if diffs else default_result['matchup_avg_point_diff'],
                'matchup_home_win_pct': (home_persp_wins / num_games) if num_games > 0 else default_result['matchup_home_win_pct'],
                'matchup_avg_total_score': np.mean(total_scores) if total_scores else default_result['matchup_avg_total_score'],
                'matchup_avg_home_score': np.mean(home_persp_scores) if home_persp_scores else default_result['matchup_avg_home_score'],
                'matchup_avg_away_score': np.mean(away_persp_scores) if away_persp_scores else default_result['matchup_avg_away_score'],
                'matchup_last_date': recent_matchups['game_date'].max(), # Get max date from the head(N) set
                'matchup_streak': int(current_streak)
            }
            return final_stats

        except Exception as e: logger.error(f"Error in _get_matchup_history_single ({home_team_norm} vs {away_team_norm}): {e}", exc_info=True); return default_result

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        """(Private Helper) Extract metrics from a single form string."""
        defaults = {'form_win_pct': self.defaults['form_win_pct'], 'current_streak': self.defaults['current_streak'], 'momentum_direction': self.defaults['momentum_direction']}
        if not form_string or pd.isna(form_string) or not isinstance(form_string, str): return defaults
        form_string = form_string.upper().strip().replace('-', '').replace('?', ''); form_len = len(form_string)
        if form_len == 0 or form_string == 'N/A': return defaults
        wins = form_string.count('W'); form_win_pct = wins / form_len; current_streak = 0
        if form_len > 0:
             streak_char = form_string[-1] # Check streak from MOST RECENT game (last char)
             streak_count = 0
             for char in reversed(form_string): # Iterate backwards
                 if char == streak_char: streak_count += 1
                 else: break
             current_streak = streak_count if streak_char == 'W' else -streak_count
        momentum_direction = 0.0
        if form_len >= 4: # Need at least 4 games to compare halves
             # Ensure split handles odd lengths reasonably (integer division)
             split_point = form_len // 2
             recent_half, older_half = form_string[-split_point:], form_string[:-split_point] # Slice from end for recent
             len_r, len_o = len(recent_half), len(older_half)
             if len_r > 0 and len_o > 0: # Check both halves exist
                 pct_r = recent_half.count('W') / len_r; pct_o = older_half.count('W') / len_o
                 if pct_r > pct_o: momentum_direction = 1.0
                 elif pct_r < pct_o: momentum_direction = -1.0
        return {'form_win_pct': form_win_pct, 'current_streak': int(current_streak), 'momentum_direction': momentum_direction}

# --- Example Usage (Conceptual - Keep outside class) ---
# ... (example usage block can remain the same) ...
if __name__ == '__main__':
    pass # Keep example logic if needed for standalone testing