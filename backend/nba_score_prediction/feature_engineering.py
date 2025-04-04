# feature_engineering.py - Unified Feature Engineering for NBA Score Prediction
"""
NBAFeatureEngine - Unified module for NBA score prediction feature engineering.
Handles the creation of various features for NBA game prediction models.
CLEANED VERSION: Removed detailed debug file/plot generation.
"""
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
from functools import wraps, lru_cache
import functools
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# --- Matplotlib and Seaborn (Optional - only needed if plotting is ever re-enabled) ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # print("WARNING: matplotlib or seaborn not found. Debug plots will not be generated.") # Keep commented unless needed

# Configure logger for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

EPSILON = 1e-6

# --- Define Output Directory (Still needed for potential future use) ---
try:
    SCRIPT_DIR_FE = Path(__file__).resolve().parent
    REPORTS_DIR_FE = SCRIPT_DIR_FE.parent.parent / 'reports'
except NameError:
    REPORTS_DIR_FE = Path('./reports')

FEATURE_DEBUG_DIR = REPORTS_DIR_FE / "feature_debug"
# --- End Directory Definition ---


# -------------------- NBAFeatureEngine Class --------------------
class NBAFeatureEngine:
    """Core class for NBA feature engineering."""

    def __init__(self, supabase_client: Optional[Any] = None, debug: bool = False):
        """ Initialize the feature engine. """
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
            # FEATURE_DEBUG_DIR.mkdir(parents=True, exist_ok=True) # No need to create if not saving files
            logger.info("DEBUG mode enabled for detailed logging.")
        else:
             logger.setLevel(logging.INFO)

        self.supabase_client = supabase_client
        logger.debug("NBAFeatureEngine Initialized.")

        # --- Default Values ---
        self.defaults = {
            'win_pct': 0.5, 'avg_pts_for': 115.0, 'avg_pts_against': 115.0,
            'home_advantage': 3.0, 'offensive_rating': 115.0, 'defensive_rating': 115.0,
            'pace': 100.0, 'estimated_possessions': 95.0, # Default possessions set to 95
            'oreb_pct': 0.23, 'dreb_pct': 0.77, 'trb_pct': 0.50, 'tov_rate': 13.0,
            'efg_pct': 0.54, 'ft_rate': 0.20, 'score_for': 115.0, 'score_against': 115.0,
            'off_rating': 115.0, 'def_rating': 115.0, 'net_rating': 0.0, # Added default
            'momentum_ewma': 0.0,
            'matchup_avg_point_diff': 0.0, 'matchup_streak': 0, 'form_win_pct': 0.5,
            'current_streak': 0, 'momentum_direction': 0.0, 'rest_days': 3.0,
            'score_for_std': 10.0, 'score_against_std': 10.0, 'off_rating_std': 10.0,
            'def_rating_std': 10.0, 'net_rating_std': 10.0, # Added default
            'pace_std': 5.0, 'efg_pct_std': 0.05, 'tov_rate_std': 3.0,
            'trb_pct_std': 0.05, 'oreb_pct_std': 0.05, 'dreb_pct_std': 0.05,
            'ft_rate_std': 0.05, # Added default
            'momentum_ewma_std': 5.0, 'matchup_num_games': 0,
            'matchup_avg_total_score': 230.0, 'matchup_avg_home_score': 115.0,
            'matchup_avg_away_score': 115.0, 'matchup_home_win_pct': 0.5,
            'vegas_home_spread': 0.0, 'vegas_over_under': 230.0,
            'games_last_7_days_home': 2, 'games_last_14_days_home': 4,
            'games_last_7_days_away': 2, 'games_last_14_days_away': 4,
        }
        self.league_averages = {
            'score': self.defaults['avg_pts_for'],
            'quarter_scores': {1: 28.5, 2: 28.5, 3: 28.0, 4: 29.0}
        }

    # --- Logging and Utility Methods ---
    @staticmethod
    def profile_time(func=None, debug_mode=None):
        # ... (Keep profiler code as is) ...
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = f(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                message = f"{f.__name__} executed in {execution_time:.4f} seconds"
                is_debug_instance = False
                if args and isinstance(args[0], NBAFeatureEngine): is_debug_instance = args[0].debug
                final_debug_mode = debug_mode if debug_mode is not None else is_debug_instance
                if final_debug_mode: logger.debug(f"[Profiler] {message}")
                return result
            return wrapper
        return decorator if func is None else decorator(func)

    @lru_cache(maxsize=512)
    def normalize_team_name(self, team_name: Optional[str]) -> str:
        # ... (Keep normalize_team_name code as is) ...
        if not isinstance(team_name, str): return "Unknown"
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
            "la":"lakers",
            "east": "east", "west": "west", "team lebron": "other_team", "team durant":"other_team",
            "chuck’s global stars": "other_team", "shaq’s ogs": "other_team",
            "kenny’s young stars": "other_team", "candace’s rising stars": "other_team",
        }
        if team_lower in mapping: return mapping[team_lower]
        for name, norm in mapping.items():
            if len(team_lower) > 3 and team_lower in name: return norm
            if len(name) > 3 and name in team_lower: return norm
        logger.warning(f"Team name '{team_name}' normalized to '{team_lower}' - no mapping found!")
        return team_lower

    def _determine_season(self, game_date: pd.Timestamp) -> str:
        # ... (Keep _determine_season code as is) ...
        if pd.isna(game_date): logger.warning("Missing game_date."); return "Unknown_Season"
        year = game_date.year; month = game_date.month
        start_year = year if month >= 9 else year - 1
        return f"{start_year}-{start_year + 1}"

    # --- Feature Calculation Methods ---

    @profile_time
    def add_intra_game_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (Keep momentum code as is) ...
        logger.debug("Adding intra-game momentum features...")
        if df is None or df.empty: return df
        result_df = df.copy()
        qtr_cols = [f'{loc}_q{i}' for loc in ['home', 'away'] for i in range(1, 5)]
        missing_cols = [col for col in qtr_cols if col not in result_df.columns]
        if missing_cols: logger.warning(f"Momentum: Missing quarter cols: {missing_cols}. Filling with 0.")
        for col in qtr_cols: result_df[col] = pd.to_numeric(result_df.get(col), errors='coerce').fillna(0)
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
        else: result_df['momentum_score_ewma_q4'] = 0.0; result_df['momentum_score_ewma_q3'] = 0.0
        logger.debug("Finished adding intra-game momentum features.")
        return result_df

    @profile_time
    def integrate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """CLEANED VERSION: Calculates advanced metrics using user's last logic (Poss clip[50,120], Ratings/TOV clip active)."""
        logger.debug("Integrating advanced metrics (v5 User Version - Clipped Poss/Ratings/TOV)...") # Updated log message
        result_df = df.copy()
        # Ensure required stats exist and are numeric, filling missing with 0
        stat_cols = ['home_score', 'away_score', 'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted','home_3pm', 'home_3pa', 'away_3pm', 'away_3pa', 'home_ft_made', 'home_ft_attempted','away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb', 'home_total_reb','away_off_reb', 'away_def_reb', 'away_total_reb', 'home_turnovers', 'away_turnovers','home_ot', 'away_ot']
        for col in stat_cols:
            if col not in result_df.columns: result_df[col] = 0 # Add missing cols as 0
            # Convert to numeric, coerce errors, fill NaNs with 0
            # This assumes 0 is acceptable if loading is correct
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        def safe_divide(numerator, denominator, default_val):
            num = pd.to_numeric(numerator, errors='coerce')
            den = pd.to_numeric(denominator, errors='coerce').replace(0, np.nan)
            result = num / den
            result.replace([np.inf, -np.inf], np.nan, inplace=True)
            return result.fillna(default_val)

        # Basic Shooting / Rebounding / FT Rate (as per user's last code version)
        result_df['home_efg_pct'] = safe_divide(result_df['home_fg_made'] + 0.5 * result_df['home_3pm'], result_df['home_fg_attempted'], self.defaults['efg_pct'])
        result_df['away_efg_pct'] = safe_divide(result_df['away_fg_made'] + 0.5 * result_df['away_3pm'], result_df['away_fg_attempted'], self.defaults['efg_pct'])
        result_df['home_ft_rate'] = safe_divide(result_df['home_ft_attempted'], result_df['home_fg_attempted'], self.defaults['ft_rate'])
        result_df['away_ft_rate'] = safe_divide(result_df['away_ft_attempted'], result_df['away_fg_attempted'], self.defaults['ft_rate'])
        result_df['home_oreb_pct'] = safe_divide(result_df['home_off_reb'], result_df['home_off_reb'] + result_df['away_def_reb'], self.defaults['oreb_pct'])
        result_df['away_dreb_pct'] = safe_divide(result_df['away_def_reb'], result_df['away_def_reb'] + result_df['home_off_reb'], self.defaults['dreb_pct'])
        result_df['away_oreb_pct'] = safe_divide(result_df['away_off_reb'], result_df['away_off_reb'] + result_df['home_def_reb'], self.defaults['oreb_pct'])
        result_df['home_dreb_pct'] = safe_divide(result_df['home_def_reb'], result_df['home_def_reb'] + result_df['away_off_reb'], self.defaults['dreb_pct'])
        result_df['home_trb_pct'] = safe_divide(result_df['home_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct'])
        result_df['away_trb_pct'] = safe_divide(result_df['away_total_reb'], result_df['home_total_reb'] + result_df['away_total_reb'], self.defaults['trb_pct'])

        # Possessions/Pace (Using user's version with clip[50,120])
        home_poss = result_df['home_fg_attempted'] + 0.44 * result_df['home_ft_attempted'] - result_df['home_off_reb'] + result_df['home_turnovers']
        away_poss = result_df['away_fg_attempted'] + 0.44 * result_df['away_ft_attempted'] - result_df['away_off_reb'] + result_df['away_turnovers']
        possessions = 0.5 * (home_poss + away_poss)
        possessions = possessions.clip(lower=50, upper=120) # User's clipping
        result_df['possessions_est'] = possessions.fillna(self.defaults['estimated_possessions'])
        result_df['home_possessions'] = result_df['possessions_est'].replace(0, np.nan) # Prepare for safe_divide
        result_df['away_possessions'] = result_df['possessions_est'].replace(0, np.nan) # Prepare for safe_divide
        num_ot = np.where((result_df['home_ot'] > 0) | (result_df['away_ot'] > 0), 1, 0)
        game_minutes_calc = 48.0 + num_ot * 5.0
        result_df['game_minutes_played'] = np.where(game_minutes_calc <= 0, 48.0, game_minutes_calc)
        result_df['game_pace'] = safe_divide(result_df['possessions_est'] * 48.0, result_df['game_minutes_played'], self.defaults['pace'])
        result_df['home_pace'] = result_df['game_pace']; result_df['away_pace'] = result_df['game_pace']

        # Efficiency (Ratings) - Keeping user's version WITH clipping active
        result_df['home_offensive_rating'] = safe_divide(result_df['home_score'] * 100, result_df['home_possessions'], self.defaults['offensive_rating'])
        result_df['away_offensive_rating'] = safe_divide(result_df['away_score'] * 100, result_df['away_possessions'], self.defaults['offensive_rating'])
        result_df['home_defensive_rating'] = result_df['away_offensive_rating']; result_df['away_defensive_rating'] = result_df['home_offensive_rating']
        rating_cols = ['home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating']
        for col in rating_cols: result_df[col] = result_df[col].clip(lower=70, upper=150).fillna(self.defaults.get(col.split('_')[1]+'_rating', 115.0)) # CLIP ACTIVE
        result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
        result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']

        # Turnover Rate - Keeping user's version WITH clipping active
        result_df['home_tov_rate'] = safe_divide(result_df['home_turnovers'] * 100, result_df['home_possessions'], self.defaults['tov_rate'])
        result_df['away_tov_rate'] = safe_divide(result_df['away_turnovers'] * 100, result_df['away_possessions'], self.defaults['tov_rate'])
        result_df['home_tov_rate'] = result_df['home_tov_rate'].clip(lower=5, upper=25).fillna(self.defaults['tov_rate']) # CLIP ACTIVE
        result_df['away_tov_rate'] = result_df['away_tov_rate'].clip(lower=5, upper=25).fillna(self.defaults['tov_rate']) # CLIP ACTIVE

        # Differentials and Convenience Columns
        result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating']
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
        result_df['oreb_pct_diff'] = result_df['home_oreb_pct'] - result_df['away_oreb_pct']
        result_df['dreb_pct_diff'] = result_df['home_dreb_pct'] - result_df['away_dreb_pct']
        result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']
        result_df['pace_differential'] = 0.0
        result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate']
        if 'total_score' not in result_df.columns: result_df['total_score'] = result_df['home_score'] + result_df['away_score']
        if 'point_diff' not in result_df.columns: result_df['point_diff'] = result_df['home_score'] - result_df['away_score']
        result_df['home_possessions'] = result_df['home_possessions'].fillna(self.defaults['estimated_possessions'])
        result_df['away_possessions'] = result_df['away_possessions'].fillna(self.defaults['estimated_possessions'])

        # --- REMOVED Debug Stats Generation Block ---

        logger.debug("Finished integrating advanced features (Cleaned - User Version).")
        return result_df

    # --- add_rolling_features ---
    @profile_time
    def add_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10]) -> pd.DataFrame:
        """CLEANED VERSION: Adds rolling mean AND std dev features."""
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
                grouped_col = team_view.groupby('team_norm', observed=True)[col]
                shifted_data = grouped_col.shift(1) # Shift BEFORE rolling
                rolling_op = shifted_data.rolling(window=window, min_periods=min_p)
                team_view[roll_mean_col] = rolling_op.mean().fillna(default_mean)
                std_dev = rolling_op.std().fillna(default_std)
                team_view[roll_std_col] = np.maximum(0, std_dev) # Ensure non-negative std
                rolling_cols_generated.extend([roll_mean_col, roll_std_col])
        logger.debug("Rolling: Merging stats back...")
        try:
            team_view['merge_key_rolling'] = team_view['game_id'].astype(str) + "_" + team_view['team_norm']
            local_df['merge_key_home'] = local_df['game_id'].astype(str) + "_" + local_df['home_team_norm']
            local_df['merge_key_away'] = local_df['game_id'].astype(str) + "_" + local_df['away_team_norm']
            cols_to_merge = ['merge_key_rolling'] + [c for c in rolling_cols_generated if c in team_view.columns]
            merge_data = team_view[cols_to_merge].drop_duplicates(subset=['merge_key_rolling'], keep='last')
            def get_rename_dict(prefix, base_cols, windows): rd = {}; [rd.update({f'rolling_{b}_{s}_{w}': f'{prefix}_rolling_{b}_{s}_{w}' for s in ['mean','std'] for w in windows}) for b in base_cols]; return rd
            local_df = pd.merge(local_df, merge_data, how='left', left_on='merge_key_home', right_on='merge_key_rolling').rename(columns=get_rename_dict('home', cols_to_roll_final, window_sizes)).drop(columns=['merge_key_rolling'], errors='ignore')
            local_df = pd.merge(local_df, merge_data, how='left', left_on='merge_key_away', right_on='merge_key_rolling').rename(columns=get_rename_dict('away', cols_to_roll_final, window_sizes)).drop(columns=['merge_key_rolling'], errors='ignore')
        except Exception as e:
            logger.error(f"Rolling: Error merging rolling stats back: {e}", exc_info=True)
            for prefix in ['home', 'away']:
                 for col_base in cols_to_roll_generic:
                     for stat_type in ['mean', 'std']:
                         for w in window_sizes:
                              col_name = f'{prefix}_rolling_{col_base}_{stat_type}_{w}'
                              if col_name not in local_df.columns: local_df[col_name] = self.defaults.get(f'{col_base}_std' if stat_type == 'std' else col_base, 0.0)

        logger.debug("Rolling: Filling NaNs and calculating diffs...")
        primary_window = max(window_sizes) if window_sizes else 10
        for stat_type in ['mean', 'std']:
             for col_base in cols_to_roll_final:
                 for w in window_sizes:
                      for prefix in ['home', 'away']:
                           col_name = f'{prefix}_rolling_{col_base}_{stat_type}_{w}'
                           default_key = f'{col_base}_std' if stat_type == 'std' else col_base; default = self.defaults.get(default_key, 0.0)
                           # Use correct logic instead of setdefault
                           if col_name not in local_df.columns: local_df[col_name] = default
                           else: local_df[col_name] = local_df[col_name].fillna(default)
                           if stat_type == 'std': local_df[col_name] = np.maximum(0, local_df[col_name])
             # Calculate diffs
             w = primary_window
             hm = self._calc_diff_safe(local_df, 'home_rolling_score_for_{stat_type}_{w}', 'home_rolling_score_against_{stat_type}_{w}', stat_type, w)
             am = self._calc_diff_safe(local_df, 'away_rolling_score_for_{stat_type}_{w}', 'away_rolling_score_against_{stat_type}_{w}', stat_type, w)
             local_df[f'rolling_margin_diff_{stat_type}'] = (hm - am).fillna(0.0)
             hne = self._calc_diff_safe(local_df, 'home_rolling_off_rating_{stat_type}_{w}', 'home_rolling_def_rating_{stat_type}_{w}', stat_type, w)
             ane = self._calc_diff_safe(local_df, 'away_rolling_off_rating_{stat_type}_{w}', 'away_rolling_def_rating_{stat_type}_{w}', stat_type, w)
             local_df[f'rolling_eff_diff_{stat_type}'] = (hne - ane).fillna(0.0)
             if 'pace' in cols_to_roll_final: local_df[f'rolling_pace_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_pace_{stat_type}_{w}', 'away_rolling_pace_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'efg_pct' in cols_to_roll_final: local_df[f'rolling_efg_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_efg_pct_{stat_type}_{w}', 'away_rolling_efg_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'tov_rate' in cols_to_roll_final: local_df[f'rolling_tov_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'away_rolling_tov_rate_{stat_type}_{w}', 'home_rolling_tov_rate_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'trb_pct' in cols_to_roll_final: local_df[f'rolling_trb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_trb_pct_{stat_type}_{w}', 'away_rolling_trb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'oreb_pct' in cols_to_roll_final: local_df[f'rolling_oreb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_oreb_pct_{stat_type}_{w}', 'away_rolling_oreb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'dreb_pct' in cols_to_roll_final: local_df[f'rolling_dreb_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_dreb_pct_{stat_type}_{w}', 'away_rolling_dreb_pct_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'ft_rate' in cols_to_roll_final: local_df[f'rolling_ft_rate_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_ft_rate_{stat_type}_{w}', 'away_rolling_ft_rate_{stat_type}_{w}', stat_type, w).fillna(0.0)
             if 'momentum_ewma' in cols_to_roll_final: local_df[f'rolling_momentum_diff_{stat_type}'] = self._calc_diff_safe(local_df, 'home_rolling_momentum_ewma_{stat_type}_{w}', 'away_rolling_momentum_ewma_{stat_type}_{w}', stat_type, w).fillna(0.0)

        # --- REMOVED Debug Stats Generation Block for Rolling Features ---

        logger.debug("Rolling: Cleaning up intermediate columns...")
        local_df = local_df.drop(columns=['merge_key_home', 'merge_key_away', 'home_team_norm', 'away_team_norm'], errors='ignore')
        logger.debug("Finished adding rolling features (v2).")
        return local_df

    # --- _calc_diff_safe ---
    def _calc_diff_safe(self, df: pd.DataFrame, home_template: str, away_template: str, stat_type: str, window: int) -> pd.Series:
        # ... (Keep implementation as is) ...
        home_col = home_template.format(stat_type=stat_type, w=window)
        away_col = away_template.format(stat_type=stat_type, w=window)
        if home_col in df.columns and away_col in df.columns:
            home_vals = pd.to_numeric(df[home_col], errors='coerce'); away_vals = pd.to_numeric(df[away_col], errors='coerce')
            return (home_vals - away_vals)
        else: logger.warning(f"Columns {home_col} or {away_col} not found for diff."); return pd.Series(np.nan, index=df.index)

    # --- add_rest_features_vectorized (v5.1) ---
    @profile_time
    def add_rest_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (Keep implementation as is - v5.1 time-based with key merge) ...
        logger.debug("Adding rest features (vectorized, v5.1 - Time-Based w/ Merge)...")
        if df is None or df.empty or not all(c in df.columns for c in ['game_id', 'game_date', 'home_team', 'away_team']):
            logger.warning("Input df empty/missing essential cols for rest features.")
            placeholder_cols = ['rest_days_home', 'games_last_7_days_home', 'games_last_14_days_home', 'rest_days_away', 'games_last_7_days_away', 'games_last_14_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage']
            temp_df = df.copy() if df is not None else pd.DataFrame()
            for col in placeholder_cols:
                 if col not in temp_df.columns: temp_df[col] = self.defaults.get(col.replace('_home','').replace('_away','').replace('_advantage',''), 0.0)
            return temp_df
        df_copy = df.copy().reset_index(drop=True)
        try: df_copy['game_date'] = pd.to_datetime(df_copy['game_date'], errors='coerce')
        except Exception as e: logger.warning(f"Error processing game_date: {e}. Cannot calc rest features.")
        if df_copy['game_date'].isnull().all(): # Check after potential coercion failure
            logger.warning("All game_date values invalid/missing. Cannot calc rest features.")
            placeholder_cols = ['rest_days_home', 'prev_home_game_date', 'rest_days_away', 'prev_away_game_date', 'games_last_7_days_home', 'games_last_14_days_home', 'games_last_7_days_away', 'games_last_14_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage', 'schedule_advantage']
            for col in placeholder_cols:
                if col not in df_copy.columns: df_copy[col] = self.defaults.get(col.replace('_home','').replace('_away','').replace('_advantage','').replace('prev_','').replace('_date',''), 0.0) if 'days' in col or 'games' in col or 'advantage' in col else pd.NaT
            return df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date'], errors='ignore')
        # --- Rest Days ---
        df_copy_sorted_home = df_copy.sort_values(['home_team', 'game_date'])
        df_copy['prev_home_game_date'] = df_copy_sorted_home.groupby('home_team')['game_date'].shift(1).reindex(df_copy.index)
        df_copy['rest_days_home'] = (df_copy['game_date'] - df_copy['prev_home_game_date']).dt.days.fillna(self.defaults['rest_days'])
        df_copy_sorted_away = df_copy.sort_values(['away_team', 'game_date'])
        df_copy['prev_away_game_date'] = df_copy_sorted_away.groupby('away_team')['game_date'].shift(1).reindex(df_copy.index)
        df_copy['rest_days_away'] = (df_copy['game_date'] - df_copy['prev_away_game_date']).dt.days.fillna(self.defaults['rest_days'])
        # --- Rolling Counts ---
        try:
            df_copy = df_copy.sort_values(['game_date', 'game_id']).reset_index(drop=True); df_copy['temp_calc_id'] = df_copy.index
            home_view = df_copy[['temp_calc_id', 'game_date', 'game_id', 'home_team']].rename(columns={'home_team': 'team'})
            away_view = df_copy[['temp_calc_id', 'game_date', 'game_id', 'away_team']].rename(columns={'away_team': 'team'})
            team_game_log = pd.concat([home_view, away_view], ignore_index=True).sort_values(['team', 'game_date', 'temp_calc_id']).reset_index(drop=True)
            team_game_log_indexed = team_game_log.set_index('game_date')
            counts_7d_series = team_game_log_indexed.groupby('team')['game_id'].rolling('7D', closed='left').count()
            counts_14d_series = team_game_log_indexed.groupby('team')['game_id'].rolling('14D', closed='left').count()
            counts_7d_df = counts_7d_series.reset_index().rename(columns={'game_id': 'games_last_7d'})
            counts_14d_df = counts_14d_series.reset_index().rename(columns={'game_id': 'games_last_14d'})
            team_counts_merged = pd.merge(counts_7d_df, counts_14d_df, on=['team', 'game_date'], how='outer')
            team_game_log_with_counts = pd.merge(team_game_log, team_counts_merged, on=['team', 'game_date'], how='left')
            final_counts = team_game_log_with_counts.sort_values(['team', 'game_date', 'temp_calc_id']).drop_duplicates(subset=['temp_calc_id'], keep='last')
            home_counts_to_merge = final_counts[final_counts['team'].isin(df_copy['home_team'].unique())][['temp_calc_id', 'games_last_7d', 'games_last_14d']].rename(columns={'games_last_7d': 'games_last_7_days_home', 'games_last_14d': 'games_last_14_days_home'})
            away_counts_to_merge = final_counts[final_counts['team'].isin(df_copy['away_team'].unique())][['temp_calc_id', 'games_last_7d', 'games_last_14d']].rename(columns={'games_last_7d': 'games_last_7_days_away', 'games_last_14d': 'games_last_14_days_away'})
            df_copy = pd.merge(df_copy, home_counts_to_merge, on='temp_calc_id', how='left')
            df_copy = pd.merge(df_copy, away_counts_to_merge, on='temp_calc_id', how='left')
        except Exception as rolling_e:
            logger.error(f"Error during time-based rolling count: {rolling_e}", exc_info=True)
            df_copy.setdefault('games_last_7_days_home', self.defaults['games_last_7_days_home'])
            df_copy.setdefault('games_last_14_days_home', self.defaults['games_last_14_days_home'])
            df_copy.setdefault('games_last_7_days_away', self.defaults['games_last_7_days_away'])
            df_copy.setdefault('games_last_14_days_away', self.defaults['games_last_14_days_away'])
        # --- Fill NaNs & Finalize ---
        fill_cols = ['games_last_7_days_home', 'games_last_14_days_home', 'games_last_7_days_away', 'games_last_14_days_away']
        for col in fill_cols: df_copy[col] = df_copy[col].fillna(self.defaults.get(col, 0)).astype(int) # Use defaults dict
        df_copy['is_back_to_back_home'] = (df_copy['rest_days_home'] == 1).astype(int) if 'rest_days_home' in df_copy else 0
        df_copy['is_back_to_back_away'] = (df_copy['rest_days_away'] == 1).astype(int) if 'rest_days_away' in df_copy else 0
        df_copy['rest_advantage'] = df_copy.get('rest_days_home', self.defaults['rest_days']) - df_copy.get('rest_days_away', self.defaults['rest_days'])
        df_copy['schedule_advantage'] = df_copy['games_last_7_days_away'] - df_copy['games_last_7_days_home']
        df_copy = df_copy.drop(columns=['prev_home_game_date', 'prev_away_game_date', 'temp_calc_id'], errors='ignore')
        logger.debug("Finished adding rest features (v5.1).")
        return df_copy

    # --- add_matchup_history_features ---
    @profile_time
    def add_matchup_history_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame], max_games: int = 5) -> pd.DataFrame:
        # ... (Keep implementation as is) ...
        logger.debug(f"Adding H2H features (last {max_games} games)...")
        if df is None or df.empty: return df
        result_df = df.copy()
        placeholder_cols = list(self._get_matchup_history_single("", "", pd.DataFrame(), 0).keys())
        if historical_df is None or historical_df.empty:
            logger.warning("Historical DataFrame empty. Adding H2H placeholders.")
            for col in placeholder_cols:
                 if col not in result_df.columns: result_df[col] = self.defaults.get(col.replace('matchup_',''), 0.0 if col != 'matchup_last_date' else pd.NaT)
            return result_df
        try:
            hist_df = historical_df.copy()
            for col in ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']:
                 if col not in hist_df.columns: raise ValueError(f"H2H: Hist DF missing: {col}")
                 if col == 'game_date': hist_df[col] = pd.to_datetime(hist_df[col], errors='coerce').dt.tz_localize(None)
                 elif 'score' in col: hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
            hist_df = hist_df.dropna(subset=['game_date', 'home_score', 'away_score'])
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None)
            result_df = result_df.dropna(subset=['game_date'])
            if result_df.empty: logger.warning("H2H: No valid input rows."); return df
            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name); result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            hist_df['home_team_norm'] = hist_df['home_team'].apply(self.normalize_team_name); hist_df['away_team_norm'] = hist_df['away_team'].apply(self.normalize_team_name)
            result_df['matchup_key'] = result_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            hist_df['matchup_key'] = hist_df.apply(lambda row: "_vs_".join(sorted([row['home_team_norm'], row['away_team_norm']])), axis=1)
            logger.debug("Grouping H2H history...")
            hist_lookup = {key: group.copy() for key, group in hist_df.sort_values('game_date').groupby('matchup_key')}
            logger.debug("Applying H2H calculation...")
            h2h_results = result_df.apply(lambda row: self._get_matchup_history_single(row['home_team_norm'], row['away_team_norm'], hist_lookup.get(row['matchup_key'], pd.DataFrame()), max_games, current_game_date=row['game_date']), axis=1)
            if not h2h_results.empty: result_df = result_df.join(pd.DataFrame(h2h_results.tolist(), index=result_df.index), how='left')
            else: logger.warning("No H2H results generated.")
            logger.debug("Finalizing H2H features...")
            for col in placeholder_cols:
                 default = self.defaults.get(col.replace('matchup_', ''), 0.0 if col != 'matchup_last_date' else pd.NaT)
                 if col not in result_df.columns: result_df[col] = default # Add if missing
                 else: result_df[col] = result_df[col].fillna(default) # Fill NaNs
                 if col == 'matchup_last_date': result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                 else: result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
                 if col in ['matchup_num_games', 'matchup_streak']: result_df[col] = result_df[col].round().astype(int)
            result_df = result_df.drop(columns=['home_team_norm', 'away_team_norm', 'matchup_key'], errors='ignore')
        except Exception as e:
            logger.error(f"Error adding H2H features: {e}", exc_info=True)
            for col in placeholder_cols:
                if col not in result_df.columns: result_df[col] = self.defaults.get(col.replace('matchup_',''), 0.0 if col != 'matchup_last_date' else pd.NaT)
        logger.debug("Finished H2H features.")
        return result_df

    # --- add_season_context_features ---
    @profile_time
    def add_season_context_features(self, df: pd.DataFrame, team_stats_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        # ... (Keep implementation as is - uses corrected fillna logic) ...
        logger.info("Adding season context features (v2+)...")
        result_df = df.copy()
        placeholder_cols = ['home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for', 'away_season_avg_pts_for', 'home_season_avg_pts_against', 'away_season_avg_pts_against', 'home_current_form', 'away_current_form', 'season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff', 'home_season_net_rating', 'away_season_net_rating', 'season_net_rating_diff']
        req_ts_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
        team_stats_available = False
        if team_stats_df is not None and not team_stats_df.empty:
            ts_df = team_stats_df.copy(); missing_ts_cols = [col for col in req_ts_cols if col not in ts_df.columns]
            if missing_ts_cols: logger.warning(f"Team stats DF missing: {missing_ts_cols}. Filling defaults.")
            for col in req_ts_cols: # Use corrected fillna
                if col not in ts_df.columns: ts_df[col] = 'N/A' if col == 'current_form' else (0.5 if 'percentage' in col else 0.0)
            team_stats_available = True
        else: logger.warning("`team_stats_df` empty/None.")
        if 'game_date' not in result_df.columns or pd.isna(result_df['game_date']).all(): logger.error("`game_date` missing/all NaN."); team_stats_available = False
        if not team_stats_available:
            for col in placeholder_cols:
                 default = self.defaults.get(col.replace('home_','').replace('away_','').replace('season_',''), 0.0 if 'form' not in col else 'N/A')
                 if col not in result_df.columns: result_df[col] = default
                 else: result_df[col] = result_df[col].fillna(default)
            for col in placeholder_cols: result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0) if 'form' not in col else result_df[col].astype(str)
            return result_df
        try: # Main merge logic
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce').dt.tz_localize(None); result_df = result_df.dropna(subset=['game_date']);
            if result_df.empty: raise ValueError("No valid game dates.")
            result_df['season'] = result_df['game_date'].apply(self._determine_season)
            result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name); result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
            ts_df['team_name_norm'] = ts_df['team_name'].apply(self.normalize_team_name)
            ts_merge = ts_df[['team_name_norm', 'season'] + [c for c in req_ts_cols if c not in ['team_name', 'season']]].copy()
            for col in ['wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all']: ts_merge[col] = pd.to_numeric(ts_merge.get(col), errors='coerce')
            ts_merge['season'] = ts_merge['season'].astype(str); ts_merge['merge_key'] = ts_merge['team_name_norm'] + "_" + ts_merge['season']
            ts_merge = ts_merge.drop(columns=['team_name_norm', 'season']).drop_duplicates(subset=['merge_key'], keep='last')
            result_df['merge_key_home'] = result_df['home_team_norm'] + "_" + result_df['season']; result_df['merge_key_away'] = result_df['away_team_norm'] + "_" + result_df['season']
            home_rename = {'wins_all_percentage': 'home_season_win_pct', 'points_for_avg_all': 'home_season_avg_pts_for','points_against_avg_all': 'home_season_avg_pts_against', 'current_form': 'home_current_form'}
            away_rename = {'wins_all_percentage': 'away_season_win_pct', 'points_for_avg_all': 'away_season_avg_pts_for','points_against_avg_all': 'away_season_avg_pts_against', 'current_form': 'away_current_form'}
            logger.debug(f"Attempting to merge season stats...")
            result_df = pd.merge(result_df, ts_merge.rename(columns=home_rename), how='left', left_on='merge_key_home', right_on='merge_key', indicator='_merge_home')
            result_df = pd.merge(result_df, ts_merge.rename(columns=away_rename), how='left', left_on='merge_key_away', right_on='merge_key', suffixes=('', '_away_dup'), indicator='_merge_away')
            if '_merge_home' in result_df.columns and '_merge_away' in result_df.columns:
                 home_success = (result_df['_merge_home'] == 'both').mean(); away_success = (result_df['_merge_away'] == 'both').mean(); logger.info(f"Season stats merge success: Home={home_success:.1%}, Away={away_success:.1%}")
                 if home_success < 0.9 or away_success < 0.9: logger.warning("Low merge success rate for season stats.")
                 result_df = result_df.drop(columns=['_merge_home', '_merge_away'], errors='ignore')
            else: logger.warning("Merge indicators missing.")
            result_df = result_df.drop(columns=[c for c in result_df.columns if '_away_dup' in c or c == 'merge_key'], errors='ignore')
            result_df = result_df.drop(columns=['season', 'home_team_norm', 'away_team_norm', 'merge_key_home', 'merge_key_away'], errors='ignore')
        except Exception as merge_e: logger.error(f"Error during season merge: {merge_e}", exc_info=True)
        logger.debug("Finalizing season context features...") # Final fillna/diff calculation
        for col in placeholder_cols: # Use corrected fillna logic
            default_key = col.replace('home_','').replace('away_','').replace('season_','').replace('_diff','').replace('_advantage','')
            default = self.defaults.get(default_key, 0.0 if 'form' not in col else 'N/A')
            if 'diff' in col: default = 0.0
            if col not in result_df.columns: result_df[col] = default
            else: result_df[col] = result_df[col].fillna(default)
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default) if 'form' not in col else result_df[col].astype(str)
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
        # ... (Keep implementation as is - uses corrected fillna logic) ...
        logger.debug("Adding features derived from form strings (v2+)...")
        result_df = df.copy()
        form_metric_keys = list(self._extract_form_metrics_single("").keys())
        home_cols = [f'home_{k}' for k in form_metric_keys]; away_cols = [f'away_{k}' for k in form_metric_keys]
        diff_cols = ['form_win_pct_diff', 'streak_advantage', 'momentum_diff']; placeholder_cols = home_cols + away_cols + diff_cols
        home_form_col = 'home_current_form'; away_form_col = 'away_current_form'
        if home_form_col not in result_df.columns or away_form_col not in result_df.columns: logger.warning(f"Missing form columns.")
        else: # Calculate if columns exist
             try:
                valid_home = result_df[home_form_col].fillna('').astype(str).replace('N/A','').str.len() > 0
                valid_away = result_df[away_form_col].fillna('').astype(str).replace('N/A','').str.len() > 0
                logger.info(f"Processing forms: Valid Home={valid_home.mean():.1%}, Away={valid_away.mean():.1%}")
                result_df[home_form_col] = result_df[home_form_col].fillna('').astype(str); result_df[away_form_col] = result_df[away_form_col].fillna('').astype(str)
                home_metrics = result_df[home_form_col].apply(self._extract_form_metrics_single); away_metrics = result_df[away_form_col].apply(self._extract_form_metrics_single)
                result_df = result_df.join(pd.DataFrame(home_metrics.tolist(), index=result_df.index).add_prefix('home_'))
                result_df = result_df.join(pd.DataFrame(away_metrics.tolist(), index=result_df.index).add_prefix('away_'))
                result_df['form_win_pct_diff'] = result_df.get('home_form_win_pct', self.defaults['form_win_pct']) - result_df.get('away_form_win_pct', self.defaults['form_win_pct'])
                result_df['streak_advantage'] = result_df.get('home_current_streak', self.defaults['current_streak']) - result_df.get('away_current_streak', self.defaults['current_streak'])
                result_df['momentum_diff'] = result_df.get('home_momentum_direction', self.defaults['momentum_direction']) - result_df.get('away_momentum_direction', self.defaults['momentum_direction'])
             except Exception as e: logger.error(f"Error processing form strings: {e}", exc_info=True)
        # Finalize all placeholder cols
        logger.debug("Finalizing form features...")
        for col in placeholder_cols:
             default = self.defaults.get(col.replace('home_','').replace('away_','').replace('_diff','').replace('_advantage',''), 0.0)
             if col not in result_df.columns: result_df[col] = default # Add if missing
             else: result_df[col] = result_df[col].fillna(default) # Fill NAs
             result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default) # Ensure numeric
             if 'streak' in col and 'advantage' not in col: result_df[col] = result_df[col].round().astype(int) # Ensure int for streak
        logger.debug("Finished form string features.")
        return result_df

    # --- add_betting_odds ---
    @profile_time
    def add_betting_odds(self, df: pd.DataFrame, betting_odds_data: Optional[Union[pd.DataFrame, Dict[Any, Dict]]] = None) -> pd.DataFrame:
        # ... (Keep implementation as is - uses corrected fillna logic) ...
        logger.debug("Adding betting odds features (if provided)...")
        result_df = df.copy(); odds_cols = ['vegas_home_spread', 'vegas_over_under']
        default_spread = self.defaults['vegas_home_spread']; default_ou = self.defaults['vegas_over_under']
        for col in odds_cols: # Ensure columns exist and are numeric with defaults
             default = default_spread if col == 'vegas_home_spread' else default_ou
             if col not in result_df.columns: result_df[col] = default
             else: result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default)
        if betting_odds_data is None: logger.debug("No betting odds data provided."); return result_df
        try: # Merge/Extract logic
            if isinstance(betting_odds_data, pd.DataFrame):
                 if 'game_id' not in betting_odds_data.columns: raise ValueError("Odds DF missing 'game_id'.")
                 odds_to_merge = [c for c in odds_cols if c in betting_odds_data.columns];
                 if not odds_to_merge: raise ValueError("Odds DF missing relevant columns.")
                 logger.debug(f"Merging odds: {odds_to_merge} from DataFrame.")
                 odds_df = betting_odds_data[['game_id'] + odds_to_merge].copy()
                 result_df['gid_merge'] = result_df['game_id'].astype(str); odds_df['gid_merge'] = odds_df['game_id'].astype(str)
                 result_df = pd.merge(result_df, odds_df.drop(columns=['game_id']), on='gid_merge', how='left', suffixes=('', '_odds'))
                 for col in odds_to_merge: # Overwrite existing column with merged values where available
                     if f"{col}_odds" in result_df.columns: result_df[col] = pd.to_numeric(result_df[f"{col}_odds"], errors='coerce').combine_first(result_df[col]); result_df = result_df.drop(columns=[f"{col}_odds"])
                 result_df = result_df.drop(columns=['gid_merge'], errors='ignore')
            elif isinstance(betting_odds_data, dict):
                 logger.debug("Extracting odds from dictionary."); raise NotImplementedError("Dict handling for odds needs review/update if used.") # Simplified for now
            else: raise TypeError("Unsupported betting_odds_data format.")
            # Final fillna after merge/extract
            result_df['vegas_home_spread'] = result_df['vegas_home_spread'].fillna(default_spread)
            result_df['vegas_over_under'] = result_df['vegas_over_under'].fillna(default_ou)
            logger.debug("Finished adding betting odds features.")
        except Exception as e: logger.error(f"Error adding betting odds: {e}", exc_info=True); # Keep defaults added initially
        return result_df

    # --- Private Helpers ---
    def _get_matchup_history_single(self, home_team_norm: str, away_team_norm: str, historical_subset: pd.DataFrame, max_games: int = 5, current_game_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        # ... (Keep implementation as is) ...
        default_result = {'matchup_num_games': self.defaults['matchup_num_games'], 'matchup_avg_point_diff': self.defaults['matchup_avg_point_diff'], 'matchup_home_win_pct': self.defaults['matchup_home_win_pct'], 'matchup_avg_total_score': self.defaults['matchup_avg_total_score'], 'matchup_avg_home_score': self.defaults['matchup_avg_home_score'], 'matchup_avg_away_score': self.defaults['matchup_avg_away_score'], 'matchup_last_date': pd.NaT, 'matchup_streak': self.defaults['matchup_streak']}
        if historical_subset.empty or max_games <= 0 or pd.isna(current_game_date): return default_result
        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        if past_games_df.empty: return default_result
        try:
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            if recent_matchups.empty: return default_result
            recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce'); recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
            recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score'])
            if recent_matchups.empty: return default_result
            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []; home_persp_wins, current_streak, last_winner_norm = 0, 0, None
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
                h_score = game['home_score']; a_score = game['away_score']; g_home = game.get('home_team_norm'); g_away = game.get('away_team_norm')
                if g_home == home_team_norm: diff, won, h_persp, a_persp = h_score - a_score, (h_score > a_score), h_score, a_score
                elif g_away == home_team_norm: diff, won, h_persp, a_persp = a_score - h_score, (a_score > h_score), a_score, h_score
                else: continue
                diffs.append(diff); total_scores.append(h_score + a_score); home_persp_scores.append(h_persp); away_persp_scores.append(a_persp)
                if won: home_persp_wins += 1
                winner = home_team_norm if won else away_team_norm
                if last_winner_norm is None: last_winner_norm = winner; current_streak = 1 if won else -1
                elif winner == last_winner_norm: current_streak += (1 if won else -1)
                else: last_winner_norm = winner; current_streak = 1 if won else -1
            num_games = len(diffs);
            if num_games == 0: return default_result
            final_stats = {'matchup_num_games': num_games, 'matchup_avg_point_diff': np.mean(diffs), 'matchup_home_win_pct': home_persp_wins / num_games, 'matchup_avg_total_score': np.mean(total_scores), 'matchup_avg_home_score': np.mean(home_persp_scores), 'matchup_avg_away_score': np.mean(away_persp_scores), 'matchup_last_date': recent_matchups['game_date'].max(), 'matchup_streak': int(current_streak)}
            for k, v in default_result.items(): final_stats.setdefault(k,v); final_stats[k] = final_stats[k] if pd.notna(final_stats[k]) else v
            return final_stats
        except Exception as e: logger.error(f"Error in _get_matchup_history_single ({home_team_norm} vs {away_team_norm}): {e}"); return default_result

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        # ... (Keep implementation as is) ...
        defaults = {'form_win_pct': self.defaults['form_win_pct'], 'current_streak': self.defaults['current_streak'], 'momentum_direction': self.defaults['momentum_direction']}
        if not form_string or pd.isna(form_string) or not isinstance(form_string, str): return defaults
        form_string = form_string.upper().strip().replace('-','').replace('?',''); form_len = len(form_string)
        if form_len == 0 or form_string == 'N/A': return defaults
        wins = form_string.count('W'); form_win_pct = wins / form_len; current_streak = 0
        if form_len > 0:
             streak_char = form_string[-1]; streak_count = 0
             for char in reversed(form_string):
                 if char == streak_char: streak_count += 1
                 else: break
             current_streak = streak_count if streak_char == 'W' else -streak_count
        momentum_direction = 0.0
        if form_len >= 4:
             split_point = form_len // 2; recent_half, older_half = form_string[-split_point:], form_string[:-split_point]
             len_r, len_o = len(recent_half), len(older_half)
             if len_r > 0 and len_o > 0:
                 pct_r = recent_half.count('W') / len_r; pct_o = older_half.count('W') / len_o
                 if pct_r > pct_o: momentum_direction = 1.0
                 elif pct_r < pct_o: momentum_direction = -1.0
        return {'form_win_pct': form_win_pct, 'current_streak': int(current_streak), 'momentum_direction': momentum_direction}

    # --- Main Orchestration Method ---
    @profile_time
    def generate_all_features(self, df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None,
                              betting_odds_data: Optional[Union[pd.DataFrame, Dict[Any, Dict]]] = None,
                              rolling_windows: List[int] = [5, 10],
                              h2h_window: int = 5) -> pd.DataFrame:
        """Applies all feature engineering steps in sequence. CLEANED VERSION."""
        logger.info("Starting comprehensive feature generation pipeline...")
        start_time_total = time.time()
        # --- Input Validation & Prep ---
        if df is None or df.empty: logger.error("Input df empty."); return pd.DataFrame()
        essential_cols = ['game_id', 'game_date', 'home_team', 'away_team'];
        if not all(c in df.columns for c in essential_cols): logger.error(f"Input df missing essentials."); return pd.DataFrame()
        target_games_df = df.copy()
        try: target_games_df['game_date'] = pd.to_datetime(target_games_df['game_date'], errors='coerce').dt.tz_localize(None); target_games_df = target_games_df.dropna(subset=['game_date'])
        except Exception as e: logger.error(f"Error processing target game_date: {e}"); return pd.DataFrame()
        if target_games_df.empty: logger.error("No valid target game dates."); return pd.DataFrame()
        logger.info(f"Target date range: {target_games_df['game_date'].min().date()} to {target_games_df['game_date'].max().date()}")

        # --- Combine History & Target ---
        base_calc_df = None; hist_df_processed = None
        if historical_games_df is not None and not historical_games_df.empty:
            logger.debug(f"Processing {len(historical_games_df)} historical games...")
            try:
                hist_df_processed = historical_games_df.copy(); hist_df_processed['game_date'] = pd.to_datetime(hist_df_processed['game_date'], errors='coerce').dt.tz_localize(None); hist_df_processed = hist_df_processed.dropna(subset=['game_date'])
                hist_cols = set(hist_df_processed.columns); target_cols = set(target_games_df.columns); all_cols_union = list(hist_cols.union(target_cols))
                for col in all_cols_union: # Align columns
                    if col not in hist_df_processed.columns: hist_df_processed[col] = np.nan
                    if col not in target_games_df.columns: target_games_df[col] = np.nan
                logger.info(f"Combining {len(hist_df_processed)} hist / {len(target_games_df)} target games...")
                base_calc_df = pd.concat([hist_df_processed[all_cols_union], target_games_df[all_cols_union]], ignore_index=True).sort_values(['game_date','game_id']).reset_index(drop=True)
                # Deduplicate
                initial_rows = len(base_calc_df); base_calc_df['game_id'] = base_calc_df['game_id'].astype(str); base_calc_df = base_calc_df.drop_duplicates('game_id', keep='last'); rows_dropped = initial_rows - len(base_calc_df)
                if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} duplicate game_id rows.")
                logger.info(f"Created base_calc_df: {len(base_calc_df)} rows.")
            except Exception as e: logger.error(f"Error combining data: {e}", exc_info=True); base_calc_df = target_games_df.copy().sort_values(['game_date','game_id']).reset_index(drop=True)
        else: logger.warning("No historical data provided."); base_calc_df = target_games_df.copy().sort_values(['game_date','game_id']).reset_index(drop=True)
        if base_calc_df is None or base_calc_df.empty: logger.error("base_calc_df empty."); return pd.DataFrame()

        # --- Feature Generation Sequence ---
        try:
            logger.info("Step 1/9: Adding intra-game momentum features...")
            base_calc_df = self.add_intra_game_momentum(base_calc_df)
            logger.info("Step 2/9: Integrating advanced metrics (v5 User)...") # User's version with poss clip[50,120], rate clips active
            base_calc_df = self.integrate_advanced_features(base_calc_df)
            logger.info("Step 3/9: Adding rolling features (v2)...")
            base_calc_df = self.add_rolling_features(base_calc_df, window_sizes=rolling_windows)
            logger.info("Step 4/9: Adding rest & schedule density features (v5.1)...")
            base_calc_df = self.add_rest_features_vectorized(base_calc_df)
            logger.info("Step 5/9: Adding H2H matchup features...")
            base_calc_df = self.add_matchup_history_features(base_calc_df, hist_df_processed if hist_df_processed is not None else pd.DataFrame(), max_games=h2h_window)
            logger.info("Step 6/9: Adding season context features (v2+)...")
            base_calc_df = self.add_season_context_features(base_calc_df, team_stats_df)
            logger.info("Step 7/9: Adding form string features (v2)...")
            base_calc_df = self.add_form_string_features(base_calc_df)
            logger.info("Step 8/9: Adding betting odds features...")
            base_calc_df = self.add_betting_odds(base_calc_df, betting_odds_data)

            # --- Final Filtering ---
            logger.info("Step 9/9: Filtering back to target games...")
            original_game_ids = df['game_id'].astype(str).unique(); num_unique_original_ids = len(original_game_ids)
            logger.debug(f"Filtering for {num_unique_original_ids} unique target game IDs...")
            if 'game_id' not in base_calc_df.columns: raise ValueError("'game_id' missing.")
            base_calc_df['game_id'] = base_calc_df['game_id'].astype(str)
            final_df = base_calc_df[base_calc_df['game_id'].isin(original_game_ids)].copy()
            logger.info(f"Shape of final_df after filtering: {final_df.shape}")
            if len(final_df) != num_unique_original_ids: # Use corrected check logic
                 logger.error(f"Final Filtering Mismatch! Expected: {num_unique_original_ids}, Got: {len(final_df)}.")
                 return pd.DataFrame()
            else: logger.info("Final filtering successful.")

        except Exception as e: logger.error(f"Error during feature pipeline: {e}", exc_info=True); return pd.DataFrame()

        total_time = time.time() - start_time_total
        logger.info(f"Feature generation pipeline complete for {len(final_df)} games in {total_time:.2f}s.")
        return final_df.reset_index(drop=True)

    # --- Private Helpers ---
    def _get_matchup_history_single(self, home_team_norm: str, away_team_norm: str, historical_subset: pd.DataFrame, max_games: int = 5, current_game_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        # ... (Keep implementation as is) ...
        default_result = {'matchup_num_games': self.defaults['matchup_num_games'], 'matchup_avg_point_diff': self.defaults['matchup_avg_point_diff'], 'matchup_home_win_pct': self.defaults['matchup_home_win_pct'], 'matchup_avg_total_score': self.defaults['matchup_avg_total_score'], 'matchup_avg_home_score': self.defaults['matchup_avg_home_score'], 'matchup_avg_away_score': self.defaults['matchup_avg_away_score'], 'matchup_last_date': pd.NaT, 'matchup_streak': self.defaults['matchup_streak']}
        if historical_subset.empty or max_games <= 0 or pd.isna(current_game_date): return default_result
        past_games_df = historical_subset[historical_subset['game_date'] < current_game_date].copy()
        if past_games_df.empty: return default_result
        try:
            recent_matchups = past_games_df.sort_values('game_date', ascending=False).head(max_games)
            if recent_matchups.empty: return default_result
            recent_matchups['home_score'] = pd.to_numeric(recent_matchups['home_score'], errors='coerce'); recent_matchups['away_score'] = pd.to_numeric(recent_matchups['away_score'], errors='coerce')
            recent_matchups = recent_matchups.dropna(subset=['home_score', 'away_score'])
            if recent_matchups.empty: return default_result
            diffs, total_scores, home_persp_scores, away_persp_scores = [], [], [], []; home_persp_wins, current_streak, last_winner_norm = 0, 0, None
            for _, game in recent_matchups.sort_values('game_date', ascending=True).iterrows():
                h_score = game['home_score']; a_score = game['away_score']; g_home = game.get('home_team_norm'); g_away = game.get('away_team_norm')
                if g_home == home_team_norm: diff, won, h_persp, a_persp = h_score - a_score, (h_score > a_score), h_score, a_score
                elif g_away == home_team_norm: diff, won, h_persp, a_persp = a_score - h_score, (a_score > h_score), a_score, h_score
                else: continue
                diffs.append(diff); total_scores.append(h_score + a_score); home_persp_scores.append(h_persp); away_persp_scores.append(a_persp)
                if won: home_persp_wins += 1
                winner = home_team_norm if won else away_team_norm
                if last_winner_norm is None: last_winner_norm = winner; current_streak = 1 if won else -1
                elif winner == last_winner_norm: current_streak += (1 if won else -1)
                else: last_winner_norm = winner; current_streak = 1 if won else -1
            num_games = len(diffs);
            if num_games == 0: return default_result
            final_stats = {'matchup_num_games': num_games, 'matchup_avg_point_diff': np.mean(diffs), 'matchup_home_win_pct': home_persp_wins / num_games, 'matchup_avg_total_score': np.mean(total_scores), 'matchup_avg_home_score': np.mean(home_persp_scores), 'matchup_avg_away_score': np.mean(away_persp_scores), 'matchup_last_date': recent_matchups['game_date'].max(), 'matchup_streak': int(current_streak)}
            for k, v in default_result.items(): final_stats.setdefault(k,v); final_stats[k] = final_stats[k] if pd.notna(final_stats[k]) else v # Ensure all keys exist
            return final_stats
        except Exception as e: logger.error(f"Error in _get_matchup_history_single ({home_team_norm} vs {away_team_norm}): {e}"); return default_result

    def _extract_form_metrics_single(self, form_string: Optional[str]) -> Dict[str, float]:
        # ... (Keep implementation as is) ...
        defaults = {'form_win_pct': self.defaults['form_win_pct'], 'current_streak': self.defaults['current_streak'], 'momentum_direction': self.defaults['momentum_direction']}
        if not form_string or pd.isna(form_string) or not isinstance(form_string, str): return defaults
        form_string = form_string.upper().strip().replace('-','').replace('?',''); form_len = len(form_string)
        if form_len == 0 or form_string == 'N/A': return defaults
        wins = form_string.count('W'); form_win_pct = wins / form_len; current_streak = 0
        if form_len > 0:
             streak_char = form_string[-1]; streak_count = 0
             for char in reversed(form_string):
                 if char == streak_char: streak_count += 1
                 else: break
             current_streak = streak_count if streak_char == 'W' else -streak_count
        momentum_direction = 0.0
        if form_len >= 4:
             split_point = form_len // 2; recent_half, older_half = form_string[-split_point:], form_string[:-split_point]
             len_r, len_o = len(recent_half), len(older_half)
             if len_r > 0 and len_o > 0:
                 pct_r = recent_half.count('W') / len_r; pct_o = older_half.count('W') / len_o
                 if pct_r > pct_o: momentum_direction = 1.0
                 elif pct_r < pct_o: momentum_direction = -1.0
        return {'form_win_pct': form_win_pct, 'current_streak': int(current_streak), 'momentum_direction': momentum_direction}

# --- Example Usage ---
if __name__ == '__main__':
    pass