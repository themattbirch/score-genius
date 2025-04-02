# backend/nba_score_prediction/train_models.py

from __future__ import annotations # Keep this at the top
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING: # Keep TYPE_CHECKING block if needed for linters
    from supabase import Client
    # Define Pipeline type hint for type checkers
    from sklearn.pipeline import Pipeline

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Scikit-learn Imports ---
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
# --- Need Pipeline for tune_and_evaluate_predictor ---
from sklearn.pipeline import Pipeline
# --- Tuning & CV ---
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform # For defining parameter distributions
# --- Metrics ---
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)

from .evaluation import (
    plot_residuals_analysis_detailed, 
    plot_conditional_bias, 
    plot_temporal_bias,
    plot_feature_importances,
    plot_actual_vs_predicted 
) 

# --- Database Imports (Optional) ---
try: from sqlalchemy import create_engine, text
except ImportError: create_engine = None; print("WARNING: SQLAlchemy not installed.")

# --- Supabase Import ---
try: from supabase import create_client
except ImportError: create_client = None; print("WARNING: Supabase library not installed.")

# --- Conditional XGBoost Import ---
try: import xgboost as xgb; XGBOOST_AVAILABLE = True
except ImportError: xgb = None; XGBOOST_AVAILABLE = False; print("WARNING: XGBoost library not found.")

# --- Project-Specific Imports ---
try:
    from .feature_engineering import NBAFeatureEngine
    from .models import (RandomForestScorePredictor, RidgeScorePredictor)
    from .models import compute_recency_weights # Assumes this is the correct one now
    # --- Import evaluation functions ---
    from .evaluation import (
        # Define custom metrics locally if needed, or rely on standard sklearn scorers
        plot_feature_importances,
        plot_actual_vs_predicted,
        plot_residuals_analysis_detailed
        # Add others if needed e.g. generate_evaluation_report
    )
   # ------------------------------------
    if XGBOOST_AVAILABLE:
        from .models import XGBoostScorePredictor
    else:
        class XGBoostScorePredictor: # Dummy class if XGBoost not available
             def __init__(self, *args, **kwargs): pass
             def _build_pipeline(self, *args, **kwargs): return None
             def train(self, *args, **kwargs): pass
             def predict(self, *args, **kwargs): return None
             def save_model(self, *args, **kwargs): pass
             def load_model(self, *args, **kwargs): pass
             feature_names_in_ = None
             training_timestamp = None
             training_duration = None

    from backend import config  # Corrected import
    LOCAL_MODULES_IMPORTED = True
except ImportError as e:
    print(f"ERROR: Could not import local modules: {e}. Using dummy classes.")
    # Define dummy classes/functions
    class NBAFeatureEngine: pass
    class XGBoostScorePredictor: # Dummy class
         def __init__(self, *args, **kwargs): pass
         def _build_pipeline(self, *args, **kwargs): return None
         def train(self, *args, **kwargs): pass
         def predict(self, *args, **kwargs): return None
         def save_model(self, *args, **kwargs): pass
         def load_model(self, *args, **kwargs): pass
         feature_names_in_ = None
         training_timestamp = None
         training_duration = None
    class RandomForestScorePredictor(XGBoostScorePredictor): pass # Inherit dummy structure
    class RidgeScorePredictor(XGBoostScorePredictor): pass # Inherit dummy structure
    def compute_recency_weights(*args, **kwargs): return np.ones(len(args[0])) if args else np.array([])
    def plot_feature_importances(*args, **kwargs): logger.error("Dummy plot_feature_importances called!")
    def plot_actual_vs_predicted(*args, **kwargs): logger.error("Dummy plot_actual_vs_predicted called!")
    def plot_residuals_analysis_detailed(*args, **kwargs): logger.error("Dummy plot_residuals_analysis_detailed called!")
    try: from backend import config
    except ImportError: config = None; print("ERROR: Could not import config module.")
    LOCAL_MODULES_IMPORTED = False

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

MODELS_BASE_DIR = BACKEND_DIR / 'models'
MAIN_MODELS_DIR = MODELS_BASE_DIR / 'saved'
QUARTERLY_MODELS_DIR = MODELS_BASE_DIR / 'quarterly'
REPORTS_DIR = PROJECT_ROOT / 'reports'

MAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
QUARTERLY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = ['home_score', 'away_score']
SEED = 42
DEFAULT_CV_FOLDS = 5 # Default splits for TimeSeriesSplit

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 0: CLIENT INITIALIZATION (No Changes Needed)
# ==============================================================================
def get_db_engine():
    db_url = getattr(config, 'DATABASE_URL', None)
    if db_url and create_engine:
        try:
            engine = create_engine(db_url)
            logger.info("DB engine created.")
            return engine
        except Exception as e:
            logger.error(f"Error creating DB engine: {e}", exc_info=True)
    elif db_url and not create_engine:
        logger.error("DB URL set but SQLAlchemy missing.")
    else:
        logger.debug("DB URL not set.")
    return None

def get_supabase_client() -> Optional["Client"]:
    supa_url = getattr(config, 'SUPABASE_URL', None)
    supa_key = getattr(config, 'SUPABASE_ANON_KEY', None)
    if supa_url and supa_key and create_client:
        try:
            supabase = create_client(supa_url, supa_key)
            logger.info("Supabase client created.")
            return supabase
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}", exc_info=True)
    elif (supa_url or supa_key) and not create_client:
        logger.error("Supabase URL/Key found but library missing.")
    else:
        logger.debug("Supabase URL/Key not set.")
    return None

# ==============================================================================
# SECTION 1: DATA LOADING (No Changes Needed)
# ==============================================================================
def load_data_source(source_type: str, lookback_days: int, args: argparse.Namespace,
                     db_engine: Optional[Any] = None, supabase_client: Optional["Client"] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # (Implementation remains the same as provided in the last version)
    logger.info(f"Attempting to load data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()
    hist_required_cols = [
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot', 'away_q1', 'away_q2',
        'away_q3', 'away_q4', 'away_ot', 'home_fg_made', 'home_fg_attempted', 'away_fg_made',
        'away_fg_attempted', 'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa', 'home_ft_made',
        'home_ft_attempted', 'away_ft_made', 'away_ft_attempted', 'home_off_reb', 'home_def_reb',
        'home_total_reb', 'away_off_reb', 'away_def_reb', 'away_total_reb', 'home_turnovers',
        'away_turnovers', 'home_assists', 'home_steals', 'home_blocks', 'home_fouls',
        'away_assists', 'away_steals', 'away_blocks', 'away_fouls'
    ]
    hist_numeric_cols = [col for col in hist_required_cols if col not in ['game_id', 'game_date', 'home_team', 'away_team']]
    team_required_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
    team_numeric_cols = ['wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all']

    if source_type == "supabase":
        if not supabase_client:
            logger.error("Supabase client unavailable.")
        else:
            logger.info("Loading historical games from Supabase...")
            threshold_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            try:
                all_historical_data = []
                page_size = 1000
                start_index = 0
                has_more = True
                select_str = ", ".join(hist_required_cols)
                while has_more:
                    response = supabase_client.table("nba_historical_game_stats") \
                        .select(select_str) \
                        .gte("game_date", threshold_date) \
                        .order('game_date') \
                        .range(start_index, start_index + page_size - 1) \
                        .execute()
                    batch = response.data
                    b_size = len(batch)
                    all_historical_data.extend(batch)
                    logger.debug(f"Retrieved {b_size} hist records...")
                    if b_size < page_size:
                        has_more = False
                    else:
                        start_index += page_size
                if all_historical_data:
                    hist_df = pd.DataFrame(all_historical_data)
                    logger.info(f"Loaded {len(hist_df)} hist records.")
                    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
                    hist_df = hist_df.dropna(subset=['game_date'])
                else:
                    logger.warning(f"No hist data since {threshold_date}.")
            except Exception as e:
                logger.error(f"Error loading historical games from Supabase: {e}", exc_info=True)
            # --- Load Team Stats ---
            logger.info("Loading team stats from Supabase...")
            try:
                select_str = ", ".join(team_required_cols)
                response = supabase_client.table("nba_historical_team_stats").select(select_str).execute()
                if response.data:
                    team_stats_df = pd.DataFrame(response.data)
                    logger.info(f"Loaded {len(team_stats_df)} team stat records.")
                else:
                    logger.warning("No team stats found.")
            except Exception as e:
                logger.error(f"Error loading team stats from Supabase: {e}", exc_info=True)
    # Add logic for other source_type (csv, database) if needed here
    elif source_type == "csv":
         logger.info(f"Loading historical data from CSV: {args.historical_csv_path}")
         try:
             hist_df = pd.read_csv(args.historical_csv_path)
             hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
             hist_df = hist_df.dropna(subset=['game_date'])
             threshold_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
             hist_df = hist_df[hist_df['game_date'] >= threshold_date]
             logger.info(f"Loaded {len(hist_df)} historical records from CSV.")
         except FileNotFoundError:
             logger.error(f"Historical CSV not found at {args.historical_csv_path}")
         except Exception as e:
             logger.error(f"Error loading historical CSV: {e}", exc_info=True)

         logger.info(f"Loading team stats from CSV: {args.team_stats_csv_path}")
         try:
             team_stats_df = pd.read_csv(args.team_stats_csv_path)
             logger.info(f"Loaded {len(team_stats_df)} team stat records from CSV.")
         except FileNotFoundError:
             logger.error(f"Team stats CSV not found at {args.team_stats_csv_path}")
         except Exception as e:
             logger.error(f"Error loading team stats CSV: {e}", exc_info=True)

    # --- Final Cleaning ---
    if not hist_df.empty:
        missing_hist_cols = [col for col in hist_required_cols if col not in hist_df.columns]
        if missing_hist_cols:
             logger.warning(f"Historical data missing columns: {missing_hist_cols}. Filling with 0.")
             for col in missing_hist_cols: hist_df[col] = 0
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)
        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)

    if not team_stats_df.empty:
        missing_team_cols = [col for col in team_required_cols if col not in team_stats_df.columns]
        if missing_team_cols:
             logger.warning(f"Team stats data missing columns: {missing_team_cols}. Filling.")
             for col in missing_team_cols: team_stats_df[col] = 0.0 if col in team_numeric_cols else ''
        for col in team_numeric_cols:
            if col in team_stats_df.columns:
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
        if 'current_form' in team_stats_df.columns:
            team_stats_df['current_form'] = team_stats_df['current_form'].astype(str).fillna('')

    logger.info(f"Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df


# ==============================================================================
# SECTION 2: UTILITY & HELPER FUNCTIONS (No Changes Needed)
# ==============================================================================
def visualize_recency_weights(dates, weights, title="Recency Weights Distribution", save_path=None):
     """ Visualizes the distribution of computed recency weights. """
     if dates is None or weights is None or len(dates) != len(weights):
          logger.warning("Invalid input for visualizing recency weights.")
          return
     try:
          df = pd.DataFrame({'date': pd.to_datetime(dates), 'weight': weights}).sort_values('date')
          plt.figure(figsize=(12, 6))
          plt.plot(df['date'], df['weight'], marker='.', linestyle='-')
          plt.title(title)
          plt.xlabel("Date")
          plt.ylabel("Sample Weight")
          plt.grid(True, linestyle='--', alpha=0.7)
          plt.tight_layout()
          if save_path:
               save_path = Path(save_path)
               save_path.parent.mkdir(parents=True, exist_ok=True)
               plt.savefig(save_path)
               logger.info(f"Recency weights plot saved to {save_path}")
          plt.show()
     except Exception as e:
          logger.error(f"Error visualizing recency weights: {e}", exc_info=True)

def select_features(feature_engine: Optional[NBAFeatureEngine],
                    data_df: pd.DataFrame,
                    target_context: str = 'pregame_score') -> List[str]:
    """
    Selects features suitable for pre-game score prediction, ensuring numeric types
    and excluding post-game raw stats and direct derivatives.
    """
    logger.info(f"Selecting features for target context: {target_context}")
    selected_features: List[str] = []

    # Define columns unavailable PRE-GAME (Targets, Raw Box Scores, Post-Game Calculated Advanced Stats)
    pregame_exclude_cols = set([
        # --- Identifiers / Metadata ---
        'id', 'game_id', 'home_team_id', 'away_team_id', 'season', 'league_id',
        # --- TARGETS ---
        'home_score', 'away_score', 'point_diff', 'total_score',
        # --- RAW BOX SCORE STATS ---
        'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
        'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
        'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
        'home_off_reb', 'home_def_reb', 'home_total_reb', 'away_off_reb', 'away_def_reb', 'away_total_reb',
        'home_turnovers', 'away_turnovers', 'home_assists', 'home_steals', 'home_blocks', 'home_fouls',
        'away_assists', 'away_steals', 'away_blocks', 'away_fouls',
        # --- RAW QUARTER SCORES ---
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot',
        'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_ot',
        # --- FEATURES DERIVED DIRECTLY FROM *THIS GAME's* RAW STATS (POST-GAME) ---
        'home_efg_pct', 'away_efg_pct', 'efg_pct_diff', 'home_ft_rate', 'away_ft_rate', 'ft_rate_diff',
        'home_oreb_pct', 'away_dreb_pct', 'away_oreb_pct', 'home_dreb_pct', 'oreb_pct_diff', 'dreb_pct_diff',
        'home_trb_pct', 'away_trb_pct', 'trb_pct_diff', 'possessions_est', 'home_possessions', 'away_possessions',
        'game_minutes_played', 'game_pace', 'home_pace', 'away_pace', 'pace_differential',
        'home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating',
        'home_net_rating', 'away_net_rating', 'efficiency_differential', 'home_tov_rate', 'away_tov_rate', 'tov_rate_diff',
        # --- INTRA-GAME MOMENTUM (Calculated from Quarter Scores) ---
        'q1_margin', 'q2_margin', 'q3_margin', 'q4_margin', 'end_q1_diff', 'end_q2_diff',
        'end_q3_diff', 'end_q4_reg_diff', 'q2_margin_change', 'q3_margin_change',
        'q4_margin_change', 'momentum_score_ewma_q4', 'momentum_score_ewma_q3',
        # --- Other Exclusions ---
        'home_team', 'away_team', 'game_date',             # Raw Info (keep game_date index if needed separately)
        'home_team_norm', 'away_team_norm', 'matchup_key', # Intermediate keys
        'current_form',                                     # Raw form string (use parsed features instead)
        'matchup_last_date',                                # Datetime column (use recency derived from it)
    ])

    try:
        # Start with ALL columns available in the data
        all_cols = data_df.columns.tolist()
        # Exclude the post-game/target columns
        candidate_features = [col for col in all_cols if col not in pregame_exclude_cols]
        # Select only purely numeric types from the candidates
        temp_numeric_df = data_df[candidate_features].select_dtypes(include=np.number)
        selected_features = temp_numeric_df.columns.tolist()
        # Optional: Remove columns that are all NaN or all Zero after selection
        cols_to_drop = [col for col in selected_features if data_df[col].isnull().all() or (data_df[col] == 0).all()]
        if cols_to_drop:
            logger.warning(f"Dropping constant/all-NaN features: {cols_to_drop}")
            selected_features = [col for col in selected_features if col not in cols_to_drop]

    except Exception as e:
        logger.error(f"Error during feature selection: {e}", exc_info=True)
        selected_features = []  # Reset on error

    if not selected_features:
        logger.error("No numeric pre-game features selected! Cannot train models.")
    else:
        logger.info(f"Selected {len(selected_features)} purely numeric pre-game features.")
        logger.debug(f"Selected features sample: {selected_features[:10]}") # Log first few

    return selected_features


# ==============================================================================
# SECTION 3: CORE METRIC & CUSTOM LOSS FUNCTIONS (Keep local definitions)
# ==============================================================================
# Define metrics locally to resolve potential import issues or keep self-contained
def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]
                                 ) -> Dict[str, float]:
    """ Calculates standard regression metrics (MSE, RMSE, MAE, R2). """
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred): logger.error(f"Length mismatch: {len(y_true)} vs {len(y_pred)}"); return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true) == 0: logger.warning("Empty input arrays."); return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        mse = mean_squared_error(y_true, y_pred); rmse = np.sqrt(mse); mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2 }
    except Exception as e: logger.error(f"Error calculating regression metrics: {e}"); return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

def nba_score_loss(y_true, y_pred, spread_weight=0.6, total_weight=0.4):
    """ Custom loss: weighted MSE of point spread and total score errors. """
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2) # Assume home, away if flat
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2) # Assume home, away if flat
        if y_true.shape[1] != 2 or y_pred.shape[1] != 2: logger.error("Need 2 columns (home, away) for nba_score_loss"); return np.inf
        true_home, true_away = y_true[:, 0], y_true[:, 1]; pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        if np.isnan(pred_home).any() or np.isnan(pred_away).any(): logger.warning("NaNs in predictions for nba_score_loss"); return np.inf
        diff_error = ((true_home - true_away) - (pred_home - pred_away)) ** 2
        total_error = ((true_home + true_away) - (pred_home + pred_away)) ** 2
        return np.mean(spread_weight * diff_error + total_weight * total_error)
    except Exception as e: logger.error(f"Error in nba_score_loss: {e}"); return np.inf

def nba_distribution_loss(y_true, y_pred): # y_true is not used, matches scorer signature
    """ Custom loss: penalizes predictions deviating from typical NBA score distributions. """
    try:
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2) # Assume home, away if flat
        if y_pred.shape[1] != 2: logger.error("Need 2 columns (home, away) for nba_distribution_loss"); return np.inf
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        if np.isnan(pred_home).any() or np.isnan(pred_away).any(): logger.warning("NaNs in predictions for nba_distribution_loss"); return np.inf
        # Use reasonable defaults for expected distributions (can be tuned)
        home_mean, home_std = 114, 13.5; away_mean, away_std = 112, 13.5
        diff_mean, diff_std = 2.5, 13.5; total_mean, total_std = 226, 23.0
        pred_diff, pred_total = pred_home - pred_away, pred_home + pred_away
        z_home = ((pred_home - home_mean) / home_std)**2 if home_std > 0 else np.zeros_like(pred_home)
        z_away = ((pred_away - away_mean) / away_std)**2 if away_std > 0 else np.zeros_like(pred_away)
        z_diff = ((pred_diff - diff_mean) / diff_std)**2 if diff_std > 0 else np.zeros_like(pred_diff)
        z_total = ((pred_total - total_mean) / total_std)**2 if total_std > 0 else np.zeros_like(pred_total)
        # Weights can be adjusted (e.g., penalize total/diff deviations more)
        return np.mean(0.2 * z_home + 0.2 * z_away + 0.3 * z_diff + 0.3 * z_total)
    except Exception as e: logger.error(f"Error in nba_distribution_loss: {e}"); return np.inf

def combined_nba_loss(y_true, y_pred, accuracy_weight=0.7, distribution_weight=0.3):
    """ Combined loss: weighted sum of nba_score_loss and nba_distribution_loss. """
    score_loss = nba_score_loss(y_true, y_pred); dist_loss = nba_distribution_loss(y_true, y_pred)
    if np.isinf(score_loss) or np.isinf(dist_loss): return np.inf
    return accuracy_weight * score_loss + distribution_weight * dist_loss

# Make scorers using the locally defined functions for use in RandomizedSearch if desired
# Note: Make sure the scorer functions accept (y_true, y_pred) signature
nba_score_scorer = make_scorer(nba_score_loss, greater_is_better=False, needs_proba=False)
nba_distribution_scorer = make_scorer(nba_distribution_loss, greater_is_better=False, needs_proba=False)
combined_scorer = make_scorer(combined_nba_loss, greater_is_better=False, needs_proba=False)

def calculate_betting_metrics(y_true, y_pred, vegas_lines: Optional[pd.DataFrame] = None):
    """ Calculates basic betting-related metrics (e.g., win prediction accuracy). """
    metrics = {}
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2) # Assume home, away
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2) # Assume home, away
        if y_true.shape[1] != 2 or y_pred.shape[1] != 2: logger.error("Need 2 columns for betting metrics"); return metrics
        true_home, true_away = y_true[:, 0], y_true[:, 1]; pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        true_diff = true_home - true_away; pred_diff = pred_home - pred_away
        metrics['win_prediction_accuracy'] = np.mean((true_diff > 0) == (pred_diff > 0))
        # Add more metrics here if vegas_lines are provided (ATS, O/U accuracy)
    except Exception as e: logger.error(f"Error calculating betting metrics: {e}")
    return metrics

# ==============================================================================
# SECTION 4: QUARTERLY MODEL TRAINING UTILITIES (Placeholders - Unchanged)
# ==============================================================================
def initialize_trainable_models(*args, **kwargs): pass
def train_quarter_models(*args, **kwargs): pass

# ==============================================================================
# SECTION 5: HYPERPARAMETER TUNING & MAIN PREDICTOR TRAINING (FUNCTION REMAINS)
# ==============================================================================
# In train_models.py - REPLACE the existing function with this:

def tune_and_evaluate_predictor(
    predictor_class,
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str,
    feature_list: List[str],
    param_dist: Optional[Dict[str, Any]], # Parameter distribution
    n_iter: int = 10,
    n_splits: int = 5,
    scoring: str = 'neg_mean_absolute_error', # Default scoring metric
    use_recency_weights: bool = False,
    weight_method: str = 'exponential',
    weight_half_life: int = 90,
    visualize: bool = True,
    save_plots: bool = False
    ) -> Optional[Dict]:
    """
    Tunes hyperparameters using RandomizedSearchCV with TimeSeriesSplit,
    trains the final model on combined Train+Val data, evaluates on the Test set AND Train+Val set,
    saves the model, and generates performance plots.
    Args:
        # ... (arguments remain the same as before) ...
    Returns:
        Dictionary containing evaluation metrics (including training set metrics) and metadata, or None on critical failure.
    """
    model_full_name = f"{model_name_prefix}_score_predictor"

    # --- Input Validation ---
    if predictor_class is None:
        logger.error(f"Predictor class provided for {model_full_name} is None. Cannot proceed.")
        return None
    if model_name_prefix == 'xgboost' and not XGBOOST_AVAILABLE:
        logger.error("XGBoost requested but unavailable (checked inside function).")
        return None
    if not feature_list:
        logger.error(f"Feature list is empty for {model_full_name}. Cannot train.")
        return None
    # --- End Input Validation ---

    logger.info(f"\n--- Tuning and Training {predictor_class.__name__} ({model_full_name}) ---")
    start_tune_time = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {} # Use model defaults if tuning skipped/fails

    try:
        # --- Data Prep for Final Fit ---
        train_val_date_col = []
        if use_recency_weights and 'game_date' in X_train.columns and 'game_date' in X_val.columns:
            train_val_date_col = ['game_date']
        cols_train = [col for col in feature_list + train_val_date_col if col in X_train.columns]
        cols_val = [col for col in feature_list + train_val_date_col if col in X_val.columns]
        X_train_full = pd.concat([X_train[cols_train], X_val[cols_val]], ignore_index=False)
        y_train_home_full = pd.concat([y_train_home, y_val_home])
        y_train_away_full = pd.concat([y_train_away, y_val_away])
        logger.info(f"Combined Train+Val size for final fit: {len(X_train_full)} samples.")

        # --- Prepare Data for Tuning ---
        X_tune_features = X_train[feature_list].copy()
        y_tune = y_train_home.copy()

        # --- Calculate Sample Weights for Tuning ---
        # ... (tuning weight calculation logic remains the same) ...
        tune_fit_params = {}
        if use_recency_weights:
            if 'game_date' in X_train.columns:
                logger.info("Calculating recency weights for tuning dataset (X_train)...")
                dates_for_weights_tune = X_train.loc[X_tune_features.index, 'game_date']
                sample_weights_tune = compute_recency_weights(dates_for_weights_tune, method=weight_method, half_life=weight_half_life)
                if sample_weights_tune is not None and len(sample_weights_tune) == len(X_tune_features):
                    try:
                         temp_predictor = predictor_class()
                         temp_pipeline = temp_predictor._build_pipeline({}) if hasattr(temp_predictor, '_build_pipeline') else None
                         if temp_pipeline and hasattr(temp_pipeline, 'steps'):
                             model_step_name = temp_pipeline.steps[-1][0]
                             weight_param_name = f"{model_step_name}__sample_weight"
                             tune_fit_params[weight_param_name] = sample_weights_tune
                             logger.info(f"Prepared sample weights for tuning using key: {weight_param_name}")
                         else: logger.warning("Could not build temporary pipeline to determine weight parameter name.")
                    except Exception as e: logger.warning(f"Could not get model step name for sample weights: {e}")
                else: logger.warning("Failed to compute or align tuning weights.")
            else: logger.warning("`use_recency_weights` is True, but 'game_date' column not found in X_train for tuning.")


        # --- Hyperparameter Tuning ---
        # ... (tuning logic remains the same) ...
        if param_dist:
            logger.info(f"Starting RandomizedSearch (n_iter={n_iter}, cv={n_splits}, scoring='{scoring}')...")
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                temp_predictor = predictor_class()
                search_pipeline = temp_predictor._build_pipeline({}) if hasattr(temp_predictor, '_build_pipeline') else None
                if search_pipeline is None:
                    raise RuntimeError(f"Could not build pipeline for tuning {predictor_class.__name__}")
                rs = RandomizedSearchCV(
                    estimator=search_pipeline, param_distributions=param_dist, n_iter=n_iter,
                    scoring=scoring, cv=tscv, n_jobs=-1, verbose=3, random_state=SEED,
                    error_score='raise', refit=False
                )
                rs.fit(X_tune_features, y_tune, **tune_fit_params)
                logger.debug("RandomizedSearch CV Results:")
                try:
                    cv_results_df = pd.DataFrame(rs.cv_results_)
                    cols_to_show = ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
                    cols_to_show = [c for c in cols_to_show if c in cv_results_df.columns]
                    sort_col = 'rank_test_score' if 'rank_test_score' in cols_to_show else ('mean_test_score' if 'mean_test_score' in cols_to_show else None)
                    if sort_col: print(cv_results_df[cols_to_show].sort_values(by=sort_col, ascending=True).to_string())
                    else: print(cv_results_df.head())
                except Exception as report_e: logger.warning(f"Could not print detailed cv_results_: {report_e}")

                best_params_raw = rs.best_params_
                metrics['best_cv_score'] = rs.best_score_
                logger.info(f"Tuning complete. Best CV Score ({scoring}): {metrics['best_cv_score']:.4f}")
                logger.info(f"Best Raw Params Found: {best_params_raw}")
                best_params_final = {k.split('__', 1)[1]: v for k, v in best_params_raw.items()}
                metrics['best_params'] = best_params_final

            except Exception as search_e:
                 logger.error(f"RandomizedSearchCV failed: {search_e}", exc_info=True)
                 logger.warning("Proceeding with default parameters due to tuning error.")
                 best_params_final = {}

            tune_duration = time.time() - start_tune_time
            logger.info(f"Tuning finished in {tune_duration:.2f} seconds.")
            metrics['tuning_duration'] = tune_duration
        else:
             logger.info("Skipping hyperparameter tuning (no parameter distribution provided).")
             metrics['best_cv_score'] = None
             metrics['best_params'] = 'default'

        # --- Train Final Model on Combined Train + Validation Data ---
        logger.info("Training final model on Train+Val data using best/default parameters...")
        final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
        X_train_full_for_fit = X_train_full[feature_list].copy() # Select only features for fitting
        y_train_home_full_aligned = y_train_home_full.loc[X_train_full_for_fit.index] # Align targets
        y_train_away_full_aligned = y_train_away_full.loc[X_train_full_for_fit.index]

        # --- Calculate Final Sample Weights ---
        # ... (final weight calculation logic remains the same) ...
        final_sample_weights = None
        if use_recency_weights:
            if 'game_date' in X_train_full.columns:
                 dates_for_weights_full = X_train_full.loc[X_train_full_for_fit.index, 'game_date']
                 logger.info("Calculating recency weights for final training dataset...")
                 final_sample_weights = compute_recency_weights(dates_for_weights_full, method=weight_method, half_life=weight_half_life)
                 if final_sample_weights is None or len(final_sample_weights) != len(X_train_full_for_fit):
                     logger.warning("Failed to compute or align final sample weights. Training without weights.")
                     final_sample_weights = None
                 else:
                      logger.info(f"Final sample weights computed ({len(final_sample_weights)} values).")
                      if visualize: visualize_recency_weights(dates_for_weights_full, final_sample_weights, title=f"Final Training Weights - {model_full_name}")
            else:
                 logger.warning("`use_recency_weights` is True, but 'game_date' column not found in combined training data.")

        # --- Actual Training Call ---
        final_predictor.train(
            X_train=X_train_full_for_fit,
            y_train_home=y_train_home_full_aligned,
            y_train_away=y_train_away_full_aligned,
            hyperparams_home=best_params_final,
            hyperparams_away=best_params_final,
            sample_weights=final_sample_weights
        )
        metrics['training_duration_final'] = final_predictor.training_duration

        # --- Saving the Final Model ---
        save_path = final_predictor.save_model(filename=f"{model_full_name}_tuned_{final_predictor.training_timestamp}.joblib")
        logger.info(f"Final tuned {model_full_name} saved to {save_path}")
        metrics['save_path'] = str(save_path)
        metrics['feature_count'] = len(final_predictor.feature_names_in_) if hasattr(final_predictor, 'feature_names_in_') and final_predictor.feature_names_in_ is not None else len(feature_list)

        # --- Final Evaluation on Test Set ---
        logger.info(f"Evaluating final tuned {model_full_name} on test set ({len(X_test)} samples)...")
        X_test_final = X_test[feature_list].copy()
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or 'predicted_home_score' not in predictions_df_test.columns:
            raise ValueError(f"Final prediction on test set failed for {model_full_name}.")
        pred_home_test = predictions_df_test['predicted_home_score']
        pred_away_test = predictions_df_test['predicted_away_score']
        y_test_home_aligned = y_test_home.loc[X_test_final.index]
        y_test_away_aligned = y_test_away.loc[X_test_final.index]
        y_true_comb_test = np.vstack((y_test_home_aligned.values, y_test_away_aligned.values)).T
        y_pred_comb_test = np.vstack((pred_home_test.values, pred_away_test.values)).T

        # --- Calculate Final Test Metrics ---
        test_metrics_home = calculate_regression_metrics(y_test_home_aligned, pred_home_test)
        test_metrics_away = calculate_regression_metrics(y_test_away_aligned, pred_away_test)
        metrics['test_mae_home']=test_metrics_home.get('mae', np.nan); metrics['test_rmse_home']=test_metrics_home.get('rmse', np.nan); metrics['test_r2_home']=test_metrics_home.get('r2', np.nan)
        metrics['test_mae_away']=test_metrics_away.get('mae', np.nan); metrics['test_rmse_away']=test_metrics_away.get('rmse', np.nan); metrics['test_r2_away']=test_metrics_away.get('r2', np.nan)
        try:
            metrics['test_mae_total']=mean_absolute_error(y_test_home_aligned + y_test_away_aligned, pred_home_test + pred_away_test)
            metrics['test_mae_diff']=mean_absolute_error(y_test_home_aligned - y_test_away_aligned, pred_home_test - pred_away_test)
        except Exception as basic_metric_e:
            logger.error(f"Error calculating basic MAE metrics: {basic_metric_e}")
            metrics['test_mae_total'] = np.nan; metrics['test_mae_diff'] = np.nan
        metrics['test_nba_score_loss']=nba_score_loss(y_true_comb_test, y_pred_comb_test)
        metrics['test_nba_dist_loss']=nba_distribution_loss(y_true_comb_test, y_pred_comb_test)
        metrics['test_combined_loss']=combined_nba_loss(y_true_comb_test, y_pred_comb_test)
        metrics['betting_metrics']=calculate_betting_metrics(y_true_comb_test, y_pred_comb_test, vegas_lines=None)

        # --- Log Final Test Metrics ---
        logger.info(f"  FINAL Test MAE : Home={metrics['test_mae_home']:.3f}, Away={metrics['test_mae_away']:.3f}, Total={metrics['test_mae_total']:.3f}, Diff={metrics['test_mae_diff']:.3f}")
        logger.info(f"  FINAL Test RMSE: Home={metrics['test_rmse_home']:.3f}, Away={metrics['test_rmse_away']:.3f}")
        logger.info(f"  FINAL Test R2  : Home={metrics['test_r2_home']:.3f}, Away={metrics['test_r2_away']:.3f}")
        logger.info(f"  FINAL Custom Losses: Score={metrics['test_nba_score_loss']:.3f}, Dist={metrics['test_nba_dist_loss']:.3f}, Combined={metrics['test_combined_loss']:.3f}")
        logger.info(f"  FINAL Betting Metrics: {metrics['betting_metrics']}")

        # ***** START: NEW CODE FOR TRAINING SET EVALUATION *****
        logger.info(f"Evaluating final tuned {model_full_name} on training set ({len(X_train_full_for_fit)} samples)...")
        try:
            predictions_df_train = final_predictor.predict(X_train_full_for_fit)
            if predictions_df_train is None or 'predicted_home_score' not in predictions_df_train.columns:
                 raise ValueError(f"Final prediction on train set failed for {model_full_name}.")
            pred_home_train = predictions_df_train['predicted_home_score']
            pred_away_train = predictions_df_train['predicted_away_score']

            # Calculate Training Set Metrics
            train_metrics_home = calculate_regression_metrics(y_train_home_full_aligned, pred_home_train)
            train_metrics_away = calculate_regression_metrics(y_train_away_full_aligned, pred_away_train)
            metrics['train_mae_home']=train_metrics_home.get('mae', np.nan); metrics['train_rmse_home']=train_metrics_home.get('rmse', np.nan); metrics['train_r2_home']=train_metrics_home.get('r2', np.nan)
            metrics['train_mae_away']=train_metrics_away.get('mae', np.nan); metrics['train_rmse_away']=train_metrics_away.get('rmse', np.nan); metrics['train_r2_away']=train_metrics_away.get('r2', np.nan)
            try:
                metrics['train_mae_total']=mean_absolute_error(y_train_home_full_aligned + y_train_away_full_aligned, pred_home_train + pred_away_train)
                metrics['train_mae_diff']=mean_absolute_error(y_train_home_full_aligned - y_train_away_full_aligned, pred_home_train - pred_away_train)
            except Exception as basic_train_metric_e:
                 logger.error(f"Error calculating basic TRAIN MAE metrics: {basic_train_metric_e}")
                 metrics['train_mae_total'] = np.nan; metrics['train_mae_diff'] = np.nan

            # Log Final Training Metrics
            logger.info(f"  FINAL Train MAE : Home={metrics['train_mae_home']:.3f}, Away={metrics['train_mae_away']:.3f}, Total={metrics['train_mae_total']:.3f}, Diff={metrics['train_mae_diff']:.3f}")
            logger.info(f"  FINAL Train RMSE: Home={metrics['train_rmse_home']:.3f}, Away={metrics['train_rmse_away']:.3f}")
            logger.info(f"  FINAL Train R2  : Home={metrics['train_r2_home']:.3f}, Away={metrics['train_r2_away']:.3f}")

        except Exception as train_eval_e:
            logger.error(f"FAILED evaluating on training set: {train_eval_e}", exc_info=True)
            # Add NaN placeholders if training evaluation fails
            metrics['train_mae_home']=np.nan; metrics['train_rmse_home']=np.nan; metrics['train_r2_home']=np.nan
            metrics['train_mae_away']=np.nan; metrics['train_rmse_away']=np.nan; metrics['train_r2_away']=np.nan
            metrics['train_mae_total']=np.nan; metrics['train_mae_diff']=np.nan
        # ***** END: NEW CODE FOR TRAINING SET EVALUATION *****

        # --- Record Data Sizes ---
        metrics['samples_train_final']=len(X_train_full); metrics['samples_test']=len(X_test)

        # --- Visualization ---
        if visualize or save_plots:
            plot_dir = REPORTS_DIR / f"{model_full_name}_tuned_performance"
            plot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Generating performance plots for final {model_full_name}...")
            try:
                # Test Set Plots
                plot_actual_vs_predicted(y_test_home_aligned, pred_home_test, f"Tuned {model_full_name} - Test Actual vs Pred (Home)", {'rmse':metrics['test_rmse_home'],'r2':metrics['test_r2_home']}, save_path=plot_dir / "test_actual_vs_pred_home.png")
                plot_actual_vs_predicted(y_test_away_aligned, pred_away_test, f"Tuned {model_full_name} - Test Actual vs Pred (Away)", {'rmse':metrics['test_rmse_away'],'r2':metrics['test_r2_away']}, save_path=plot_dir / "test_actual_vs_pred_away.png")
                plot_residuals_analysis_detailed(y_test_home_aligned, pred_home_test, f"Tuned {model_full_name} (Home) - Test Set", save_dir=plot_dir)
                # Add Training Set Plots (Optional - uncomment if desired)
                # plot_actual_vs_predicted(y_train_home_full_aligned, pred_home_train, f"Tuned {model_full_name} - Train Actual vs Pred (Home)", {'rmse':metrics['train_rmse_home'],'r2':metrics['train_r2_home']}, save_path=plot_dir / "train_actual_vs_pred_home.png")
                # plot_actual_vs_predicted(y_train_away_full_aligned, pred_away_train, f"Tuned {model_full_name} - Train Actual vs Pred (Away)", {'rmse':metrics['train_rmse_away'],'r2':metrics['train_r2_away']}, save_path=plot_dir / "train_actual_vs_pred_away.png")
                # plot_residuals_analysis_detailed(y_train_home_full_aligned, pred_home_train, f"Tuned {model_full_name} (Home) - Train Set", save_dir=plot_dir)


                # Feature Importance Plot
                logger.info(f"Generating feature importance plots for final {model_full_name}...")
                pipeline_home = getattr(final_predictor, 'pipeline_home', None)
                pipeline_away = getattr(final_predictor, 'pipeline_away', None)
                features_in = getattr(final_predictor, 'feature_names_in_', None)
                if pipeline_home and pipeline_away and features_in:
                    models_to_plot = { f"{model_full_name}_Home": pipeline_home, f"{model_full_name}_Away": pipeline_away }
                    plot_feature_importances(models_dict=models_to_plot, feature_names=features_in, top_n=20, plot_groups=True, save_dir=plot_dir / "feature_importance")
                else: logger.warning("Cannot generate feature importance: Missing final pipeline or feature names attribute.")
            except Exception as plot_e: logger.error(f"Failed generating plots: {plot_e}", exc_info=True)

    except Exception as e:
        logger.error(f"FAILED Tuning/Training/Evaluation: {predictor_class.__name__} ({model_full_name}) - {e}", exc_info=True)
        metrics['error'] = str(e) # Record error in metrics

    metrics['total_duration'] = time.time() - start_tune_time
    logger.info(f"--- Finished Tuning & Training {model_full_name} in {metrics['total_duration']:.2f}s ---")
    return metrics


# ==============================================================================
# SECTION 6: EVALUATION & ANALYSIS FUNCTIONS (Placeholders - Unchanged)
# ==============================================================================
def analyze_main_predictors_learning_curves(*args, **kwargs): pass
# ... (other analysis function placeholders) ...

# ==============================================================================
# SECTION 7: MAIN EXECUTION BLOCK (UPDATED PARAM DISTRIBUTIONS)
# ==============================================================================

def run_training_pipeline(args):
    """ Main function to orchestrate the tuning, training and evaluation pipeline. """
    start_pipeline_time = time.time()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled.")
    logger.info("--- Starting NBA Model Tuning & Training Pipeline ---")
    logger.info(f"Arguments: {vars(args)}")
    if not LOCAL_MODULES_IMPORTED and not args.allow_dummy:
        logger.critical("Required local modules not found and dummy run not allowed. Exiting.")
        sys.exit(1)
    if not config:
        logger.critical("Config module missing. Exiting.")
        sys.exit(1)

    db_engine = get_db_engine()
    supabase_client = get_supabase_client()
    historical_df, team_stats_df = load_data_source(args.data_source, args.lookback_days, args, db_engine, supabase_client)
    if historical_df.empty:
        logger.error("Failed to load historical data. Exiting.")
        sys.exit(1)
    if team_stats_df.empty:
        logger.warning("Team stats data is empty. Season context features will use defaults.")

    # --- Feature Generation (Do this ONCE before the loop) ---
    logger.info("Initializing Feature Engine...")
    feature_engine = NBAFeatureEngine(debug=args.debug)
    logger.info("Generating features for ALL historical data...")
    # Ensure rolling windows are parsed correctly from args
    rolling_windows_list = [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [5, 10]
    logger.info(f"Using rolling windows: {rolling_windows_list}")  # Log the windows being used

    features_df = feature_engine.generate_all_features(
        df=historical_df.copy(),  # Use the full loaded historical data
        historical_games_df=historical_df.copy(),  # Pass history again for H2H etc.
        team_stats_df=team_stats_df.copy() if not team_stats_df.empty else None,
        rolling_windows=rolling_windows_list,  # USE THE PARSED LIST CONSISTENTLY
        h2h_window=args.h2h_window
    )
    if features_df.empty:
        logger.error("Feature generation failed. Exiting.")
        sys.exit(1)

    # Ensure target columns exist before dropping NaNs
    for col in TARGET_COLUMNS:
        if col not in features_df.columns and col in historical_df.columns:
            logger.info(f"Merging target column '{col}' from original historical data.")
            # Merge carefully to avoid duplication issues if game_id isn't unique or index changed
            features_df = features_df.merge(historical_df[['game_id', col]], on='game_id', how='left')
        elif col not in features_df.columns:
            logger.error(f"Target column '{col}' not found in features or historical data. Cannot proceed.")
            sys.exit(1)

    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    rows_dropped = initial_rows - len(features_df)
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows due to missing target values.")
    if features_df.empty:
        logger.error("No rows remaining after dropping missing targets. Exiting.")
        sys.exit(1)
    logger.info(f"Feature generation complete. Final shape for splitting: {features_df.shape}")

    # --- Time-Based Data Splitting ---
    logger.info("Splitting data (time-based)...")
    features_df = features_df.sort_values('game_date').reset_index(drop=True)
    n = len(features_df)
    test_split_idx = int(n * (1 - args.test_size))
    # Ensure val_size doesn't overlap with test_size
    val_split_frac = min(args.val_size, 1.0 - args.test_size - 0.01)  # Leave a tiny gap if needed
    val_split_idx = int(n * (1 - args.test_size - val_split_frac))
    train_df = features_df.iloc[:val_split_idx]
    val_df = features_df.iloc[val_split_idx:test_split_idx]
    test_df = features_df.iloc[test_split_idx:]
    logger.info(f"Data Split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        logger.error("Empty data split generated. Check sizes or data. Exiting.")
        sys.exit(1)

    # --- Feature Selection (Based ONLY on Training Data) ---
    logger.info("Selecting features based on training data...")
    # Pass feature_engine instance if its selection logic is implemented
    selected_features = select_features(feature_engine, train_df)
    if not selected_features:
        logger.error("No features selected. Exiting.")
        sys.exit(1)
    logger.info(f"Selected {len(selected_features)} features for model training.")

    # --- Prepare Final Data Splits for Models ---
    # Keep 'game_date' in X_train/X_val IF weighting is enabled and column exists
    train_cols_to_keep = selected_features[:]  # Start with selected features
    if args.use_weights and 'game_date' in train_df.columns:
        train_cols_to_keep.append('game_date')
    val_cols_to_keep = selected_features[:]
    if args.use_weights and 'game_date' in val_df.columns:
        val_cols_to_keep.append('game_date')

    X_train = train_df[train_cols_to_keep]
    y_train_home = train_df[TARGET_COLUMNS[0]]
    y_train_away = train_df[TARGET_COLUMNS[1]]
    X_val = val_df[val_cols_to_keep]
    y_val_home = val_df[TARGET_COLUMNS[0]]
    y_val_away = val_df[TARGET_COLUMNS[1]]
    X_test = test_df[selected_features]
    y_test_home = test_df[TARGET_COLUMNS[0]]
    y_test_away = test_df[TARGET_COLUMNS[1]]  # Test set only needs features

    # --- Define Parameter Distributions for RandomizedSearchCV ---
    # Distributions match Scikit-learn pipeline naming conventions (e.g., 'xgb__n_estimators')
    XGB_PARAM_DIST = {
        'xgb__n_estimators': randint(100, 701),  # Explore 100-700 trees
        'xgb__max_depth': randint(4, 11),  # Explore depths 4-10
        'xgb__learning_rate': uniform(0.01, 0.19),  # Explore 0.01 to 0.20 (uniform takes loc, scale)
        'xgb__min_child_weight': randint(1, 8),  # Default 1, explore up to 7
        'xgb__subsample': uniform(0.6, 0.4),  # Explore 0.6 to 1.0
        'xgb__colsample_bytree': uniform(0.6, 0.4),  # Explore 0.6 to 1.0
        'xgb__reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],  # Explore L1 regularization including zero
        'xgb__reg_lambda': [0, 0.5, 1.0, 2.0, 5.0]  # Explore L2 regularization including zero
    }
    # *** UPDATED RandomForest Distribution ***
    RF_PARAM_DIST = {
        'rf__n_estimators': randint(200, 801),  # Explore 200-800 trees
        'rf__max_depth': [10, 15, 20, 25, 30, None],  # Explore depths including unlimited
        'rf__min_samples_split': randint(2, 11),  # *** REFINED: Explore 2-10 ***
        'rf__min_samples_leaf': randint(1, 5),  # *** REFINED: Explore 1-4 ***
        'rf__max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9]  # Explore different strategies
    }
    RIDGE_PARAM_DIST = {
        # Explore alpha on a logarithmic scale
        'ridge__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    }
    param_dist_map = {"xgboost": XGB_PARAM_DIST, "random_forest": RF_PARAM_DIST, "ridge": RIDGE_PARAM_DIST}
    predictor_map = {
        "xgboost": XGBoostScorePredictor if XGBOOST_AVAILABLE else None,
        "random_forest": RandomForestScorePredictor,
        "ridge": RidgeScorePredictor
    }

    # --- Model Tuning & Training Loop ---
    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    logger.info(f"Starting tuning & training for models: {models_to_run}")
    all_metrics = []
    for model_key in models_to_run:
        PredictorClass = predictor_map.get(model_key)
        if PredictorClass is None or (model_key == 'xgboost' and not XGBOOST_AVAILABLE):
            logger.warning(f"Model class for '{model_key}' unavailable. Skipping.")
            continue

        # Get param dist unless skipping tuning
        param_dist_current = param_dist_map.get(model_key) if not args.skip_tuning else None

        # Call the tuning/evaluation function
        # It now receives data splits already based on the CONSISTENT feature set
        metrics = tune_and_evaluate_predictor(
            predictor_class=PredictorClass,
            X_train=X_train.copy(), y_train_home=y_train_home.copy(), y_train_away=y_train_away.copy(),
            X_val=X_val.copy(), y_val_home=y_val_home.copy(), y_val_away=y_val_away.copy(),
            X_test=X_test.copy(), y_test_home=y_test_home.copy(), y_test_away=y_test_away.copy(),
            model_name_prefix=model_key,
            feature_list=selected_features,  # Pass the consistently selected features
            param_dist=param_dist_current,
            n_iter=args.tune_iterations,
            n_splits=args.cv_splits,
            scoring=args.scoring_metric,  # Use specified scoring metric
            use_recency_weights=args.use_weights,
            weight_method=args.weight_method,
            weight_half_life=args.weight_half_life,
            visualize=args.visualize,
            save_plots=args.save_plots
        )
        if metrics:
            all_metrics.append(metrics)

    # --- Report Overall Results ---
    logger.info("\n--- Overall Training Summary ---")
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        # Add model_name as index if it exists and is unique
        if 'model_name' in summary_df.columns and summary_df['model_name'].is_unique:
            summary_df = summary_df.set_index('model_name')

        display_cols = [
            'predictor_class', 'best_cv_score', 'test_mae_home', 'test_mae_away',
            'test_rmse_home', 'test_rmse_away', 'test_r2_home', 'test_r2_away',
            'test_combined_loss', 'feature_count', 'training_duration_final', 'tuning_duration'
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns]  # Show only available cols
        # Print with adjusted formatting
        print(summary_df[display_cols].to_string(float_format="%.4f"))

        summary_path = REPORTS_DIR / "training_summary_tuned.csv"
        try:
            summary_df.to_csv(summary_path)
            logger.info(f"Summary saved: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary CSV: {e}")
    else:
        logger.warning("No models were successfully trained/evaluated.")

    # --- Optional Analysis (Placeholder) ---
    if args.run_analysis:
        logger.info("Running analysis (Placeholders)...")

    total_pipeline_time = time.time() - start_pipeline_time
    logger.info(f"--- NBA Model Training Pipeline Finished in {total_pipeline_time:.2f} seconds ---")



if __name__ == "__main__":
    # --- Argument Parsing (Updated) ---
    parser = argparse.ArgumentParser(description="NBA Score Prediction Model Tuning & Training Pipeline")
    # Data Args
    parser.add_argument("--data-source", type=str, default="supabase", choices=["csv", "supabase", "database", "dummy"], help="Data source type")
    parser.add_argument("--historical-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_game_stats.csv'), help="Path to historical games CSV")
    parser.add_argument("--team-stats-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_team_stats.csv'), help="Path to team stats CSV")
    parser.add_argument("--lookback-days", type=int, default=1095, help="Days of historical data to load (e.g., 1095 for ~3 seasons)")
    # Model Args
    parser.add_argument("--models", type=str, default="xgboost,random_forest,ridge", help="Comma-separated models to train (e.g., 'xgboost,random_forest')")
    parser.add_argument("--rolling-windows", type=str, default="5,10,20", help="Comma-separated rolling window sizes (e.g., '5,10')")
    parser.add_argument("--h2h-window", type=int, default=7, help="Number of games for H2H features")
    # Training Args
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of data for the final test set (e.g., 0.15 for 15%)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of data for the validation set (used for tuning, taken from remaining data after test split)")
    parser.add_argument("--use-weights", action="store_true", help="Use recency weighting during training")
    parser.add_argument("--weight-method", type=str, default="exponential", choices=["exponential", "half_life"], help="Recency weighting method")
    parser.add_argument("--weight-half-life", type=int, default=90, help="Half-life in days for 'half_life' weights")
    # --- Tuning Args ---
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning and train with model defaults")
    parser.add_argument("--tune-iterations", type=int, default=30, help="Number of iterations for RandomizedSearchCV (e.g., 30-100)")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS, help="Number of splits for TimeSeriesSplit CV (e.g., 5)")
    parser.add_argument("--scoring-metric", type=str, default='neg_mean_absolute_error',
                        help="Scoring metric for tuning (sklearn format, e.g., 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2')")
    # --------------------
    # Analysis Args
    parser.add_argument("--run-analysis", action="store_true", help="Run all optional analysis functions (Placeholders)")
    # Output Args
    parser.add_argument("--visualize", action="store_true", help="Show plots interactively during/after training")
    parser.add_argument("--save-plots", action="store_true", help="Save generated plots to the reports directory")
    # Misc Args
    parser.add_argument("--allow-dummy", action="store_true", help="Allow script to run using dummy classes if module imports fail")
    parser.add_argument("--debug", action="store_true", help="Enable debug level logging")

    cli_args = parser.parse_args()

    # --- Run Main Function ---
    run_training_pipeline(cli_args)