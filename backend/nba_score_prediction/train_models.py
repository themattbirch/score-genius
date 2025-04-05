from __future__ import annotations  # Keep this at the top
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # Keep TYPE_CHECKING block if needed for linters
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform  # For defining parameter distributions
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)

# --- Import evaluation functions ---
from .evaluation import (
    plot_residuals_analysis_detailed, 
    plot_conditional_bias, 
    plot_temporal_bias,
    plot_feature_importances,
    plot_actual_vs_predicted 
)

# --- Database Imports (Optional) ---
try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine = None
    print("WARNING: SQLAlchemy not installed.")

# --- Supabase Import ---
try:
    from supabase import create_client
except ImportError:
    create_client = None
    print("WARNING: Supabase library not installed.")

# --- Conditional XGBoost Import ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost library not found.")

# --- Project-Specific Imports ---
try:
    from .feature_engineering import NBAFeatureEngine
    from .models import (RandomForestScorePredictor, RidgeScorePredictor, compute_recency_weights)
    if XGBOOST_AVAILABLE:
        from .models import XGBoostScorePredictor
    else:
        class XGBoostScorePredictor:
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
    class NBAFeatureEngine: pass
    class XGBoostScorePredictor:
        def __init__(self, *args, **kwargs): pass
        def _build_pipeline(self, *args, **kwargs): return None
        def train(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return None
        def save_model(self, *args, **kwargs): pass
        def load_model(self, *args, **kwargs): pass
        feature_names_in_ = None
        training_timestamp = None
    class RandomForestScorePredictor(XGBoostScorePredictor): pass
    class RidgeScorePredictor(XGBoostScorePredictor): pass
    def compute_recency_weights(*args, **kwargs): return np.ones(len(args[0])) if args else np.array([])
    def plot_feature_importances(*args, **kwargs): logging.error("Dummy plot_feature_importances called!")
    def plot_actual_vs_predicted(*args, **kwargs): logging.error("Dummy plot_actual_vs_predicted called!")
    def plot_residuals_analysis_detailed(*args, **kwargs): logging.error("Dummy plot_residuals_analysis_detailed called!")
    try:
        from backend import config
    except ImportError:
        config = None
        print("ERROR: Could not import config module.")
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
DEFAULT_CV_FOLDS = 5  # Default splits for TimeSeriesSplit

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 0: CLIENT INITIALIZATION
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
# SECTION 1: DATA LOADING
# ==============================================================================
def load_data_source(source_type: str, lookback_days: int, args: argparse.Namespace,
                     db_engine: Optional[Any] = None, supabase_client: Optional["Client"] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Attempting to load data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()
    hist_required_cols = [
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot', 'away_q1', 'away_q2',
        'away_q3', 'away_q4', 'away_ot', 'home_fg_made', 'home_fg_attempted', 'away_fg_made',
        'away_fg_attempted', 'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa', 'home_ft_made',
        'home_ft_attempted', 'away_ft_made', 'away_ft_attempted'
    ]
    hist_numeric_cols = [col for col in hist_required_cols if col not in ['game_id', 'game_date', 'home_team', 'away_team']]
    team_required_cols = ['team_name', 'season', 'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all', 'current_form']
    team_numeric_cols = [col for col in team_required_cols if col not in ['team_name', 'season', 'current_form']]

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
                    logger.debug(f"Retrieved {b_size} historical records...")
                    if b_size < page_size:
                        has_more = False
                    else:
                        start_index += page_size
                if all_historical_data:
                    hist_df = pd.DataFrame(all_historical_data)
                    logger.info(f"Loaded {len(hist_df)} historical records.")
                    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
                    hist_df = hist_df.dropna(subset=['game_date'])
                else:
                    logger.warning(f"No historical data since {threshold_date}.")
            except Exception as e:
                logger.error(f"Error loading historical games from Supabase: {e}", exc_info=True)
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

    if not hist_df.empty:
        missing_hist_cols = [col for col in hist_required_cols if col not in hist_df.columns]
        if missing_hist_cols:
             logger.warning(f"Historical data missing columns: {missing_hist_cols}. Filling with NaN.")
             for col in missing_hist_cols:
                 hist_df[col] = np.nan
        logger.info("Converting historical numeric columns and handling NaNs...")
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                numeric_col = pd.to_numeric(hist_df[col], errors='coerce')
                hist_df[col] = numeric_col.fillna(np.nan)
            else:
                 logger.warning(f"Numeric column '{col}' unexpectedly missing; adding as NaN.")
                 hist_df[col] = np.nan
        if args.debug:
            key_cols = ['home_fg_attempted', 'home_ft_attempted', 'away_fg_attempted', 'away_ft_attempted']
            logger.debug("--- NaN Check After Loading ---")
            for k in key_cols:
                if k in hist_df.columns:
                    nan_count = hist_df[k].isnull().sum()
                    logger.debug(f"NaN count in hist_df['{k}']: {nan_count} ({nan_count*100.0/len(hist_df):.1f}%)")
                else:
                    logger.warning(f"Column '{k}' not found for NaN check.")
        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)
    if not team_stats_df.empty:
        pass

    logger.info(f"Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df

# ==============================================================================
# SECTION 2: UTILITY & HELPER FUNCTIONS
# ==============================================================================
def visualize_recency_weights(dates, weights, title="Recency Weights Distribution", save_path=None):
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

# ==============================================================================
# SECTION 2.1: FEATURE SELECTION UTILITY FUNCTIONS
# ==============================================================================
REFINED_TOP_100_FEATURES = [
    'home_season_avg_pts_against',
    'away_season_avg_pts_for',
    'away_rolling_score_for_mean_10',
    'home_rolling_score_against_mean_20',
    'season_net_rating_diff',
    'season_win_pct_diff',
    'home_rolling_score_against_mean_10',
    'away_season_avg_pts_against',
    'home_season_avg_pts_for',
    'home_rolling_score_for_mean_5',
    'away_rolling_score_against_mean_10',
    'away_season_net_rating',
    'home_rolling_pace_mean_20',
    'away_rolling_pace_mean_10',
    'away_form_win_pct',
    'home_season_win_pct',
    'home_momentum_direction',
    'matchup_avg_away_score',
    'season_pts_for_diff',
    'home_rolling_dreb_pct_mean_20',
    'away_rolling_efg_pct_mean_10',
    'season_pts_against_diff',
    'home_season_net_rating',
    'rolling_margin_diff_mean',
    'away_rolling_pace_std_20',
    'games_last_14_days_away',
    'away_rolling_off_rating_mean_20',
    'home_rolling_def_rating_mean_20',
    'home_rolling_tov_rate_mean_20',
    'matchup_home_win_pct',
    'away_rolling_dreb_pct_std_5',
    'matchup_avg_total_score',
    'away_season_win_pct',
    'home_rolling_ft_rate_std_20',
    'home_rolling_net_rating_mean_20',
    'home_rolling_trb_pct_std_10',
    'home_rolling_tov_rate_std_10',
    'rolling_efg_diff_std',
    'away_rolling_dreb_pct_std_10',
    'home_rolling_ft_rate_std_10',
    'home_rolling_tov_rate_std_5',
    'home_rolling_ft_rate_std_5',
    'home_rolling_dreb_pct_std_10',
    'rolling_eff_diff_std',
    'away_rolling_net_rating_std_20',
    'home_rolling_oreb_pct_std_20',
    'home_rolling_ft_rate_mean_20',
    'away_rolling_ft_rate_std_20',
    'home_rolling_trb_pct_mean_20',
    'away_rolling_tov_rate_std_20',
    'matchup_streak',
    'away_rolling_score_against_std_20',
    'away_rolling_trb_pct_std_20',
    'home_rolling_dreb_pct_std_20',
    'is_back_to_back_away',
    'away_rolling_net_rating_mean_5',
    'home_rolling_net_rating_std_20',
    'rolling_eff_diff_mean',
    'away_rolling_off_rating_std_10',
    'away_rolling_oreb_pct_std_20',
    'away_rolling_momentum_ewma_mean_20',
    'away_rolling_def_rating_std_10',
    'home_rolling_net_rating_std_10',
    'home_rolling_pace_std_10',
    'away_rolling_dreb_pct_std_20',
    'matchup_avg_home_score',
    'home_rolling_efg_pct_std_5',
    'away_rolling_trb_pct_std_10',
    'home_rolling_pace_std_20',
    'away_rolling_def_rating_std_20',
    'home_rolling_off_rating_std_20',
    'rolling_momentum_diff_mean',
    'away_rolling_score_for_std_5',
    'rolling_trb_diff_mean',
    'home_rolling_momentum_ewma_mean_20',
    'home_rolling_score_for_std_20',
    'home_rolling_ft_rate_mean_10',
    'games_last_14_days_home',
    'matchup_avg_point_diff',
    'away_rolling_dreb_pct_mean_10',
    'away_rolling_efg_pct_mean_20',
    'rest_days_away',
    'away_rolling_ft_rate_std_5',
    'away_rolling_tov_rate_std_5',
    'rest_advantage',
    'home_rolling_oreb_pct_std_5',
    'home_rolling_momentum_ewma_std_20',
    'home_rolling_trb_pct_mean_10',
    'away_rolling_def_rating_mean_5',
    'away_rolling_momentum_ewma_std_20',
    'rolling_ft_rate_diff_std',
    'away_rolling_efg_pct_std_5',
    'away_rolling_oreb_pct_std_5',
    'home_rolling_def_rating_std_10',
    'away_rolling_efg_pct_std_20',
    'away_rolling_dreb_pct_mean_5',
    'rolling_dreb_diff_std',
    'away_rolling_net_rating_std_5',
    'away_rolling_tov_rate_mean_10',
    'rolling_trb_diff_std'
]

def select_features(feature_engine: Optional[Any], data_df: pd.DataFrame, target_context: str = 'pregame_score') -> List[str]:
    """
    Selects features based on a predefined list of top features (REFINED_TOP_100_FEATURES).
    Only returns features present in the provided DataFrame.
    
    Args:
        feature_engine: An instance of NBAFeatureEngine (not used here but maintained for signature consistency).
        data_df: The DataFrame containing all generated features.
        target_context: Context string (unused in this implementation).
    
    Returns:
        A list of feature names from REFINED_TOP_100_FEATURES that exist in data_df.
    """
    logger.info(f"Selecting features based on predefined REFINED_TOP_100_FEATURES for context: {target_context}")
    if data_df is None or data_df.empty:
        logger.error("Input DataFrame is empty. Cannot select features.")
        return []
    
    available_features = [feature for feature in REFINED_TOP_100_FEATURES if feature in data_df.columns]
    num_expected = len(REFINED_TOP_100_FEATURES)
    num_available = len(available_features)
    
    if num_available == 0:
        logger.error("CRITICAL: None of the predefined features were found in the input DataFrame!")
        return []
    elif num_available < num_expected:
        missing_features = set(REFINED_TOP_100_FEATURES) - set(available_features)
        logger.warning(f"Found {num_available} out of {num_expected} predefined features. Missing: {missing_features}")
    else:
        logger.info(f"Successfully selected all {num_available} predefined features.")
    
    logger.debug(f"Final selected features sample: {available_features[:10]}")
    return available_features

# ==============================================================================
# SECTION 3: CORE METRIC & CUSTOM LOSS FUNCTIONS
# ==============================================================================
def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred):
            logger.error(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true) == 0:
            logger.warning("Empty input arrays for metrics.")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}")
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

def nba_score_loss(y_true, y_pred, spread_weight=0.6, total_weight=0.4):
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)):
            y_pred = y_pred.values
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 2)
        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("nba_score_loss requires 2 columns (home, away).")
            return np.inf
        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        diff_error = ((true_home - true_away) - (pred_home - pred_away)) ** 2
        total_error = ((true_home + true_away) - (pred_home + pred_away)) ** 2
        return np.mean(spread_weight * diff_error + total_weight * total_error)
    except Exception as e:
        logger.error(f"Error in nba_score_loss: {e}")
        return np.inf

def nba_distribution_loss(y_true, y_pred):
    try:
        if isinstance(y_pred, (pd.DataFrame, pd.Series)):
            y_pred = y_pred.values
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 2)
        if y_pred.shape[1] != 2:
            logger.error("nba_distribution_loss requires 2 columns (home, away).")
            return np.inf
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        home_mean, home_std = 114, 13.5
        away_mean, away_std = 112, 13.5
        diff_mean, diff_std = 2.5, 13.5
        total_mean, total_std = 226, 23.0
        pred_diff, pred_total = pred_home - pred_away, pred_home + pred_away
        z_home = ((pred_home - home_mean) / home_std) ** 2 if home_std > 0 else np.zeros_like(pred_home)
        z_away = ((pred_away - away_mean) / away_std) ** 2 if away_std > 0 else np.zeros_like(pred_away)
        z_diff = ((pred_diff - diff_mean) / diff_std) ** 2 if diff_std > 0 else np.zeros_like(pred_diff)
        z_total = ((pred_total - total_mean) / total_std) ** 2 if total_std > 0 else np.zeros_like(pred_total)
        return np.mean(0.2 * z_home + 0.2 * z_away + 0.3 * z_diff + 0.3 * z_total)
    except Exception as e:
        logger.error(f"Error in nba_distribution_loss: {e}")
        return np.inf

def combined_nba_loss(y_true, y_pred, accuracy_weight=0.7, distribution_weight=0.3):
    score_loss = nba_score_loss(y_true, y_pred)
    dist_loss = nba_distribution_loss(y_true, y_pred)
    if np.isinf(score_loss) or np.isinf(dist_loss):
        return np.inf
    return accuracy_weight * score_loss + distribution_weight * dist_loss

nba_score_scorer = make_scorer(nba_score_loss, greater_is_better=False, needs_proba=False)
nba_distribution_scorer = make_scorer(nba_distribution_loss, greater_is_better=False, needs_proba=False)
combined_scorer = make_scorer(combined_nba_loss, greater_is_better=False, needs_proba=False)

def calculate_betting_metrics(y_true, y_pred, vegas_lines: Optional[pd.DataFrame] = None):
    metrics = {}
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)):
            y_pred = y_pred.values
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 2)
        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("Betting metrics require 2 columns (home, away).")
            return metrics
        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        true_diff = true_home - true_away
        pred_diff = pred_home - pred_away
        metrics['win_prediction_accuracy'] = np.mean((true_diff > 0) == (pred_diff > 0))
    except Exception as e:
        logger.error(f"Error calculating betting metrics: {e}")
    return metrics

# ==============================================================================
# SECTION 4: QUARTERLY MODEL TRAINING UTILITIES (Placeholders)
# ==============================================================================
def initialize_trainable_models(*args, **kwargs): pass
def train_quarter_models(*args, **kwargs): pass

# ==============================================================================
# SECTION 5: HYPERPARAMETER TUNING & MAIN PREDICTOR TRAINING
# ==============================================================================
import time
import logging
import traceback
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (make_scorer, mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import randint, uniform

# Assuming these are correctly imported elsewhere as in the original code
from .evaluation import (
    plot_residuals_analysis_detailed,
    plot_conditional_bias,
    plot_temporal_bias,
    plot_feature_importances,
    plot_actual_vs_predicted
)
from .models import compute_recency_weights # And other model classes
from .feature_engineering import NBAFeatureEngine # For type hint/consistency if needed later

# Placeholder for XGBoost availability check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- Constants (assuming these are defined elsewhere) ---
logger = logging.getLogger(__name__) # Assume logger is configured
MAIN_MODELS_DIR = Path("./models/saved") # Example path
REPORTS_DIR = Path("./reports") # Example path
SEED = 42
TARGET_COLUMNS = ['home_score', 'away_score']
# --- END Constants ---


# ==============================================================================
# SECTION 5: HYPERPARAMETER TUNING & MAIN PREDICTOR TRAINING (REWRITTEN)
# ==============================================================================
def tune_and_evaluate_predictor(
    predictor_class,
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str,
    feature_list: List[str], # Takes the already selected feature list as input
    param_dist: Optional[Dict[str, Any]],
    n_iter: int = 10,
    n_splits: int = 5,
    scoring: str = 'neg_mean_absolute_error',
    use_recency_weights: bool = False,
    weight_method: str = 'exponential',
    weight_half_life: int = 90,
    visualize: bool = True,
    save_plots: bool = False
) -> Optional[Dict]:
    """
    Tunes hyperparameters using RandomizedSearchCV on the training set,
    trains the final model on the combined training and validation sets using
    the best parameters (or defaults if tuning skipped/failed), and evaluates
    on the test set.

    Args:
        predictor_class: The class of the predictor model to use (e.g., XGBoostScorePredictor).
        X_train, y_train_home, y_train_away: Training data and targets.
        X_val, y_val_home, y_val_away: Validation data and targets.
        X_test, y_test_home, y_test_away: Test data and targets.
        model_name_prefix: Base name for the model (e.g., 'xgboost').
        feature_list: List of feature names selected *before* calling this function.
        param_dist: Parameter distribution for RandomizedSearchCV. If None, tuning is skipped.
        n_iter: Number of iterations for RandomizedSearchCV.
        n_splits: Number of splits for TimeSeriesSplit cross-validation.
        scoring: Scoring metric string for RandomizedSearchCV.
        use_recency_weights: Whether to apply sample weights based on game date.
        weight_method: Method for calculating recency weights ('exponential' or 'half_life').
        weight_half_life: Half-life in days for weight calculation.
        visualize: Whether to display plots interactively.
        save_plots: Whether to save generated plots.

    Returns:
        A dictionary containing evaluation metrics and model information, or None if failed.
    """
    # --- REMOVED INCORRECT RELATIVE IMPORTS FOR .metrics and .utility ---
    # The necessary functions (calculate_regression_metrics, visualize_recency_weights, etc.)
    # are assumed to be defined in the global scope of train_models.py as per the original code.

    logger.info(f"tune_and_evaluate_predictor received feature_list with {len(feature_list)} features.")
    model_full_name = f"{model_name_prefix}_score_predictor"

    # --- Initial Checks ---
    if predictor_class is None:
        logger.error(f"Predictor class for {model_full_name} is None. Aborting.")
        return None
    if model_name_prefix == 'xgboost' and not XGBOOST_AVAILABLE:
        logger.error("XGBoost requested but unavailable. Aborting.")
        return None
    if not feature_list:
        logger.error(f"Input feature list is empty for {model_full_name}. Aborting.")
        return None

    logger.info(f"--- Tuning and Training {predictor_class.__name__} ({model_full_name}) ---")
    start_tune_time = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {}

    # --- Ensure Feature List is Unique ---
    feature_list_unique = list(pd.Index(feature_list).unique())
    if len(feature_list_unique) != len(feature_list):
        logger.warning(f"Duplicate features found in input list. Using {len(feature_list_unique)} unique features.")
    logger.debug(f"Using unique feature list ({len(feature_list_unique)} features): {feature_list_unique[:20]}")


    # --- Combine Train and Validation Data for Final Fit ---
    train_val_date_cols = []
    if use_recency_weights and 'game_date' in X_train.columns and 'game_date' in X_val.columns:
        train_val_date_cols = ['game_date']
    cols_train = [col for col in feature_list_unique + train_val_date_cols if col in X_train.columns]
    cols_val = [col for col in feature_list_unique + train_val_date_cols if col in X_val.columns]
    try:
        X_train_full = pd.concat([X_train[cols_train], X_val[cols_val]], ignore_index=False)
        y_train_home_full = pd.concat([y_train_home, y_val_home])
        y_train_away_full = pd.concat([y_train_away, y_val_away])
        y_train_home_full = y_train_home_full.loc[X_train_full.index]
        y_train_away_full = y_train_away_full.loc[X_train_full.index]
        logger.info(f"Combined Train+Val size for final fit: {len(X_train_full)} samples.")
    except Exception as concat_err:
        logger.error(f"Error combining Train and Val sets: {concat_err}", exc_info=True)
        return None

    # --- Prepare Data Specifically for Tuning (Using only Train set) ---
    logger.debug(f"Preparing data for hyperparameter tuning using X_train.")
    try:
        X_tune_features = X_train[feature_list_unique].copy()
        y_tune = y_train_home.loc[X_tune_features.index].copy()
        logger.debug(f"Tuning data shapes: X={X_tune_features.shape}, y={y_tune.shape}")
    except KeyError as ke:
        logger.error(f"KeyError preparing tuning data. Missing features? {ke}", exc_info=True)
        missing_in_xtrain = set(feature_list_unique) - set(X_train.columns)
        if missing_in_xtrain: logger.error(f"Features missing in X_train: {missing_in_xtrain}")
        return None
    except Exception as prep_err:
         logger.error(f"Error preparing tuning data: {prep_err}", exc_info=True)
         return None

    # --- Calculate Sample Weights for Tuning (if enabled) ---
    tune_fit_params = {}
    if use_recency_weights:
        if 'game_date' in X_train.columns:
            logger.info("Calculating recency weights for tuning dataset (X_train)...")
            dates_for_weights_tune = X_train.loc[X_tune_features.index, 'game_date']
            # Use compute_recency_weights directly (assuming it's globally available)
            sample_weights_tune = compute_recency_weights(dates_for_weights_tune, method=weight_method, half_life=weight_half_life)
            if sample_weights_tune is not None and len(sample_weights_tune) == len(X_tune_features):
                try:
                    temp_predictor = predictor_class()
                    temp_pipeline = temp_predictor._build_pipeline({})
                    if temp_pipeline and hasattr(temp_pipeline, 'steps') and len(temp_pipeline.steps) > 0:
                        model_step_name = temp_pipeline.steps[-1][0]
                        weight_param_name = f"{model_step_name}__sample_weight"
                        tune_fit_params[weight_param_name] = sample_weights_tune
                        logger.info(f"Sample weights for tuning prepared using key: '{weight_param_name}'")
                    else:
                        logger.warning("Could not determine model step name for sample weights. Weights may not be applied during tuning.")
                except Exception as e:
                    logger.warning(f"Error determining weight parameter name for tuning: {e}. Weights may not be applied.")
            else:
                logger.warning(f"Failed to compute or align tuning sample weights. Proceeding without tuning weights.")
        else:
            logger.warning("'use_recency_weights' enabled but 'game_date' column missing in X_train. Cannot apply weights during tuning.")

    # --- Hyperparameter Tuning (RandomizedSearchCV) ---
    if param_dist:
        logger.info(f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={n_splits}, scoring='{scoring}')...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        try:
            temp_predictor = predictor_class()
            search_pipeline = temp_predictor._build_pipeline({})
            if search_pipeline is None:
                raise RuntimeError(f"Could not build pipeline for tuning {predictor_class.__name__}")
            rs = RandomizedSearchCV( # ... (rest of RandomizedSearchCV setup) ...
                estimator=search_pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring,
                cv=tscv,
                n_jobs=-1,
                verbose=3,
                random_state=SEED,
                error_score='raise',
                refit=False
            )
            rs.fit(X_tune_features, y_tune, **tune_fit_params)
            try: # Log CV results
                cv_results_df = pd.DataFrame(rs.cv_results_).sort_values(by='rank_test_score')
                cols_to_show = ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
                logger.info("--- Top CV Results ---")
                logger.info("\n" + cv_results_df[cols_to_show].head().to_string())
            except Exception as report_e:
                logger.warning(f"Could not print detailed CV results: {report_e}")

            best_params_raw = rs.best_params_
            metrics['best_cv_score'] = rs.best_score_
            logger.info(f"Tuning complete. Best CV Score ({scoring}): {metrics['best_cv_score']:.4f}")
            logger.info(f"Best Raw Params (with prefix): {best_params_raw}")
            best_params_final = {k.split('__', 1)[1]: v for k, v in best_params_raw.items()}
            metrics['best_params'] = best_params_final
            logger.info(f"Best Final Params (no prefix): {best_params_final}")
        except Exception as search_e:
            logger.error(f"RandomizedSearchCV failed: {search_e}", exc_info=True)
            logger.warning("Proceeding with default parameters due to tuning error.")
            best_params_final = {}
            metrics['best_cv_score'] = None
            metrics['best_params'] = 'default (tuning failed)'
        tune_duration = time.time() - start_tune_time
        logger.info(f"Tuning finished in {tune_duration:.2f} seconds.")
        metrics['tuning_duration'] = tune_duration
    else:
        logger.info("Skipping hyperparameter tuning (no parameter distribution provided). Using defaults.")
        metrics['best_cv_score'] = None
        metrics['best_params'] = 'default (tuning skipped)'
        metrics['tuning_duration'] = 0.0

    # --- Train Final Model on Combined Train+Val Data ---
    logger.info("Training final model on combined Train+Val data using best/default parameters...")
    final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
    try:
        X_train_full_for_fit = X_train_full[feature_list_unique].copy()
        y_train_home_full_aligned = y_train_home_full
        y_train_away_full_aligned = y_train_away_full
        logger.info(f"Shape of final training features X_train_full_for_fit: {X_train_full_for_fit.shape}")
    except KeyError as ke:
        logger.error(f"KeyError preparing final training data. Missing features? {ke}", exc_info=True)
        missing_in_xtf = set(feature_list_unique) - set(X_train_full.columns)
        if missing_in_xtf: logger.error(f"Features missing in X_train_full: {missing_in_xtf}")
        return None
    except Exception as final_prep_err:
        logger.error(f"Error preparing final training data: {final_prep_err}", exc_info=True)
        return None

    # --- Calculate Sample Weights for Final Training (if enabled) ---
    final_sample_weights = None
    if use_recency_weights:
        if 'game_date' in X_train_full.columns:
            dates_for_weights_full = X_train_full.loc[X_train_full_for_fit.index, 'game_date']
            logger.info("Calculating recency weights for final training dataset (Train+Val)...")
            # Use compute_recency_weights directly
            final_sample_weights = compute_recency_weights(dates_for_weights_full, method=weight_method, half_life=weight_half_life)
            if final_sample_weights is None or len(final_sample_weights) != len(X_train_full_for_fit):
                logger.warning(f"Final sample weights computation failed or length mismatch. Training without weights.")
                final_sample_weights = None
            else:
                logger.info(f"Final sample weights computed ({len(final_sample_weights)} values). Min: {np.min(final_sample_weights):.4f}, Max: {np.max(final_sample_weights):.4f}, Mean: {np.mean(final_sample_weights):.4f}")
                if visualize:
                    # --- MODIFIED: Removed the incorrect import ---
                    try:
                        # Call visualize_recency_weights directly
                        save_path = REPORTS_DIR / f"{model_full_name}_final_weights.png" if save_plots else None
                        visualize_recency_weights(dates_for_weights_full, final_sample_weights, title=f"Final Training Weights - {model_full_name}", save_path=save_path)
                    # Keep general exception handling for the plotting call itself
                    except NameError:
                         logger.error("visualize_recency_weights function not found. Ensure it's defined or imported correctly elsewhere.")
                    except Exception as plot_weight_e:
                        logger.error(f"Error visualizing final weights: {plot_weight_e}")
        else:
            logger.warning("'use_recency_weights' enabled but 'game_date' not found in final combined training data. Cannot apply final weights.")

    # --- Fit the Final Model ---
    try:
        train_start_time = time.time()
        final_predictor.train( # ... (rest of train call) ...
            X_train=X_train_full_for_fit,
            y_train_home=y_train_home_full_aligned,
            y_train_away=y_train_away_full_aligned,
            hyperparams_home=best_params_final,
            hyperparams_away=best_params_final,
            sample_weights=final_sample_weights
        )
        train_duration = time.time() - train_start_time
        metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', train_duration)
        logger.info(f"Final model training completed in {metrics['training_duration_final']:.2f} seconds.")
        metrics['feature_count'] = len(getattr(final_predictor, 'feature_names_in_', feature_list_unique))
        if not getattr(final_predictor, 'feature_names_in_', None):
             logger.warning("Could not retrieve feature_names_in_ from trained model. Using input list length.")
    except Exception as train_e:
        logger.error(f"Training final model failed: {train_e}", exc_info=True)
        metrics['training_duration_final'] = None
        metrics['feature_count'] = len(feature_list_unique)
        return metrics # Return partial metrics

    # --- Save the Final Model ---
    try:
        timestamp = getattr(final_predictor, 'training_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_filename = f"{model_full_name}_tuned_{timestamp}.joblib"
        save_path = final_predictor.save_model(filename=save_filename)
        if save_path:
            logger.info(f"Final tuned {model_full_name} saved to {save_path}")
            metrics['save_path'] = str(save_path)
        else:
            logger.error(f"Final model saving returned no path for {model_full_name}.")
            metrics['save_path'] = "Save failed (no path returned)"
    except Exception as save_e:
        logger.error(f"Error saving final model: {save_e}", exc_info=True)
        metrics['save_path'] = f"Save failed ({type(save_e).__name__})"

    # --- Evaluate Final Model on Test Set ---
    logger.info(f"Evaluating final tuned {model_full_name} on test set ({len(X_test)} samples)...")
    try:
        X_test_final = X_test[feature_list_unique].copy()
        logger.debug(f"Shape of test features X_test_final: {X_test_final.shape}")
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or 'predicted_home_score' not in predictions_df_test.columns or 'predicted_away_score' not in predictions_df_test.columns:
            raise ValueError(f"Final prediction on test set failed or returned unexpected format for {model_full_name}.")
        pred_home_test = predictions_df_test['predicted_home_score']
        pred_away_test = predictions_df_test['predicted_away_score']
        y_test_home_aligned = y_test_home.loc[pred_home_test.index]
        y_test_away_aligned = y_test_away.loc[pred_away_test.index]

        # Use globally available metric functions
        test_metrics_home = calculate_regression_metrics(y_test_home_aligned, pred_home_test)
        test_metrics_away = calculate_regression_metrics(y_test_away_aligned, pred_away_test)
        metrics['test_mae_home'] = test_metrics_home.get('mae', np.nan)
        metrics['test_rmse_home'] = test_metrics_home.get('rmse', np.nan)
        # ... (rest of metric calculations using globally available functions) ...
        metrics['test_r2_home'] = test_metrics_home.get('r2', np.nan)
        metrics['test_mae_away'] = test_metrics_away.get('mae', np.nan)
        metrics['test_rmse_away'] = test_metrics_away.get('rmse', np.nan)
        metrics['test_r2_away'] = test_metrics_away.get('r2', np.nan)
        y_true_comb_test = np.vstack((y_test_home_aligned.values, y_test_away_aligned.values)).T
        y_pred_comb_test = np.vstack((pred_home_test.values, pred_away_test.values)).T
        metrics['test_mae_total'] = mean_absolute_error(y_test_home_aligned + y_test_away_aligned, pred_home_test + pred_away_test)
        metrics['test_mae_diff'] = mean_absolute_error(y_test_home_aligned - y_test_away_aligned, pred_home_test - pred_away_test)
        metrics['test_nba_score_loss'] = nba_score_loss(y_true_comb_test, y_pred_comb_test)
        metrics['test_nba_dist_loss'] = nba_distribution_loss(y_true_comb_test, y_pred_comb_test)
        metrics['test_combined_loss'] = combined_nba_loss(y_true_comb_test, y_pred_comb_test)
        metrics['betting_metrics'] = calculate_betting_metrics(y_true_comb_test, y_pred_comb_test, vegas_lines=None)

        logger.info(f"FINAL Test MAE : Home={metrics['test_mae_home']:.3f}, Away={metrics['test_mae_away']:.3f}, Total={metrics['test_mae_total']:.3f}, Diff={metrics['test_mae_diff']:.3f}")
        # ... (rest of logging) ...
        logger.info(f"FINAL Test RMSE: Home={metrics['test_rmse_home']:.3f}, Away={metrics['test_rmse_away']:.3f}")
        logger.info(f"FINAL Test R2  : Home={metrics['test_r2_home']:.3f}, Away={metrics['test_r2_away']:.3f}")
        logger.info(f"FINAL Custom Losses: Score={metrics['test_nba_score_loss']:.3f}, Dist={metrics['test_nba_dist_loss']:.3f}, Combined={metrics['test_combined_loss']:.3f}")
        logger.info(f"FINAL Betting Metrics: {metrics['betting_metrics']}")

    except KeyError as ke: # ... (rest of error handling) ...
        logger.error(f"KeyError evaluating on test set. Missing features? {ke}", exc_info=True)
        missing_in_xtest = set(feature_list_unique) - set(X_test.columns)
        if missing_in_xtest: logger.error(f"Features missing in X_test: {missing_in_xtest}")
        for k in ['test_mae_home', 'test_rmse_home', 'test_r2_home', 'test_mae_away', 'test_rmse_away', 'test_r2_away', 'test_mae_total', 'test_mae_diff', 'test_nba_score_loss', 'test_nba_dist_loss', 'test_combined_loss']: metrics[k] = metrics.get(k, np.nan)
        metrics['betting_metrics'] = metrics.get('betting_metrics', {})
    except Exception as test_eval_e: # ... (rest of error handling) ...
        logger.error(f"Failed evaluating final model on test set: {test_eval_e}", exc_info=True)
        for k in ['test_mae_home', 'test_rmse_home', 'test_r2_home', 'test_mae_away', 'test_rmse_away', 'test_r2_away', 'test_mae_total', 'test_mae_diff', 'test_nba_score_loss', 'test_nba_dist_loss', 'test_combined_loss']: metrics[k] = metrics.get(k, np.nan)
        metrics['betting_metrics'] = metrics.get('betting_metrics', {})

    # --- (Optional) Evaluate Final Model on Training Set ---
    logger.info(f"Evaluating final tuned {model_full_name} on training set (Train+Val) ({len(X_train_full_for_fit)} samples)...")
    try: # ... (rest of training set evaluation using global metric functions) ...
        predictions_df_train = final_predictor.predict(X_train_full_for_fit)
        if predictions_df_train is None or 'predicted_home_score' not in predictions_df_train.columns:
            raise ValueError(f"Final prediction on train set failed for {model_full_name}.")
        pred_home_train = predictions_df_train['predicted_home_score']
        pred_away_train = predictions_df_train['predicted_away_score']
        train_metrics_home = calculate_regression_metrics(y_train_home_full_aligned, pred_home_train)
        train_metrics_away = calculate_regression_metrics(y_train_away_full_aligned, pred_away_train)
        metrics['train_mae_home'] = train_metrics_home.get('mae', np.nan)
        metrics['train_rmse_home'] = train_metrics_home.get('rmse', np.nan)
        # ... etc ...
        metrics['train_r2_home'] = train_metrics_home.get('r2', np.nan)
        metrics['train_mae_away'] = train_metrics_away.get('mae', np.nan)
        metrics['train_rmse_away'] = train_metrics_away.get('rmse', np.nan)
        metrics['train_r2_away'] = train_metrics_away.get('r2', np.nan)
        metrics['train_mae_total'] = mean_absolute_error(y_train_home_full_aligned + y_train_away_full_aligned, pred_home_train + pred_away_train)
        metrics['train_mae_diff'] = mean_absolute_error(y_train_home_full_aligned - y_train_away_full_aligned, pred_home_train - pred_away_train)
        logger.info(f"FINAL Train MAE : Home={metrics['train_mae_home']:.3f}, Away={metrics['train_mae_away']:.3f}, Total={metrics['train_mae_total']:.3f}, Diff={metrics['train_mae_diff']:.3f}")
        logger.info(f"FINAL Train RMSE: Home={metrics['train_rmse_home']:.3f}, Away={metrics['train_rmse_away']:.3f}")
        logger.info(f"FINAL Train R2  : Home={metrics['train_r2_home']:.3f}, Away={metrics['train_r2_away']:.3f}")

    except Exception as train_eval_e: # ... (rest of error handling) ...
        logger.error(f"Failed evaluating on training set: {train_eval_e}", exc_info=True)
        metrics['train_mae_home'] = np.nan
        metrics['train_rmse_home'] = np.nan
        # ... etc ...
        metrics['train_r2_home'] = np.nan
        metrics['train_mae_away'] = np.nan
        metrics['train_rmse_away'] = np.nan
        metrics['train_r2_away'] = np.nan
        metrics['train_mae_total'] = np.nan
        metrics['train_mae_diff'] = np.nan

    # --- Store Sample Sizes ---
    metrics['samples_train_final'] = len(X_train_full_for_fit)
    metrics['samples_test'] = len(X_test_final) if 'X_test_final' in locals() else len(X_test)

    # --- Generate Performance Plots (if enabled) ---
    if visualize or save_plots:
        # Use timestamp from save_path if available, otherwise generate one fallback
        timestamp_str = metrics.get('save_path')
        if timestamp_str and isinstance(timestamp_str, str):
             # Try to extract timestamp like YYYYMMDD_HHMMSS from filename part
             filename_part = Path(timestamp_str).name
             ts_parts = filename_part.split('_')
             # Look for a part that matches the datetime format
             potential_ts = [p.split('.')[0] for p in ts_parts if p.replace('.', '').isdigit() and len(p.split('.')[0]) == 15]
             timestamp = potential_ts[0] if potential_ts else datetime.now().strftime("%Y%m%d_%H%M%S") # Fallback if format not found
        else:
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Fallback if save_path missing

        plot_dir = REPORTS_DIR / f"{model_full_name}_tuned_performance_{timestamp}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating performance plots for {model_full_name} in {plot_dir}")

        if 'pred_home_test' in locals() and 'y_test_home_aligned' in locals():
            try:
                # --- CORRECTED FIRST CALL (metrics_dict added back) ---
                plot_actual_vs_predicted(
                    y_true=y_test_home_aligned,
                    y_pred=pred_home_test,
                    title=f"Tuned {model_full_name} - Test Actual vs Pred (Home)",
                    # metrics_dict added back:
                    metrics_dict={
                        'rmse': metrics.get('test_rmse_home', np.nan),
                        'r2': metrics.get('test_r2_home', np.nan),
                        'mae': metrics.get('test_mae_home', np.nan)
                    },
                    save_path=plot_dir / "test_actual_vs_pred_home.png" if save_plots else None,
                    show_plot=visualize # Correct parameter name
                )
                # --- CORRECTED SECOND CALL (metrics_dict added back) ---
                plot_actual_vs_predicted(
                    y_true=y_test_away_aligned,
                    y_pred=pred_away_test,
                    title=f"Tuned {model_full_name} - Test Actual vs Pred (Away)",
                    # metrics_dict added back:
                    metrics_dict={
                        'rmse': metrics.get('test_rmse_away', np.nan),
                        'r2': metrics.get('test_r2_away', np.nan),
                        'mae': metrics.get('test_mae_away', np.nan)
                    },
                    save_path=plot_dir / "test_actual_vs_pred_away.png" if save_plots else None,
                    show_plot=visualize # Correct parameter name
                )

                # --- Residual Analysis Plots (show_plots changed to show_plot) ---
                plot_residuals_analysis_detailed(
                    y_true=y_test_home_aligned,
                    y_pred=pred_home_test,
                    title_prefix=f"Tuned {model_full_name} (Home) - Test Set",
                    save_dir=plot_dir if save_plots else None,
                    show_plot=visualize # Changed from show_plots
                )
                plot_residuals_analysis_detailed(
                    y_true=y_test_away_aligned,
                    y_pred=pred_away_test,
                    title_prefix=f"Tuned {model_full_name} (Away) - Test Set",
                    save_dir=plot_dir if save_plots else None,
                    show_plot=visualize # Changed from show_plots
                )

                # --- Feature Importance Plots (show_plots changed to show_plot) ---
                logger.info(f"Attempting to generate feature importance plots for {model_full_name}...")
                pipeline_home = getattr(final_predictor, 'pipeline_home', None)
                pipeline_away = getattr(final_predictor, 'pipeline_away', None)
                features_in = getattr(final_predictor, 'feature_names_in_', feature_list_unique)

                if pipeline_home and pipeline_away and features_in:
                    models_to_plot = {f"{model_full_name}_Home": pipeline_home, f"{model_full_name}_Away": pipeline_away}
                    plot_feature_importances(
                         models_dict=models_to_plot,
                         feature_names=features_in,
                         top_n=30,
                         plot_groups=True,
                         save_dir=plot_dir / "feature_importance" if save_plots else None,
                         show_plot=visualize # Changed from show_plots
                    )
                else:
                    logger.warning("Missing final pipeline components or feature names; cannot generate feature importance plots.")

            except NameError as ne:
                logger.error(f"Plotting function not found: {ne}. Ensure it's defined or imported correctly elsewhere.")
            except Exception as plot_e:
                logger.error(f"Failed generating one or more plots: {plot_e}", exc_info=True)

        else:
            logger.warning("Skipping plot generation as test predictions are unavailable.")
    # --- END OF 'if visualize or save_plots:' block ---

    # --- Finalize and Return ---
    metrics['total_duration'] = time.time() - start_tune_time
    logger.info(f"--- Finished Tuning & Training {model_full_name} in {metrics['total_duration']:.2f}s ---")
    return metrics

# ==============================================================================
# SECTION 6: EVALUATION & ANALYSIS FUNCTIONS (Placeholders)
# ==============================================================================
def analyze_main_predictors_learning_curves(*args, **kwargs): pass

# ==============================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# ==============================================================================
def run_training_pipeline(args):
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

    logger.info("Initializing Feature Engine...")
    feature_engine = NBAFeatureEngine(debug=args.debug)
    logger.info("Generating features for ALL historical data...")
    rolling_windows_list = [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [5, 10]
    logger.info(f"Using rolling windows: {rolling_windows_list}")
    logger.info("Generating features for ALL historical data...")
    features_df = feature_engine.generate_all_features(
        df=historical_df.copy(),
        historical_games_df=historical_df.copy(),
        team_stats_df=team_stats_df.copy() if not team_stats_df.empty else None,
        rolling_windows=rolling_windows_list,
        h2h_window=args.h2h_window
    )
    if features_df.empty:
        logger.error("Feature generation failed. Exiting.")
        sys.exit(1)

    logger.info(f"Shape of features_df before duplicate removal: {features_df.shape}")
    if features_df.columns.duplicated().any():
        logger.warning("Duplicate column names found in features_df! Removing duplicates, keeping first occurrence.")
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep='first')]
        logger.info(f"Shape after duplicate removal: {features_df.shape}")
    else:
        logger.info("No duplicate column names found in features_df.")

    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    rows_dropped = initial_rows - len(features_df)
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows due to missing target values.")
    if features_df.empty:
        logger.error("No rows remaining after dropping missing targets. Exiting.")
        sys.exit(1)
    logger.info(f"Feature generation complete. Final shape: {features_df.shape}")

    logger.info("Splitting data (time-based)...")
    features_df = features_df.sort_values('game_date').reset_index(drop=True)
    n = len(features_df)
    test_split_idx = int(n * (1 - args.test_size))
    val_split_frac = min(args.val_size, 1.0 - args.test_size - 0.01)
    val_split_idx = int(n * (1 - args.test_size - val_split_frac))
    train_df = features_df.iloc[:val_split_idx]
    val_df = features_df.iloc[val_split_idx:test_split_idx]
    test_df = features_df.iloc[test_split_idx:]
    logger.info(f"Data Split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        logger.error("Empty data split generated. Exiting.")
        sys.exit(1)

    logger.info("Selecting features based on training data...")
    selected_features = select_features(feature_engine, train_df)
    if not selected_features:
        logger.error("No features selected. Exiting.")
        sys.exit(1)
    logger.info(f"Selected {len(selected_features)} features initially.")
    logger.debug(f"Initial selected features sample: {selected_features[:10]}")

    train_cols = selected_features[:]
    if args.use_weights and 'game_date' in train_df.columns:
        train_cols.append('game_date')
    val_cols = selected_features[:]
    if args.use_weights and 'game_date' in val_df.columns:
        val_cols.append('game_date')
    selected_features_unique = list(pd.Index(selected_features).unique())

    logger.debug(f"Unique columns in X_train: {len(list(pd.Index(train_cols).unique()))}")
    logger.debug(f"Unique columns in X_val: {len(list(pd.Index(val_cols).unique()))}")
    logger.debug(f"Unique columns in X_test: {len(selected_features_unique)}")

    X_train = train_df[list(pd.Index(train_cols).unique())].copy()
    y_train_home = train_df[TARGET_COLUMNS[0]].copy()
    y_train_away = train_df[TARGET_COLUMNS[1]].copy()
    X_val = val_df[list(pd.Index(val_cols).unique())].copy()
    y_val_home = val_df[TARGET_COLUMNS[0]].copy()
    y_val_away = val_df[TARGET_COLUMNS[1]].copy()
    X_test = test_df[selected_features_unique].copy()
    y_test_home = test_df[TARGET_COLUMNS[0]].copy()
    y_test_away = test_df[TARGET_COLUMNS[1]].copy()

    logger.debug(f"X_train shape: {X_train.shape}, unique cols: {len(X_train.columns.unique())}")
    logger.debug(f"X_val shape: {X_val.shape}, unique cols: {len(X_val.columns.unique())}")
    logger.debug(f"X_test shape: {X_test.shape}, unique cols: {len(X_test.columns.unique())}")

    # --- Define Parameter Distributions for RandomizedSearchCV ---
    XGB_PARAM_DIST = {
        'xgb__max_depth': randint(3, 7),
        'xgb__learning_rate': uniform(0.01, 0.1),
        'xgb__min_child_weight': randint(3, 10),
        'xgb__gamma': [0.0, 0.1, 0.5, 1.0, 2.0],
        'xgb__reg_alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'xgb__reg_lambda': [1.0, 2.0, 5.0, 10.0],
        'xgb__n_estimators': randint(100, 701),
        'xgb__subsample': uniform(0.6, 0.4),
        'xgb__colsample_bytree': uniform(0.6, 0.4)
    }
    RF_PARAM_DIST = {
        'rf__n_estimators': randint(100, 501),
        'rf__max_depth': [8, 10, 12, 15, 20],
        'rf__min_samples_split': randint(10, 41),
        'rf__min_samples_leaf': randint(5, 21),
        'rf__max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9]
    }
    RIDGE_PARAM_DIST = {
        'ridge__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    }
    param_dist_map = {"xgboost": XGB_PARAM_DIST, "random_forest": RF_PARAM_DIST, "ridge": RIDGE_PARAM_DIST}
    predictor_map = {
        "xgboost": XGBoostScorePredictor if XGBOOST_AVAILABLE else None,
        "random_forest": RandomForestScorePredictor,
        "ridge": RidgeScorePredictor
    }

    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    logger.info(f"Starting tuning & training for models: {models_to_run}")
    all_metrics = []
    for model_key in models_to_run:
        PredictorClass = predictor_map.get(model_key)
        if PredictorClass is None or (model_key == 'xgboost' and not XGBOOST_AVAILABLE):
            logger.warning(f"Model class for '{model_key}' unavailable. Skipping.")
            continue
        param_dist_current = param_dist_map.get(model_key) if not args.skip_tuning else None

        metrics = tune_and_evaluate_predictor(
            predictor_class=PredictorClass,
            X_train=X_train.copy(), y_train_home=y_train_home.copy(), y_train_away=y_train_away.copy(),
            X_val=X_val.copy(), y_val_home=y_val_home.copy(), y_val_away=y_val_away.copy(),
            X_test=X_test.copy(), y_test_home=y_test_home.copy(), y_test_away=y_test_away.copy(),
            model_name_prefix=model_key,
            feature_list=selected_features_unique,
            param_dist=param_dist_current,
            n_iter=args.tune_iterations,
            n_splits=args.cv_splits,
            scoring=args.scoring_metric,
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
        # This block executes only if all_metrics is True (or evaluates to True)
        summary_df = pd.DataFrame(all_metrics)

        # Add model_name as index if it exists and is unique
        if 'model_name' in summary_df.columns and summary_df['model_name'].is_unique:
            # This line executes only if the inner 'if' condition is true
            summary_df = summary_df.set_index('model_name')

        display_cols = [
            'predictor_class', 'best_cv_score', 'test_mae_home', 'test_mae_away',
            'test_rmse_home', 'test_rmse_away', 'test_r2_home', 'test_r2_away',
            'test_combined_loss', 'feature_count', 'training_duration_final', 'tuning_duration'
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns] # Show only available cols

        # --- Choose ONE of the following options for `report_string` ---

        # Option A: Save only the display columns, formatted like the print statement
        # report_df_to_save = summary_df[display_cols]
        # report_string = report_df_to_save.to_string(float_format="%.4f")
        # # Print this version (optional, if you want console output to match file)
        # print(report_string)

        # Option B: Save ALL columns from summary_df, but nicely formatted
        report_string = summary_df.to_string(float_format="%.4f")
        print(summary_df[display_cols].to_string(float_format="%.4f")) # Keep original print if desired

        # --- Save the chosen string to a TXT file ---
        summary_path = REPORTS_DIR / "training_summary_tuned.txt"
        try:
            # This block attempts to open and write to the file
            with open(summary_path, 'w') as f:
                # These lines execute only within the 'with' block
                f.write("--- Overall Training Summary ---\n") # Optional header
                f.write(report_string)
                f.write("\n") # Optional trailing newline
            logger.info(f"Formatted summary saved: {summary_path}")
        except Exception as e:
            # This block executes if an error occurs in the 'try' block
            logger.error(f"Failed to save summary TXT: {e}")
        # Note: The 'else' block associated with 'try...except' is removed
        # as it wasn't present in your original logic (usually used if no exception occurs)

    else:
        # This block executes if all_metrics is False (or evaluates to False)
        logger.warning("No models were successfully trained/evaluated.")

    # These lines are outside the main 'if all_metrics' block and will execute regardless
    if args.run_analysis:
        # This line executes only if args.run_analysis is true
        logger.info("Running analysis (placeholders)...")

    total_pipeline_time = time.time() - start_pipeline_time
    logger.info(f"--- NBA Model Training Pipeline Finished in {total_pipeline_time:.2f} seconds ---")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Score Prediction Model Tuning & Training Pipeline")
    # Data Args
    parser.add_argument("--data-source", type=str, default="supabase", choices=["csv", "supabase", "database", "dummy"], help="Data source type")
    parser.add_argument("--historical-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_game_stats.csv'), help="Path to historical games CSV")
    parser.add_argument("--team-stats-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_team_stats.csv'), help="Path to team stats CSV")
    parser.add_argument("--lookback-days", type=int, default=1095, help="Days of historical data to load")
    # Model Args
    parser.add_argument("--models", type=str, default="xgboost,random_forest,ridge", help="Comma-separated models to train")
    parser.add_argument("--rolling-windows", type=str, default="5,10,20", help="Comma-separated rolling window sizes")
    parser.add_argument("--h2h-window", type=int, default=7, help="Number of games for H2H features")
    # Training Args
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for test set")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction for validation set")
    parser.add_argument("--use-weights", action="store_true", help="Use recency weighting during training")
    parser.add_argument("--weight-method", type=str, default="exponential", choices=["exponential", "half_life"], help="Recency weighting method")
    parser.add_argument("--weight-half-life", type=int, default=90, help="Half-life in days for weights")
    # Tuning Args
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--tune-iterations", type=int, default=30, help="Iterations for RandomizedSearchCV")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS, help="CV splits for TimeSeriesSplit")
    parser.add_argument("--scoring-metric", type=str, default='neg_mean_absolute_error',
                        help="Scoring metric for tuning")
    # Analysis Args
    parser.add_argument("--run-analysis", action="store_true", help="Run optional analysis functions")
    # Output Args
    parser.add_argument("--visualize", action="store_true", help="Show plots interactively")
    parser.add_argument("--save-plots", action="store_true", help="Save generated plots to reports directory")
    # Misc Args
    parser.add_argument("--allow-dummy", action="store_true", help="Allow script to run using dummy classes if imports fail")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    cli_args = parser.parse_args()

    run_training_pipeline(cli_args)
