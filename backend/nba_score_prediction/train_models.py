from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, uniform

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from supabase import Client
from datetime import datetime, timedelta
from pathlib import Path

# --- Scikit-learn Imports ---
from sklearn.linear_model import Ridge as MetaRidge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge as MetaRidge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, HalvingRandomSearchCV
from scipy.stats import randint, uniform, reciprocal  
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
    from . import utils
    if XGBOOST_AVAILABLE:
        from .models import XGBoostScorePredictor
    else:
        # Dummy XGBoost class
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

    try:
        import lightgbm as lgb  # Check library first
        LGBM_AVAILABLE = True
        from .models import LightGBMScorePredictor  # Import your new class
    except ImportError:
        LGBM_AVAILABLE = False
        LightGBMScorePredictor = None  # Crucial: Set to None if import fails
        print("WARNING: LightGBM library not found or LightGBMScorePredictor class import failed.")

    from backend import config
    LOCAL_MODULES_IMPORTED = True
except ImportError as e:
    print(f"ERROR: Could not import local modules: {e}. Using dummy classes.")
    # Define dummy classes/functions
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
        training_duration = None
    class RandomForestScorePredictor(XGBoostScorePredictor): pass
    class RidgeScorePredictor(XGBoostScorePredictor): pass
    def compute_recency_weights(*args, **kwargs): return np.ones(len(args[0])) if args else np.array([])
    def plot_feature_importances(*args, **kwargs): logging.error("Dummy plot_feature_importances called!")
    def plot_actual_vs_predicted(*args, **kwargs): logging.error("Dummy plot_actual_vs_predicted called!")
    def plot_residuals_analysis_detailed(*args, **kwargs): logging.error("Dummy plot_residuals_analysis_detailed called!")
    if 'LightGBMScorePredictor' not in locals():
        class LightGBMScorePredictor(XGBoostScorePredictor):
            def __init__(self, *args, **kwargs):
                 pass
        LGBM_AVAILABLE = False
    class utils:
        @staticmethod
        def remove_duplicate_columns(df): return df
        @staticmethod
        def slice_dataframe_by_features(df, feature_list, fill_value=0.0): return df[feature_list] if df is not None and feature_list else pd.DataFrame()
        @staticmethod
        def fill_missing_numeric(df, default_value=0.0): return df.fillna(default_value) if df is not None else pd.DataFrame()
        @staticmethod
        def to_datetime_naive(series): return pd.to_datetime(series, errors='coerce')
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_BASE_DIR = BACKEND_DIR / 'models'
MAIN_MODELS_DIR = PROJECT_ROOT / 'models' / 'saved' 
REPORTS_DIR = PROJECT_ROOT / 'reports'
MAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COLUMNS = ['home_score', 'away_score']
SEED = 42
DEFAULT_CV_FOLDS = 5
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

def load_data_source(source_type: str, lookback_days: int, args: argparse.Namespace,
                     db_engine: Optional[Any] = None, supabase_client: Optional["Client"] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads historical game data and team stats based on the source type."""
    logger.info(f"Attempting to load data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()

    hist_required_cols = [
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot', 'away_q1', 'away_q2',
        'away_q3', 'away_q4', 'away_ot',
        'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
        'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
        'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
        'home_assists', 'home_steals', 'home_blocks', 'home_turnovers', 'home_fouls',
        'away_assists', 'away_steals', 'away_blocks', 'away_turnovers', 'away_fouls',
        'home_off_reb', 'home_def_reb', 'home_total_reb',
        'away_off_reb', 'away_def_reb', 'away_total_reb'
    ]
    logger.debug(f"DEBUG RUN: Requesting {len(hist_required_cols)} columns: {hist_required_cols}")
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
                logger.debug(f"Supabase select string: {select_str}")

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
                    logger.debug(f"Raw Supabase response processing {len(all_historical_data)} records.")
                    try:
                        first_record_keys = list(all_historical_data[0].keys())
                        num_keys_raw = len(first_record_keys)
                        logger.debug(f"Keys count in first raw record: {num_keys_raw}")
                        logger.debug(f"Keys sample in first raw record: {sorted(first_record_keys)}")
                        missing_keys_in_raw = sorted([col for col in hist_required_cols if col not in first_record_keys])
                        if missing_keys_in_raw:
                            logger.error(f"COLUMNS MISSING IN RAW SUPABASE RESPONSE ({len(missing_keys_in_raw)} missing): {missing_keys_in_raw}")
                        else:
                            logger.info("All requested columns appear present in raw Supabase response.")
                    except IndexError:
                         logger.error("Cannot inspect raw Supabase response keys: all_historical_data is empty list.")
                    except Exception as e_debug_raw:
                         logger.error(f"Error inspecting raw Supabase response: {e_debug_raw}")

                    logger.debug("Attempting to create DataFrame from raw data...")
                    try:
                        hist_df = pd.DataFrame(all_historical_data)
                        logger.info(f"Loaded {len(hist_df)} historical records into DataFrame.")
                        df_cols = sorted(list(hist_df.columns))
                        num_cols_df = len(df_cols)
                        logger.debug(f"Columns count in created DataFrame: {num_cols_df}")
                        logger.debug(f"Columns sample in created DataFrame: {df_cols}")
                        missing_cols_in_df = sorted([col for col in hist_required_cols if col not in df_cols])
                        if missing_cols_in_df:
                            logger.error(f"COLUMNS MISSING AFTER DataFrame CREATION ({len(missing_cols_in_df)} missing): {missing_cols_in_df}")
                        else:
                            logger.info("All requested columns appear present in created DataFrame.")
                        problematic_cols = []
                        for col in hist_required_cols:
                             if col in hist_df.columns and hist_df[col].isnull().all():
                                  problematic_cols.append(col)
                        if problematic_cols:
                             logger.warning(f"Columns present in DataFrame but contain ONLY NaN/None values: {problematic_cols}")
                    except Exception as e_df_create:
                         logger.error(f"Error creating DataFrame from Supabase data: {e_df_create}")
                         hist_df = pd.DataFrame()
                else:
                     logger.warning("No historical data retrieved from Supabase (all_historical_data is empty).")

                if not hist_df.empty:
                    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
                    hist_df = hist_df.dropna(subset=['game_date'])
                    logger.debug(f"DataFrame shape after date processing: {hist_df.shape}")
                else:
                     logger.warning("hist_df is empty after Supabase load/debug checks.")

            except Exception as e:
                logger.error(f"Error loading historical games from Supabase: {e}", exc_info=True)
                hist_df = pd.DataFrame()

    elif source_type == "csv":
        pass

    if source_type != "csv":
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

    if not hist_df.empty:
        missing_hist_cols = [col for col in hist_required_cols if col not in hist_df.columns]
        if missing_hist_cols:
            logger.error(f"CRITICAL FINAL CHECK: Columns missing before numeric conversion: {missing_hist_cols}. Check DataFrame creation logs.")
        logger.info("Converting historical numeric columns and handling NaNs...")
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                hist_df[col] = hist_df[col].fillna(np.nan)
            else:
                 logger.warning(f"Numeric column '{col}' still missing before fillna; adding as NaN.")
                 hist_df[col] = np.nan

        if args.debug:
            pass

        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)

    if not hist_df.empty:
         final_df_cols = sorted(list(hist_df.columns))
         logger.debug(f"Final columns in hist_df before return ({len(final_df_cols)}): {final_df_cols}")
         missing_final = sorted([col for col in hist_required_cols if col not in final_df_cols])
         if missing_final:
              logger.error(f"Columns confirmed missing in final hist_df before return: {missing_final}")

    logger.info(f"Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df

def visualize_recency_weights(dates, weights, title="Recency Weights Distribution", save_path=None):
    if dates is None or weights is None or len(dates) != len(weights):
        logger.warning("Invalid input for visualizing recency weights.")
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df_plot = pd.DataFrame({'date': pd.to_datetime(dates), 'weight': weights}).sort_values('date')
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['date'], df_plot['weight'], marker='.', linestyle='-')
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
        plt.close()
    except ImportError:
         logger.error("Matplotlib or Pandas not found. Cannot generate recency weights plot.")
    except Exception as e:
         logger.error(f"Error visualizing recency weights: {e}", exc_info=True)

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

def compute_weights(validation_metrics: Dict[str, float]) -> Dict[str, float]:
    inv_errors = {}
    logger.info(f"Computing weights based on validation metrics: {validation_metrics}")
    for model, error in validation_metrics.items():
        if pd.isna(error) or error <= 0:
             logger.warning(f"Invalid error ({error}) for model '{model}'. Excluding from weighting.")
             inv_errors[model] = 0.0
        else:
            inv_errors[model] = 1.0 / (error + 1e-8)
    total = sum(inv_errors.values())
    weights = {}
    if total <= 1e-8:
        logger.warning("Total inverse error is near zero. Assigning equal weights to models with valid errors.")
        num_valid_models = sum(1 for err in validation_metrics.values() if pd.notna(err) and err > 0)
        equal_weight = 1.0 / num_valid_models if num_valid_models > 0 else 0.0
        weights = {model: equal_weight if pd.notna(error) and error > 0 else 0.0
                   for model, error in validation_metrics.items()}
    else:
        weights = {model: inv_error / total for model, inv_error in inv_errors.items()}
    logger.info(f"Calculated normalized weights: {weights}")
    return weights

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

from .evaluation import (
    plot_residuals_analysis_detailed,
    plot_conditional_bias,
    plot_temporal_bias,
    plot_feature_importances,
    plot_actual_vs_predicted
)
from .models import compute_recency_weights
from .feature_engineering import NBAFeatureEngine

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MAIN_MODELS_DIR = Path("./models/saved")
REPORTS_DIR = Path("./reports")
SEED = 42
TARGET_COLUMNS = ['home_score', 'away_score']

def tune_model(estimator, param_dist, X, y, cv, n_iter, scoring, random_state, fit_params=None):
    """Helper function to run RandomizedSearchCV."""
    # Ensure necessary import is available
    from sklearn.model_selection import RandomizedSearchCV
    if fit_params is None:
         fit_params = {}

    # Use RandomizedSearchCV again
    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,              # Use n_iter again
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,                  # Set verbosity (1 shows progress)
        random_state=random_state,
        error_score='raise',
        refit=False                 # We refit the best model later
    )

    # Pass fit_params (which might include sample_weight and early stopping args)
    rs.fit(X, y, **fit_params)

    # Extract results
    cv_results = pd.DataFrame(rs.cv_results_).sort_values(by='rank_test_score')
    # Strip potential prefixes from parameter names
    best_params = {k.split('__', 1)[1]: v for k, v in rs.best_params_.items()}
    best_score = rs.best_score_
    return best_params, best_score, cv_results

def tune_and_evaluate_predictor(
    predictor_class,
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str,
    feature_list: List[str],
    param_dist: Optional[Dict[str, Any]],
    n_iter: int = 10,
    n_splits: int = 5,
    scoring: str = 'neg_mean_absolute_error',
    use_recency_weights: bool = False,
    weight_method: str = 'exponential',
    weight_half_life: int = 90,
    visualize: bool = True,
    save_plots: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    logger.debug(f"tune_and_evaluate_predictor received feature_list with {len(feature_list)} features.")
    model_full_name = f"{model_name_prefix}_score_predictor"
    if predictor_class is None:
        logger.error(f"Predictor class for {model_full_name} is None.")
        return None, None
    if model_name_prefix.lower() == 'xgboost' and not XGBOOST_AVAILABLE:
        logger.error("XGBoost unavailable.")
        return None, None
    if not feature_list:
        logger.error(f"Input feature list is empty for {model_full_name}.")
        return None, None

    logger.info(f"--- Tuning and Training {predictor_class.__name__} ({model_full_name}) ---")
    start_tune_time = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {}

    feature_list_unique = list(pd.Index(feature_list).unique())
    if len(feature_list_unique) != len(feature_list):
        logger.warning(f"Duplicate features found. Using {len(feature_list_unique)} unique features.")

    train_val_date_cols = ['game_date'] if use_recency_weights and 'game_date' in X_train.columns and 'game_date' in X_val.columns else []
    cols_train = [col for col in feature_list_unique + train_val_date_cols if col in X_train.columns]
    cols_val = [col for col in feature_list_unique + train_val_date_cols if col in X_val.columns]
    try:
        X_train_full = pd.concat([X_train[cols_train], X_val[cols_val]], ignore_index=True)
        y_train_home_full = pd.concat([y_train_home, y_val_home], ignore_index=True)
        y_train_away_full = pd.concat([y_train_away, y_val_away], ignore_index=True)
        logger.info(f"Combined Train+Val size for final fit: {len(X_train_full)} samples.")
    except Exception as concat_err:
        logger.error(f"Error combining Train and Val sets: {concat_err}", exc_info=True)
        return None, None

    try:
        X_tune_features = X_train[feature_list_unique].copy()
        y_tune_home = y_train_home.loc[X_tune_features.index].copy()
    except KeyError as ke:
        logger.error(f"KeyError preparing tuning data targets: {ke}", exc_info=True)
        return None, None
    except Exception as prep_err:
        logger.error(f"Error preparing tuning data targets: {prep_err}", exc_info=True)
        return None, None

    tune_fit_params = {}
    if use_recency_weights:
        if 'game_date' in X_train.columns:
            dates_for_weights_tune = X_train.loc[X_tune_features.index, 'game_date']
            sample_weights_tune = compute_recency_weights(dates_for_weights_tune, method=weight_method, half_life=weight_half_life)
            if sample_weights_tune is not None and len(sample_weights_tune) == len(X_tune_features):
                try:
                    temp_predictor = predictor_class()
                    temp_pipeline = temp_predictor._build_pipeline({})
                    model_step_name = temp_pipeline.steps[-1][0]
                    weight_param_name = f"{model_step_name}__sample_weight"
                    tune_fit_params[weight_param_name] = sample_weights_tune
                    logger.info(f"Sample weights for tuning prepared using key: '{weight_param_name}'")
                except Exception as e:
                    logger.warning(f"Error determining weight parameter name for tuning: {e}. Weights may not be applied.")
            else:
                logger.warning("Failed tuning sample weights computation/alignment.")
        else:
            logger.warning("'use_recency_weights' enabled but 'game_date' missing in X_train.")

    # Initialize variables that might be set during tuning
    best_params_final = {}
    metrics['best_cv_score'] = None # Initialize in metrics dict
    metrics['best_params'] = 'default (tuning skipped or failed)' # Default
    metrics['tuning_duration'] = 0.0 # Default

    # Check if tuning is requested for the current model being processed
    if param_dist:
        # <<< CHANGE Log message back >>>
        logger.info(f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={n_splits}, scoring='{scoring}')...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        tuning_start_time = time.time()
        try:
            temp_predictor = predictor_class()
            search_pipeline = temp_predictor._build_pipeline({})
            if search_pipeline is None:
                raise RuntimeError(f"Could not build pipeline for {predictor_class.__name__}")

            # --- Prepare fit_params (including potential early stopping) ---
            current_fit_params = tune_fit_params.copy() # Start with sample weights if they exist

            # <<< ADD EARLY STOPPING FOR LIGHTGBM >>>
            if model_name_prefix.lower() == 'lightgbm' and LGBM_AVAILABLE and lgb is not None:
                try:
                    # Prepare eval set using the main validation data passed to tune_and_evaluate_predictor
                    # Ensure X_val has the right features selected by LASSO
                    X_val_eval = X_val[feature_list_unique].copy()
                    # Align y_val_home with the index of X_val_eval
                    y_val_home_eval = y_val_home.loc[X_val_eval.index]

                    # Add LGBM specific fit parameters for early stopping
                    # NOTE: We use the main validation set here. This isn't perfect time-series validation
                    # within each CV fold, but it's the most practical way to enable early stopping
                    # during the RandomizedSearchCV process without major restructuring.
                    current_fit_params[f'lgbm__eval_set'] = [(X_val_eval, y_val_home_eval)]
                    current_fit_params[f'lgbm__eval_metric'] = 'mae' # Make sure metric matches objective/scoring
                    current_fit_params[f'lgbm__callbacks'] = [
                        lgb.early_stopping(stopping_rounds=15, # Stop if MAE doesn't improve for 15 rounds
                                           verbose=False)      # Suppress early stopping messages
                        ]
                    logger.info("Added Early Stopping parameters for LightGBM tuning.")
                except KeyError as ke:
                    logger.error(f"KeyError preparing eval_set for LightGBM early stopping: {ke}. Skipping early stopping.")
                except Exception as es_err:
                     logger.error(f"Error setting up early stopping for LightGBM: {es_err}. Skipping early stopping.")
            # <<< END EARLY STOPPING FOR LIGHTGBM >>>


            # Call the tune_model helper (now using RandomizedSearchCV)
            # Pass the potentially modified current_fit_params
            best_params_final, best_score, cv_results_df = tune_model(
                estimator=search_pipeline,
                param_dist=param_dist,
                X=X_tune_features,
                y=y_tune_home,
                cv=tscv,
                scoring=scoring,
                n_iter=n_iter,            # Pass n_iter for RandomizedSearchCV
                random_state=SEED,
                fit_params=current_fit_params # Pass weights and potentially early stopping args
            )

            # Log results consistently after successful tuning
            try:
                logger.info("--- Top CV Results ---")
                logger.info("\n" + cv_results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']].head().to_string())
            except Exception as report_e:
                logger.warning(f"Could not print CV results: {report_e}")

            metrics['best_cv_score'] = best_score
            metrics['best_params'] = best_params_final
            logger.info(f"Tuning complete. Best CV Score ({scoring}): {best_score:.4f}")
            logger.info(f"Best Final Params: {best_params_final}")


        except Exception as search_e:
            # <<< CHANGE Log message back >>>
            logger.error(f"RandomizedSearchCV failed for {model_name_prefix}: {search_e}", exc_info=True)
            best_params_final = {}
            metrics['best_cv_score'] = None
            metrics['best_params'] = 'default (tuning failed)'

        metrics['tuning_duration'] = time.time() - tuning_start_time
        logger.info(f"Tuning finished in {metrics['tuning_duration']:.2f}s.")

    else: # Handles --skip-tuning case
        logger.info("Skipping hyperparameter tuning.")

    # --- Train Final Model ---
    logger.info("Training final model on combined Train+Val data...")

    # --- Train Final Model ---
    # This section now correctly uses 'best_params_final' populated (or left empty) above
    logger.info("Training final model on combined Train+Val data...")
    final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
    try:
        X_train_full_for_fit = X_train_full[feature_list_unique].copy()
        y_train_home_full_aligned = y_train_home_full
        y_train_away_full_aligned = y_train_away_full
    except KeyError as ke:
        logger.error(f"KeyError preparing final train data: {ke}", exc_info=True)
        return None, None
    except Exception as final_prep_err:
        logger.error(f"Error preparing final train data: {final_prep_err}", exc_info=True)
        return None, None

    final_sample_weights = None
    if use_recency_weights:
        if 'game_date' in X_train_full.columns:
            dates_for_weights_full = X_train_full.loc[X_train_full_for_fit.index, 'game_date']
            final_sample_weights = compute_recency_weights(dates_for_weights_full, method=weight_method, half_life=weight_half_life)
            if final_sample_weights is None or len(final_sample_weights) != len(X_train_full_for_fit):
                logger.warning("Final sample weights failed/mismatched. Training without weights.")
                final_sample_weights = None
            else:
                logger.info(f"Final sample weights computed. Min: {np.min(final_sample_weights):.4f}, Max: {np.max(final_sample_weights):.4f}")
                if visualize:
                    try:
                        save_path_w = REPORTS_DIR / f"{model_full_name}_final_weights.png" if save_plots else None
                        visualize_recency_weights(dates_for_weights_full, final_sample_weights, title=f"Final Training Weights - {model_full_name}", save_path=save_path_w)
                    except NameError:
                        logger.error("visualize_recency_weights function not found.")
                    except Exception as plot_w_e:
                        logger.error(f"Error visualizing final weights: {plot_w_e}")
        else:
            logger.warning("'use_recency_weights' enabled but 'game_date' missing in final data.")

    try:
        train_start_time = time.time()
        final_predictor.train(
            X_train=X_train_full_for_fit,
            y_train_home=y_train_home_full_aligned,
            y_train_away=y_train_away_full_aligned,
            hyperparams_home=best_params_final,
            hyperparams_away=best_params_final,
            sample_weights=final_sample_weights
        )
        metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', time.time() - train_start_time)
        logger.info(f"Final model training completed in {metrics['training_duration_final']:.2f} seconds.")
        metrics['feature_count'] = len(getattr(final_predictor, 'feature_names_in_', feature_list_unique))
        if not getattr(final_predictor, 'feature_names_in_', None):
            logger.warning("feature_names_in_ not found on model.")
    except Exception as train_e:
        logger.error(f"Training final model failed: {train_e}", exc_info=True)
        metrics['training_duration_final'] = None
        metrics['feature_count'] = len(feature_list_unique)
        return metrics, None

    try:
        timestamp = getattr(final_predictor, 'training_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_filename = f"{model_full_name}.joblib"
        save_path = final_predictor.save_model(filename=save_filename)
        if save_path:
            logger.info(f"Final tuned model saved to {save_path}")
            metrics['save_path'] = str(save_path)
        else:
            logger.error("Model saving returned no path.")
            metrics['save_path'] = "Save failed"
    except Exception as save_e:
        logger.error(f"Error saving final model: {save_e}", exc_info=True)
        metrics['save_path'] = f"Save failed ({type(save_e).__name__})"

    logger.info(f"Evaluating final {model_full_name} on test set ({len(X_test)})...")
    try:
        X_test_final = X_test[feature_list_unique].copy()
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or 'predicted_home_score' not in predictions_df_test.columns:
            raise ValueError("Test prediction failed/invalid format.")
        pred_home_test = predictions_df_test['predicted_home_score']
        pred_away_test = predictions_df_test['predicted_away_score']
        y_test_home_aligned = y_test_home.loc[pred_home_test.index]
        y_test_away_aligned = y_test_away.loc[pred_away_test.index]

        test_metrics_home = calculate_regression_metrics(y_test_home_aligned, pred_home_test)
        test_metrics_away = calculate_regression_metrics(y_test_away_aligned, pred_away_test)
        metrics['test_mae_home'] = test_metrics_home.get('mae', np.nan)
        metrics['test_rmse_home'] = test_metrics_home.get('rmse', np.nan)
        metrics['test_r2_home'] = test_metrics_home.get('r2', np.nan)
        metrics['test_mae_away'] = test_metrics_away.get('mae', np.nan)
        metrics['test_rmse_away'] = test_metrics_away.get('rmse', np.nan)
        metrics['test_r2_away'] = test_metrics_away.get('r2', np.nan)
        metrics['test_mae_total'] = mean_absolute_error(y_test_home_aligned + y_test_away_aligned, pred_home_test + pred_away_test)
        metrics['test_mae_diff'] = mean_absolute_error(y_test_home_aligned - y_test_away_aligned, pred_home_test - pred_away_test)

        try:
            y_true_comb_test = np.vstack((y_test_home_aligned.values, y_test_away_aligned.values)).T
            y_pred_comb_test = np.vstack((pred_home_test.values, pred_away_test.values)).T
            metrics['test_nba_score_loss'] = nba_score_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_nba_dist_loss'] = nba_distribution_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_combined_loss'] = combined_nba_loss(y_true_comb_test, y_pred_comb_test)
            metrics['betting_metrics'] = calculate_betting_metrics(y_true_comb_test, y_pred_comb_test, vegas_lines=None)
        except NameError as ne:
            logger.warning(f"Custom loss/betting function not found: {ne}")
        except Exception as custom_loss_e:
            logger.error(f"Error calculating custom losses: {custom_loss_e}")

        logger.info(f"FINAL Test MAE : Home={metrics.get('test_mae_home',np.nan):.3f}, Away={metrics.get('test_mae_away',np.nan):.3f}, Total={metrics.get('test_mae_total',np.nan):.3f}, Diff={metrics.get('test_mae_diff',np.nan):.3f}")
        logger.info(f"FINAL Test RMSE: Home={metrics.get('test_rmse_home',np.nan):.3f}, Away={metrics.get('test_rmse_away',np.nan):.3f}")
        logger.info(f"FINAL Test R2  : Home={metrics.get('test_r2_home',np.nan):.3f}, Away={metrics.get('test_r2_away',np.nan):.3f}")
        logger.info(f"FINAL Custom Losses: Score={metrics.get('test_nba_score_loss',np.nan):.3f}, Dist={metrics.get('test_nba_dist_loss',np.nan):.3f}, Combined={metrics.get('test_combined_loss',np.nan):.3f}")
        logger.info(f"FINAL Betting Metrics: {metrics.get('betting_metrics', {})}")
    except KeyError as ke:
        logger.error(f"KeyError evaluating test set: {ke}", exc_info=True)
        [metrics.setdefault(k, np.nan) for k in ['test_mae_home','test_mae_away', 'test_mae_total','test_mae_diff']]
    except Exception as test_eval_e:
        logger.error(f"Failed evaluating test set: {test_eval_e}", exc_info=True)
        [metrics.setdefault(k, np.nan) for k in ['test_mae_home','test_mae_away', 'test_mae_total','test_mae_diff']]

    try:
        predictions_df_train = final_predictor.predict(X_train_full_for_fit)
        if predictions_df_train is None or 'predicted_home_score' not in predictions_df_train.columns:
            raise ValueError("Train prediction failed.")
        pred_home_train = predictions_df_train['predicted_home_score']
        pred_away_train = predictions_df_train['predicted_away_score']
        train_metrics_home = calculate_regression_metrics(y_train_home_full_aligned, pred_home_train)
        train_metrics_away = calculate_regression_metrics(y_train_away_full_aligned, pred_away_train)
        metrics['train_mae_home'] = train_metrics_home.get('mae', np.nan)
        metrics['train_rmse_home'] = train_metrics_home.get('rmse', np.nan)
        metrics['train_r2_home'] = train_metrics_home.get('r2', np.nan)
        metrics['train_mae_away'] = train_metrics_away.get('mae', np.nan)
        metrics['train_rmse_away'] = train_metrics_away.get('rmse', np.nan)
        metrics['train_r2_away'] = train_metrics_away.get('r2', np.nan)
        metrics['train_mae_total'] = mean_absolute_error(y_train_home_full_aligned + y_train_away_full_aligned, pred_home_train + pred_away_train)
        metrics['train_mae_diff'] = mean_absolute_error(y_train_home_full_aligned - y_train_away_full_aligned, pred_home_train - pred_away_train)
    except Exception as train_eval_e:
        logger.error(f"Failed evaluating on training set: {train_eval_e}", exc_info=True)
        [metrics.setdefault(k, np.nan) for k in ['train_mae_home','train_mae_away', 'train_mae_total','train_mae_diff']]

    metrics['samples_train_final'] = len(X_train_full_for_fit)
    metrics['samples_test'] = len(X_test_final) if 'X_test_final' in locals() else len(X_test)

    if visualize or save_plots:
        timestamp_str = metrics.get('save_path')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp_str and isinstance(timestamp_str, str):
            try:
                ts_match = re.search(r'(\d{8}_\d{6})\.joblib$', Path(timestamp_str).name)
                if ts_match:
                    timestamp = ts_match.group(1)
                else:
                     timestamp = getattr(final_predictor, 'training_timestamp', timestamp)
            except Exception:
                 timestamp = getattr(final_predictor, 'training_timestamp', timestamp)
        plot_dir = REPORTS_DIR / f"{model_full_name}_tuned_performance_{timestamp}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating performance plots in {plot_dir}")
        if 'pred_home_test' in locals() and 'y_test_home_aligned' in locals():
            try:
                plot_actual_vs_predicted(y_true=y_test_home_aligned, y_pred=pred_home_test, title=f"Tuned {model_full_name} - Test Actual vs Pred (Home)",
                    metrics_dict={'rmse': metrics.get('test_rmse_home'), 'r2': metrics.get('test_r2_home'), 'mae': metrics.get('test_mae_home')},
                    save_path=plot_dir / "test_actual_vs_pred_home.png" if save_plots else None, show_plot=visualize)
                plot_actual_vs_predicted(y_true=y_test_away_aligned, y_pred=pred_away_test, title=f"Tuned {model_full_name} - Test Actual vs Pred (Away)",
                    metrics_dict={'rmse': metrics.get('test_rmse_away'), 'r2': metrics.get('test_r2_away'), 'mae': metrics.get('test_mae_away')},
                    save_path=plot_dir / "test_actual_vs_pred_away.png" if save_plots else None, show_plot=visualize)
                plot_residuals_analysis_detailed(y_true=y_test_home_aligned, y_pred=pred_home_test, title_prefix=f"Tuned {model_full_name} (Home) - Test Set", save_dir=plot_dir if save_plots else None, show_plot=visualize)
                plot_residuals_analysis_detailed(y_true=y_test_away_aligned, y_pred=pred_away_test, title_prefix=f"Tuned {model_full_name} (Away) - Test Set", save_dir=plot_dir if save_plots else None, show_plot=visualize)
                pipeline_home = getattr(final_predictor, 'pipeline_home', None)
                pipeline_away = getattr(final_predictor, 'pipeline_away', None)
                features_in = getattr(final_predictor, 'feature_names_in_', feature_list_unique)
                if pipeline_home and pipeline_away and features_in:
                    plot_feature_importances(models_dict={f"{model_full_name}_Home": pipeline_home, f"{model_full_name}_Away": pipeline_away},
                        feature_names=features_in, top_n=30, plot_groups=True, save_dir=plot_dir / "feature_importance" if save_plots else None, show_plot=visualize)
                else:
                    logger.warning("Cannot generate feature importance plots (missing pipeline/features).")
            except NameError as ne:
                logger.error(f"Plotting function not found: {ne}.")
            except Exception as plot_e:
                logger.error(f"Failed generating plots: {plot_e}", exc_info=True)
        else:
            logger.warning("Skipping plots (test predictions unavailable).")

    val_predictions_dict = None
    try:
        X_val_final = X_val[feature_list_unique].copy()
        predictions_df_val = final_predictor.predict(X_val_final)
        if predictions_df_val is None or 'predicted_home_score' not in predictions_df_val.columns:
            raise ValueError(f"Prediction on validation set failed or returned invalid format for {model_full_name}.")
        y_val_home_aligned = y_val_home.loc[X_val_final.index]
        y_val_away_aligned = y_val_away.loc[X_val_final.index]
        pred_home_val = predictions_df_val['predicted_home_score'].loc[X_val_final.index]
        pred_away_val = predictions_df_val['predicted_away_score'].loc[X_val_final.index]
        val_predictions_dict = {
            'pred_home': pred_home_val, 'pred_away': pred_away_val,
            'true_home': y_val_home_aligned, 'true_away': y_val_away_aligned
        }
    except KeyError as ke_val:
        logger.error(f"FAILED generating validation predictions: KeyError. Features missing in X_val? {ke_val}", exc_info=True)
        missing_val = set(feature_list_unique) - set(X_val.columns)
        if missing_val:
            logger.error(f"Features missing in X_val: {missing_val}")
        val_predictions_dict = None
    except Exception as val_pred_e:
        logger.error(f"FAILED generating validation set predictions for {model_full_name}: {val_pred_e}", exc_info=True)
        val_predictions_dict = None

    metrics['total_duration'] = time.time() - start_tune_time
    logger.info(f"--- Finished Tuning & Training {model_full_name} in {metrics['total_duration']:.2f}s ---")
    return metrics, val_predictions_dict

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

    historical_df, team_stats_df = load_data_source(
        args.data_source, args.lookback_days, args, db_engine, supabase_client
    )

    if historical_df.empty:
        logger.error("Failed to load historical data. Exiting.")
        sys.exit(1)

    if team_stats_df.empty:
        logger.warning("Team stats data is empty. Season context features will use defaults.")

    logger.info("Initializing Feature Engine...")
    feature_engine = NBAFeatureEngine(debug=args.debug)

    logger.info("Generating features for ALL historical data...")
    rolling_windows_list = (
        [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [5, 10]
    )
    logger.info(f"Using rolling windows: {rolling_windows_list}")

    logger.info(f"Processing {len(historical_df)} games for feature generation...")
    logger.info("Generating features...")
    features_df = feature_engine.generate_all_features(
        df=historical_df.copy(),
        historical_games_df=historical_df.copy(),
        team_stats_df=team_stats_df.copy() if not team_stats_df.empty else None,
        rolling_windows=rolling_windows_list,
        h2h_window=args.h2h_window,
    )

    if features_df.empty:
        logger.error("Feature generation failed. Exiting.")
        sys.exit(1)

    if features_df.columns.duplicated().any():
        logger.warning("Duplicate column names found in features_df! Removing duplicates, keeping first occurrence.")
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep='first')]
        logger.info(f"Shape after duplicate removal: {features_df.shape}")
    else:
        logger.info("No duplicate column names found in features_df.")

    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    logger.info(f"Dropped {initial_rows - len(features_df)} rows due to missing target values.")

    if features_df.empty:
        logger.error("No rows remaining after dropping missing targets. Exiting.")
        sys.exit(1)

    logger.info("Analyzing generated feature values...")

    if not features_df.empty:
        try:
            numeric_features_df = features_df.select_dtypes(include=np.number)
            if not numeric_features_df.empty:
                desc_stats = numeric_features_df.describe().transpose()
                zero_pct = (numeric_features_df == 0).mean() * 100
                zero_pct.rename('zero_percentage', inplace=True)
                feature_summary = pd.concat([desc_stats, zero_pct], axis=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = f"feature_value_summary_{timestamp}.txt"
                summary_path = REPORTS_DIR / summary_filename
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(summary_path, 'w') as f:
                    f.write(f"Feature Value Summary ({timestamp})\n")
                    f.write(f"Total Numeric Features Analyzed: {len(feature_summary)}\n")
                    f.write("=" * 50 + "\n\n")
                    problem_threshold_std = 1e-6
                    problem_threshold_zero_pct = 98.0
                    problematic_features = feature_summary[
                        (feature_summary['std'].fillna(0) < problem_threshold_std) |
                        (feature_summary['zero_percentage'] > problem_threshold_zero_pct)
                    ]
                    if not problematic_features.empty:
                        f.write("--- POTENTIALLY PROBLEMATIC FEATURES ---\n")
                        f.write(
                            f"(Features with std < {problem_threshold_std} OR zero % > {problem_threshold_zero_pct}%)\n\n"
                        )
                        f.write(problematic_features.to_string(float_format="%.4f"))
                        f.write("\n\n" + "=" * 50 + "\n\n")
                    else:
                        f.write("--- No obvious problematic features found based on thresholds. ---\n\n")
                    f.write("--- FULL FEATURE SUMMARY ---\n\n")
                    with pd.option_context('display.max_rows', None):
                        f.write(feature_summary.to_string(float_format="%.4f"))
                logger.info(f"Feature value summary saved to: {summary_path}")
                if not problematic_features.empty:
                    logger.warning(
                        f"Found {len(problematic_features)} potentially problematic features (low variance or high zero %). See {summary_filename}"
                    )
            else:
                logger.warning("No numeric features found in features_df to analyze for summary.")
        except Exception as e_summary:
            logger.error(f"Error generating feature value summary: {e_summary}", exc_info=True)
    else:
        logger.warning("features_df is empty, skipping feature value analysis.")

    logger.info("--- Starting LASSO Feature Selection ---")
    potential_feature_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = TARGET_COLUMNS + ['game_id', 'game_date']
    logger.info(f"Initial potential numeric feature count (excluding targets/IDs): {len(potential_feature_cols) - len(exclude_cols)}")

    safe_prefixes = (
        'home_rolling_', 'away_rolling_', 'rolling_',
        'home_season_', 'away_season_', 'season_',
        'matchup_', 'rest_days_', 'is_back_to_back_',
        'games_last_', 'home_form_', 'away_form_'
    )
    safe_exact_names = {
        'rest_advantage', 'schedule_advantage', 'form_win_pct_diff',
        'streak_advantage', 'momentum_diff', 'home_current_streak',
        'home_momentum_direction', 'away_current_streak', 'away_momentum_direction',
        'game_importance_rank', 'home_win_last10', 'away_win_last10',
        'home_trend_rating', 'away_trend_rating', 'home_rank', 'away_rank'
    }

    feature_candidates = []
    leaked_or_excluded = []
    temp_candidates = [col for col in potential_feature_cols if col not in exclude_cols and not features_df[col].isnull().all()]

    for col in temp_candidates:
        is_safe = False
        if col.startswith(safe_prefixes):
            is_safe = True
        elif col in safe_exact_names:
            is_safe = True
        if is_safe:
            feature_candidates.append(col)
        else:
            leaked_or_excluded.append(col)

    logger.info(f"Filtered down to {len(feature_candidates)} PRE-GAME feature candidates for LASSO.")
    if leaked_or_excluded:
        logger.debug(f"Excluded {len(leaked_or_excluded)} potential leaky/post-game features: {leaked_or_excluded}")

    X_lasso_raw = features_df[feature_candidates].copy()
    y_home_lasso = features_df[TARGET_COLUMNS[0]].copy()
    y_away_lasso = features_df[TARGET_COLUMNS[1]].copy()

    logger.info("Checking for NaNs before LASSO scaling...")
    nan_counts = X_lasso_raw.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        logger.error(f"UNEXPECTED NaNs found before LASSO in columns: {cols_with_nan.index.tolist()}. Fix upstream feature engineering or implement robust NaN handling here.")
    else:
        logger.info("Confirmed: No NaNs found in feature candidates before LASSO.")

    logger.info("Checking for and removing zero-variance features...")
    variances = X_lasso_raw.var()
    zero_var_cols = variances[variances < 1e-8].index.tolist()
    if zero_var_cols:
        logger.warning(f"Removing {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        feature_candidates_final = variances[variances >= 1e-8].index.tolist()
        X_lasso_final = X_lasso_raw[feature_candidates_final].copy()
        logger.info(f"Proceeding with {X_lasso_final.shape[1]} non-zero-variance features.")
    else:
        logger.info("No zero-variance features found.")
        X_lasso_final = X_lasso_raw

    if X_lasso_final.empty:
        logger.error("LASSO: No features remaining after removing zero-variance columns.")
        sys.exit(1)

    logger.info("Scaling features for LASSO...")
    scaler = StandardScaler()
    X_lasso_scaled = scaler.fit_transform(X_lasso_final)

    logger.info("Running LassoCV for Home Score...")
    lasso_cv_home = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=2000)
    lasso_cv_home.fit(X_lasso_scaled, y_home_lasso)
    logger.info(f"LassoCV (Home) completed. Optimal alpha: {lasso_cv_home.alpha_:.6f}")

    logger.info("Running LassoCV for Away Score...")
    lasso_cv_away = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=2000)
    lasso_cv_away.fit(X_lasso_scaled, y_away_lasso)
    logger.info(f"LassoCV (Away) completed. Optimal alpha: {lasso_cv_away.alpha_:.6f}")

    selector_home = SelectFromModel(lasso_cv_home, prefit=True, threshold=1e-5)
    selector_away = SelectFromModel(lasso_cv_away, prefit=True, threshold=1e-5)
    selected_mask_home = selector_home.get_support()
    selected_mask_away = selector_away.get_support()
    combined_mask = selected_mask_home | selected_mask_away
    selected_features_lasso = X_lasso_final.columns[combined_mask].tolist()
    num_selected = len(selected_features_lasso)

    logger.info(f"LASSO selected {num_selected} features (union of home/away selections).")
    if num_selected == 0:
        logger.error("LASSO selected 0 features. Check data, scaling, or LassoCV parameters. Exiting.")
        sys.exit(1)
    elif num_selected < 10:
        logger.warning(f"LASSO selected a very small number of features ({num_selected}). Model performance might be affected.")

    logger.debug(f"Selected features: {selected_features_lasso}")

    essential_non_feature_cols = ['game_id', 'game_date'] + TARGET_COLUMNS
    final_cols_for_split = essential_non_feature_cols + selected_features_lasso
    missing_final_cols = [col for col in final_cols_for_split if col not in features_df.columns]
    if missing_final_cols:
        logger.error(f"Columns required for final split DataFrame are missing from features_df: {missing_final_cols}. Exiting.")
        sys.exit(1)

    features_df_selected = features_df[final_cols_for_split].copy()
    logger.info(f"Created features_df_selected with shape: {features_df_selected.shape}")

    logger.info("Splitting data (time-based) using LASSO-selected features...")
    features_df_selected = features_df_selected.sort_values('game_date').reset_index(drop=True)
    n = len(features_df_selected)
    test_split_idx = int(n * (1 - args.test_size))
    val_split_frac = min(args.val_size, 1.0 - args.test_size - 0.01)
    val_split_idx = int(n * (1 - args.test_size - val_split_frac))
    train_df = features_df_selected.iloc[:val_split_idx].copy()
    val_df = features_df_selected.iloc[val_split_idx:test_split_idx].copy()
    test_df = features_df_selected.iloc[test_split_idx:].copy()

    X_train = train_df[selected_features_lasso + (['game_date'] if args.use_weights else [])].copy()
    X_val = val_df[selected_features_lasso + (['game_date'] if args.use_weights else [])].copy()
    X_test = test_df[selected_features_lasso].copy()
    y_train_home = train_df[TARGET_COLUMNS[0]].copy()
    y_train_away = train_df[TARGET_COLUMNS[1]].copy()
    y_val_home = val_df[TARGET_COLUMNS[0]].copy()
    y_val_away = val_df[TARGET_COLUMNS[1]].copy()
    y_test_home = test_df[TARGET_COLUMNS[0]].copy()
    y_test_away = test_df[TARGET_COLUMNS[1]].copy()

    logger.info(f"Data Split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    logger.debug(f"X_train shape: {X_train.shape}")
    logger.debug(f"X_val shape: {X_val.shape}")
    logger.debug(f"X_test shape: {X_test.shape}")

    final_feature_list_for_models = selected_features_lasso
    logger.info(f"Using {len(final_feature_list_for_models)} features selected by LASSO for model training.")

    XGB_PARAM_DIST = {
        'xgb__n_estimators': randint(100, 1001),
        'xgb__max_depth': randint(3, 11),
        'xgb__learning_rate': uniform(0.01, 0.14),
        'xgb__subsample': uniform(0.5, 0.4),
        'xgb__colsample_bytree': uniform(0.5, 0.4),
        'xgb__gamma': uniform(0.0, 0.5),
        'xgb__min_child_weight': randint(1, 11)
    }

    RF_PARAM_DIST = {
        'rf__n_estimators': randint(100, 801),
        'rf__max_depth': randint(3, 11),
        'rf__min_samples_split': randint(20, 61),
        'rf__min_samples_leaf': randint(10, 31),
        'rf__max_features': uniform(0.2, 0.3),
        'rf__ccp_alpha': uniform(0.0, 0.02),
        'rf__bootstrap': [True, False]
    }

    RIDGE_PARAM_DIST = {
        'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'ridge__fit_intercept': [True, False],
        'ridge__solver': ['auto', 'svd', 'cholesky']
    }

    # Note: LGBM_PARAM_DIST is defined above if model_name_prefix is "lightgbm"
    param_dist_map = {
            "xgboost": XGB_PARAM_DIST,
            "random_forest": RF_PARAM_DIST,
            "ridge": RIDGE_PARAM_DIST,
            "lightgbm": {}  # Already handled above
        }

    predictor_map = {
        "xgboost": XGBoostScorePredictor if XGBOOST_AVAILABLE else None,
        "random_forest": RandomForestScorePredictor,
        "ridge": RidgeScorePredictor,
        "lightgbm": LightGBMScorePredictor if LGBM_AVAILABLE else None
    }

    # --- Base Model Training Loop ---
    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    logger.info(f"Starting tuning & training for base models: {models_to_run}")
    all_metrics = []
    validation_predictions_collector = {}

    for model_key in models_to_run:
        PredictorClass = predictor_map.get(model_key)
        if PredictorClass is None or (model_key == 'xgboost' and not XGBOOST_AVAILABLE):
            logger.warning(f"Model class for '{model_key}' unavailable. Skipping.")
            continue

        param_dist_current = param_dist_map.get(model_key) if not args.skip_tuning else None

        results_tuple = tune_and_evaluate_predictor(
            predictor_class=PredictorClass,
            X_train=X_train.copy(), y_train_home=y_train_home.copy(), y_train_away=y_train_away.copy(),
            X_val=X_val.copy(), y_val_home=y_val_home.copy(), y_val_away=y_val_away.copy(),
            X_test=X_test.copy(), y_test_home=y_test_home.copy(), y_test_away=y_test_away.copy(),
            model_name_prefix=model_key,
            feature_list=final_feature_list_for_models,
            param_dist=param_dist_current,
            n_iter=args.tune_iterations,
            n_splits=args.cv_splits,
            scoring=args.scoring_metric,
            use_recency_weights=args.use_weights,
            weight_method=args.weight_method,
            weight_half_life=args.weight_half_life,
            visualize=args.visualize,
            save_plots=args.save_plots,
        )

        if results_tuple:
            metrics_result, val_preds = results_tuple
            if metrics_result:
                all_metrics.append(metrics_result)
            if val_preds:
                if (pd.notna(val_preds.get('pred_home')).any() and
                    pd.notna(val_preds.get('pred_away')).any() and
                    len(val_preds.get('pred_home', pd.Series(dtype=float))) == len(X_val)):
                    validation_predictions_collector[model_key] = val_preds
                    logger.info(f"Collected validation predictions for {model_key}.")
                else:
                    logger.warning(f"Validation predictions for {model_key} appear invalid or empty. Skipping collection.")
            else:
                logger.warning(f"Did not receive validation predictions for {model_key}.")
        else:
            logger.error(f"Training/Evaluation failed entirely for {model_key}.")

    successful_base_models = list(validation_predictions_collector.keys())
    if not successful_base_models:
        logger.error("No base models produced valid validation predictions. Cannot proceed to calculate weights.")
        sys.exit(1)
    logger.info(f"Base models with validation predictions: {successful_base_models}")

    logger.info("\n--- Analyzing Base Model Validation Prediction Correlations ---")
    try:
        meta_features_list_corr = []
        reference_index_corr = None
        first_valid_key_corr = next((k for k in successful_base_models if k in validation_predictions_collector), None)
        if first_valid_key_corr:
            reference_index_corr = validation_predictions_collector[first_valid_key_corr]['pred_home'].index
        else:
            raise ValueError("Could not get validation predictions for correlation analysis.")

        for model_key in successful_base_models:
            preds_dict = validation_predictions_collector[model_key]
            df_home = preds_dict['pred_home'].rename(f"{model_key}_pred_home").reindex(reference_index_corr)
            df_away = preds_dict['pred_away'].rename(f"{model_key}_pred_away").reindex(reference_index_corr)
            meta_features_list_corr.extend([df_home, df_away])

        X_meta_train_corr = pd.concat(meta_features_list_corr, axis=1).dropna()
        if not X_meta_train_corr.empty:
            logger.info(f"Constructed meta-feature matrix for correlation with shape: {X_meta_train_corr.shape}")
            logger.info("Calculating correlations between base model validation predictions...")
            correlation_matrix = X_meta_train_corr.corr()
            logger.info("\nCorrelation Matrix:\n" + correlation_matrix.to_string())
            if args.save_plots or args.visualize:
                try:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title("Correlation Matrix of Base Model Validation Predictions")
                    plt.tight_layout()
                    if args.save_plots:
                        plot_path = REPORTS_DIR / f"validation_pred_correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                        plt.savefig(plot_path)
                        logger.info(f"Correlation heatmap saved to {plot_path}")
                    if args.visualize:
                        plt.show()
                    plt.close()
                except ImportError:
                    logger.warning("Seaborn/Matplotlib not found, cannot generate heatmap plot.")
                except Exception as plot_err:
                    logger.error(f"Error generating correlation heatmap: {plot_err}")
        else:
            logger.warning("No valid data remaining for correlation calculation after dropna.")
    except Exception as prep_corr_e:
        logger.error(f"Error preparing data for correlation analysis: {prep_corr_e}", exc_info=True)

    logger.info("\n--- Calculating Validation MAEs and Ensemble Weights ---")
    validation_mae_dict = {}
    for model_key, val_preds_dict in validation_predictions_collector.items():
        try:
            true_home = val_preds_dict['true_home']
            pred_home = val_preds_dict['pred_home']
            true_away = val_preds_dict['true_away']
            pred_away = val_preds_dict['pred_away']
            if (true_home.isnull().any() or pred_home.isnull().any() or
                true_away.isnull().any() or pred_away.isnull().any()):
                logger.warning(f"NaNs detected in validation predictions/targets for {model_key}. Skipping MAE calculation for this model.")
                validation_mae_dict[model_key] = np.nan
                continue
            mae_home = mean_absolute_error(true_home, pred_home)
            mae_away = mean_absolute_error(true_away, pred_away)
            avg_mae = (mae_home + mae_away) / 2.0
            validation_mae_dict[model_key] = avg_mae
            logger.info(f"Validation MAE for {model_key}: Home={mae_home:.4f}, Away={mae_away:.4f}, Avg={avg_mae:.4f}")
        except KeyError as ke:
            logger.error(f"Missing expected keys in validation predictions for {model_key}: {ke}. Cannot calculate MAE.")
            validation_mae_dict[model_key] = np.nan
        except Exception as mae_err:
            logger.error(f"Error calculating validation MAE for {model_key}: {mae_err}")
            validation_mae_dict[model_key] = np.nan

    ensemble_weights = compute_weights(validation_mae_dict)
    if ensemble_weights:
        weights_path = MAIN_MODELS_DIR / "ensemble_weights.json"
        try:
            with open(weights_path, 'w') as f:
                json.dump(ensemble_weights, f, indent=4)
            logger.info(f"Saved ensemble weights (based on validation MAE) to {weights_path}")
        except Exception as e:
            logger.error(f"Failed to save ensemble weights: {e}", exc_info=True)
    else:
        logger.error("Ensemble weights could not be computed. Weights file not saved.")

    logger.info("\n--- Evaluating Weighted Ensemble (Inverse Validation MAE Weights) on Test Set ---")
    base_model_test_preds = {}
    models_loaded_for_eval = {}

    logger.info("Loading saved base models for ensemble evaluation...")
    models_to_reload = list(predictor_map.keys())
    if ensemble_weights:
        models_to_reload = list(ensemble_weights.keys())

    for model_key in models_to_reload:
        model_loaded = False
        for m in all_metrics:
            full_model_name = m.get('model_name', '')
            save_path = m.get('save_path')
            if full_model_name.split('_score_predictor')[0].lower() == model_key.lower() and save_path and os.path.exists(save_path):
                 try:
                     PredictorClass = predictor_map.get(model_key)
                     if PredictorClass:
                          predictor = PredictorClass(model_dir=str(MAIN_MODELS_DIR), model_name=f"{model_key}_score_predictor")
                          predictor.load_model(filepath=save_path)
                          if getattr(predictor, 'pipeline_home', None) and getattr(predictor, 'pipeline_away', None):
                              models_loaded_for_eval[model_key] = predictor
                              model_loaded = True
                              logger.info(f"Successfully loaded model {model_key} from {save_path}")
                              break
                          else:
                              logger.error(f"Loaded model {model_key} from {save_path} but pipelines missing.")
                     else:
                          logger.error(f"Predictor class not found for {model_key} during reload.")
                 except Exception as load_err:
                     logger.error(f"Failed to reload model {model_key} from {save_path}: {load_err}")
        if not model_loaded:
            logger.error(f"Could not reload model {model_key}. It cannot be used in evaluation.")

    logger.info("Generating test set predictions from reloaded base models...")
    if not models_loaded_for_eval:
         logger.error("No base models successfully reloaded. Cannot evaluate any weighted ensemble.")
    else:
        try:
            X_test_eval = X_test[final_feature_list_for_models].copy()
            for model_key, predictor in models_loaded_for_eval.items():
                logger.debug(f"Predicting test set with reloaded {model_key}...")
                preds_df = predictor.predict(X_test_eval)
                if preds_df is not None and not preds_df.empty and all(c in preds_df for c in ['predicted_home_score', 'predicted_away_score']):
                    base_model_test_preds[model_key] = preds_df.loc[X_test_eval.index, ['predicted_home_score', 'predicted_away_score']]
                    logger.debug(f"Stored test predictions for {model_key}")
                else:
                     logger.warning(f"Failed to get valid test predictions from reloaded {model_key}.")
                     base_model_test_preds[model_key] = None
        except Exception as repred_err:
            logger.error(f"Error during re-prediction for ensemble evaluation: {repred_err}", exc_info=True)
            base_model_test_preds = {}

        if not ensemble_weights:
            logger.info("Skipping evaluation using inverse validation MAE weights (weights unavailable).")
        elif not base_model_test_preds:
            logger.error("Skipping evaluation using inverse validation MAE weights (base predictions missing).")
        else:
            logger.info("Calculating weighted average predictions for test set (using Inv Val MAE weights)...")
            weighted_home_sum = pd.Series(0.0, index=X_test.index)
            weighted_away_sum = pd.Series(0.0, index=X_test.index)
            total_weight_applied_series = pd.Series(0.0, index=X_test.index)
            successful_models_for_inv_mae_avg = [k for k, v in base_model_test_preds.items() if v is not None]
            for model_key in successful_models_for_inv_mae_avg:
                 weight = ensemble_weights.get(model_key, 0.0)
                 preds_df = base_model_test_preds.get(model_key)
                 if weight > 1e-6 and preds_df is not None:
                    preds_df = preds_df.reindex(X_test.index)
                    valid_idx = preds_df['predicted_home_score'].notna() & preds_df['predicted_away_score'].notna()
                    weighted_home_sum.loc[valid_idx] += preds_df.loc[valid_idx, 'predicted_home_score'] * weight
                    weighted_away_sum.loc[valid_idx] += preds_df.loc[valid_idx, 'predicted_away_score'] * weight
                    total_weight_applied_series.loc[valid_idx] += weight
            final_pred_home_inv_mae = pd.Series(np.nan, index=X_test.index)
            final_pred_away_inv_mae = pd.Series(np.nan, index=X_test.index)
            valid_weight_idx = total_weight_applied_series > 1e-6
            final_pred_home_inv_mae.loc[valid_weight_idx] = weighted_home_sum.loc[valid_weight_idx] / total_weight_applied_series.loc[valid_weight_idx]
            final_pred_away_inv_mae.loc[valid_weight_idx] = weighted_away_sum.loc[valid_weight_idx] / total_weight_applied_series.loc[valid_weight_idx]
            fallback_idx = ~valid_weight_idx
            if fallback_idx.any():
                logger.warning(f"Applying simple average fallback for {fallback_idx.sum()} test samples (Inv Val MAE weights).")
                valid_dfs_for_avg = [df for df in base_model_test_preds.values() if df is not None]
                if valid_dfs_for_avg:
                     simple_avg_home = pd.concat([df['predicted_home_score'] for df in valid_dfs_for_avg], axis=1).mean(axis=1)
                     simple_avg_away = pd.concat([df['predicted_away_score'] for df in valid_dfs_for_avg], axis=1).mean(axis=1)
                     final_pred_home_inv_mae.loc[fallback_idx] = simple_avg_home.loc[fallback_idx]
                     final_pred_away_inv_mae.loc[fallback_idx] = simple_avg_away.loc[fallback_idx]
                final_pred_home_inv_mae = final_pred_home_inv_mae.fillna(115.0)
                final_pred_away_inv_mae = final_pred_away_inv_mae.fillna(115.0)
            logger.info("Evaluating weighted ensemble predictions (Inv Val MAE weights)...")
            try:
                 y_test_home_aligned_w = y_test_home.reindex(final_pred_home_inv_mae.index)
                 y_test_away_aligned_w = y_test_away.reindex(final_pred_away_inv_mae.index)
                 valid_final_idx = final_pred_home_inv_mae.notna() & final_pred_away_inv_mae.notna() & y_test_home_aligned_w.notna() & y_test_away_aligned_w.notna()
                 if not valid_final_idx.all():
                     logger.warning(f"Dropping {(~valid_final_idx).sum()} samples with NaNs before Inv Val MAE weighted evaluation.")
                 final_pred_home_inv_mae = final_pred_home_inv_mae[valid_final_idx]
                 final_pred_away_inv_mae = final_pred_away_inv_mae[valid_final_idx]
                 y_test_home_aligned_w = y_test_home_aligned_w[valid_final_idx]
                 y_test_away_aligned_w = y_test_away_aligned_w[valid_final_idx]
                 if final_pred_home_inv_mae.empty:
                     logger.error("No valid predictions remaining for Inv Val MAE weighted evaluation.")
                 else:
                    ens_metrics_home = calculate_regression_metrics(y_test_home_aligned_w, final_pred_home_inv_mae)
                    ens_metrics_away = calculate_regression_metrics(y_test_away_aligned_w, final_pred_away_inv_mae)
                    ens_mae_total = mean_absolute_error(y_test_home_aligned_w + y_test_away_aligned_w, final_pred_home_inv_mae + final_pred_away_inv_mae)
                    ens_mae_diff = mean_absolute_error(y_test_home_aligned_w - y_test_away_aligned_w, final_pred_home_inv_mae - final_pred_away_inv_mae)
                    y_true_comb_ens = np.vstack((y_test_home_aligned_w.values, y_test_away_aligned_w.values)).T
                    y_pred_comb_ens = np.vstack((final_pred_home_inv_mae.values, final_pred_away_inv_mae.values)).T
                    ens_betting = calculate_betting_metrics(y_true_comb_ens, y_pred_comb_ens)
                    logger.info(f"WEIGHTED ENSEMBLE (Inv Val MAE) Test MAE : Home={ens_metrics_home.get('mae', np.nan):.3f}, Away={ens_metrics_away.get('mae', np.nan):.3f}, Total={ens_mae_total:.3f}, Diff={ens_mae_diff:.3f}")
                    logger.info(f"WEIGHTED ENSEMBLE (Inv Val MAE) Test RMSE: Home={ens_metrics_home.get('rmse', np.nan):.3f}, Away={ens_metrics_away.get('rmse', np.nan):.3f}")
                    logger.info(f"WEIGHTED ENSEMBLE (Inv Val MAE) Test R2  : Home={ens_metrics_home.get('r2', np.nan):.3f}, Away={ens_metrics_away.get('r2', np.nan):.3f}")
                    logger.info(f"WEIGHTED ENSEMBLE (Inv Val MAE) Betting Metrics: {ens_betting}")
            except Exception as ens_eval_err:
                 logger.error(f"Error evaluating Inv Val MAE weighted ensemble: {ens_eval_err}", exc_info=True)

        logger.info("\n--- Evaluating MANUALLY Weighted Ensemble on Test Set ---")
        manual_weights = {'ridge': 0.55, 'random_forest': 0.25, 'xgboost': 0.20}
        logger.info(f"Using Manual Weights: {manual_weights}")
        if not base_model_test_preds:
            logger.error("Missing base model test predictions. Cannot calculate manually weighted ensemble performance.")
        else:
            logger.info("Calculating MANUALLY weighted average predictions for test set...")
            manual_weighted_home_sum = pd.Series(0.0, index=X_test.index)
            manual_weighted_away_sum = pd.Series(0.0, index=X_test.index)
            manual_total_weight_applied_series = pd.Series(0.0, index=X_test.index)
            models_in_manual_weights = list(manual_weights.keys())
            for model_key in models_in_manual_weights:
                 weight = manual_weights.get(model_key, 0.0)
                 preds_df = base_model_test_preds.get(model_key)
                 if weight > 1e-6 and preds_df is not None:
                     preds_df = preds_df.reindex(X_test.index)
                     valid_idx = preds_df['predicted_home_score'].notna() & preds_df['predicted_away_score'].notna()
                     manual_weighted_home_sum.loc[valid_idx] += preds_df.loc[valid_idx, 'predicted_home_score'] * weight
                     manual_weighted_away_sum.loc[valid_idx] += preds_df.loc[valid_idx, 'predicted_away_score'] * weight
                     manual_total_weight_applied_series.loc[valid_idx] += weight
                 elif weight > 1e-6 and preds_df is None:
                      logger.warning(f"Model {model_key} needed for manual weights but test predictions missing. Skipping.")
            manual_final_pred_home = pd.Series(np.nan, index=X_test.index)
            manual_final_pred_away = pd.Series(np.nan, index=X_test.index)
            manual_valid_weight_idx = manual_total_weight_applied_series > 1e-6
            manual_final_pred_home.loc[manual_valid_weight_idx] = manual_weighted_home_sum.loc[manual_valid_weight_idx] / manual_total_weight_applied_series.loc[manual_valid_weight_idx]
            manual_final_pred_away.loc[manual_valid_weight_idx] = manual_weighted_away_sum.loc[manual_valid_weight_idx] / manual_total_weight_applied_series.loc[manual_valid_weight_idx]
            manual_fallback_idx = ~manual_valid_weight_idx
            if manual_fallback_idx.any():
                logger.warning(f"Applying simple average fallback for {manual_fallback_idx.sum()} test samples (manual weights).")
                valid_dfs_for_avg = [df for df in base_model_test_preds.values() if df is not None]
                if valid_dfs_for_avg:
                     simple_avg_home = pd.concat([df['predicted_home_score'] for df in valid_dfs_for_avg], axis=1).mean(axis=1)
                     simple_avg_away = pd.concat([df['predicted_away_score'] for df in valid_dfs_for_avg], axis=1).mean(axis=1)
                     manual_final_pred_home.loc[manual_fallback_idx] = simple_avg_home.loc[manual_fallback_idx]
                     manual_final_pred_away.loc[manual_fallback_idx] = simple_avg_away.loc[manual_fallback_idx]
                manual_final_pred_home = manual_final_pred_home.fillna(115.0)
                manual_final_pred_away = manual_final_pred_away.fillna(115.0)
            logger.info("Evaluating MANUALLY weighted ensemble predictions...")
            try:
                 y_test_home_aligned_w = y_test_home.reindex(manual_final_pred_home.index)
                 y_test_away_aligned_w = y_test_away.reindex(manual_final_pred_away.index)
                 valid_final_idx_man = manual_final_pred_home.notna() & manual_final_pred_away.notna() & y_test_home_aligned_w.notna() & y_test_away_aligned_w.notna()
                 if not valid_final_idx_man.all():
                     logger.warning(f"Dropping {(~valid_final_idx_man).sum()} samples with NaNs before final manual weighted evaluation.")
                 manual_final_pred_home = manual_final_pred_home[valid_final_idx_man]
                 manual_final_pred_away = manual_final_pred_away[valid_final_idx_man]
                 y_test_home_aligned_w = y_test_home_aligned_w[valid_final_idx_man]
                 y_test_away_aligned_w = y_test_away_aligned_w[valid_final_idx_man]
                 if manual_final_pred_home.empty:
                     logger.error("No valid predictions remaining for manual weighted ensemble evaluation.")
                 else:
                    man_ens_metrics_home = calculate_regression_metrics(y_test_home_aligned_w, manual_final_pred_home)
                    man_ens_metrics_away = calculate_regression_metrics(y_test_away_aligned_w, manual_final_pred_away)
                    man_ens_mae_total = mean_absolute_error(y_test_home_aligned_w + y_test_away_aligned_w, manual_final_pred_home + manual_final_pred_away)
                    man_ens_mae_diff = mean_absolute_error(y_test_home_aligned_w - y_test_away_aligned_w, manual_final_pred_home - manual_final_pred_away)
                    y_true_comb_ens_man = np.vstack((y_test_home_aligned_w.values, y_test_away_aligned_w.values)).T
                    y_pred_comb_ens_man = np.vstack((manual_final_pred_home.values, manual_final_pred_away.values)).T
                    man_ens_betting = calculate_betting_metrics(y_true_comb_ens_man, y_pred_comb_ens_man)
                    logger.info(f"MANUAL WEIGHTED ENSEMBLE Test MAE : Home={man_ens_metrics_home.get('mae', np.nan):.3f}, Away={man_ens_metrics_away.get('mae', np.nan):.3f}, Total={man_ens_mae_total:.3f}, Diff={man_ens_mae_diff:.3f}")
                    logger.info(f"MANUAL WEIGHTED ENSEMBLE Test RMSE: Home={man_ens_metrics_home.get('rmse', np.nan):.3f}, Away={man_ens_metrics_away.get('rmse', np.nan):.3f}")
                    logger.info(f"MANUAL WEIGHTED ENSEMBLE Test R2  : Home={man_ens_metrics_home.get('r2', np.nan):.3f}, Away={man_ens_metrics_away.get('r2', np.nan):.3f}")
                    logger.info(f"MANUAL WEIGHTED ENSEMBLE Betting Metrics: {man_ens_betting}")
            except Exception as man_ens_eval_err:
                 logger.error(f"Error evaluating manually weighted ensemble performance: {man_ens_eval_err}", exc_info=True)
    logger.info("\n--- Training Pipeline Summary ---")
    if all_metrics:
         metrics_df = pd.DataFrame(all_metrics)
         cols_to_show = [
             'model_name', 'feature_count', 'training_duration_final',
             'test_mae_home', 'test_mae_away', 'test_r2_home', 'test_r2_away',
             'test_mae_total', 'test_mae_diff'
         ]
         if 'test_combined_loss' in metrics_df.columns:
             cols_to_show.append('test_combined_loss')
         if 'betting_metrics' in metrics_df.columns:
             cols_to_show.append('betting_metrics')
         cols_to_show_present = [col for col in cols_to_show if col in metrics_df.columns]
         logger.info("Base Model Performance (Test Set):")
         try:
             metrics_df_display = metrics_df[cols_to_show_present].copy()
             if 'betting_metrics' in metrics_df_display.columns:
                  metrics_df_display['betting_metrics'] = metrics_df_display['betting_metrics'].astype(str)
             logger.info("\n" + metrics_df_display.to_string(index=False))
         except Exception as e:
              logger.warning(f"Could not format summary table: {e}")
              logger.info("\n" + metrics_df[cols_to_show_present].to_string(index=False))
    if ensemble_weights:
        logger.info("\nCalculated Ensemble Weights (Inverse Validation MAE):")
        logger.info(json.dumps(ensemble_weights, indent=4))
    else:
         logger.warning("\nEnsemble weights (Inv Val MAE) were not calculated successfully.")

    logger.info(f"\nManual Ensemble Weights Used for Evaluation: {manual_weights}")
    end_time_pipeline = time.time()
    logger.info(f"\n--- NBA Model Training Pipeline Finished in {end_time_pipeline - start_pipeline_time:.2f} seconds ---")
 
def load_base_model(model_key: str, all_metrics: list, default_key: str = 'model') -> Optional[Any]:
    for m in all_metrics:
        full_model_name = m.get('model_name', '')
        if full_model_name.split('_score_predictor')[0].lower() == model_key.lower():
            saved_path = m.get('save_path')
            if not saved_path or not os.path.exists(saved_path):
                logger.warning(f"Saved path missing or invalid for {model_key}: '{saved_path}'.")
                continue
            try:
                loaded_data = joblib.load(saved_path)
                if isinstance(loaded_data, dict) and default_key in loaded_data:
                    logger.info(f"Loaded predictor for {model_key} using key '{default_key}'.")
                    return loaded_data[default_key]
                elif hasattr(loaded_data, 'predict'):
                    logger.info(f"Loaded predictor for {model_key} as a standalone object.")
                    return loaded_data
                else:
                    logger.error(f"Loaded object for {model_key} from {saved_path} does not contain key '{default_key}' or a predict method.")
            except Exception as e:
                logger.error(f"Failed to load base model {model_key} from {saved_path}: {e}", exc_info=True)
    logger.warning(f"Could not load base model for {model_key}.")
    return None

parser = argparse.ArgumentParser(description="NBA Score Prediction Model Tuning & Training Pipeline")
parser.add_argument("--data-source", type=str, default="supabase", choices=["csv", "supabase", "database", "dummy"], help="Data source type")
parser.add_argument("--historical-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_game_stats.csv'), help="Path to historical games CSV")
parser.add_argument("--team-stats-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_team_stats.csv'), help="Path to team stats CSV")
parser.add_argument("--lookback-days", type=int, default=1095, help="Days of historical data to load")
parser.add_argument("--models", type=str, default="xgboost,random_forest,ridge", help="Comma-separated models to train")
parser.add_argument("--rolling-windows", type=str, default="5,10,20", help="Comma-separated rolling window sizes")
parser.add_argument("--h2h-window", type=int, default=5, help="Number of games for H2H features")
parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for test set")
parser.add_argument("--val-size", type=float, default=0.15, help="Fraction for validation set")
parser.add_argument("--use-weights", action="store_true", help="Use recency weighting during training")
parser.add_argument("--weight-method", type=str, default="exponential", choices=["exponential", "half_life"], help="Recency weighting method")
parser.add_argument("--weight-half-life", type=int, default=90, help="Half-life in days for weights")
parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning")
parser.add_argument("--tune-iterations", type=int, default=30, help="Iterations for RandomizedSearchCV")
parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS, help="CV splits for TimeSeriesSplit")
parser.add_argument("--scoring-metric", type=str, default='neg_mean_absolute_error', help="Scoring metric for tuning")
parser.add_argument("--run-analysis", action="store_true", help="Run optional analysis functions")
parser.add_argument("--visualize", action="store_true", help="Show plots interactively")
parser.add_argument("--save-plots", action="store_true", help="Save generated plots to reports directory")
parser.add_argument("--allow-dummy", action="store_true", help="Allow script to run using dummy classes if imports fail")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")

if __name__ == '__main__':
    cli_args = parser.parse_args()
    run_training_pipeline(cli_args)
