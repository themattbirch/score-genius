# backend/nba_score_prediction/train_models.py

from __future__ import annotations  # Keep this at the top
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from sklearn.linear_model import Ridge as MetaRidge

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
from sklearn.linear_model import Ridge as MetaRidge
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
    from .prediction import fetch_and_parse_betting_odds # Import the function
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
    # Define dummy utils functions if main import failed
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
logging.getLogger("httpcore").setLevel(logging.WARNING) # httpcore is often used by httpx
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

# In train_models.py

def load_data_source(source_type: str, lookback_days: int, args: argparse.Namespace,
                     db_engine: Optional[Any] = None, supabase_client: Optional["Client"] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads historical game data and team stats based on the source type."""
    logger.info(f"Attempting to load data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()

    # --- Re-ADD desired columns to hist_required_cols for debugging ---
    hist_required_cols = [
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot', 'away_q1', 'away_q2',
        'away_q3', 'away_q4', 'away_ot',
        'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
        'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
        'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
        # Re-add columns previously removed, assuming they exist in Supabase table schema:
        'home_assists', 'home_steals', 'home_blocks', 'home_turnovers', 'home_fouls',
        'away_assists', 'away_steals', 'away_blocks', 'away_turnovers', 'away_fouls',
        'home_off_reb', 'home_def_reb', 'home_total_reb',
        'away_off_reb', 'away_def_reb', 'away_total_reb'
        # Add any other columns you expect from the Supabase table if needed
    ]
    logger.debug(f"DEBUG RUN: Requesting {len(hist_required_cols)} columns: {hist_required_cols}")
    # Update hist_numeric_cols based on the potentially expanded hist_required_cols
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
                select_str = ", ".join(hist_required_cols) # Use the expanded list
                logger.debug(f"Supabase select string: {select_str}") # Log the exact select string

                while has_more:
                    response = supabase_client.table("nba_historical_game_stats") \
                        .select(select_str) \
                        .gte("game_date", threshold_date) \
                        .order('game_date') \
                        .range(start_index, start_index + page_size - 1) \
                        .execute()
                    # Basic check on response status if possible (depends on library version)
                    # if hasattr(response, 'status_code') and response.status_code != 200:
                    #    logger.warning(f"Supabase request returned status {response.status_code}")

                    batch = response.data
                    b_size = len(batch)
                    all_historical_data.extend(batch)
                    logger.debug(f"Retrieved {b_size} historical records...")
                    if b_size < page_size:
                        has_more = False
                    else:
                        start_index += page_size
                        # Optional: Add a small delay if hitting rate limits, though unlikely for reads
                        # time.sleep(0.1)

                # --- DEBUGGING BLOCK START ---
                if all_historical_data:
                    logger.debug(f"Raw Supabase response processing {len(all_historical_data)} records.")
                    try:
                        # Check the keys of the *first* record returned by Supabase
                        first_record_keys = list(all_historical_data[0].keys())
                        num_keys_raw = len(first_record_keys)
                        logger.debug(f"Keys count in first raw record: {num_keys_raw}")
                        logger.debug(f"Keys sample in first raw record: {sorted(first_record_keys)}") # Log sorted keys

                        missing_keys_in_raw = sorted([col for col in hist_required_cols if col not in first_record_keys])
                        if missing_keys_in_raw:
                            logger.error(f"COLUMNS MISSING IN RAW SUPABASE RESPONSE ({len(missing_keys_in_raw)} missing): {missing_keys_in_raw}")
                        else:
                            logger.info("All requested columns appear present in raw Supabase response.")

                    except IndexError:
                         logger.error("Cannot inspect raw Supabase response keys: all_historical_data is empty list.")
                    except Exception as e_debug_raw:
                         logger.error(f"Error inspecting raw Supabase response: {e_debug_raw}")

                    # Create DataFrame
                    logger.debug("Attempting to create DataFrame from raw data...")
                    try:
                        hist_df = pd.DataFrame(all_historical_data)
                        logger.info(f"Loaded {len(hist_df)} historical records into DataFrame.")

                        # Check columns *after* creating the DataFrame
                        df_cols = sorted(list(hist_df.columns))
                        num_cols_df = len(df_cols)
                        logger.debug(f"Columns count in created DataFrame: {num_cols_df}")
                        logger.debug(f"Columns sample in created DataFrame: {df_cols}") # Log sorted columns

                        missing_cols_in_df = sorted([col for col in hist_required_cols if col not in df_cols])
                        if missing_cols_in_df:
                            # This is where the problem likely lies if keys were present in raw data
                            logger.error(f"COLUMNS MISSING AFTER DataFrame CREATION ({len(missing_cols_in_df)} missing): {missing_cols_in_df}")
                        else:
                            logger.info("All requested columns appear present in created DataFrame.")

                        # Check for columns that are ALL NaN/None after creation
                        problematic_cols = []
                        for col in hist_required_cols:
                             if col in hist_df.columns and hist_df[col].isnull().all():
                                  problematic_cols.append(col)
                        if problematic_cols:
                             logger.warning(f"Columns present in DataFrame but contain ONLY NaN/None values: {problematic_cols}")

                    except Exception as e_df_create:
                         logger.error(f"Error creating DataFrame from Supabase data: {e_df_create}")
                         hist_df = pd.DataFrame() # Ensure it's empty

                else:
                     logger.warning("No historical data retrieved from Supabase (all_historical_data is empty).")
                # --- DEBUGGING BLOCK END ---

                # Proceed with standard processing if df created
                if not hist_df.empty:
                    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
                    hist_df = hist_df.dropna(subset=['game_date'])
                    logger.debug(f"DataFrame shape after date processing: {hist_df.shape}")
                else:
                     logger.warning("hist_df is empty after Supabase load/debug checks.")

            except Exception as e:
                logger.error(f"Error loading historical games from Supabase: {e}", exc_info=True)
                hist_df = pd.DataFrame() # Ensure it's empty on error

    # --- Load Team Stats (remains the same) ---
    elif source_type == "csv":
        # ... (CSV loading logic remains the same) ...
        pass # Placeholder for brevity

    # --- Load Team Stats from Supabase (remains the same) ---
    if source_type != "csv": # Avoid double loading if source is CSV
        logger.info("Loading team stats from Supabase...")
        # ... (Team stats loading logic remains the same) ...
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


    # --- Final Processing & NaN Checks (remains the same) ---
    if not hist_df.empty:
        # Check for completely missing columns *before* conversion attempt
        missing_hist_cols = [col for col in hist_required_cols if col not in hist_df.columns]
        if missing_hist_cols:
            logger.error(f"CRITICAL FINAL CHECK: Columns missing before numeric conversion: {missing_hist_cols}. Check DataFrame creation logs.")
            # Optionally add them here if needed for downstream, though the error indicates a deeper issue
            # for col in missing_hist_cols: hist_df[col] = np.nan

        logger.info("Converting historical numeric columns and handling NaNs...")
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                # Store original dtype for debug
                # original_dtype = str(hist_df[col].dtype)
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                # Log if conversion resulted in all NaNs for a previously non-all-NaN column
                # if not original_all_nan and hist_df[col].isnull().all():
                #      logger.warning(f"Column '{col}' became all NaN after pd.to_numeric (original dtype: {original_dtype}). Check source data type.")
                hist_df[col] = hist_df[col].fillna(np.nan) # Use NaN for fill, handle later if needed
            else:
                 # This shouldn't happen if the check above worked, but as a safeguard:
                 logger.warning(f"Numeric column '{col}' still missing before fillna; adding as NaN.")
                 hist_df[col] = np.nan

        if args.debug:
            # ... (NaN check logic remains the same) ...
            pass

        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)

    # Log final check before returning
    if not hist_df.empty:
         final_df_cols = sorted(list(hist_df.columns))
         logger.debug(f"Final columns in hist_df before return ({len(final_df_cols)}): {final_df_cols}")
         missing_final = sorted([col for col in hist_required_cols if col not in final_df_cols])
         if missing_final:
              logger.error(f"Columns confirmed missing in final hist_df before return: {missing_final}")

    logger.info(f"Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df

def visualize_recency_weights(dates, weights, title="Recency Weights Distribution", save_path=None):
    """Generates and optionally saves a plot of recency weights over time."""
    if dates is None or weights is None or len(dates) != len(weights):
        logger.warning("Invalid input for visualizing recency weights.")
        return
    try:
        # Ensure matplotlib is available if we reach here
        import matplotlib.pyplot as plt
        import pandas as pd # Make sure pandas is imported if not already globally

        df_plot = pd.DataFrame({'date': pd.to_datetime(dates), 'weight': weights}).sort_values('date')
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['date'], df_plot['weight'], marker='.', linestyle='-')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sample Weight")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path) # Ensure it's a Path object
            save_path.parent.mkdir(parents=True, exist_ok=True) # Create directory if needed
            plt.savefig(save_path)
            logger.info(f"Recency weights plot saved to {save_path}")

        # Decide whether to show the plot interactively - depends on `visualize` flag passed to the outer function
        # The outer function `tune_and_evaluate_predictor` controls the show_plot logic for other plots.
        # For simplicity here, we can rely on the standard matplotlib behavior (shows if not in non-interactive mode)
        # or explicitly add plt.show() if needed, but it might interfere if running non-interactively.
        # Let's keep plt.show() commented out to avoid issues in non-interactive runs,
        # relying on the save_path for output if save_plots=True.
        # plt.show()
        plt.close() # Close the plot figure to free memory, especially if generating many plots

    except ImportError:
         logger.error("Matplotlib or Pandas not found. Cannot generate recency weights plot.")
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
    Ensures that the data_df has exactly the columns in REFINED_TOP_100_FEATURES.
    Missing features are added with a default value of 0.0.
    Returns the full list of predefined features.
    """
    logger.info(f"Selecting features based on predefined REFINED_TOP_100_FEATURES for context: {target_context}")
    # At this point, data_df is assumed to have been reindexed.
    logger.info(f"DataFrame now has {len(data_df.columns)} columns after reindexing.")
    return REFINED_TOP_100_FEATURES


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
) -> Tuple[Optional[Dict], Optional[Dict]]: # MODIFIED Return type hint
    """
    Tunes hyperparameters using RandomizedSearchCV on the training set,
    trains the final model on the combined training and validation sets using
    the best parameters (or defaults if tuning skipped/failed), evaluates
    on the test set, and generates predictions on the validation set.

    Args:
        # ... (Arguments remain the same) ...
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
        A tuple containing:
            - A dictionary with evaluation metrics and model info (or None if failed).
            - A dictionary with validation set predictions (pred_home, pred_away,
              true_home, true_away) keyed by index (or None if failed).
    """
    # Assuming metric/utility functions are globally available as per OLD code structure
    # (e.g., calculate_regression_metrics, compute_recency_weights, plotting functions)

    logger.info(f"tune_and_evaluate_predictor received feature_list with {len(feature_list)} features.")
    model_full_name = f"{model_name_prefix}_score_predictor"

    # --- Initial Checks ---
    if predictor_class is None: logger.error(f"Predictor class for {model_full_name} is None."); return None, None
    if model_name_prefix == 'xgboost' and not XGBOOST_AVAILABLE: logger.error("XGBoost unavailable."); return None, None
    if not feature_list: logger.error(f"Input feature list is empty for {model_full_name}."); return None, None

    logger.info(f"--- Tuning and Training {predictor_class.__name__} ({model_full_name}) ---")
    start_tune_time = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {}

    # --- Ensure Feature List is Unique ---
    # Use this unique list consistently throughout the function
    feature_list_unique = list(pd.Index(feature_list).unique())
    if len(feature_list_unique) != len(feature_list):
        logger.warning(f"Duplicate features found. Using {len(feature_list_unique)} unique features.")
    logger.debug(f"Using unique feature list ({len(feature_list_unique)} features).")

    # --- Combine Train and Validation Data for Final Fit ---
    # Include 'game_date' if needed for weights later
    train_val_date_cols = []
    if use_recency_weights and 'game_date' in X_train.columns and 'game_date' in X_val.columns:
        train_val_date_cols = ['game_date']
    # Select only the unique features + date column if needed
    cols_train = [col for col in feature_list_unique + train_val_date_cols if col in X_train.columns]
    cols_val = [col for col in feature_list_unique + train_val_date_cols if col in X_val.columns]
    try:
      
        X_train_full = pd.concat([X_train[cols_train], X_val[cols_val]], ignore_index=True)
        y_train_home_full = pd.concat([y_train_home, y_val_home], ignore_index=True)
        y_train_away_full = pd.concat([y_train_away, y_val_away], ignore_index=True)
        # (No need to re-align using .loc[X_train_full.index] anymore.)
        logger.info(f"Combined Train+Val size for final fit: {len(X_train_full)} samples.")
    except Exception as concat_err:
        logger.error(f"Error combining Train and Val sets: {concat_err}", exc_info=True)
        return None, None # Return None for both if setup fails

    # --- Prepare Data Specifically for Tuning (Using only Train set features) ---
    logger.debug(f"Preparing data for hyperparameter tuning using X_train.")
    try:
        # Select unique features from X_train
        X_tune_features = X_train[feature_list_unique].copy()
        # Align y_train using the index of the selected features
        y_tune_home = y_train_home.loc[X_tune_features.index].copy()
        # Away target usually not needed directly for tuning single-output MAE/RMSE
        # y_tune_away = y_train_away.loc[X_tune_features.index].copy()
        logger.debug(f"Tuning data shapes: X={X_tune_features.shape}, y={y_tune_home.shape}")
    except KeyError as ke:
        logger.error(f"KeyError preparing tuning data. Missing features? {ke}", exc_info=True)
        missing = set(feature_list_unique) - set(X_train.columns)
        if missing: logger.error(f"Features missing in X_train: {missing}")
        return None, None
    except Exception as prep_err:
         logger.error(f"Error preparing tuning data: {prep_err}", exc_info=True)
         return None, None

    # --- Calculate Sample Weights for Tuning (if enabled) ---
    tune_fit_params = {}
    if use_recency_weights:
        if 'game_date' in X_train.columns:
            logger.info("Calculating recency weights for tuning dataset (X_train)...")
            dates_for_weights_tune = X_train.loc[X_tune_features.index, 'game_date']
            sample_weights_tune = compute_recency_weights(dates_for_weights_tune, method=weight_method, half_life=weight_half_life)
            if sample_weights_tune is not None and len(sample_weights_tune) == len(X_tune_features):
                try: # Determine the correct parameter name (e.g., 'model__sample_weight')
                    temp_predictor = predictor_class(); temp_pipeline = temp_predictor._build_pipeline({})
                    model_step_name = temp_pipeline.steps[-1][0]
                    weight_param_name = f"{model_step_name}__sample_weight"
                    tune_fit_params[weight_param_name] = sample_weights_tune
                    logger.info(f"Sample weights for tuning prepared using key: '{weight_param_name}'")
                except Exception as e: logger.warning(f"Error determining weight parameter name for tuning: {e}. Weights may not be applied.")
            else: logger.warning(f"Failed tuning sample weights computation/alignment.")
        else: logger.warning("'use_recency_weights' enabled but 'game_date' missing in X_train.")

    # --- Hyperparameter Tuning (RandomizedSearchCV) ---
    if param_dist:
        logger.info(f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={n_splits}, scoring='{scoring}')...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        try:
            temp_predictor = predictor_class(); search_pipeline = temp_predictor._build_pipeline({})
            if search_pipeline is None: raise RuntimeError(f"Could not build pipeline for {predictor_class.__name__}")
            rs = RandomizedSearchCV(
                estimator=search_pipeline, param_distributions=param_dist, n_iter=n_iter, scoring=scoring,
                cv=tscv, n_jobs=-1, verbose=1, random_state=SEED, error_score='raise', refit=False # verbose=1 less noisy
            )
            # Fit using home target (or adjust if scorer needs both/different target)
            rs.fit(X_tune_features, y_tune_home, **tune_fit_params)
            # Log results...
            try:
                cv_results_df = pd.DataFrame(rs.cv_results_).sort_values(by='rank_test_score')
                logger.info("--- Top CV Results ---"); logger.info("\n" + cv_results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']].head().to_string())
            except Exception as report_e: logger.warning(f"Could not print CV results: {report_e}")
            best_params_raw = rs.best_params_; metrics['best_cv_score'] = rs.best_score_
            logger.info(f"Tuning complete. Best CV Score ({scoring}): {metrics['best_cv_score']:.4f}")
            # Strip pipeline prefix (e.g., 'model__n_estimators' -> 'n_estimators')
            best_params_final = {k.split('__', 1)[1]: v for k, v in best_params_raw.items()}
            metrics['best_params'] = best_params_final; logger.info(f"Best Final Params: {best_params_final}")
        except Exception as search_e:
            logger.error(f"RandomizedSearchCV failed: {search_e}", exc_info=True)
            best_params_final = {}; metrics['best_cv_score'] = None; metrics['best_params'] = 'default (tuning failed)'
        metrics['tuning_duration'] = time.time() - start_tune_time; logger.info(f"Tuning finished in {metrics['tuning_duration']:.2f}s.")
    else:
        logger.info("Skipping hyperparameter tuning."); metrics['best_params'] = 'default (tuning skipped)'; metrics['tuning_duration'] = 0.0

    # --- Train Final Model on Combined Train+Val Data ---
    logger.info("Training final model on combined Train+Val data...")
    # Instantiate the predictor, it will handle pipeline creation internally
    final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
    try:
        # Select the unique features from the combined Train+Val set
        X_train_full_for_fit = X_train_full[feature_list_unique].copy()
        # y targets are already aligned from the concat step
        y_train_home_full_aligned = y_train_home_full
        y_train_away_full_aligned = y_train_away_full
        logger.info(f"Shape of final training features X_train_full_for_fit: {X_train_full_for_fit.shape}")
    except KeyError as ke: logger.error(f"KeyError preparing final train data: {ke}", exc_info=True); return None, None
    except Exception as final_prep_err: logger.error(f"Error preparing final train data: {final_prep_err}", exc_info=True); return None, None

    # --- Calculate Sample Weights for Final Training (if enabled) ---
    final_sample_weights = None
    if use_recency_weights:
        if 'game_date' in X_train_full.columns:
            dates_for_weights_full = X_train_full.loc[X_train_full_for_fit.index, 'game_date']
            logger.info("Calculating recency weights for final training (Train+Val)...")
            final_sample_weights = compute_recency_weights(dates_for_weights_full, method=weight_method, half_life=weight_half_life)
            if final_sample_weights is None or len(final_sample_weights) != len(X_train_full_for_fit):
                logger.warning(f"Final sample weights failed/mismatched. Training without weights."); final_sample_weights = None
            else:
                logger.info(f"Final sample weights computed. Min: {np.min(final_sample_weights):.4f}, Max: {np.max(final_sample_weights):.4f}")
                if visualize:
                    try: # Visualize final weights
                        save_path_w = REPORTS_DIR / f"{model_full_name}_final_weights.png" if save_plots else None
                        visualize_recency_weights(dates_for_weights_full, final_sample_weights, title=f"Final Training Weights - {model_full_name}", save_path=save_path_w)
                    except NameError: logger.error("visualize_recency_weights function not found.")
                    except Exception as plot_w_e: logger.error(f"Error visualizing final weights: {plot_w_e}")
        else: logger.warning("'use_recency_weights' enabled but 'game_date' missing in final data.")

    # --- Fit the Final Model ---
    try:
        train_start_time = time.time()
        # Pass final weights directly to train method
        final_predictor.train(
            X_train=X_train_full_for_fit,
            y_train_home=y_train_home_full_aligned,
            y_train_away=y_train_away_full_aligned,
            hyperparams_home=best_params_final, # Use tuned params for both for now
            hyperparams_away=best_params_final,
            sample_weights=final_sample_weights # Pass computed weights (or None)
        )
        metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', time.time() - train_start_time)
        logger.info(f"Final model training completed in {metrics['training_duration_final']:.2f} seconds.")
        # Store final feature count used by the model
        metrics['feature_count'] = len(getattr(final_predictor, 'feature_names_in_', feature_list_unique))
        if not getattr(final_predictor, 'feature_names_in_', None): logger.warning("feature_names_in_ not found on model.")
    except Exception as train_e:
        logger.error(f"Training final model failed: {train_e}", exc_info=True)
        metrics['training_duration_final'] = None; metrics['feature_count'] = len(feature_list_unique)
        return metrics, None # Return partial metrics, None for val preds

    # --- Save the Final Model ---
    try:
        timestamp = getattr(final_predictor, 'training_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_filename = f"{model_full_name}_tuned_{timestamp}.joblib"
        save_path = final_predictor.save_model(filename=save_filename)
        if save_path: logger.info(f"Final tuned model saved to {save_path}"); metrics['save_path'] = str(save_path)
        else: logger.error(f"Model saving returned no path."); metrics['save_path'] = "Save failed"
    except Exception as save_e: logger.error(f"Error saving final model: {save_e}", exc_info=True); metrics['save_path'] = f"Save failed ({type(save_e).__name__})"

    # --- Evaluate Final Model on Test Set ---
    logger.info(f"Evaluating final {model_full_name} on test set ({len(X_test)})...")
    try:
        # Use unique feature list for test set prediction
        X_test_final = X_test[feature_list_unique].copy()
        logger.debug(f"Shape of test features X_test_final: {X_test_final.shape}")
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or 'predicted_home_score' not in predictions_df_test.columns: raise ValueError("Test prediction failed/invalid format.")
        pred_home_test = predictions_df_test['predicted_home_score']
        pred_away_test = predictions_df_test['predicted_away_score']
        # Align y_test using the prediction index
        y_test_home_aligned = y_test_home.loc[pred_home_test.index]
        y_test_away_aligned = y_test_away.loc[pred_away_test.index]

        # Calculate metrics (assuming functions are globally available)
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
        # Assuming custom loss functions are available
        try:
            y_true_comb_test = np.vstack((y_test_home_aligned.values, y_test_away_aligned.values)).T
            y_pred_comb_test = np.vstack((pred_home_test.values, pred_away_test.values)).T
            metrics['test_nba_score_loss'] = nba_score_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_nba_dist_loss'] = nba_distribution_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_combined_loss'] = combined_nba_loss(y_true_comb_test, y_pred_comb_test)
            metrics['betting_metrics'] = calculate_betting_metrics(y_true_comb_test, y_pred_comb_test, vegas_lines=None) # Pass lines if available
        except NameError as ne: logger.warning(f"Custom loss/betting function not found: {ne}")
        except Exception as custom_loss_e: logger.error(f"Error calculating custom losses: {custom_loss_e}")

        logger.info(f"FINAL Test MAE : Home={metrics.get('test_mae_home',np.nan):.3f}, Away={metrics.get('test_mae_away',np.nan):.3f}, Total={metrics.get('test_mae_total',np.nan):.3f}, Diff={metrics.get('test_mae_diff',np.nan):.3f}")
        logger.info(f"FINAL Test RMSE: Home={metrics.get('test_rmse_home',np.nan):.3f}, Away={metrics.get('test_rmse_away',np.nan):.3f}")
        logger.info(f"FINAL Test R2  : Home={metrics.get('test_r2_home',np.nan):.3f}, Away={metrics.get('test_r2_away',np.nan):.3f}")
        logger.info(f"FINAL Custom Losses: Score={metrics.get('test_nba_score_loss',np.nan):.3f}, Dist={metrics.get('test_nba_dist_loss',np.nan):.3f}, Combined={metrics.get('test_combined_loss',np.nan):.3f}")
        logger.info(f"FINAL Betting Metrics: {metrics.get('betting_metrics', {})}")

    except KeyError as ke: logger.error(f"KeyError evaluating test set: {ke}", exc_info=True); [metrics.setdefault(k, np.nan) for k in ['test_mae_home','test_mae_away', 'test_mae_total','test_mae_diff']] # etc.
    except Exception as test_eval_e: logger.error(f"Failed evaluating test set: {test_eval_e}", exc_info=True); [metrics.setdefault(k, np.nan) for k in ['test_mae_home','test_mae_away', 'test_mae_total','test_mae_diff']] # etc.


    # --- (Optional) Evaluate Final Model on Training Set ---
    # This evaluates on the same data the model was trained on (Train+Val)
    # Useful for checking over/underfitting
    logger.info(f"Evaluating final {model_full_name} on training set (Train+Val)...")
    try:
        # Predict on the same data used for final fitting
        predictions_df_train = final_predictor.predict(X_train_full_for_fit)
        if predictions_df_train is None or 'predicted_home_score' not in predictions_df_train.columns: raise ValueError("Train prediction failed.")
        pred_home_train = predictions_df_train['predicted_home_score']
        pred_away_train = predictions_df_train['predicted_away_score']
        # Use the already aligned y_train_..._full_aligned targets
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
        logger.info(f"FINAL Train MAE : Home={metrics.get('train_mae_home',np.nan):.3f}, Away={metrics.get('train_mae_away',np.nan):.3f}, Total={metrics.get('train_mae_total',np.nan):.3f}, Diff={metrics.get('train_mae_diff',np.nan):.3f}")
    except Exception as train_eval_e: logger.error(f"Failed evaluating on training set: {train_eval_e}", exc_info=True); [metrics.setdefault(k, np.nan) for k in ['train_mae_home','train_mae_away', 'train_mae_total','train_mae_diff']] # etc.


    # --- Store Sample Sizes ---
    metrics['samples_train_final'] = len(X_train_full_for_fit)
    metrics['samples_test'] = len(X_test_final) if 'X_test_final' in locals() else len(X_test) # Use len(X_test) as fallback

    # --- Generate Performance Plots (if enabled) ---
    if visualize or save_plots:
        timestamp_str = metrics.get('save_path') # Use saved model timestamp if available
        if timestamp_str and isinstance(timestamp_str, str):
             filename_part = Path(timestamp_str).name; ts_parts = filename_part.split('_')
             potential_ts = [p.split('.')[0] for p in ts_parts if p.replace('.','').isdigit() and len(p.split('.')[0])==15]
             timestamp = potential_ts[0] if potential_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
        else: timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = REPORTS_DIR / f"{model_full_name}_tuned_performance_{timestamp}"; plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating performance plots in {plot_dir}")
        if 'pred_home_test' in locals() and 'y_test_home_aligned' in locals():
            try: # Plotting block
                plot_actual_vs_predicted(y_true=y_test_home_aligned, y_pred=pred_home_test, title=f"Tuned {model_full_name} - Test Actual vs Pred (Home)",
                    metrics_dict={'rmse': metrics.get('test_rmse_home'), 'r2': metrics.get('test_r2_home'), 'mae': metrics.get('test_mae_home')},
                    save_path=plot_dir / "test_actual_vs_pred_home.png" if save_plots else None, show_plot=visualize)
                plot_actual_vs_predicted(y_true=y_test_away_aligned, y_pred=pred_away_test, title=f"Tuned {model_full_name} - Test Actual vs Pred (Away)",
                    metrics_dict={'rmse': metrics.get('test_rmse_away'), 'r2': metrics.get('test_r2_away'), 'mae': metrics.get('test_mae_away')},
                    save_path=plot_dir / "test_actual_vs_pred_away.png" if save_plots else None, show_plot=visualize)
                plot_residuals_analysis_detailed(y_true=y_test_home_aligned, y_pred=pred_home_test, title_prefix=f"Tuned {model_full_name} (Home) - Test Set", save_dir=plot_dir if save_plots else None, show_plot=visualize)
                plot_residuals_analysis_detailed(y_true=y_test_away_aligned, y_pred=pred_away_test, title_prefix=f"Tuned {model_full_name} (Away) - Test Set", save_dir=plot_dir if save_plots else None, show_plot=visualize)
                logger.info(f"Attempting feature importance for {model_full_name}...")
                pipeline_home = getattr(final_predictor, 'pipeline_home', None); pipeline_away = getattr(final_predictor, 'pipeline_away', None)
                features_in = getattr(final_predictor, 'feature_names_in_', feature_list_unique)
                if pipeline_home and pipeline_away and features_in:
                    plot_feature_importances(models_dict={f"{model_full_name}_Home": pipeline_home, f"{model_full_name}_Away": pipeline_away},
                         feature_names=features_in, top_n=30, plot_groups=True, save_dir=plot_dir / "feature_importance" if save_plots else None, show_plot=visualize)
                else: logger.warning("Cannot generate feature importance plots (missing pipeline/features).")
            except NameError as ne: logger.error(f"Plotting function not found: {ne}.")
            except Exception as plot_e: logger.error(f"Failed generating plots: {plot_e}", exc_info=True)
        else: logger.warning("Skipping plots (test predictions unavailable).")


    # --- <<< ADDED: Generate and Return Validation Set Predictions >>> ---
    val_predictions_dict = None # Initialize as None
    logger.info(f"Generating validation set predictions for {model_full_name} ({len(X_val)} samples)...")
    try:
        # Ensure X_val uses the same unique feature list used for training
        # This assumes X_val was passed in containing these columns
        X_val_final = X_val[feature_list_unique].copy()

        # Predict using the final trained model
        predictions_df_val = final_predictor.predict(X_val_final)
        if predictions_df_val is None or 'predicted_home_score' not in predictions_df_val.columns:
            raise ValueError(f"Prediction on validation set failed or returned invalid format for {model_full_name}.")

        # Align indices for safety before combining
        # Use index from X_val_final as the reference
        y_val_home_aligned = y_val_home.loc[X_val_final.index]
        y_val_away_aligned = y_val_away.loc[X_val_final.index]
        pred_home_val = predictions_df_val['predicted_home_score'].loc[X_val_final.index]
        pred_away_val = predictions_df_val['predicted_away_score'].loc[X_val_final.index]

        # Store predictions and true values indexed by the validation set's original index
        val_predictions_dict = {
            'pred_home': pred_home_val, # Already a Series with correct index
            'pred_away': pred_away_val, # Already a Series with correct index
            'true_home': y_val_home_aligned, # Already a Series with correct index
            'true_away': y_val_away_aligned  # Already a Series with correct index
        }
        logger.info(f"Successfully generated validation predictions for {model_full_name}.")

    except KeyError as ke_val:
        logger.error(f"FAILED generating validation predictions: KeyError. Features missing in X_val? {ke_val}", exc_info=True)
        missing_val = set(feature_list_unique) - set(X_val.columns)
        if missing_val: logger.error(f"Features missing in X_val: {missing_val}")
        val_predictions_dict = None # Ensure it's None on failure
    except Exception as val_pred_e:
        logger.error(f"FAILED generating validation set predictions for {model_full_name}: {val_pred_e}", exc_info=True)
        val_predictions_dict = None # Ensure it's None on failure
    # --- <<< END ADDED BLOCK >>> ---

    # --- Finalize and Return ---
    metrics['total_duration'] = time.time() - start_tune_time
    logger.info(f"--- Finished Tuning & Training {model_full_name} in {metrics['total_duration']:.2f}s ---")

    # --- MODIFIED RETURN ---
    # Return both metrics and validation predictions dictionary
    return metrics, val_predictions_dict
# ==============================================================================
# SECTION 6: EVALUATION & ANALYSIS FUNCTIONS (Placeholders)
# ==============================================================================
def analyze_main_predictors_learning_curves(*args, **kwargs): pass

# ==============================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# ==============================================================================
def run_training_pipeline(args):
    start_pipeline_time = time.time() # Make sure time is imported
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
    logger.info("Generating features for ALL historical data...") # Message is okay
    rolling_windows_list = [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [5, 10]
    logger.info(f"Using rolling windows: {rolling_windows_list}")

    # --- MODIFICATION START ---
    # Pass the *same* historical_df to both 'df' and 'historical_games_df'.
    # This ensures generate_all_features works on the full dataset first.
    # The internal logic of generate_all_features will handle the combination correctly
    # if both inputs are the same initially. The filtering step at the end becomes
    # slightly redundant but harmless in this case.
    logger.info(f"Processing {len(historical_df)} games for feature generation...")
    # --- Fetch Odds ---
    # Get relevant game IDs (e.g., from historical_df)
    all_game_ids = historical_df['game_id'].astype(str).unique().tolist()
    logger.info("Fetching betting odds data...")
    # Make sure fetch_and_parse_betting_odds is imported or defined
    live_odds_dict = fetch_and_parse_betting_odds(supabase_client, all_game_ids)
    if not live_odds_dict:
        logger.warning("No betting odds were fetched or parsed.")

    # --- Generate Features ---
    logger.info("Generating features...")
    features_df = feature_engine.generate_all_features(
        df=historical_df.copy(),
        historical_games_df=historical_df.copy(),
        team_stats_df=team_stats_df.copy() if not team_stats_df.empty else None,
        betting_odds_data=live_odds_dict, # <--- PASS THE FETCHED DICTIONARY HERE
        rolling_windows=rolling_windows_list,
        h2h_window=args.h2h_window
    )
        # --- MODIFICATION END ---

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

    # <<< --- START FEATURE VALUE DEBUGGING --- >>>
    logger.info("Analyzing generated feature values...")
    if not features_df.empty:
        try:
            # Select only numeric columns for analysis
            numeric_features_df = features_df.select_dtypes(include=np.number)

            if not numeric_features_df.empty:
                # 1. Calculate Descriptive Statistics
                desc_stats = numeric_features_df.describe().transpose()

                # 2. Calculate Percentage of Zeros
                zero_pct = (numeric_features_df == 0).mean() * 100
                zero_pct.rename('zero_percentage', inplace=True)

                # 3. Combine Stats
                feature_summary = pd.concat([desc_stats, zero_pct], axis=1)

                # 4. Define Output Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = f"feature_value_summary_{timestamp}.txt"
                summary_path = REPORTS_DIR / summary_filename

                # 5. Save to TXT file
                with open(summary_path, 'w') as f:
                    f.write(f"Feature Value Summary ({timestamp})\n")
                    f.write(f"Total Features Analyzed: {len(feature_summary)}\n")
                    f.write("="*50 + "\n\n")

                    # Highlight potentially problematic features
                    problem_threshold_std = 1e-6 # Very low standard deviation threshold
                    problem_threshold_zero_pct = 98.0 # High zero percentage threshold

                    problematic_features = feature_summary[
                        (feature_summary['std'] < problem_threshold_std) |
                        (feature_summary['zero_percentage'] > problem_threshold_zero_pct)
                    ]

                    if not problematic_features.empty:
                        f.write("--- POTENTIALLY PROBLEMATIC FEATURES ---\n")
                        f.write(f"(Features with std < {problem_threshold_std} OR zero % > {problem_threshold_zero_pct}%)\n\n")
                        f.write(problematic_features.to_string(float_format="%.4f"))
                        f.write("\n\n" + "="*50 + "\n\n")
                    else:
                        f.write("--- No obvious problematic features found based on thresholds. ---\n\n")


                    f.write("--- FULL FEATURE SUMMARY ---\n\n")
                    f.write(feature_summary.to_string(float_format="%.4f"))

                logger.info(f"Feature value summary saved to: {summary_path}")
                if not problematic_features.empty:
                    logger.warning(f"Found {len(problematic_features)} potentially problematic features (low variance or high zero %). See {summary_filename}")

            else:
                logger.warning("No numeric features found in features_df to analyze.")

        except Exception as e_summary:
            logger.error(f"Error generating feature value summary: {e_summary}", exc_info=True)
    else:
        logger.warning("features_df is empty, skipping feature value analysis.")
    # <<< --- END FEATURE VALUE DEBUGGING --- >>>


    logger.info("Splitting data (time-based)...")
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

    # --- Split the data into train, validation, and test sets ---
    train_df = features_df.iloc[:val_split_idx].copy()
    val_df   = features_df.iloc[val_split_idx:test_split_idx].copy()
    test_df  = features_df.iloc[test_split_idx:].copy()

    # Extract target values from each split (row order remains intact)
    y_train_home = train_df[TARGET_COLUMNS[0]].copy()
    y_train_away = train_df[TARGET_COLUMNS[1]].copy()
    y_val_home   = val_df[TARGET_COLUMNS[0]].copy()
    y_val_away   = val_df[TARGET_COLUMNS[1]].copy()
    y_test_home  = test_df[TARGET_COLUMNS[0]].copy()
    y_test_away  = test_df[TARGET_COLUMNS[1]].copy()

    # Reindex each split to force exactly the REFINED_TOP_100_FEATURES columns
    X_train = train_df.reindex(columns=REFINED_TOP_100_FEATURES, fill_value=0.0)
    X_val   = val_df.reindex(columns=REFINED_TOP_100_FEATURES, fill_value=0.0)
    X_test  = test_df.reindex(columns=REFINED_TOP_100_FEATURES, fill_value=0.0)

    logger.info(f"Data Split: Train={len(X_train)} samples, Validation={len(X_val)} samples, Test={len(X_test)} samples.")
    logger.debug(f"X_train shape: {X_train.shape}, unique cols: {len(X_train.columns.unique())}")
    logger.debug(f"X_val shape: {X_val.shape}, unique cols: {len(X_val.columns.unique())}")
    logger.debug(f"X_test shape: {X_test.shape}, unique cols: {len(X_test.columns.unique())}")

    # Use these reindexed DataFrames for feature selection and subsequent modeling.
    selected_features = select_features(feature_engine, X_train)
    logger.info(f"Selected {len(selected_features)} features for model training.")

    # If sample weights are used, append 'game_date' to the feature set.
    train_cols = selected_features[:]
    if args.use_weights and 'game_date' in X_train.columns:
        train_cols.append('game_date')
    val_cols = selected_features[:]
    if args.use_weights and 'game_date' in X_val.columns:
        val_cols.append('game_date')
    selected_features_unique = list(pd.Index(selected_features).unique())

    logger.debug(f"Unique columns in X_train: {len(list(pd.Index(train_cols).unique()))}")
    logger.debug(f"Unique columns in X_val: {len(list(pd.Index(val_cols).unique()))}")
    logger.debug(f"Unique columns in X_test: {len(selected_features_unique)}")

    # Reassign X_train, X_val, X_test using the reindexed DataFrames (do not revert to original train_df, etc.)
    X_train = X_train[list(pd.Index(train_cols).unique())].copy()
    X_val   = X_val[list(pd.Index(val_cols).unique())].copy()
    X_test  = X_test[selected_features_unique].copy()

    logger.debug(f"Final X_train shape: {X_train.shape}, unique cols: {len(X_train.columns.unique())}")
    logger.debug(f"Final X_val shape: {X_val.shape}, unique cols: {len(X_val.columns.unique())}")
    logger.debug(f"Final X_test shape: {X_test.shape}, unique cols: {len(X_test.columns.unique())}")

    # --- Define Parameter Distributions for RandomizedSearchCV ---
    XGB_PARAM_DIST = {
        # ... (keep existing definitions) ...
       'xgb__n_estimators': randint(100, 701),
       'xgb__subsample': uniform(0.6, 0.4),
       'xgb__colsample_bytree': uniform(0.6, 0.4)
    }
    RF_PARAM_DIST = {
        # ... (keep existing definitions) ...
       'rf__max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9]
    }
    RIDGE_PARAM_DIST = {
        # ... (keep existing definitions) ...
       'ridge__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    }
    param_dist_map = {"xgboost": XGB_PARAM_DIST, "random_forest": RF_PARAM_DIST, "ridge": RIDGE_PARAM_DIST}
    predictor_map = {
        "xgboost": XGBoostScorePredictor if XGBOOST_AVAILABLE else None,
        "random_forest": RandomForestScorePredictor,
        "ridge": RidgeScorePredictor
    }

    # --- Model Tuning & Training Loop (MODIFIED to collect validation preds) ---
    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    logger.info(f"Starting tuning & training for models: {models_to_run}")
    all_metrics = []
    validation_predictions_collector = {} # ADDED: Dictionary to collect validation preds

    for model_key in models_to_run:
        PredictorClass = predictor_map.get(model_key)
        if PredictorClass is None or (model_key == 'xgboost' and not XGBOOST_AVAILABLE):
            logger.warning(f"Model class for '{model_key}' unavailable. Skipping.")
            continue

        param_dist_current = param_dist_map.get(model_key) if not args.skip_tuning else None

        # --- MODIFIED CALL ---
        results_tuple = tune_and_evaluate_predictor(
            predictor_class=PredictorClass,
            X_train=X_train.copy(), y_train_home=y_train_home.copy(), y_train_away=y_train_away.copy(),
            X_val=X_val.copy(), y_val_home=y_val_home.copy(), y_val_away=y_val_away.copy(),
            X_test=X_test.copy(), y_test_home=y_test_home.copy(), y_test_away=y_test_away.copy(),
            model_name_prefix=model_key,
            feature_list=selected_features_unique, # Pass the unique list
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

        # --- Collect Results ---
        if results_tuple:
            metrics, val_preds = results_tuple # Unpack the tuple
            if metrics:
                all_metrics.append(metrics)
            if val_preds:
                # Store validation preds using model_key as the key
                validation_predictions_collector[model_key] = val_preds
                logger.info(f"Collected validation predictions for {model_key}.")
            else:
                 logger.warning(f"Did not receive validation predictions for {model_key}.")
        else:
            logger.error(f"Training/Evaluation failed entirely for {model_key}.")
    # <<< END OF MODEL TRAINING LOOP >>>

        # --- Peek & Filter Context Flags (before stacking) ---
    possible_flags = [col for col in val_df.columns 
                      if 'back_to_back' in col or 'rest' in col or 'venue' in col]
    logger.info(f"Potential context flags in val_df: {possible_flags}")

    candidate_flags = [
        'is_back_to_back',
        'is_back_to_back_home',
        'is_back_to_back_away',
        'rest_days_home',
        'rest_days_away',
        'rest_advantage',
        'venue_home_flag',
        'home_venue',
        'away_venue'
    ]
    context_flags_to_try = [f for f in candidate_flags if f in val_df.columns]
    logger.info(f"Using context flags: {context_flags_to_try}")

    # --- Enriched Meta-Model Training Logic (COMMENTED OUT) ---
    """
    logger.info("\n--- Starting Enriched Meta-Model Training (Stacking) ---")
    required_for_stacking = [m for m in ["xgboost", "random_forest", "ridge"] if m in models_to_run]
    meta_model_trained_and_saved = False

    if not all(key in validation_predictions_collector for key in required_for_stacking):
        logger.error("Missing validation predictions; cannot train meta-model.")
    elif 'val_df' not in locals() or val_df.empty:
        logger.error("Validation DataFrame unavailable; cannot extract context features.")
    else:
        logger.info(f"Preparing data for enriched meta-model training using base models: {required_for_stacking}...")
        try:
            # 1. Establish reference_index from validation preds
            first_valid_key = next((k for k in required_for_stacking if k in validation_predictions_collector), None)
            pred_series = validation_predictions_collector[first_valid_key]['pred_home']
            reference_index = pred_series.index

            # 2. Build and filter meta-features (predictions only)
            meta_df_full = pd.concat(meta_features_series_list, axis=1)
            base_meta_df = meta_df_full[[c for c in meta_df_full.columns if "_resid_" not in c]]

            # 3. Add context flags
            if not context_flags_to_try:
                context_df = pd.DataFrame(index=reference_index)
            else:
                context_df = val_df.loc[reference_index, context_flags_to_try].copy()

            # 4. Combine meta-features
            X_meta_train = pd.concat([base_meta_df, context_df], axis=1)
            logger.debug(f"Combined X_meta_train shape: {X_meta_train.shape}")

            # 5. Handle NaNs in features & targets
            if X_meta_train.isnull().any().any() or y_val_true_home.isnull().any() or y_val_true_away.isnull().any():
                feat_nan = X_meta_train.columns[X_meta_train.isnull().any()].tolist()
                if feat_nan:
                    logger.warning(f"Filling NaNs in features: {feat_nan}")
                    X_meta_train = X_meta_train.fillna(X_meta_train.mean())
                drop_idx = X_meta_train.index[
                    X_meta_train.isnull().any(axis=1) |
                    y_val_true_home.isnull() |
                    y_val_true_away.isnull()
                ]
                if not drop_idx.empty:
                    logger.warning(f"Dropping {len(drop_idx)} rows with NaN in features/targets")
                    X_meta_train = X_meta_train.drop(index=drop_idx)
                    y_val_true_home = y_val_true_home.drop(index=drop_idx)
                    y_val_true_away = y_val_true_away.drop(index=drop_idx)
                if X_meta_train.empty:
                    raise ValueError("No data left after NaN handling for meta-model.")

            # Align targets
            y_val_true_home = y_val_true_home.loc[X_meta_train.index]
            y_val_true_away = y_val_true_away.loc[X_meta_train.index]

            # 6. Train & Save enriched meta‑models
            meta_feature_names = list(X_meta_train.columns)
            logger.info(f"Enriched meta‑features ({len(meta_feature_names)}): {meta_feature_names}")
            logger.info(f"Training samples: {len(X_meta_train)}")

            meta_model_home = MetaRidge(alpha=1.0, random_state=SEED)
            meta_model_home.fit(X_meta_train, y_val_true_home)

            meta_model_away = MetaRidge(alpha=1.0, random_state=SEED)
            meta_model_away.fit(X_meta_train, y_val_true_away)

            meta_model_save_path = MAIN_MODELS_DIR / "stacking_meta_model_enriched.joblib"
            joblib.dump({
                'meta_model_home': meta_model_home,
                'meta_model_away': meta_model_away,
                'meta_feature_names': meta_feature_names,
                'base_models_used': required_for_stacking,
                'training_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }, meta_model_save_path)
            logger.info(f"Enriched stacking meta-models saved to {meta_model_save_path}")
            meta_model_trained_and_saved = True

            # (Test evaluation code for enriched meta-model would follow here)
        except Exception as e:
            logger.error(f"ERROR during enriched meta-model training: {e}", exc_info=True)

    if not meta_model_trained_and_saved:
        logger.warning("Enriched meta-model was not trained or saved due to previous errors.")
    """

    # --- Original Meta-Model Training (Stacking) ---
    logger.info("\n--- Starting Meta-Model Training (Stacking) ---")
    required_for_stacking = [m for m in ["xgboost", "random_forest", "ridge"] if m in models_to_run]
    meta_model_trained_and_saved = False

    if not all(key in validation_predictions_collector for key in required_for_stacking):
        logger.error(f"Missing validation predictions for one or more base models ({required_for_stacking}) needed for stacking. Cannot train meta-model.")
    elif not required_for_stacking:
        logger.error("No base models were successfully trained or validation predictions collected. Cannot train meta-model.")
    else:
        logger.info(f"Preparing data for meta-model training using base models: {required_for_stacking}...")
        try:
            meta_features_list = []
            y_val_true_home = None
            y_val_true_away = None
            reference_index = None

            # Use the index from the first available validation prediction set as reference
            first_valid_key = next((k for k in required_for_stacking if k in validation_predictions_collector), None)
            if first_valid_key:
                reference_index = validation_predictions_collector[first_valid_key]['pred_home'].index
            else:
                raise ValueError("Could not get validation predictions for any required base models.")

            if reference_index is None:
                raise ValueError("Could not establish a reference index from validation predictions.")

            logger.debug(f"Using reference index (length {len(reference_index)}) for meta-model alignment.")

            for model_key in required_for_stacking:
                preds_dict = validation_predictions_collector[model_key]
                pred_home_col = f"{model_key}_pred_home"
                pred_away_col = f"{model_key}_pred_away"

                if 'pred_home' not in preds_dict or 'pred_away' not in preds_dict:
                    raise KeyError(f"Missing 'pred_home' or 'pred_away' in validation predictions for {model_key}")

                df_home = preds_dict['pred_home'].rename(pred_home_col).reindex(reference_index)
                df_away = preds_dict['pred_away'].rename(pred_away_col).reindex(reference_index)
                meta_features_list.extend([df_home, df_away])

                if y_val_true_home is None:
                    if 'true_home' not in preds_dict or 'true_away' not in preds_dict:
                        raise KeyError(f"Missing 'true_home' or 'true_away' in validation predictions for {model_key}")
                    y_val_true_home = preds_dict['true_home'].reindex(reference_index)
                    y_val_true_away = preds_dict['true_away'].reindex(reference_index)

            X_meta_train = pd.concat(meta_features_list, axis=1)
            logger.debug(f"Shape of X_meta_train before NaN check: {X_meta_train.shape}")

            # Check and handle NaNs
            if X_meta_train.isnull().any().any() or y_val_true_home.isnull().any() or y_val_true_away.isnull().any():
                logger.warning("NaNs detected in meta-model training data. Handling NaNs...")
                X_meta_train = X_meta_train.fillna(X_meta_train.mean())
                rows_to_drop_idx = X_meta_train.index[X_meta_train.isnull().any(axis=1) | y_val_true_home.isnull() | y_val_true_away.isnull()]
                if not rows_to_drop_idx.empty:
                    logger.warning(f"Dropping {len(rows_to_drop_idx)} rows due to NaN values.")
                    X_meta_train = X_meta_train.drop(index=rows_to_drop_idx)
                    y_val_true_home = y_val_true_home.drop(index=rows_to_drop_idx)
                    y_val_true_away = y_val_true_away.drop(index=rows_to_drop_idx)
                if X_meta_train.empty:
                    raise ValueError("No valid data remaining after NaN handling for meta-model.")

            meta_feature_names = list(X_meta_train.columns)
            logger.info(f"Meta-model training features ({len(meta_feature_names)}): {meta_feature_names}")
            logger.info(f"Meta-model training samples: {len(X_meta_train)}")

            # Train meta-models (Ridge)
            logger.info("Training Home Score Meta-Model (Ridge)...")
            meta_model_home = MetaRidge(alpha=1.0, random_state=SEED)
            meta_model_home.fit(X_meta_train, y_val_true_home)

            logger.info("Training Away Score Meta-Model (Ridge)...")
            meta_model_away = MetaRidge(alpha=1.0, random_state=SEED)
            meta_model_away.fit(X_meta_train, y_val_true_away)

            # Save the meta-models
            meta_model_filename = "stacking_meta_model.joblib"
            meta_model_save_path = MAIN_MODELS_DIR / meta_model_filename
            meta_model_payload = {
                'meta_model_home': meta_model_home,
                'meta_model_away': meta_model_away,
                'meta_feature_names': meta_feature_names,
                'base_models_used': required_for_stacking,
                'training_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            joblib.dump(meta_model_payload, meta_model_save_path)
            logger.info(f"Stacking meta-models saved successfully to {meta_model_save_path}")
            meta_model_trained_and_saved = True

        except Exception as meta_e:
            logger.error(f"Failed to train or save meta-model: {meta_e}", exc_info=True)

    if not meta_model_trained_and_saved:
        logger.warning("Meta-model was not trained or saved due to previous errors.")


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
parser.add_argument("--scoring-metric", type=str, default='neg_mean_absolute_error', help="Scoring metric for tuning")
# Analysis Args
parser.add_argument("--run-analysis", action="store_true", help="Run optional analysis functions")
# Output Args
parser.add_argument("--visualize", action="store_true", help="Show plots interactively")
parser.add_argument("--save-plots", action="store_true", help="Save generated plots to reports directory")
# Misc Args
parser.add_argument("--allow-dummy", action="store_true", help="Allow script to run using dummy classes if imports fail")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")

# Parse arguments
cli_args = parser.parse_args()

# Run the main pipeline function
run_training_pipeline(cli_args)
