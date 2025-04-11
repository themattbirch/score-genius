# backend/nba_score_prediction/train_models.py
"""
NBA Score Prediction Model Training Pipeline

This script orchestrates the process of training and evaluating models for predicting NBA game scores.
Key steps include:
1. Loading historical game data and team statistics.
2. Generating a comprehensive set of features using NBAFeatureEngine.
3. Performing feature selection using LassoCV on pre-game features to prevent data leakage.
4. Splitting the data chronologically into training, validation, and test sets.
5. Optionally tuning hyperparameters for base models (Ridge, SVR) using RandomizedSearchCV
   on the training set.
6. Training final base models on combined Train+Val data using the best (or default) parameters.
7. Evaluating base models on the test set.
8. Generating non-leaked validation predictions for potential ensemble weighting.
9. Evaluating a manually weighted (50/50 Ridge/SVR) ensemble on the test set.
10. Saving trained models and generating performance reports/plots.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from supabase import Client, create_client

# --- Scikit-learn Imports ---
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, RidgeCV  # RidgeCV might be used later
from sklearn.linear_model import Ridge as MetaRidge # Used for potential meta-model later? Keep for now.
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# --- SciPy Imports ---
from scipy.optimize import minimize
from scipy.stats import loguniform, randint, uniform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Project-Specific Imports ---
# Use try-except for robustness, especially if run outside standard environment
try:
    from .evaluation import (
        plot_actual_vs_predicted, plot_conditional_bias,
        plot_feature_importances, plot_residuals_analysis_detailed,
        plot_temporal_bias
    )
    from .feature_engineering import NBAFeatureEngine
    from .models import (
        RidgeScorePredictor, SVRScorePredictor,
        compute_recency_weights
    )
    from . import utils  # Assumes utils.py is in the same directory
    from backend import config  # Assumes config.py is in the parent 'backend' directory
    LOCAL_MODULES_IMPORTED = True
except ImportError as e:
    logger.error("Local modules could not be imported; falling back to dummy implementations.")
    from .dummy_modules import (
        NBAFeatureEngine,
        SVRScorePredictor, RidgeScorePredictor,
        compute_recency_weights,
        plot_feature_importances,
        plot_actual_vs_predicted,
        plot_residuals_analysis_detailed,
        plot_conditional_bias,
        plot_temporal_bias,
        utils
    )
    LOCAL_MODULES_IMPORTED = False

# ==============================================================================
# Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce verbosity from http libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
# Use config paths if available, otherwise default relative to project root
MAIN_MODELS_DIR = Path(getattr(config, 'MAIN_MODELS_DIR', PROJECT_ROOT / 'models' / 'saved'))
REPORTS_DIR = Path(getattr(config, 'REPORTS_DIR', PROJECT_ROOT / 'reports'))

# Ensure directories exist
MAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
TARGET_COLUMNS = ['home_score', 'away_score']
SEED = 42
DEFAULT_CV_FOLDS = 5
# Columns required from historical data source
HISTORICAL_REQUIRED_COLS = [
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
# Columns required from team stats data source
TEAM_STATS_REQUIRED_COLS = [
    'team_name', 'season', 'wins_all_percentage', 'points_for_avg_all',
    'points_against_avg_all', 'current_form'
]

# --- Plotting and Warnings ---
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# Data Loading & Client Initialization
# ==============================================================================

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
                     supabase_client: Optional[Client] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads historical game data and team stats based on the source type."""
    logger.info(f"Attempting to load data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()

    hist_numeric_cols = [col for col in HISTORICAL_REQUIRED_COLS if col not in ['game_id', 'game_date', 'home_team', 'away_team']]
    team_numeric_cols = [col for col in TEAM_STATS_REQUIRED_COLS if col not in ['team_name', 'season', 'current_form']]

    if source_type == "supabase":
        if not supabase_client:
            logger.error("Supabase client unavailable for loading data.")
            return hist_df, team_stats_df # Return empty DataFrames

        # --- Load Historical Game Data from Supabase ---
        logger.info("Loading historical games from Supabase...")
        threshold_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        try:
            all_historical_data = []
            page_size = 1000
            start_index = 0
            has_more = True
            select_str = ", ".join(HISTORICAL_REQUIRED_COLS)
            logger.debug(f"Supabase historical select string: {select_str}")

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
                logger.info(f"Loaded {len(hist_df)} historical records into DataFrame.")

                # Simple check for required columns after loading
                missing_cols_in_df = sorted([col for col in HISTORICAL_REQUIRED_COLS if col not in hist_df.columns])
                if missing_cols_in_df:
                    logger.error(f"COLUMNS MISSING AFTER DataFrame CREATION ({len(missing_cols_in_df)} missing): {missing_cols_in_df}")
                    # Decide how to handle: return empty, fill, or proceed with warning?
                    # For now, log error and proceed, conversion below might fail.
                else:
                     logger.info("All requested historical columns appear present in created DataFrame.")

            else:
                logger.warning("No historical data retrieved from Supabase.")

        except Exception as e:
            logger.error(f"Error loading historical games from Supabase: {e}", exc_info=True)
            hist_df = pd.DataFrame() # Ensure empty on error

        # --- Load Team Stats Data from Supabase ---
        # (Assumes team stats are always needed if source is supabase for historical)
        logger.info("Loading team stats from Supabase...")
        try:
            select_str_team = ", ".join(TEAM_STATS_REQUIRED_COLS)
            response_team = supabase_client.table("nba_historical_team_stats").select(select_str_team).execute()
            if response_team.data:
                team_stats_df = pd.DataFrame(response_team.data)
                logger.info(f"Loaded {len(team_stats_df)} team stat records.")
            else:
                logger.warning("No team stats found in Supabase.")
        except Exception as e:
            logger.error(f"Error loading team stats from Supabase: {e}", exc_info=True)
            team_stats_df = pd.DataFrame() # Ensure empty on error

    elif source_type == "csv":
        logger.info("Loading data from CSV files...")
        try:
            hist_path = Path(args.historical_csv_path)
            team_path = Path(args.team_stats_csv_path)
            if hist_path.is_file():
                hist_df = pd.read_csv(hist_path)
                logger.info(f"Loaded {len(hist_df)} historical records from {hist_path}")
            else:
                logger.error(f"Historical CSV file not found at: {hist_path}")
            if team_path.is_file():
                team_stats_df = pd.read_csv(team_path)
                logger.info(f"Loaded {len(team_stats_df)} team stat records from {team_path}")
            else:
                logger.warning(f"Team stats CSV file not found at: {team_path}")
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}", exc_info=True)
            hist_df, team_stats_df = pd.DataFrame(), pd.DataFrame()

    else:
        logger.error(f"Unsupported data source type: {source_type}")
        return hist_df, team_stats_df

    # --- Common Post-Processing ---
    if not hist_df.empty:
        # Date Conversion
        if 'game_date' in hist_df.columns:
            hist_df['game_date'] = pd.to_datetime(hist_df['game_date'], errors='coerce').dt.tz_localize(None)
            hist_df = hist_df.dropna(subset=['game_date'])
            logger.debug(f"Historical DataFrame shape after date processing: {hist_df.shape}")
        else:
            logger.error("'game_date' column missing from historical data. Cannot proceed reliably.")
            return pd.DataFrame(), team_stats_df # Return empty if date is crucial

        # Numeric Conversion and NaN Handling for Historical Data
        logger.info("Converting historical numeric columns and filling NaNs with 0...")
        # Convert historical numeric columns and fill NaNs with 0...
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Historical column '{col}' missing. Adding and filling with 0.")
                hist_df[col] = 0
        # Sort by date
        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)

    if not team_stats_df.empty:
        logger.info("Converting team stats numeric columns and filling NaNs with 0...")
        for col in team_numeric_cols:
             if col in team_stats_df.columns:
                  team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
             else:
                  logger.warning(f"Team stats column '{col}' missing. Adding and filling with 0.")
                  team_stats_df[col] = 0

    logger.info(f"Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df

# ==============================================================================
# Utilities
# ==============================================================================

def visualize_recency_weights(dates, weights, title="Recency Weights Distribution", save_path=None):
    """Generates and optionally saves a plot of recency weights vs. date."""
    if dates is None or weights is None or len(dates) != len(weights):
        logger.warning("Invalid input for visualizing recency weights.")
        return
    try:
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
        plt.close() # Close plot to free memory
    except Exception as e:
         logger.error(f"Error visualizing recency weights: {e}", exc_info=True)

# ==============================================================================
# Metrics, Loss Functions, and Scorers
# ==============================================================================

def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Calculates standard regression metrics (MSE, RMSE, MAE, R2)."""
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            logger.error(f"Length mismatch in metric calculation: {len(y_true)} vs {len(y_pred)}")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true) == 0:
            logger.warning("Empty arrays passed for metric calculation.")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

        # Handle potential NaNs by calculating metrics only on valid pairs
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not valid_mask.any():
            logger.warning("No valid (non-NaN) pairs for metric calculation.")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0: # Should be caught above, but double-check
             logger.warning("No valid pairs after NaN filtering.")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

        mse = mean_squared_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        # R2 requires at least 2 samples and non-constant true values
        r2 = r2_score(y_true_valid, y_pred_valid) if len(y_true_valid) >= 2 and np.var(y_true_valid) > 1e-9 else np.nan

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

def nba_score_loss(y_true, y_pred, spread_weight=0.6, total_weight=0.4) -> float:
    """Custom loss function penalizing errors in predicted spread and total."""
    try:
        # Ensure numpy arrays
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        # Ensure correct shape (N, 2)
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)

        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("nba_score_loss requires 2 columns (home, away).")
            return np.inf

        # Handle potential NaNs
        valid_mask = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any(): return np.inf # No valid rows
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]

        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]

        diff_error = ((true_home - true_away) - (pred_home - pred_away)) ** 2
        total_error = ((true_home + true_away) - (pred_home + pred_away)) ** 2

        return np.mean(spread_weight * diff_error + total_weight * total_error)
    except Exception as e:
        logger.error(f"Error in nba_score_loss: {e}", exc_info=True)
        return np.inf

def nba_distribution_loss(y_true, y_pred) -> float:
    """Custom loss function penalizing predictions deviating from typical NBA score distributions."""
    try:
        # y_true is not actually used here, only y_pred distributions
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)
        if y_pred.shape[1] != 2:
            logger.error("nba_distribution_loss requires 2 prediction columns (home, away).")
            return np.inf

        # Handle potential NaNs
        valid_mask = ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any(): return np.inf
        y_pred = y_pred[valid_mask]

        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]

        # Typical historical averages/std devs (could be made dynamic later)
        home_mean, home_std = 114, 13.5
        away_mean, away_std = 112, 13.5
        diff_mean, diff_std = 2.5, 13.5   # home - away
        total_mean, total_std = 226, 23.0 # home + away

        pred_diff = pred_home - pred_away
        pred_total = pred_home + pred_away

        # Calculate squared Z-scores (penalty for deviation)
        z_home = ((pred_home - home_mean) / home_std) ** 2 if home_std > 1e-6 else np.zeros_like(pred_home)
        z_away = ((pred_away - away_mean) / away_std) ** 2 if away_std > 1e-6 else np.zeros_like(pred_away)
        z_diff = ((pred_diff - diff_mean) / diff_std) ** 2 if diff_std > 1e-6 else np.zeros_like(pred_diff)
        z_total = ((pred_total - total_mean) / total_std) ** 2 if total_std > 1e-6 else np.zeros_like(pred_total)

        # Weighted average of squared Z-scores
        return np.mean(0.2 * z_home + 0.2 * z_away + 0.3 * z_diff + 0.3 * z_total)
    except Exception as e:
        logger.error(f"Error in nba_distribution_loss: {e}", exc_info=True)
        return np.inf

def combined_nba_loss(y_true, y_pred, accuracy_weight=0.7, distribution_weight=0.3) -> float:
    """Combines accuracy loss (spread/total error) and distribution loss."""
    score_loss = nba_score_loss(y_true, y_pred)
    dist_loss = nba_distribution_loss(y_true, y_pred)

    if np.isinf(score_loss) or np.isinf(dist_loss):
        return np.inf

    return accuracy_weight * score_loss + distribution_weight * dist_loss

# --- Scorers for use with scikit-learn ---
# Note: greater_is_better=False because these are loss functions (lower is better)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
nba_score_scorer = make_scorer(nba_score_loss, greater_is_better=False)
nba_distribution_scorer = make_scorer(nba_distribution_loss, greater_is_better=False)
combined_scorer = make_scorer(combined_nba_loss, greater_is_better=False)

def calculate_betting_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculates basic betting-related metrics (e.g., win prediction accuracy)."""
    metrics = {'win_prediction_accuracy': np.nan} # Initialize
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)

        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("Betting metrics require 2 columns (home, away).")
            return metrics

        # Handle potential NaNs
        valid_mask = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any():
             logger.warning("No valid pairs for betting metrics calculation.")
             return metrics
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]

        if len(y_true) == 0: # Should be caught above
            logger.warning("Empty arrays after NaN filtering for betting metrics.")
            return metrics

        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]

        true_diff = true_home - true_away
        pred_diff = pred_home - pred_away

        # Correct winner prediction accuracy
        metrics['win_prediction_accuracy'] = np.mean((true_diff > 0) == (pred_diff > 0))

        # Add more metrics later (e.g., spread accuracy, total accuracy vs lines)
        # if vegas_lines DataFrame is passed and aligned.

    except Exception as e:
        logger.error(f"Error calculating betting metrics: {e}", exc_info=True)
        metrics['win_prediction_accuracy'] = np.nan # Ensure NaN on error
    return metrics

# ==============================================================================
# Ensemble Weighting
# ==============================================================================

def compute_inverse_error_weights(validation_metrics: Dict[str, float]) -> Dict[str, float]:
    """Computes weights based on the inverse of validation errors (e.g., MAE)."""
    inv_errors = {}
    logger.info(f"Computing inverse error weights based on validation metrics: {validation_metrics}")

    for model, error in validation_metrics.items():
        if pd.isna(error) or error <= 1e-9: # Use small threshold instead of 0
             logger.warning(f"Invalid or zero error ({error}) for model '{model}'. Assigning zero weight.")
             inv_errors[model] = 0.0
        else:
            inv_errors[model] = 1.0 / error # Direct inverse

    total_inv_error = sum(inv_errors.values())
    weights = {}

    if total_inv_error <= 1e-9:
        logger.warning("Total inverse error is near zero. Falling back to equal weights for models with valid errors.")
        num_valid_models = sum(1 for err in validation_metrics.values() if pd.notna(err) and err > 1e-9)
        equal_weight = 1.0 / num_valid_models if num_valid_models > 0 else 0.0
        weights = {model: equal_weight if pd.notna(error) and error > 1e-9 else 0.0
                   for model, error in validation_metrics.items()}
    else:
        # Normalize weights
        weights = {model: inv_error / total_inv_error for model, inv_error in inv_errors.items()}

    logger.info(f"Calculated normalized inverse error weights: {weights}")
    # Save weights to JSON file
    try:
        weights_path = MAIN_MODELS_DIR / "ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=4)
        logger.info(f"Ensemble weights saved to {weights_path}")
    except Exception as e:
        logger.error(f"Failed to save ensemble weights: {e}", exc_info=True)

    return weights

def optimize_ensemble_weights(
    validation_predictions: Dict[str, Dict[str, pd.Series]],
    model_keys: List[str],
    metric_to_minimize: str = 'avg_mae', # Options: 'avg_mae', 'mae_home', 'mae_away'
    l2_lambda: float = 0.1 # Regularization strength
) -> Optional[Dict[str, float]]:
    """
    Finds optimal ensemble weights by minimizing a REGULARIZED metric on validation predictions.
    Uses L2 regularization (lambda * sum(w^2)) to prevent overfitting weights.
    """
    logger.info(f"Starting REGULARIZED weight optimization for models: {model_keys} "
                f"using metric: {metric_to_minimize} with L2 lambda: {l2_lambda}")
    num_models = len(model_keys)

    if num_models < 2:
        logger.warning("At least 2 models are required to optimize weights. Returning equal weights if possible.")
        return {k: 1.0 / num_models for k in model_keys} if num_models > 0 else {}

    # --- Prepare Data ---
    try:
        # Use the index of the first model's predictions as the reference
        ref_idx = validation_predictions[model_keys[0]]['true_home'].index
        y_true_h = validation_predictions[model_keys[0]]['true_home'].loc[ref_idx].values
        y_true_a = validation_predictions[model_keys[0]]['true_away'].loc[ref_idx].values

        # Create prediction matrices [n_samples, n_models]
        preds_h = np.zeros((len(ref_idx), num_models))
        preds_a = np.zeros((len(ref_idx), num_models))
        for i, key in enumerate(model_keys):
            # Align predictions to the reference index, filling missing with NaN
            preds_h[:, i] = validation_predictions[key]['pred_home'].reindex(ref_idx).values
            preds_a[:, i] = validation_predictions[key]['pred_away'].reindex(ref_idx).values

        # Identify rows with any NaNs in true values or any model's predictions
        valid_rows_mask = (~np.isnan(y_true_h)) & (~np.isnan(y_true_a)) & \
                          (~np.isnan(preds_h).any(axis=1)) & (~np.isnan(preds_a).any(axis=1))

        if not valid_rows_mask.all():
            n_dropped = (~valid_rows_mask).sum()
            logger.warning(f"Dropping {n_dropped} rows with NaNs before weight optimization.")
            y_true_h, y_true_a = y_true_h[valid_rows_mask], y_true_a[valid_rows_mask]
            preds_h, preds_a = preds_h[valid_rows_mask, :], preds_a[valid_rows_mask, :]

        if len(y_true_h) == 0:
            logger.error("No valid samples remain after NaN check; optimization aborted.")
            return None
    except Exception as e:
        logger.error(f"Error preparing data for weight optimization: {e}", exc_info=True)
        return None

    # --- Define Regularized Objective Function ---
    def objective_function(weights: np.ndarray, y_true_h, y_true_a, preds_h, preds_a, metric, lambda_reg) -> float:
        """Calculates the blend MAE + L2 penalty given weights."""
        # Optimizer might slightly violate constraints, ensure weights sum to 1 and are non-negative
        weights = np.maximum(0, weights)
        weights_sum = np.sum(weights)
        if weights_sum > 1e-9:
            weights = weights / weights_sum
        else: # Handle case where all weights are zeroed
            return np.inf # Return infinity if weights are invalid

        blend_pred_h = np.dot(preds_h, weights)
        blend_pred_a = np.dot(preds_a, weights)

        mae_h = mean_absolute_error(y_true_h, blend_pred_h)
        mae_a = mean_absolute_error(y_true_a, blend_pred_a)

        if metric == 'avg_mae': base_metric = (mae_h + mae_a) / 2.0
        elif metric == 'mae_home': base_metric = mae_h
        elif metric == 'mae_away': base_metric = mae_a
        else: base_metric = (mae_h + mae_a) / 2.0 # Default to average MAE

        # Add L2 regularization penalty
        l2_penalty = lambda_reg * np.sum(weights**2)
        return base_metric + l2_penalty

    # --- Constraints & Bounds ---
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) # Weights must sum to 1
    bounds = tuple((0.0, 1.0) for _ in range(num_models))      # Weights between 0 and 1

    # --- Initial Guess ---
    initial_weights = np.array([1.0 / num_models] * num_models)

    # --- Run Optimization ---
    try:
        result = minimize(
            objective_function,
            initial_weights,
            args=(y_true_h, y_true_a, preds_h, preds_a, metric_to_minimize, l2_lambda), # Pass extra args
            method='SLSQP', # Suitable for constrained optimization
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-6, 'maxiter': 1000}
        )

        if result.success:
            optimized_weights_raw = result.x
            # Log the optimized *regularized* objective value
            logger.info(f"Optimization successful. Optimal regularized validation {metric_to_minimize}: {result.fun:.4f}")

            # Calculate and log the *unregularized* MAE for reference using the found weights
            unreg_mae = objective_function(optimized_weights_raw, y_true_h, y_true_a, preds_h, preds_a, metric_to_minimize, 0.0) # Set lambda=0
            logger.info(f"Unregularized validation {metric_to_minimize} with optimal weights: {unreg_mae:.4f}")

            # Clip negative weights (due to numerical precision) and renormalize
            optimized_weights = np.maximum(optimized_weights_raw, 0)
            final_sum = np.sum(optimized_weights)
            if final_sum > 1e-9:
                 optimized_weights /= final_sum
            else:
                 logger.warning("All optimized weights were zero after clipping. Returning equal weights.")
                 optimized_weights = np.array([1.0 / num_models] * num_models) # Fallback

            final_weights_dict = dict(zip(model_keys, optimized_weights))
            logger.info(f"Optimal Weights (Regularized, Lambda={l2_lambda}): {final_weights_dict}")

            # Save optimized weights
            try:
                weights_path = MAIN_MODELS_DIR / "ensemble_weights_optimized.json"
                with open(weights_path, 'w') as f:
                    json.dump(final_weights_dict, f, indent=4)
                logger.info(f"Optimized ensemble weights saved to {weights_path}")
            except Exception as e:
                logger.error(f"Failed to save optimized ensemble weights: {e}", exc_info=True)

            return final_weights_dict
        else:
            logger.error(f"Weight optimization failed: {result.message}")
            return None # Indicate failure
    except Exception as e:
        logger.error(f"Exception during weight optimization: {e}", exc_info=True)
        return None

# ==============================================================================
# Core Model Tuning and Training
# ==============================================================================

def tune_model_with_randomizedsearch(
    estimator: Pipeline,
    param_dist: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_iter: int,
    scoring: str,
    random_state: int,
    fit_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    """Helper function to run RandomizedSearchCV and return results."""
    if fit_params is None:
         fit_params = {}

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1, # Show progress during tuning
        random_state=random_state,
        error_score='raise', # Raise errors during CV
        refit=False          # We refit the best model separately on Train+Val
    )

    logger.info(f"Starting RandomizedSearchCV fitting with {n_iter} iterations...")
    rs.fit(X, y, **fit_params)
    logger.info("RandomizedSearchCV fitting complete.")

    # Extract and process results
    cv_results_df = pd.DataFrame(rs.cv_results_).sort_values(by='rank_test_score')

    # Strip potential pipeline prefixes (e.g., 'ridge__alpha' -> 'alpha') from best_params
    best_params_cleaned = {k.split('__', 1)[-1]: v for k, v in rs.best_params_.items()}
    best_score = rs.best_score_

    return best_params_cleaned, best_score, cv_results_df


def tune_and_evaluate_predictor(
    predictor_class: type, # Pass the class itself
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str,
    feature_list: List[str],
    param_dist: Optional[Dict[str, Any]], # Parameter distribution for tuning
    n_iter: int = 50, # Number of iterations for RandomizedSearch
    n_splits: int = 5, # Number of CV splits for TimeSeriesSplit
    scoring: str = 'neg_mean_absolute_error', # Scorer for tuning
    use_recency_weights: bool = False,
    weight_method: str = 'exponential',
    weight_half_life: int = 90,
    visualize: bool = True,
    save_plots: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Tunes hyperparameters, trains a final model, evaluates it, and generates
    non-leaked validation predictions for a given predictor class.

    Args:
        predictor_class: The class of the predictor (e.g., RidgeScorePredictor).
        X_train, y_train_home, ...: Training, validation, and test data splits.
        model_name_prefix: Short name for the model (e.g., "ridge").
        feature_list: List of features selected (e.g., by LASSO) to use.
        param_dist: Parameter distribution for RandomizedSearchCV. If None, tuning is skipped.
        n_iter: Number of parameter settings sampled by RandomizedSearchCV.
        n_splits: Number of splits for TimeSeriesSplit during tuning.
        scoring: Scoring metric for tuning (e.g., 'neg_mean_absolute_error').
        use_recency_weights: Whether to apply sample weights based on date.
        weight_method: Method for calculating weights ('exponential' or 'half_life').
        weight_half_life: Half-life for weighting (if method is 'half_life').
        visualize: Whether to show plots interactively.
        save_plots: Whether to save plots to the reports directory.

    Returns:
        A tuple containing:
        - metrics_dict: Dictionary of performance metrics and metadata.
        - validation_predictions_dict: Dictionary containing non-leaked predictions
          on the validation set ('pred_home', 'pred_away', 'true_home', 'true_away').
          Returns None if generation fails.
    """
    model_full_name = f"{model_name_prefix}_score_predictor"
    if predictor_class is None:
        logger.error(f"Predictor class for {model_full_name} is None.")
        return None, None

    logger.info(f"--- Starting Pipeline for {predictor_class.__name__} ({model_full_name}) ---")
    start_time_model = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {} # Will store best params found during tuning

    # --- Data Preparation ---
    if not feature_list:
        logger.error(f"Input feature list is empty for {model_full_name}.")
        return None, None
    feature_list_unique = list(pd.Index(feature_list).unique())
    if len(feature_list_unique) != len(feature_list):
        logger.warning(f"Removed {len(feature_list) - len(feature_list_unique)} duplicate features.")
    metrics['feature_count'] = len(feature_list_unique)

    # Ensure all required features are present in the data splits
    required_cols_train = feature_list_unique + (['game_date'] if use_recency_weights else [])
    required_cols_val = feature_list_unique + (['game_date'] if use_recency_weights else [])
    required_cols_test = feature_list_unique # Test set doesn't need date for prediction

    missing_train = [col for col in required_cols_train if col not in X_train.columns]
    missing_val = [col for col in required_cols_val if col not in X_val.columns]
    missing_test = [col for col in required_cols_test if col not in X_test.columns]

    if missing_train or missing_val or missing_test:
        logger.error(f"Missing required features. Train: {missing_train}, Val: {missing_val}, Test: {missing_test}")
        return None, None

    try:
        # Data for tuning (using only the training set)
        X_tune = X_train[feature_list_unique].copy()
        y_tune_home = y_train_home.loc[X_tune.index].copy()
        y_tune_away = y_train_away.loc[X_tune.index].copy() # Needed for temporary model training later

        # Data for final model training (combined Train + Val)
        X_train_val = pd.concat([X_train[required_cols_train], X_val[required_cols_val]], ignore_index=False) # Keep original index
        X_train_val_features = X_train_val[feature_list_unique]
        y_train_val_home = pd.concat([y_train_home, y_val_home], ignore_index=False).loc[X_train_val_features.index]
        y_train_val_away = pd.concat([y_train_away, y_val_away], ignore_index=False).loc[X_train_val_features.index]
        logger.info(f"Combined Train+Val size for final fit: {len(X_train_val_features)} samples.")

        # Data for testing
        X_test_final = X_test[feature_list_unique].copy()

    except KeyError as ke:
        logger.error(f"KeyError preparing data splits: {ke}. Check feature names.", exc_info=True)
        return None, None
    except Exception as prep_err:
        logger.error(f"Error preparing data splits: {prep_err}", exc_info=True)
        return None, None

    # --- Prepare Sample Weights (if enabled) ---
    tune_fit_params = {}
    final_fit_sample_weights = None
    temp_fit_sample_weights = None # For temporary model training later

    if use_recency_weights:
        logger.info("Calculating recency weights...")
        if 'game_date' in X_train.columns:
            dates_tune = X_train.loc[X_tune.index, 'game_date']
            temp_fit_sample_weights = compute_recency_weights(dates_tune, method=weight_method, half_life=weight_half_life)

            if temp_fit_sample_weights is not None and len(temp_fit_sample_weights) == len(X_tune):
                try:
                    # Determine the correct parameter name (e.g., 'ridge__sample_weight')
                    temp_predictor_instance = predictor_class()
                    temp_pipeline = temp_predictor_instance._build_pipeline({}) # Build with default params
                    model_step_name = temp_pipeline.steps[-1][0] # Get name of final estimator step
                    weight_param_name_tune = f"{model_step_name}__sample_weight"
                    tune_fit_params[weight_param_name_tune] = temp_fit_sample_weights
                    logger.info(f"Sample weights prepared for tuning using key: '{weight_param_name_tune}'")
                except Exception as e:
                    logger.warning(f"Could not determine weight parameter name for tuning: {e}. Weights might not be applied during tuning.")
                    temp_fit_sample_weights = None # Nullify if key is wrong
            else:
                logger.warning("Tuning sample weights calculation failed or length mismatched.")
                temp_fit_sample_weights = None
        else:
            logger.warning("'use_recency_weights' is True, but 'game_date' column is missing in X_train.")

        if 'game_date' in X_train_val.columns:
            dates_final = X_train_val.loc[X_train_val_features.index, 'game_date']
            final_fit_sample_weights = compute_recency_weights(dates_final, method=weight_method, half_life=weight_half_life)

            if final_fit_sample_weights is None or len(final_fit_sample_weights) != len(X_train_val_features):
                logger.warning("Final sample weights calculation failed or length mismatched. Training final model without weights.")
                final_fit_sample_weights = None
            else:
                logger.info(f"Final sample weights computed. Min: {np.min(final_fit_sample_weights):.4f}, Max: {np.max(final_fit_sample_weights):.4f}")
                if visualize or save_plots:
                     plot_save_path = REPORTS_DIR / f"{model_full_name}_final_weights.png" if save_plots else None
                     visualize_recency_weights(dates_final, final_fit_sample_weights,
                                               title=f"Final Training Weights - {model_full_name}",
                                               save_path=plot_save_path)
        else:
             logger.warning("'use_recency_weights' is True, but 'game_date' column is missing in combined Train+Val data.")

    # --- Hyperparameter Tuning (Optional) ---
    metrics['tuning_duration'] = 0.0
    metrics['best_params'] = 'default'
    metrics['best_cv_score'] = None

    if param_dist:
        logger.info(f"Starting hyperparameter tuning (RandomizedSearchCV, n_iter={n_iter}, cv={n_splits}, scoring='{scoring}')...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        tuning_start_time = time.time()
        try:
            # Create a temporary predictor instance to build the pipeline for tuning
            temp_predictor_tune = predictor_class()
            tuning_pipeline = temp_predictor_tune._build_pipeline({}) # Build with defaults
            if tuning_pipeline is None:
                raise RuntimeError(f"Could not build pipeline for {predictor_class.__name__}")

            # Tune using the helper function (targets home score by default, assuming similar params work for away)
            best_params_final, best_score, cv_results_df = tune_model_with_randomizedsearch(
                estimator=tuning_pipeline,
                param_dist=param_dist,
                X=X_tune,
                y=y_tune_home,
                cv=tscv,
                scoring=scoring,
                n_iter=n_iter,
                random_state=SEED,
                fit_params=tune_fit_params # Pass potential sample weights
            )

            metrics['tuning_duration'] = time.time() - tuning_start_time
            metrics['best_cv_score'] = best_score
            metrics['best_params'] = best_params_final

            logger.info(f"Tuning complete in {metrics['tuning_duration']:.2f}s.")
            logger.info(f"Best CV Score ({scoring}): {best_score:.4f}")
            logger.info(f"Best Params Found: {best_params_final}")
            try:
                # Log top CV results for inspection
                logger.info("--- Top CV Tuning Results ---")
                logger.info("\n" + cv_results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']].head().to_string())
            except Exception as report_e:
                logger.warning(f"Could not display CV results: {report_e}")

        except Exception as search_e:
            logger.error(f"Hyperparameter tuning failed for {model_name_prefix}: {search_e}", exc_info=True)
            metrics['best_params'] = 'default (tuning failed)' # Revert to default if tuning fails
            best_params_final = {} # Ensure empty dict if failed
    else:
        logger.info("Skipping hyperparameter tuning as requested.")
        best_params_final = {} # Use empty dict -> defaults

    # --- Train Final Model ---
    logger.info("Training final model on combined Train+Val data...")
    final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
    try:
        train_start_time = time.time()
        # Pass the combined data and the best (or default) hyperparameters
        final_predictor.train(
            X_train=X_train_val_features, # Use combined Train+Val features
            y_train_home=y_train_val_home,
            y_train_away=y_train_val_away,
            hyperparams_home=best_params_final, # Use tuned params (or {} for defaults)
            hyperparams_away=best_params_final, # Use same params for away
            sample_weights=final_fit_sample_weights # Use weights calculated on combined set
        )
        # Use training duration reported by the model if available, otherwise estimate
        metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', time.time() - train_start_time)
        logger.info(f"Final model training completed in {metrics['training_duration_final']:.2f} seconds.")

        # Save the trained model
        save_filename = f"{model_full_name}.joblib" # Use consistent naming convention
        save_path = final_predictor.save_model(filename=save_filename)
        if save_path:
            logger.info(f"Final tuned model saved to {save_path}")
            metrics['save_path'] = str(save_path)
        else:
            raise RuntimeError("Model saving returned no path.")

    except Exception as train_e:
        logger.error(f"Training or saving final model failed: {train_e}", exc_info=True)
        metrics['training_duration_final'] = None
        metrics['save_path'] = "Save failed"
        return metrics, None # Cannot proceed if final model fails

    # --- Evaluate Final Model on Test Set ---
    logger.info(f"Evaluating final {model_full_name} on test set ({len(X_test_final)} samples)...")
    try:
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or 'predicted_home_score' not in predictions_df_test.columns:
            raise ValueError("Test prediction failed or returned invalid format.")

        pred_home_test = predictions_df_test['predicted_home_score']
        pred_away_test = predictions_df_test['predicted_away_score']

        # Align true test labels
        y_test_home_aligned = y_test_home.loc[pred_home_test.index]
        y_test_away_aligned = y_test_away.loc[pred_away_test.index]

        # Calculate standard regression metrics
        test_metrics_home = calculate_regression_metrics(y_test_home_aligned, pred_home_test)
        test_metrics_away = calculate_regression_metrics(y_test_away_aligned, pred_away_test)
        metrics['test_mae_home'] = test_metrics_home.get('mae')
        metrics['test_rmse_home'] = test_metrics_home.get('rmse')
        metrics['test_r2_home'] = test_metrics_home.get('r2')
        metrics['test_mae_away'] = test_metrics_away.get('mae')
        metrics['test_rmse_away'] = test_metrics_away.get('rmse')
        metrics['test_r2_away'] = test_metrics_away.get('r2')

        # Calculate combined metrics (Total Score MAE, Point Differential MAE)
        valid_test_mask = y_test_home_aligned.notna() & y_test_away_aligned.notna() & \
                          pred_home_test.notna() & pred_away_test.notna()
        if valid_test_mask.any():
            metrics['test_mae_total'] = mean_absolute_error(
                (y_test_home_aligned + y_test_away_aligned)[valid_test_mask],
                (pred_home_test + pred_away_test)[valid_test_mask]
            )
            metrics['test_mae_diff'] = mean_absolute_error(
                (y_test_home_aligned - y_test_away_aligned)[valid_test_mask],
                (pred_home_test - pred_away_test)[valid_test_mask]
            )
            # Calculate custom losses and betting metrics
            y_true_comb_test = np.vstack((y_test_home_aligned[valid_test_mask].values, y_test_away_aligned[valid_test_mask].values)).T
            y_pred_comb_test = np.vstack((pred_home_test[valid_test_mask].values, pred_away_test[valid_test_mask].values)).T
            metrics['test_nba_score_loss'] = nba_score_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_nba_dist_loss'] = nba_distribution_loss(y_true_comb_test, y_pred_comb_test)
            metrics['test_combined_loss'] = combined_nba_loss(y_true_comb_test, y_pred_comb_test)
            metrics['betting_metrics'] = calculate_betting_metrics(y_true_comb_test, y_pred_comb_test)
        else:
            logger.warning("No valid pairs for combined test metric calculation.")
            metrics['test_mae_total'], metrics['test_mae_diff'] = np.nan, np.nan
            metrics['test_nba_score_loss'], metrics['test_nba_dist_loss'], metrics['test_combined_loss'] = np.nan, np.nan, np.nan
            metrics['betting_metrics'] = {}

        logger.info(f"FINAL Test MAE : Home={metrics.get('test_mae_home',np.nan):.3f}, Away={metrics.get('test_mae_away',np.nan):.3f}, "
                    f"Total={metrics.get('test_mae_total',np.nan):.3f}, Diff={metrics.get('test_mae_diff',np.nan):.3f}")
        logger.info(f"FINAL Test R2  : Home={metrics.get('test_r2_home',np.nan):.3f}, Away={metrics.get('test_r2_away',np.nan):.3f}")
        logger.info(f"FINAL Custom Losses: Score={metrics.get('test_nba_score_loss',np.nan):.3f}, Dist={metrics.get('test_nba_dist_loss',np.nan):.3f}, "
                    f"Combined={metrics.get('test_combined_loss',np.nan):.3f}")
        logger.info(f"FINAL Betting Metrics: {metrics.get('betting_metrics', {})}")

    except Exception as test_eval_e:
        logger.error(f"Failed evaluating test set: {test_eval_e}", exc_info=True)
        # Ensure metrics exist even if calculation fails
        for k in ['test_mae_home','test_rmse_home','test_r2_home','test_mae_away','test_rmse_away','test_r2_away',
                  'test_mae_total','test_mae_diff', 'test_nba_score_loss','test_nba_dist_loss','test_combined_loss']:
            metrics.setdefault(k, np.nan)
        metrics['betting_metrics'] = {}

    # --- Evaluate Final Model on Training Set (optional, for diagnosing overfitting) ---
    try:
        logger.info("Evaluating final model on training data (Train+Val)...")
        predictions_df_train = final_predictor.predict(X_train_val_features) # Predict on combined Train+Val features
        if predictions_df_train is None: raise ValueError("Train prediction failed.")

        pred_home_train = predictions_df_train['predicted_home_score']
        pred_away_train = predictions_df_train['predicted_away_score']

        train_metrics_home = calculate_regression_metrics(y_train_val_home, pred_home_train)
        train_metrics_away = calculate_regression_metrics(y_train_val_away, pred_away_train)
        metrics['train_mae_home'] = train_metrics_home.get('mae')
        metrics['train_r2_home'] = train_metrics_home.get('r2')
        metrics['train_mae_away'] = train_metrics_away.get('mae')
        metrics['train_r2_away'] = train_metrics_away.get('r2')

        valid_train_mask = y_train_val_home.notna() & y_train_val_away.notna() & \
                           pred_home_train.notna() & pred_away_train.notna()
        if valid_train_mask.any():
             metrics['train_mae_total'] = mean_absolute_error(
                 (y_train_val_home + y_train_val_away)[valid_train_mask],
                 (pred_home_train + pred_away_train)[valid_train_mask]
             )
             metrics['train_mae_diff'] = mean_absolute_error(
                 (y_train_val_home - y_train_val_away)[valid_train_mask],
                 (pred_home_train - pred_away_train)[valid_train_mask]
             )
        else:
             metrics['train_mae_total'], metrics['train_mae_diff'] = np.nan, np.nan

        logger.info(f"FINAL Train MAE: Home={metrics.get('train_mae_home',np.nan):.3f}, Away={metrics.get('train_mae_away',np.nan):.3f}, "
                    f"Total={metrics.get('train_mae_total',np.nan):.3f}, Diff={metrics.get('train_mae_diff',np.nan):.3f}")
        logger.info(f"FINAL Train R2 : Home={metrics.get('train_r2_home',np.nan):.3f}, Away={metrics.get('train_r2_away',np.nan):.3f}")

    except Exception as train_eval_e:
        logger.warning(f"Failed evaluating on training set: {train_eval_e}", exc_info=True)
        for k in ['train_mae_home','train_r2_home','train_mae_away','train_r2_away','train_mae_total','train_mae_diff']:
            metrics.setdefault(k, np.nan)

    metrics['samples_train_final'] = len(X_train_val_features)
    metrics['samples_test'] = len(X_test_final)

    # --- Generate Performance Plots ---
    if visualize or save_plots:
        timestamp = getattr(final_predictor, 'training_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        plot_dir = REPORTS_DIR / f"{model_full_name}_performance_{timestamp}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating performance plots in {plot_dir}")

        # Check if test predictions were successfully generated
        if 'pred_home_test' in locals() and 'y_test_home_aligned' in locals():
            try:
                # Actual vs Predicted Plots
                plot_actual_vs_predicted(
                    y_true=y_test_home_aligned, y_pred=pred_home_test,
                    title=f"{model_full_name} - Test Actual vs Pred (Home)",
                    metrics_dict={'RMSE': metrics.get('test_rmse_home'), 'R2': metrics.get('test_r2_home'), 'MAE': metrics.get('test_mae_home')},
                    save_path=plot_dir / "test_actual_vs_pred_home.png" if save_plots else None,
                    show_plot=visualize
                )
                plot_actual_vs_predicted(
                    y_true=y_test_away_aligned, y_pred=pred_away_test,
                    title=f"{model_full_name} - Test Actual vs Pred (Away)",
                    metrics_dict={'RMSE': metrics.get('test_rmse_away'), 'R2': metrics.get('test_r2_away'), 'MAE': metrics.get('test_mae_away')},
                    save_path=plot_dir / "test_actual_vs_pred_away.png" if save_plots else None,
                    show_plot=visualize
                )

                # Residual Analysis Plots
                plot_residuals_analysis_detailed(
                    y_true=y_test_home_aligned, y_pred=pred_home_test,
                    title_prefix=f"{model_full_name} (Home) - Test Set",
                    save_dir=plot_dir if save_plots else None, show_plot=visualize
                )
                plot_residuals_analysis_detailed(
                    y_true=y_test_away_aligned, y_pred=pred_away_test,
                    title_prefix=f"{model_full_name} (Away) - Test Set",
                    save_dir=plot_dir if save_plots else None, show_plot=visualize
                )

                                # Feature Importance Plots (if model provides them)
                pipeline_home = getattr(final_predictor, 'pipeline_home', None)
                pipeline_away = getattr(final_predictor, 'pipeline_away', None)
                features_in = getattr(final_predictor, 'feature_names_in_', None)

                # --- ADD CHECK FOR IMPORTANCE ATTRIBUTES ---
                can_plot_importance = False
                if pipeline_home and pipeline_away and features_in:
                    # Check if either coef_ or feature_importances_ exists on the final step
                    model_step_home = pipeline_home.steps[-1][1]
                    model_step_away = pipeline_away.steps[-1][1]
                    if hasattr(model_step_home, 'coef_') or hasattr(model_step_home, 'feature_importances_') or \
                       hasattr(model_step_away, 'coef_') or hasattr(model_step_away, 'feature_importances_'):
                        can_plot_importance = True
                # --- END ADDED CHECK ---

                if can_plot_importance: # <<< MODIFIED Condition
                    logger.info(f"Attempting to plot feature importance for {model_full_name}...")
                    plot_feature_importances(
                        models_dict={f"{model_full_name}_Home": pipeline_home, f"{model_full_name}_Away": pipeline_away},
                        feature_names=features_in, top_n=30, plot_groups=True,
                        save_dir=plot_dir / "feature_importance" if save_plots else None,
                        show_plot=visualize
                    )
                else:
                    # Log skipping clearly instead of just warning about missing pipeline/features
                    logger.info(f"Skipping feature importance plots for {model_full_name} "
                                f"(model type does not provide standard importance attributes like coef_ or feature_importances_).")


            except NameError as ne:
                 logger.error(f"Plotting function not found: {ne}. Ensure evaluation module is imported correctly.")
            except Exception as plot_e:
                logger.error(f"Failed generating plots: {plot_e}", exc_info=True)
        else:
            logger.warning("Skipping test set plots (test predictions unavailable).")


    # --- Generate NON-LEAKED Validation Predictions ---
    # Strategy: Train a temporary model using ONLY the training data (X_tune)
    # with the best hyperparameters found, then predict on the validation set (X_val).
    logger.info(f"Generating non-leaked validation predictions for {model_full_name}...")
    val_predictions_dict = None
    try:
        # Use features for validation set
        X_val_features = X_val[feature_list_unique].copy()

        # Instantiate a *new*, temporary predictor
        temp_val_predictor = predictor_class(model_dir=None, model_name=f"{model_full_name}_val_pred_temp") # Not saved

        logger.debug(f"Training temporary model on X_train ({len(X_tune)} samples) for validation prediction...")
        # Train temporary model using ONLY X_train data and best hyperparameters
        temp_val_predictor.train(
            X_train=X_tune,                # Use original training features
            y_train_home=y_tune_home,      # Original training targets
            y_train_away=y_tune_away,      # Original training targets
            hyperparams_home=best_params_final, # Best params from tuning
            hyperparams_away=best_params_final, # Best params from tuning
            sample_weights=temp_fit_sample_weights # Use weights calculated ONLY on training set
        )
        logger.debug("Temporary validation model trained.")

        # Predict on the validation set using the temporary model
        logger.debug("Predicting X_val with temporary model...")
        predictions_df_val = temp_val_predictor.predict(X_val_features) # Predict on X_val features

        if predictions_df_val is None or 'predicted_home_score' not in predictions_df_val.columns:
            raise ValueError("Prediction on validation set failed or returned invalid format.")

        # Align true validation targets (passed as input to this function)
        y_val_home_aligned = y_val_home.loc[X_val_features.index]
        y_val_away_aligned = y_val_away.loc[X_val_features.index]

        # Extract predictions and align index just in case
        pred_home_val = predictions_df_val['predicted_home_score'].reindex(X_val_features.index)
        pred_away_val = predictions_df_val['predicted_away_score'].reindex(X_val_features.index)

        if pred_home_val.isnull().any() or pred_away_val.isnull().any():
            logger.warning(f"NaNs found in non-leaked validation predictions for {model_full_name}.")

        # Store results
        val_predictions_dict = {
            'pred_home': pred_home_val, 'pred_away': pred_away_val,
            'true_home': y_val_home_aligned, 'true_away': y_val_away_aligned
        }
        logger.info(f"Successfully generated non-leaked validation predictions for {model_full_name}.")

    except Exception as val_pred_e:
        logger.error(f"FAILED generating non-leaked validation set predictions for {model_full_name}: {val_pred_e}", exc_info=True)
        val_predictions_dict = None # Ensure None on error

    # --- Finish ---
    metrics['total_duration'] = time.time() - start_time_model
    logger.info(f"--- Finished Pipeline for {model_full_name} in {metrics['total_duration']:.2f}s ---")

    return metrics, val_predictions_dict

# ==============================================================================
# Main Execution Block
# ==============================================================================

def run_training_pipeline(args: argparse.Namespace):
    """Main function to run the complete training and evaluation pipeline."""
    start_pipeline_time = time.time()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers: # Ensure handlers also reflect level
             handler.setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled.")

    logger.info("--- Starting NBA Model Tuning & Training Pipeline ---")
    logger.info(f"Run Arguments: {vars(args)}")

    # Check if local modules were imported correctly
    if not LOCAL_MODULES_IMPORTED and not args.allow_dummy:
        logger.critical("Required local modules not found and --allow-dummy not set. Exiting.")
        sys.exit(1)
    if not config and not args.allow_dummy: # Check config specifically too
        logger.critical("Config module missing or failed to import and --allow-dummy not set. Exiting.")
        sys.exit(1)

    # --- Initialize Clients ---
    supabase_client = get_supabase_client() # Will be None if not configured/installed

    # --- Load Data ---
    historical_df, team_stats_df = load_data_source(
        args.data_source, args.lookback_days, args, supabase_client
    )
    if historical_df.empty:
        logger.error("Failed to load historical data. Exiting.")
        sys.exit(1)
    if team_stats_df.empty:
        logger.warning("Team stats data is empty. Context features might be limited.")

    # --- Feature Engineering ---
    logger.info("Initializing Feature Engine...")
    try:
        # Pass client only if it was successfully created
        feature_engine = NBAFeatureEngine(supabase_client=supabase_client, debug=args.debug)
    except TypeError: # Handle case where dummy class doesn't accept args
         feature_engine = NBAFeatureEngine()
         logger.warning("Using dummy NBAFeatureEngine without arguments.")
    except Exception as fe_init_e:
         logger.error(f"Failed to initialize NBAFeatureEngine: {fe_init_e}", exc_info=True)
         sys.exit(1)


    logger.info("Generating features for ALL historical data...")
    rolling_windows_list = [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [5, 10]
    logger.info(f"Using rolling windows: {rolling_windows_list}")

    try:
        # Assuming generate_all_features is the primary method
        features_df = feature_engine.generate_all_features(
            df=historical_df.copy(), # Pass base data
            historical_games_df=historical_df.copy(), # Pass historical context
            team_stats_df=team_stats_df.copy() if not team_stats_df.empty else None,
            rolling_windows=rolling_windows_list,
            h2h_window=args.h2h_window,
        )
    except AttributeError:
         logger.error("NBAFeatureEngine instance missing 'generate_all_features' method (likely dummy class).")
         sys.exit(1)
    except Exception as fe_gen_e:
         logger.error(f"Feature generation failed: {fe_gen_e}", exc_info=True)
         sys.exit(1)

    if features_df is None or features_df.empty:
        logger.error("Feature generation returned an empty DataFrame. Exiting.")
        sys.exit(1)
    logger.info(f"Feature generation completed. Shape: {features_df.shape}")

    # --- Feature Cleaning & Pre-selection ---
    # Remove duplicates
    if features_df.columns.duplicated().any():
        logger.warning("Duplicate column names found! Removing duplicates, keeping first occurrence.")
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep='first')]
        logger.info(f"Shape after duplicate removal: {features_df.shape}")

    # Drop rows with missing targets
    initial_rows = len(features_df)
    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    rows_dropped = initial_rows - len(features_df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows due to missing target values ({TARGET_COLUMNS}).")
    if features_df.empty:
        logger.error("No rows remaining after dropping missing targets. Exiting.")
        sys.exit(1)

    # --- Feature Value Analysis (Optional but Recommended) ---
    if args.run_analysis:
        logger.info("Analyzing generated feature values...")
        try:
            numeric_features_df = features_df.select_dtypes(include=np.number)
            if not numeric_features_df.empty:
                desc_stats = numeric_features_df.describe().transpose()
                # Calculate percentage of zero values
                zero_pct = (numeric_features_df == 0).mean().mul(100).rename('zero_percentage')
                # Calculate percentage of NaN values (should be low after target drop/fills)
                nan_pct = numeric_features_df.isnull().mean().mul(100).rename('nan_percentage')

                feature_summary = pd.concat([desc_stats, zero_pct, nan_pct], axis=1)

                # Identify potentially problematic features (low variance, high zero %, high NaN %)
                problem_threshold_std = 1e-7
                problem_threshold_zero_pct = 99.0
                problem_threshold_nan_pct = 50.0
                problematic_features = feature_summary[
                    (feature_summary['std'].fillna(0) < problem_threshold_std) |
                    (feature_summary['zero_percentage'] > problem_threshold_zero_pct) |
                    (feature_summary['nan_percentage'] > problem_threshold_nan_pct)
                ]

                # Save summary to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = f"feature_value_summary_{timestamp}.txt"
                summary_path = REPORTS_DIR / summary_filename
                with open(summary_path, 'w') as f:
                    f.write(f"Feature Value Summary ({timestamp})\n")
                    f.write(f"Total Numeric Features Analyzed: {len(feature_summary)}\n")
                    f.write("=" * 80 + "\n")
                    if not problematic_features.empty:
                        f.write("\n--- POTENTIALLY PROBLEMATIC FEATURES ---\n")
                        f.write(f"(std < {problem_threshold_std:.1E} OR zero % > {problem_threshold_zero_pct}% OR NaN % > {problem_threshold_nan_pct}%)\n\n")
                        f.write(problematic_features.to_string(float_format="%.4f"))
                        f.write("\n\n" + "=" * 80 + "\n")
                    else:
                        f.write("\n--- No obvious problematic features found based on thresholds. ---\n\n")
                    f.write("\n--- FULL FEATURE SUMMARY ---\n\n")
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                        f.write(feature_summary.to_string(float_format="%.4f"))
                logger.info(f"Feature value summary saved to: {summary_path}")
                if not problematic_features.empty:
                    logger.warning(f"Found {len(problematic_features)} potentially problematic features. Review {summary_filename}")
            else:
                logger.warning("No numeric features found to generate value summary.")
        except Exception as e_summary:
            logger.error(f"Error generating feature value summary: {e_summary}", exc_info=True)

    # --- LASSO Feature Selection (on Pre-Game Features) ---
    logger.info("--- Starting LASSO Feature Selection ---")
    # Define prefixes/names considered safe (representing pre-game information)
    # This is crucial for preventing data leakage from post-game info.
    SAFE_PREFIXES = (
        'home_rolling_', 'away_rolling_', 'rolling_', # Rolling stats before the game
        'home_season_', 'away_season_', 'season_',     # Season averages/stats before the game
        'matchup_',                                    # Head-to-head stats before the game
        'rest_days_', 'is_back_to_back_',              # Rest and schedule factors
        'games_last_',                                 # Games played in recent window
        'home_form_', 'away_form_'                     # Team form/momentum indicators
    )
    SAFE_EXACT_NAMES = {
        # Manually curated list of other potentially safe features
        'rest_advantage', 'schedule_advantage', 'form_win_pct_diff',
        'streak_advantage', 'momentum_diff', 'home_current_streak',
        'home_momentum_direction', 'away_current_streak', 'away_momentum_direction',
        'game_importance_rank', 'home_win_last10', 'away_win_last10',
        'home_trend_rating', 'away_trend_rating', 'home_rank', 'away_rank'
    }

    potential_feature_cols = features_df.select_dtypes(include=np.number).columns
    # Exclude target variables and identifiers
    cols_to_exclude_lasso = set(TARGET_COLUMNS + ['game_id', 'game_date'])

    feature_candidates_for_lasso = []
    excluded_or_leaky_lasso = []

    for col in potential_feature_cols:
        if col in cols_to_exclude_lasso:
            continue
        # Check if feature has non-zero variance and isn't all NaN
        if features_df[col].isnull().all() or features_df[col].var() < 1e-8:
            excluded_or_leaky_lasso.append(f"{col} (all_nan_or_zero_variance)")
            continue

        is_safe = False
        if col.startswith(SAFE_PREFIXES) or col in SAFE_EXACT_NAMES:
            is_safe = True

        if is_safe:
            feature_candidates_for_lasso.append(col)
        else:
            excluded_or_leaky_lasso.append(f"{col} (potentially_leaky)")

    logger.info(f"Identified {len(feature_candidates_for_lasso)} candidate features for LASSO based on naming conventions.")
    if excluded_or_leaky_lasso and args.debug:
        logger.debug(f"Excluded {len(excluded_or_leaky_lasso)} features from LASSO pool: {excluded_or_leaky_lasso}")

    if not feature_candidates_for_lasso:
        logger.error("No candidate features remaining for LASSO selection. Exiting.")
        sys.exit(1)

    # Prepare data for LASSO (handle NaNs robustly before scaling)
    X_lasso = features_df[feature_candidates_for_lasso].copy()
    y_home_lasso = features_df[TARGET_COLUMNS[0]].copy()
    y_away_lasso = features_df[TARGET_COLUMNS[1]].copy()

    # Impute remaining NaNs (e.g., with median) BEFORE scaling
    # This should ideally not happen if feature generation is robust, but good safeguard.
    if X_lasso.isnull().any().any():
        logger.warning("NaNs found in LASSO feature candidates! Imputing with median.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_lasso = pd.DataFrame(imputer.fit_transform(X_lasso), columns=X_lasso.columns, index=X_lasso.index)

    logger.info("Scaling features for LASSO...")
    scaler = StandardScaler()
    X_lasso_scaled = scaler.fit_transform(X_lasso)

    logger.info("Running LassoCV for Home Score...")
    lasso_cv_home = LassoCV(cv=args.cv_splits, random_state=SEED, n_jobs=-1, max_iter=3000, tol=1e-3) # Increased max_iter/tol
    lasso_cv_home.fit(X_lasso_scaled, y_home_lasso)
    logger.info(f"LassoCV (Home) completed. Optimal alpha: {lasso_cv_home.alpha_:.6f}")

    logger.info("Running LassoCV for Away Score...")
    lasso_cv_away = LassoCV(cv=args.cv_splits, random_state=SEED, n_jobs=-1, max_iter=3000, tol=1e-3) # Increased max_iter/tol
    lasso_cv_away.fit(X_lasso_scaled, y_away_lasso)
    logger.info(f"LassoCV (Away) completed. Optimal alpha: {lasso_cv_away.alpha_:.6f}")

    # Combine selected features (union of features selected for home or away)
    selector_home = SelectFromModel(lasso_cv_home, prefit=True, threshold=1e-5) # Use small threshold
    selector_away = SelectFromModel(lasso_cv_away, prefit=True, threshold=1e-5)
    selected_mask = selector_home.get_support() | selector_away.get_support() # Union
    final_feature_list_for_models = X_lasso.columns[selected_mask].tolist()

    num_selected = len(final_feature_list_for_models)
    logger.info(f"LASSO selected {num_selected} features in total (union of home/away).")

    if num_selected == 0:
        logger.error("LASSO selected 0 features. Cannot proceed with model training. Exiting.")
        sys.exit(1)
    elif num_selected < 15: # Adjust threshold as needed
        logger.warning(f"LASSO selected a relatively small number of features ({num_selected}). Model performance might be limited.")
    if args.debug:
        logger.debug(f"Final selected features by LASSO: {final_feature_list_for_models}")

    # --- Data Splitting (using LASSO-selected features) ---
    logger.info("Splitting data (time-based) using LASSO-selected features...")
    essential_non_feature_cols = ['game_id', 'game_date'] + TARGET_COLUMNS
    cols_for_split_df = essential_non_feature_cols + final_feature_list_for_models
    # Ensure all needed columns are present before selecting
    missing_split_cols = [col for col in cols_for_split_df if col not in features_df.columns]
    if missing_split_cols:
         logger.error(f"Columns required for splitting are missing from features_df: {missing_split_cols}")
         sys.exit(1)

    features_df_selected = features_df[cols_for_split_df].sort_values('game_date').reset_index(drop=True)
    logger.info(f"Created features_df_selected for splitting with shape: {features_df_selected.shape}")

    n_total = len(features_df_selected)
    test_split_idx = int(n_total * (1 - args.test_size))
    # Ensure validation size doesn't overlap test set if test_size + val_size > 1
    val_split_frac_adjusted = min(args.val_size, 1.0 - args.test_size - 0.01) # Leave small gap if needed
    val_split_idx = int(n_total * (1 - args.test_size - val_split_frac_adjusted))

    if val_split_idx <= 0 or test_split_idx <= val_split_idx:
         logger.error(f"Invalid split indices: Train end={val_split_idx}, Val end={test_split_idx}. Check test/val sizes.")
         sys.exit(1)

    train_df = features_df_selected.iloc[:val_split_idx].copy()
    val_df   = features_df_selected.iloc[val_split_idx:test_split_idx].copy()
    test_df  = features_df_selected.iloc[test_split_idx:].copy()

    # Prepare X and y splits, include 'game_date' if weighting is enabled
    feature_cols_with_date = final_feature_list_for_models + (['game_date'] if args.use_weights else [])

    X_train = train_df[[col for col in feature_cols_with_date if col in train_df.columns]].copy()
    X_val   = val_df[[col for col in feature_cols_with_date if col in val_df.columns]].copy()
    X_test  = test_df[final_feature_list_for_models].copy() # Test set prediction doesn't need date

    y_train_home, y_train_away = train_df[TARGET_COLUMNS[0]], train_df[TARGET_COLUMNS[1]]
    y_val_home, y_val_away     = val_df[TARGET_COLUMNS[0]], val_df[TARGET_COLUMNS[1]]
    y_test_home, y_test_away   = test_df[TARGET_COLUMNS[0]], test_df[TARGET_COLUMNS[1]]

    logger.info(f"Data Split Sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    if X_train.empty or X_val.empty or X_test.empty:
         logger.error("One or more data splits are empty after splitting. Exiting.")
         sys.exit(1)

    # --- Define Base Models and Tuning Parameters ---
    # Parameter distributions for RandomizedSearchCV
    RIDGE_PARAM_DIST = {
        'ridge__alpha': loguniform(1e-3, 1e3), # Log-uniform distribution for alpha
        'ridge__fit_intercept': [True, False],
        'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'] # More solvers
    }
    SVR_PARAM_DIST = {
        'svr__kernel': ['rbf', 'linear'],            # Common effective kernels
        'svr__C': loguniform(0.1, 100),              # Regularization strength (log scale)
        'svr__gamma': ['scale', 'auto'] + list(loguniform(1e-4, 1e-1).rvs(size=5, random_state=SEED)), # Kernel coefficient (incl. some specific values)
        'svr__epsilon': uniform(0.05, 0.2)           # Margin of tolerance (uniform distribution)
    }

    # Mappings for easy access
    predictor_map = {
        "ridge": RidgeScorePredictor,
        "svr": SVRScorePredictor,
    }
    param_dist_map = {
        "ridge": RIDGE_PARAM_DIST,
        "svr": SVR_PARAM_DIST,
    }

    # --- Base Model Training Loop ---
    models_to_run = [m.strip().lower() for m in args.models.split(',') if m.strip().lower() in predictor_map]
    if not models_to_run:
        logger.error("No valid models specified to run. Check --models argument.")
        sys.exit(1)

    logger.info(f"--- Starting Tuning & Training for Base Models: {models_to_run} ---")
    all_metrics = []
    validation_predictions_collector = {} # Stores non-leaked validation predictions

    for model_key in models_to_run:
        PredictorClass = predictor_map[model_key]
        param_dist_current = param_dist_map.get(model_key) if not args.skip_tuning else None

        # Call the core training and evaluation function
        metrics_result, val_preds = tune_and_evaluate_predictor(
            predictor_class=PredictorClass,
            # Pass data splits explicitly
            X_train=X_train, y_train_home=y_train_home, y_train_away=y_train_away,
            X_val=X_val, y_val_home=y_val_home, y_val_away=y_val_away,
            X_test=X_test, y_test_home=y_test_home, y_test_away=y_test_away,
            model_name_prefix=model_key,
            feature_list=final_feature_list_for_models, # Use LASSO selected features
            param_dist=param_dist_current,
            # Pass tuning/training args
            n_iter=args.tune_iterations,
            n_splits=args.cv_splits,
            scoring=args.scoring_metric,
            use_recency_weights=args.use_weights,
            weight_method=args.weight_method,
            weight_half_life=args.weight_half_life,
            visualize=args.visualize,
            save_plots=args.save_plots,
        )

        if metrics_result:
            all_metrics.append(metrics_result)
        if val_preds:
            # Basic validation of the returned prediction dictionary
            if (isinstance(val_preds, dict) and
                'pred_home' in val_preds and 'true_home' in val_preds and
                len(val_preds['pred_home']) == len(X_val)): # Check length against X_val
                validation_predictions_collector[model_key] = val_preds
                logger.info(f"Collected non-leaked validation predictions for {model_key}.")
            else:
                logger.warning(f"Validation predictions for {model_key} appear invalid or length mismatch. Skipping collection.")
        else:
            logger.warning(f"Did not receive validation predictions dictionary for {model_key}.")

    # --- Post-Training Analysis and Ensemble Preparation ---
    successful_base_models = list(validation_predictions_collector.keys())
    if not successful_base_models:
        logger.error("No base models produced valid validation predictions. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Base models with validation predictions: {successful_base_models}")

    # --- Calculate and Save Inverse MAE Weights (Optional but useful) ---
    logger.info("\n--- Calculating Validation MAEs for Weighting ---")
    validation_mae_dict = {}
    for model_key, val_preds_dict in validation_predictions_collector.items():
        try:
            mae_home = mean_absolute_error(val_preds_dict['true_home'], val_preds_dict['pred_home'])
            mae_away = mean_absolute_error(val_preds_dict['true_away'], val_preds_dict['pred_away'])
            avg_mae = (mae_home + mae_away) / 2.0
            if pd.notna(avg_mae):
                validation_mae_dict[model_key] = avg_mae
                logger.info(f"Validation MAE for {model_key}: Home={mae_home:.4f}, Away={mae_away:.4f}, Avg={avg_mae:.4f}")
            else:
                logger.warning(f"Calculated average MAE for {model_key} is NaN. Excluding from weighting.")
                validation_mae_dict[model_key] = np.nan
        except Exception as mae_err:
            logger.error(f"Error calculating validation MAE for {model_key}: {mae_err}")
            validation_mae_dict[model_key] = np.nan

    # Compute and save weights based on inverse MAE
    if validation_mae_dict:
        compute_inverse_error_weights(validation_mae_dict)
    else:
        logger.warning("No valid validation MAEs to compute inverse error weights.")

    # --- Optimize Ensemble Weights (Optional - Requires >1 model) ---
    if len(successful_base_models) > 1:
         # Example: Optimize using average MAE and L2 regularization
         optimized_weights = optimize_ensemble_weights(
             validation_predictions=validation_predictions_collector,
             model_keys=successful_base_models,
             metric_to_minimize='avg_mae',
             l2_lambda=0.05 # Adjust regularization strength as needed
         )
         if optimized_weights:
              logger.info("Optimized weights found.")
              # Can potentially use these optimized_weights for evaluation below
         else:
              logger.warning("Optimized weight calculation failed.")
    else:
         logger.info("Skipping optimized weight calculation (only one successful base model).")


    # --- Validation Prediction Correlation Analysis ---
    if len(successful_base_models) > 1 and args.run_analysis:
        logger.info("\n--- Analyzing Base Model Validation Prediction Correlations ---")
        try:
            meta_features_list_corr = []
            ref_idx_corr = validation_predictions_collector[successful_base_models[0]]['pred_home'].index
            for key in successful_base_models:
                preds_dict = validation_predictions_collector[key]
                df_home = preds_dict['pred_home'].rename(f"{key}_pred_home").reindex(ref_idx_corr)
                df_away = preds_dict['pred_away'].rename(f"{key}_pred_away").reindex(ref_idx_corr)
                meta_features_list_corr.extend([df_home, df_away])

            X_meta_train_corr = pd.concat(meta_features_list_corr, axis=1).dropna()
            if not X_meta_train_corr.empty:
                logger.info(f"Correlation matrix input shape: {X_meta_train_corr.shape}")
                correlation_matrix = X_meta_train_corr.corr()
                logger.info("\nCorrelation Matrix (Validation Predictions):\n" + correlation_matrix.to_string())
                # Plot heatmap
                if args.save_plots or args.visualize:
                    try:
                        plt.figure(figsize=(max(6, len(X_meta_train_corr.columns)*0.8), max(5, len(X_meta_train_corr.columns)*0.6)))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                        plt.title("Correlation Matrix of Base Model Validation Predictions")
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        if args.save_plots:
                            plot_path = REPORTS_DIR / f"validation_pred_correlation_heatmap_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                            plt.savefig(plot_path)
                            logger.info(f"Correlation heatmap saved to {plot_path}")
                        if args.visualize:
                            plt.show()
                        plt.close()
                    except Exception as plot_err:
                        logger.error(f"Error generating correlation heatmap: {plot_err}")
            else:
                logger.warning("No valid data remaining for correlation calculation after dropna.")
        except Exception as corr_err:
            logger.error(f"Error during correlation analysis: {corr_err}", exc_info=True)

    # --- Evaluate Manually Weighted Ensemble (50/50 Ridge/SVR) ---
    # This section requires base models to be reloaded or kept in memory.
    # For simplicity, we reload them based on the saved paths in metrics.
    logger.info("\n--- Evaluating MANUALLY Weighted Ensemble (50/50 Ridge/SVR) on Test Set ---")
    manual_weights = {'ridge': 0.5, 'svr': 0.5}
    models_needed_manual = list(manual_weights.keys())
    base_model_test_preds_manual = {}

    if not all(m in predictor_map for m in models_needed_manual):
         logger.error("One or more models needed for manual weighting not defined in predictor_map.")
    else:
        logger.info("Loading models needed for manual ensemble...")
        models_loaded_manual = {}
        for model_key in models_needed_manual:
            loaded = False
            for m in all_metrics: # Find saved path from THIS run's metrics
                 if m.get('model_name', '').startswith(model_key) and m.get('save_path') and Path(m['save_path']).is_file():
                      try:
                           PredictorClass = predictor_map[model_key]
                           predictor = PredictorClass(model_dir=None) # Dir doesn't matter when loading specific path
                           predictor.load_model(filepath=m['save_path'])
                           models_loaded_manual[model_key] = predictor
                           logger.info(f"Loaded {model_key} from {m['save_path']} for manual ensemble.")
                           loaded = True
                           break # Found the model from this run
                      except Exception as load_err:
                           logger.error(f"Failed to load {model_key} for manual ensemble: {load_err}")
            if not loaded:
                 logger.error(f"Could not find/load saved model for {model_key} from this run.")

        if len(models_loaded_manual) == len(models_needed_manual):
            logger.info("Generating test predictions for manual ensemble models...")
            try:
                 # Ensure using correct features
                 X_test_manual_eval = X_test[final_feature_list_for_models].copy()
                 for key, predictor in models_loaded_manual.items():
                     preds = predictor.predict(X_test_manual_eval)
                     if preds is not None:
                         base_model_test_preds_manual[key] = preds.reindex(X_test_manual_eval.index)
                     else:
                         logger.warning(f"Prediction failed for {key} during manual ensemble eval.")
            except Exception as pred_err:
                 logger.error(f"Error predicting for manual ensemble: {pred_err}")
                 base_model_test_preds_manual = {} # Clear if error

            if len(base_model_test_preds_manual) == len(models_needed_manual):
                 logger.info("Calculating manually weighted predictions...")
                 manual_blend_home = pd.Series(0.0, index=X_test.index)
                 manual_blend_away = pd.Series(0.0, index=X_test.index)
                 for key, weight in manual_weights.items():
                     manual_blend_home += base_model_test_preds_manual[key]['predicted_home_score'] * weight
                     manual_blend_away += base_model_test_preds_manual[key]['predicted_away_score'] * weight

                 logger.info("Evaluating manually weighted predictions...")
                 try:
                     man_ens_metrics_home = calculate_regression_metrics(y_test_home, manual_blend_home)
                     man_ens_metrics_away = calculate_regression_metrics(y_test_away, manual_blend_away)
                     valid_man_mask = y_test_home.notna() & y_test_away.notna() & manual_blend_home.notna() & manual_blend_away.notna()

                     if valid_man_mask.any():
                          man_ens_mae_total = mean_absolute_error((y_test_home + y_test_away)[valid_man_mask], (manual_blend_home + manual_blend_away)[valid_man_mask])
                          man_ens_mae_diff = mean_absolute_error((y_test_home - y_test_away)[valid_man_mask], (manual_blend_home - manual_blend_away)[valid_man_mask])
                          y_true_comb_man = np.vstack((y_test_home[valid_man_mask].values, y_test_away[valid_man_mask].values)).T
                          y_pred_comb_man = np.vstack((manual_blend_home[valid_man_mask].values, manual_blend_away[valid_man_mask].values)).T
                          man_ens_betting = calculate_betting_metrics(y_true_comb_man, y_pred_comb_man)

                          logger.info(f"MANUAL WEIGHTED (50/50) ENSEMBLE Test MAE : Home={man_ens_metrics_home.get('mae', np.nan):.3f}, Away={man_ens_metrics_away.get('mae', np.nan):.3f}, Total={man_ens_mae_total:.3f}, Diff={man_ens_mae_diff:.3f}")
                          logger.info(f"MANUAL WEIGHTED (50/50) ENSEMBLE Test R2  : Home={man_ens_metrics_home.get('r2', np.nan):.3f}, Away={man_ens_metrics_away.get('r2', np.nan):.3f}")
                          logger.info(f"MANUAL WEIGHTED (50/50) ENSEMBLE Betting Metrics: {man_ens_betting}")
                     else: logger.warning("No valid data for manual ensemble combined metrics.")

                 except Exception as man_eval_err:
                      logger.error(f"Error evaluating manual ensemble: {man_eval_err}")
            else:
                logger.error("Failed to get predictions for all models required for manual ensemble.")
        else:
            logger.error("Failed to load all models required for manual ensemble evaluation.")


    # --- Final Summary Logging ---
    logger.info("\n" + "="*80)
    logger.info("--- Training Pipeline Summary ---")
    logger.info("="*80)
    if all_metrics:
        logger.info("Base Model Performance (Test Set):")
        try:
            metrics_df = pd.DataFrame(all_metrics)
            # Select and order columns for display
            cols_to_show = [
                'model_name', 'feature_count', 'training_duration_final',
                'test_mae_home', 'test_mae_away', 'test_r2_home', 'test_r2_away',
                'test_mae_total', 'test_mae_diff', 'test_combined_loss'
                # Add betting accuracy if available and desired
            ]
            if 'betting_metrics' in metrics_df.columns:
                 # Try to extract win prediction accuracy specifically
                 metrics_df['win_pred_acc'] = metrics_df['betting_metrics'].apply(lambda x: x.get('win_prediction_accuracy', np.nan) if isinstance(x, dict) else np.nan)
                 cols_to_show.append('win_pred_acc')

            cols_present = [col for col in cols_to_show if col in metrics_df.columns]
            metrics_df_display = metrics_df[cols_present].copy()

            # Define float formatters for better readability
            float_format = "{:.3f}".format
            formatters = {col: float_format for col in metrics_df_display.select_dtypes(include=['float']).columns}

            logger.info("\n" + metrics_df_display.to_string(index=False, formatters=formatters, na_rep='NaN'))

            # Save detailed metrics to CSV
            metrics_filename = f"training_metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            metrics_save_path = REPORTS_DIR / metrics_filename
            try:
                 # Clean up dict columns before saving if needed
                 metrics_df_save = metrics_df.copy()
                 for col in ['best_params', 'betting_metrics']: # Example columns that might be dicts
                      if col in metrics_df_save.columns:
                           metrics_df_save[col] = metrics_df_save[col].astype(str)
                 metrics_df_save.to_csv(metrics_save_path, index=False, float_format="%.4f")
                 logger.info(f"Detailed metrics saved to: {metrics_save_path}")
            except Exception as save_csv_e:
                 logger.error(f"Failed to save detailed metrics CSV: {save_csv_e}")

        except Exception as summary_e:
            logger.error(f"Error generating summary table: {summary_e}", exc_info=True)
            # Fallback: Print raw metrics list
            for m in all_metrics: logger.info(m)
    else:
        logger.warning("No model metrics were collected during the run.")

    end_time_pipeline = time.time()
    logger.info(f"\n--- NBA Model Training Pipeline Finished in {end_time_pipeline - start_pipeline_time:.2f} seconds ---")


# ==============================================================================
# Command-Line Interface
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NBA Score Prediction Model Tuning & Training Pipeline")

    # Data Source Arguments
    parser.add_argument("--data-source", type=str, default="supabase", choices=["csv", "supabase"], help="Data source type (Supabase recommended)")
    parser.add_argument("--historical-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_game_stats.csv'), help="Path to historical games CSV (if using --data-source csv)")
    parser.add_argument("--team-stats-csv-path", type=str, default=str(PROJECT_ROOT / 'data' / 'nba_historical_team_stats.csv'), help="Path to team stats CSV (if using --data-source csv)")
    parser.add_argument("--lookback-days", type=int, default=1095, help="Number of days of historical data to load (approx 3 seasons)")

    # Feature Engineering Arguments
    parser.add_argument("--rolling-windows", type=str, default="5,10,20", help="Comma-separated rolling window sizes for feature generation")
    parser.add_argument("--h2h-window", type=int, default=5, help="Number of recent games for Head-to-Head features")

    # Model Selection Arguments
    parser.add_argument("--models", type=str, default="ridge,svr", help="Comma-separated models to train (e.g., 'ridge,svr')")

    # Data Splitting Arguments
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of data for the final test set")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of data for the validation set (taken from before test set)")

    # Training Options
    parser.add_argument("--use-weights", action="store_true", help="Use recency weighting during model training")
    parser.add_argument("--weight-method", type=str, default="exponential", choices=["exponential", "half_life"], help="Method for recency weighting")
    parser.add_argument("--weight-half-life", type=int, default=90, help="Half-life in days (if using half_life weighting)")

    # Hyperparameter Tuning Arguments
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning (train with default parameters)")
    parser.add_argument("--tune-iterations", type=int, default=50, help="Number of iterations for RandomizedSearchCV")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS, help="Number of time-series cross-validation splits for tuning")
    parser.add_argument("--scoring-metric", type=str, default='neg_mean_absolute_error', help="Scoring metric used for hyperparameter tuning (e.g., 'neg_mean_absolute_error', 'r2')")

    # Output and Debugging Arguments
    parser.add_argument("--run-analysis", action="store_true", help="Run optional analysis (feature summary, correlations)")
    parser.add_argument("--visualize", action="store_true", help="Show performance plots interactively during the run")
    parser.add_argument("--save-plots", action="store_true", help="Save generated plots to the reports directory")
    parser.add_argument("--allow-dummy", action="store_true", help="Allow script to run using dummy classes if local module imports fail")
    parser.add_argument("--debug", action="store_true", help="Enable detailed DEBUG level logging")

    cli_args = parser.parse_args()

    # --- Run Pipeline ---
    run_training_pipeline(cli_args)