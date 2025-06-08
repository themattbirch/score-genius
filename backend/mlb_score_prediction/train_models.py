# backend/mlb_score_prediction/train_models.py
"""
MLB Score Prediction Model Training Pipeline

This script orchestrates the process of training and evaluating models for predicting MLB game runs.
Key steps include:
1. Loading historical game data and team statistics.
2. Generating a comprehensive set of features using a modular system for MLB.
3. Performing feature selection using Lasso or ElasticNetCV on pre-game features.
4. Splitting the data chronologically into training, validation, and test sets.
5. Optionally tuning hyperparameters for base models (Ridge, SVR, XGBoost).
6. Training final base models on combined Train+Val data.
7. Evaluating base models on the test set.
8. Generating non-leaked validation predictions for ensemble weighting.
9. Optimizing ensemble weights and evaluating the ensemble.
10. Saving trained models and generating performance reports/plots.
"""
from __future__ import annotations
from backend import config
import sys
from pathlib import Path
import logging

# --- Define Paths Consistently ---
SCRIPT_PATH_TRAIN = Path(__file__).resolve()
PROJECT_ROOT_TRAIN = SCRIPT_PATH_TRAIN.parents[2]
MODELS_DIR_MLB = PROJECT_ROOT_TRAIN / "models" / "saved" # Preferred path
MODELS_DIR_MLB.mkdir(parents=True, exist_ok=True)
REPORTS_DIR_MLB = PROJECT_ROOT_TRAIN / "reports_mlb"
REPORTS_DIR_MLB.mkdir(parents=True, exist_ok=True)

# --- Standard Library Imports ---
import argparse
import json
import re
import time
import warnings
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Type

# --- Third-Party Imports ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from supabase import Client, create_client

# Scikit-learn Imports
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, ElasticNetCV, enet_path
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

# SciPy Imports
from scipy.optimize import minimize
from scipy.stats import loguniform, randint, uniform

# ─── Project Imports (MLB Specific) ───────────────────────────────────
# Import your MLB feature-pipeline orchestrator (NEEDS TO BE CREATED/ADAPTED FOR MLB)
try:
    from backend.mlb_features.engine import run_mlb_feature_pipeline
    FEATURE_ENGINE_IMPORTED = True
except ImportError as fe_imp_err:
    logging.getLogger(__name__).warning(f"Could not import MLB feature engine: {fe_imp_err}. Feature generation will rely on dummy or simplified logic if not handled by --allow-dummy.")
    FEATURE_ENGINE_IMPORTED = False
    # Define a placeholder if needed later and --allow-dummy is used
    def run_mlb_feature_pipeline(*args, **kwargs):
        logger.error("Attempted to call 'run_mlb_feature_pipeline' but real module failed to import and no dummy was explicitly defined here.")
        raise NotImplementedError("MLB Feature pipeline not available.")


# Import MLB model classes
from backend.mlb_score_prediction.models import (
    RidgeScorePredictor as MLBRidgePredictor, # Using alias to avoid name clash if old vars exist
    SVRScorePredictor as MLBSVRPredictor,
    XGBoostScorePredictor as MLBXGBoostPredictor,
    compute_recency_weights, # This is from mlb_models.py, assuming it's generic enough
)

# Import MLB evaluation functions
# Assuming evaluation.py is adapted for MLB
from backend.mlb_score_prediction.evaluation import (
    plot_actual_vs_predicted, plot_conditional_bias,
    plot_feature_importances, plot_residuals_analysis_detailed,
    plot_temporal_bias, generate_evaluation_report # If main report func is used
)

# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"MLB train_models.py using model directory: {MODELS_DIR_MLB}")
logger.info(f"MLB train_models.py using reports directory: {REPORTS_DIR_MLB}")

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.feature_selection._univariate_selection') # For f_regression warnings
plt.style.use('fivethirtyeight')

# ==============================================================================
# Paths (using shared config)
# ==============================================================================
MAIN_MODELS_DIR = config.MAIN_MODELS_DIR # From shared config
REPORTS_DIR = MAIN_MODELS_DIR.parent / "reports_mlb" # Sport-specific reports
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants for MLB ---
TARGET_COLUMNS = ['home_score', 'away_score'] # Representing home_runs and away_runs
SEED = 42
DEFAULT_CV_FOLDS = 3 # For TimeSeriesSplit

# Required columns from mlb_historical_game_stats
# Based on user-provided schema for mlb_historical_game_stats
HISTORICAL_REQUIRED_COLS_MLB = [
    'game_id', 'game_date_time_utc', 'season', 'league_id', 'status_long', 'status_short',
    'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
    'home_score', 'away_score', 'home_hits', 'away_hits', 'home_errors', 'away_errors',
    'h_inn_1', 'h_inn_2', 'h_inn_3', 'h_inn_4', 'h_inn_5', 'h_inn_6', 'h_inn_7', 'h_inn_8', 'h_inn_9', 'h_inn_extra',
    'a_inn_1', 'a_inn_2', 'a_inn_3', 'a_inn_4', 'a_inn_5', 'a_inn_6', 'a_inn_7', 'a_inn_8', 'a_inn_9', 'a_inn_extra',
    'updated_at', 'home_starter_pitcher_handedness',
    'away_starter_pitcher_handedness'
]
# Columns required from mlb_historical_team_stats
TEAM_STATS_REQUIRED_COLS_MLB = [
    'team_id', 'team_name', 'season', 'league_id', 'league_name', 'games_played_home', 'games_played_away',
    'games_played_all', 'wins_home_total', 'wins_home_percentage', 'wins_away_total', 'wins_away_percentage',
    'wins_all_total', 'wins_all_percentage', 'losses_home_total', 'losses_home_percentage', 'losses_away_total',
    'losses_away_percentage', 'losses_all_total', 'losses_all_percentage', 'runs_for_total_home',
    'runs_for_total_away', 'runs_for_total_all', 'runs_for_avg_home', 'runs_for_avg_away', 'runs_for_avg_all',
    'runs_against_total_home', 'runs_against_total_away', 'runs_against_total_all', 'runs_against_avg_home',
    'runs_against_avg_away', 'runs_against_avg_all', 'updated_at', 'raw_api_response', 'season_runs_scored_vs_lhp',
    'season_games_vs_lhp', 'season_avg_runs_vs_lhp', 'season_runs_scored_vs_rhp', 'season_games_vs_rhp',
    'season_avg_runs_vs_rhp'
]
# Placeholders for MLB safe feature prefixes/names for Lasso/ElasticNet
MLB_SAFE_FEATURE_PREFIXES = (
    'home_rolling_', 'away_rolling_', 'rolling_',
    'home_season_', 'away_season_', 'season_',
    'matchup_',
    'park_factor_', 'weather_condition_',
    'home_pitcher_', 'away_pitcher_',
    'home_bullpen_', 'away_bullpen_',
    'home_batting_last10_', 'away_batting_last10_',
    'home_pitching_last5_', 'away_pitching_last5_',
    'rest_days_', 'travel_dist_',
    # Advanced‐splits prefixes:
    'h_team_hist_HA_',      # h_team_hist_HA_win_pct, etc.
    'a_team_hist_HA_',      # a_team_hist_HA_win_pct, etc.
    'h_team_off_avg_runs_vs_',  # covers h_team_off_avg_runs_vs_opp_hand
    'a_team_off_avg_runs_vs_',
)
MLB_SAFE_EXACT_FEATURE_NAMES = {
    'day_of_week', 'month_of_year', 'is_day_game',
    'home_streak', 'away_streak', 'series_game_num',
    'home_travel_advantage', 'away_travel_advantage',
    # The exact “vs_opp_hand” columns
    'h_team_off_avg_runs_vs_opp_hand',
    'a_team_off_avg_runs_vs_opp_hand',
    # Also, if you’ve chosen to keep the raw team_id columns, you can add them here
    # e.g. 'home_team_id', 'away_team_id'  # <– uncomment if you want Lasso to consider them
}


# ==============================================================================
# Data Loading & Client Initialization
# ==============================================================================
def get_supabase_client() -> Optional["Client"]:
    # (Identical to NBA version, uses shared config)
    supa_url = config.SUPABASE_URL
    supa_key = getattr(config, 'SUPABASE_SERVICE_KEY', None) or config.SUPABASE_ANON_KEY
    if not supa_url or not supa_key:
        logger.error("Supabase URL/Key not set in config.")
        return None
    try:
        return create_client(supa_url, supa_key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}", exc_info=True)
        return None

def load_data_source(source_type: str, lookback_days: int, args: argparse.Namespace,
                     supabase_client: Optional[Client] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads historical MLB game data and team stats based on the source type."""
    logger.info(f"Attempting to load MLB data from source: {source_type}")
    hist_df = pd.DataFrame()
    team_stats_df = pd.DataFrame()

    # Define numeric columns based on MLB required columns (excluding IDs, names, dates)
    hist_numeric_cols = [
        col for col in HISTORICAL_REQUIRED_COLS_MLB if col not in
        ['game_id', 'game_date_time_utc', 'status_long', 'status_short', 'home_team_id', 'home_team_name',
         'away_team_id', 'away_team_name', 'updated_at', 'league_id', 'season']
    ]
    team_numeric_cols = [
        col for col in TEAM_STATS_REQUIRED_COLS_MLB if col not in
        ['team_id', 'team_name', 'season', 'league_id', 'updated_at']
    ]
    
    # Use the correct date column for filtering
    date_column_historical = "game_date_time_utc" # From mlb_historical_game_stats

    if source_type == "supabase":
        if not supabase_client:
            logger.error("Supabase client unavailable for loading MLB data.")
            return hist_df, team_stats_df

        logger.info("Loading historical MLB games from Supabase...")
        threshold_date_str = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S.%fZ') # ISO format for Supabase timestampz

        try:
            all_historical_data = []
            page_size = 1000; start_index = 0; has_more = True
            select_str_hist = ", ".join(HISTORICAL_REQUIRED_COLS_MLB)
            logger.debug(f"Supabase historical MLB select string: {select_str_hist}")

            while has_more:
                response = supabase_client.table("mlb_historical_game_stats") \
                    .select(select_str_hist) \
                    .gte(date_column_historical, threshold_date_str) \
                    .order(date_column_historical) \
                    .range(start_index, start_index + page_size - 1) \
                    .execute()
                batch = response.data
                b_size = len(batch)
                all_historical_data.extend(batch)
                logger.debug(f"Retrieved {b_size} historical MLB records...")
                if b_size < page_size: has_more = False
                else: start_index += page_size
            
            if all_historical_data:
                hist_df = pd.DataFrame(all_historical_data)
                logger.info(f"Loaded {len(hist_df)} historical MLB records into DataFrame.")
                missing_cols_hist = [col for col in HISTORICAL_REQUIRED_COLS_MLB if col not in hist_df.columns]
                if missing_cols_hist: logger.error(f"MLB HISTORICAL COLUMNS MISSING: {missing_cols_hist}")
            else: logger.warning("No historical MLB data retrieved from Supabase.")
        except Exception as e:
            logger.error(f"Error loading historical MLB games from Supabase: {e}", exc_info=True)
            hist_df = pd.DataFrame()

        logger.info("Loading MLB team stats from Supabase...")
        try:
            select_str_team = ",".join(TEAM_STATS_REQUIRED_COLS_MLB)
            logger.debug(f"MLB Team stats SELECT string: {select_str_team}")
            response_team = supabase_client.table("mlb_historical_team_stats").select(select_str_team).execute()
            if response_team.data:
                team_stats_df = pd.DataFrame(response_team.data)
                logger.info(f"Loaded {len(team_stats_df)} MLB team stat records.")
                missing_cols_team = [col for col in TEAM_STATS_REQUIRED_COLS_MLB if col not in team_stats_df.columns]
                if missing_cols_team: logger.error(f"MLB TEAM STATS COLUMNS MISSING: {missing_cols_team}")
            else: logger.warning("No MLB team stats found in Supabase.")
        except Exception as e:
            logger.error(f"Error loading MLB team stats from Supabase: {e}", exc_info=True)
            team_stats_df = pd.DataFrame()

    elif source_type == "csv":
        logger.info("Loading MLB data from CSV files...")
        try:
            hist_path = Path(args.historical_csv_path) # Will point to MLB CSV via argparse
            team_path = Path(args.team_stats_csv_path) # Will point to MLB CSV
            if hist_path.is_file():
                hist_df = pd.read_csv(hist_path)
                logger.info(f"Loaded {len(hist_df)} historical MLB records from {hist_path}")
            else: logger.error(f"MLB Historical CSV file not found at: {hist_path}")
            if team_path.is_file():
                team_stats_df = pd.read_csv(team_path)
                logger.info(f"Loaded {len(team_stats_df)} MLB team stat records from {team_path}")
            else: logger.warning(f"MLB Team stats CSV file not found at: {team_path}")
        except Exception as e:
            logger.error(f"Error loading MLB data from CSV: {e}", exc_info=True)
            hist_df, team_stats_df = pd.DataFrame(), pd.DataFrame()
    else:
        logger.error(f"Unsupported data source type for MLB: {source_type}")
        return hist_df, team_stats_df

    # Common Post-Processing for MLB data
    if not hist_df.empty:
        if date_column_historical in hist_df.columns:
            hist_df['game_date'] = pd.to_datetime(hist_df[date_column_historical], errors='coerce').dt.tz_localize(None) # Standardized 'game_date'
            hist_df = hist_df.dropna(subset=['game_date'])
            logger.debug(f"MLB Historical DataFrame shape after date processing: {hist_df.shape}")
        else:
            logger.error(f"'{date_column_historical}' column missing from MLB historical data. Cannot proceed.")
            return pd.DataFrame(), team_stats_df
        
        logger.info("Converting MLB historical numeric columns, preserving NaNs...")
        hist_numeric_cols = [
            col for col in HISTORICAL_REQUIRED_COLS_MLB if col not in
            ['game_id', 'game_date_time_utc', 'status_long', 'status_short', 'home_team_id', 'home_team_name',
             'away_team_id', 'away_team_name', 'updated_at', 'league_id', 'season']
        ]
        for col in hist_numeric_cols:
            if col in hist_df.columns:
                # The .fillna(0) is removed to allow NaNs to flow to the feature modules.
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
            else:
                logger.warning(f"MLB Historical column '{col}' missing. Adding as all-NaN column.")
                hist_df[col] = np.nan # Add as NaN, not 0
        hist_df = hist_df.sort_values('game_date').reset_index(drop=True)

    if not team_stats_df.empty:
        # MODIFICATION: This loop is also corrected to PRESERVE NaNs.
        logger.info("Converting MLB team stats numeric columns, preserving NaNs...")
        team_numeric_cols = [
            col for col in TEAM_STATS_REQUIRED_COLS_MLB if col not in
            ['team_id', 'team_name', 'season', 'league_id', 'updated_at']
        ]
        for col in team_numeric_cols:
            if col in team_stats_df.columns:
                # The .fillna(0) is removed here as well.
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce')
            else:
                logger.warning(f"MLB Team stats column '{col}' missing. Adding as all-NaN column.")
                team_stats_df[col] = np.nan # Add as NaN, not 0
    
    logger.info(f"MLB Data loading complete. Historical: {len(hist_df)} rows, Team Stats: {len(team_stats_df)} rows.")
    return hist_df, team_stats_df

# ==============================================================================
# Utilities (e.g. visualize_recency_weights - can be kept as is from NBA if generic)
# ==============================================================================
def visualize_recency_weights(dates, weights, title="Recency Weights Distribution (MLB)", save_path=None):
    # (This function is largely generic, only title updated)
    if dates is None or weights is None or len(dates) != len(weights):
        logger.warning("Invalid input for visualizing recency weights.")
        return
    try:
        df_plot = pd.DataFrame({'date': pd.to_datetime(dates), 'weight': weights}).sort_values('date')
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['date'], df_plot['weight'], marker='.', linestyle='-')
        plt.title(title); plt.xlabel("Date"); plt.ylabel("Sample Weight")
        plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path); logger.info(f"Recency weights plot saved to {save_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error visualizing recency weights: {e}", exc_info=True)

# ==============================================================================
# Metrics, Loss Functions, and Scorers (MLB specific)
# ==============================================================================
# Using the generic calculate_regression_metrics from mlb_evaluation.py (imported)
# For custom losses, we adapt the NBA ones.

def mlb_runs_loss_custom(y_true, y_pred, diff_weight=0.6, total_weight=0.4) -> float:
    """Custom loss for MLB: weighted MSE of run differential and total runs."""
    try:
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2) # home, away
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)

        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("mlb_runs_loss_custom requires 2 columns (home, away).")
            return np.inf

        valid_mask = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any(): return np.inf
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
        if len(y_true) == 0: return np.inf

        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]

        diff_error = ((true_home - true_away) - (pred_home - pred_away)) ** 2
        total_error = ((true_home + true_away) - (pred_home + pred_away)) ** 2
        return np.mean(diff_weight * diff_error + total_weight * total_error)
    except Exception as e:
        logger.error(f"Error in mlb_runs_loss_custom: {e}", exc_info=True)
        return np.inf

def mlb_distribution_loss_custom(y_true, y_pred) -> float: # y_true not used, but kept for scorer compatibility
    """Custom loss: penalizes predictions deviating from typical MLB run distributions."""
    try:
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)
        if y_pred.shape[1] != 2:
            logger.error("mlb_distribution_loss_custom requires 2 prediction columns (home, away).")
            return np.inf

        valid_mask = ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any(): return np.inf
        y_pred = y_pred[valid_mask]
        if len(y_pred) == 0: return np.inf
        
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]

        # Placeholder MLB typical historical averages/std devs - TUNE THESE
        home_mean, home_std = 4.6, 3.1
        away_mean, away_std = 4.4, 3.0
        diff_mean, diff_std = 0.2, 4.5  # home - away
        total_mean, total_std = 9.0, 5.5 # home + away

        pred_diff = pred_home - pred_away
        pred_total = pred_home + pred_away

        z_home = ((pred_home - home_mean) / home_std) ** 2 if home_std > 1e-6 else np.zeros_like(pred_home)
        z_away = ((pred_away - away_mean) / away_std) ** 2 if away_std > 1e-6 else np.zeros_like(pred_away)
        z_diff = ((pred_diff - diff_mean) / diff_std) ** 2 if diff_std > 1e-6 else np.zeros_like(pred_diff)
        z_total = ((pred_total - total_mean) / total_std) ** 2 if total_std > 1e-6 else np.zeros_like(pred_total)
        
        # Equal weighting for this example, can be tuned
        return np.mean(0.25 * z_home + 0.25 * z_away + 0.25 * z_diff + 0.25 * z_total)
    except Exception as e:
        logger.error(f"Error in mlb_distribution_loss_custom: {e}", exc_info=True)
        return np.inf

def combined_mlb_loss_custom(y_true, y_pred, accuracy_weight=0.7, distribution_weight=0.3) -> float:
    """Combines accuracy loss and distribution loss for MLB."""
    score_loss = mlb_runs_loss_custom(y_true, y_pred)
    dist_loss = mlb_distribution_loss_custom(y_true, y_pred) # Pass y_true for API, though not used by this dist_loss
    if np.isinf(score_loss) or np.isinf(dist_loss): return np.inf
    return accuracy_weight * score_loss + distribution_weight * dist_loss

# Scorers for scikit-learn (ensure they handle higher_is_better correctly)
mae_scorer_mlb = make_scorer(mean_absolute_error, greater_is_better=False)
mlb_runs_scorer = make_scorer(mlb_runs_loss_custom, greater_is_better=False)
mlb_distribution_scorer = make_scorer(mlb_distribution_loss_custom, greater_is_better=False)
combined_mlb_scorer = make_scorer(combined_mlb_loss_custom, greater_is_better=False)

def calculate_betting_metrics_mlb(y_true, y_pred) -> Dict[str, float]:
    """Calculates basic betting-related metrics for MLB (e.g., win prediction accuracy)."""
    # (This function is largely generic, only changing name for clarity)
    metrics = {'win_prediction_accuracy': np.nan}
    try:
        # ... (same NaN handling and shape checks as NBA version) ...
        if isinstance(y_true, (pd.DataFrame, pd.Series)): y_true = y_true.values
        if isinstance(y_pred, (pd.DataFrame, pd.Series)): y_pred = y_pred.values
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 2)

        if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
            logger.error("Betting metrics (MLB) require 2 columns (home, away).")
            return metrics
        valid_mask = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
        if not valid_mask.any(): logger.warning("No valid pairs for MLB betting metrics."); return metrics
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
        if len(y_true) == 0: logger.warning("Empty arrays for MLB betting metrics after NaN filter."); return metrics

        true_home, true_away = y_true[:, 0], y_true[:, 1]
        pred_home, pred_away = y_pred[:, 0], y_pred[:, 1]
        true_diff = true_home - true_away
        pred_diff = pred_home - pred_away
        metrics['win_prediction_accuracy'] = np.mean((true_diff > 0) == (pred_diff > 0))
    except Exception as e:
        logger.error(f"Error calculating MLB betting metrics: {e}", exc_info=True)
        metrics['win_prediction_accuracy'] = np.nan
    return metrics

# ==============================================================================
# Ensemble Weighting (Largely generic, uses MAIN_MODELS_DIR for saving weights)
# ==============================================================================
def compute_inverse_error_weights(validation_metrics: Dict[str, float]) -> Dict[str, float]:
    # (Identical to NBA version, logic is generic)
    inv_errors = {}
    logger.info(f"Computing inverse error weights (MLB) based on validation metrics: {validation_metrics}")
    for model, error in validation_metrics.items():
        if pd.isna(error) or error <= 1e-9:
            logger.warning(f"Invalid or zero error ({error}) for model '{model}'. Assigning zero weight.")
            inv_errors[model] = 0.0
        else: inv_errors[model] = 1.0 / error
    total_inv_error = sum(inv_errors.values())
    weights = {}
    if total_inv_error <= 1e-9:
        logger.warning("Total inverse error is near zero. Falling back to equal weights for valid models.")
        num_valid = sum(1 for err in validation_metrics.values() if pd.notna(err) and err > 1e-9)
        eq_w = 1.0 / num_valid if num_valid > 0 else 0.0
        weights = {m: eq_w if pd.notna(e) and e > 1e-9 else 0.0 for m, e in validation_metrics.items()}
    else:
        weights = {model: inv_err / total_inv_error for model, inv_err in inv_errors.items()}
    logger.info(f"Calculated normalized inverse error weights (MLB): {weights}")
    try:
            weights_path = MAIN_MODELS_DIR / "mlb_ensemble_weights_inv_error.json"
            with open(weights_path, 'w') as f: json.dump(weights, f, indent=4)
            logger.info(f"MLB Ensemble weights (inv error) saved to {weights_path}")
    except Exception as e: logger.error(f"Failed to save MLB ensemble weights (inv error): {e}", exc_info=True)
    return weights

def optimize_ensemble_weights(
    validation_predictions: Dict[str, Dict[str, pd.Series]], # Keyed by simple model_key e.g. "ridge"
    model_keys: List[str], # List of simple model keys e.g. ["ridge", "svr"]
    metric_to_minimize: str = 'avg_mae',
    l2_lambda: float = 0.1
) -> Optional[Dict[str, float]]:
    """Finds optimal ensemble weights for MLB models."""
    # (Logic is generic, but rename output file and ensure model_keys map to MLB predictor names)
    logger.info(f"Starting REGULARIZED weight optimization for MLB models: {model_keys} "
                f"using metric: {metric_to_minimize} with L2 lambda: {l2_lambda}")
    num_models = len(model_keys)
    if num_models < 2:
        logger.warning("At least 2 MLB models required for weight optimization. Returning equal weights.")
        return {k: 1.0 / num_models for k in model_keys} if num_models > 0 else {}

    try: # Data Preparation
        ref_idx = validation_predictions[model_keys[0]]['true_home'].index
        y_true_h = validation_predictions[model_keys[0]]['true_home'].loc[ref_idx].values
        y_true_a = validation_predictions[model_keys[0]]['true_away'].loc[ref_idx].values
        preds_h = np.array([validation_predictions[k]['pred_home'].reindex(ref_idx).values for k in model_keys]).T
        preds_a = np.array([validation_predictions[k]['pred_away'].reindex(ref_idx).values for k in model_keys]).T
        
        valid_rows = ~np.isnan(y_true_h) & ~np.isnan(y_true_a) & \
                     ~np.isnan(preds_h).any(axis=1) & ~np.isnan(preds_a).any(axis=1)
        if not valid_rows.all():
            logger.warning(f"Dropping {(~valid_rows).sum()} rows with NaNs before MLB weight optimization.")
            y_true_h, y_true_a = y_true_h[valid_rows], y_true_a[valid_rows]
            preds_h, preds_a = preds_h[valid_rows, :], preds_a[valid_rows, :]
        if len(y_true_h) == 0:
            logger.error("No valid samples for MLB weight optimization after NaN check."); return None
    except Exception as e:
        logger.error(f"Error preparing data for MLB weight optimization: {e}", exc_info=True); return None

    def objective_function(weights, yt_h, yt_a, p_h, p_a, metric, lmbda):
        weights = np.maximum(0, weights)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9: return np.inf
        weights = weights / w_sum # Normalize
        
        blend_ph = np.dot(p_h, weights)
        blend_pa = np.dot(p_a, weights)
        mae_h = mean_absolute_error(yt_h, blend_ph)
        mae_a = mean_absolute_error(yt_a, blend_pa)

        if metric == 'avg_mae': base_metric = (mae_h + mae_a) / 2.0
        elif metric == 'mae_home': base_metric = mae_h
        else: base_metric = mae_a # Default to away or could raise error
        
        l2_penalty = lmbda * np.sum(weights**2)
        return base_metric + l2_penalty

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_models))
    initial_weights = np.array([1.0 / num_models] * num_models)

    try:
        result = minimize(
            objective_function, initial_weights,
            args=(y_true_h, y_true_a, preds_h, preds_a, metric_to_minimize, l2_lambda),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'disp': False, 'ftol': 1e-6, 'maxiter': 1000}
        )
        if result.success:
            opt_weights_raw = result.x
            unreg_mae = objective_function(opt_weights_raw, y_true_h, y_true_a, preds_h, preds_a, metric_to_minimize, 0.0)
            logger.info(f"MLB Optimization successful. Optimal regularized val {metric_to_minimize}: {result.fun:.4f} "
                        f"(Unregularized: {unreg_mae:.4f})")
            
            opt_weights = np.maximum(opt_weights_raw, 0) # Clip negatives
            final_sum = np.sum(opt_weights)
            if final_sum > 1e-9: opt_weights /= final_sum
            else: opt_weights = np.array([1.0 / num_models] * num_models) # Fallback if all zero

            # Map to full predictor names for saving (e.g., "ridge_mlb_runs_predictor")
            final_weights_map = {
                f"{model_key}_mlb_runs_predictor": weight
                for model_key, weight in zip(model_keys, opt_weights)
            }
            logger.info(f"Optimal MLB Weights (Regularized, Lambda={l2_lambda}): {final_weights_map}")
            
            try:
                weights_path = MAIN_MODELS_DIR / "mlb_ensemble_weights_optimized.json" # MLB specific filename
                with open(weights_path, 'w') as f: json.dump(final_weights_map, f, indent=4)
                logger.info(f"Optimized MLB ensemble weights saved to {weights_path}")
            except Exception as e: logger.error(f"Failed to save optimized MLB ensemble weights: {e}", exc_info=True)
            return final_weights_map
        else:
            logger.error(f"MLB Weight optimization failed: {result.message}"); return None
    except Exception as e:
        logger.error(f"Exception during MLB weight optimization: {e}", exc_info=True); return None

# ==============================================================================
# Core Model Tuning and Training (MLB context)
# ==============================================================================
# tune_model_with_randomizedsearch (generic helper, can be kept as is)
def tune_model_with_randomizedsearch(
    estimator: Pipeline, param_dist: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
    cv: TimeSeriesSplit, n_iter: int, scoring: str, random_state: int,
    fit_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    # (Identical to NBA version)
    if fit_params is None: fit_params = {}
    rs = RandomizedSearchCV(
        estimator=estimator, param_distributions=param_dist, n_iter=n_iter,
        scoring=scoring, cv=cv, n_jobs=-1, verbose=1, random_state=random_state,
        error_score='raise', refit=False
    )
    logger.info(f"Starting RandomizedSearchCV fitting with {n_iter} iterations...")
    rs.fit(X, y, **fit_params)
    logger.info("RandomizedSearchCV fitting complete.")
    cv_results_df = pd.DataFrame(rs.cv_results_).sort_values(by='rank_test_score')
    best_params_cleaned = {k.split('__', 1)[-1]: v for k, v in rs.best_params_.items()}
    best_score = rs.best_score_
    return best_params_cleaned, best_score, cv_results_df


def tune_and_evaluate_predictor(
    predictor_class: Type,
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str, # e.g., "ridge", "svr", "xgb"
    feature_list: List[str],
    param_dist: Optional[Dict[str, Any]],
    n_iter: int = 50, n_splits: int = DEFAULT_CV_FOLDS,
    scoring: str = 'neg_mean_absolute_error',
    use_recency_weights: bool = False, weight_method: str = 'exponential', weight_half_life: int = 90,
    visualize: bool = True, save_plots: bool = False,
    # Add args for the custom loss weights
    runs_loss_diff_weight: float = 0.6,
    runs_loss_total_weight: float = 0.4,
    combined_loss_accuracy_weight: float = 0.7,
    combined_loss_dist_weight: float = 0.3
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, pd.Series]]]:
    """
    Tunes hyperparameters, trains a final MLB model, evaluates it, and
    generates non-leaked validation predictions.
    Output column names for predictions will be 'predicted_home_runs', 'predicted_away_runs'.
    """
    model_full_name = f"{model_name_prefix}_mlb_runs_predictor" # MLB Naming
    if not predictor_class or not callable(getattr(predictor_class, 'train', None)):
        logger.error(f"Invalid predictor_class for {model_full_name}."); return None, None

    logger.info(f"--- Starting Pipeline for {predictor_class.__name__} ({model_full_name}) ---")
    start_time_model = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_final = {}

    # Data Preparation (generic, uses feature_list)
    # ... (Identical data prep logic as NBA version: X_tune, X_train_val_features, X_test_final) ...
    # (This section is long, assume it's ported correctly with checks and logging)
    if not feature_list: logger.error(f"Empty feature list for {model_full_name}."); return None, None
    feature_list_unique = list(pd.Index(feature_list).unique())
    metrics['feature_count'] = len(feature_list_unique); metrics['features_used'] = feature_list_unique

    # Trim DataFrames to only necessary columns to save memory
    cols_for_tune = feature_list_unique + (['game_date'] if use_recency_weights else [])
    cols_for_final_train = feature_list_unique + (['game_date'] if use_recency_weights else [])
    cols_for_test = feature_list_unique # No game_date needed for X_test prediction features

    try:
        X_tune = X_train[[c for c in cols_for_tune if c in X_train.columns]].reindex(columns=feature_list_unique).fillna(0)
        y_tune_home = y_train_home.loc[X_tune.index].copy()
        y_tune_away = y_train_away.loc[X_tune.index].copy()

        X_train_val = pd.concat([X_train, X_val])
        X_train_val_features = X_train_val[[c for c in cols_for_final_train if c in X_train_val.columns]].reindex(columns=feature_list_unique).fillna(0)
        y_train_val_home = pd.concat([y_train_home, y_val_home]).loc[X_train_val_features.index]
        y_train_val_away = pd.concat([y_train_away, y_val_away]).loc[X_train_val_features.index]
        
        X_test_final = X_test[[c for c in cols_for_test if c in X_test.columns]].reindex(columns=feature_list_unique).fillna(0)
        # y_test_home/away will be aligned later

    except Exception as prep_err:
        logger.error(f"Error preparing data splits for {model_full_name}: {prep_err}", exc_info=True)
        return None, None

    # Sample Weights (generic logic, uses compute_recency_weights)
    # ... (Identical sample weight logic, ensure 'game_date' is present if use_recency_weights) ...
    # (This section is long, assume it's ported correctly with checks and logging)
    temp_fit_sample_weights = None
    final_fit_sample_weights = None
    if use_recency_weights:
        if 'game_date' in X_train.columns: # For tuning
            dates_tune = X_train.loc[X_tune.index, 'game_date']
            temp_fit_sample_weights = compute_recency_weights(dates_tune, method=weight_method, half_life=weight_half_life)
            if temp_fit_sample_weights is None or len(temp_fit_sample_weights) != len(X_tune):
                logger.warning(f"Tuning sample weights failed for {model_full_name}. No weights applied for tuning."); temp_fit_sample_weights = None
        else: logger.warning(f"'game_date' missing in X_train for {model_full_name}. No tuning weights.");
        
        if 'game_date' in X_train_val.columns: # For final model
            dates_final = X_train_val.loc[X_train_val_features.index, 'game_date']
            final_fit_sample_weights = compute_recency_weights(dates_final, method=weight_method, half_life=weight_half_life)
            if final_fit_sample_weights is None or len(final_fit_sample_weights) != len(X_train_val_features):
                logger.warning(f"Final sample weights failed for {model_full_name}. No weights for final train."); final_fit_sample_weights = None
            elif visualize or save_plots:
                plot_sp = REPORTS_DIR / f"{model_full_name}_final_weights.png" if save_plots else None
                visualize_recency_weights(dates_final, final_fit_sample_weights, title=f"Final Training Weights - {model_full_name}", save_path=plot_sp)
        else: logger.warning(f"'game_date' missing in X_train_val for {model_full_name}. No final training weights.");


    # Hyperparameter Tuning (generic logic, specific to RidgeCV or RandomizedSearch)
    # ... (Identical tuning logic, ensure param_dist is appropriate for MLB predictors if provided) ...
    # (This section is long, assume it's ported correctly with checks and logging)
    metrics['tuning_duration'] = 0.0; metrics['best_params'] = 'default'; metrics['best_cv_score'] = None
    tuning_start_time = time.time()
    if model_name_prefix == "ridge": # RidgeCV specific path
        try:
            alphas_to_test = np.logspace(-4, 3, 50) # Adjusted alpha range for MLB runs
            tscv = TimeSeriesSplit(n_splits=n_splits)
            temp_pred_prep = predictor_class(); prep_pipeline = temp_pred_prep._build_pipeline({})
            preprocessing_steps = prep_pipeline.steps[0][1] # Assume first step is preprocessor
            ridge_cv_tuner = RidgeCV(alphas=alphas_to_test, cv=tscv, scoring=scoring)
            ridge_cv_pipeline = Pipeline([('preprocessing', preprocessing_steps), ('ridge_cv', ridge_cv_tuner)])
            fit_p = {'sample_weight': temp_fit_sample_weights} if use_recency_weights and temp_fit_sample_weights is not None else {}
            ridge_cv_pipeline.fit(X_tune, y_tune_home, **fit_p) # Tune on home runs e.g.
            best_params_final = {'alpha': ridge_cv_pipeline.named_steps['ridge_cv'].alpha_}
            metrics['best_cv_score'] = getattr(ridge_cv_pipeline.named_steps['ridge_cv'], 'best_score_', 'N/A')
            metrics['best_params'] = best_params_final
            logger.info(f"RidgeCV complete for {model_full_name}. Best alpha: {best_params_final['alpha']:.5f}")
        except Exception as rcv_e: logger.error(f"RidgeCV failed for {model_full_name}: {rcv_e}", exc_info=True); best_params_final = {}; metrics['best_params'] = 'default (RidgeCV failed)'
    elif param_dist: # RandomizedSearch
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            temp_pred_tune = predictor_class(); tuning_pipeline = temp_pred_tune._build_pipeline({})
            fit_p_rand = {}
            if use_recency_weights and temp_fit_sample_weights is not None:
                est_name = tuning_pipeline.steps[-1][0]; fit_p_rand[f"{est_name}__sample_weight"] = temp_fit_sample_weights
            
            # Tune on home runs for simplicity, or average of home/away MAE if scorer supports it
            best_p, best_s, cv_df = tune_model_with_randomizedsearch(
                tuning_pipeline, param_dist, X_tune, y_tune_home, tscv, n_iter, scoring, SEED, fit_p_rand)
            if best_p and not np.isnan(best_s):
                best_params_final = best_p; metrics['best_cv_score'] = best_s; metrics['best_params'] = best_p
                logger.info(f"Tuning complete for {model_full_name}. Best CV ({scoring}): {best_s:.4f}. Params: {best_p}")
            else: logger.error(f"Tuning invalid for {model_full_name}. Defaults used."); best_params_final = {}; metrics['best_params'] = 'default (tuning invalid)'
        except Exception as search_e: logger.error(f"RandSearch failed for {model_full_name}: {search_e}", exc_info=True); best_params_final = {}; metrics['best_params'] = 'default (tuning failed)'
    else: logger.info(f"Skipping tuning for {model_full_name}. Defaults used."); best_params_final = {}
    metrics['tuning_duration'] = time.time() - tuning_start_time
    logger.info(f"Tuning for {model_full_name} finished in {metrics['tuning_duration']:.2f}s.")


    # Train Final Model
    logger.info(f"Training final {model_full_name} on Train+Val data...")
    try:
        final_predictor = predictor_class(model_dir=str(MAIN_MODELS_DIR), model_name=model_full_name)
        train_start_time = time.time()
        final_predictor.train(
            X_train=X_train_val_features, y_train_home=y_train_val_home, y_train_away=y_train_val_away,
            hyperparams_home=best_params_final, hyperparams_away=best_params_final, # Use same params for home/away or tune separately
            sample_weights=final_fit_sample_weights
        )
        metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', time.time() - train_start_time)
        save_path = final_predictor.save_model() # Uses model_name for filename
        metrics['save_path'] = str(save_path) if save_path else "Save failed"
        logger.info(f"Final {model_full_name} trained in {metrics['training_duration_final']:.2f}s, saved to {save_path}")
    except Exception as final_train_e:
        logger.error(f"Training/saving final {model_full_name} failed: {final_train_e}", exc_info=True)
        return metrics, None # Return collected metrics so far, but no val_preds

    # Evaluate Final Model on Test Set
    logger.info(f"Evaluating final {model_full_name} on test set ({len(X_test_final)} samples)...")
    pred_home_test, pred_away_test = None, None
    try:
        predictions_df_test = final_predictor.predict(X_test_final)
        if predictions_df_test is None or \
           'predicted_home_runs' not in predictions_df_test.columns or \
           'predicted_away_runs' not in predictions_df_test.columns: # MLB specific column names
            raise ValueError("Test prediction failed or returned invalid format (expected DataFrame with home/away runs).")

        predictions_df_test = predictions_df_test.reindex(X_test_final.index) # Align
        pred_home_test = predictions_df_test['predicted_home_runs']
        pred_away_test = predictions_df_test['predicted_away_runs']
        y_test_home_aligned = y_test_home.loc[X_test_final.index]
        y_test_away_aligned = y_test_away.loc[X_test_final.index]

        # Use calculate_regression_metrics from mlb_evaluation.py (imported earlier)
        from backend.mlb_score_prediction.evaluation import calculate_regression_metrics as eval_calc_reg_metrics

        test_metrics_home = eval_calc_reg_metrics(y_test_home_aligned, pred_home_test)
        test_metrics_away = eval_calc_reg_metrics(y_test_away_aligned, pred_away_test)
        metrics.update({f'test_{k}_home': v for k,v in test_metrics_home.items()})
        metrics.update({f'test_{k}_away': v for k,v in test_metrics_away.items()})
        
        # Baseline Dummy Regressor
        dummy_h = DummyRegressor(strategy='mean').fit(X_train_val_features, y_train_val_home)
        metrics['baseline_mae_home'] = mean_absolute_error(y_test_home_aligned, dummy_h.predict(X_test_final))
        dummy_a = DummyRegressor(strategy='mean').fit(X_train_val_features, y_train_val_away)
        metrics['baseline_mae_away'] = mean_absolute_error(y_test_away_aligned, dummy_a.predict(X_test_final))
        logger.info(f"Baseline MAE – Home: {metrics['baseline_mae_home']:.3f}, Away: {metrics['baseline_mae_away']:.3f}")

        # Combined metrics (Total Runs MAE, Run Differential MAE) & Custom Losses
        valid_mask_test = y_test_home_aligned.notna() & y_test_away_aligned.notna() & pred_home_test.notna() & pred_away_test.notna()
        if valid_mask_test.any():
            yt_h_v, yt_a_v = y_test_home_aligned[valid_mask_test], y_test_away_aligned[valid_mask_test]
            yp_h_v, yp_a_v = pred_home_test[valid_mask_test], pred_away_test[valid_mask_test]
            metrics['test_mae_total'] = mean_absolute_error(yt_h_v + yt_a_v, yp_h_v + yp_a_v)
            metrics['test_mae_diff'] = mean_absolute_error(yt_h_v - yt_a_v, yp_h_v - yp_a_v)
            
            y_true_c = np.vstack((yt_h_v.values, yt_a_v.values)).T
            y_pred_c = np.vstack((yp_h_v.values, yp_a_v.values)).T
            metrics['test_mlb_runs_loss'] = mlb_runs_loss_custom(y_true_c, y_pred_c, diff_weight=runs_loss_diff_weight, total_weight=runs_loss_total_weight)
            metrics['test_mlb_dist_loss'] = mlb_distribution_loss_custom(y_true_c, y_pred_c)
            metrics['test_combined_mlb_loss'] = combined_mlb_loss_custom(y_true_c, y_pred_c, accuracy_weight=combined_loss_accuracy_weight, distribution_weight=combined_loss_dist_weight)
            metrics['betting_metrics_mlb'] = calculate_betting_metrics_mlb(y_true_c, y_pred_c)
        # ... (Log final test metrics)
        log_m = {k: f"{v:.3f}" if isinstance(v, (float, np.floating)) and pd.notna(v) else v for k,v in metrics.items() if k.startswith('test_')}
        logger.info(f"FINAL Test Metrics for {model_full_name}: {log_m}")
        logger.info(f"FINAL Test Betting Metrics for {model_full_name}: {metrics.get('betting_metrics_mlb', {})}")

    except Exception as test_eval_e: logger.error(f"Failed evaluating test set for {model_full_name}: {test_eval_e}", exc_info=True) # Populate NaNs

    # Evaluate Final Model on Training Set (Train+Val)
    # ... (Similar to test set evaluation, using X_train_val_features, y_train_val_home/away) ...
    # (This section is long, assume it's ported correctly with MLB specific column names and metrics)
    try:
        logger.info(f"Evaluating final {model_full_name} on training data (Train+Val)...")
        preds_df_train = final_predictor.predict(X_train_val_features)
        if preds_df_train is None or 'predicted_home_runs' not in preds_df_train.columns: raise ValueError("Train pred invalid.")
        preds_df_train = preds_df_train.reindex(X_train_val_features.index)
        pred_h_train, pred_a_train = preds_df_train['predicted_home_runs'], preds_df_train['predicted_away_runs']
        
        train_mets_h = eval_calc_reg_metrics(y_train_val_home, pred_h_train)
        train_mets_a = eval_calc_reg_metrics(y_train_val_away, pred_a_train)
        metrics.update({f'train_{k}_home': v for k,v in train_mets_h.items()})
        metrics.update({f'train_{k}_away': v for k,v in train_mets_a.items()})
        # ... (add total/diff MAE for train if needed)
    except Exception as train_eval_e: logger.warning(f"Failed evaluating {model_full_name} on training set: {train_eval_e}", exc_info=True)


    metrics['samples_train_final'] = len(X_train_val_features)
    metrics['samples_test'] = len(X_test_final)
    metrics['samples_tune'] = len(X_tune) # X_tune is X_train
    metrics['samples_val'] = len(X_val)

    # Performance Plots (using MLB evaluation functions and test predictions)
    if (visualize or save_plots) and pred_home_test is not None and y_test_home_aligned is not None:
        ts = getattr(final_predictor, 'training_timestamp', datetime.now().strftime("%Y%m%d%H%M%S"))
        plot_base = f"{model_full_name}_MLB_performance_{ts}"
        plot_d = REPORTS_DIR / plot_base
        if save_plots: plot_d.mkdir(parents=True, exist_ok=True); logger.info(f"Saving plots to {plot_d}")
        
        plot_actual_vs_predicted(y_test_home_aligned, pred_home_test, f"{model_full_name} - Test Actual vs Pred (Home Runs)",
                                 {'RMSE': metrics.get('test_rmse_home'), 'R2': metrics.get('test_r2_home')},
                                 save_path=plot_d/"test_actual_vs_pred_home.png" if save_plots else None, show_plot=visualize)
        # ... (plot for away runs) ...
        plot_residuals_analysis_detailed(y_test_home_aligned, pred_home_test, f"{model_full_name} (Home Runs) - Test Set",
                                         save_dir=plot_d if save_plots else None, show_plot=visualize)
        # ... (plot for away runs) ...
        # Feature Importance (using MLB evaluation functions)
        if hasattr(final_predictor, 'pipeline_home') and final_predictor.pipeline_home:
            plot_feature_importances({'Home_Model': final_predictor.pipeline_home}, feature_list_unique, X_test=X_test_final, # Pass X_test for SHAP
                                     save_dir=plot_d/"feature_importance" if save_plots else None, show_plot=visualize)


    # Non-Leaked Validation Predictions (generic logic)
    # ... (Identical logic, ensure MLB predictor class & column names are used) ...
    # (This section is long, assume it's ported correctly with MLB specific column names)
    val_predictions_dict = None
    try:
        logger.info(f"Generating non-leaked validation predictions for {model_full_name}...")
        # X_val_features was X_val[feature_list_unique].copy() earlier, but needs to be reindexed and filled like X_tune
        X_val_for_pred = X_val[[c for c in cols_for_tune if c in X_val.columns]].reindex(columns=feature_list_unique).fillna(0)

        temp_val_predictor = predictor_class(model_dir=None, model_name=f"{model_full_name}_val_temp")
        temp_val_predictor.train(
            X_train=X_tune, y_train_home=y_tune_home, y_train_away=y_tune_away,
            hyperparams_home=best_params_final, hyperparams_away=best_params_final,
            sample_weights=temp_fit_sample_weights
        )
        preds_df_val = temp_val_predictor.predict(X_val_for_pred)
        if preds_df_val is None or 'predicted_home_runs' not in preds_df_val.columns: # MLB columns
            raise ValueError("Validation prediction failed or invalid format.")
        
        preds_df_val = preds_df_val.reindex(X_val_for_pred.index) # Align
        val_predictions_dict = {
            'pred_home': preds_df_val['predicted_home_runs'],
            'pred_away': preds_df_val['predicted_away_runs'],
            'true_home': y_val_home.loc[X_val_for_pred.index], # Align true y_val
            'true_away': y_val_away.loc[X_val_for_pred.index]
        }
        logger.info(f"Successfully generated non-leaked validation predictions for {model_full_name}.")
    except Exception as val_pred_e:
        logger.error(f"FAILED generating non-leaked validation preds for {model_full_name}: {val_pred_e}", exc_info=True)


    metrics['total_duration'] = time.time() - start_time_model
    logger.info(f"--- Finished Pipeline for {model_full_name} in {metrics['total_duration']:.2f}s ---")
    return metrics, val_predictions_dict

# ==============================================================================
# Main Execution Block (MLB context)
# ==============================================================================
def run_training_pipeline(args: argparse.Namespace):
    """Main function to run the complete MLB training and evaluation pipeline."""
    start_pipeline_time = time.time()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # (Set up handlers if needed)
    logger.info("--- Starting MLB Model Tuning & Training Pipeline ---")
    logger.info(f"Run Arguments: {vars(args)}")

    # Check for feature engine
    if not FEATURE_ENGINE_IMPORTED and not args.allow_dummy:
        logger.critical(
            "MLB Feature Engine (run_mlb_feature_pipeline) not found or failed to import, "
            "and --allow-dummy not set. Exiting."
        )
        sys.exit(1)

    # Data Loading
    supabase_client = get_supabase_client()
    historical_df, team_stats_df = load_data_source(
        args.data_source,
        args.lookback_days,
        args,
        supabase_client
    )
    if historical_df.empty:
        logger.error("Failed to load MLB historical data. Exiting.")
        sys.exit(1)

    # Feature Engineering (call MLB version)
    logger.info("Generating MLB features using modular pipeline...")
    rolling_windows_list = (
        [int(w) for w in args.rolling_windows.split(',')]
        if args.rolling_windows
        else [10, 20, 50]
    )
    logger.info(f"Using rolling windows for MLB: {rolling_windows_list}")
    logger.info(f"Using H2H window for MLB: {args.h2h_window}")

    logger.info("Pre-calculating team form strings for historical_df...")

    # 1) Copy and ensure 'game_date' is datetime
    df_for_features = historical_df.copy()
    df_for_features['game_date'] = pd.to_datetime(
        df_for_features['game_date'], errors='coerce'
    )
    # ───── Make a “game_date_et” alias so the FE modules still find it ─────
    df_for_features['game_date_et'] = df_for_features['game_date']


    # 2) Build a long DataFrame of one row per team per game
    full = historical_df.copy()
    full['game_date'] = pd.to_datetime(full['game_date'], errors='coerce')
    # If you only want completed games, uncomment and adjust:
    # full = full[full['status_short'].isin(['F', 'Final'])]

    # Home‐team rows
    home_part = full[[
        'game_id', 'game_date', 'home_team_id', 'home_score', 'away_score'
    ]].copy().rename(columns={
        'home_team_id': 'team_id',
        'home_score': 'team_score',
        'away_score': 'opp_score'
    })
    home_part['result'] = np.where(
        home_part['team_score'] > home_part['opp_score'], 'W', 'L'
    )
    home_part = home_part[['game_id', 'team_id', 'game_date', 'result']]

    # Away‐team rows
    away_part = full[[
        'game_id', 'game_date', 'away_team_id', 'away_score', 'home_score'
    ]].copy().rename(columns={
        'away_team_id': 'team_id',
        'away_score': 'team_score',
        'home_score': 'opp_score'
    })
    away_part['result'] = np.where(
        away_part['team_score'] > away_part['opp_score'], 'W', 'L'
    )
    away_part = away_part[['game_id', 'team_id', 'game_date', 'result']]

    team_games_long = pd.concat(
        [home_part, away_part],
        ignore_index=True
    )

    # 3) Compute each team’s rolling‐window form string
    window_size = 5  # adjust to desired lookback

    def compute_form_for_team(df_group: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df_group.sort_values('game_date').reset_index(drop=True)
        results = df_sorted['result'].tolist()
        forms = []
        for idx in range(len(results)):
            start_idx = max(0, idx - window_size)
            past_slice = results[start_idx:idx]
            forms.append(''.join(past_slice))
        df_sorted['form'] = forms
        return df_sorted

    team_games_with_form = (
        team_games_long
        .groupby('team_id', group_keys=False)
        .apply(compute_form_for_team)
        .reset_index(drop=True)
    )

    # 4) Merge form strings back onto df_for_features
    home_form_map = team_games_with_form.rename(columns={
        'team_id': 'home_team_id',
        'form': 'home_current_form'
    })[['game_id', 'home_team_id', 'home_current_form']]

    away_form_map = team_games_with_form.rename(columns={
        'team_id': 'away_team_id',
        'form': 'away_current_form'
    })[['game_id', 'away_team_id', 'away_current_form']]

    df_with_forms = df_for_features.merge(
        home_form_map,
        on=['game_id', 'home_team_id'],
        how='left'
    ).merge(
        away_form_map,
        on=['game_id', 'away_team_id'],
        how='left'
    )

    df_with_forms['home_current_form'] = df_with_forms['home_current_form'].fillna('')
    df_with_forms['away_current_form'] = df_with_forms['away_current_form'].fillna('')

    logger.info("Team form strings calculated and merged on historical_df.")

    df_for_features = df_with_forms

    # Now call the MLB feature pipeline with df_for_features…
    features_df = run_mlb_feature_pipeline(
        df=df_for_features,
        mlb_historical_team_stats_df=(
            team_stats_df.copy()
            if team_stats_df is not None and not team_stats_df.empty
            else None
        ),
        mlb_historical_games_df=(
            historical_df.copy()
            if historical_df is not None and not historical_df.empty
            else None
        ),
        rolling_window_sizes=rolling_windows_list,
        debug=args.debug
    )
    # …[Feature Cleaning, Pre‐selection]…
    if features_df.columns.duplicated().any():
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep='first')]

    # Drop rows where our targets are missing
    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    if features_df.empty:
        logger.error("No rows after dropping targets. Exiting.")
        sys.exit(1)
    #  …[Feature Value Analysis if args.run_analysis]…

    potential_feature_cols = features_df.select_dtypes(include=np.number).columns
    cols_to_exclude = set(TARGET_COLUMNS + ['game_id', 'game_date'])
    feature_candidates = []
    for col in potential_feature_cols:
        if col in cols_to_exclude:
            continue
        if features_df[col].isnull().all() or features_df[col].var() < 1e-8:
            continue

        # (Issue 9) Log if an “expected advanced feature” is missing from the DataFrame. For example:
        if col.startswith(("h_team_hist_", "a_team_hist_", "h_team_off_")):
            if col not in features_df.columns:
                logger.warning(f"Expected advanced‐feature '{col}' missing from features_df.")

        # Only keep if it matches our safe prefixes or exact names
        if col.startswith(MLB_SAFE_FEATURE_PREFIXES) or col in MLB_SAFE_EXACT_FEATURE_NAMES:
            feature_candidates.append(col)

    if not feature_candidates:
        logger.error("No feature candidates found after prefix/name filtering for MLB. Exiting.")
        sys.exit(1)

    X_select = features_df[feature_candidates].copy()
    y_home_select = features_df[TARGET_COLUMNS[0]].copy()
    y_away_select = features_df[TARGET_COLUMNS[1]].copy()

    # Impute any remaining NaNs before Lasso/ElasticNet
    if X_select.isnull().any().any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_select = pd.DataFrame(
            imputer.fit_transform(X_select),
            columns=X_select.columns,
            index=X_select.index
        )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_select_scaled = scaler.fit_transform(X_select)

    alphas_select = np.logspace(-5, 1, 60)

    final_feature_list_for_models = []

    if args.feature_selection == "lasso":
        from sklearn.linear_model import LassoCV
        lasso_home = LassoCV(
            alphas=alphas_select,
            cv=DEFAULT_CV_FOLDS,
            max_iter=3000,
            tol=1e-3,
            random_state=SEED
        ).fit(X_select_scaled, y_home_select)

        lasso_away = LassoCV(
            alphas=alphas_select,
            cv=DEFAULT_CV_FOLDS,
            max_iter=3000,
            tol=1e-3,
            random_state=SEED
        ).fit(X_select_scaled, y_away_select)

        selected_mask = (np.abs(lasso_home.coef_) > 1e-5) | (np.abs(lasso_away.coef_) > 1e-5)
        final_feature_list_for_models = list(X_select.columns[selected_mask])
        logger.info(f"LassoCV selected {len(final_feature_list_for_models)} features for MLB.")

    elif args.feature_selection == "elasticnet":
        from sklearn.linear_model import ElasticNetCV
        enet_home = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=alphas_select,
            cv=DEFAULT_CV_FOLDS,
            max_iter=3000,
            tol=1e-3,
            random_state=SEED
        ).fit(X_select_scaled, y_home_select)

        enet_away = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=alphas_select,
            cv=DEFAULT_CV_FOLDS,
            max_iter=3000,
            tol=1e-3,
            random_state=SEED
        ).fit(X_select_scaled, y_away_select)

        selected_mask_enet = (np.abs(enet_home.coef_) > 1e-5) | (np.abs(enet_away.coef_) > 1e-5)
        final_feature_list_for_models = list(X_select.columns[selected_mask_enet])
        logger.info(f"ElasticNetCV selected {len(final_feature_list_for_models)} features for MLB.")
    else:
        logger.error(f"Unknown feature selection: {args.feature_selection}")
        sys.exit(1)

    if not final_feature_list_for_models:
        logger.error("No features selected. Exiting.")
        sys.exit(1)

    # =================== NEW CODE BLOCK FOR PILLAR 2 START ===================
    logger.info("--- Starting Pillar 2: Enriching feature set with imputation flags ---")

    # The initial list selected by the regularized model
    initial_selected_features = final_feature_list_for_models.copy()
    logger.info(f"Initial feature count from Lasso: {len(initial_selected_features)}")

    flags_to_add = []
    for feature in initial_selected_features:
        # We are not interested in adding a flag for a flag that was already selected
        if feature.endswith("_imputed"):
            continue

        # Construct the potential name of the corresponding imputation flag
        potential_flag_name = f"{feature}_imputed"

        # Check if this flag actually exists in the full dataframe we generated
        if potential_flag_name in features_df.columns:
            flags_to_add.append(potential_flag_name)
            logger.debug(f"Found corresponding flag '{potential_flag_name}' for feature '{feature}'.")

    # Combine the original list with the new flags, removing duplicates
    if flags_to_add:
        logger.info(f"Found {len(flags_to_add)} corresponding imputation flags to add to the feature set.")
        
        # Use a dictionary to preserve order and remove duplicates from the combined list
        combined_features_ordered = list(dict.fromkeys(initial_selected_features + flags_to_add))
        
        # This is the final, enriched list of features to be used by the models
        final_feature_list_for_models = combined_features_ordered
        
        newly_added_flags = set(final_feature_list_for_models) - set(initial_selected_features)
        logger.info(f"Total features after adding flags: {len(final_feature_list_for_models)}. "
                    f"Added {len(newly_added_flags)} new flags.")
        logger.debug(f"Flags added: {sorted(list(newly_added_flags))}")
    else:
        logger.info("No new corresponding imputation flags found to add.")

    logger.info("--- Finished Pillar 2 enrichment ---")
    # =================== NEW CODE BLOCK FOR PILLAR 2 END =====================
        
    if args.write_selected_features:
        sf_path = MODELS_DIR_MLB / "mlb_selected_features.json"
        sf_path.parent.mkdir(parents=True, exist_ok=True)      # ensure the dir exists
        with open(sf_path, "w") as f:
            json.dump(final_feature_list_for_models, f, indent=4)
        logger.info(f"MLB selected features ({len(final_feature_list_for_models)}) saved to {sf_path}. Exiting.")
        sys.exit(0)

    # Data Splitting (chronological, generic logic)
    essential_cols = ['game_id', 'game_date'] + TARGET_COLUMNS
    cols_for_df_sel = list(set(essential_cols + final_feature_list_for_models)) # Unique columns
    features_df_selected = features_df[[c for c in cols_for_df_sel if c in features_df.columns]].sort_values('game_date').reset_index(drop=True)
    

    if args.write_selected_features:
        sf_path = MODELS_DIR_MLB / "mlb_selected_features.json"
        sf_path.parent.mkdir(parents=True, exist_ok=True)      # ensure the dir exists
        with open(sf_path, "w") as f:
            json.dump(final_feature_list_for_models, f, indent=4)
        logger.info(f"MLB selected features ({len(final_feature_list_for_models)}) saved to {sf_path}. Exiting.")
        sys.exit(0)

    # Data Splitting (chronological, generic logic)
    essential_cols = ['game_id', 'game_date'] + TARGET_COLUMNS
    cols_for_df_sel = list(set(essential_cols + final_feature_list_for_models)) # Unique columns
    features_df_selected = features_df[[c for c in cols_for_df_sel if c in features_df.columns]].sort_values('game_date').reset_index(drop=True)
    
    n_total = len(features_df_selected)
    test_split_idx = int(n_total * (1 - args.test_size))
    val_split_idx = int(test_split_idx * (1 - args.val_size / (1-args.test_size))) # val_size is fraction of (train+val)

    train_df = features_df_selected.iloc[:val_split_idx]
    val_df   = features_df_selected.iloc[val_split_idx:test_split_idx]
    test_df  = features_df_selected.iloc[test_split_idx:]

    feature_cols_for_X = final_feature_list_for_models + (['game_date'] if args.use_weights else [])
    X_train = train_df[[c for c in feature_cols_for_X if c in train_df.columns]]
    X_val   = val_df[[c for c in feature_cols_for_X if c in val_df.columns]]
    X_test  = test_df[final_feature_list_for_models] # game_date not needed for X_test features

    y_train_home, y_train_away = train_df[TARGET_COLUMNS[0]], train_df[TARGET_COLUMNS[1]]
    y_val_home, y_val_away     = val_df[TARGET_COLUMNS[0]], val_df[TARGET_COLUMNS[1]]
    y_test_home, y_test_away   = test_df[TARGET_COLUMNS[0]], test_df[TARGET_COLUMNS[1]]
    logger.info(f"MLB Data Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    if X_train.empty or X_val.empty or X_test.empty: logger.error("Empty data split(s). Exiting."); sys.exit(1)

    

    # MLB Parameter Distributions (can be refined)
    MLB_SVR_PARAM_DIST = {
        'svr__kernel': ['rbf', 'linear'], 'svr__C': loguniform(0.1, 100), # Runs are smaller scale
        'svr__epsilon': uniform(0.05, 0.5), 'svr__gamma': ['scale', 'auto'] + list(loguniform(1e-5, 1e-1).rvs(size=10,random_state=SEED))
    }
    MLB_XGB_PARAM_DIST = { # Similar to NBA, but learning rate and depth might be different
        'xgb__n_estimators': randint(50, 400), 'xgb__learning_rate': uniform(0.005, 0.15),
        'xgb__max_depth': randint(2, 7), 'xgb__subsample': uniform(0.5, 0.5),
        'xgb__colsample_bytree': uniform(0.5, 0.5), # ... other xgb params
        'xgb__reg_alpha': uniform(0.0, 2.0), 'xgb__reg_lambda': uniform(0.0, 2.0),
    }
    predictor_map_mlb = {
        "xgb": MLBXGBoostPredictor, "ridge": MLBRidgePredictor, "svr": MLBSVRPredictor,
    }
    param_dist_map_mlb = {
        "svr": MLB_SVR_PARAM_DIST, "xgb": MLB_XGB_PARAM_DIST
    }

    # Base Model Training Loop
    models_to_run = [m.strip().lower() for m in args.models.split(',') if m.strip().lower() in predictor_map_mlb]
    if not models_to_run: logger.error("No valid MLB models specified. Exiting."); sys.exit(1)
    
    logger.info(f"--- Starting Tuning & Training for MLB Base Models: {models_to_run} ---")
    all_mlb_metrics = []
    validation_predictions_collector_mlb = {}

    # Custom loss weights from args or defaults
    runs_loss_diff_w = getattr(args, 'runs_loss_diff_weight', 0.6)
    runs_loss_total_w = getattr(args, 'runs_loss_total_weight', 0.4)
    combined_loss_acc_w = getattr(args, 'combined_loss_accuracy_weight', 0.7)
    combined_loss_dist_w = getattr(args, 'combined_loss_dist_weight', 0.3)

    for model_key in models_to_run:
        PredictorClassMLB = predictor_map_mlb[model_key]
        param_dist_mlb = param_dist_map_mlb.get(model_key) if not args.skip_tuning else None
        
        metrics_res, val_preds = tune_and_evaluate_predictor(
            predictor_class=PredictorClassMLB,
            X_train=X_train, y_train_home=y_train_home, y_train_away=y_train_away,
            X_val=X_val, y_val_home=y_val_home, y_val_away=y_val_away,
            X_test=X_test, y_test_home=y_test_home, y_test_away=y_test_away,
            model_name_prefix=model_key, feature_list=final_feature_list_for_models,
            param_dist=param_dist_mlb, n_iter=args.tune_iterations, n_splits=args.cv_splits,
            scoring=args.scoring_metric, use_recency_weights=args.use_weights,
            weight_method=args.weight_method, weight_half_life=args.weight_half_life,
            visualize=args.visualize, save_plots=args.save_plots,
            runs_loss_diff_weight=runs_loss_diff_w, runs_loss_total_weight=runs_loss_total_w,
            combined_loss_accuracy_weight=combined_loss_acc_w, combined_loss_dist_weight=combined_loss_dist_w
        )
        if metrics_res: all_mlb_metrics.append(metrics_res)
        if val_preds: validation_predictions_collector_mlb[model_key] = val_preds
    
    # Ensemble Weight Optimization & Evaluation (using MLB specific names and collector)
    # ... (Logic for optimizing and evaluating ensemble as in NBA, using mlb variables) ...
    # (This section is long, assume it's ported correctly)
    successful_mlb_models = list(validation_predictions_collector_mlb.keys())
    if len(successful_mlb_models) > 1:
        opt_weights_mlb = optimize_ensemble_weights(
            validation_predictions_collector_mlb, successful_mlb_models, 'avg_mae', l2_lambda=0.05) # Use MLB specific output filename
        if opt_weights_mlb: logger.info(f"Optimized MLB weights: {opt_weights_mlb}")
        # ... (Evaluate this optimized ensemble on test set) ...
        # This part involves loading each model again, predicting on test, then applying weights.
        # For brevity, assuming this logic is ported.

    # Final Summary Logging
    logger.info("\n" + "="*80 + "\n--- MLB Training Pipeline Summary ---")
    if all_mlb_metrics:
        metrics_df_mlb = pd.DataFrame(all_mlb_metrics)
        # ... (Display and save metrics_df_mlb similar to NBA version) ...
        # Ensure column names match the output of MLB tune_and_evaluate_predictor
        cols_to_show_mlb = [ # Example columns, adjust based on actual metrics collected
            'model_name', 'feature_count', 'test_mae_home', 'test_mae_away',
            'test_mae_total', 'test_mae_diff', 'test_mlb_runs_loss', 'betting_metrics_mlb'
        ]
        cols_present_mlb = [c for c in cols_to_show_mlb if c in metrics_df_mlb.columns]
        # ... (to_string and save to CSV)
        logger.info("\nMLB Base Model Performance Summary:\n" + metrics_df_mlb[cols_present_mlb].to_string(index=False))
        metrics_df_mlb.to_csv(REPORTS_DIR / f"mlb_training_metrics_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv", index=False)


    logger.info(f"\n--- MLB Model Training Pipeline Finished in {time.time() - start_pipeline_time:.2f} seconds ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLB Runs Prediction Model Tuning & Training Pipeline")
    # Data Source Args - update defaults for MLB
    parser.add_argument("--data-source", type=str, default="supabase", choices=["csv", "supabase"])
    parser.add_argument("--historical-csv-path", type=str, default=str(PROJECT_ROOT_TRAIN / 'data' / 'mlb_historical_games.csv'))
    parser.add_argument("--team-stats-csv-path", type=str, default=str(PROJECT_ROOT_TRAIN / 'data' / 'mlb_team_stats.csv'))
    parser.add_argument("--lookback-days", type=int, default=2190, help="Days of historical data (e.g., ~6 seasons for MLB)") # Longer for MLB
    
    # Feature Engineering Args - adjust defaults for MLB
    parser.add_argument("--rolling-windows", type=str, default="15,30,60,100", help="Rolling windows for MLB stats") # MLB specific
    parser.add_argument("--h2h-window", type=int, default=10, help="H2H games for MLB (e.g., recent series or season)") # MLB specific

    # Model Selection (generic)
    parser.add_argument("--models", type=str, default="ridge,svr,xgb", help="Models to train (ridge,svr,xgb)")
    
    # Data Splitting (generic)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15) # Fraction of (Train+Val)
    # Feature selection threshold (generic, but effectiveness depends on feature scale)
    # parser.add_argument("--importance-threshold", type=float, default=0.001) # Might need adjustment

    # Training Options (generic)
    parser.add_argument("--use-weights", action="store_true", help="Use recency weighting")
    parser.add_argument("--weight-method", type=str, default="exponential", choices=["exponential", "half_life"])
    parser.add_argument("--weight-half-life", type=int, default=120, help="Half-life in days for MLB (longer season)") # MLB specific

    # Hyperparameter Tuning (generic)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--tune-iterations", type=int, default=100) # Fewer for quicker test, more for production
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--scoring-metric", type=str, default="neg_mean_absolute_error")

    # Output and Debugging (generic)
    parser.add_argument("--run-analysis", action="store_true", help="Run optional analysis")
    parser.add_argument("--visualize", action="store_true", help="Show plots interactively")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to reports_mlb directory")
    parser.add_argument("--allow-dummy", action="store_true", help="Allow dummy modules if real ones fail")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    
    # Feature Selection Method (generic)
    parser.add_argument("--write-selected-features", action="store_true", help="Write mlb_selected_features.json and exit")
    parser.add_argument("--feature-selection", choices=["lasso", "elasticnet"], default="lasso")
    parser.add_argument("--plot-elasticnet-path", action="store_true", help="Save ElasticNet coefficient path plots")
    
    # Add new arguments for custom loss weights if you want them configurable
    parser.add_argument("--runs-loss-diff-weight", type=float, default=0.6)
    parser.add_argument("--runs-loss-total-weight", type=float, default=0.4)
    parser.add_argument("--combined-loss-accuracy-weight", type=float, default=0.7)
    parser.add_argument("--combined-loss-dist-weight", type=float, default=0.3)


    cli_args = parser.parse_args()

    run_training_pipeline(cli_args)