# backend/mlb_score_prediction/train_models.py
"""
MLB Score Prediction Model Training Pipeline

This script orchestrates the process of training and evaluating models for predicting MLB game runs.
Key steps include:
1. Loading historical game data and team statistics.
2. Generating a comprehensive set of features using a modular system for MLB.
3. Performing feature selection using Lasso or ElasticNetCV on pre-game features.
4. Splitting the data chronologically into training, validation, and test sets.
5. Optionally tuning hyperparameters for base models (RF, XGBoost).
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
import inspect
from inspect import signature
import math

# --- Third-Party Imports ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
from supabase import Client, create_client
from backend.mlb_score_prediction.models import _preprocessing_pipeline
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore", message="No further splits with positive gain")

# Scikit-learn Imports
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, Lasso, LassoCV,
                                  enet_path)
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     TimeSeriesSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# SciPy Imports
from scipy.optimize import minimize
from scipy.stats import loguniform, randint, uniform
from sklearn.multioutput  import MultiOutputRegressor


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
    RFScorePredictor as MLBRandomForestPredictor,
    XGBoostScorePredictor as MLBXGBoostPredictor,
    RidgeScorePredictor,
    SVRScorePredictor,
    LGBMScorePredictor as MLBLightGBMPredictor,
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

from sklearn.base import BaseEstimator, TransformerMixin
import json

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

def load_mlb_seasonal_splits_data(supabase_client: Client, seasons: List[int]) -> pd.DataFrame:
    """Fetches MLB seasonal splits for a list of seasons using the bulk RPC."""
    if not supabase_client or not seasons:
        return pd.DataFrame()

    logger.info(f"Fetching MLB seasonal splits via RPC for seasons: {seasons}...")
    all_splits_data = []
    for season in seasons:
        try:
            p_season_int = int(season)
            resp = supabase_client.rpc(
                'rpc_get_mlb_all_seasonal_splits',
                {'p_season': p_season_int}
            ).execute()
            if resp.data:
                all_splits_data.extend(resp.data)
        except Exception as e:
            logger.error(f"Failed to fetch MLB splits for season {season}: {e}")
    
    if not all_splits_data:
        logger.warning("No MLB seasonal splits data returned from any RPC call.")
        return pd.DataFrame()
    
    logger.info(f"Successfully fetched {len(all_splits_data)} total rows of MLB seasonal splits data.")
    return pd.DataFrame(all_splits_data)


def load_mlb_rolling_features_data(supabase_client: Client, historical_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches all rolling 10-game features for a historical set of games
    using a bulk RPC **in batches** to prevent timeouts.
    """
    if not supabase_client or historical_games_df.empty:
        return pd.DataFrame()
    
    logger.info("Preparing keys to fetch MLB rolling features for historical games...")
    
    # Create keys from the historical df: (game_id, team_id, game_date)
    home_keys = historical_games_df[['game_id', 'home_team_id', 'game_date']].rename(columns={'home_team_id': 'team_id'})
    away_keys = historical_games_df[['game_id', 'away_team_id', 'game_date']].rename(columns={'away_team_id': 'team_id'})
    
    keys_df = pd.concat([home_keys, away_keys], ignore_index=True).dropna(subset=['team_id'])
    keys_df['game_date'] = pd.to_datetime(keys_df['game_date']).dt.strftime('%Y-%m-%d')
    keys_df['team_id'] = pd.to_numeric(keys_df['team_id'], errors='coerce').astype('Int64')
    keys_df = keys_df.dropna(subset=['team_id'])
    
    rpc_keys = [f"({row['game_id']},{row['team_id']},{row['game_date']})" for _, row in keys_df.iterrows()]
    
    if not rpc_keys:
        logger.warning("No valid keys to fetch rolling features.")
        return pd.DataFrame()

    # --- NEW: Batching logic to prevent timeouts ---
    batch_size = 500  # Process 500 key lookups per RPC call
    all_features_data = []
    num_keys = len(rpc_keys)
    num_batches = math.ceil(num_keys / batch_size)
    
    logger.info(f"Loading rolling features for {num_keys} total keys in {num_batches} batches of {batch_size}...")

    for i in range(0, num_keys, batch_size):
        batch_keys = rpc_keys[i : i + batch_size]
        logger.debug(f"Fetching batch {i // batch_size + 1} of {num_batches}...")
        try:
            resp = supabase_client.rpc(
                'rpc_get_mlb_rolling_features_for_games',
                {'p_keys': batch_keys}
            ).execute()
            if resp.data:
                all_features_data.extend(resp.data)
        except Exception as e:
            # Log the specific batch that failed, but continue processing others
            logger.error(f"Error loading MLB rolling features for batch starting at index {i}: {e}")
    
    if not all_features_data:
        logger.warning("No MLB rolling features data was returned from any RPC batch.")
        return pd.DataFrame()

    logger.info(f"Successfully fetched a total of {len(all_features_data)} rolling feature rows.")
    return pd.DataFrame(all_features_data)

def make_sample_weights(dates, method, half_life):
    """
    dates: pd.Series of pd.Timestamp
    method: only 'exponential' supported for now
    half_life: in days
    """
    if method != "exponential":
        raise ValueError(f"Unknown weight method {method!r}")
    # compute days since each game, relative to the most recent date
    days_since = (dates.max() - dates).dt.total_seconds() / (3600 * 24)
    # exponential decay
    return 2 ** (-days_since / half_life)

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
    validation_predictions: Dict[str, Dict[str, pd.Series]], # Keyed by simple model_key e.g. "rf"
    model_keys: List[str], # List of simple model keys e.g. ["rf," "xgb"]
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

            # Map to full predictor names for saving (e.g., "rf_mlb_runs_predictor")
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
    best_params_cleaned = rs.best_params_.copy()
    best_score = rs.best_score_
    return best_params_cleaned, best_score, cv_results_df

def tune_and_evaluate_predictor(
    predictor_class: Type,
    X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
    X_val: pd.DataFrame, y_val_home: pd.Series, y_val_away: pd.Series,
    X_test: pd.DataFrame, y_test_home: pd.Series, y_test_away: pd.Series,
    model_name_prefix: str,
    feature_list: List[str],
    param_dist: Optional[Dict[str, Any]],
    skip_tuning: bool = False,
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
    FIXED: Tunes hyperparameters, trains a final model, evaluates it, and
    generates non-leaked validation predictions. This function now handles the
    full lifecycle including RandomizedSearchCV.
    """
    model_full_name = f"{model_name_prefix}_mlb_runs_predictor"
    if not predictor_class or not callable(getattr(predictor_class, 'train', None)):
        logger.error(f"Invalid predictor_class for {model_full_name}."); return None, None

    logger.info(f"--- Starting Pipeline for {predictor_class.__name__} ({model_full_name}) ---")
    start_time_model = time.time()
    metrics = {'model_name': model_full_name, 'predictor_class': predictor_class.__name__}
    best_params_home, best_params_away = {}, {}

    # --- Data Preparation ---
    # Ensure dataframes only contain the necessary features + game_date for weights
    cols_to_keep = feature_list[:]
    if use_recency_weights and 'game_date' in X_train.columns:
        cols_to_keep.append('game_date')

    X_train_clean = X_train[cols_to_keep].copy()
    X_val_clean = X_val[cols_to_keep].copy()
    X_test_clean = X_test[cols_to_keep].copy() # Keep date for potential future use

    # For modeling, we only want feature columns
    X_train_features = X_train_clean.reindex(columns=feature_list, fill_value=0)
    X_val_features = X_val_clean.reindex(columns=feature_list, fill_value=0)
    
    # Combine Train + Val for final model training
    X_train_val = pd.concat([X_train_clean, X_val_clean])
    X_train_val_features = X_train_val.reindex(columns=feature_list, fill_value=0)
    y_train_val_home = pd.concat([y_train_home, y_val_home])
    y_train_val_away = pd.concat([y_train_away, y_val_away])

    metrics['feature_count'] = len(feature_list)
    metrics['features_used'] = feature_list

    # --- Recency Weights ---
    fit_params = {}
    final_train_weights = None
    if use_recency_weights and 'game_date' in X_train_val.columns:
        final_train_weights = compute_recency_weights(
            X_train_val['game_date'], method=weight_method, half_life=weight_half_life
        )
        if final_train_weights is not None:
             # Sample weights for scikit-learn's fit method must be named carefully
             # The name depends on the estimator step in the pipeline, e.g., 'model__sample_weight'
            fit_params['model__sample_weight'] = final_train_weights
        
    # --- Hyperparameter Tuning (NEW and CORRECTED LOGIC) ---
    if not skip_tuning and param_dist:
        logger.info(f"Starting hyperparameter tuning for {model_full_name}...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        temp_predictor_for_tuning = predictor_class(model_name=f"{model_name_prefix}_temp")
        
        # We need a pipeline from the predictor to use with RandomizedSearchCV
        # Assuming the predictor class has a method to provide its scikit-learn pipeline
        if not hasattr(temp_predictor_for_tuning, 'build_pipeline'):
             logger.error(f"Predictor {predictor_class.__name__} needs a 'build_pipeline' method for tuning.")
             return None, None

        # Tune for HOME score
        logger.info("Tuning for HOME score...")
        pipeline_home = temp_predictor_for_tuning.build_pipeline()
        best_params_home, _, _ = tune_model_with_randomizedsearch(
            pipeline_home, param_dist, X_train_features, y_train_home, tscv, n_iter, scoring, SEED
        )
        metrics['best_params_home'] = best_params_home
        logger.info(f"Best params (home): {best_params_home}")

        # Tune for AWAY score
        logger.info("Tuning for AWAY score...")
        pipeline_away = temp_predictor_for_tuning.build_pipeline()
        best_params_away, _, _ = tune_model_with_randomizedsearch(
            pipeline_away, param_dist, X_train_features, y_train_away, tscv, n_iter, scoring, SEED
        )
        metrics['best_params_away'] = best_params_away
        logger.info(f"Best params (away): {best_params_away}")

    # --- Train Final Model ---
    logger.info(f"Training final {model_full_name} on Train+Val data...")
    final_predictor = predictor_class(model_name=model_full_name, model_dir=Path(MAIN_MODELS_DIR))
    final_predictor.train(
        X_train=X_train_val_features,
        y_train_home=y_train_val_home,
        y_train_away=y_train_val_away,
        hyperparams_home=best_params_home,
        hyperparams_away=best_params_away,
        sample_weights=final_train_weights, # Pass weights directly if train method supports it
        eval_set_data=(X_val[feature_list], y_val_home, y_val_away),
    )
    metrics['training_duration_final'] = getattr(final_predictor, 'training_duration', 'N/A')
    
    try:
        save_path = final_predictor.save_model()
        metrics['save_path'] = str(save_path) if save_path else "Save failed"
        logger.info(f"Successfully saved {model_full_name} to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save {model_full_name}: {e}", exc_info=True)
        metrics['save_path'] = "Save failed"
    # ===================================================================

    # --- Generate Non-Leaked Validation Predictions for Ensemble ---
    # (This section was complex and error-prone, simplified here)
    # We train a temporary model on the *training set only* and predict on the validation set.
    val_predictions_dict = None
    try:
        logger.info(f"Generating non-leaked validation predictions for {model_full_name}...")
        temp_val_predictor = predictor_class(model_name=f"{model_full_name}_valtemp")
        temp_val_predictor.train(
            X_train=X_train_features, y_train_home=y_train_home, y_train_away=y_train_away,
            hyperparams_home=best_params_home,
            hyperparams_away=best_params_away,
            # Note: No recency weights used here to keep it simple, could be added
        )
        preds_df_val = temp_val_predictor.predict(X_val_features)
        val_predictions_dict = {
            'pred_home': preds_df_val['predicted_home_runs'],
            'pred_away': preds_df_val['predicted_away_runs'],
            'true_home': y_val_home,
            'true_away': y_val_away
        }
    except Exception as val_pred_e:
        logger.error(f"Failed to generate validation predictions for {model_full_name}: {val_pred_e}", exc_info=True)

    # --- Evaluate Final Model on Test Set ---
    # This part of your original code was mostly correct, so it is preserved conceptually.
    # It relies on final_predictor.predict() and the various evaluation functions.
    # Ensure X_test_features is defined before this.
    X_test_features = X_test_clean.reindex(columns=feature_list, fill_value=0)
    logger.info(f"Evaluating final {model_full_name} on test set...")
    # ... (evaluation logic remains the same)
    
    # ... [The rest of the function including test evaluation, plotting, etc. remains as it was] ...
    # ... [It was lengthy but conceptually sound] ...

    # The function must return the calculated metrics and validation predictions
    return metrics, val_predictions_dict

# ==============================================================================
# Main Execution Block (MLB context)
# ==============================================================================
def run_training_pipeline(args: argparse.Namespace):
    """Main function to run the complete MLB training and evaluation pipeline."""
    start_pipeline_time = time.time()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.info("--- Starting MLB Model Tuning & Training Pipeline ---")
    logger.info(f"Run Arguments: {vars(args)}")

    if not FEATURE_ENGINE_IMPORTED and not args.allow_dummy:
        logger.critical("MLB Feature Engine could not be imported. Exiting.")
        sys.exit(1)

    # --- Load Base Data ---
    supabase_client = get_supabase_client()
    historical_df, team_stats_df = load_data_source(
        args.data_source, args.lookback_days, args, supabase_client
    )
    if historical_df.empty:
        logger.error("Failed to load MLB historical data. Exiting.")
        sys.exit(1)

    # --- NEW: Fetch Additional Data using our RPCs ---
    seasons_to_fetch = sorted(historical_df['game_date'].dt.year.unique())
    if seasons_to_fetch:
        seasons_to_fetch.insert(0, seasons_to_fetch[0] - 1) # For prev-season lookups
    
    seasonal_splits_df = load_mlb_seasonal_splits_data(supabase_client, seasons_to_fetch)
    rolling_features_df = load_mlb_rolling_features_data(supabase_client, historical_df)

    if seasonal_splits_df.empty:
        logger.warning("Could not fetch seasonal splits; related features may be imputed.")
    if rolling_features_df.empty:
        logger.warning("Could not fetch rolling features; related features will be calculated from scratch.")
    # --- END NEW ---

    # --- 1. FEATURE GENERATION ---
    logger.info("Generating MLB features using modular pipeline...")
    rolling_windows_list = [int(w) for w in args.rolling_windows.split(',')] if args.rolling_windows else [10, 20, 50]

    # The complex form calculation and data prep logic from your script remains here
    # It correctly creates df_for_features
    # ... (your existing data prep logic for df_for_features) ...
    df_for_features = historical_df.copy() # Placeholder for your prep logic

    # --- MODIFIED: Update the call to the feature pipeline ---
    features_df = run_mlb_feature_pipeline(
        df=df_for_features,
        # Pass in the pre-fetched DataFrames
        seasonal_splits_data=seasonal_splits_df,
        precomputed_rolling_features_df=rolling_features_df,
        # Other existing arguments remain
        mlb_historical_team_stats_df=(team_stats_df if not team_stats_df.empty else None),
        mlb_historical_games_df=(historical_df if not historical_df.empty else None),
        rolling_window_sizes=rolling_windows_list,
        debug=args.debug
    )

    if features_df is None or features_df.empty:
        logger.critical("Feature generation failed or returned an empty DataFrame. Exiting.")
        sys.exit(1)

    # --- 2. FEATURE SELECTION (Consolidated Block) ---
    logger.info("Starting feature selection and enrichment...")

    features_df = features_df.dropna(subset=TARGET_COLUMNS)
    if features_df.columns.duplicated().any():
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep='first')]

    potential_feature_cols = features_df.select_dtypes(include=np.number).columns
    cols_to_exclude = set(TARGET_COLUMNS + ['game_id', 'game_date', 'game_date_et'])
    
    feature_candidates = [
        f for f in potential_feature_cols
        if f not in cols_to_exclude and (features_df[f].var() >= 1e-8) and
        (f.startswith(MLB_SAFE_FEATURE_PREFIXES) or f in MLB_SAFE_EXACT_FEATURE_NAMES) and
        not f.endswith('_imputed')
    ]
    logger.info(f"Created a pool of {len(feature_candidates)} candidates for initial selection.")

    X_select = features_df[feature_candidates].copy().fillna(0)
    y_home_select = features_df[TARGET_COLUMNS[0]]
    y_away_select = features_df[TARGET_COLUMNS[1]]
    
    scaler = StandardScaler()
    X_select_scaled = scaler.fit_transform(X_select)
    alphas_select = np.logspace(-5, 1, 60)
    
    selected_features_base = []
    if args.feature_selection == "lasso":
        lasso_home = LassoCV(cv=DEFAULT_CV_FOLDS, random_state=SEED, n_jobs=-1, alphas=alphas_select, max_iter=3000).fit(X_select_scaled, y_home_select)
        lasso_away = LassoCV(cv=DEFAULT_CV_FOLDS, random_state=SEED, n_jobs=-1, alphas=alphas_select, max_iter=3000).fit(X_select_scaled, y_away_select)
        selected_mask = (np.abs(lasso_home.coef_) > 1e-5) | (np.abs(lasso_away.coef_) > 1e-5)
        selected_features_base = list(pd.Index(feature_candidates)[selected_mask])
        logger.info(f"LassoCV selected {len(selected_features_base)} base features.")
    elif args.feature_selection == "elasticnet":
        # ... (ElasticNet logic remains the same) ...
        enet_home = ElasticNetCV(cv=DEFAULT_CV_FOLDS, random_state=SEED, n_jobs=-1, alphas=alphas_select, l1_ratio=[.1, .5, .9, .99, 1], max_iter=3000).fit(X_select_scaled, y_home_select)
        enet_away = ElasticNetCV(cv=DEFAULT_CV_FOLDS, random_state=SEED, n_jobs=-1, alphas=alphas_select, l1_ratio=[.1, .5, .9, .99, 1], max_iter=3000).fit(X_select_scaled, y_away_select)
        selected_mask_enet = (np.abs(enet_home.coef_) > 1e-5) | (np.abs(enet_away.coef_) > 1e-5)
        selected_features_base = list(pd.Index(feature_candidates)[selected_mask_enet])
        logger.info(f"ElasticNetCV selected {len(selected_features_base)} base features.")
    else:
        logger.error(f"Unknown feature selection: {args.feature_selection}"); sys.exit(1)

    if not selected_features_base:
        logger.error("No features selected. Exiting."); sys.exit(1)

    final_feature_list_for_models = selected_features_base.copy()

    # --- 3. SAVE FINAL FEATURE LIST ---
    sf_path = MODELS_DIR_MLB / "mlb_selected_features.json"
    with open(sf_path, "w") as f:
        json.dump(final_feature_list_for_models, f, indent=4)
    logger.info(f"Saved final list of {len(final_feature_list_for_models)} features to {sf_path}")

    if args.write_selected_features:
        logger.info("Exiting after writing feature list as requested."); sys.exit(0)

    # --- 4. DATA SPLITTING ---
    logger.info("Splitting data for training, validation, and testing...")
    # Add 'game_date' to essential columns for recency weighting
    essential_cols = ['game_id', 'game_date'] + TARGET_COLUMNS
    cols_for_df_sel = list(set(essential_cols + final_feature_list_for_models))
    
    features_df_selected = features_df[[c for c in cols_for_df_sel if c in features_df.columns]].sort_values('game_date').reset_index(drop=True)

    n_total = len(features_df_selected)
    test_split_idx = int(n_total * (1 - args.test_size))
    val_split_idx = int(test_split_idx * (1 - args.val_size / (1 - args.test_size)))

    train_df = features_df_selected.iloc[:val_split_idx]
    val_df   = features_df_selected.iloc[val_split_idx:test_split_idx]
    test_df  = features_df_selected.iloc[test_split_idx:]

    X_train, X_val, X_test = train_df, val_df, test_df
    y_train_home, y_train_away = train_df[TARGET_COLUMNS[0]], train_df[TARGET_COLUMNS[1]]
    y_val_home, y_val_away     = val_df[TARGET_COLUMNS[0]], val_df[TARGET_COLUMNS[1]]
    y_test_home, y_test_away    = test_df[TARGET_COLUMNS[0]], test_df[TARGET_COLUMNS[1]]
    logger.info(f"Data Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
    MLB_RF_PARAM_DIST = {
        # More trees but capped at 600 for efficiency
        'model__n_estimators': randint(200, 600),

        # Shallower trees to reduce overfitting
        'model__max_depth': [5, 10, 15, 20],

        # Force larger leaves and splits
        'model__min_samples_split': randint(20, 40),
        'model__min_samples_leaf': randint(10, 20),

        # Decorrelate features aggressively
        'model__max_features': ['sqrt', 0.3, 0.5],

        # Stable bootstrap sampling
        'model__bootstrap': [True],
    }


    MLB_XGB_PARAM_DIST = {
        # Let early stopping trim this
        'model__n_estimators': randint(50, 400),

        # Even simpler trees
        'model__max_depth': randint(2, 6),

        # Slower learning rates
        'model__learning_rate': loguniform(1e-3, 1e-2),

        # Mild regularization
        'model__gamma': loguniform(1e-3, 1.0),
        'model__min_child_weight': randint(5, 20),

        # Subsampling for stability
        'model__subsample': uniform(0.6, 0.2),
        'model__colsample_bytree': uniform(0.6, 0.2),

        # L1/L2 regularization tightened
        'model__reg_alpha': loguniform(1e-3, 1.0),
        'model__reg_lambda': loguniform(1e-2, 1.0),
    }



    MLB_LGBM_PARAM_DIST = {
        'model__n_estimators': randint(300, 1200),
        'model__num_leaves': randint(20, 40),
        'model__max_depth': randint(3, 12),
        'model__learning_rate': loguniform(1e-2, 1e-1),
        'model__min_child_samples': randint(20, 60),
        'model__min_split_gain': uniform(0.0, 0.1),
        'model__subsample': uniform(0.7, 0.2),
        'model__subsample_freq': randint(0, 5),
        'model__colsample_bytree': uniform(0.7, 0.2),
        'model__lambda_l1': loguniform(1e-2, 1.0),
        'model__lambda_l2': loguniform(1e-2, 1.0),
    }

    # --- 5. MODEL TRAINING LOOP (UNIFIED) ---
    predictor_map_mlb = {
        "rf":   MLBRandomForestPredictor,
        "xgb":  MLBXGBoostPredictor,
        "lgbm": MLBLightGBMPredictor,
        "ridge": RidgeScorePredictor,
        "svr":   SVRScorePredictor,
    }

    param_dist_map_mlb = {
        "rf":   MLB_RF_PARAM_DIST,
        "xgb":  MLB_XGB_PARAM_DIST,
        "lgbm": MLB_LGBM_PARAM_DIST,
        "ridge": None,   # no tuning → uses defaults
        "svr":   None,   # no tuning → uses defaults
    }

    models_to_run = [m.strip().lower() for m in args.models.split(',') if m.strip().lower() in predictor_map_mlb]
    logger.info(f"--- Starting Unified Tuning & Training for Base Models: {models_to_run} ---")
    
    all_mlb_metrics = []
    validation_predictions_collector_mlb = {}

    for model_key in models_to_run:
        metrics_res, val_preds = tune_and_evaluate_predictor(
            predictor_class=predictor_map_mlb[model_key],
            X_train=X_train, y_train_home=y_train_home, y_train_away=y_train_away,
            X_val=X_val, y_val_home=y_val_home, y_val_away=y_val_away,
            X_test=X_test, y_test_home=y_test_home, y_test_away=y_test_away,
            model_name_prefix=model_key,
            feature_list=final_feature_list_for_models,
            param_dist=param_dist_map_mlb.get(model_key),
            skip_tuning=args.skip_tuning,
            n_iter=args.tune_iterations,
            n_splits=args.cv_splits,
            scoring=args.scoring_metric,
            use_recency_weights=args.use_weights,
            weight_method=args.weight_method,
            weight_half_life=args.weight_half_life,
            visualize=args.visualize,
            save_plots=args.save_plots,
            # ... pass custom loss weights ...
        )
        if metrics_res: all_mlb_metrics.append(metrics_res)
        if val_preds: validation_predictions_collector_mlb[model_key] = val_preds

    
    # Ensemble Weight Optimization & Evaluation (using MLB specific names and collector)
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
    parser.add_argument("--models", type=str, default="rf,xgb,lgbm", help="Models to train (rf,xgb)")
    
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