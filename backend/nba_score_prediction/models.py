# backend/nba_score_prediction/models.py

"""
models.py - NBA Score Prediction Models Module

This module defines:
    - Utility functions for recency weight computation.
    - An abstract base predictor class (BaseScorePredictor) with common training,
      prediction, saving, and loading logic.
    - Specific model classes: XGBoostScorePredictor, RandomForestScorePredictor, and RidgeScorePredictor.
    - A QuarterSpecificModelSystem for quarter-level predictions with ensemble weighting.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import re

# --- Scikit-learn Imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# --- Imports ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost library not found...")

#try:
    #import lightgbm as lgb
    #LGBM_AVAILABLE = True
#except ImportError:
    #lgb = None
    #LGBM_AVAILABLE = False
    # Keep logger definition consistent with how it's handled for XGBoost
    #logging.warning("LightGBM library not found...")

# --- Configuration ---
# Determine Project Root relative to this file
SCRIPT_DIR = Path(__file__).resolve().parent # backend/nba_score_prediction
BACKEND_DIR = SCRIPT_DIR.parent             # backend
PROJECT_ROOT = BACKEND_DIR.parent           # project_root

# Use the same primary directory as train_models.py/prediction.py for the default
MODELS_BASE_DIR_DEFAULT = PROJECT_ROOT / 'models' / 'saved'
MODELS_BASE_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SEED = 42

# --- Utility Functions ---
def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """
    Calculate recency weights for samples based on dates.
    
    Args:
        dates: Pandas Series of dates.
        method: Weighting method ('half_life' or 'exponential').
        half_life: Half-life (in days) if using 'half_life'.
        decay_rate: Decay rate if using 'exponential'.
    
    Returns:
        A NumPy array of normalized weights matching the original Series index.
    """
    if not isinstance(dates, pd.Series) or dates.empty:
        logger.warning("Recency weights: input not a valid Series.")
        return None
    if not pd.api.types.is_datetime64_any_dtype(dates):
        try:
            dates = pd.to_datetime(dates, errors='coerce').dt.tz_localize(None)
        except Exception as e:
            logger.error(f"Recency weights: date conversion error: {e}")
            return None

    original_index = dates.index
    valid_dates = dates.dropna()
    if valid_dates.empty:
        logger.warning("Recency weights: no valid dates after dropna.")
        return np.ones(len(dates))
    sorted_dates = valid_dates.sort_values()
    latest_date = sorted_dates.max()
    if pd.isna(latest_date):
        logger.warning("Recency weights: no valid latest date found.")
        return np.ones(len(dates))

    days_from_latest = (latest_date - sorted_dates).dt.total_seconds() / (3600 * 24.0)
    days_from_latest = days_from_latest.astype(float)
    weights = np.ones(len(sorted_dates))
    if method == 'half_life':
        if half_life <= 0:
            logger.warning("Recency weights: Half-life must be positive. Using equal weights.")
        else:
            weights = 0.5 ** (days_from_latest / float(half_life))
    elif method == 'exponential':
        if not (0 < decay_rate < 1):
            logger.warning("Recency weights: Exponential decay_rate must be between 0 and 1. Using equal weights.")
        else:
            weights = decay_rate ** days_from_latest
    else:
        logger.warning(f"Recency weights: Unknown method: {method}. Using equal weights.")

    weights[~np.isfinite(weights)] = 0.0
    mean_weight = np.mean(weights) if len(weights) > 0 else 0.0
    if mean_weight > 1e-9:
        weights = weights / mean_weight
    else:
        logger.warning("Recency weights: Mean weight near zero. Using equal weights.")
        weights = np.ones(len(sorted_dates))
    weights_series = pd.Series(weights, index=valid_dates.index)
    return weights_series.reindex(original_index, fill_value=0.0).values

def _generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """Generate standardized rolling feature column name."""
    return f"{prefix}_rolling_{base}_{stat_type}_{window}"

def _preprocessing_pipeline() -> Pipeline:
    """
    Returns a common preprocessing pipeline used by all predictors:
    Imputation followed by Standard Scaling.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

# --- Base Predictor Class ---
class BaseScorePredictor:
    """
    Abstract base class for NBA score predictors.
    Provides common training, prediction, saving, and loading logic.
    """
    # Make model_dir Optional in signature and handle None
    def __init__(self, model_name: str, model_dir: Optional[Union[str, Path]] = MODELS_BASE_DIR_DEFAULT):
         # <<< FIX: Handle None input for model_dir >>>
         resolved_model_dir = model_dir if model_dir is not None else MODELS_BASE_DIR_DEFAULT
         # Ensure the directory exists, using the resolved path
         Path(resolved_model_dir).mkdir(parents=True, exist_ok=True)
         # Assign the Path object using the resolved path
         self.model_dir: Path = Path(resolved_model_dir)
         # <<< END FIX >>>

         self.model_name: str = model_name
         self.pipeline_home: Optional[Pipeline] = None
         self.pipeline_away: Optional[Pipeline] = None
         self.feature_names_in_: Optional[List[str]] = None
         self.training_timestamp: Optional[str] = None
         self.training_duration: Optional[float] = None

    def _build_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        raise NotImplementedError("Subclasses must implement _build_pipeline")

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement train")

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement predict")

    def save_model(self, filename: Optional[str] = None) -> str:
        if not self.pipeline_home or not self.pipeline_away:
            raise RuntimeError("Models must be trained before saving.")

        timestamp = self.training_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"{self.model_name}_{timestamp}.joblib"
        else:
            if not filename.endswith(".joblib"):
                filename += ".joblib"

        save_path = Path(self.model_dir) / filename
        # Create the directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.feature_names_in_ is None:
            logger.warning(f"Attempting to save model {self.model_name} but feature_names_in_ is not set.")

        model_data = {
            'pipeline_home': self.pipeline_home,
            'pipeline_away': self.pipeline_away,
            'feature_names_in_': self.feature_names_in_,
            'training_timestamp': self.training_timestamp,
            'training_duration': self.training_duration,
            'model_name': self.model_name,
            'model_class': self.__class__.__name__
        }

        try:
            joblib.dump(model_data, save_path)
            logger.info(f"{self.__class__.__name__} model saved successfully to {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}", exc_info=True)
            raise

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "BaseScorePredictor":
        """
        Loads the model pipelines and metadata from a file.
        """
        load_path_str = filepath
        if load_path_str is None:
            try:
                search_name = model_name or self.model_name
                model_dir_path = self.model_dir
                if not model_dir_path.is_dir():
                    raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

                # --- REPLACEMENT LOGIC ---
                files = []
                for path in model_dir_path.iterdir():
                    if path.is_file() and \
                       path.name.startswith(search_name) and \
                       path.name.endswith(".joblib"):
                        files.append(path)
                # --- END REPLACEMENT LOGIC ---

                if not files:
                    raise FileNotFoundError(f"No model file found starting with '{search_name}' in {self.model_dir}")

                # --- Keep the sorting logic the same ---
                def get_timestamp_from_filename(path: Path) -> datetime:
                    # (Your existing timestamp extraction logic - using regex or split)
                    try:
                        # Example using regex (adjust if needed)
                        match = re.search(r'_(\d{8}_\d{6})$', path.stem)
                        if match:
                            timestamp_str = match.group(1)
                            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        else:
                            return datetime.fromtimestamp(path.stat().st_mtime) # Fallback
                    except (IndexError, ValueError, AttributeError): # Added AttributeError just in case
                        return datetime.fromtimestamp(path.stat().st_mtime) # Fallback

                files.sort(key=get_timestamp_from_filename, reverse=True)
                load_path_str = str(files[0])
                logger.info(f"Loading latest model: {load_path_str}")
            except Exception as e:
                logger.error(f"Error finding latest model file: {e}", exc_info=True)
                raise

        load_path = Path(load_path_str)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model file not found at specified path: {load_path}")
        try:
            model_data = joblib.load(load_path)
            loaded_class_name = model_data.get('model_class')
            if loaded_class_name and loaded_class_name != self.__class__.__name__:
                raise TypeError(f"Loaded model class ({loaded_class_name}) does not match expected ({self.__class__.__name__}).")
            if 'pipeline_home' not in model_data or 'pipeline_away' not in model_data:
                raise ValueError("Loaded model file is missing required pipeline data.")
            if not isinstance(model_data['pipeline_home'], Pipeline) or not isinstance(model_data['pipeline_away'], Pipeline):
                raise TypeError("Loaded pipelines are not valid sklearn Pipeline objects.")
            required_features = model_data.get('feature_names_in_')
            if required_features is not None and not isinstance(required_features, list):
                logger.warning("Feature names in loaded model are not a list; attempting conversion.")
                try:
                    self.feature_names_in_ = list(required_features)
                except TypeError:
                    self.feature_names_in_ = None
            else:
                self.feature_names_in_ = required_features
            if self.feature_names_in_ is None:
                logger.error(f"CRITICAL: Loaded model {self.model_name} is missing the feature list ('feature_names_in_'). Prediction may fail.")
            self.pipeline_home = model_data['pipeline_home']
            self.pipeline_away = model_data['pipeline_away']
            self.training_timestamp = model_data.get('training_timestamp')
            self.training_duration = model_data.get('training_duration')
            self.model_name = model_data.get('model_name', self.model_name)
            logger.info(f"{self.__class__.__name__} model loaded successfully from {load_path}")
            return self  
        except Exception as e:
            logger.error(f"Error loading model from {load_path}: {e}", exc_info=True)
            raise


    def _common_train_logic(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
                              hyperparams_home: Optional[Dict[str, Any]],
                              hyperparams_away: Optional[Dict[str, Any]],
                              sample_weights: Optional[np.ndarray],
                              default_params: Dict[str, Any],
                              fit_params: Optional[Dict[str, Any]] = None,
                              eval_set_data: Optional[Tuple] = None) -> None:
        """
        Common logic for training home and away pipelines.
        Handles input preprocessing, pipeline construction, sample weights, and eval_set for XGBoost.
        """
        start_time = time.time()
        logger.debug(f"X_train shape at start: {X_train.shape}")
        try:
            X_train_numeric = X_train.select_dtypes(include=np.number)
            if X_train_numeric.shape[1] != X_train.shape[1]:
                dropped_cols = set(X_train.columns) - set(X_train_numeric.columns)
                logger.warning(f"Non-numeric columns dropped: {dropped_cols}. Using numeric subset.")
                X_train_for_fit = X_train_numeric
            else:
                X_train_for_fit = X_train
            self.feature_names_in_ = list(X_train_for_fit.columns)
            logger.info(f"Using {len(self.feature_names_in_)} features: {self.feature_names_in_[:15]}...")
        except Exception as e:
            logger.error(f"Error processing X_train columns: {e}", exc_info=True)
            self.feature_names_in_ = list(X_train.columns)
            X_train_for_fit = X_train

        if not self.feature_names_in_:
            logger.error("CRITICAL: No features for training. Aborting.")
            return

        logger.info(f"Starting training for {self.model_name}...")
        params_home = default_params.copy()
        params_away = default_params.copy()
        if hyperparams_home:
            params_home.update(hyperparams_home)
            logger.info("Custom hyperparameters applied for home model.")
        if hyperparams_away:
            params_away.update(hyperparams_away)
            logger.info("Custom hyperparameters applied for away model.")
        params_home.pop('normalize', None)
        params_away.pop('normalize', None)

        try:
            self.pipeline_home = self._build_pipeline(params_home)
            self.pipeline_away = self._build_pipeline(params_away)
        except Exception as build_e:
            logger.error(f"Pipeline build failed: {build_e}", exc_info=True)
            return

        fit_kwargs_home = fit_params.copy() if fit_params else {}
        fit_kwargs_away = fit_params.copy() if fit_params else {}

        if sample_weights is not None:
            if len(sample_weights) == len(X_train_for_fit):
                try:
                    final_estimator_name = self.pipeline_home.steps[-1][0]
                    weight_param_name = f"{final_estimator_name}__sample_weight"
                    fit_kwargs_home[weight_param_name] = sample_weights
                    fit_kwargs_away[weight_param_name] = sample_weights
                    logger.info(f"Sample weights applied with parameter: {weight_param_name}")
                except Exception as sw_e:
                    logger.error(f"Could not set sample weights: {sw_e}", exc_info=True)
            else:
                logger.warning(f"Sample weights length mismatch: {len(sample_weights)} vs {len(X_train_for_fit)}. Ignoring weights.")

        # Handle XGBoost eval_set if applicable
        is_xgb = False
        try:
            final_estimator_home = self.pipeline_home.steps[-1][1]
            if XGBOOST_AVAILABLE and isinstance(final_estimator_home, xgb.XGBRegressor):
                is_xgb = True
        except Exception:
            pass

        if is_xgb and eval_set_data:
            logger.info("Preparing eval_set for XGBoost...")
            if isinstance(eval_set_data, tuple) and len(eval_set_data) == 3:
                try:
                    X_val, y_val_home, y_val_away = eval_set_data
                    X_val_aligned = X_val[self.feature_names_in_]
                    y_val_home_aligned = y_val_home.loc[X_val_aligned.index]
                    y_val_away_aligned = y_val_away.loc[X_val_aligned.index]
                    final_estimator_name = self.pipeline_home.steps[-1][0]
                    fit_kwargs_home[f'{final_estimator_name}__eval_set'] = [(X_val_aligned.values, y_val_home_aligned.values)]
                    fit_kwargs_away[f'{final_estimator_name}__eval_set'] = [(X_val_aligned.values, y_val_away_aligned.values)]
                    logger.info(f"XGBoost eval_set applied using prefix '{final_estimator_name}__'.")
                except Exception as e:
                    logger.error(f"Eval set alignment error: {e}. Ignoring eval_set.", exc_info=True)
            else:
                logger.warning("Incorrect format for eval_set_data. Ignoring.")

        try:
            logger.info(f"Training home model with fit args: {list(fit_kwargs_home.keys())}")
            self.pipeline_home.fit(X_train_for_fit, y_train_home, **fit_kwargs_home)
            logger.info(f"Training away model with fit args: {list(fit_kwargs_away.keys())}")
            self.pipeline_away.fit(X_train_for_fit, y_train_away, **fit_kwargs_away)
        except Exception as e:
            logger.error(f"Error during model fitting: {e}", exc_info=True)
            raise

        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start_time
        logger.info(f"Training completed in {self.training_duration:.2f} seconds.")

    def _common_predict_logic(self, X: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Validates input features and returns predictions for home and away scores.
        """
        logger.debug(f"Prediction input shape: {X.shape}")
        if not self.pipeline_home or not self.pipeline_away:
            logger.error(f"Prediction failed for {self.model_name}: pipelines not available.")
            return None
        required_features = self.feature_names_in_
        if not required_features:
            logger.error(f"Prediction failed for {self.model_name}: missing feature list.")
            return None
        missing_cols = set(required_features) - set(X.columns)
        if missing_cols:
            logger.error(f"Input missing required features: {missing_cols}.")
            return None
        if set(X.columns) - set(required_features):
            logger.warning(f"Extra columns detected; selecting only required features.")
            try:
                X_predict = X[required_features]
            except Exception as slice_e:
                logger.error(f"Error slicing features: {slice_e}", exc_info=True)
                return None
        elif list(X.columns) != required_features:
            logger.warning("Reordering input features to match training order.")
            try:
                X_predict = X[required_features]
            except Exception as reorder_e:
                logger.error(f"Error reordering features: {reorder_e}", exc_info=True)
                return None
        else:
            X_predict = X

        logger.debug(f"Aligned X_predict shape: {X_predict.shape}")
        if X_predict.shape[1] != len(required_features):
            logger.error(f"Mismatch in feature count: {X_predict.shape[1]} vs {len(required_features)}.")
            return None

        logger.info(f"Predicting scores for {len(X_predict)} samples using {self.model_name}...")
        try:
            pred_home = self.pipeline_home.predict(X_predict)
            pred_away = self.pipeline_away.predict(X_predict)
            logger.info(f"Prediction finished for {self.model_name}.")
            return pred_home, pred_away
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return None


# --- Ridge Model Definition ---
class RidgeScorePredictor(BaseScorePredictor):
    """Ridge-based predictor with a preprocessing pipeline."""
    def __init__(self, model_dir: Union[str, Path] = MODELS_BASE_DIR_DEFAULT, model_name: str = "ridge_score_predictor"): # Corrected default model_name
        # --- CORRECTED LINE USING NAMED ARGUMENTS ---
        super().__init__(model_name=model_name, model_dir=model_dir)
        # --- END CORRECTION ---
        self._default_ridge_params: Dict[str, Any] = {
            'alpha': 1.0, 'fit_intercept': True, 'solver': 'auto', 'random_state': SEED
        }

    def _build_pipeline(self, ridge_params: Dict[str, Any]) -> Pipeline:
        final_params = self._default_ridge_params.copy()
        final_params.update(ridge_params)
        preprocessing = _preprocessing_pipeline()
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('ridge', Ridge(**final_params))
        ])
        try:
            pipeline.set_output(transform="pandas")
            logger.debug("Ridge pipeline set to output pandas DataFrame.")
        except AttributeError:
            logger.debug("Could not set pipeline output for Ridge (older sklearn).")
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None) -> None:
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_ridge_params)

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "RidgeScorePredictor":
        loaded_instance = super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
        return loaded_instance

    def _validate_loaded_model_type(self) -> None:
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], Ridge):
                raise TypeError("Loaded home pipeline is not a Ridge regressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], Ridge):
                raise TypeError("Loaded away pipeline is not a Ridge regressor.")
            logger.debug("Validated Ridge pipelines successfully.")
        except Exception as e:
            logger.error(f"Validation error for Ridge model: {e}", exc_info=True)
            raise
        
# class LightGBMScorePredictor(BaseScorePredictor):
#     """LightGBM-based predictor with a preprocessing pipeline."""
#     def __init__(self, model_dir: Union[str, Path] = MODELS_BASE_DIR_DEFAULT, model_name: str = "xgboost_score_predictor"):
#         super().__init__(model_dir, model_name)
#         self._default_lgbm_params: Dict[str, Any] = {
#              'objective': 'regression_l1', # MAE objective is often good for scores
#              'metric': 'mae',              # Use MAE for evaluation/early stopping if enabled
#              'n_estimators': 300,
#              'learning_rate': 0.05,
#              'num_leaves': 31,          # Default, adjust based on max_depth if needed
#              'max_depth': -1,           # Default (no limit unless constrained in tuning)
#              'feature_fraction': 0.8,   # Equivalent to colsample_bytree
#              'bagging_fraction': 0.8,   # Equivalent to subsample
#              'bagging_freq': 1,         # Perform bagging at every iteration
#              'reg_alpha': 0.1,          # L1 regularization
#              'reg_lambda': 0.1,         # L2 regularization
#              'min_child_samples': 20,   # Default minimum samples in a leaf
#              'random_state': SEED,      # For reproducibility
#              'n_jobs': -1,              # Use all available cores
#              'boosting_type': 'gbdt'    # Standard gradient boosting decision tree
#              # Add 'verbose': -1 to suppress LightGBM's internal verbosity during fit/predict
#          }
# 
#     def _build_pipeline(self, lgbm_params: Dict[str, Any]) -> Pipeline:
#          """Builds the scikit-learn pipeline for LightGBM."""
#          if not LGBM_AVAILABLE:
#              raise ImportError("LightGBM library required for LightGBMScorePredictor.")
# 
#          final_params = self._default_lgbm_params.copy()
#          final_params.update(lgbm_params)
#          # Remove verbose if it was passed during tuning to avoid conflict
#          final_params.pop('verbose', None)
# 
#          preprocessing = _preprocessing_pipeline() # Get the standard Imputer + Scaler
#          pipeline = Pipeline([
#              ('preprocessing', preprocessing),
#              # Use 'lgbm' prefix consistent with other models for hyperparameter tuning keys
#              ('lgbm', lgb.LGBMRegressor(**final_params, verbose=-1)) # Set verbose=-1 here
#          ])
#          try: # Set output format for scikit-learn >= 1.0
#              pipeline.set_output(transform="pandas")
#              logger.debug("LightGBM pipeline set to output pandas DataFrame.")
#          except AttributeError:
#              logger.debug("Could not set pipeline output for LightGBM (older scikit-learn version).")
#          return pipeline
# 
#     def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
#               hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
#               sample_weights: Optional[np.ndarray] = None,
#               # Accept fit_params which might contain eval_set from RandomizedSearchCV wrapper
#               fit_params: Optional[Dict[str, Any]] = None,
#               # Allow eval_set_data for consistency, though might not be directly used by common logic if fit_params handles it
#               eval_set_data: Optional[Tuple] = None) -> None:
#         """Trains the LightGBM model using the common training logic."""
#         if not LGBM_AVAILABLE:
#             raise ImportError("LightGBM library not found; cannot train LightGBMScorePredictor.")
# 
#         # Prepare fit parameters specifically for LGBM if needed (e.g., eval_set structure)
#         # Note: sample_weight is passed directly to _common_train_logic
#         # RandomizedSearchCV usually handles passing fit_params correctly if keys match
#         # We might need adjustments in _common_train_logic if LGBM requires fit_params keys
#         # different from XGBoost (e.g., 'lgbm__eval_set' vs 'xgb__eval_set')
# 
#         # For simplicity now, assume RandomizedSearchCV handles passing relevant fit_params correctly
#         # based on the pipeline structure ('lgbm__...')
# 
#         self._common_train_logic(X_train, y_train_home, y_train_away,
#                                  hyperparams_home, hyperparams_away, sample_weights,
#                                  self._default_lgbm_params,
#                                  fit_params=fit_params, # Pass along fit_params
#                                  eval_set_data=eval_set_data # Pass along eval_set_data
#                                  )
# 
# 
#     def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
#          """Generates predictions using the common prediction logic."""
#          predictions = self._common_predict_logic(X)
#          if predictions is None: return None
#          pred_home, pred_away = predictions
#          # Post-process predictions same as other models
#          pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
#          pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
#          return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)
# 
#     def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "LightGBMScorePredictor":
#          """Loads the model and validates its type."""
#          loaded_instance = super().load_model(filepath, model_name)
#          self._validate_loaded_model_type()
#          # Need to ignore type checking here as superclass returns BaseScorePredictor
#          return loaded_instance # type: ignore
# 
#     def _validate_loaded_model_type(self) -> None:
#         """Validates that the loaded model is indeed a LightGBM regressor."""
#         if not LGBM_AVAILABLE: return # Skip if library not found
#         try:
#             # Check the actual estimator instance in the final step of the pipeline
#             final_estimator_home = self.pipeline_home.steps[-1][1] if self.pipeline_home else None
#             final_estimator_away = self.pipeline_away.steps[-1][1] if self.pipeline_away else None
# 
#             if final_estimator_home and not isinstance(final_estimator_home, lgb.LGBMRegressor):
#                  raise TypeError(f"Loaded home pipeline's final step is not an LGBMRegressor.")
#             if final_estimator_away and not isinstance(final_estimator_away, lgb.LGBMRegressor):
#                  raise TypeError(f"Loaded away pipeline's final step is not an LGBMRegressor.")
#             logger.debug("Validated LightGBM pipelines successfully.")
#         except Exception as e:
#              logger.error(f"Validation error for LightGBM model: {e}", exc_info=True)
#              raise # Re-raise the error after logging

# --- SVR Model Definition ---
# --- Corrected SVR Model Definition ---
class SVRScorePredictor(BaseScorePredictor):
    """SVR-based predictor with a preprocessing pipeline."""
    def __init__(self, model_dir: Union[str, Path] = MODELS_BASE_DIR_DEFAULT, model_name: str = "svr_score_predictor"):
         super().__init__(model_dir, model_name)
         # <<< FIX: Define the _default_svr_params attribute >>>
         self._default_svr_params: Dict[str, Any] = {
             'kernel': 'rbf',      # Default kernel
             'C': 1.0,             # Default regularization
             'gamma': 'scale',     # Default gamma for rbf
             'epsilon': 0.1,       # Default epsilon
             # 'cache_size': 500   # Optional: Can increase if needed
             # No 'random_state' for SVR itself
         }
         # <<< END FIX >>>

    def _build_pipeline(self, svr_params: Dict[str, Any]) -> Pipeline:
         """Builds the scikit-learn pipeline for SVR."""
         final_params = self._default_svr_params.copy() # Now uses the defined attribute
         final_params.update(svr_params)

         preprocessing = _preprocessing_pipeline() # Get the standard Imputer + Scaler

         pipeline = Pipeline([
             ('preprocessing', preprocessing),
             ('svr', SVR(**final_params)) # Use 'svr' prefix
         ])
         try: # Set output format for scikit-learn >= 1.0
             pipeline.set_output(transform="pandas")
             logger.debug("SVR pipeline set to output pandas DataFrame.")
         except AttributeError:
             logger.debug("Could not set pipeline output for SVR (older scikit-learn version).")
         return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None,
              fit_params: Optional[Dict[str, Any]] = None, # Keep signature consistent
              eval_set_data: Optional[Tuple] = None) -> None: # Keep signature consistent
         """Trains the SVR model using the common training logic."""
         # Pass fit_params down; _common_train_logic will extract sample_weight if key matches
         self._common_train_logic(X_train, y_train_home, y_train_away,
                                  hyperparams_home, hyperparams_away, sample_weights,
                                  self._default_svr_params, # Now uses the defined attribute
                                  fit_params=fit_params,
                                  eval_set_data=eval_set_data) # Pass along, though SVR won't use eval_set


    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
         """Generates predictions using the common prediction logic."""
         predictions = self._common_predict_logic(X)
         if predictions is None: return None
         pred_home, pred_away = predictions
         # Post-process SVR predictions (same as others)
         pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
         pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
         return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "SVRScorePredictor":
         """Loads the model and validates its type."""
         loaded_instance = super().load_model(filepath, model_name)
         self._validate_loaded_model_type()
         return loaded_instance # type: ignore

    def _validate_loaded_model_type(self) -> None:
        """Validates that the loaded model is indeed an SVR."""
        try:
            final_estimator_home = self.pipeline_home.steps[-1][1] if self.pipeline_home else None
            final_estimator_away = self.pipeline_away.steps[-1][1] if self.pipeline_away else None

            if final_estimator_home and not isinstance(final_estimator_home, SVR):
                 raise TypeError(f"Loaded home pipeline's final step is not an SVR.")
            if final_estimator_away and not isinstance(final_estimator_away, SVR):
                 raise TypeError(f"Loaded away pipeline's final step is not an SVR.")
            logger.debug("Validated SVR pipelines successfully.")
        except Exception as e:
             logger.error(f"Validation error for SVR model: {e}", exc_info=True)
             raise


# --- Quarter-Specific Model System ---
class QuarterSpecificModelSystem:
    """
    Manages quarter-specific models for in-game predictions.
    Supports loading primary models, fallback models (Ridge-based), and caching of predictions.
    """
    def __init__(self, feature_generator: Optional[Any] = None, debug: bool = False):
        self.feature_generator = feature_generator
        self.debug = debug
        self.models: Dict[int, Any] = {}       # Primary quarter-specific models
        self.fallback_models: Dict[int, Any] = {}  # Fallback Ridge models
        self.quarter_feature_sets: Dict[int, List[str]] = self._get_consolidated_feature_sets()
        self.error_history: Dict[str, Dict[int, float]] = {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.prediction_cache: Dict[str, float] = {}
        self._create_fallback_models()
        self.log("QuarterSpecificModelSystem initialized for prediction.", level="DEBUG")

    def log(self, message: str, level: str = "INFO") -> None:
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[QuarterSystem] {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        base_sets: Dict[int, List[str]] = {}
        if self.feature_generator:
            try:
                if hasattr(self.feature_generator, 'get_prediction_feature_sets'):
                    base_sets = self.feature_generator.get_prediction_feature_sets()
                elif hasattr(self.feature_generator, '_get_optimized_feature_sets'):
                    base_sets = self.feature_generator._get_optimized_feature_sets()
                elif hasattr(self.feature_generator, 'feature_sets'):
                    base_sets = self.feature_generator.feature_sets
                if not isinstance(base_sets, dict):
                    self.log(f"Feature set attribute not a dict ({type(base_sets)}). Using empty sets.", level="WARNING")
                    base_sets = {}
            except Exception as e:
                self.log(f"Error retrieving feature sets: {e}. Using empty sets.", level="ERROR")
                base_sets = {}
        final_sets = {q: base_sets.get(q, []) for q in range(1, 5)}
        self.log(f"Consolidated feature sets: Q1({len(final_sets.get(1, []))}), Q2({len(final_sets.get(2, []))}), Q3({len(final_sets.get(3, []))}), Q4({len(final_sets.get(4, []))})", level="DEBUG")
        return final_sets

    def _create_fallback_models(self) -> None:
        self.log("Ensuring fallback Ridge models are available...", level="DEBUG")
        for quarter in range(1, 5):
            if quarter not in self.fallback_models:
                try:
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=1.0, random_state=42))
                    ])
                    self.fallback_models[quarter] = model
                    self.log(f"Fallback Ridge model created for Q{quarter}.", level="DEBUG")
                except Exception as e:
                    self.log(f"Failed to create fallback model for Q{quarter}: {e}", level="ERROR")

    def predict_quarter(self, X: pd.DataFrame, quarter: int) -> float:
        """
        Predicts a single-quarter home score using the appropriate model.
        Expects X to be a single-row DataFrame.
        """
        if not (1 <= quarter <= 4):
            self.log(f"Invalid quarter {quarter}.", level="ERROR")
            return 0.0
        if X is None or X.empty or X.shape[0] != 1:
            self.log(f"predict_quarter expects a single-row DataFrame. Got shape: {X.shape if X is not None else 'None'}.", level="ERROR")
            return 0.0

        features_expected = self.quarter_feature_sets.get(quarter, [])
        if not features_expected:
            self.log(f"No expected features for Q{quarter}. Using default score.", level="WARNING")
            return 25.0

        features_available = [f for f in features_expected if f in X.columns]
        if not features_available:
            default_score = 25.0
            if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
            self.log(f"Expected Q{quarter} features not in input. Using default score: {default_score}", level="WARNING")
            return default_score

        X_predict_qtr = X[features_available]

        # Generate a cache key using a simple hash of the row's values.
        cache_key = None
        try:
            key_tuple = tuple(X_predict_qtr.iloc[0].fillna(0).values)
            cache_key = f"q{quarter}_{hash(key_tuple)}"
            if cache_key in self.prediction_cache:
                self.log(f"Using cached prediction for Q{quarter}.", level="DEBUG")
                return self.prediction_cache[cache_key]
        except Exception as e:
            self.log(f"Cache key generation failed for Q{quarter}: {e}.", level="WARNING")
            cache_key = None

        model_to_use = self.models.get(quarter) or self.fallback_models.get(quarter)
        model_source = "primary" if quarter in self.models and self.models[quarter] is not None else "fallback"
        if model_to_use is None:
            default_score = 25.0
            if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
            self.log(f"No model for Q{quarter}. Using default score: {default_score}", level="ERROR")
            if cache_key:
                self.prediction_cache[cache_key] = default_score
            return default_score

        try:
            # Optionally validate that input features match the model's expected features if stored.
            if hasattr(model_to_use, 'steps'):
                model_instance = model_to_use.steps[-1][1]
                if hasattr(model_instance, 'feature_names_in_') and model_instance.feature_names_in_ is not None:
                    required_model_features = list(model_instance.feature_names_in_)
                    if set(required_model_features).issubset(set(features_available)):
                        X_predict_qtr = X_predict_qtr[required_model_features]
                    else:
                        missing_req = set(required_model_features) - set(features_available)
                        self.log(f"Missing features for loaded model: {missing_req}. Using default.", level="ERROR")
                        default_score = 25.0
                        if cache_key:
                            self.prediction_cache[cache_key] = default_score
                        return default_score
                else:
                    self.log(f"Model for Q{quarter} lacks stored feature names. Proceeding with available features.", level="DEBUG")
            else:
                self.log(f"Model for Q{quarter} is not a Pipeline instance.", level="WARNING")
        except Exception as feat_e:
            self.log(f"Error during feature validation for Q{quarter}: {feat_e}", level="ERROR")
            default_score = 25.0
            if cache_key:
                self.prediction_cache[cache_key] = default_score
            return default_score

        self.log(f"Predicting Q{quarter} using {model_source} model with {X_predict_qtr.shape[1]} features.", level="DEBUG")
        try:
            prediction = model_to_use.predict(X_predict_qtr)[0]
        except Exception as e:
            self.log(f"Error during Q{quarter} prediction with {model_source} model: {e}", level="ERROR")
            if model_source == "primary":
                self.log(f"Attempting fallback prediction for Q{quarter}...", level="WARNING")
                fallback_model = self.fallback_models.get(quarter)
                if fallback_model:
                    try:
                        fb_input = X[features_available].fillna(0)
                        prediction = fallback_model.predict(fb_input)[0]
                        model_source = "fallback (after primary error)"
                    except Exception as fb_e:
                        self.log(f"Fallback prediction error for Q{quarter}: {fb_e}", level="ERROR")
                        prediction = None
            else:
                prediction = None

        default_score = 25.0
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
            default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
        final_prediction = default_score
        if prediction is not None:
            try:
                final_prediction = max(0.0, float(prediction))
            except Exception:
                self.log(f"Non-numeric prediction for Q{quarter}. Using default.", level="ERROR")
                final_prediction = default_score
        else:
            self.log(f"Prediction failed for Q{quarter}. Using default: {final_prediction}", level="ERROR")
            model_source += " -> default"
        self.log(f"Q{quarter} Prediction ({model_source}): {final_prediction:.3f}", level="DEBUG")
        if cache_key:
            self.prediction_cache[cache_key] = final_prediction
        return final_prediction

    def predict_remaining_quarters(self, game_data: Dict, current_quarter: int) -> Dict[str, float]:
        """
        Iteratively predicts remaining quarters (Q(current+1) to Q4) by updating game state.
        """
        if not isinstance(game_data, dict):
            self.log("game_data must be a dict.", level="ERROR")
            return {}
        if not isinstance(current_quarter, int) or current_quarter < 0:
            self.log(f"Invalid current_quarter: {current_quarter}.", level="ERROR")
            return {}
        if current_quarter >= 4:
            self.log("No remaining quarters to predict.", level="DEBUG")
            return {}
        results = {}
        current_state = game_data.copy()
        try:
            X = pd.DataFrame([current_state])
            if self.feature_generator:
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X = self.feature_generator.generate_features_for_prediction(X)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X = self.feature_generator.integrate_advanced_features(X)
                else:
                    self.log("Feature generator missing expected methods.", level="ERROR")
                    return {}
            else:
                self.log("Feature generator not provided.", level="ERROR")
                return {}
            if X.empty:
                raise ValueError("Feature generation returned empty DataFrame.")
        except Exception as e:
            self.log(f"Error generating features for remaining quarters: {e}", level="ERROR", )
            return {}
        for q in range(current_quarter + 1, 5):
            try:
                q_pred = self.predict_quarter(X.iloc[[0]], q)
                results[f'q{q}'] = q_pred
                current_state[f'home_q{q}'] = q_pred
                self.log(f"Predicted Q{q} = {q_pred:.2f}. Regenerating features...", level="DEBUG")
                X_next = pd.DataFrame([current_state])
                if self.feature_generator:
                    if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                        X = self.feature_generator.generate_features_for_prediction(X_next)
                    elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                        X = self.feature_generator.integrate_advanced_features(X_next)
                    else:
                        self.log("Feature generator lacks methods; cannot regenerate features.", level="WARNING")
                        if f'home_q{q}' not in X.columns:
                            X = X.assign(**{f'home_q{q}': q_pred})
                        else:
                            X.loc[X.index[0], f'home_q{q}'] = q_pred
                else:
                    if f'home_q{q}' not in X.columns:
                        X = X.assign(**{f'home_q{q}': q_pred})
                    else:
                        X.loc[X.index[0], f'home_q{q}'] = q_pred
                if X.empty:
                    raise ValueError(f"Feature regeneration failed after Q{q}.")
            except Exception as e:
                self.log(f"Error predicting/updating features for Q{q}: {e}", level="ERROR", )
                default_q = 25.0
                if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                    default_q = self.feature_generator.league_averages.get('quarter_scores', {}).get(q, 25.0)
                results[f'q{q}'] = default_q
                current_state[f'home_q{q}'] = default_q
                try:
                    X_next = pd.DataFrame([current_state])
                    if self.feature_generator:
                        if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                            X = self.feature_generator.generate_features_for_prediction(X_next)
                        elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                            X = self.feature_generator.integrate_advanced_features(X_next)
                        else:
                            X = None
                    else:
                        X = None
                    if X is None or X.empty:
                        self.log(f"Feature regeneration failed after Q{q} error. Stopping predictions.", level="ERROR")
                        break
                except Exception as regen_e:
                    self.log(f"Error regenerating features after Q{q} error: {regen_e}. Stopping predictions.", level="ERROR")
                    break
        self.log(f"Remaining quarters predictions: {results}", level="DEBUG")
        return results

    def predict_final_score(self, game_data: Dict, main_model_prediction: Optional[float] = None,
                            weight_manager: Optional[Any] = None) -> Tuple[float, float, Dict]:
        """
        Combines the main model prediction and quarter-specific predictions using an ensemble weight manager
        to produce a final total score prediction.
        """
        self.log("Predicting final score using ensemble weighting...", level="DEBUG")
        league_avg_score = 110.0
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
            league_avg_score = self.feature_generator.league_averages.get('score', 110.0)
        fallback_pred = main_model_prediction if main_model_prediction is not None else league_avg_score
        try:
            fallback_pred = float(fallback_pred)
        except Exception:
            fallback_pred = league_avg_score

        if not isinstance(game_data, dict):
            self.log("game_data must be a dict.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Invalid game_data type'}
        if weight_manager is None or not hasattr(weight_manager, 'calculate_ensemble'):
            self.log("WeightManager instance with calculate_ensemble method is required.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Missing WeightManager'}

        try:
            current_quarter = int(game_data.get('current_quarter', 0))
        except Exception:
            self.log("Invalid current_quarter value; defaulting to 0.", level="WARNING")
            current_quarter = 0

        home_score_so_far = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        away_score_so_far = sum([float(game_data.get(f'away_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        if main_model_prediction is None:
            if current_quarter > 0 and (home_score_so_far > 0 or away_score_so_far > 0):
                main_model_prediction = (home_score_so_far + away_score_so_far) * (4.0 / max(1, current_quarter))
                main_model_prediction = main_model_prediction / 2
                self.log(f"Extrapolated main model prediction: ~{main_model_prediction:.2f}", level="WARNING")
            else:
                main_model_prediction = league_avg_score / 2
                self.log(f"Using league average for main model prediction: ~{main_model_prediction:.2f}", level="WARNING")
        try:
            main_model_prediction = float(main_model_prediction)
        except Exception:
            main_model_prediction = league_avg_score / 2

        if current_quarter <= 0:
            self.log("Pregame prediction. Using main model prediction only.", level="DEBUG")
            confidence = 0.5
            breakdown = {
                'main_model_pred': main_model_prediction,
                'quarter_model_sum': 0,
                'weights': {'main': 1.0, 'quarter': 0.0},
                'quarter_predictions': {},
                'current_score_home': 0,
                'current_score_away': 0,
                'score_differential': 0,
                'momentum': 0,
                'current_quarter': 0,
                'time_remaining_minutes_est': 48.0
            }
            final_pred_home = main_model_prediction
            final_pred_away = main_model_prediction
            final_total = final_pred_home + final_pred_away
            return final_total, confidence, breakdown

        remaining_quarters_pred = self.predict_remaining_quarters(game_data, current_quarter)
        if not isinstance(remaining_quarters_pred, dict):
            self.log("Remaining quarters prediction failed. Using fallback.", level="ERROR")
            main_total = main_model_prediction * 2
            return main_total, 0.3, {'error': 'Failed to predict remaining quarters', 'main_model': main_total}

        predicted_remaining_home = sum(remaining_quarters_pred.values())
        quarter_sum_home = home_score_so_far + predicted_remaining_home
        score_diff = home_score_so_far - away_score_so_far
        momentum = 0.0
        time_remaining = max(0.0, 12.0 * (4.0 - current_quarter))
        try:
            X_context = pd.DataFrame([game_data])
            if self.feature_generator:
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X_context = self.feature_generator.generate_features_for_prediction(X_context)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X_context = self.feature_generator.integrate_advanced_features(X_context)
                if not X_context.empty:
                    if 'score_differential' in X_context.columns and pd.notna(X_context['score_differential'].iloc[0]):
                        score_diff = float(X_context['score_differential'].iloc[0])
                    if 'cumulative_momentum' in X_context.columns and pd.notna(X_context['cumulative_momentum'].iloc[0]):
                        momentum = float(X_context['cumulative_momentum'].iloc[0])
                    if 'time_remaining_seconds' in X_context.columns and pd.notna(X_context['time_remaining_seconds'].iloc[0]):
                        time_remaining = max(0.0, float(X_context['time_remaining_seconds'].iloc[0]) / 60.0)
            else:
                self.log("Feature generator missing. Using basic context estimates.", level="WARNING")
        except Exception as e:
            self.log(f"Error extracting context features: {e}. Using defaults.", level="WARNING")
        try:
            ensemble_pred_home, weight_main, weight_quarter = weight_manager.calculate_ensemble(
                main_prediction=main_model_prediction,
                quarter_prediction=quarter_sum_home,
                current_quarter=current_quarter,
                score_differential=score_diff,
                momentum=momentum,
                time_remaining=time_remaining
            )
            confidence = 0.6 + 0.4 * weight_main
        except Exception as e:
            self.log(f"Error in ensemble weighting: {e}. Using fallback.", level="ERROR", exc_info=True)
            ensemble_pred_home = main_model_prediction
            confidence = 0.3
            weight_main = 1.0
            weight_quarter = 0.0

        try:
            final_ensemble_home = float(ensemble_pred_home)
            final_confidence = float(confidence)
            avg_diff = 2.5
            final_ensemble_away = final_ensemble_home - avg_diff
            final_total = final_ensemble_home + final_ensemble_away
            breakdown = {
                'main_model_pred': main_model_prediction,
                'quarter_model_sum': quarter_sum_home,
                'weights': {'main': weight_main, 'quarter': weight_quarter},
                'quarter_predictions': remaining_quarters_pred,
                'current_score_home': home_score_so_far,
                'current_score_away': away_score_so_far,
                'score_differential': score_diff,
                'momentum': momentum,
                'current_quarter': current_quarter,
                'time_remaining_minutes_est': time_remaining
            }
            self.log(f"Final Ensemble: Home {final_ensemble_home:.2f}, Away {final_ensemble_away:.2f}, Total {final_total:.2f} (Confidence: {final_confidence:.2f})", level="DEBUG")
            return final_total, final_confidence, breakdown
        except Exception as final_e:
            self.log(f"Error finalizing ensemble prediction: {final_e}. Using fallback.", level="ERROR")
            main_total = main_model_prediction * 2
            return main_total, 0.0, {'error': 'Non-numeric ensemble result'}