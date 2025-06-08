# backend/mlb_score_prediction/models.py

"""
models.py - MLB Score Prediction Models Module
...
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import joblib

# --- Scikit-learn Imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler

# Get logger for this module
logger = logging.getLogger(__name__) # Define logger once, early

# --- Configuration for Model Directory ---
# Define MODELS_DIR consistently based on project root
SCRIPT_PATH_MODELS = Path(__file__).resolve()
PROJECT_ROOT_MODELS = SCRIPT_PATH_MODELS.parents[2]
SHARED_MODELS_BASE_DIR = PROJECT_ROOT_MODELS / "models" / "saved" # Your preferred path

# Ensure the directory exists
SHARED_MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Log the path being used *after* it has been defined
logger.info(f"MLB models.py will use model directory: {SHARED_MODELS_BASE_DIR}")

# --- Logger Configuration ---
# (Logger setup can be here or handled globally in your app)
# Ensure logger is configured before first use if not done globally
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

SEED = 42
NUM_INNINGS_REGULATION = 9 # Standard number of innings in an MLB game


# --- Utility Functions ---
def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """
    Calculate recency weights for samples based on dates.
    Applicable to MLB; half_life/decay_rate might need tuning for MLB season/game dynamics.
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
        return np.ones(len(dates))  # Return array of ones with original length

    sorted_dates = valid_dates.sort_values()
    latest_date = sorted_dates.max()
    if pd.isna(latest_date):
        logger.warning("Recency weights: no valid latest date found.")
        return np.ones(len(dates))  # Return array of ones with original length

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
    if mean_weight > 1e-9:  # Avoid division by zero if all weights are tiny
        weights = weights / mean_weight
    else:
        logger.warning("Recency weights: Mean weight near zero. Using equal weights for sorted dates.")
        weights = np.ones(len(sorted_dates))

    weights_series = pd.Series(weights, index=sorted_dates.index)  # Use sorted_dates' index
    # Reindex to original_index, fill missing (originally NaN dates) with 0 weight, or 1 if all were NaNs
    fill_val = 0.0 if not valid_dates.empty else 1.0
    return weights_series.reindex(original_index, fill_value=fill_val).values

def _generate_rolling_column_name(prefix: str, base: str, stat_type: str, window: int) -> str:
    """Generate standardized rolling feature column name."""
    return f"{prefix}_rolling_{base}_{stat_type}_{window}"

def _preprocessing_pipeline(numeric_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            # === THE FINAL FIX: Use the median to fill missing values ===
            # This is a standard, robust method that avoids creating outliers.
            ("imputer", SimpleImputer(strategy="median")),
            
            ("scaler",  RobustScaler(quantile_range=(25.0, 75.0))),
            
            # This second imputer is still a good safety measure to catch
            # any NaNs created by the scaler for zero-variance columns.
            ("post_scale_imputer", SimpleImputer(strategy="constant", fill_value=0.0))
        ]
    )
# --- Base Predictor Class ---
class BaseScorePredictor:
    """
    Abstract base class for score predictors (MLB or other sports).
    Provides common training, prediction, saving, and loading logic for predicting
    home and away team final scores/runs.
    Models will be saved in the SHARED_MODELS_BASE_DIR, differentiated by model_name.
    """
    def __init__(self, model_name: str, model_dir: Optional[Union[str, Path]] = SHARED_MODELS_BASE_DIR):
         resolved_model_dir = model_dir if model_dir is not None else SHARED_MODELS_BASE_DIR
         Path(resolved_model_dir).mkdir(parents=True, exist_ok=True)
         self.model_dir: Path = Path(resolved_model_dir)

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

        # Determine the final save path using a fixed name based on self.model_name
        if filename is None:
            # Use a non-timestamped, fixed filename for this model_name
            final_filename = f"{self.model_name}.joblib"
        else:
            # If a specific filename is provided externally, use that
            final_filename = filename
            if not final_filename.endswith(".joblib"):
                final_filename += ".joblib"
        
        final_save_path = self.model_dir / final_filename
        final_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Define a temporary save path
        # Suffix with a unique timestamp to avoid collision if multiple processes try to save simultaneously
        temp_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        temp_save_path = self.model_dir / f"{self.model_name}.joblib_temp_{temp_id}"

        if self.feature_names_in_ is None:
            logger.warning(f"Attempting to save model {self.model_name} but feature_names_in_ is not set.")

        model_data = {
            'pipeline_home': self.pipeline_home,
            'pipeline_away': self.pipeline_away,
            'feature_names_in_': self.feature_names_in_,
            'training_timestamp': self.training_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_duration': self.training_duration,
            'model_name': self.model_name,
            'model_class': self.__class__.__name__
        }

        try:
            # 1. Save to the temporary file
            joblib.dump(model_data, temp_save_path)
            
            # 2. If successful, atomically replace the final destination file
            temp_save_path.replace(final_save_path)
            
            logger.info(f"{self.__class__.__name__} model saved successfully to {final_save_path} (overwriting previous).")
            return str(final_save_path)
        except Exception as e:
            logger.error(f"Error saving model to {final_save_path} (via temp file {temp_save_path}): {e}", exc_info=True)
            # Clean up the temporary file if it exists and an error occurred
            if temp_save_path.exists():
                try:
                    temp_save_path.unlink()
                except OSError:
                    logger.warning(f"Could not remove temporary save file {temp_save_path} after error.")
            raise


    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "BaseScorePredictor":
        load_path_str: Optional[str] = filepath
        effective_model_name = model_name or self.model_name
        load_path: Path  # Declare load_path here

        if load_path_str is None:
            # Primary target: non-timestamped file based on the effective_model_name
            target_filename = f"{effective_model_name}.joblib"
            load_path = self.model_dir / target_filename
            if load_path.is_file():
                load_path_str = str(load_path)
                logger.info(f"Loading model for '{effective_model_name}' from fixed path: {load_path_str}")
            else:
                logger.error(f"Model file {load_path} not found for model name '{effective_model_name}'.")
                raise FileNotFoundError(f"Model file {target_filename} not found in {self.model_dir} for {effective_model_name}.")
        else:
            load_path = Path(load_path_str)
            if not load_path.is_file():
                raise FileNotFoundError(f"Specified model file not found at path: {load_path}")

        try:
            model_data = joblib.load(load_path)
            loaded_class_name = model_data.get('model_class')
            if loaded_class_name and loaded_class_name != self.__class__.__name__:
                raise TypeError(f"Loaded model class ('{loaded_class_name}') does not match expected ('{self.__class__.__name__}').")
            
            if 'pipeline_home' not in model_data or 'pipeline_away' not in model_data:
                raise ValueError("Loaded model file is missing required pipeline data ('pipeline_home' or 'pipeline_away').")
            if not isinstance(model_data['pipeline_home'], Pipeline) or not isinstance(model_data['pipeline_away'], Pipeline):
                raise TypeError("Loaded pipelines are not valid scikit-learn Pipeline objects.")
            
            required_features = model_data.get('feature_names_in_')
            if required_features is not None and not isinstance(required_features, list):
                logger.warning("Feature names in loaded model ('feature_names_in_') are not a list; attempting conversion.")
                try:
                    self.feature_names_in_ = list(required_features)
                except TypeError:
                    logger.error("Failed to convert 'feature_names_in_' to list. Setting to None.")
                    self.feature_names_in_ = None
            else:
                self.feature_names_in_ = required_features

            if self.feature_names_in_ is None:
                logger.error(f"CRITICAL: Loaded model '{self.model_name}' is missing the feature list ('feature_names_in_'). Prediction integrity is at risk.")
            
            self.pipeline_home = model_data['pipeline_home']
            self.pipeline_away = model_data['pipeline_away']
            self.training_timestamp = model_data.get('training_timestamp')
            self.training_duration = model_data.get('training_duration')
            self.model_name = model_data.get('model_name', effective_model_name)
            logger.info(f"{self.__class__.__name__} model ('{self.model_name}') loaded successfully from {load_path}")
            return self
        except FileNotFoundError:
            logger.error(f"Model file not found at {load_path} during joblib.load attempt.", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading model data from {load_path}: {e}", exc_info=True)
            raise


    def _common_train_logic(self,
                            X_train: pd.DataFrame,
                            y_train_home: pd.Series,
                            y_train_away: pd.Series,
                            hyperparams_home: Optional[Dict[str, Any]],
                            hyperparams_away: Optional[Dict[str, Any]],
                            sample_weights: Optional[np.ndarray],
                            default_params: Dict[str, Any],
                            fit_params: Optional[Dict[str, Any]] = None,
                            eval_set_data: Optional[Tuple] = None) -> None:
        start_time = time.time()
        logger.debug(f"X_train shape at start: {X_train.shape}")
        
        # === FIX START: Set feature names BEFORE building the pipeline ===
        try:
            # Drop non-numeric columns and store the final feature list
            X_train_for_fit = X_train.select_dtypes(include=np.number)
            if X_train_for_fit.shape[1] != X_train.shape[1]:
                dropped_cols = set(X_train.columns) - set(X_train_for_fit.columns)
                logger.warning(f"Non-numeric columns dropped: {dropped_cols}. Using numeric subset.")
            
            self.feature_names_in_ = list(X_train_for_fit.columns)
            logger.info(f"Using {len(self.feature_names_in_)} features: {self.feature_names_in_[:10]}...")

            if not self.feature_names_in_:
                logger.error("CRITICAL: No features for training. Aborting.")
                return

        except Exception as e:
            logger.error(f"Error processing X_train columns: {e}", exc_info=True)
            self.feature_names_in_ = list(X_train.columns) # Fallback
            X_train_for_fit = X_train
        # === FIX END ===

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
            # Now, self.feature_names_in_ is available to _build_pipeline
            self.pipeline_home = self._build_pipeline(params_home)
            self.pipeline_away = self._build_pipeline(params_away)
        except Exception as build_e:
            logger.error(f"Pipeline build failed: {build_e}", exc_info=True)
            return

        fit_kwargs_home = fit_params.copy() if fit_params else {}
        fit_kwargs_away = fit_params.copy() if fit_params else {}

        if eval_set_data is not None and 'xgb' in self.pipeline_home.named_steps:  # XGB specific
            X_val_raw, y_val_home, y_val_away = eval_set_data
            preprocessing = self.pipeline_home.named_steps.get('preprocessing')
            if preprocessing is None:
                raise RuntimeError("Cannot find 'preprocessing' step in pipeline for eval_set transformation")

            # Fit preprocessing on training data if not already done (though build_pipeline might do it)
            if not hasattr(preprocessing, "mean_"):  # A simple check if scaler is fitted
                preprocessing.fit(X_train_for_fit)

            X_val_transformed = preprocessing.transform(X_val_raw[self.feature_names_in_])
            if not hasattr(X_val_transformed, 'columns'):
                X_val_transformed = pd.DataFrame(
                    X_val_transformed,
                    columns=self.feature_names_in_,  # Use consistent feature names
                    index=X_val_raw.index
                )

            fit_kwargs_home['xgb__eval_set'] = [(X_val_transformed, y_val_home)]
            fit_kwargs_home['xgb__early_stopping_rounds'] = 30  # Example, make configurable
            fit_kwargs_home['xgb__eval_metric'] = 'rmse'  # Example

            fit_kwargs_away['xgb__eval_set'] = [(X_val_transformed, y_val_away)]
            fit_kwargs_away['xgb__early_stopping_rounds'] = 30
            fit_kwargs_away['xgb__eval_metric'] = 'rmse'

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
        logger.debug(f"[{self.model_name}] Prediction input shape: {X.shape}")
        if not self.pipeline_home or not self.pipeline_away:
            logger.error(f"Prediction failed for {self.model_name}: pipelines not available.")
            return None

        required_features = self.feature_names_in_
        if not required_features:
            logger.error(f"Prediction failed for {self.model_name}: missing feature list (feature_names_in_).")
            return None

        # --- Start of new diagnostic section ---
        X_predict = X.copy()
        
        missing_cols = set(required_features) - set(X_predict.columns)
        if missing_cols:
            logger.error(f"[{self.model_name}] Input missing required features: {missing_cols}. Cannot predict.")
            return None

        # Ensure column order matches training order
        if list(X_predict.columns) != required_features:
            logger.warning(f"[{self.model_name}] Reordering input features to match training order.")
            X_predict = X_predict[required_features]

        # 1. Log stats of the raw input data
        logger.info(f"--- [{self.model_name}] DIAGNOSTICS: RAW INPUT DATA (Top 5 Rows) ---")
        logger.info("\n" + X_predict.head().to_string())
        
        # 2. Log stats AFTER the full preprocessing pipeline
        try:
            preprocessor = self.pipeline_home.named_steps['preprocessing']
            X_transformed = preprocessor.transform(X_predict)
            if not isinstance(X_transformed, pd.DataFrame):
                 X_transformed = pd.DataFrame(X_transformed, index=X_predict.index, columns=X_predict.columns)

            logger.info(f"--- [{self.model_name}] DIAGNOSTICS: POST-PREPROCESSING DATA (Top 5 Rows) ---")
            logger.info("\n" + X_transformed.head().to_string())
            
            nan_counts = X_transformed.isnull().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"[{self.model_name}] WARNING: NaNs detected AFTER preprocessing step. Columns with NaNs:\n{nan_counts[nan_counts > 0]}")

        except Exception as e:
            logger.error(f"[{self.model_name}] Error during diagnostic transform step: {e}")
        # --- End of new diagnostic section ---


        logger.info(f"Predicting scores for {len(X_predict)} samples using {self.model_name}...")
        try:
            pred_home = self.pipeline_home.predict(X_predict)
            pred_away = self.pipeline_away.predict(X_predict)
            
            # 3. Log the raw prediction values BEFORE the np.maximum(0, ...) step
            logger.info(f"--- [{self.model_name}] DIAGNOSTICS: RAW PREDICTION OUTPUT ---")
            logger.info(f"Raw Home Preds: {pred_home[:5]}")
            logger.info(f"Raw Away Preds: {pred_away[:5]}")
            
            logger.info(f"Prediction finished for {self.model_name}.")
            return pred_home, pred_away
        except Exception as e:
            logger.error(f"Error during prediction for {self.model_name}: {e}", exc_info=True)
            return None

# --- Ridge Model Definition ---
class RidgeScorePredictor(BaseScorePredictor):
    """Ridge-based predictor for MLB runs, with a preprocessing pipeline."""
    def __init__(self, model_dir: Union[str, Path] = SHARED_MODELS_BASE_DIR, model_name: str = "ridge_mlb_runs_predictor"): # CHANGED model_dir default
        super().__init__(model_name=model_name, model_dir=model_dir)
        # Default hyperparameters; consider MLB-specific tuning.
        self._default_ridge_params: Dict[str, Any] = {
            'alpha': 1.0, 'fit_intercept': True, 'solver': 'auto', 'random_state': SEED
        }

    def _build_pipeline(self, ridge_params: Dict[str, Any]) -> Pipeline:
        """
        Build a scikit-learn Pipeline for a Ridge regression model.
        """
        final_params = self._default_ridge_params.copy()
        final_params.update(ridge_params)

        # === FIX: Use self.feature_names_in_ which is now set before this is called ===
        preprocessing = _preprocessing_pipeline(numeric_features=self.feature_names_in_)

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('ridge', Ridge(**final_params))
        ])
        try:
            pipeline.set_output(transform="pandas")  # sklearn >=1.2
            logger.debug("Ridge pipeline set to output pandas DataFrame.")
        except AttributeError:
            logger.debug("Could not set pipeline output for Ridge (older scikit-learn).")
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
        # MLB runs are non-negative integers
        pred_home_processed = np.maximum(0.0, pred_home)
        pred_away_processed = np.maximum(0.0, pred_away)

        return pd.DataFrame({'predicted_home_runs': pred_home_processed, 'predicted_away_runs': pred_away_processed}, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "RidgeScorePredictor":
        loaded_instance = super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
        return loaded_instance # type: ignore

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
        
# --- SVR Model Definition ---
class SVRScorePredictor(BaseScorePredictor):
    """SVR-based predictor for MLB runs, with a preprocessing pipeline."""
    def __init__(self, model_dir: Union[str, Path] = SHARED_MODELS_BASE_DIR, model_name: str = "svr_mlb_runs_predictor"): # CHANGED model_dir default
        super().__init__(model_name=model_name, model_dir=model_dir)
        # Default hyperparameters; consider MLB-specific tuning.
        self._default_svr_params: Dict[str, Any] = {
            'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1,
        }

    def _build_pipeline(self, svr_params: Dict[str, Any]) -> Pipeline:
         final_params = self._default_svr_params.copy() 
         final_params.update(svr_params)
         # === FIX: Pass the required 'numeric_features' argument ===
         preprocessing = _preprocessing_pipeline(numeric_features=self.feature_names_in_) 
         pipeline = Pipeline([
             ('preprocessing', preprocessing),
             ('svr', SVR(**final_params)) 
         ])
         try: 
             pipeline.set_output(transform="pandas") # Requires sklearn 1.2+
             logger.debug("SVR pipeline set to output pandas DataFrame.")
         except AttributeError:
             logger.debug("Could not set pipeline output for SVR (older scikit-learn version).")
         return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None,
              fit_params: Optional[Dict[str, Any]] = None, 
              eval_set_data: Optional[Tuple] = None) -> None: # eval_set_data may not be used by SVR
         self._common_train_logic(X_train, y_train_home, y_train_away,
                                  hyperparams_home, hyperparams_away, sample_weights,
                                  self._default_svr_params,
                                  fit_params=fit_params,
                                  eval_set_data=eval_set_data) # Pass along, even if SVR doesn't use all parts

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
         predictions = self._common_predict_logic(X)
         if predictions is None: return None
         pred_home, pred_away = predictions
         # MLB runs are non-negative integers
         pred_home_processed = np.maximum(0.0, pred_home)
         pred_away_processed = np.maximum(0.0, pred_away)


         return pd.DataFrame({'predicted_home_runs': pred_home_processed, 'predicted_away_runs': pred_away_processed}, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "SVRScorePredictor":
         loaded_instance = super().load_model(filepath, model_name)
         self._validate_loaded_model_type()
         return loaded_instance # type: ignore

    def _validate_loaded_model_type(self) -> None:
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

# --- XGBoost Model Definition ---
class XGBoostScorePredictor(BaseScorePredictor):
    """XGBoost-based predictor for MLB runs, with imputation+scaling pipeline."""
    def __init__(self, model_dir: Union[str, Path] = SHARED_MODELS_BASE_DIR, model_name="xgb_mlb_runs_predictor"): # CHANGED model_dir default
        super().__init__(model_name=model_name, model_dir=model_dir)
        # Default hyperparameters; consider MLB-specific tuning.
        self._default_xgb_params = {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': SEED,
            'objective': 'reg:squarederror',
            'verbosity': 0,
        }

    def _build_pipeline(self, xgb_params: Dict[str, Any]) -> Pipeline:
        params = self._default_xgb_params.copy()
        params.update(xgb_params)
        # === FIX: Pass the required 'numeric_features' argument ===
        preprocessing = _preprocessing_pipeline(numeric_features=self.feature_names_in_)
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('xgb', XGBRegressor(**params))
        ])
        # XGBoost pipeline does not have set_output like scikit-learn directly
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None, 
              fit_params: Optional[Dict[str, Any]] = None, 
              eval_set_data: Optional[Tuple] = None) -> None:
        self._common_train_logic(
            X_train, y_train_home, y_train_away,
            hyperparams_home, hyperparams_away,
            sample_weights, self._default_xgb_params,
            fit_params=fit_params, eval_set_data=eval_set_data
        )

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        preds = self._common_predict_logic(X)
        if preds is None: return None
        ph, pa = preds
        # MLB runs are non-negative integers
        return pd.DataFrame({
            'predicted_home_runs': np.maximum(0.0, ph),
            'predicted_away_runs': np.maximum(0.0, pa)
        }, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "XGBoostScorePredictor":
        inst = super().load_model(filepath, model_name)
        # Optionally validate instance of XGBRegressor for deeper check
        self._validate_loaded_model_type()
        return inst # type: ignore

    def _validate_loaded_model_type(self) -> None:
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], XGBRegressor):
                raise TypeError("Loaded home pipeline is not an XGBRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], XGBRegressor):
                raise TypeError("Loaded away pipeline is not an XGBRegressor.")
            logger.debug("Validated XGBoost pipelines successfully.")
        except Exception as e:
            logger.error(f"Validation error for XGBoost model: {e}", exc_info=True)
            raise
        
class RFScorePredictor(BaseScorePredictor):
    """RandomForest-based predictor for MLB runs, with imputation+scaling pipeline."""
    def __init__(self, model_dir: Union[str, Path] = SHARED_MODELS_BASE_DIR, model_name="rf_mlb_runs_predictor"):
        super().__init__(model_name=model_name, model_dir=model_dir)
        # Default hyperparameters for RandomForest
        self._default_rf_params = {
            'n_estimators': 100, 
            'max_depth': None, 
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': SEED,
            'n_jobs': -1  # Use all available cores
        }

    def _build_pipeline(self, rf_params: Dict[str, Any]) -> Pipeline:
        params = self._default_rf_params.copy()
        params.update(rf_params)
        
        preprocessing = _preprocessing_pipeline(numeric_features=self.feature_names_in_)
        
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            # Use the 'rf' prefix to match your hyperparameter grid
            ('rf', RandomForestRegressor(**params))
        ])
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None, 
              fit_params: Optional[Dict[str, Any]] = None, 
              eval_set_data: Optional[Tuple] = None) -> None:
        # This method calls the generic training logic from the base class
        self._common_train_logic(
            X_train, y_train_home, y_train_away,
            hyperparams_home, hyperparams_away,
            sample_weights, self._default_rf_params, # Pass the correct default params
            fit_params=fit_params, eval_set_data=eval_set_data
        )

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        # This method calls the generic prediction logic from the base class
        preds = self._common_predict_logic(X)
        if preds is None: return None
        ph, pa = preds
        
        # Ensure predictions are non-negative
        return pd.DataFrame({
            'predicted_home_runs': np.maximum(0.0, ph),
            'predicted_away_runs': np.maximum(0.0, pa)
        }, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None) -> "RFScorePredictor":
        inst = super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
        return inst # type: ignore

    def _validate_loaded_model_type(self) -> None:
        # This ensures the loaded model is of the correct type
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], RandomForestRegressor):
                raise TypeError("Loaded home pipeline is not a RandomForestRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], RandomForestRegressor):
                raise TypeError("Loaded away pipeline is not a RandomForestRegressor.")
            logger.debug("Validated RandomForest pipelines successfully.")
        except Exception as e:
            logger.error(f"Validation error for RandomForest model: {e}", exc_info=True)
            raise


# --- Inning-Specific Model System ---
class InningSpecificModelSystem:
    """
    Manages inning-specific models for in-game MLB run predictions.
    Supports loading primary models, fallback models (Ridge-based), and caching.
    Handles typically 9 innings; extra innings logic would be an enhancement.
    """
    def __init__(self, feature_generator: Optional[Any] = None, debug: bool = False):
        self.feature_generator = feature_generator  # Expected to handle MLB-specific features
        self.debug = debug
        self.models: Dict[int, Any] = {}  # Models for each inning (1-9 for home team runs)
        self.fallback_models: Dict[int, Any] = {}
        self.inning_feature_sets: Dict[int, List[str]] = self._get_consolidated_feature_sets()
        # Example error history (in runs); these would be tuned/learned
        self.error_history: Dict[str, Dict[int, float]] = {
            'main_model': {i: 2.0 - (i-1)*0.1 for i in range(1, NUM_INNINGS_REGULATION + 1)},  # Lower error as game progresses
            'inning_model': {i: 1.5 - (i-1)*0.05 for i in range(1, NUM_INNINGS_REGULATION + 1)}
        }
        self.prediction_cache: Dict[str, float] = {}
        self._create_fallback_models()
        self.log(f"InningSpecificModelSystem initialized for {NUM_INNINGS_REGULATION}-inning prediction structure.", level="DEBUG")

    def log(self, message: str, level: str = "INFO") -> None:
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[InningSystem] {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        base_sets: Dict[int, List[str]] = {}
        if self.feature_generator:
            try: # Try to get MLB-specific feature sets per inning
                if hasattr(self.feature_generator, 'get_prediction_feature_sets_by_inning'):
                    base_sets = self.feature_generator.get_prediction_feature_sets_by_inning()
                elif hasattr(self.feature_generator, 'inning_feature_sets'): # Fallback attribute name
                    base_sets = self.feature_generator.inning_feature_sets
                if not isinstance(base_sets, dict):
                    self.log(f"Feature set attribute not a dict ({type(base_sets)}). Using empty sets.", level="WARNING")
                    base_sets = {}
            except Exception as e:
                self.log(f"Error retrieving inning feature sets: {e}. Using empty sets.", level="ERROR")
                base_sets = {}
        
        # Ensure sets for all regulation innings
        final_sets = {inn: base_sets.get(inn, []) for inn in range(1, NUM_INNINGS_REGULATION + 1)}
        log_msg = "Consolidated feature sets per inning: "
        log_msg += ", ".join([f"I{inn}({len(final_sets.get(inn, []))})" for inn in range(1, NUM_INNINGS_REGULATION + 1)])
        self.log(log_msg, level="DEBUG")
        return final_sets

    def _create_fallback_models(self) -> None:
        self.log("Ensuring fallback Ridge models are available for each inning...", level="DEBUG")
        for inning in range(1, NUM_INNINGS_REGULATION + 1):  # For innings 1 through 9
            if inning not in self.fallback_models:
                try:
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=1.0, random_state=SEED))  # Basic params
                    ])
                    self.fallback_models[inning] = model
                    self.log(f"Fallback Ridge model created for Inning {inning}.", level="DEBUG")
                except Exception as e:
                    self.log(f"Failed to create fallback model for Inning {inning}: {e}", level="ERROR")

    def predict_inning(self, X: pd.DataFrame, inning: int) -> float:
        """
        Predicts a single-inning home team runs using the appropriate model.
        Expects X to be a single-row DataFrame with current game state features.
        """
        if not (1 <= inning <= NUM_INNINGS_REGULATION):
            self.log(f"Invalid inning {inning} for standard prediction. Max is {NUM_INNINGS_REGULATION}.", level="ERROR")
            return 0.0

        if X is None or X.empty or X.shape[0] != 1:
            self.log(f"predict_inning expects a single-row DataFrame. Got shape: {X.shape if X is not None else 'None'}.", level="ERROR")
            return 0.0

        features_expected = self.inning_feature_sets.get(inning, [])
        default_inning_runs = 0.5  # MLB: Average runs per inning is low
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
            default_inning_runs = self.feature_generator.league_averages.get('inning_runs', {}).get(inning, 0.5)

        if not features_expected:
            self.log(f"No expected features for Inning {inning}. Using default runs: {default_inning_runs}", level="WARNING")
            return default_inning_runs

        features_available = [f for f in features_expected if f in X.columns]
        if not features_available:
            self.log(f"Expected Inning {inning} features not in input. Using default runs: {default_inning_runs}", level="WARNING")
            return default_inning_runs

        X_predict_inning = X[features_available]

        cache_key = None
        try:
            key_tuple = tuple(X_predict_inning.iloc[0].fillna(0).values)
            cache_key = f"i{inning}_{hash(key_tuple)}"
            if cache_key in self.prediction_cache:
                self.log(f"Using cached prediction for Inning {inning}.", level="DEBUG")
                return self.prediction_cache[cache_key]
        except Exception as e:
            self.log(f"Cache key generation failed for Inning {inning}: {e}.", level="WARNING")
            cache_key = None

        model_to_use = self.models.get(inning) or self.fallback_models.get(inning)
        model_source = "primary" if inning in self.models and self.models[inning] is not None else "fallback"

        if model_to_use is None:
            self.log(f"No model (incl. fallback) for Inning {inning}. Using default runs: {default_inning_runs}", level="ERROR")
            if cache_key:
                self.prediction_cache[cache_key] = default_inning_runs
            return default_inning_runs

        try:  # Feature alignment for loaded model
            if hasattr(model_to_use, 'steps') and model_to_use.steps:  # Is a pipeline
                final_estimator = model_to_use.steps[-1][1]
                if hasattr(final_estimator, 'feature_names_in_') and final_estimator.feature_names_in_ is not None:
                    required_model_features = list(final_estimator.feature_names_in_)
                    if set(required_model_features).issubset(set(X_predict_inning.columns)):
                        X_predict_inning = X_predict_inning[required_model_features]  # Reorder/select
                    else:
                        missing_req = set(required_model_features) - set(X_predict_inning.columns)
                        self.log(f"Model for I{inning} requires features not in current selection: {missing_req}. Using default.", level="ERROR")
                        if cache_key:
                            self.prediction_cache[cache_key] = default_inning_runs
                        return default_inning_runs
                else:
                    self.log(f"Model for I{inning} lacks stored feature names. Using available intersection: {list(X_predict_inning.columns)}", level="DEBUG")
        except Exception as feat_e:
            self.log(f"Error during feature validation for I{inning} model: {feat_e}. Using default.", level="ERROR")
            if cache_key:
                self.prediction_cache[cache_key] = default_inning_runs
            return default_inning_runs

        self.log(f"Predicting Inning {inning} using {model_source} model with {X_predict_inning.shape[1]} features.", level="DEBUG")
        prediction_val = None
        try:
            prediction_val = model_to_use.predict(X_predict_inning)[0]
        except Exception as e:
            self.log(f"Error during Inning {inning} prediction with {model_source} model: {e}", level="ERROR")
            if model_source == "primary":
                self.log(f"Attempting fallback prediction for Inning {inning}...", level="WARNING")
                fallback_model = self.fallback_models.get(inning)
                if fallback_model:
                    try:
                        fb_input = X[features_available].fillna(0)
                        prediction_val = fallback_model.predict(fb_input)[0]
                        model_source = "fallback (after primary error)"
                    except Exception as fb_e:
                        self.log(f"Fallback prediction error for Inning {inning}: {fb_e}", level="ERROR")
                        prediction_val = None
            else:
                prediction_val = None

        final_prediction_runs = default_inning_runs
        if prediction_val is not None:
            try:
                final_prediction_runs = max(0.0, float(prediction_val))
            except (ValueError, TypeError):
                self.log(f"Non-numeric prediction for I{inning} ('{prediction_val}'). Using default.", level="ERROR")
                final_prediction_runs = default_inning_runs
        else:
            self.log(f"Prediction failed for I{inning} after all attempts. Using default: {final_prediction_runs}", level="ERROR")
            model_source += " -> default"

        self.log(f"Inning {inning} Prediction ({model_source}): {final_prediction_runs:.3f} runs", level="DEBUG")
        if cache_key:
            self.prediction_cache[cache_key] = final_prediction_runs
        return final_prediction_runs

    def predict_final_score(self, game_data: Dict, 
                        main_model_pred_home: Optional[float] = None, 
                        main_model_pred_away: Optional[float] = None,
                        weight_manager: Optional[Any] = None) -> Tuple[float, float, float, Dict]:
        """
        Combines main model full-game predictions and inning-specific predictions
        using an ensemble weight manager (if provided) to produce final total runs predictions
        for home and away teams.
        Returns: (final_total_runs_home, final_total_runs_away, confidence, breakdown_dict)
        """
        self.log("Predicting final MLB score using ensemble weighting...", level="DEBUG")

        league_avg_runs_per_team = 4.5
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
            league_avg_runs_per_team = self.feature_generator.league_averages.get(
                'runs_per_game_per_team', 4.5
            )

        default_pred_home = main_model_pred_home if main_model_pred_home is not None else league_avg_runs_per_team
        default_pred_away = main_model_pred_away if main_model_pred_away is not None else league_avg_runs_per_team
        try:
            default_pred_home = float(default_pred_home)
            default_pred_away = float(default_pred_away)
        except (ValueError, TypeError):
            default_pred_home = league_avg_runs_per_team
            default_pred_away = league_avg_runs_per_team

        if not isinstance(game_data, dict):
            self.log("game_data must be a dict.", level="ERROR")
            return default_pred_home, default_pred_away, 0.0, {'error': 'Invalid game_data type'}

        if weight_manager is None or not hasattr(weight_manager, 'calculate_ensemble_mlb'):
            self.log(
                "WeightManager with 'calculate_ensemble_mlb' method not provided. "
                "Using main model predictions or extrapolation.",
                level="WARNING"
            )
            # Pre-game fallback logic (could return here if desired)
            # For now, we just log and continue to compute home/away so-far.

        home_runs_so_far = float(game_data.get('home_score', 0) or 0)
        away_runs_so_far = float(game_data.get('away_score', 0) or 0)

        try:
            current_inning_completed = int(game_data.get('current_inning_completed', 0))
        except (ValueError, TypeError):
            self.log("Invalid current_inning_completed; defaulting to 0 (pre-game).", level="WARNING")
            current_inning_completed = 0
        # if the main 'home_score'/'away_score' keys are missing or zero.
        if current_inning_completed > 0 and \
           (game_data.get('home_score') is None or (home_runs_so_far == 0 and away_runs_so_far == 0)):
            self.log("Overall 'home_score'/'away_score' might be missing or zero for in-progress game. Attempting to sum from inning-specific keys (e.g., 'h_inn_1', 'a_inn_1').", level="DEBUG")
            home_runs_so_far = sum([float(game_data.get(f'h_inn_{i}', 0) or 0) for i in range(1, current_inning_completed + 1)])
            away_runs_so_far = sum([float(game_data.get(f'a_inn_{i}', 0) or 0) for i in range(1, current_inning_completed + 1)])
            # Update game_data with these summed scores if they were missing, for consistency
            game_data['home_score'] = home_runs_so_far
            game_data['away_score'] = away_runs_so_far
        # --- >> END OF FALLBACK SUMMATION LOGIC << ---

        # Now, proceed with extrapolation using the (potentially updated) home_runs_so_far
        if main_model_pred_home is None:
            if (current_inning_completed > 0 and
                (home_runs_so_far > 0 or current_inning_completed > 3)): # Ensure home_runs_so_far here is the corrected one
                main_model_pred_home = (home_runs_so_far / current_inning_completed) * NUM_INNINGS_REGULATION if current_inning_completed > 0 else league_avg_runs_per_team # Added check for current_inning_completed > 0 before division
                self.log(
                    f"Extrapolated main model home runs: ~{main_model_pred_home:.2f}",
                    level="DEBUG"
                )
            else: # Pre-game or very early, use league average
                main_model_pred_home = league_avg_runs_per_team
                self.log(
                    f"Using league average for main model home runs: ~{main_model_pred_home:.2f}",
                    level="DEBUG"
                )
        # ... (similar extrapolation for main_model_pred_away) ...
        if main_model_pred_away is None:
            if (current_inning_completed > 0 and
                (away_runs_so_far > 0 or current_inning_completed > 3)): # Ensure away_runs_so_far here is the corrected one
                main_model_pred_away = (away_runs_so_far / current_inning_completed) * NUM_INNINGS_REGULATION if current_inning_completed > 0 else league_avg_runs_per_team # Added check
                self.log(
                    f"Extrapolated main model away runs: ~{main_model_pred_away:.2f}",
                    level="DEBUG"
                )
            else:
                main_model_pred_away = league_avg_runs_per_team
                self.log(
                    f"Using league average for main model away runs: ~{main_model_pred_away:.2f}",
                    level="DEBUG"
                )

        try: # Ensure float after potential extrapolation
            main_model_pred_home = float(main_model_pred_home)
            main_model_pred_away = float(main_model_pred_away)
        except (ValueError, TypeError):
            main_model_pred_home = league_avg_runs_per_team
            main_model_pred_away = league_avg_runs_per_team
        
        # Handle pre-game or no weight manager case
        if current_inning_completed == 0 or weight_manager is None:
            self.log("Pre-game or no weight manager. Using main model predictions as final.", level="DEBUG")
            # ... (rest of the pre-game return logic as you have it) ...
            confidence = 0.5
            breakdown = {
                'main_model_pred_home': main_model_pred_home,
                'main_model_pred_away': main_model_pred_away,
                'inning_model_sum_home': 0,
                'inning_model_sum_away': 0,
                'weights': {'main': 1.0, 'inning': 0.0},
                'inning_predictions': {},
                'current_runs_home': 0, # Should be home_runs_so_far, but pre-game it's 0
                'current_runs_away': 0, # Should be away_runs_so_far, but pre-game it's 0
                'run_differential': 0,
                'momentum_metric': 0,
                'current_inning_completed': 0,
                'innings_remaining_est': NUM_INNINGS_REGULATION
            }
            # Ensure breakdown uses correct values for pre-game if current_inning_completed is truly 0.
            # If current_inning_completed != 0 but weight_manager is None, current scores might not be 0.
            if current_inning_completed == 0:
                breakdown['current_runs_home'] = 0
                breakdown['current_runs_away'] = 0
            else: # No weight manager, but game in progress
                breakdown['current_runs_home'] = home_runs_so_far
                breakdown['current_runs_away'] = away_runs_so_far
                breakdown['run_differential'] = home_runs_so_far - away_runs_so_far
            return main_model_pred_home, main_model_pred_away, confidence, breakdown

        # In-game: Combine current score with predicted remaining inning scores
        remaining_innings_pred_home_dict = self.predict_remaining_innings(
            game_data, current_inning_completed
        )
        if not isinstance(remaining_innings_pred_home_dict, dict):
            self.log(
                "Remaining innings prediction (home) failed. "
                "Using main model adjusted by current score.",
                level="ERROR"
            )
            final_pred_home = home_runs_so_far + max(
                0, main_model_pred_home - (main_model_pred_home * current_inning_completed / NUM_INNINGS_REGULATION)
            )
            final_pred_away = away_runs_so_far + max(
                0, main_model_pred_away - (main_model_pred_away * current_inning_completed / NUM_INNINGS_REGULATION)
            )
            return final_pred_home, final_pred_away, 0.3, {
                'error': 'Failed to predict remaining innings for home team'
            }

        predicted_remaining_home_runs = sum(
            v for k, v in remaining_innings_pred_home_dict.items() if k.startswith('home_i')
        )
        inning_model_total_home_runs = home_runs_so_far + predicted_remaining_home_runs

        # Placeholder for away team's remaining runs prediction
        predicted_remaining_away_runs_est = max(0, main_model_pred_away - away_runs_so_far)
        inning_model_total_away_runs = away_runs_so_far + predicted_remaining_away_runs_est

        run_diff = home_runs_so_far - away_runs_so_far
        momentum = 0.0
        innings_left_in_regulation = NUM_INNINGS_REGULATION - current_inning_completed

        # Extract more context features if available
        try:
            X_context = pd.DataFrame([game_data])
            if self.feature_generator:
                if hasattr(self.feature_generator, 'generate_live_context_features'):
                    X_context = self.feature_generator.generate_live_context_features(X_context)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X_context = self.feature_generator.integrate_advanced_features(X_context)
                if not X_context.empty:
                    if ('current_run_differential' in X_context.columns and
                        pd.notna(X_context['current_run_differential'].iloc[0])):
                        run_diff = float(X_context['current_run_differential'].iloc[0])
                    if ('home_team_momentum_ ostatnich_3_inn' in X_context.columns and
                        pd.notna(X_context['home_team_momentum_ ostatnich_3_inn'].iloc[0])):
                        momentum = float(X_context['home_team_momentum_ ostatnich_3_inn'].iloc[0])
                    if ('outs_remaining_in_game_est' in X_context.columns and
                        pd.notna(X_context['outs_remaining_in_game_est'].iloc[0])):
                        pass
            else:
                self.log(
                    "Feature generator missing for live context. Using basic estimates.",
                    level="WARNING"
                )
        except Exception as e:
            self.log(
                f"Error extracting live context features: {e}. Using basic defaults.",
                level="WARNING"
            )

        try:
            (ensembled_home_runs, ensembled_away_runs,
                weight_main, weight_inning_model) = weight_manager.calculate_ensemble_mlb(
                main_pred_home=main_model_pred_home,
                main_pred_away=main_model_pred_away,
                inning_model_total_pred_home=inning_model_total_home_runs,
                inning_model_total_pred_away=inning_model_total_away_runs,
                current_inning=current_inning_completed,
                run_differential=run_diff,
                momentum=momentum,
                innings_remaining=innings_left_in_regulation
            )
            confidence = (
                0.6 + 0.3 * min(weight_main, weight_inning_model) /
                max(weight_main, weight_inning_model, 1e-6)
            ) if max(weight_main, weight_inning_model) > 1e-6 else 0.6
        except Exception as e:
            self.log(
                f"Error in ensemble weighting: {e}. Falling back to simpler sum.",
                level="ERROR", exc_info=True
            )
            ensembled_home_runs = inning_model_total_home_runs
            ensembled_away_runs = inning_model_total_away_runs
            confidence = 0.3
            weight_main, weight_inning_model = 0.0, 1.0

        try:
            final_pred_home_runs = float(ensembled_home_runs)
            final_pred_away_runs = float(ensembled_away_runs)
            final_confidence = float(confidence)

            breakdown = {
                'main_model_pred_home': main_model_pred_home,
                'main_model_pred_away': main_model_pred_away,
                'inning_model_sum_home': inning_model_total_home_runs,
                'inning_model_sum_away': inning_model_total_away_runs,
                'weights': {'main_model': weight_main, 'inning_model': weight_inning_model},
                'inning_predictions_home': remaining_innings_pred_home_dict,
                'current_runs_home': home_runs_so_far,
                'current_runs_away': away_runs_so_far,
                'run_differential': run_diff,
                'momentum_metric': momentum,
                'current_inning_completed': current_inning_completed,
                'innings_remaining_est': innings_left_in_regulation
            }
            self.log(
                f"Final Ensembled MLB Prediction: Home {final_pred_home_runs:.2f}, "
                f"Away {final_pred_away_runs:.2f} (Confidence: {final_confidence:.2f})",
                level="DEBUG"
            )
            return final_pred_home_runs, final_pred_away_runs, final_confidence, breakdown

        except Exception as final_e:
            self.log(
                f"Error finalizing ensemble MLB prediction: {final_e}. "
                "Using basic extrapolation.",
                level="ERROR"
            )
            ext_home = (
                (home_runs_so_far / current_inning_completed) * NUM_INNINGS_REGULATION
                if current_inning_completed > 0 else league_avg_runs_per_team
            )
            ext_away = (
                (away_runs_so_far / current_inning_completed) * NUM_INNINGS_REGULATION
                if current_inning_completed > 0 else league_avg_runs_per_team
            )
            return ext_home, ext_away, 0.1, {
                'error': 'Non-numeric or other error in final ensemble result'
            }
