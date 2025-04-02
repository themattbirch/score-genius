# backend/nba_score_prediction/models.py

import pandas as pd
import os  # Keep os for os.path.join used in save_model/load_model fallbacks
import numpy as np
import joblib
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path  # Make sure Path is imported

# --- Scikit-learn Imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# --- Optional XGBoost Import ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost library not found...")

# --- Type Hinting Placeholders ---
NBAFeatureEngine = Any
EnsembleWeightManager = Any

# --- Configuration ---
# Define MODELS_BASE_DIR using pathlib
MODELS_BASE_DIR = Path(__file__).resolve().parent.parent / 'models' / 'saved'
MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Define QUARTERLY_MODELS_DIR for quarter-specific models
QUARTERLY_MODELS_DIR = Path(__file__).resolve().parent.parent / 'models' / 'quarterly'
QUARTERLY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
# !!! END CONFIGURATION SECTION !!!

# --- Utility Functions ---

def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """ Calculate recency weights for samples based on dates. """
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
        logger.warning("Recency weights: no valid dates.")
        return np.ones(len(dates))
    sorted_dates = valid_dates.sort_values()
    latest_date = sorted_dates.max()
    if pd.isna(latest_date):
        logger.warning("Recency weights: no valid latest date.")
        return np.ones(len(dates))
    days_from_latest = (latest_date - sorted_dates).dt.total_seconds() / (3600 * 24.0)
    days_from_latest = days_from_latest.astype(float)
    weights = np.ones(len(sorted_dates))
    if method == 'half_life':
        if half_life <= 0:
            logger.warning("Recency weights: Half-life must be positive.")
        else:
            weights = 0.5 ** (days_from_latest / float(half_life))
    elif method == 'exponential':
        if not (0 < decay_rate < 1):
            logger.warning("Recency weights: Exponential decay_rate must be between 0 and 1.")
        else:
            weights = decay_rate ** days_from_latest
    else:
        logger.warning(f"Recency weights: Unknown method: {method}.")
    weights[~np.isfinite(weights)] = 0.0
    mean_weight = np.mean(weights)
    if mean_weight > 1e-9:
        weights = weights / mean_weight
    else:
        logger.warning("Recency weights: Mean weight near zero.")
    weights_series = pd.Series(weights, index=valid_dates.index)
    return weights_series.reindex(original_index, fill_value=0.0).values

# --- Base Predictor Class ---
class BaseScorePredictor:
    """Abstract base class for score predictors."""
    def __init__(self, model_dir: str, model_name: str):
        self.model_dir = str(model_dir)
        self.model_name = model_name
        self.pipeline_home: Optional[Pipeline] = None
        self.pipeline_away: Optional[Pipeline] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.training_timestamp: Optional[str] = None
        self.training_duration: Optional[float] = None

    def _build_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        raise NotImplementedError

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series, **kwargs):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def save_model(self, filename: Optional[str] = None) -> str:
        if not self.pipeline_home or not self.pipeline_away:
            raise RuntimeError("Models must be trained before saving.")
        timestamp = self.training_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"{self.model_name}_{timestamp}.joblib"
        else:
            if not filename.endswith(".joblib"):
                filename += ".joblib"
        save_path = os.path.join(self.model_dir, filename)
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
            return save_path
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}")
            raise

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        load_path = filepath
        if load_path is None:
            try:
                search_name = model_name or self.model_name
                if not os.path.isdir(self.model_dir):
                    raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
                files = [f for f in os.listdir(self.model_dir) if f.startswith(search_name) and f.endswith(".joblib")]
                if not files:
                    raise FileNotFoundError(f"No model file found for '{search_name}' in {self.model_dir}")
                files.sort(reverse=True)
                load_path = os.path.join(self.model_dir, files[0])
                logger.info(f"Loading latest model: {load_path}")
            except Exception as e:
                logger.error(f"Error finding latest model file: {e}")
                raise
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")
        try:
            model_data = joblib.load(load_path)
            loaded_class_name = model_data.get('model_class')
            if loaded_class_name and loaded_class_name != self.__class__.__name__:
                raise TypeError(f"Loaded model class ({loaded_class_name}) does not match expected ({self.__class__.__name__}).")
            if 'pipeline_home' not in model_data or 'pipeline_away' not in model_data:
                raise ValueError("Loaded model file is missing required pipeline data.")
            if not isinstance(model_data['pipeline_home'], Pipeline) or not isinstance(model_data['pipeline_away'], Pipeline):
                raise TypeError("Loaded pipelines are not valid sklearn Pipeline objects.")
            self.pipeline_home = model_data['pipeline_home']
            self.pipeline_away = model_data['pipeline_away']
            self.feature_names_in_ = model_data.get('feature_names_in_')
            self.training_timestamp = model_data.get('training_timestamp')
            self.training_duration = model_data.get('training_duration')
            self.model_name = model_data.get('model_name', self.model_name)
            if self.feature_names_in_ is not None and not isinstance(self.feature_names_in_, list):
                logger.warning("Feature names in loaded model are not a list; resetting to None.")
                self.feature_names_in_ = None
            logger.info(f"{self.__class__.__name__} model loaded successfully from {load_path}")
        except Exception as e:
            logger.error(f"Error loading model from {load_path}: {e}")
            raise

    def _common_train_logic(self, X_train, y_train_home, y_train_away,
                              hyperparams_home, hyperparams_away, sample_weights,
                              default_params,
                              fit_params: Optional[Dict[str, Any]] = None,
                              eval_set_data: Optional[Tuple] = None):
        start_time = time.time()
        try:
            X_train_numeric = X_train.select_dtypes(include=np.number)
            if X_train_numeric.shape[1] != X_train.shape[1]:
                dropped_cols = set(X_train.columns) - set(X_train_numeric.columns)
                logger.warning(f"_common_train_logic received non-numeric columns: {dropped_cols}. Using only numeric.")
                X_train = X_train_numeric
            self.feature_names_in_ = list(X_train.columns)
        except Exception as e:
            logger.error(f"Error processing X_train columns: {e}")
            self.feature_names_in_ = list(X_train.columns)
        logger.info(f"Starting {self.__class__.__name__} training for {self.model_name}...")
        logger.debug(f"Training with {len(self.feature_names_in_)} features: {self.feature_names_in_[:10]}...")
        params_home = default_params.copy()
        params_away = default_params.copy()
        if hyperparams_home:
            params_home.update(hyperparams_home)
            logger.info("Using custom hyperparameters for home.")
        if hyperparams_away:
            params_away.update(hyperparams_away)
            logger.info("Using custom hyperparameters for away.")
        params_home.pop('normalize', None)
        params_away.pop('normalize', None)
        self.pipeline_home = self._build_pipeline(params_home)
        self.pipeline_away = self._build_pipeline(params_away)
        fit_kwargs_home = fit_params.copy() if fit_params else {}
        fit_kwargs_away = fit_params.copy() if fit_params else {}
        step_name = self.pipeline_home.steps[-1][0]
        weight_param_name = f"{step_name}__sample_weight"
        if sample_weights is not None:
            if len(sample_weights) == len(X_train):
                fit_kwargs_home[weight_param_name] = sample_weights
                fit_kwargs_away[weight_param_name] = sample_weights
                logger.info(f"Applying sample weights to {step_name}.")
            else:
                logger.warning(f"Sample weights length mismatch. Ignoring.")
        is_xgb = False
        if XGBOOST_AVAILABLE:
            try:
                if isinstance(self.pipeline_home.steps[-1][1], xgb.XGBRegressor):
                    is_xgb = True
            except (AttributeError, IndexError):
                pass
        if is_xgb and eval_set_data:
            if isinstance(eval_set_data, tuple) and len(eval_set_data) == 3:
                try:
                    X_val, y_val_home, y_val_away = eval_set_data
                    X_val_aligned = X_val[self.feature_names_in_]
                    fit_kwargs_home['xgb__eval_set'] = [(X_val_aligned, y_val_home)]
                    fit_kwargs_away['xgb__eval_set'] = [(X_val_aligned, y_val_away)]
                    logger.info("Applying XGBoost eval_set separately for home/away.")
                except KeyError as ke:
                    logger.error(f"Eval set missing required columns: {ke}. Ignoring eval_set.")
                except Exception as e:
                    logger.warning(f"Could not parse/align 'eval_set_data': {e}. Ignoring.")
            else:
                logger.warning("Incorrect eval_set_data format. Ignoring.")
        try:
            logger.info(f"Training home score model... Fit args: {list(fit_kwargs_home.keys())}")
            self.pipeline_home.fit(X_train, y_train_home, **fit_kwargs_home)
            logger.info(f"Training away score model... Fit args: {list(fit_kwargs_away.keys())}")
            self.pipeline_away.fit(X_train, y_train_away, **fit_kwargs_away)
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start_time
        logger.info(f"{self.__class__.__name__} training completed in {self.training_duration:.2f} seconds.")

    def _common_predict_logic(self, X: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.pipeline_home or not self.pipeline_away:
            logger.error("Prediction failed: pipelines are not available.")
            return None
        required_features = self.feature_names_in_
        if required_features:
            missing_cols = set(required_features) - set(X.columns)
            extra_cols = set(X.columns) - set(required_features)
            if missing_cols:
                logger.error(f"Predict input missing required: {missing_cols}.")
                return None
            if extra_cols:
                logger.warning(f"Predict input has extra: {extra_cols}. Selecting required.")
                X_predict = X[required_features]
            elif list(X.columns) != required_features:
                logger.warning("Predict input features match but order differs. Reordering.")
                X_predict = X[required_features]
            else:
                X_predict = X
        else:
            logger.warning(f"{self.__class__.__name__} skipping predict feature validation.")
            X_predict = X
        logger.info(f"Predicting scores using {self.__class__.__name__} for {len(X_predict)} samples...")
        try:
            pred_home = self.pipeline_home.predict(X_predict)
            pred_away = self.pipeline_away.predict(X_predict)
            logger.info(f"{self.__class__.__name__} prediction finished.")
            return pred_home, pred_away
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return None

# --- XGBoost Model Definition ---
class XGBoostScorePredictor(BaseScorePredictor):
    """ Trains and predicts using XGBoost models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "xgboost_score_predictor"):
        super().__init__(model_dir, model_name)
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost library not found.")
        self._default_xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1
        }
    def _build_pipeline(self, xgb_params: Dict[str, Any]) -> Pipeline:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost library required.")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor(**xgb_params))
        ])
        try:
            pipeline.set_output(transform="pandas")
            logger.debug("Set XGBoost pipeline output pandas.")
        except AttributeError:
            logger.debug("Could not set pipeline output pandas (sklearn < 1.2).")
        return pipeline
    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None, fit_params: Optional[Dict[str, Any]] = None,
              eval_set_data: Optional[Tuple] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("Cannot train: XGBoost library not found.")
        self._common_train_logic(X_train, y_train_home, y_train_away, hyperparams_home, hyperparams_away,
                                 sample_weights, self._default_xgb_params, fit_params=fit_params, eval_set_data=eval_set_data)
    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)
    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
    def _validate_loaded_model_type(self):
        if not XGBOOST_AVAILABLE:
            return
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], xgb.XGBRegressor):
                raise TypeError("Loaded home pipeline is not an XGBRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], xgb.XGBRegressor):
                raise TypeError("Loaded away pipeline is not an XGBRegressor.")
        except (AttributeError, IndexError) as e:
            logger.error(f"Error during validation of XGBoost model: {e}")
            raise TypeError("Loaded model validation failed.")

# --- RandomForest Model Definition ---
class RandomForestScorePredictor(BaseScorePredictor):
    """ Trains and predicts using RandomForest models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "random_forest_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': False
        }
    def _build_pipeline(self, rf_params: Dict[str, Any]) -> Pipeline:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(**rf_params))
        ])
        try:
            pipeline.set_output(transform="pandas")
            logger.debug("Set RF pipeline output pandas.")
        except AttributeError:
            logger.debug("Could not set pipeline output pandas.")
        return pipeline
    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        self._common_train_logic(X_train, y_train_home, y_train_away, hyperparams_home, hyperparams_away,
                                 sample_weights, self._default_rf_params)
    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)
    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
    def _validate_loaded_model_type(self):
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], RandomForestRegressor):
                raise TypeError("Loaded home pipeline is not a RandomForestRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], RandomForestRegressor):
                raise TypeError("Loaded away pipeline is not a RandomForestRegressor.")
        except (AttributeError, IndexError):
            logger.error("Error validating RandomForest model.")
            raise TypeError("Loaded model validation failed.")

# --- Ridge Model Definition ---
class RidgeScorePredictor(BaseScorePredictor):
    """ Trains and predicts using Ridge models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "ridge_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_ridge_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'solver': 'auto',
            'random_state': 42
        }
    def _build_pipeline(self, ridge_params: Dict[str, Any]) -> Pipeline:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**ridge_params))
        ])
        try:
            pipeline.set_output(transform="pandas")
            logger.debug("Set Ridge pipeline output pandas.")
        except AttributeError:
            logger.debug("Could not set pipeline output pandas.")
        return pipeline
    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        self._common_train_logic(X_train, y_train_home, y_train_away, hyperparams_home, hyperparams_away,
                                 sample_weights, self._default_ridge_params)
    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)
    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        super().load_model(filepath, model_name)
        self._validate_loaded_model_type()
    def _validate_loaded_model_type(self):
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], Ridge):
                raise TypeError("Loaded home pipeline is not a Ridge regressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], Ridge):
                raise TypeError("Loaded away pipeline is not a Ridge regressor.")
        except (AttributeError, IndexError):
            logger.error("Error validating Ridge model.")
            raise TypeError("Loaded model validation failed.")

# --- Quarter-Specific Model System (Prediction Focus) ---
class QuarterSpecificModelSystem:
    """
    Manages and utilizes quarter-specific models for prediction,
    including loading pre-trained models and handling fallbacks.
    """
    def __init__(self, feature_generator: NBAFeatureEngine, debug: bool = False):
        self.feature_generator = feature_generator
        self.debug = debug

        # --- Model Storage (for Prediction) ---
        self.models: Dict[int, Any] = {}  # Stores loaded primary models for prediction
        self.fallback_models: Dict[int, Any] = {}  # Stores fallback models (Ridge)

        # --- Feature Set Definitions (Used for Prediction) ---
        self.quarter_feature_sets = self._get_consolidated_feature_sets()

        # --- Other Attributes ---
        self.error_history = {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.prediction_cache: Dict[str, float] = {}

        self._create_fallback_models()
        self.log("QuarterSpecificModelSystem initialized for prediction.", level="DEBUG")

    def log(self, message, level="INFO"):
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[QuarterSystem] {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        base_sets = {}
        try:
            if hasattr(self.feature_generator, 'get_prediction_feature_sets'):
                base_sets = self.feature_generator.get_prediction_feature_sets()
                if not isinstance(base_sets, dict):
                    base_sets = {}
            elif hasattr(self.feature_generator, '_get_optimized_feature_sets'):
                base_sets = self.feature_generator._get_optimized_feature_sets()
                if not isinstance(base_sets, dict):
                    base_sets = {}
            elif hasattr(self.feature_generator, 'feature_sets'):
                base_sets = self.feature_generator.feature_sets
                if not isinstance(base_sets, dict):
                    base_sets = {}
            else:
                self.log("Feature generator missing expected feature set definitions. Using empty sets.", level="WARNING")

            final_sets = {q: base_sets.get(q, []) for q in range(1, 5)}
            self.log(f"Consolidated prediction feature sets from generator: Q1({len(final_sets.get(1, []))}), Q2({len(final_sets.get(2, []))}), Q3({len(final_sets.get(3, []))}), Q4({len(final_sets.get(4, []))}) features.", level="DEBUG")
            return final_sets

        except Exception as e:
            self.log(f"Error getting feature sets from generator: {e}. Using empty sets.", level="ERROR")
            return {q: [] for q in range(1, 5)}

    def load_models(self, model_dir: Optional[str] = None):
        """
        Load pre-trained quarter-specific models.

        Args:
            model_dir: Directory containing the quarter model files. Defaults to the globally
                       defined QUARTERLY_MODELS_DIR path.
        """
        if model_dir is None:
            if 'QUARTERLY_MODELS_DIR' in globals():
                model_dir = str(QUARTERLY_MODELS_DIR)
                logger.debug(f"Defaulting quarterly model directory to: {model_dir}")
            else:
                logger.error("QUARTERLY_MODELS_DIR not defined globally. Cannot determine default quarterly model path.")
                fallback_path = Path(__file__).resolve().parent.parent / 'models' / 'quarterly'
                model_dir = str(fallback_path)
                logger.warning(f"Attempting fallback quarterly model path: {model_dir}")

        self.log(f"Loading quarter prediction models from '{model_dir}'...")
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            self.log(f"Error creating directory {model_dir}: {e}", level="ERROR")
            return

        for quarter in range(1, 5):
            model_filename = f'q{quarter}_model.pkl'
            model_path = os.path.join(model_dir, model_filename)

            if os.path.exists(model_path):
                try:
                    self.models[quarter] = joblib.load(model_path)
                    self.log(f"Loaded Q{quarter} prediction model from {model_path}")
                except Exception as e:
                    self.log(f"Error loading Q{quarter} prediction model from {model_path}: {e}", level="ERROR")
                    self.models[quarter] = None
            else:
                self.log(f"Prediction model for Q{quarter} ('{model_filename}') not found in {model_dir}. Will use fallback.", level="WARNING")
                self.models[quarter] = None

        self._create_fallback_models()

    def _create_fallback_models(self):
        """Create simple Ridge regression models as fallbacks for prediction."""
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
                    self.log(f"Fallback Ridge pipeline created/verified for Q{quarter}.", level="DEBUG")
                except Exception as e:
                    self.log(f"Failed to create fallback model for Q{quarter}: {e}", level="ERROR")

    def predict_quarter(self, X: pd.DataFrame, quarter: int) -> float:
        features_to_use = self.quarter_feature_sets.get(quarter, [])
        cache_key = None
        if features_to_use and all(f in X.columns for f in features_to_use):
            try:
                relevant_data_tuple = tuple(X.iloc[0][features_to_use].fillna(0).values)
                cache_key = f"q{quarter}_{hash(relevant_data_tuple)}"
                if cache_key in self.prediction_cache:
                    self.log(f"Using cached prediction for Q{quarter}", level="DEBUG")
                    return self.prediction_cache[cache_key]
            except Exception as e:
                self.log(f"Could not generate or check cache key: {e}. Proceeding without cache.", level="WARNING")
                cache_key = None
        else:
            self.log("No features defined or available for cache key generation.", level="DEBUG")

        if not (1 <= quarter <= 4):
            self.log(f"Invalid quarter ({quarter}) requested.", level="ERROR")
            return 0.0

        available_features = [f for f in features_to_use if f in X.columns]
        default_score = 25.0
        if hasattr(self.feature_generator, 'league_averages') and isinstance(self.feature_generator.league_averages, dict):
            default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)

        if not available_features:
            self.log(f"No specified features available for Q{quarter} prediction. Using default: {default_score}", level="WARNING")
            if cache_key:
                self.prediction_cache[cache_key] = default_score
            return default_score

        model_to_use = self.models.get(quarter)
        model_source = "primary"
        if model_to_use is None:
            model_to_use = self.fallback_models.get(quarter)
            model_source = "fallback"
            if model_to_use is None:
                self.log(f"No primary or fallback model for Q{quarter}. Using default: {default_score}", level="ERROR")
                if cache_key:
                    self.prediction_cache[cache_key] = default_score
                return default_score

        final_features_used = available_features
        required_model_features = []
        model_instance_for_check = model_to_use.steps[-1][1] if isinstance(model_to_use, Pipeline) else model_to_use
        if hasattr(model_instance_for_check, 'feature_names_in_'):
            required_model_features = list(model_instance_for_check.feature_names_in_)
            if set(available_features) >= set(required_model_features):
                final_features_used = required_model_features
            else:
                missing_req = set(required_model_features) - set(available_features)
                self.log(f"Feature mismatch for Q{quarter} {model_source}. Missing required: {missing_req}. Cannot predict accurately.", level="ERROR")
                if cache_key:
                    self.prediction_cache[cache_key] = default_score
                return default_score
        else:
            self.log(f"Model for Q{quarter} doesn't store feature names. Using available features: {available_features}", level="DEBUG")

        prediction = None
        self.log(f"Predicting Q{quarter} using {model_source} model with {len(final_features_used)} features.", level="DEBUG")
        try:
            prediction_input = X[final_features_used]
            prediction = model_to_use.predict(prediction_input)[0]
        except Exception as e:
            self.log(f"Error during Q{quarter} prediction with {model_source}: {e}", level="ERROR")
            if model_source == "primary":
                self.log(f"Attempting Q{quarter} prediction with fallback pipeline...", level="WARNING")
                fallback_model = self.fallback_models.get(quarter)
                if fallback_model:
                    try:
                        fb_available_features = [f for f in self.quarter_feature_sets.get(quarter, []) if f in X.columns]
                        if not fb_available_features:
                            raise ValueError("No features for fallback")
                        prediction = fallback_model.predict(X[fb_available_features])[0]
                        model_source = "fallback (after primary error)"
                        self.log(f"Successfully used fallback pipeline for Q{quarter}.", level="DEBUG")
                    except Exception as fb_e:
                        self.log(f"Error during Q{quarter} prediction with fallback pipeline: {fb_e}", level="ERROR")
                        prediction = None

        final_prediction = default_score
        if prediction is not None:
            try:
                final_prediction = max(0.0, float(prediction))
            except (ValueError, TypeError):
                self.log(f"Prediction result '{prediction}' is not numeric for Q{quarter}. Using default.", level="ERROR")
                final_prediction = default_score
        else:
            self.log(f"Prediction failed for Q{quarter} even after fallback. Using default score: {final_prediction}", level="ERROR")
            model_source += " -> default (after errors)"

        self.log(f"Q{quarter} Prediction ({model_source}): {final_prediction:.3f}", level="DEBUG")
        if cache_key:
            self.prediction_cache[cache_key] = final_prediction
        return final_prediction

    def predict_remaining_quarters(self, game_data: Dict, current_quarter: int) -> Dict[str, float]:
        if current_quarter >= 4:
            self.log("Game is in Q4 or later, no remaining quarters to predict.", level="DEBUG")
            return {}

        results = {}
        current_game_state_dict = game_data.copy()

        try:
            X = pd.DataFrame([current_game_state_dict])
            if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                X = self.feature_generator.generate_features_for_prediction(X)
            elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                X = self.feature_generator.integrate_advanced_features(X)
            else:
                self.log("Feature generator lacks prediction feature methods.", level="ERROR")
                return {}
        except Exception as e:
            self.log(f"Error generating initial features for remaining quarters: {e}", level="ERROR")
            return {}

        for q in range(max(1, current_quarter + 1), 5):
            try:
                pred_score = self.predict_quarter(X.iloc[[0]], q)
                results[f'q{q}'] = pred_score

                predicted_col_home = f'home_q{q}'
                current_game_state_dict[predicted_col_home] = pred_score
                self.log(f"Regenerating features after predicting Q{q}={pred_score:.2f}...", level="DEBUG")
                X_next = pd.DataFrame([current_game_state_dict])
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X = self.feature_generator.generate_features_for_prediction(X_next)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X = self.feature_generator.integrate_advanced_features(X_next)
                else:
                    self.log("Cannot regenerate features; subsequent predictions may be less accurate.", level="WARNING")
                    if predicted_col_home not in X.columns:
                        X = X.assign(**{predicted_col_home: pred_score})
                    else:
                        X.loc[0, predicted_col_home] = pred_score
            except Exception as e:
                self.log(f"Error predicting or updating features for Q{q}: {e}", level="ERROR")
                default_q_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(q, 25.0)
                results[f'q{q}'] = default_q_score
                current_game_state_dict[f'home_q{q}'] = default_q_score
                try:
                    X_next = pd.DataFrame([current_game_state_dict])
                    if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                        X = self.feature_generator.generate_features_for_prediction(X_next)
                    elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                        X = self.feature_generator.integrate_advanced_features(X_next)
                except Exception as regen_e:
                    self.log(f"Error regenerating features after Q{q} prediction error: {regen_e}. Stopping remaining predictions.", level="ERROR")
                    break

        self.log(f"Predicted remaining quarters ({current_quarter+1}-4): {results}", level="DEBUG")
        return results

    def predict_final_score(self, game_data: Dict, main_model_prediction: Optional[float] = None, weight_manager: Optional[EnsembleWeightManager] = None) -> Tuple[float, float, Dict]:
        self.log("Predicting final score using WeightManager integration...", level="DEBUG")
        league_avg_score = self.feature_generator.league_averages.get('score', 110.0) if hasattr(self.feature_generator, 'league_averages') else 110.0
        fallback_pred = main_model_prediction if main_model_prediction is not None else league_avg_score
        try:
            fallback_pred = float(fallback_pred)
        except (ValueError, TypeError):
            fallback_pred = league_avg_score

        if not isinstance(game_data, dict):
            self.log("Input game_data must be a dictionary.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Invalid game_data type'}
        if weight_manager is None:
            self.log("WeightManager instance IS REQUIRED for predict_final_score.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Missing WeightManager'}
        if not hasattr(weight_manager, 'calculate_ensemble'):
            self.log(f"Provided weight_manager is missing 'calculate_ensemble' method.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Invalid WeightManager instance or method'}

        current_quarter = int(game_data.get('current_quarter', 0))
        home_score_so_far = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        away_score_so_far = sum([float(game_data.get(f'away_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])

        if main_model_prediction is None:
            if current_quarter > 0 and home_score_so_far > 0:
                main_model_prediction = home_score_so_far * (4.0 / max(1, current_quarter))
                self.log(f"Main model prediction missing. Extrapolating: {main_model_prediction:.2f}", level="WARNING")
            else:
                main_model_prediction = league_avg_score
                self.log(f"Main model prediction missing pregame/score=0. Using league average: {main_model_prediction:.2f}", level="WARNING")
        try:
            main_model_prediction = float(main_model_prediction)
        except (ValueError, TypeError):
            self.log(f"Main model prediction ('{main_model_prediction}') invalid. Using league average.", level="ERROR")
            main_model_prediction = league_avg_score

        if current_quarter <= 0:
            self.log("Pregame prediction. Using main model prediction only.", level="DEBUG")
            confidence = 0.5
            breakdown = {
                'main_model_pred': main_model_prediction, 'quarter_model_pred': 0,
                'weights': {'main': 1.0, 'quarter': 0.0}, 'quarter_predictions': {},
                'current_score_home': 0, 'current_score_away': 0, 'score_differential': 0,
                'momentum': 0, 'current_quarter': 0, 'time_remaining_minutes_est': 48.0
            }
            return main_model_prediction, confidence, breakdown

        remaining_quarters_pred = self.predict_remaining_quarters(game_data, current_quarter)
        if not isinstance(remaining_quarters_pred, dict):
            self.log("predict_remaining_quarters failed. Returning main prediction.", level="ERROR")
            return main_model_prediction, 0.3, {'error': 'Failed to predict remaining quarters', 'main_model': main_model_prediction}

        predicted_score_remaining = sum(remaining_quarters_pred.values())
        quarter_sum_prediction = home_score_so_far + predicted_score_remaining

        score_differential = home_score_so_far - away_score_so_far
        momentum = 0.0
        try:
            current_features_df = pd.DataFrame()
            if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                current_features_df = self.feature_generator.generate_features_for_prediction(pd.DataFrame([game_data]))
            elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                current_features_df = self.feature_generator.integrate_advanced_features(pd.DataFrame([game_data]))

            if not current_features_df.empty:
                if 'score_differential' in current_features_df.columns:
                    score_differential = float(current_features_df['score_differential'].iloc[0])
                if 'cumulative_momentum' in current_features_df.columns:
                    momentum = float(current_features_df['cumulative_momentum'].iloc[0])
            else:
                self.log("Feature generation for context returned empty DataFrame. Using score-based defaults.", level="WARNING")
        except Exception as e:
            self.log(f"Error getting context features: {e}. Using score-based defaults.", level="WARNING")

        time_remaining_minutes = max(0.0, 12.0 * (4 - current_quarter))

        try:
            ensemble_pred, confidence, final_weight_main, final_weight_quarter = weight_manager.calculate_ensemble(
                main_prediction=main_model_prediction,
                quarter_prediction=float(quarter_sum_prediction),
                current_quarter=int(current_quarter),
                score_differential=float(score_differential),
                momentum=float(momentum),
                time_remaining_minutes=float(time_remaining_minutes)
            )
        except Exception as e:
            self.log(f"Error calling weight_manager.calculate_ensemble: {e}. Using fallback.", level="ERROR")
            import traceback
            self.log(traceback.format_exc())
            ensemble_pred = fallback_pred
            confidence = 0.3
            final_weight_main = 1.0
            final_weight_quarter = 0.0

        breakdown = {
            'main_model_pred': main_model_prediction,
            'quarter_model_pred': quarter_sum_prediction,
            'weights': {'main': final_weight_main, 'quarter': final_weight_quarter},
            'quarter_predictions': remaining_quarters_pred,
            'current_score_home': home_score_so_far,
            'current_score_away': away_score_so_far,
            'score_differential': score_differential,
            'momentum': momentum,
            'current_quarter': current_quarter,
            'time_remaining_minutes_est': time_remaining_minutes
        }
        self.log(f"Final Ensemble Prediction: {ensemble_pred:.2f} (Confidence: {confidence:.2f})", level="DEBUG")

        try:
            final_ensemble_pred = float(ensemble_pred)
            final_confidence = float(confidence)
        except (ValueError, TypeError):
            self.log("Ensemble prediction or confidence not numeric. Returning fallback.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Non-numeric ensemble result'}

        return final_ensemble_pred, final_confidence, breakdown
