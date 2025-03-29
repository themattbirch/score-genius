# backend/nba_score_prediction/models.py

import pandas as pd
import numpy as np
import os
import joblib
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

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
    xgb = None # Define xgb as None if not available
    XGBOOST_AVAILABLE = False
    # Log this information when the module is loaded
    logging.warning("XGBoost library not found. XGBoostScorePredictor will not be fully functional.")

# --- Type Hinting Placeholders (Replace with actual imports if possible) ---
# from .feature_engineering import NBAFeatureEngine # Example relative import
# from .ensemble import EnsembleWeightManager # Example relative import
NBAFeatureEngine = Any # Placeholder type hint
EnsembleWeightManager = Any # Placeholder type hint


# --- Configuration ---
BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved')
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module


# --- Utility Functions ---

def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """
    Calculate recency weights for samples based on dates. More recent games get higher weights.

    Args:
        dates: Series of game dates. Index should align with features/target data.
        method: Weighting method ('half_life' or 'exponential').
        half_life: Half-life in days for 'half_life' method.
        decay_rate: Decay rate (0-1) for 'exponential' method (per day).

    Returns:
        Array of weights corresponding to each date, or None if input is invalid.
    """
    if not isinstance(dates, pd.Series) or dates.empty:
        logger.warning("Recency weights requested but 'dates' is not a valid Series. Returning None.")
        return None
    if not pd.api.types.is_datetime64_any_dtype(dates):
        try:
            dates = pd.to_datetime(dates, errors='coerce')
        except Exception as e:
             logger.error(f"Error converting dates to datetime: {e}. Returning None for weights.")
             return None
        if dates.isnull().any():
            logger.warning("Some dates could not be parsed. Recency weighting might be affected.")

    # Ensure dates are sorted chronologically for calculation, but return weights in original order
    original_index = dates.index
    # Handle potential NaT values gracefully during sorting
    sorted_dates = dates.sort_values(na_position='first')

    latest_date = sorted_dates.max()
    if pd.isna(latest_date):
        logger.warning("No valid dates found for recency weighting. Returning equal weights.")
        return np.ones(len(dates))

    # Calculate days from latest, handle NaT resulting in large number (inf)
    days_from_latest = (latest_date - sorted_dates).dt.days.astype(float).fillna(np.inf)

    if method == 'half_life':
        if half_life <= 0:
            logger.warning("Half-life must be positive. Using equal weights.")
            weights = np.ones(len(sorted_dates))
        else:
            weights = 0.5 ** (days_from_latest / float(half_life))
    elif method == 'exponential':
        if not (0 < decay_rate < 1):
             logger.warning("Exponential decay_rate must be between 0 and 1. Using equal weights.")
             weights = np.ones(len(sorted_dates))
        else:
            weights = decay_rate ** days_from_latest
    else:
        logger.warning(f"Unknown weighting method: {method}. Using equal weights.")
        weights = np.ones(len(sorted_dates))

    # Handle potential NaNs or Infs in weights (e.g., from NaT dates or calculations)
    weights[~np.isfinite(weights)] = 0.0

    # Normalize weights so the average weight is 1
    mean_weight = np.mean(weights)
    if mean_weight > 1e-9: # Avoid division by zero
        weights = weights / mean_weight
    else:
        # If mean is near zero, weights are likely all zero, return as is.
        logger.warning("Mean weight is near zero; cannot normalize. Returning unnormalized weights (mostly zeros).")

    # Reindex weights to match the original order of the dates Series
    try:
        weights_series = pd.Series(weights, index=sorted_dates.index)
        return weights_series.reindex(original_index).values
    except Exception as e:
         logger.error(f"Error reindexing weights: {e}. Returning None.")
         return None


# --- Base Predictor Class (Optional but good practice) ---
class BaseScorePredictor:
    """Abstract base class for score predictors."""
    def __init__(self, model_dir: str, model_name: str):
        self.model_dir = model_dir
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
             if not filename.endswith(".joblib"): filename += ".joblib"

        save_path = os.path.join(self.model_dir, filename)

        model_data = {
            'pipeline_home': self.pipeline_home,
            'pipeline_away': self.pipeline_away,
            'feature_names_in_': self.feature_names_in_,
            'training_timestamp': self.training_timestamp,
            'training_duration': self.training_duration,
            'model_name': self.model_name,
            'model_class': self.__class__.__name__ # Store class name for validation
        }
        try:
            joblib.dump(model_data, save_path)
            logger.info(f"{self.__class__.__name__} model saved successfully to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving {self.__class__.__name__} model to {save_path}: {e}")
            raise

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        load_path = filepath
        if load_path is None:
            try:
                search_name = model_name or self.model_name
                # Ensure model_dir exists before listing
                if not os.path.isdir(self.model_dir):
                    raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
                files = [f for f in os.listdir(self.model_dir)
                         if f.startswith(search_name) and f.endswith(".joblib")]
                if not files:
                    raise FileNotFoundError(f"No {self.__class__.__name__} model file found for '{search_name}' in {self.model_dir}")
                files.sort(reverse=True)
                load_path = os.path.join(self.model_dir, files[0])
                logger.info(f"No filepath provided, loading latest {self.__class__.__name__} model: {load_path}")
            except Exception as e:
                logger.error(f"Error finding latest {self.__class__.__name__} model file: {e}")
                raise

        if not os.path.exists(load_path):
             raise FileNotFoundError(f"{self.__class__.__name__} Model file not found at {load_path}")

        try:
            model_data = joblib.load(load_path)

            # --- Validation Checks ---
            loaded_class_name = model_data.get('model_class')
            if loaded_class_name and loaded_class_name != self.__class__.__name__:
                 raise TypeError(f"Attempting to load a '{loaded_class_name}' model into a '{self.__class__.__name__}' instance.")
            if 'pipeline_home' not in model_data or 'pipeline_away' not in model_data:
                 raise ValueError("Loaded model data missing required 'pipeline_home' or 'pipeline_away'.")
            if not isinstance(model_data['pipeline_home'], Pipeline) or not isinstance(model_data['pipeline_away'], Pipeline):
                 raise TypeError("Loaded object does not contain valid scikit-learn Pipelines.")
            # Further validation specific to model type done in child classes

            self.pipeline_home = model_data['pipeline_home']
            self.pipeline_away = model_data['pipeline_away']
            self.feature_names_in_ = model_data.get('feature_names_in_') # Use get for safety
            self.training_timestamp = model_data.get('training_timestamp')
            self.training_duration = model_data.get('training_duration')
            self.model_name = model_data.get('model_name', self.model_name)

            if not isinstance(self.feature_names_in_, list) and self.feature_names_in_ is not None:
                 logger.warning("Loaded model 'feature_names_in_' is not a list. Input validation may fail.")
                 self.feature_names_in_ = None # Reset if invalid type

            logger.info(f"{self.__class__.__name__} model loaded successfully from {load_path}")

        except Exception as e:
            logger.error(f"Error loading {self.__class__.__name__} model from {load_path}: {e}")
            raise

    def _common_train_logic(self, X_train, y_train_home, y_train_away, hyperparams_home, hyperparams_away, sample_weights, default_params, fit_params_logic=None):
        """ Encapsulates common training logic. """
        start_time = time.time()
        self.feature_names_in_ = list(X_train.columns)
        logger.info(f"Starting {self.__class__.__name__} training for {self.model_name}...")

        params_home = default_params.copy()
        if hyperparams_home:
            params_home.update(hyperparams_home)
            logger.info("Using custom hyperparameters for home model.")

        params_away = default_params.copy()
        if hyperparams_away:
            params_away.update(hyperparams_away)
            logger.info("Using custom hyperparameters for away model.")

        # Specific model params adjustment (e.g., remove 'normalize' for Ridge)
        params_home.pop('normalize', None)
        params_away.pop('normalize', None)

        self.pipeline_home = self._build_pipeline(params_home)
        self.pipeline_away = self._build_pipeline(params_away)

        fit_kwargs_home = {}
        fit_kwargs_away = {}

        # Handle sample weights (must be passed with step prefix, e.g., 'xgb__sample_weight')
        step_name = self.pipeline_home.steps[-1][0] # Get name of the final step (e.g., 'xgb', 'rf', 'ridge')
        weight_param_name = f"{step_name}__sample_weight"

        if sample_weights is not None:
            if len(sample_weights) == len(X_train):
                fit_kwargs_home[weight_param_name] = sample_weights
                fit_kwargs_away[weight_param_name] = sample_weights
                logger.info(f"Applying sample weights to {step_name} training.")
            else:
                logger.warning(f"Length mismatch: sample_weights ({len(sample_weights)}) vs X_train ({len(X_train)}). Ignoring weights.")

        # Apply model-specific fit parameter logic (like eval_set for XGBoost)
        if fit_params_logic:
            specific_kwargs_home, specific_kwargs_away = fit_params_logic()
            fit_kwargs_home.update(specific_kwargs_home)
            fit_kwargs_away.update(specific_kwargs_away)

        try:
            logger.info("Training home score model...")
            self.pipeline_home.fit(X_train, y_train_home, **fit_kwargs_home)
            logger.info("Training away score model...")
            self.pipeline_away.fit(X_train, y_train_away, **fit_kwargs_away)
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise # Re-raise the exception after logging

        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start_time
        logger.info(f"{self.__class__.__name__} training completed in {self.training_duration:.2f} seconds.")

    def _common_predict_logic(self, X: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ Encapsulates common prediction logic including feature validation. """
        if not self.pipeline_home or not self.pipeline_away:
            logger.error(f"{self.__class__.__name__} models have not been trained or loaded. Call train() or load_model() first.")
            return None

        if self.feature_names_in_:
            if list(X.columns) != self.feature_names_in_:
                logger.warning(f"Feature mismatch for {self.__class__.__name__}. Model trained on {len(self.feature_names_in_)} features ({self.feature_names_in_[:3]}...), input has {len(X.columns)} ({list(X.columns[:3])}...).")
                try:
                    missing = set(self.feature_names_in_) - set(X.columns)
                    extra = set(X.columns) - set(self.feature_names_in_)
                    if missing:
                        logger.error(f"Input missing required features: {missing}. Cannot predict.")
                        return None
                    if extra:
                        logger.warning(f"Input has extra features not seen during training: {extra}. Selecting required features.")
                        X = X[self.feature_names_in_] # Select only needed features in correct order
                    else: # Features match but order might be wrong
                        logger.warning("Input features seem to match but order might differ. Reordering features.")
                        X = X[self.feature_names_in_] # Reorder features

                except KeyError as e:
                     logger.error(f"Input missing required features: {e}. Cannot predict.")
                     return None
                except Exception as e:
                    logger.error(f"Error during feature validation/selection: {e}. Cannot predict.")
                    return None
        else:
            logger.warning(f"{self.__class__.__name__} loaded without feature names. Skipping input feature validation.")

        logger.info(f"Predicting scores using {self.__class__.__name__} for {len(X)} samples...")
        try:
            pred_home = self.pipeline_home.predict(X)
            pred_away = self.pipeline_away.predict(X)
            logger.info(f"{self.__class__.__name__} prediction finished.")
            return pred_home, pred_away
        except Exception as e:
             logger.error(f"Error during {self.__class__.__name__} prediction: {e}")
             import traceback
             logger.error(traceback.format_exc())
             return None


# --- XGBoost Model Definition ---

class XGBoostScorePredictor(BaseScorePredictor):
    """
    Trains and predicts NBA home and away scores using two separate XGBoost models
    within a preprocessing pipeline.
    """
    def __init__(self, model_dir: str = BASE_MODEL_DIR, model_name: str = "xgboost_score_predictor"):
        super().__init__(model_dir, model_name)
        if not XGBOOST_AVAILABLE:
             logger.error("XGBoost library not found. XGBoostScorePredictor cannot be used.")
             # Consider raising an error here or handling it in train/predict
        self._default_xgb_params = {
            'objective': 'reg:squarederror', 'n_estimators': 300,
            'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 3,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
            'reg_alpha': 1.0, 'reg_lambda': 2.0, 'random_state': 42, 'n_jobs': -1
        }

    def _build_pipeline(self, xgb_params: Dict[str, Any]) -> Pipeline:
        """Helper function to build the preprocessing and XGBoost model pipeline."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost library is required for XGBoostScorePredictor.")
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor(**xgb_params))
        ])

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None,
              hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None,
              fit_params: Optional[Dict[str, Any]] = None): # Specific param for XGBoost eval_set etc.
        if not XGBOOST_AVAILABLE:
             raise ImportError("Cannot train XGBoost model: XGBoost library not found.")

        def _xgb_fit_params_logic():
            # Logic to handle XGBoost specific fit_params like eval_set
            kwargs_home = {}
            kwargs_away = {}
            if fit_params:
                if 'xgb__eval_set' in fit_params and isinstance(fit_params['xgb__eval_set'], list):
                    try:
                        # Assume structure: [(X_val, y_val_home, y_val_away)]
                        if len(fit_params['xgb__eval_set'][0]) == 3:
                            X_val, y_val_home, y_val_away = fit_params['xgb__eval_set'][0]
                            kwargs_home['xgb__eval_set'] = [(X_val, y_val_home)]
                            kwargs_away['xgb__eval_set'] = [(X_val, y_val_away)]
                            logger.info("Applying XGBoost eval_set separately for home/away.")
                        else:
                            logger.warning("eval_set in fit_params has unexpected structure. Ignoring.")
                    except Exception as e:
                        logger.warning(f"Could not parse 'xgb__eval_set': {e}. Ignoring.")
                # Copy other params that don't need special handling
                for k, v in fit_params.items():
                    if k != 'xgb__eval_set':
                        kwargs_home[k] = v
                        kwargs_away[k] = v
            return kwargs_home, kwargs_away

        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_xgb_params, _xgb_fit_params_logic)

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """ Predicts scores and applies post-processing. """
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        # Post-processing
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({
            'predicted_home_score': pred_home_processed,
            'predicted_away_score': pred_away_processed
        }, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        """ Loads model and performs XGBoost specific validation. """
        super().load_model(filepath, model_name)
        # Validate the final step is XGBRegressor
        if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], xgb.XGBRegressor):
            raise TypeError("Loaded home pipeline final step is not an XGBRegressor.")
        if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], xgb.XGBRegressor):
            raise TypeError("Loaded away pipeline final step is not an XGBRegressor.")


# --- RandomForest Model Definition ---

class RandomForestScorePredictor(BaseScorePredictor):
    """
    Trains and predicts NBA home and away scores using two separate RandomForest models
    within a preprocessing pipeline.
    """
    def __init__(self, model_dir: str = BASE_MODEL_DIR, model_name: str = "random_forest_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_rf_params = {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
            'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1, 'oob_score': False
        }

    def _build_pipeline(self, rf_params: Dict[str, Any]) -> Pipeline:
        """Helper function to build the preprocessing and RandomForest model pipeline."""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(**rf_params))
        ])

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None,
              hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_rf_params) # No specific fit_params logic needed

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """ Predicts scores and applies post-processing. """
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        # Post-processing
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({
            'predicted_home_score': pred_home_processed,
            'predicted_away_score': pred_away_processed
        }, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        """ Loads model and performs RandomForest specific validation. """
        super().load_model(filepath, model_name)
        if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], RandomForestRegressor):
            raise TypeError("Loaded home pipeline final step is not a RandomForestRegressor.")
        if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], RandomForestRegressor):
            raise TypeError("Loaded away pipeline final step is not a RandomForestRegressor.")


# --- Ridge Model Definition ---

class RidgeScorePredictor(BaseScorePredictor):
    """
    Trains and predicts NBA home and away scores using two separate Ridge regression models
    within a preprocessing pipeline.
    """
    def __init__(self, model_dir: str = BASE_MODEL_DIR, model_name: str = "ridge_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_ridge_params = {
            'alpha': 1.0, 'fit_intercept': True, 'solver': 'auto', 'random_state': 42
        }

    def _build_pipeline(self, ridge_params: Dict[str, Any]) -> Pipeline:
        """Helper function to build the preprocessing and Ridge model pipeline."""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**ridge_params))
        ])

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None,
              hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_ridge_params) # No specific fit_params logic needed

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """ Predicts scores and applies post-processing. """
        predictions = self._common_predict_logic(X)
        if predictions is None:
            return None
        pred_home, pred_away = predictions
        # Post-processing
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({
            'predicted_home_score': pred_home_processed,
            'predicted_away_score': pred_away_processed
        }, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        """ Loads model and performs Ridge specific validation. """
        super().load_model(filepath, model_name)
        if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], Ridge):
            raise TypeError("Loaded home pipeline final step is not a Ridge regressor.")
        if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], Ridge):
            raise TypeError("Loaded away pipeline final step is not a Ridge regressor.")


# --- Quarter-Specific Model System (Prediction Focus) ---

class QuarterSpecificModelSystem:
    """
    Manages and utilizes quarter-specific models for prediction,
    including loading pre-trained models and handling fallbacks.
    """
    def __init__(self, feature_generator: NBAFeatureEngine, debug: bool = False):
        """
        Initialize the system for prediction.

        Args:
            feature_generator: Instance of the feature engineering class (e.g., NBAFeatureEngine).
            debug: Verbosity flag.
        """
        self.feature_generator = feature_generator
        self.debug = debug # Use logger level instead?

        # --- Model Storage (for Prediction) ---
        self.models: Dict[int, Any] = {} # Stores loaded primary models for prediction
        self.fallback_models: Dict[int, Any] = {} # Stores fallback models (Ridge)

        # --- Feature Set Definitions (Used for Prediction) ---
        # Determines the features expected by the prediction methods.
        # Assumes feature generator provides the necessary definitions.
        self.quarter_feature_sets = self._get_consolidated_feature_sets()

        # --- Other Attributes ---
        # Example structure - should ideally be loaded from config/disk or updated dynamically
        self.error_history = {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7}, # Example/Default values
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5} # Example/Default values
        }
        self.prediction_cache: Dict[str, float] = {} # Simple in-memory cache

        self._create_fallback_models()
        self.log("QuarterSpecificModelSystem initialized for prediction.", level="DEBUG")

    def log(self, message, level="INFO"):
        """Log messages using the module's logger."""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[QuarterSystem] {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        """
        Returns the primary feature set expected by prediction models for each quarter,
        derived from the feature generator.
        """
        base_sets = {}
        try:
            if hasattr(self.feature_generator, 'get_prediction_feature_sets'):
                 base_sets = self.feature_generator.get_prediction_feature_sets() # Ideal method
                 if not isinstance(base_sets, dict): base_sets = {}
            elif hasattr(self.feature_generator, '_get_optimized_feature_sets'): # Fallback access
                 base_sets = self.feature_generator._get_optimized_feature_sets()
                 if not isinstance(base_sets, dict): base_sets = {}
            elif hasattr(self.feature_generator, 'feature_sets'): # Alternative access pattern
                 base_sets = self.feature_generator.feature_sets
                 if not isinstance(base_sets, dict): base_sets = {}
            else:
                 self.log("Feature generator missing expected feature set definitions. Using empty sets.", level="WARNING")

            # Ensure keys are integers 1-4
            final_sets = {q: base_sets.get(q, []) for q in range(1, 5)}

            self.log(f"Consolidated prediction feature sets from generator: Q1({len(final_sets.get(1,[]))}), Q2({len(final_sets.get(2,[]))}), Q3({len(final_sets.get(3,[]))}), Q4({len(final_sets.get(4,[]))}) features.", level="DEBUG")
            return final_sets

        except Exception as e:
            self.log(f"Error getting feature sets from generator: {e}. Using empty sets.", level="ERROR")
            return {q: [] for q in range(1, 5)}


    def load_models(self, model_dir: Optional[str] = None):
        """
        Load pre-trained quarter-specific models (e.g., q1_home_model.pkl, q1_away_model.pkl)
        or potentially a single model per quarter (e.g., q1_model.pkl if it predicts both).
        Adjust logic based on how models are saved by the training script.

        Args:
            model_dir: Directory containing the quarter model files. Defaults to 'quarterly' subdir under BASE_MODEL_DIR.
        """
        if model_dir is None:
             model_dir = os.path.join(BASE_MODEL_DIR, 'quarterly') # Default subdirectory

        self.log(f"Loading quarter prediction models from '{model_dir}'...")
        os.makedirs(model_dir, exist_ok=True) # Ensure directory exists

        for quarter in range(1, 5):
            # --- Adapt this logic based on how your training script saves models ---
            # Option A: Single model per quarter (predicts home/away or total)
            model_path = os.path.join(model_dir, f'q{quarter}_model.pkl')
            # Option B: Separate home/away models per quarter
            # model_path_home = os.path.join(model_dir, f'q{quarter}_home_model.pkl')
            # model_path_away = os.path.join(model_dir, f'q{quarter}_away_model.pkl')
            # --- Choose one option based on your training output ---

            # Example using Option A:
            if os.path.exists(model_path):
                try:
                    self.models[quarter] = joblib.load(model_path)
                    self.log(f"Loaded Q{quarter} prediction model from {model_path}")
                except Exception as e:
                    self.log(f"Error loading Q{quarter} prediction model from {model_path}: {e}", level="ERROR")
                    self.models[quarter] = None
            else:
                self.log(f"Prediction model for Q{quarter} not found at {model_path}. Will use fallback.", level="WARNING")
                self.models[quarter] = None
            # --- End Example Option A ---

            # If using Option B, you'd load home/away separately and store them,
            # possibly modifying self.models structure: self.models[quarter] = {'home': model_h, 'away': model_a}

        self._create_fallback_models() # Ensure fallbacks exist

    def _create_fallback_models(self):
        """Create simple Ridge regression models as fallbacks for prediction."""
        self.log("Ensuring fallback Ridge models are available...", level="DEBUG")
        for quarter in range(1, 5):
            if quarter not in self.fallback_models:
                 try:
                    # Build a minimal Ridge pipeline for fallback
                    # Needs imputer/scaler if features aren't guaranteed clean
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=1.0, random_state=42))
                    ])
                    # NOTE: Fallback models are not trained here, they predict based on Ridge defaults.
                    # For better fallbacks, consider training simple Ridge models on basic features
                    # during the main training process and saving them as specific fallback models.
                    self.fallback_models[quarter] = model
                    self.log(f"Fallback Ridge pipeline created/verified for Q{quarter}.", level="DEBUG")
                 except Exception as e:
                      self.log(f"Failed to create fallback model for Q{quarter}: {e}", level="ERROR")

    def predict_quarter(self, X: pd.DataFrame, quarter: int) -> float:
        """
        Predict score for a specific quarter using the loaded model or fallback.

        Args:
            X: DataFrame containing features for the game (should be a single row).
            quarter: The quarter number (1-4) to predict.

        Returns:
            Predicted score for the quarter (e.g., predicted home score or predicted total score).
            Adapt the return/logic if predicting home & away separately here.
        """
        # --- Caching ---
        # Hash relevant features for cache key
        features_to_use = self.quarter_feature_sets.get(quarter, [])
        cache_key = None
        if features_to_use and all(f in X.columns for f in features_to_use):
             try:
                 relevant_data_tuple = tuple(X.iloc[0][features_to_use].fillna(0).values) # Handle NaN before hashing
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

        # Determine default score
        default_score = 25.0
        if hasattr(self.feature_generator, 'league_averages') and isinstance(self.feature_generator.league_averages, dict):
            default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)

        if not available_features:
            self.log(f"No specified features available for Q{quarter} prediction. Using default: {default_score}", level="WARNING")
            if cache_key: self.prediction_cache[cache_key] = default_score
            return default_score

        # --- Model Selection ---
        model_to_use = self.models.get(quarter)
        model_source = "primary"
        if model_to_use is None:
            model_to_use = self.fallback_models.get(quarter)
            model_source = "fallback"
            if model_to_use is None:
                 self.log(f"No primary or fallback model for Q{quarter}. Using default: {default_score}", level="ERROR")
                 if cache_key: self.prediction_cache[cache_key] = default_score
                 return default_score

        # --- Feature Preparation ---
        final_features_used = available_features
        required_model_features = []
        # Check if the model is a Pipeline and has feature names requirements
        model_instance_for_check = model_to_use.steps[-1][1] if isinstance(model_to_use, Pipeline) else model_to_use
        if hasattr(model_instance_for_check, 'feature_names_in_'):
            required_model_features = list(model_instance_for_check.feature_names_in_)
            if set(available_features) >= set(required_model_features): # Check if we have at least the required ones
                 final_features_used = required_model_features # Use the exact set model expects, in order
                 # Note: The pipeline's imputer/scaler will run on these features
            else:
                 missing_req = set(required_model_features) - set(available_features)
                 self.log(f"Feature mismatch for Q{quarter} {model_source}. Missing required: {missing_req}. Cannot predict accurately.", level="ERROR")
                 # Fallback to default or attempt with available (risky)? Let's fallback.
                 if cache_key: self.prediction_cache[cache_key] = default_score
                 return default_score
        else:
             # Model doesn't store feature names, assume available_features is okay
             self.log(f"Model for Q{quarter} doesn't store feature names. Using available features: {available_features}", level="DEBUG")

        # --- Prediction ---
        prediction = None
        self.log(f"Predicting Q{quarter} using {model_source} model with {len(final_features_used)} features.", level="DEBUG")
        try:
            # Predict using the selected features (Pipeline handles preprocessing)
            # Ensure X contains the final_features_used before prediction
            prediction_input = X[final_features_used]
            prediction = model_to_use.predict(prediction_input)[0]

        except Exception as e:
            self.log(f"Error during Q{quarter} prediction with {model_source}: {e}", level="ERROR")
            # If primary model fails, try fallback (if not already using it)
            if model_source == "primary":
                 self.log(f"Attempting Q{quarter} prediction with fallback pipeline...", level="WARNING")
                 fallback_model = self.fallback_models.get(quarter)
                 if fallback_model:
                     try:
                         # Use available features for fallback pipeline
                         fb_available_features = [f for f in self.quarter_feature_sets.get(quarter, []) if f in X.columns]
                         if not fb_available_features: raise ValueError("No features for fallback")
                         prediction = fallback_model.predict(X[fb_available_features])[0]
                         model_source = "fallback (after primary error)"
                         self.log(f"Successfully used fallback pipeline for Q{quarter}.", level="DEBUG")
                     except Exception as fb_e:
                          self.log(f"Error during Q{quarter} prediction with fallback pipeline: {fb_e}", level="ERROR")
                          prediction = None # Fallback also failed

        # --- Post-processing & Caching ---
        final_prediction = default_score # Start with default
        if prediction is not None:
            try:
                final_prediction = max(0.0, float(prediction))
                # Optional: Add cap? e.g., min(final_prediction, 60.0)
            except (ValueError, TypeError):
               self.log(f"Prediction result '{prediction}' is not numeric for Q{quarter}. Using default.", level="ERROR")
               final_prediction = default_score # Revert to default if conversion fails
        else:
             self.log(f"Prediction failed for Q{quarter} even after fallback. Using default score: {final_prediction}", level="ERROR")
             model_source += " -> default (after errors)"

        self.log(f"Q{quarter} Prediction ({model_source}): {final_prediction:.3f}", level="DEBUG")
        if cache_key: self.prediction_cache[cache_key] = final_prediction
        return final_prediction


    def predict_remaining_quarters(self, game_data: Dict, current_quarter: int) -> Dict[str, float]:
        """
        Predict scores for all quarters remaining in the game.
        Handles feature regeneration between quarter predictions if necessary.

        Args:
            game_data: Dictionary representing the current state of the game.
            current_quarter: The current quarter number (0 if pre-game, 1-4 during game).

        Returns:
            Dictionary mapping remaining quarter numbers (str) to predicted scores. {}.
        """
        if current_quarter >= 4:
            self.log("Game is in Q4 or later, no remaining quarters to predict.", level="DEBUG")
            return {}

        results = {}
        # Create a copy to modify for iterative predictions
        current_game_state_dict = game_data.copy()

        # Initial feature generation based on known game state
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

        # Predict sequentially for remaining quarters
        for q in range(max(1, current_quarter + 1), 5):
            try:
                pred_score = self.predict_quarter(X.iloc[[0]], q)
                results[f'q{q}'] = pred_score

                # --- Update state and Regenerate features for NEXT prediction ---
                # Add the predicted score to the dictionary representation of the game state
                predicted_col_home = f'home_q{q}' # Assuming prediction is home score
                current_game_state_dict[predicted_col_home] = pred_score
                # Add away score prediction if available/needed for features

                # Regenerate features using the updated game state dictionary
                # This simulates the game progressing with the predicted score
                self.log(f"Regenerating features after predicting Q{q}={pred_score:.2f}...", level="DEBUG")
                X_next = pd.DataFrame([current_game_state_dict]) # Create new DF from updated dict
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X = self.feature_generator.generate_features_for_prediction(X_next)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X = self.feature_generator.integrate_advanced_features(X_next)
                else:
                     # If no feature generator, we can't update features based on prediction
                     self.log("Cannot regenerate features; subsequent predictions may be less accurate.", level="WARNING")
                     # Update the existing DataFrame X directly (less safe if complex features exist)
                     if predicted_col_home not in X.columns:
                         X = X.assign(**{predicted_col_home: pred_score})
                     else:
                         X.loc[0, predicted_col_home] = pred_score


            except Exception as e:
                 self.log(f"Error predicting or updating features for Q{q}: {e}", level="ERROR")
                 default_q_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(q, 25.0)
                 results[f'q{q}'] = default_q_score
                 # Attempt to update state with default and regenerate features to continue
                 current_game_state_dict[f'home_q{q}'] = default_q_score
                 try:
                    X_next = pd.DataFrame([current_game_state_dict])
                    if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                        X = self.feature_generator.generate_features_for_prediction(X_next)
                    elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                        X = self.feature_generator.integrate_advanced_features(X_next)
                 except Exception as regen_e:
                    self.log(f"Error regenerating features after Q{q} prediction error: {regen_e}. Stopping remaining predictions.", level="ERROR")
                    break # Stop predicting further quarters if regeneration fails critically

        self.log(f"Predicted remaining quarters ({current_quarter+1}-4): {results}", level="DEBUG")
        return results


    def predict_final_score(self, game_data: Dict, main_model_prediction: Optional[float] = None, weight_manager: Optional[EnsembleWeightManager] = None) -> Tuple[float, float, Dict]:
        """
        Predict the final score using an ensemble managed by EnsembleWeightManager.

        Args:
            game_data: Dictionary representing the current state of the game.
            main_model_prediction: Optional prediction from the main full-game model.
            weight_manager: REQUIRED instance of EnsembleWeightManager.

        Returns:
            Tuple: (ensemble_prediction, confidence, breakdown_dict)
                   or (fallback_prediction, 0.0, {'error': ...}) on critical errors.
        """
        self.log("Predicting final score using WeightManager integration...", level="DEBUG")
        # Determine a reasonable fallback prediction early
        league_avg_score = self.feature_generator.league_averages.get('score', 110.0) if hasattr(self.feature_generator, 'league_averages') else 110.0
        fallback_pred = main_model_prediction if main_model_prediction is not None else league_avg_score
        try: fallback_pred = float(fallback_pred)
        except (ValueError, TypeError): fallback_pred = league_avg_score

        # --- Input Validation ---
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

        # Calculate score accumulated so far safely
        home_score_so_far = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        away_score_so_far = sum([float(game_data.get(f'away_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])

        # --- Get Main Model Prediction (Handle if missing) ---
        if main_model_prediction is None:
             if current_quarter > 0 and home_score_so_far > 0:
                 main_model_prediction = home_score_so_far * (4.0 / max(1, current_quarter))
                 self.log(f"Main model prediction missing. Extrapolating: {main_model_prediction:.2f}", level="WARNING")
             else:
                 main_model_prediction = league_avg_score
                 self.log(f"Main model prediction missing pregame/score=0. Using league average: {main_model_prediction:.2f}", level="WARNING")
        try: main_model_prediction = float(main_model_prediction)
        except (ValueError, TypeError):
             self.log(f"Main model prediction ('{main_model_prediction}') invalid. Using league average.", level="ERROR")
             main_model_prediction = league_avg_score

        # --- Handle Pre-game Scenario ---
        if current_quarter <= 0:
            self.log("Pregame prediction. Using main model prediction only.", level="DEBUG")
            confidence = 0.5 # Example pregame confidence
            breakdown = {
                'main_model_pred': main_model_prediction, 'quarter_model_pred': 0,
                'weights': {'main': 1.0, 'quarter': 0.0}, 'quarter_predictions': {},
                'current_score_home': 0, 'current_score_away': 0, 'score_differential': 0,
                'momentum': 0, 'current_quarter': 0, 'time_remaining_minutes_est': 48.0
            }
            return main_model_prediction, confidence, breakdown

        # --- Get Quarter Model Prediction Component ---
        remaining_quarters_pred = self.predict_remaining_quarters(game_data, current_quarter)
        if not isinstance(remaining_quarters_pred, dict): # Check for critical failure
             self.log("predict_remaining_quarters failed. Returning main prediction.", level="ERROR")
             return main_model_prediction, 0.3, {'error': 'Failed to predict remaining quarters', 'main_model': main_model_prediction}

        predicted_score_remaining = sum(remaining_quarters_pred.values())
        # Assuming prediction target is HOME score. Adapt if predicting total.
        quarter_sum_prediction = home_score_so_far + predicted_score_remaining

        # --- Gather Context Features for Weight Manager ---
        score_differential = home_score_so_far - away_score_so_far
        momentum = 0.0
        try:
             current_features_df = pd.DataFrame() # Init empty
             if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                  current_features_df = self.feature_generator.generate_features_for_prediction(pd.DataFrame([game_data]))
             elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                  current_features_df = self.feature_generator.integrate_advanced_features(pd.DataFrame([game_data]))

             if not current_features_df.empty:
                if 'score_differential' in current_features_df.columns: score_differential = float(current_features_df['score_differential'].iloc[0])
                if 'cumulative_momentum' in current_features_df.columns: momentum = float(current_features_df['cumulative_momentum'].iloc[0])
                # Extract other needed context features...
             else:
                 self.log("Feature generation for context returned empty DataFrame. Using score-based defaults.", level="WARNING")

        except Exception as e:
             self.log(f"Error getting context features: {e}. Using score-based defaults.", level="WARNING")

        # Estimate time remaining
        time_remaining_minutes = max(0.0, 12.0 * (4 - current_quarter)) # Simple estimate, ensures non-negative

        # --- Calculate Ensemble Prediction using WeightManager ---
        try:
             ensemble_pred, confidence, final_weight_main, final_weight_quarter = weight_manager.calculate_ensemble(
                 main_prediction=main_model_prediction, # Already float
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
             ensemble_pred = fallback_pred # Use pre-calculated fallback
             confidence = 0.3 # Low confidence on error
             final_weight_main = 1.0
             final_weight_quarter = 0.0

        # --- Compile Breakdown ---
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

        # Final type check before returning
        try:
            final_ensemble_pred = float(ensemble_pred)
            final_confidence = float(confidence)
        except (ValueError, TypeError):
           self.log("Ensemble prediction or confidence not numeric. Returning fallback.", level="ERROR")
           return fallback_pred, 0.0, {'error': 'Non-numeric ensemble result'}

        return final_ensemble_pred, final_confidence, breakdown