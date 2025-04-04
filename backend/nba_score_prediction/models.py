# backend/nba_score_prediction/models.py
import pandas as pd
import os
import numpy as np
import joblib
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
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
# Ensure logger is configured properly (might be redundant if configured elsewhere, but safe)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger specific to this module
# !!! END CONFIGURATION SECTION !!!

# --- Utility Functions ---
def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """ Calculate recency weights for samples based on dates. """
    if not isinstance(dates, pd.Series) or dates.empty:
        logger.warning("Recency weights: input not a valid Series.")
        return None
    if not pd.api.types.is_datetime64_any_dtype(dates):
        try:
            # Attempt conversion, be timezone naive for calculation consistency
            dates = pd.to_datetime(dates, errors='coerce').dt.tz_localize(None)
        except Exception as e:
            logger.error(f"Recency weights: date conversion error: {e}")
            return None
    original_index = dates.index
    valid_dates = dates.dropna()
    if valid_dates.empty:
        logger.warning("Recency weights: no valid dates after dropna.")
        # Return array of ones matching original length if no valid dates
        return np.ones(len(dates))

    sorted_dates = valid_dates.sort_values()
    latest_date = sorted_dates.max()
    if pd.isna(latest_date):
        logger.warning("Recency weights: no valid latest date found.")
        # Return array of ones matching original length
        return np.ones(len(dates))

    # Calculate difference in days
    days_from_latest = (latest_date - sorted_dates).dt.total_seconds() / (3600 * 24.0)
    days_from_latest = days_from_latest.astype(float) # Ensure float type

    weights = np.ones(len(sorted_dates)) # Initialize weights

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

    # Handle potential numerical issues and normalize
    weights[~np.isfinite(weights)] = 0.0 # Replace Inf/NaN with 0
    mean_weight = np.mean(weights) if len(weights) > 0 else 0.0
    if mean_weight > 1e-9: # Avoid division by zero or near-zero
        weights = weights / mean_weight
    else:
        logger.warning("Recency weights: Mean weight near zero. Weights might not be meaningful.")
        weights = np.ones(len(sorted_dates)) # Fallback to equal weights if mean is zero

    # Map weights back to the original index, filling missing/dropped dates with 0 weight
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
        raise NotImplementedError("Subclasses must implement _build_pipeline")

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series, **kwargs):
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
            # Ensure filename ends with .joblib
            if not filename.endswith(".joblib"):
                filename += ".joblib"

        save_path = Path(self.model_dir) / filename # Use pathlib for path joining

        # Ensure feature_names_in_ is correctly set from training before saving
        if self.feature_names_in_ is None:
            logger.warning(f"Attempting to save model {self.model_name} but feature_names_in_ is not set. This might cause issues during loading/prediction.")

        model_data = {
            'pipeline_home': self.pipeline_home,
            'pipeline_away': self.pipeline_away,
            'feature_names_in_': self.feature_names_in_, # Save the list used during training
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

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        load_path_str = filepath
        if load_path_str is None:
            try:
                search_name = model_name or self.model_name
                model_dir_path = Path(self.model_dir)
                if not model_dir_path.is_dir():
                    raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

                # Find files matching the pattern
                files = [f for f in model_dir_path.glob(f"{search_name}*.joblib")]
                if not files:
                    raise FileNotFoundError(f"No model file found starting with '{search_name}' in {self.model_dir}")

                # Sort by modification time (most recent first) or filename timestamp
                # Let's use filename timestamp assuming format is consistent
                def get_timestamp_from_filename(path: Path) -> datetime:
                    try:
                        # Example: model_name_tuned_YYYYMMDD_HHMMSS.joblib
                        timestamp_str = path.stem.split('_')[-2] + "_" + path.stem.split('_')[-1]
                        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except (IndexError, ValueError):
                        # Fallback if timestamp format is different or missing
                        return datetime.fromtimestamp(path.stat().st_mtime)

                files.sort(key=get_timestamp_from_filename, reverse=True)
                load_path_str = str(files[0])
                logger.info(f"Loading latest model: {load_path_str}")

            except Exception as e:
                logger.error(f"Error finding latest model file: {e}", exc_info=True)
                raise

        load_path = Path(load_path_str) if load_path_str else None
        if not load_path or not load_path.is_file():
            raise FileNotFoundError(f"Model file not found at specified path: {load_path}")

        try:
            model_data = joblib.load(load_path)

            # --- Validation ---
            loaded_class_name = model_data.get('model_class')
            if loaded_class_name and loaded_class_name != self.__class__.__name__:
                raise TypeError(f"Loaded model class ({loaded_class_name}) does not match expected ({self.__class__.__name__}).")

            if 'pipeline_home' not in model_data or 'pipeline_away' not in model_data:
                raise ValueError("Loaded model file is missing required pipeline data ('pipeline_home' or 'pipeline_away').")

            if not isinstance(model_data['pipeline_home'], Pipeline) or not isinstance(model_data['pipeline_away'], Pipeline):
                raise TypeError("Loaded pipelines are not valid sklearn Pipeline objects.")

            required_features = model_data.get('feature_names_in_')
            if required_features is not None and not isinstance(required_features, list):
                logger.warning("Feature names in loaded model are not a list; correcting.")
                # Attempt to correct if it's iterable, otherwise reset
                try: self.feature_names_in_ = list(required_features)
                except TypeError: self.feature_names_in_ = None
            else:
                 self.feature_names_in_ = required_features

            if self.feature_names_in_ is None:
                 logger.error(f"CRITICAL: Loaded model {self.model_name} from {load_path} is missing the feature list ('feature_names_in_'). Prediction will likely fail.")
            else:
                 logger.info(f"Loaded model includes feature list with {len(self.feature_names_in_)} features.")

            # --- Assignment ---
            self.pipeline_home = model_data['pipeline_home']
            self.pipeline_away = model_data['pipeline_away']
            self.training_timestamp = model_data.get('training_timestamp')
            self.training_duration = model_data.get('training_duration')
            self.model_name = model_data.get('model_name', self.model_name) # Update model name if stored

            logger.info(f"{self.__class__.__name__} model loaded successfully from {load_path}")

        except Exception as e:
            logger.error(f"Error loading model from {load_path}: {e}", exc_info=True)
            raise

    def _common_train_logic(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
                              hyperparams_home: Optional[Dict[str, Any]],
                              hyperparams_away: Optional[Dict[str, Any]],
                              sample_weights: Optional[np.ndarray],
                              default_params: Dict[str, Any], # Pass model defaults here
                              fit_params: Optional[Dict[str, Any]] = None,
                              eval_set_data: Optional[Tuple] = None):
        """ Common logic for training home and away pipelines. """
        start_time = time.time()
        # --- DEBUG: Log input shape ---
        logger.debug(f"Shape of X_train at start of _common_train_logic: {X_train.shape}")

        try:
            # Ensure only numeric features are used for training internal pipeline steps
            X_train_numeric = X_train.select_dtypes(include=np.number)
            if X_train_numeric.shape[1] != X_train.shape[1]:
                dropped_cols = set(X_train.columns) - set(X_train_numeric.columns)
                logger.warning(f"_common_train_logic received non-numeric columns: {dropped_cols}. Using only numeric.")
                # IMPORTANT: Use the numeric subset for fitting, but store original columns if needed elsewhere
                X_train_for_fit = X_train_numeric
            else:
                X_train_for_fit = X_train # All columns were numeric

            # --- DEBUG: Log shape after numeric selection ---
            logger.debug(f"Shape of X_train after numeric selection (X_train_for_fit): {X_train_for_fit.shape}")

            # Store the list of feature names ACTUALLY used for fitting the pipeline
            self.feature_names_in_ = list(X_train_for_fit.columns)
            # --- DEBUG: Log the count of features being used ---
            logger.info(f"Model feature_names_in_ set with {len(self.feature_names_in_)} features inside _common_train_logic.")
            logger.debug(f"Features being used for fit: {self.feature_names_in_[:15]}...") # Log sample

        except Exception as e:
            logger.error(f"Error processing X_train columns for feature list: {e}", exc_info=True)
            # Fallback: attempt to use original columns, but this might fail later
            self.feature_names_in_ = list(X_train.columns)
            X_train_for_fit = X_train # Use original in fallback case
            logger.error(f"FALLBACK: Using original {len(self.feature_names_in_)} features. Training may fail.")

        if not self.feature_names_in_:
             logger.error("CRITICAL: No features determined for training. Aborting train logic.")
             return # Cannot proceed without features

        logger.info(f"Starting {self.__class__.__name__} training for {self.model_name}...")

        # Prepare parameters, ensuring 'normalize' (if present) is handled correctly or removed if not needed
        params_home = default_params.copy()
        params_away = default_params.copy()
        if hyperparams_home: params_home.update(hyperparams_home); logger.info("Using custom hyperparameters for home.")
        if hyperparams_away: params_away.update(hyperparams_away); logger.info("Using custom hyperparameters for away.")
        params_home.pop('normalize', None); params_away.pop('normalize', None) # Remove if exists, as StandardScaler handles it

        # Build pipelines
        try:
            self.pipeline_home = self._build_pipeline(params_home)
            self.pipeline_away = self._build_pipeline(params_away)
        except Exception as build_e:
            logger.error(f"Failed to build pipeline(s): {build_e}", exc_info=True)
            return # Cannot proceed if pipelines fail to build

        # Prepare fit keyword arguments (e.g., sample weights, eval_set)
        fit_kwargs_home = fit_params.copy() if fit_params else {}
        fit_kwargs_away = fit_params.copy() if fit_params else {}

        # --- Handle Sample Weights ---
        if sample_weights is not None:
            if len(sample_weights) == len(X_train_for_fit):
                # Determine the correct parameter name (e.g., 'xgb__sample_weight')
                try:
                    # Get the name of the final step (the estimator)
                    final_estimator_name = self.pipeline_home.steps[-1][0]
                    weight_param_name = f"{final_estimator_name}__sample_weight"
                    fit_kwargs_home[weight_param_name] = sample_weights
                    fit_kwargs_away[weight_param_name] = sample_weights
                    logger.info(f"Applying sample weights using parameter name: {weight_param_name}")
                except (AttributeError, IndexError, TypeError) as sw_e:
                    logger.error(f"Could not determine sample weight parameter name: {sw_e}. Weights ignored.")
            else:
                logger.warning(f"Sample weights length ({len(sample_weights)}) mismatch with training data ({len(X_train_for_fit)}). Ignoring weights.")

        # --- Handle XGBoost Specific Eval Set ---
        is_xgb = False
        final_estimator_home = None
        try: # Safely access the final estimator instance
            final_estimator_home = self.pipeline_home.steps[-1][1]
            if XGBOOST_AVAILABLE and isinstance(final_estimator_home, xgb.XGBRegressor):
                is_xgb = True
        except (AttributeError, IndexError, TypeError): pass # Ignore if pipeline structure is unexpected

        if is_xgb and eval_set_data:
            logger.info("Preparing XGBoost eval_set...")
            if isinstance(eval_set_data, tuple) and len(eval_set_data) == 3:
                try:
                    X_val, y_val_home, y_val_away = eval_set_data
                    # Ensure validation features match training features *used for fit*
                    X_val_aligned = X_val[self.feature_names_in_] # Use the stored feature list
                    # Align validation targets with the index of the aligned validation features
                    y_val_home_aligned = y_val_home.loc[X_val_aligned.index]
                    y_val_away_aligned = y_val_away.loc[X_val_aligned.index]

                    # Add eval_set to fit_kwargs, using the correct step name prefix
                    final_estimator_name = self.pipeline_home.steps[-1][0] # Should be 'xgb' based on XGB pipeline
                    fit_kwargs_home[f'{final_estimator_name}__eval_set'] = [(X_val_aligned.values, y_val_home_aligned.values)] # Pass numpy arrays if needed by xgb
                    fit_kwargs_away[f'{final_estimator_name}__eval_set'] = [(X_val_aligned.values, y_val_away_aligned.values)]
                    logger.info(f"Applying XGBoost eval_set separately for home/away using prefix '{final_estimator_name}__'.")

                    # Add early stopping rounds if desired (passed via fit_params initially)
                    # Example: if 'early_stopping_rounds' in fit_params:
                    #   fit_kwargs_home[f'{final_estimator_name}__early_stopping_rounds'] = fit_params['early_stopping_rounds']
                    #   fit_kwargs_away[f'{final_estimator_name}__early_stopping_rounds'] = fit_params['early_stopping_rounds']

                except KeyError as ke:
                    logger.error(f"Eval set feature mismatch or target alignment error: {ke}. Check feature names used in training vs validation. Ignoring eval_set.", exc_info=True)
                except Exception as e:
                    logger.warning(f"Could not parse/align 'eval_set_data': {e}. Ignoring eval_set.", exc_info=True)
            else:
                logger.warning("Incorrect eval_set_data format provided (expected Tuple[DataFrame, Series, Series]). Ignoring.")

        # --- Fit the Pipelines ---
        try:
            logger.info(f"Training home score model... Fit args keys: {list(fit_kwargs_home.keys())}")
            # Fit using the numeric features DataFrame
            self.pipeline_home.fit(X_train_for_fit, y_train_home, **fit_kwargs_home)

            logger.info(f"Training away score model... Fit args keys: {list(fit_kwargs_away.keys())}")
            self.pipeline_away.fit(X_train_for_fit, y_train_away, **fit_kwargs_away)

        except Exception as e:
            logger.error(f"Error during model fitting: {e}", exc_info=True)
            # No need to import traceback again here, just raise
            raise

        # --- Finalize ---
        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start_time
        logger.info(f"{self.__class__.__name__} training completed in {self.training_duration:.2f} seconds.")


    def _common_predict_logic(self, X: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ Common logic for validating input and predicting with trained pipelines. """
        # --- DEBUG: Log input shape ---
        logger.debug(f"Shape of X received by _common_predict_logic: {X.shape}")

        if not self.pipeline_home or not self.pipeline_away:
            logger.error(f"Prediction failed for {self.model_name}: pipelines are not available (not trained or loaded).")
            return None

        # Use the feature list stored *during training*
        required_features = self.feature_names_in_
        if not required_features:
            logger.error(f"Prediction failed for {self.model_name}: Model was not trained with a feature list ('feature_names_in_' is None).")
            return None

        logger.debug(f"Model expects {len(required_features)} features: {required_features[:15]}...")

        # --- Feature Validation and Alignment ---
        missing_cols = set(required_features) - set(X.columns)
        extra_cols = set(X.columns) - set(required_features)

        if missing_cols:
            logger.error(f"Prediction failed for {self.model_name}: Input DataFrame is missing required features: {missing_cols}.")
            return None

        if extra_cols:
            logger.warning(f"Prediction input for {self.model_name} has {len(extra_cols)} extra columns: {list(extra_cols)[:10]}... Selecting only required features.")
            # Select only required features IN THE CORRECT ORDER
            try:
                 X_predict = X[required_features]
            except Exception as slice_e:
                 logger.error(f"Error selecting required features during prediction: {slice_e}", exc_info=True)
                 return None
        elif list(X.columns) != required_features:
            logger.warning(f"Predict input features match for {self.model_name} but order differs. Reordering.")
            # Reorder columns to match the order seen during training
            try:
                 X_predict = X[required_features]
            except Exception as reorder_e:
                 logger.error(f"Error reordering features during prediction: {reorder_e}", exc_info=True)
                 return None
        else:
            # Columns match and are in the correct order
            X_predict = X

        # --- DEBUG: Log shape before prediction ---
        logger.debug(f"Shape of X_predict before passing to pipeline: {X_predict.shape}")

        if X_predict.shape[1] != len(required_features):
             logger.error(f"CRITICAL INTERNAL ERROR: X_predict shape ({X_predict.shape[1]}) doesn't match required features ({len(required_features)}) after validation/selection.")
             return None

        # --- Prediction ---
        logger.info(f"Predicting scores using {self.model_name} for {len(X_predict)} samples...")
        try:
            pred_home = self.pipeline_home.predict(X_predict)
            pred_away = self.pipeline_away.predict(X_predict)
            logger.info(f"{self.model_name} prediction finished.")
            return pred_home, pred_away
        except Exception as e:
            # Error already includes traceback from the original exception context
            logger.error(f"Error during prediction call for {self.model_name}: {e}", exc_info=True)
            return None


# --- XGBoost Model Definition ---
class XGBoostScorePredictor(BaseScorePredictor):
    """ Trains and predicts using XGBoost models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "xgboost_score_predictor"):
        super().__init__(model_dir, model_name)
        if not XGBOOST_AVAILABLE:
            # Log error but don't raise immediately, allow object creation for dummy use cases
            logger.error("XGBoost library not found. XGBoostScorePredictor will not function.")
        # Default parameters - these might be overridden by tuning
        self._default_xgb_params = {
            'objective': 'reg:squarederror', 'n_estimators': 300, 'learning_rate': 0.05,
            'max_depth': 5, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.1, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'random_state': 42, 'n_jobs': -1
        }

    def _build_pipeline(self, xgb_params: Dict[str, Any]) -> Pipeline:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost library required to build XGBoost pipeline.")
        # Combine default and provided params, ensuring provided ones take precedence
        final_params = self._default_xgb_params.copy(); final_params.update(xgb_params)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor(**final_params)) # Use combined params
        ])
        try: pipeline.set_output(transform="pandas"); logger.debug("Set XGBoost pipeline output pandas.")
        except AttributeError: logger.debug("Could not set pipeline output pandas (sklearn < 1.2).")
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None, fit_params: Optional[Dict[str, Any]] = None,
              eval_set_data: Optional[Tuple] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("Cannot train: XGBoost library not found.")
        # Pass defaults to common logic; hyperparams override them inside
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_xgb_params, # Pass model defaults
                                 fit_params=fit_params, eval_set_data=eval_set_data)

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None: return None
        pred_home, pred_away = predictions
        # Ensure non-negative integer predictions
        pred_home_processed = np.maximum(0, np.round(pred_home)).astype(int)
        pred_away_processed = np.maximum(0, np.round(pred_away)).astype(int)
        return pd.DataFrame({'predicted_home_score': pred_home_processed, 'predicted_away_score': pred_away_processed}, index=X.index)

    def load_model(self, filepath: Optional[str] = None, model_name: Optional[str] = None):
        super().load_model(filepath, model_name)
        self._validate_loaded_model_type() # Validate after loading pipelines

    def _validate_loaded_model_type(self):
        """ Ensures the loaded pipelines contain XGBoost regressors. """
        if not XGBOOST_AVAILABLE: return # Skip if XGBoost wasn't available anyway
        try:
            if self.pipeline_home and not isinstance(self.pipeline_home.steps[-1][1], xgb.XGBRegressor):
                raise TypeError(f"Loaded home pipeline for {self.model_name} is not an XGBRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], xgb.XGBRegressor):
                raise TypeError(f"Loaded away pipeline for {self.model_name} is not an XGBRegressor.")
            logger.debug(f"Validated loaded pipelines are XGBoost for {self.model_name}.")
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Error during validation of loaded XGBoost model ({self.model_name}): {e}", exc_info=True)
            # Raise a more informative error if validation fails
            raise TypeError(f"Validation failed for loaded model {self.model_name}. Check pipeline structure.")

# --- RandomForest Model Definition ---
class RandomForestScorePredictor(BaseScorePredictor):
    """ Trains and predicts using RandomForest models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "random_forest_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_rf_params = {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
            'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1, 'oob_score': False
        }

    def _build_pipeline(self, rf_params: Dict[str, Any]) -> Pipeline:
        final_params = self._default_rf_params.copy(); final_params.update(rf_params)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(**final_params))
        ])
        try: pipeline.set_output(transform="pandas"); logger.debug("Set RF pipeline output pandas.")
        except AttributeError: logger.debug("Could not set RF pipeline output pandas.")
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        # Note: RF doesn't typically use fit_params like eval_set
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_rf_params) # Pass defaults

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None: return None
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
                raise TypeError(f"Loaded home pipeline for {self.model_name} is not a RandomForestRegressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], RandomForestRegressor):
                raise TypeError(f"Loaded away pipeline for {self.model_name} is not a RandomForestRegressor.")
            logger.debug(f"Validated loaded pipelines are RandomForest for {self.model_name}.")
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Error validating loaded RandomForest model ({self.model_name}): {e}", exc_info=True)
            raise TypeError(f"Validation failed for loaded model {self.model_name}.")


# --- Ridge Model Definition ---
class RidgeScorePredictor(BaseScorePredictor):
    """ Trains and predicts using Ridge models within a preprocessing pipeline. """
    def __init__(self, model_dir: str = str(MODELS_BASE_DIR), model_name: str = "ridge_score_predictor"):
        super().__init__(model_dir, model_name)
        self._default_ridge_params = {
            'alpha': 1.0, 'fit_intercept': True, 'solver': 'auto', 'random_state': 42
        }

    def _build_pipeline(self, ridge_params: Dict[str, Any]) -> Pipeline:
        final_params = self._default_ridge_params.copy(); final_params.update(ridge_params)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**final_params))
        ])
        try: pipeline.set_output(transform="pandas"); logger.debug("Set Ridge pipeline output pandas.")
        except AttributeError: logger.debug("Could not set Ridge pipeline output pandas.")
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train_home: pd.Series, y_train_away: pd.Series,
              hyperparams_home: Optional[Dict[str, Any]] = None, hyperparams_away: Optional[Dict[str, Any]] = None,
              sample_weights: Optional[np.ndarray] = None):
        # Note: Ridge doesn't typically use fit_params like eval_set
        self._common_train_logic(X_train, y_train_home, y_train_away,
                                 hyperparams_home, hyperparams_away, sample_weights,
                                 self._default_ridge_params) # Pass defaults

    def predict(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        predictions = self._common_predict_logic(X)
        if predictions is None: return None
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
                raise TypeError(f"Loaded home pipeline for {self.model_name} is not a Ridge regressor.")
            if self.pipeline_away and not isinstance(self.pipeline_away.steps[-1][1], Ridge):
                raise TypeError(f"Loaded away pipeline for {self.model_name} is not a Ridge regressor.")
            logger.debug(f"Validated loaded pipelines are Ridge for {self.model_name}.")
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Error validating loaded Ridge model ({self.model_name}): {e}", exc_info=True)
            raise TypeError(f"Validation failed for loaded model {self.model_name}.")


# --- Quarter-Specific Model System (Prediction Focus) ---
# NOTE: This class remains largely unchanged as it wasn't the source of the error
# and is mostly used for prediction, not the main training pipeline error.
# Ensure its methods correctly handle feature subsets if they are called elsewhere.
class QuarterSpecificModelSystem:
    """
    Manages and utilizes quarter-specific models for prediction,
    including loading pre-trained models and handling fallbacks.
    """
    def __init__(self, feature_generator: Optional[NBAFeatureEngine] = None, debug: bool = False):
        self.feature_generator = feature_generator
        self.debug = debug
        # --- Model Storage (for Prediction) ---
        self.models: Dict[int, Any] = {}  # Stores loaded primary models for prediction
        self.fallback_models: Dict[int, Any] = {}  # Stores fallback models (Ridge)
        # --- Feature Set Definitions (Used for Prediction) ---
        # Attempt to get feature sets, handle potential absence gracefully
        self.quarter_feature_sets = self._get_consolidated_feature_sets()
        # --- Other Attributes ---
        # Default error history, can be updated later
        self.error_history = {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.prediction_cache: Dict[str, float] = {} # Simple cache
        self._create_fallback_models() # Ensure fallbacks exist
        self.log("QuarterSpecificModelSystem initialized for prediction.", level="DEBUG")

    def log(self, message, level="INFO"):
        # Use the module-level logger
        log_func = getattr(logger, level.lower(), logger.info)
        # Add a prefix to distinguish logs from this class
        log_func(f"[QuarterSystem] {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        # Safely get feature sets from the feature generator if available
        base_sets = {}
        if self.feature_generator:
            try:
                # Try different potential method names for flexibility
                if hasattr(self.feature_generator, 'get_prediction_feature_sets'):
                    base_sets = self.feature_generator.get_prediction_feature_sets()
                elif hasattr(self.feature_generator, '_get_optimized_feature_sets'):
                     base_sets = self.feature_generator._get_optimized_feature_sets()
                elif hasattr(self.feature_generator, 'feature_sets'):
                     base_sets = self.feature_generator.feature_sets

                if not isinstance(base_sets, dict):
                    self.log(f"Feature generator's feature set attribute is not a dict ({type(base_sets)}). Using empty sets.", level="WARNING")
                    base_sets = {}

            except Exception as e:
                self.log(f"Error retrieving feature sets from feature generator: {e}. Using empty sets.", level="ERROR")
                base_sets = {}
        else:
             self.log("Feature generator instance not provided during initialization. Using empty feature sets.", level="WARNING")

        # Ensure all quarters 1-4 have an entry, defaulting to empty list
        final_sets = {q: base_sets.get(q, []) for q in range(1, 5)}
        self.log(f"Consolidated prediction feature sets: Q1({len(final_sets.get(1, []))}), Q2({len(final_sets.get(2, []))}), Q3({len(final_sets.get(3, []))}), Q4({len(final_sets.get(4, []))}) features.", level="DEBUG")
        return final_sets


    def load_models(self, model_dir: Optional[str] = None):
        """
        Load pre-trained quarter-specific models.
        Args:
            model_dir: Directory containing the quarter model files. Defaults to the globally
                       defined QUARTERLY_MODELS_DIR path.
        """
        model_dir_path = Path(model_dir) if model_dir else QUARTERLY_MODELS_DIR
        self.log(f"Loading quarter prediction models from '{model_dir_path}'...")

        try: model_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e: self.log(f"Error creating directory {model_dir_path}: {e}", level="ERROR"); return

        for quarter in range(1, 5):
            model_filename = f'q{quarter}_model.pkl' # Assume .pkl for joblib
            model_path = model_dir_path / model_filename
            if model_path.is_file():
                try:
                    # TODO: Add validation after loading (e.g., check type, features_in_ if saved)
                    self.models[quarter] = joblib.load(model_path)
                    self.log(f"Loaded Q{quarter} prediction model from {model_path}")
                except Exception as e:
                    self.log(f"Error loading Q{quarter} prediction model from {model_path}: {e}", level="ERROR")
                    self.models[quarter] = None # Ensure it's None on failure
            else:
                self.log(f"Prediction model file for Q{quarter} ('{model_filename}') not found in {model_dir_path}. Will use fallback.", level="WARNING")
                self.models[quarter] = None

        self._create_fallback_models() # Ensure fallbacks are present


    def _create_fallback_models(self):
        """Create simple Ridge regression models as fallbacks for prediction if not already present."""
        self.log("Ensuring fallback Ridge models are available...", level="DEBUG")
        for quarter in range(1, 5):
            if quarter not in self.fallback_models:
                try:
                    # Basic fallback pipeline
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=1.0, random_state=42)) # Consistent random state
                    ])
                    self.fallback_models[quarter] = model
                    self.log(f"Fallback Ridge pipeline created/verified for Q{quarter}.", level="DEBUG")
                except Exception as e:
                    self.log(f"Failed to create fallback model for Q{quarter}: {e}", level="ERROR")


    def predict_quarter(self, X: pd.DataFrame, quarter: int) -> float:
        """Predicts the score for a specific quarter using the appropriate model."""
        # --- Input Validation ---
        if not (1 <= quarter <= 4):
            self.log(f"Invalid quarter ({quarter}) requested for prediction.", level="ERROR")
            return 0.0 # Return a default neutral value
        if X is None or X.empty or X.shape[0] != 1:
            self.log(f"Invalid input DataFrame for predict_quarter (must be single row). Shape: {X.shape if X is not None else 'None'}", level="ERROR")
            return 0.0

        # --- Feature Preparation ---
        # Get expected features for this quarter
        features_expected = self.quarter_feature_sets.get(quarter, [])
        if not features_expected:
             self.log(f"No specific features defined for Q{quarter} prediction. Cannot reliably predict.", level="WARNING")
             # Consider if fallback logic is needed here or if default score is acceptable
        # Identify features available in the input DataFrame
        features_available = [f for f in features_expected if f in X.columns]
        if not features_available:
            default_score = 25.0 # Fallback average
            if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
            self.log(f"None of the expected Q{quarter} features ({features_expected}) are available in the input. Using default score: {default_score}", level="WARNING")
            return default_score

        # Prepare the input slice using only available features expected for this quarter
        X_predict_qtr = X[features_available]

        # --- Caching (Optional but useful for repeated calls with same state) ---
        cache_key = None
        try:
             # Create a hashable key from the relevant feature values
             relevant_data_tuple = tuple(X_predict_qtr.iloc[0].fillna(0).values)
             cache_key = f"q{quarter}_{hash(relevant_data_tuple)}"
             if cache_key in self.prediction_cache:
                 self.log(f"Using cached prediction for Q{quarter}", level="DEBUG")
                 return self.prediction_cache[cache_key]
        except Exception as e:
            self.log(f"Could not generate or check cache key for Q{quarter}: {e}. Proceeding without cache.", level="WARNING")
            cache_key = None # Ensure cache is skipped on error

        # --- Model Selection (Primary or Fallback) ---
        model_to_use = self.models.get(quarter)
        model_source = "primary"
        if model_to_use is None:
            model_to_use = self.fallback_models.get(quarter)
            model_source = "fallback"
            if model_to_use is None:
                default_score = 25.0 # Define default again just in case
                if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                    default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
                self.log(f"No primary or fallback model loaded for Q{quarter}. Using default score: {default_score}", level="ERROR")
                if cache_key: self.prediction_cache[cache_key] = default_score
                return default_score

        # --- Feature Validation Against Loaded Model (if possible) ---
        final_features_used_for_predict = features_available
        try:
            # Check if the loaded model pipeline has stored feature names
            model_instance = model_to_use.steps[-1][1] if isinstance(model_to_use, Pipeline) else model_to_use
            if hasattr(model_instance, 'feature_names_in_') and model_instance.feature_names_in_ is not None:
                required_model_features = list(model_instance.feature_names_in_)
                # Check if ALL features the model was trained on are available
                if set(required_model_features).issubset(set(features_available)):
                    # Use the exact features the model expects, in the correct order
                    final_features_used_for_predict = required_model_features
                    X_predict_qtr = X_predict_qtr[final_features_used_for_predict] # Re-slice/reorder
                    self.log(f"Aligned input features with {model_source} Q{quarter} model's expected {len(required_model_features)} features.", level="DEBUG")
                else:
                    missing_req = set(required_model_features) - set(features_available)
                    self.log(f"Input missing features required by loaded {model_source} Q{quarter} model: {missing_req}. Prediction accuracy compromised.", level="ERROR")
                    # Decide how to handle: return default, or predict with available subset?
                    # Predicting with subset might be inaccurate. Returning default might be safer.
                    default_score = 25.0 # Define default again
                    if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                        default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
                    if cache_key: self.prediction_cache[cache_key] = default_score
                    return default_score
            else:
                self.log(f"Loaded {model_source} model for Q{quarter} doesn't have 'feature_names_in_'. Predicting with available features: {features_available}", level="DEBUG")
                # Proceed with the features available that were expected for the quarter
                X_predict_qtr = X_predict_qtr[final_features_used_for_predict]

        except Exception as feat_val_e:
             self.log(f"Error during feature validation against loaded Q{quarter} model: {feat_val_e}", level="ERROR")
             # Fallback to default score on validation error
             default_score = 25.0
             if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)
             if cache_key: self.prediction_cache[cache_key] = default_score
             return default_score


        # --- Prediction ---
        prediction = None
        self.log(f"Predicting Q{quarter} using {model_source} model with {X_predict_qtr.shape[1]} features.", level="DEBUG")
        try:
            # Ensure input has no NaNs before prediction (imputer should handle this in pipeline)
            if X_predict_qtr.isnull().any().any():
                 logger.warning(f"NaNs detected in input for Q{quarter} {model_source} prediction just before calling predict. Imputer should handle.")
                 # Optionally fill here if pipeline doesn't have imputer: X_predict_qtr = X_predict_qtr.fillna(0)

            prediction = model_to_use.predict(X_predict_qtr)[0] # Predict on single row

        except Exception as e:
            self.log(f"Error during Q{quarter} prediction with {model_source}: {e}", level="ERROR")
            # Try fallback if primary failed
            if model_source == "primary":
                self.log(f"Attempting Q{quarter} prediction with fallback pipeline...", level="WARNING")
                fallback_model = self.fallback_models.get(quarter)
                if fallback_model:
                    try:
                        # Re-prepare input for fallback model if necessary (might expect different features)
                        # For simplicity, assume fallback uses same `features_available` for now
                        # TODO: Fallback model should ideally also store its expected features
                        fb_input = X[features_available] # Use original available features for fallback
                        if fb_input.isnull().any().any():
                            logger.warning("NaNs detected in fallback input, attempting fillna(0).")
                            fb_input = fb_input.fillna(0)

                        prediction = fallback_model.predict(fb_input)[0]
                        model_source = "fallback (after primary error)"
                        self.log(f"Successfully used fallback pipeline for Q{quarter}.", level="DEBUG")
                    except Exception as fb_e:
                        self.log(f"Error during Q{quarter} prediction with fallback pipeline: {fb_e}", level="ERROR")
                        prediction = None # Prediction failed even with fallback

        # --- Process Prediction Result ---
        default_score = 25.0 # Define default again
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
             default_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(quarter, 25.0)

        final_prediction = default_score # Start with default
        if prediction is not None:
            try:
                # Ensure prediction is float and non-negative
                final_prediction = max(0.0, float(prediction))
            except (ValueError, TypeError):
                self.log(f"Prediction result '{prediction}' is not numeric for Q{quarter}. Using default.", level="ERROR")
                final_prediction = default_score # Fallback to default
        else:
            # This means prediction failed even after fallback attempt
            self.log(f"Prediction failed for Q{quarter} after all attempts. Using default score: {final_prediction}", level="ERROR")
            model_source += " -> default (after errors)"

        self.log(f"Q{quarter} Prediction ({model_source}): {final_prediction:.3f}", level="DEBUG")

        # Store in cache if key was generated
        if cache_key:
            self.prediction_cache[cache_key] = final_prediction

        return final_prediction


    def predict_remaining_quarters(self, game_data: Dict, current_quarter: int) -> Dict[str, float]:
        """ Predicts scores for quarters from current_quarter + 1 through 4. """
        if not isinstance(game_data, dict):
             self.log("predict_remaining_quarters requires game_data as a dict.", level="ERROR"); return {}
        if not isinstance(current_quarter, int) or current_quarter < 0:
             self.log(f"Invalid current_quarter ({current_quarter}).", level="ERROR"); return {}
        if current_quarter >= 4:
            self.log("Game is in Q4 or later, no remaining quarters to predict.", level="DEBUG")
            return {}

        results = {}
        # Start with the current known game state
        current_game_state_dict = game_data.copy()

        # Initial feature generation based on current state
        try:
            # Convert dict to DataFrame for feature engine compatibility
            X = pd.DataFrame([current_game_state_dict])

            # Use feature generator if available and has appropriate methods
            if self.feature_generator:
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X = self.feature_generator.generate_features_for_prediction(X)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    # This might not be ideal as it expects historical context sometimes
                    X = self.feature_generator.integrate_advanced_features(X)
                else:
                    self.log("Feature generator lacks expected methods (generate_features_for_prediction or integrate_advanced_features). Cannot generate features.", level="ERROR")
                    return {} # Cannot proceed without features
            else:
                 self.log("Feature generator not available. Cannot generate features for remaining quarters.", level="ERROR")
                 return {}

            if X is None or X.empty:
                 raise ValueError("Initial feature generation returned empty DataFrame.")

        except Exception as e:
            self.log(f"Error generating initial features for remaining quarters prediction: {e}", level="ERROR", exc_info=True)
            return {} # Cannot proceed

        # Predict iteratively for each remaining quarter
        for q in range(current_quarter + 1, 5): # Quarters 1-4
            q_pred_score = 0.0 # Default score if prediction fails
            try:
                # Predict score for quarter q using current features X
                # Pass iloc[0] as predict_quarter expects a single row DataFrame
                q_pred_score = self.predict_quarter(X.iloc[[0]], q)
                results[f'q{q}'] = q_pred_score

                # Update the game state dictionary with the predicted score for this quarter
                # Assume prediction is for home team score for now (or total if model predicts total)
                # TODO: Clarify if quarter model predicts home, away, or total score for the quarter
                predicted_col_home = f'home_q{q}' # Assuming prediction is home score for the quarter
                current_game_state_dict[predicted_col_home] = q_pred_score
                # Optionally predict/set away score too if model supports it or use an assumption
                # current_game_state_dict[f'away_q{q}'] = q_pred_score # Example: Assume symmetric scoring

                # Regenerate features based on the updated game state *after* predicting quarter q
                self.log(f"Regenerating features incorporating predicted Q{q}={q_pred_score:.2f}...", level="DEBUG")
                X_next = pd.DataFrame([current_game_state_dict]) # Create DF from updated dict

                if self.feature_generator: # Re-check feature generator
                    if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                        X = self.feature_generator.generate_features_for_prediction(X_next)
                    elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                         X = self.feature_generator.integrate_advanced_features(X_next)
                    else:
                         # If feature generator failed before, it will fail again, but log warning
                         self.log("Feature generator lacks methods. Cannot regenerate features; subsequent predictions may be less accurate.", level="WARNING")
                         # Manually update the feature DataFrame if possible (less ideal)
                         if predicted_col_home not in X.columns: X = X.assign(**{predicted_col_home: q_pred_score})
                         else: X.loc[X.index[0], predicted_col_home] = q_pred_score # Update existing row
                else:
                     # Cannot regenerate if generator missing
                      self.log("Feature generator missing. Cannot regenerate features.", level="ERROR")
                      # Manually update if possible
                      if predicted_col_home not in X.columns: X = X.assign(**{predicted_col_home: q_pred_score})
                      else: X.loc[X.index[0], predicted_col_home] = q_pred_score

                if X is None or X.empty:
                     raise ValueError(f"Feature regeneration after Q{q} returned empty DataFrame.")

            except Exception as e:
                 self.log(f"Error predicting or updating features for Q{q}: {e}", level="ERROR", exc_info=True)
                 # Use default score for this quarter if error occurs
                 default_q_score = 25.0
                 if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
                    default_q_score = self.feature_generator.league_averages.get('quarter_scores', {}).get(q, 25.0)
                 results[f'q{q}'] = default_q_score
                 current_game_state_dict[f'home_q{q}'] = default_q_score # Update state with default

                 # Attempt to regenerate features even after error to allow next prediction attempt
                 try:
                     X_next = pd.DataFrame([current_game_state_dict])
                     if self.feature_generator:
                         if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                            X = self.feature_generator.generate_features_for_prediction(X_next)
                         elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                            X = self.feature_generator.integrate_advanced_features(X_next)
                         else: X = None # Cannot regenerate
                     else: X = None

                     if X is None or X.empty:
                         self.log(f"Feature regeneration failed after Q{q} error. Stopping remaining predictions.", level="ERROR")
                         break # Stop predicting further quarters

                 except Exception as regen_e:
                     self.log(f"Error regenerating features after Q{q} prediction error: {regen_e}. Stopping remaining predictions.", level="ERROR")
                     break # Stop predicting further quarters


        self.log(f"Predicted remaining quarters ({current_quarter+1}-4): {results}", level="DEBUG")
        return results


    def predict_final_score(self, game_data: Dict, main_model_prediction: Optional[float] = None, weight_manager: Optional[EnsembleWeightManager] = None) -> Tuple[float, float, Dict]:
        """ Predicts the final score using quarter models and ensemble weighting. """
        self.log("Predicting final score using WeightManager integration...", level="DEBUG")

        # --- Input Validation & Fallback ---
        league_avg_score = 110.0 # Base fallback
        if self.feature_generator and hasattr(self.feature_generator, 'league_averages'):
             league_avg_score = self.feature_generator.league_averages.get('score', 110.0)

        # Determine initial fallback prediction value
        fallback_pred = main_model_prediction if main_model_prediction is not None else league_avg_score
        try: fallback_pred = float(fallback_pred)
        except (ValueError, TypeError): fallback_pred = league_avg_score

        # Validate game_data and weight_manager
        if not isinstance(game_data, dict):
            self.log("Input game_data must be a dictionary.", level="ERROR"); return fallback_pred, 0.0, {'error': 'Invalid game_data type'}
        if weight_manager is None:
            self.log("WeightManager instance IS REQUIRED for predict_final_score.", level="ERROR"); return fallback_pred, 0.0, {'error': 'Missing WeightManager'}
        if not hasattr(weight_manager, 'calculate_ensemble'):
            self.log(f"Provided weight_manager ({type(weight_manager)}) is missing 'calculate_ensemble' method.", level="ERROR"); return fallback_pred, 0.0, {'error': 'Invalid WeightManager instance or method'}

        # --- Process Current Game State ---
        current_quarter = 0
        try: current_quarter = int(game_data.get('current_quarter', 0))
        except (ValueError, TypeError): self.log("Invalid current_quarter, defaulting to 0.", level="WARNING")

        home_score_so_far = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        away_score_so_far = sum([float(game_data.get(f'away_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])

        # Handle missing main model prediction
        if main_model_prediction is None:
            if current_quarter > 0 and (home_score_so_far > 0 or away_score_so_far > 0): # Use combined score > 0 check
                 # Extrapolate based on current score per quarter played
                 main_model_prediction = (home_score_so_far + away_score_so_far) * (4.0 / max(1, current_quarter)) # Extrapolate total
                 # Distribute based on league average or ratio? Let's stick to total extrapolation for main model gap fill.
                 # Assume main model predicts home score, need to adjust this if it predicts total or diff
                 main_model_prediction = main_model_prediction / 2 # Simple split for home score prediction
                 self.log(f"Main model prediction missing. Extrapolating based on current total: Home Est ~{main_model_prediction:.2f}", level="WARNING")
            else: # Pregame or score is zero
                 main_model_prediction = league_avg_score # Predict league average total score
                 main_model_prediction = main_model_prediction / 2 # Simple split for home score prediction
                 self.log(f"Main model prediction missing pregame/score=0. Using league average: Home Est ~{main_model_prediction:.2f}", level="WARNING")

        # Ensure main_model_prediction is float
        try: main_model_prediction = float(main_model_prediction)
        except (ValueError, TypeError):
             self.log(f"Main model prediction ('{main_model_prediction}') invalid. Using league average.", level="ERROR")
             main_model_prediction = league_avg_score / 2 # Fallback home score estimate

        # --- Handle Pre-Game Case ---
        if current_quarter <= 0:
            self.log("Pregame prediction. Using main model prediction only.", level="DEBUG")
            # Use main model prediction directly, assume confidence is neutral
            confidence = 0.5 # Default confidence for pregame
            # Basic breakdown for pregame
            breakdown = {
                'main_model_pred': main_model_prediction, 'quarter_model_sum': 0, # Quarter prediction is 0 pregame
                'weights': {'main': 1.0, 'quarter': 0.0}, 'quarter_predictions': {},
                'current_score_home': 0, 'current_score_away': 0, 'score_differential': 0,
                'momentum': 0, 'current_quarter': 0, 'time_remaining_minutes_est': 48.0
            }
            # Predict away score based on average differential or assume symmetry
            # For simplicity, assume symmetric prediction if only home is given
            final_pred_home = main_model_prediction
            # final_pred_away = main_model_prediction - avg_diff # Needs avg_diff calculation
            final_pred_away = main_model_prediction # Simple symmetric assumption
            final_total_pred = final_pred_home + final_pred_away # Calculate total based on individual preds

            return final_total_pred, confidence, breakdown # Return total score prediction for consistency? Or home? Let's return TOTAL.


        # --- In-Game Prediction ---
        # Predict remaining quarters
        remaining_quarters_pred = self.predict_remaining_quarters(game_data, current_quarter)
        if not isinstance(remaining_quarters_pred, dict):
             self.log("predict_remaining_quarters failed. Returning main prediction (total).", level="ERROR")
             # Return extrapolated total prediction as fallback
             main_total_pred = main_model_prediction * 2 # Reconstruct total if main was home pred
             return main_total_pred, 0.3, {'error': 'Failed to predict remaining quarters', 'main_model': main_total_pred}

        # Calculate the quarter model's full game prediction (HOME score focus)
        predicted_score_remaining_home = sum(remaining_quarters_pred.values()) # Assuming predict_quarter returns home score
        quarter_sum_prediction_home = home_score_so_far + predicted_score_remaining_home

        # Get context features (score diff, momentum)
        score_differential = home_score_so_far - away_score_so_far # Use actual current diff
        momentum = 0.0
        time_remaining_minutes = max(0.0, 12.0 * (4.0 - current_quarter)) # Estimate based on quarter

        # Try to get momentum from features if possible
        try:
            X_context = pd.DataFrame([game_data])
            if self.feature_generator:
                 if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                      X_context = self.feature_generator.generate_features_for_prediction(X_context)
                 elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                      X_context = self.feature_generator.integrate_advanced_features(X_context)

                 if not X_context.empty:
                     # Use calculated score_differential if feature exists and is valid
                     if 'score_differential' in X_context.columns and pd.notna(X_context['score_differential'].iloc[0]):
                         score_differential = float(X_context['score_differential'].iloc[0])
                     # Use cumulative_momentum if feature exists and is valid
                     if 'cumulative_momentum' in X_context.columns and pd.notna(X_context['cumulative_momentum'].iloc[0]):
                          momentum = float(X_context['cumulative_momentum'].iloc[0])
                     # Try to get a better time estimate if available
                     if 'time_remaining_seconds' in X_context.columns and pd.notna(X_context['time_remaining_seconds'].iloc[0]):
                          time_remaining_minutes = max(0.0, float(X_context['time_remaining_seconds'].iloc[0]) / 60.0)

            else: self.log("Feature generator missing, using basic context estimates.", level="WARNING")

        except Exception as e:
            self.log(f"Error getting context features (momentum, time): {e}. Using defaults.", level="WARNING")

        # --- Call Ensemble Weight Manager ---
        try:
            # IMPORTANT: Pass the HOME score predictions to the weight manager
            # The weight manager should return the ensemble HOME score prediction
            ensemble_pred_home, final_weight_main, final_weight_quarter = weight_manager.calculate_ensemble(
                main_prediction=main_model_prediction,            # Main model's HOME score prediction
                quarter_prediction=quarter_sum_prediction_home,   # Quarter system's HOME score prediction
                current_quarter=int(current_quarter),
                score_differential=float(score_differential),     # Current actual diff
                momentum=float(momentum),                         # Current momentum
                time_remaining=float(time_remaining_minutes)    # Use estimated time remaining
                # Pass uncertainties if available/needed by weight manager
            )
            # Confidence might need separate calculation now
            # Placeholder confidence based on weights or prediction stability
            confidence = 0.6 + 0.4 * final_weight_main # Example: Higher confidence if main model weighted more

        except Exception as e:
            self.log(f"Error calling weight_manager.calculate_ensemble: {e}. Using fallback.", level="ERROR", exc_info=True)
            ensemble_pred_home = main_model_prediction # Fallback to main model HOME score
            confidence = 0.3
            final_weight_main = 1.0
            final_weight_quarter = 0.0

        # --- Assemble Final Prediction and Breakdown ---
        try:
            final_ensemble_pred_home = float(ensemble_pred_home)
            final_confidence = float(confidence)

            # Estimate Away Score based on Home Score and Predicted Differential
            # We need a way to predict the final differential or away score.
            # Option 1: Predict Away score separately (requires away quarter models)
            # Option 2: Use the main model's predicted differential if available
            # Option 3: Assume the ensemble diff is proportional to the main model's diff
            # Option 4: Simplest - Assume ensemble diff is similar to main model diff (less accurate)

            # Let's assume main model predicts home/away, calculate its diff
            # If only home is predicted, this part needs adjustment
            # Assume main_model_prediction was only for home, we need away too.
            # Fallback: Assume average diff or fixed diff based on ensemble home pred
            # Let's use a placeholder assuming league average diff for simplicity here:
            avg_diff = 2.5 # League average home advantage (approx)
            final_ensemble_pred_away = final_ensemble_pred_home - avg_diff # Estimate away

            final_total_pred = final_ensemble_pred_home + final_ensemble_pred_away

            breakdown = {
                'main_model_pred': main_model_prediction, # Home score prediction
                'quarter_model_sum': quarter_sum_prediction_home, # Home score prediction
                'weights': {'main': final_weight_main, 'quarter': final_weight_quarter},
                'quarter_predictions': remaining_quarters_pred, # Predictions for remaining quarters (home assumed)
                'current_score_home': home_score_so_far,
                'current_score_away': away_score_so_far,
                'score_differential': score_differential, # Actual current differential
                'momentum': momentum,
                'current_quarter': current_quarter,
                'time_remaining_minutes_est': time_remaining_minutes
            }
            self.log(f"Final Ensemble Prediction (Home): {final_ensemble_pred_home:.2f}, Est Away: {final_ensemble_pred_away:.2f}, Total: {final_total_pred:.2f} (Confidence: {final_confidence:.2f})", level="DEBUG")

            # Return the TOTAL score prediction for consistency with pregame? Or return Home?
            # Let's return TOTAL score prediction.
            return final_total_pred, final_confidence, breakdown

        except (ValueError, TypeError) as final_e:
            self.log(f"Error finalizing ensemble prediction: {final_e}. Returning fallback.", level="ERROR")
            main_total_pred = main_model_prediction * 2 # Reconstruct total
            return main_total_pred, 0.0, {'error': 'Non-numeric ensemble result'}