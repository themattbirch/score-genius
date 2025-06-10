# backend/mlb_score_prediction/models.py
"""
models.py - MLB Score Prediction Models Module (rewritten 2025-06-09)
Implements Ridge, SVR, RF, XGB, LGBM predictors with shared training/prediction
logic. Fixes:
  • `build_pipeline()` now exists for every concrete predictor ▶ prevents
    NotImplementedError.
  • `BaseScorePredictor` declares abstract `build_pipeline()` for clarity.
  • Parameter application in `_common_train_logic` uses `pipeline.set_params(**…)`
    — works because all hyper-parameter keys already carry the `model__` prefix
    (from RandomizedSearchCV).
  • Numeric-only feature subset retained; preprocessing unchanged.
  • Re-exported `compute_recency_weights` from mlb_features.utils to support imports.
  • Minor lint/typing clean-ups.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Re-export utility for external imports
# -----------------------------------------------------------------------------
# Utility Functions (inlined)
# -----------------------------------------------------------------------------

def compute_recency_weights(dates: pd.Series, method: str = 'half_life', half_life: int = 60, decay_rate: float = 0.98) -> Optional[np.ndarray]:
    """
    Calculate recency weights for samples based on dates.
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
    days_from_latest = (latest_date - sorted_dates).dt.total_seconds() / (3600 * 24.0)
    weights = np.ones(len(sorted_dates))
    if method == 'half_life' and half_life > 0:
        weights = 0.5 ** (days_from_latest / float(half_life))
    elif method == 'exponential' and 0 < decay_rate < 1:
        weights = decay_rate ** days_from_latest
    else:
        logger.warning(f"Recency weights: Unknown method or invalid params: {method}, {half_life}, {decay_rate}.")

    weights[~np.isfinite(weights)] = 0.0
    mean_weight = np.mean(weights) if len(weights) > 0 else 0.0
    if mean_weight > 1e-9:
        weights = weights / mean_weight
    else:
        weights = np.ones(len(sorted_dates))

    weights_series = pd.Series(weights, index=sorted_dates.index)
    fill_val = 0.0 if not valid_dates.empty else 1.0
    return weights_series.reindex(original_index, fill_value=fill_val).values


# -----------------------------------------------------------------------------
# Global / config
# -----------------------------------------------------------------------------

SEED = 42
NUM_INNINGS_REGULATION = 9  # Standard number of innings in an MLB game

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# model directory
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"MLB models will be stored in {MODEL_DIR}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _preprocessing_pipeline() -> Pipeline:
    """Simple numeric-only impute → scale pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25, 75))),
            ("post_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )

# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------

class BaseScorePredictor:
    """Abstract base for home/away run predictors."""

    def __init__(self, model_name: str, model_dir: Path = MODEL_DIR):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.pipeline_home: Optional[Pipeline] = None
        self.pipeline_away: Optional[Pipeline] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.training_timestamp: Optional[str] = None
        self.training_duration: Optional[float] = None

    # ----- abstract stubs ----------------------------------------------------

    def build_pipeline(self) -> Pipeline:  # noqa: D401
        """Return a fresh pipeline instance. Must be overridden."""
        raise NotImplementedError("Subclasses must implement build_pipeline()")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train_home: pd.Series,
        y_train_away: pd.Series,
        *,
        hyperparams_home: Optional[Dict[str, Any]] = None,
        hyperparams_away: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
        default_params: Dict[str, Any],
        fit_params: Optional[Dict[str, Any]] = None,
        eval_set_data: Optional[Tuple] = None,
    ) -> None:
        """Shared training logic."""

        start = time.time()

        # numeric-only guard
        X_num = X_train.select_dtypes(include=np.number)
        if X_num.shape[1] != X_train.shape[1]:
            diff = set(X_train.columns) - set(X_num.columns)
            logger.warning(f"Dropping non-numeric cols: {diff}")
        self.feature_names_in_ = list(X_num.columns)

        # assemble params (RandomizedSearch returns keys with model__ prefix)
        params_h = {**default_params, **(hyperparams_home or {})}
        params_a = {**default_params, **(hyperparams_away or {})}

        # build pipelines
        self.pipeline_home = self.build_pipeline()
        self.pipeline_away = self.build_pipeline()
        self.pipeline_home.set_params(**params_h)
        self.pipeline_away.set_params(**params_a)

        fit_kw_h: Dict[str, Any] = {}
        fit_kw_a: Dict[str, Any] = {}

        # handle sample weights
        if sample_weights is not None and len(sample_weights) == len(X_num):
            sw_param = f"{self.pipeline_home.steps[-1][0]}__sample_weight"  # model__sample_weight
            fit_kw_h[sw_param] = sample_weights
            fit_kw_a[sw_param] = sample_weights

        # early-stopping for XGB
        if eval_set_data and isinstance(self.pipeline_home.named_steps["model"], XGBRegressor):
            X_val_raw, y_val_h, y_val_a = eval_set_data
            X_val_trans = self.pipeline_home.named_steps["preprocessing"].fit_transform(
                X_val_raw[self.feature_names_in_]
            )
            fit_kw_h.update({
                "model__eval_set": [(X_val_trans, y_val_h)],
                "model__early_stopping_rounds": 30
            })
            fit_kw_a.update({
                "model__eval_set": [(X_val_trans, y_val_a)],
                "model__early_stopping_rounds": 30
            })
        elif eval_set_data and isinstance(self.pipeline_home.named_steps["model"], LGBMRegressor):
            X_val_raw, y_val_h, y_val_a = eval_set_data
            X_val_trans = self.pipeline_home.named_steps["preprocessing"].fit_transform(
                X_val_raw[self.feature_names_in_]
            )
            fit_kw_h.update({
                "model__eval_set": [(X_val_trans, y_val_h)],
                "model__early_stopping_rounds": 50
            })
            fit_kw_a.update({
                "model__eval_set": [(X_val_trans, y_val_a)],
                "model__early_stopping_rounds": 50
            })
        # fit
        self.pipeline_home.fit(X_num, y_train_home, **fit_kw_h)
        self.pipeline_away.fit(X_num, y_train_away, **fit_kw_a)

        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start
        logger.info(f"{self.model_name} trained in {self.training_duration:.1f}s")

    # ----- prediction --------------------------------------------------------

    def _predict_pair(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.pipeline_home or not self.pipeline_away:
            raise RuntimeError("Models not trained")
        X_num = X[self.feature_names_in_]
        return self.pipeline_home.predict(X_num), self.pipeline_away.predict(X_num)

    # ----- persistence -------------------------------------------------------

    def save_model(self) -> str:
        if not self.pipeline_home or not self.pipeline_away:
            raise RuntimeError("Train before saving")
        path = self.model_dir / f"{self.model_name}.joblib"
        joblib.dump({
            "pipeline_home": self.pipeline_home,
            "pipeline_away": self.pipeline_away,
            "feature_names_in_": self.feature_names_in_,
        }, path)
        return str(path)

    def load_model(self, path: Optional[Union[str, Path]] = None) -> None:
        path = Path(path or (self.model_dir / f"{self.model_name}.joblib"))
        data = joblib.load(path)
        self.pipeline_home = data["pipeline_home"]
        self.pipeline_away = data["pipeline_away"]
        self.feature_names_in_ = data["feature_names_in_"]


# -----------------------------------------------------------------------------
# Concrete predictors
# -----------------------------------------------------------------------------


class RidgeScorePredictor(BaseScorePredictor):
    _default = {"model__alpha": 1.0, "model__random_state": SEED}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def build_pipeline(self) -> Pipeline:  # noqa: D401
        return Pipeline([
            ("preprocessing", _preprocessing_pipeline()),
            ("model", Ridge())
        ])

    def train(self, *args, **kwargs):
        super().train(*args, default_params=self._default, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ph, pa = self._predict_pair(X)
        return pd.DataFrame({"predicted_home_runs": np.maximum(0, ph), "predicted_away_runs": np.maximum(0, pa)}, index=X.index)


class SVRScorePredictor(BaseScorePredictor):
    _default = {"model__kernel": "rbf", "model__C": 1.0, "model__epsilon": 0.1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessing", _preprocessing_pipeline()),
            ("model", SVR())
        ])

    def train(self, *args, **kwargs):
        super().train(*args, default_params=self._default, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ph, pa = self._predict_pair(X)
        return pd.DataFrame({"predicted_home_runs": np.maximum(0, ph), "predicted_away_runs": np.maximum(0, pa)}, index=X.index)


class RFScorePredictor(BaseScorePredictor):
    _default = {"model__n_estimators": 300, "model__n_jobs": -1, "model__random_state": SEED}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessing", _preprocessing_pipeline()),
            ("model", RandomForestRegressor())
        ])

    def train(self, *args, **kwargs):
        super().train(*args, default_params=self._default, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ph, pa = self._predict_pair(X)
        return pd.DataFrame({"predicted_home_runs": np.maximum(0, ph), "predicted_away_runs": np.maximum(0, pa)}, index=X.index)


class XGBoostScorePredictor(BaseScorePredictor):
    _default = {
        "model__n_estimators": 400,
        "model__max_depth": 3,
        "model__learning_rate": 0.05,
        "model__subsample": 0.8,
        "model__colsample_bytree": 0.8,
        "model__objective": "reg:squarederror",
        "model__random_state": SEED,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessing", _preprocessing_pipeline()),
            ("model", XGBRegressor(verbosity=0))
        ])

    def train(self, *args, **kwargs):
        super().train(*args, default_params=self._default, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ph, pa = self._predict_pair(X)
        return pd.DataFrame({"predicted_home_runs": np.maximum(0, ph), "predicted_away_runs": np.maximum(0, pa)}, index=X.index)


class LGBMScorePredictor(BaseScorePredictor):
    _default = {"model__n_estimators": 400, "model__learning_rate": 0.05, "model__random_state": SEED}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessing", _preprocessing_pipeline()),
            ("model", LGBMRegressor(verbose=-1))
        ])

    def train(self, *args, **kwargs):
        super().train(*args, default_params=self._default, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ph, pa = self._predict_pair(X)
        return pd.DataFrame({"predicted_home_runs": np.maximum(0, ph), "predicted_away_runs": np.maximum(0, pa)}, index=X.index)

# InningSpecificModelSystem omitted for brevity
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