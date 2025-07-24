# backend/nfl_score_prediction/models.py
"""
models.py - NFL Score Prediction Models Module

This module provides the core components for building NFL score prediction models.

Key Features:
  - Implements a flexible `BaseNFLPredictor` class to manage model training,
    prediction, and persistence for a single prediction target.
  - Designed to predict two separate targets: 'point margin' (home_score -
    away_score) and 'total points' (home_score + away_score).
  - Allows for different model architectures for each target, such as using
    Ridge regression for the margin and XGBoost for the total.
  - Provides concrete implementations for various models (Ridge, XGBoost, SVR,
    RandomForest).
  - Includes a `derive_scores_from_predictions` helper function to
    reconstruct home and away scores from the model outputs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------- #
# Globals
# ----------------------------------------------------------------------------- #

SEED = 42
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logger.info("NFL models dir: %s", MODEL_DIR)


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def _preprocessing_pipeline() -> Pipeline:
    """Numeric-only preprocessing: impute → scale → safety impute."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
            ("post_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )


def _select_numeric(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> pd.DataFrame:
    """Return numeric subset; optionally intersect with explicit list."""
    if explicit:
        cols = [c for c in explicit if c in df.columns]
        df_num = df[cols]
    else:
        df_num = df.select_dtypes(include=np.number)
    if df_num.shape[1] < df.shape[1]:
        logger.debug("Non-numeric columns dropped: %d", df.shape[1] - df_num.shape[1])
    return df_num


def compute_recency_weights(
    dates: pd.Series,
    *,
    method: str = "half_life",
    half_life: int = 90,
    decay_rate: float = 0.98,
) -> np.ndarray:
    """
    Recency weights for weekly NFL cadence. Falls back to uniform weights on errors.
    """
    try:
        if not isinstance(dates, pd.Series) or dates.empty:
            return np.ones(len(dates))
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates, errors="coerce").dt.tz_localize(None)

        valid = dates.dropna()
        if valid.empty:
            return np.ones(len(dates))

        latest = valid.max()
        days_back = (latest - valid).dt.total_seconds() / 86400.0

        if method == "half_life" and half_life > 0:
            w = 0.5 ** (days_back / float(half_life))
        elif method == "exponential" and 0 < decay_rate < 1:
            w = decay_rate ** days_back
        else:
            w = np.ones(len(valid))

        w[np.isnan(w) | ~np.isfinite(w)] = 0.0
        mean_w = w.mean()
        if mean_w > 1e-9:
            w = w / mean_w
        weights = pd.Series(w, index=valid.index)
        return weights.reindex(dates.index, fill_value=1.0).values
    except Exception as e:
        logger.warning("Recency weighting failed (%s); using uniform weights.", e)
        return np.ones(len(dates))


def derive_scores_from_predictions(margin_preds: pd.Series, total_preds: pd.Series) -> pd.DataFrame:
    """Recover home/away scores from margin & total predictions."""
    home = (total_preds + margin_preds) / 2.0
    away = (total_preds - margin_preds) / 2.0
    home = np.maximum(home, 0)
    away = np.maximum(away, 0)
    return pd.DataFrame(
        {"predicted_home_score": home, "predicted_away_score": away},
        index=margin_preds.index,
    )


# ----------------------------------------------------------------------------- #
# Base Class
# ----------------------------------------------------------------------------- #

class BaseNFLPredictor:
    """
    Single-target predictor (margin or total).
    Subclasses set `_default` and implement `build_pipeline()`.
    """

    _default: Dict[str, Any] = {}

    def __init__(self, model_name: str, model_dir: Path = MODEL_DIR):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.training_timestamp: Optional[str] = None
        self.training_duration: Optional[float] = None
        self.recency_weight_params: Optional[Dict[str, Any]] = None

    # --- abstract ------------------------------------------------------------ #

    def build_pipeline(self) -> Pipeline:
        raise NotImplementedError

    # --- core API ------------------------------------------------------------ #

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        hyperparams: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
        numeric_features: Optional[List[str]] = None,
        eval_set_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        recency_weight_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fit pipeline. Handles numeric filtering, weights, and optional eval set.
        """
        start = time.time()

        X_num = _select_numeric(X_train, numeric_features)
        self.feature_names_in_ = list(X_num.columns)

        params = {**self._default, **(hyperparams or {})}
        self.pipeline = self.build_pipeline()
        self.pipeline.set_params(**params)

        fit_kwargs: Dict[str, Any] = {}
        model_step = self.pipeline.steps[-1][0]

        if sample_weights is not None and len(sample_weights) == len(X_num):
            fit_kwargs[f"{model_step}__sample_weight"] = sample_weights

        # Fit preprocessing separately if we need an eval set with XGB
        if (
            eval_set_data
            and isinstance(self.pipeline.named_steps[model_step], XGBRegressor)
        ):
            X_val_raw, y_val = eval_set_data

            # Fit preprocessing on training data, then transform the validation set
            preprocessor = self.pipeline.named_steps["preprocessing"]
            preprocessor.fit(X_num)  # fits on numeric training features

            X_val_num = _select_numeric(X_val_raw, self.feature_names_in_)
            X_val_transformed = preprocessor.transform(
                X_val_num.reindex(columns=self.feature_names_in_, fill_value=0.0)
            )

            fit_kwargs[f"{model_step}__eval_set"] = [(X_val_transformed, y_val)]
            fit_kwargs[f"{model_step}__early_stopping_rounds"] = 30
            fit_kwargs[f"{model_step}__verbose"] = False

        # Single pipeline.fit handles both XGB with early stopping and others
        self.pipeline.fit(X_num, y_train, **fit_kwargs)

        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_duration = time.time() - start
        self.recency_weight_params = recency_weight_params
        logger.info("Trained %s in %.2fs", self.model_name, self.training_duration)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None or self.feature_names_in_ is None:
            raise RuntimeError("Model not trained or loaded.")
        X_num = _select_numeric(X, self.feature_names_in_)
        # Ensure column order and fill missing
        X_num = X_num.reindex(columns=self.feature_names_in_, fill_value=0.0)
        preds = self.pipeline.predict(X_num)
        return pd.Series(preds, index=X.index, name=self.model_name)

    def save_model(self) -> str:
        if self.pipeline is None:
            raise RuntimeError("Train before saving.")
        path = self.model_dir / f"{self.model_name}.joblib"
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names_in_": self.feature_names_in_,
                "training_timestamp": self.training_timestamp,
                "training_duration": self.training_duration,
                "recency_weight_params": self.recency_weight_params,
            },
            path,
        )
        logger.info("Saved %s → %s", self.model_name, path)
        return str(path)

    def load_model(self, path: Optional[Union[str, Path]] = None) -> None:
        load_path = Path(path or (self.model_dir / f"{self.model_name}.joblib"))
        data = joblib.load(load_path)
        self.pipeline = data["pipeline"]
        self.feature_names_in_ = data["feature_names_in_"]
        self.training_timestamp = data.get("training_timestamp")
        self.training_duration = data.get("training_duration")
        self.recency_weight_params = data.get("recency_weight_params")
        logger.info("Loaded %s (trained %s) from %s", self.model_name, self.training_timestamp, load_path)


# ----------------------------------------------------------------------------- #
# Concrete Models
# ----------------------------------------------------------------------------- #

class RidgeMarginPredictor(BaseNFLPredictor):
    _default = {"model__alpha": 5.0, "model__random_state": SEED}

    def __init__(self, **kw):
        super().__init__("nfl_ridge_margin_predictor", **kw)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([("preprocessing", _preprocessing_pipeline()), ("model", Ridge())])


class SVRMarginPredictor(BaseNFLPredictor):
    _default = {"model__kernel": "rbf", "model__C": 1.0, "model__epsilon": 0.2}

    def __init__(self, **kw):
        super().__init__("nfl_svr_margin_predictor", **kw)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([("preprocessing", _preprocessing_pipeline()), ("model", SVR())])


class XGBoostTotalPredictor(BaseNFLPredictor):
    _default = {
        "model__n_estimators": 500,
        "model__max_depth": 3,
        "model__learning_rate": 0.05,
        "model__subsample": 0.8,
        "model__colsample_bytree": 0.8,
        "model__objective": "reg:squarederror",
        "model__random_state": SEED,
    }

    def __init__(self, **kw):
        super().__init__("nfl_xgb_total_predictor", **kw)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([("preprocessing", _preprocessing_pipeline()), ("model", XGBRegressor(verbosity=0))])


class RFTotalPredictor(BaseNFLPredictor):
    _default = {"model__n_estimators": 400, "model__n_jobs": -1, "model__random_state": SEED}

    def __init__(self, **kw):
        super().__init__("nfl_rf_total_predictor", **kw)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([("preprocessing", _preprocessing_pipeline()), ("model", RandomForestRegressor())])
