# backend/nfl_score_prediction/ensemble.py
"""
ensemble.py - NFL Pre-Game Ensemble Model Module

This module provides a class to create and manage an ensemble of pre-game NFL
prediction models. It combines predictions from multiple trained models
(e.g., Ridge, SVR, XGBoost) using a weighted-average approach to produce a
single, more robust forecast.

This system is for pre-game predictions only and does not contain live-game logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Mapping

import numpy as np
import pandas as pd

from backend.nfl_score_prediction.models import BaseNFLPredictor, MODEL_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class NFLEnsemble:
    """
    Weighted ensemble of pre-trained NFL predictors (pre-game only).
    Loads each model once and reuses it for fast inference.
    """

    def __init__(self, model_weights: Mapping[str, float], model_dir: Optional[Path] = None) -> None:
        if not model_weights:
            raise ValueError("model_weights cannot be empty.")

        total_w = float(sum(model_weights.values()))
        if not np.isfinite(total_w) or np.isclose(total_w, 0.0):
            # fallback to equal weights
            k = len(model_weights)
            logger.warning("Weights sum is invalid (%.4f). Using equal weights.", total_w)
            self.model_weights = {m: 1.0 / k for m in model_weights}
        elif not np.isclose(total_w, 1.0):
            logger.warning("Weights sum=%.4f. Normalizing to 1.0.", total_w)
            self.model_weights = {m: w / total_w for m, w in model_weights.items()}
        else:
            self.model_weights = dict(model_weights)

        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.models: Dict[str, BaseNFLPredictor] = {}
        self._loaded = False
        logger.info("NFLEnsemble init → %s", list(self.model_weights.keys()))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _load_single(self, model_name: str) -> BaseNFLPredictor:
        p = BaseNFLPredictor(model_name=model_name, model_dir=self.model_dir)
        p.load_model()
        return p

    def load_models(self) -> None:
        logger.info("Loading ensemble models...")
        for name in self.model_weights:
            try:
                self.models[name] = self._load_single(name)
                logger.debug("Loaded %s", name)
            except Exception as e:
                logger.error("Failed to load '%s': %s", name, e)
                raise
        self._loaded = True
        logger.info("All models loaded.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._loaded:
            raise RuntimeError("Call load_models() before predict().")

        out = pd.Series(0.0, index=X.index, dtype=float)
        for name, predictor in self.models.items():
            w = self.model_weights.get(name, 0.0)
            if w == 0:
                continue
            preds = predictor.predict(X)
            out = out.add(preds * w, fill_value=0.0)
        return out
