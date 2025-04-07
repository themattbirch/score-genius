# backend/nba_score_prediction/ensemble.py

"""
ensemble.py - Ensemble Weight Management Module

This module provides:
  - EnsembleWeightManager: A class to compute and adjust ensemble weights based on error history and game context.
  - generate_enhanced_predictions: A function to generate enhanced predictions for live games by combining
    a main model and quarter-specific predictions, applying uncertainty estimation and ensemble weighting.
"""

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
import joblib
from scipy import stats
import logging

# Import PredictionUncertaintyEstimator from simulation module
from backend.nba_score_prediction.simulation import PredictionUncertaintyEstimator

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------- EnsembleWeightManager --------------------
class EnsembleWeightManager:
    """
    Manages ensemble weights using both standard and adaptive strategies.
    Allows dynamic adjustments based on game context (time remaining, score differential, momentum)
    and tracks error history for potential feedback-based updates.
    """
    def __init__(self, error_history: Optional[Dict[str, Dict[int, float]]] = None, debug: bool = False) -> None:
        self.error_history: Dict[str, Dict[int, float]] = error_history or {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.weighting_strategies: Dict[str, Any] = {
            'standard': self._standard_weights,
            'adaptive': self._adaptive_weights,
        }
        self.weight_history: List[Dict[str, Any]] = []
        self.BASE_WEIGHT_MIN: float = 0.6    # Minimum base weight for main model in adaptive strategy
        self.BASE_WEIGHT_MAX: float = 0.95   # Maximum base weight for main model in adaptive strategy
        self.HEURISTIC_MAIN_WEIGHT_MAX: float = 0.95  # Overall cap for main model weight after heuristics
        self.HEURISTIC_QUARTER_WEIGHT_MAX: float = 0.4  # Cap for quarter model weight after heuristics
        self.HISTORIC_WEIGHT_SMOOTHING: float = 0.7  # Smoothing factor for updating error history
        self.debug: bool = debug

    def log(self, message: str, level: str = "INFO") -> None:
        """Logs messages using the module logger."""
        log_func = getattr(logger, level.lower(), logger.info)
        if self.debug or level.upper() != "DEBUG":
            log_func(f"[WeightManager] {message}")

    def _standard_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """Returns standard fixed weights for a given quarter."""
        weights = {1: (0.80, 0.20), 2: (0.85, 0.15), 3: (0.90, 0.10), 4: (0.95, 0.05)}
        return weights.get(quarter, (0.90, 0.10))

    def _adaptive_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """
        Computes adaptive weights based on error history.
        Returns a tuple (main_weight, quarter_weight) where higher weight is given to the model with lower error.
        """
        if (quarter not in self.error_history.get('main_model', {})) or (quarter not in self.error_history.get('quarter_model', {})):
            self.log(f"Insufficient error history for Q{quarter}. Using standard weights.", level="DEBUG")
            return self._standard_weights(quarter)

        main_error = self.error_history['main_model'][quarter]
        quarter_error = self.error_history['quarter_model'][quarter]
        total_error = main_error + quarter_error
        if total_error <= 0:
            self.log(f"Total error non-positive for Q{quarter}. Using standard weights.", level="WARNING")
            return self._standard_weights(quarter)

        # Weight inversely proportional: main_weight is quarter_error / total_error.
        main_weight = quarter_error / total_error
        main_weight = min(max(main_weight, self.BASE_WEIGHT_MIN), self.BASE_WEIGHT_MAX)
        quarter_weight = 1.0 - main_weight
        self.log(f"Adaptive base weights for Q{quarter}: Main={main_weight:.3f}, Quarter={quarter_weight:.3f} (Errors: Main={main_error:.2f}, Quarter={quarter_error:.2f})", level="DEBUG")
        return main_weight, quarter_weight

    def get_base_weights(self, quarter: int, strategy: str = 'adaptive', **kwargs) -> Tuple[float, float]:
        """Retrieves base weights using the specified strategy."""
        strategy_func = self.weighting_strategies.get(strategy, self._adaptive_weights)
        return strategy_func(quarter, **kwargs)

    def calculate_ensemble(
        self,
        main_prediction: float,
        quarter_prediction: float,
        current_quarter: int,
        weighting_strategy: str = 'adaptive',
        score_differential: float = 0,
        momentum: float = 0,
        time_remaining: Optional[float] = None,
        main_uncertainty: Optional[float] = None,
        quarter_uncertainty: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculates the final ensemble prediction based on main and quarter predictions,
        adjusted by game context factors.

        Returns:
            Tuple of (final_ensemble_prediction, final_main_weight, final_quarter_weight).
        """
        if not (1 <= current_quarter <= 4):
            self.log(f"Invalid quarter {current_quarter} provided to calculate_ensemble. Returning main_prediction.", level="WARNING")
            return main_prediction, 1.0, 0.0

        base_main_weight, base_quarter_weight = self.get_base_weights(
            current_quarter,
            weighting_strategy,
            main_uncertainty=main_uncertainty,
            quarter_uncertainty=quarter_uncertainty
        )
        main_w, quarter_w = base_main_weight, base_quarter_weight

        self.log(f"Initial base weights for Q{current_quarter}: Main={main_w:.3f}, Quarter={quarter_w:.3f}.", level="DEBUG")

        # Time remaining adjustment: Increase main model weight as game progresses.
        if time_remaining is not None and time_remaining >= 0:
            total_minutes = 48.0
            elapsed = total_minutes - time_remaining
            progress = np.clip(elapsed / total_minutes, 0.0, 1.0)
            sigmoid_progress = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
            adjustment = (1.0 - main_w) * sigmoid_progress
            main_w += adjustment
            quarter_w = 1.0 - main_w
            self.log(f"Time Adjustment (Progress: {progress:.2f}): Weights adjusted to Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        # Close game adjustment: Increase quarter weight when score differential is small.
        if abs(score_differential) < 8:
            close_adjust = 0.05 * (1.0 - abs(score_differential) / 8.0)
            quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + close_adjust)
            main_w = 1.0 - quarter_w
            self.log(f"Close Game Adjustment: Weights adjusted to Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        # Momentum adjustment: Increase quarter weight if momentum is high.
        momentum_threshold = 0.3
        if abs(momentum) > momentum_threshold:
            momentum_adjust = 0.05 * (abs(momentum) - momentum_threshold) / (1.0 - momentum_threshold)
            quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + momentum_adjust)
            main_w = 1.0 - quarter_w
            self.log(f"Momentum Adjustment: Weights adjusted to Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        # Prediction gap adjustment: Increase main weight if predictions diverge significantly.
        prediction_gap = abs(main_prediction - quarter_prediction)
        gap_threshold = 15.0
        if prediction_gap > gap_threshold:
            gap_adjust = min(0.3, 0.01 * (prediction_gap - gap_threshold))
            main_w = min(self.HEURISTIC_MAIN_WEIGHT_MAX, main_w + gap_adjust)
            quarter_w = 1.0 - main_w
            self.log(f"Prediction Gap Adjustment (Gap: {prediction_gap:.1f}): Weights adjusted to Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        main_w = min(max(main_w, 1.0 - self.HEURISTIC_QUARTER_WEIGHT_MAX), self.HEURISTIC_MAIN_WEIGHT_MAX)
        quarter_w = 1.0 - main_w
        self.log(f"Final Adjusted Weights: Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        final_ensemble_prediction = main_w * main_prediction + quarter_w * quarter_prediction
        return final_ensemble_prediction, main_w, quarter_w

    def track_weight_usage(self, game_id: str, quarter: int, main_weight: float, quarter_weight: float,
                             prediction_error: Optional[float] = None) -> None:
        """Tracks weight usage and associated prediction error for analysis."""
        record = {
            'game_id': str(game_id),
            'quarter': quarter,
            'main_weight': main_weight,
            'quarter_weight': quarter_weight,
            'timestamp': pd.Timestamp.now(),
            'prediction_error': prediction_error
        }
        self.weight_history.append(record)
        self.log(f"Tracked weights for game {game_id}, Q{quarter} (Error: {prediction_error}).", level="DEBUG")

    def update_error_history_from_validation(self, validation_summary: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        Updates error history based on a validation summary DataFrame.
        The DataFrame should include columns:
          - 'quarter'
          - 'avg_abs_main_error'
          - 'avg_abs_quarter_error'
        """
        self.log("Updating error history from validation summary...", level="INFO")
        updated = False
        for _, row in validation_summary.iterrows():
            q = int(row['quarter'])
            avg_main_err = row.get('avg_abs_main_error')
            avg_quarter_err = row.get('avg_abs_quarter_error')
            if q not in self.error_history.get('main_model', {}) or q not in self.error_history.get('quarter_model', {}):
                self.log(f"Quarter {q} missing in error history. Skipping update.", level="WARNING")
                continue
            if pd.notna(avg_main_err):
                curr_main = self.error_history['main_model'][q]
                new_main = self.HISTORIC_WEIGHT_SMOOTHING * avg_main_err + (1 - self.HISTORIC_WEIGHT_SMOOTHING) * curr_main
                self.error_history['main_model'][q] = new_main
                self.log(f"Updated main_model error for Q{q}: {curr_main:.2f} -> {new_main:.2f}", level="DEBUG")
                updated = True
            if pd.notna(avg_quarter_err):
                curr_quarter = self.error_history['quarter_model'][q]
                new_quarter = self.HISTORIC_WEIGHT_SMOOTHING * avg_quarter_err + (1 - self.HISTORIC_WEIGHT_SMOOTHING) * curr_quarter
                self.error_history['quarter_model'][q] = new_quarter
                self.log(f"Updated quarter_model error for Q{q}: {curr_quarter:.2f} -> {new_quarter:.2f}", level="DEBUG")
                updated = True
        if updated:
            self.log("Error history updated from validation summary.", level="INFO")
        else:
            self.log("No updates applied to error history.", level="DEBUG")
        return self.error_history

# -------------------- generate_enhanced_predictions Function --------------------
def generate_enhanced_predictions(
    live_games_df: pd.DataFrame,
    model_payload_path: str,
    feature_generator: Any,
    quarter_system: Any,
    uncertainty_estimator: Any,
    confidence_viz: Any,
    historical_games_df: Optional[pd.DataFrame] = None,
    team_stats_df: Optional[pd.DataFrame] = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generates enhanced predictions for live games by combining a main model with quarter-specific predictions.
    It loads the model payload, generates features, applies the main model, and refines predictions using the quarter system.
    
    Returns a DataFrame with predictions, uncertainty bounds, and a confidence SVG indicator.
    """
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model']
        required_features = model_payload['features']
        logger.info(f"Loaded model payload with {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        logger.error(f"Model payload file not found at {model_payload_path}.")
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"Model payload missing key: {e}. Requires 'model' and 'features'.")
        return pd.DataFrame()
    except Exception as e:
        logger.exception(f"Failed to load model payload from {model_payload_path}: {e}")
        return pd.DataFrame()

    try:
        features_df = feature_generator.generate_all_features(
            live_games_df,
            historical_games_df=historical_games_df,
            team_stats_df=team_stats_df
        )
        if features_df.empty or features_df.shape[0] != live_games_df.shape[0]:
            logger.error(f"Feature generation returned unexpected shape. Input: {live_games_df.shape}, Output: {features_df.shape}")
            return pd.DataFrame()
        logger.info(f"Feature generation complete. Shape: {features_df.shape}")
    except Exception as e:
        logger.exception(f"Exception during feature generation: {e}")
        return pd.DataFrame()

    for f in required_features:
        if f not in features_df.columns:
            features_df[f] = 0  # Fill missing with default 0
            logger.warning(f"Missing required feature '{f}' added with default 0.")
    try:
        X_main = features_df.reindex(columns=required_features, fill_value=0)
        if X_main.isnull().any().any():
            logger.warning("Null values detected in X_main after reindexing.")
    except Exception as e:
        logger.exception(f"Error selecting required features: {e}")
        return pd.DataFrame()

    try:
        main_predictions = main_model.predict(X_main)
        logger.info(f"Main model predictions generated for {len(main_predictions)} samples.")
    except Exception as e:
        logger.exception(f"Error during main model prediction: {e}")
        avg_score = feature_generator.defaults.get('avg_pts_for', 110.0)
        main_predictions = np.full(len(features_df), avg_score)
        logger.info(f"Using fallback predictions with average score {avg_score}.")

    results = []
    try:
        # Obtain historical accuracy stats from the uncertainty estimator, if available.
        if hasattr(uncertainty_estimator, 'get_coverage_stats'):
            historic_accuracy = uncertainty_estimator.get_coverage_stats().set_index('quarter').to_dict('index')
        else:
            historic_accuracy = None
    except Exception as e:
        logger.warning(f"Failed to retrieve historical accuracy stats: {e}")
        historic_accuracy = None

    features_df = features_df.reset_index(drop=True)
    for i, game_row in features_df.iterrows():
        game_id_log = game_row.get('game_id', f'index_{i}')
        try:
            game_data_dict = game_row.to_dict()
            main_pred = float(main_predictions[i])
            # Assume quarter_system has a weight_manager attribute
            ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
                game_data_dict=game_data_dict,
                main_model_prediction=main_pred,
                weight_manager=quarter_system.weight_manager
            )
            current_quarter = int(game_data_dict.get('current_quarter', 0))
            home_score_live = float(game_data_dict.get('home_score', 0))
            away_score_live = float(game_data_dict.get('away_score', 0))
            score_diff = abs(float(game_data_dict.get('score_differential', home_score_live - away_score_live)))
            momentum = float(game_data_dict.get('cumulative_momentum', 0.0))
            lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
                prediction=float(ensemble_pred),
                current_quarter=current_quarter,
                historic_accuracy=historic_accuracy
            )
            svg_indicator = confidence_viz.create_confidence_svg(
                prediction=float(ensemble_pred),
                lower_bound=float(lower_b),
                upper_bound=float(upper_b),
                current_quarter=current_quarter,
                current_home_score=home_score_live
            )
            result_row = {
                'game_id': game_data_dict.get('game_id'),
                'home_team': game_data_dict.get('home_team'),
                'away_team': game_data_dict.get('away_team'),
                'game_date': game_data_dict.get('game_date'),
                'current_quarter': current_quarter,
                'home_score': home_score_live,
                'away_score': away_score_live,
                'main_model_pred': main_pred,
                'quarter_model_sum_pred': breakdown.get('quarter_model_sum'),
                'ensemble_pred': float(ensemble_pred),
                'lower_bound': float(lower_b),
                'upper_bound': float(upper_b),
                'confidence_pct': float(conf_pct),
                'confidence_svg': svg_indicator,
                'main_weight': breakdown.get('weights', {}).get('main'),
                'quarter_weight': breakdown.get('weights', {}).get('quarter')
            }
            results.append(result_row)
        except Exception as e:
            logger.exception(f"Error processing game {game_id_log}: {e}")
            results.append({
                'game_id': game_id_log,
                'home_team': game_row.get('home_team'),
                'away_team': game_row.get('away_team'),
                'game_date': game_row.get('game_date'),
                'error': str(e),
                'current_quarter': game_row.get('current_quarter', 0),
                'home_score': game_row.get('home_score', 0),
                'away_score': game_row.get('away_score', 0),
                'main_model_pred': np.nan,
                'quarter_model_sum_pred': np.nan,
                'ensemble_pred': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'confidence_pct': np.nan,
                'confidence_svg': '<svg>Error</svg>',
                'main_weight': np.nan,
                'quarter_weight': np.nan,
            })

    return pd.DataFrame(results)
    