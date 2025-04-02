# backend/nba_score_prediction/ensemble.py

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
import joblib
from scipy import stats

# -------------------- Ensemble Weight Manager --------------------
class EnsembleWeightManager:
    """
    Manages ensemble weights, including base strategies and dynamic adjustments,
    to calculate the final ensemble prediction.
    """
    def __init__(self, error_history: Optional[Dict[str, Dict[int, float]]] = None, debug: bool = False):
        self.error_history = error_history or {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.weighting_strategies = {
            'standard': self._standard_weights,
            'adaptive': self._adaptive_weights,
        }
        self.weight_history: List[Dict[str, Any]] = []
        # Constants for adaptive weighting and heuristic adjustments
        self.BASE_WEIGHT_MIN = 0.6  # Min weight for main model in adaptive strategy
        self.BASE_WEIGHT_MAX = 0.95 # Max weight for main model in adaptive strategy
        self.HEURISTIC_MAIN_WEIGHT_MAX = 0.95 # Overall cap after heuristics
        self.HEURISTIC_QUARTER_WEIGHT_MAX = 0.4 # Cap for quarter weight after heuristics (implicitly caps main at 0.6)

        self.HISTORIC_WEIGHT_SMOOTHING = 0.7 # Smoothing factor for error history updates
        self.debug = debug

    def log(self, message, level="INFO"):
         """Log messages based on debug flag."""
         if self.debug or level != "DEBUG":
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [WeightManager] {level}: {message}")

    def _standard_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """ Standard weighting strategy. """
        weights = { 1: (0.80, 0.20), 2: (0.85, 0.15), 3: (0.90, 0.10), 4: (0.95, 0.05) }
        return weights.get(quarter, (0.90, 0.10)) # Default for Q0 or invalid

    def _adaptive_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """ Adaptive weighting based on historical error rates. """
        if not self.error_history or quarter not in self.error_history.get('main_model', {}) or quarter not in self.error_history.get('quarter_model', {}):
            self.log(f"Insufficient error history for Q{quarter}. Using standard weights.", level="DEBUG")
            return self._standard_weights(quarter)

        main_error = self.error_history['main_model'][quarter]
        quarter_error = self.error_history['quarter_model'][quarter]

        total_error = main_error + quarter_error
        if total_error <= 0: # Avoid division by zero
             self.log(f"Total error is zero or negative for Q{quarter}. Using standard weights.", level="WARNING")
             return self._standard_weights(quarter)

        # Higher weight to model with lower error (weight = other_error / total_error)
        main_weight = quarter_error / total_error

        # Constrain base weights
        main_weight = min(max(main_weight, self.BASE_WEIGHT_MIN), self.BASE_WEIGHT_MAX)
        quarter_weight = 1.0 - main_weight
        self.log(f"Adaptive base weights for Q{quarter}: Main={main_weight:.3f}, Quarter={quarter_weight:.3f} (Errors M:{main_error:.2f}, Q:{quarter_error:.2f})", level="DEBUG")
        return main_weight, quarter_weight

    def get_base_weights(self, quarter: int, strategy: str = 'adaptive', **kwargs) -> Tuple[float, float]:
        """ Get base ensemble weights using the specified strategy. """
        strategy_func = self.weighting_strategies.get(strategy, self._adaptive_weights)
        return strategy_func(quarter, **kwargs)

    def calculate_ensemble(
        self,
        main_prediction: float,
        quarter_prediction: float, # This is the quarter model's prediction for the *full game* (current + remaining)
        current_quarter: int,
        weighting_strategy: str = 'adaptive',
        score_differential: float = 0,
        momentum: float = 0,
        time_remaining: Optional[float] = None, # Assume minutes
        main_uncertainty: Optional[float] = None, # Pass through if needed by base strategy
        quarter_uncertainty: Optional[float] = None # Pass through if needed by base strategy
    ) -> Tuple[float, float, float]:
        """
        Calculates the final ensemble prediction after applying heuristic adjustments
        to the base weights.

        Args:
            main_prediction: Prediction from the main full-game model.
            quarter_prediction: Full-game prediction derived from quarter models (current score + predicted remaining).
            current_quarter: Current quarter of the game (1-4, assume 0 pre-game handled before calling).
            weighting_strategy: Strategy for getting base weights ('adaptive', 'standard').
            score_differential: Current score difference (home - away).
            momentum: Current game momentum metric.
            time_remaining: Estimated time remaining in minutes.
            main_uncertainty: Optional uncertainty measure for main model.
            quarter_uncertainty: Optional uncertainty measure for quarter model.

        Returns:
            Tuple: (final_ensemble_prediction, final_main_weight, final_quarter_weight)
        """
        if not (1 <= current_quarter <= 4):
             self.log(f"calculate_ensemble called with invalid quarter {current_quarter}. Returning main prediction.", level="WARNING")
             return main_prediction, 1.0, 0.0

        # 1. Get Base Weights
        base_main_weight, base_quarter_weight = self.get_base_weights(
            current_quarter,
            weighting_strategy,
            main_uncertainty=main_uncertainty,
            quarter_uncertainty=quarter_uncertainty
        )
        main_w, quarter_w = base_main_weight, base_quarter_weight # Start with base weights

        # 2. Apply Heuristic Adjustments (Based on global function logic)
        self.log(f"Applying heuristics to base weights (M:{main_w:.3f}, Q:{quarter_w:.3f}). Context: Diff={score_differential:.1f}, Mom={momentum:.2f}, TimeRem={time_remaining}", level="DEBUG")

        # Adjust based on time remaining (Increase main weight as game progresses)
        if time_remaining is not None and time_remaining >= 0:
            total_minutes = 48.0
            elapsed = total_minutes - time_remaining
            progress = min(1.0, max(0.0, elapsed / total_minutes))
            # Sigmoid function to smoothly increase main model weight as game progresses
            sigmoid_progress = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5))) # Range [~0, ~1]
            adjustment = (1.0 - main_w) * sigmoid_progress # How much closer to 1.0 can we get?
            main_w = main_w + adjustment
            quarter_w = 1.0 - main_w
            self.log(f"  Time Adjustment (Progress {progress:.2f}): Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if game is close (Give quarter model slightly more weight)
        if abs(score_differential) < 8:
            # Scale adjustment from 0.05 (at diff 0) down to 0 (at diff 8)
            close_game_adjustment = 0.05 * (1.0 - abs(score_differential) / 8.0)
            # Apply adjustment but cap quarter weight
            quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + close_game_adjustment)
            main_w = 1.0 - quarter_w
            self.log(f"  Close Game Adjustment: Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if high momentum (Give quarter model slightly more weight)
        momentum_threshold = 0.3
        if abs(momentum) > momentum_threshold:
             # Scale adjustment based on momentum magnitude beyond threshold
             momentum_adjustment = 0.05 * (abs(momentum) - momentum_threshold) / (1.0 - momentum_threshold) # Normalize effect range
             # Apply adjustment but cap quarter weight
             quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + momentum_adjustment)
             main_w = 1.0 - quarter_w
             self.log(f"  Momentum Adjustment: Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if predictions diverge significantly (Trust main model more)
        prediction_gap = abs(main_prediction - quarter_prediction)
        gap_threshold = 15.0
        if prediction_gap > gap_threshold:
            # Scale adjustment based on how large the gap is
            discrepancy_adjustment = min(0.3, 0.01 * (prediction_gap - gap_threshold)) # Apply adjustment only beyond threshold
            # Apply adjustment but cap main weight
            main_w = min(self.HEURISTIC_MAIN_WEIGHT_MAX, main_w + discrepancy_adjustment)
            quarter_w = 1.0 - main_w
            self.log(f"  Prediction Gap Adjustment (Gap {prediction_gap:.1f}): Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # --- Final Check on Weights ---
        # Ensure weights sum to 1 and are within reasonable bounds after all adjustments
        main_w = min(max(main_w, 1.0 - self.HEURISTIC_QUARTER_WEIGHT_MAX), self.HEURISTIC_MAIN_WEIGHT_MAX)
        quarter_w = 1.0 - main_w
        self.log(f"Final Adjusted Weights: Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        # 3. Calculate Final Weighted Prediction
        final_ensemble_prediction = main_w * main_prediction + quarter_w * quarter_prediction

        # 4. Return prediction and final weights (Confidence calculation removed)
        return final_ensemble_prediction, main_w, quarter_w


    def track_weight_usage(
        self,
        game_id: str,
        quarter: int,
        main_weight: float,
        quarter_weight: float,
        prediction_error: Optional[float] = None
    ) -> None:
        """ Tracks the weights used and their performance for later analysis. """
        # (Implementation remains the same as provided in Code Block 1)
        if not isinstance(game_id, str): game_id = str(game_id) # Ensure string game_id
        self.weight_history.append({
            'game_id': game_id,
            'quarter': quarter,
            'main_weight': main_weight,
            'quarter_weight': quarter_weight,
            'timestamp': pd.Timestamp.now(),
            'prediction_error': prediction_error
        })
        self.log(f"Tracked weights for game {game_id}, Q{quarter}. Error={prediction_error}", level="DEBUG")


    def update_weights_from_feedback(
        self,
        recent_window: int = 10
    ) -> Dict[str, Dict[int, float]]:
        """ Updates error history based on recent prediction performance tracked. """
        # (Implementation remains the same as provided in Code Block 1)
        if len(self.weight_history) < recent_window:
             self.log(f"Not enough history ({len(self.weight_history)}/{recent_window}) to update error metrics.", level="DEBUG")
             return self.error_history

        recent_data = self.weight_history[-recent_window:]
        recent_df = pd.DataFrame(recent_data)

        if 'prediction_error' not in recent_df.columns or recent_df['prediction_error'].isna().all():
            self.log("No valid prediction errors found in recent history. Cannot update.", level="DEBUG")
            return self.error_history

        # Calculate average *absolute* error by quarter might be more stable for weighting
        recent_df['abs_error'] = recent_df['prediction_error'].abs()
        quarter_avg_abs_errors = recent_df.groupby('quarter')['abs_error'].mean().to_dict()
        self.log(f"Recent avg absolute errors by quarter: {quarter_avg_abs_errors}", level="DEBUG")

        updated = False
        # Update error history using exponential moving average approach
        for q, avg_abs_error in quarter_avg_abs_errors.items():
             if q in self.error_history.get('main_model', {}) and q in self.error_history.get('quarter_model', {}):
                  # How to update? Assume the 'prediction_error' tracked was for the *ensemble*.
                  # We need errors for main and quarter models separately to properly update adaptive weights.
                  # This current update logic assumes the ensemble error reflects 'main_model' error, which is incorrect.
                  # TODO: Requires tracking separate errors or a different update strategy for adaptive weights.
                  # Placeholder: Update 'main_model' error history based on ensemble error for now.
                  current_error = self.error_history['main_model'][q]
                  new_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_error +
                               (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_error)
                  self.error_history['main_model'][q] = new_error
                  self.log(f"Updated main_model error history for Q{q}: {current_error:.2f} -> {new_error:.2f}", level="DEBUG")
                  updated = True
             else:
                  self.log(f"Quarter {q} not found in error history structure. Skipping update.", level="WARNING")

        if updated:
             self.log("Error history updated based on feedback.", level="INFO")
        else:
             self.log("Error history not updated.", level="DEBUG")

        return self.error_history
    
    # Modify the update function signature to accept processed validation data
    def update_error_history_from_validation(self, validation_summary: pd.DataFrame):
        """
        Updates error history based on aggregated validation results.

        Args:
            validation_summary: A DataFrame with columns like
                                'quarter', 'avg_abs_main_error', 'avg_abs_quarter_error'.
                                (This needs to be calculated from your validation run)
        """
        self.log("Updating error history from validation summary...")
        updated = False
        for _, row in validation_summary.iterrows():
            q = int(row['quarter'])
            avg_abs_main_err = row.get('avg_abs_main_error')
            avg_abs_quarter_err = row.get('avg_abs_quarter_error')

            if q not in self.error_history.get('main_model', {}) or \
            q not in self.error_history.get('quarter_model', {}):
                self.log(f"Quarter {q} not found in error history structure. Skipping.", level="WARNING")
                continue

            if pd.notna(avg_abs_main_err):
                current_main_error = self.error_history['main_model'][q]
                # Use smoothing (EMA - Exponential Moving Average) or just replace
                new_main_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_main_err +
                                (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_main_error)
                self.error_history['main_model'][q] = new_main_error
                self.log(f"Updated main_model error history for Q{q}: {current_main_error:.2f} -> {new_main_error:.2f}", level="DEBUG")
                updated = True

            if pd.notna(avg_abs_quarter_err):
                current_quarter_error = self.error_history['quarter_model'][q]
                new_quarter_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_quarter_err +
                                    (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_quarter_error)
                self.error_history['quarter_model'][q] = new_quarter_error
                self.log(f"Updated quarter_model error history for Q{q}: {current_quarter_error:.2f} -> {new_quarter_error:.2f}", level="DEBUG")
                updated = True

        if updated:
            self.log("Error history updated based on validation feedback.", level="INFO")
        else:
            self.log("Error history not updated (no valid data?).", level="DEBUG")
        return self.error_history

    # Note: The old 'update_weights_from_feedback' based on live ensemble error
    # should probably be removed or significantly rethought if you want truly adaptive live weights.
    # Relying on periodic updates from validation is more robust.

def generate_enhanced_predictions(
    live_games_df: pd.DataFrame,
    model_payload_path: str, # Path to the saved model payload
    feature_generator: Any, # Instance of NBAFeatureEngine
    quarter_system: Any, # Instance of QuarterSpecificModelSystem
    uncertainty_estimator: Any, # Instance of PredictionUncertaintyEstimator
    confidence_viz: Any, # Instance of ConfidenceVisualizer
    historical_games_df: Optional[pd.DataFrame] = None,
    team_stats_df: Optional[pd.DataFrame] = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generates enhanced predictions for live games using a loaded model payload.

    Args:
        live_games_df: DataFrame of games to predict.
        model_payload_path: Path to the file containing the model and required features.
        feature_generator: Instance of NBAFeatureEngine.
        quarter_system: Instance of QuarterSpecificModelSystem.
        uncertainty_estimator: Instance of PredictionUncertaintyEstimator.
        confidence_viz: Instance of ConfidenceVisualizer.
        historical_games_df: Optional DataFrame of historical games for feature generation.
        team_stats_df: Optional DataFrame of team stats for feature generation.
        debug: Whether to print debug messages.

    Returns:
        DataFrame with predictions, uncertainty bounds, and confidence SVG.
    """
    # --- Load Model and Required Features ---
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model']
        required_features = model_payload['features']
        if debug: print(f"[generate_enhanced_predictions] Loaded model and {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        print(f"ERROR: Model payload file not found at {model_payload_path}. Cannot generate predictions.")
        return pd.DataFrame() # Return empty DataFrame on critical error
    except KeyError as e:
         print(f"ERROR: Model payload at {model_payload_path} is missing key: {e}. Requires 'model' and 'features'.")
         return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to load model payload from {model_payload_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    # --- Generate Features ---
    # Use the generate_all_features wrapper method
    # Ensure it generates all features needed by both main_model and quarter_system
    try:
        features_df = feature_generator.generate_all_features(
            live_games_df,
            historical_games_df=historical_games_df,
            team_stats_df=team_stats_df
        )
        if features_df.empty or features_df.shape[0] != live_games_df.shape[0]:
             print(f"ERROR: Feature generation failed or returned unexpected shape. Input: {live_games_df.shape}, Output: {features_df.shape}")
             return pd.DataFrame()
        if debug: print(f"[generate_enhanced_predictions] Feature generation complete. Shape: {features_df.shape}")
    except Exception as e:
        print(f"ERROR: Exception during feature generation: {e}")
        traceback.print_exc()
        return pd.DataFrame()


    # --- Main Model Prediction ---
    if not required_features: # Should have been caught during loading, but double-check
         print("ERROR: No required features list available for the main model. Cannot predict.")
         return pd.DataFrame()

    # Ensure all required features are present in the generated features_df
    missing_features_check = []
    for f in required_features:
        if f not in features_df.columns:
            features_df[f] = 0 # Use 0 as default, or implement better default logic
            missing_features_check.append(f)
    if missing_features_check:
        if debug: print(f"[generate_enhanced_predictions] Warning: Added {len(missing_features_check)} missing columns with default 0 for main model: {missing_features_check[:5]}...")
    else:
         if debug: print(f"[generate_enhanced_predictions] All {len(required_features)} required features found.")

    # Select ONLY required features IN THE CORRECT ORDER using reindex
    try:
        # Use fill_value=0 for any columns that might still be missing after the check (shouldn't happen often)
        X_main = features_df.reindex(columns=required_features, fill_value=0)
        if X_main.isnull().any().any():
             print("[generate_enhanced_predictions] Warning: Null values detected in feature matrix X_main after reindex. Check defaults/data.")
             # Optionally fill NaNs again if needed: X_main = X_main.fillna(0)
    except Exception as e:
         print(f"[generate_enhanced_predictions] CRITICAL Error selecting/reindexing required features: {e}. Cannot predict.")
         traceback.print_exc()
         return pd.DataFrame()

    # --- Predict ---
    try:
        main_predictions = main_model.predict(X_main)
        if debug: print(f"[generate_enhanced_predictions] Main model predictions generated ({len(main_predictions)}).")
    except Exception as e:
        print(f"[generate_enhanced_predictions] Error during main model prediction: {e}")
        traceback.print_exc()
        # Fallback prediction using feature_generator defaults
        avg_score = feature_generator.defaults.get('avg_pts_for', 110.0)
        main_predictions = np.full(len(features_df), avg_score)
        if debug: print(f"[generate_enhanced_predictions] Using fallback predictions ({avg_score}).")


    # --- Process Each Game for Ensemble & Uncertainty ---
    results = []
    # Placeholder for historical accuracy - load real stats if available for dynamic intervals
    historic_accuracy = uncertainty_estimator.get_coverage_stats().set_index('quarter').to_dict('index') if hasattr(uncertainty_estimator, 'get_coverage_stats') else None

    # Use iterrows carefully on the potentially large features_df. Consider optimization if this loop is slow.
    # Ensure index aligns with main_predictions
    features_df = features_df.reset_index() # Ensure index is 0, 1, 2... matching main_predictions array

    for i, game_row in features_df.iterrows():
        game_id_log = game_row.get('game_id', f'index_{i}') # For logging
        try:
            # Convert row to dict for quarter_system processing
            # Ensure all necessary columns (like quarter scores, current_quarter) are present
            game_data_dict = game_row.to_dict()
            main_pred = float(main_predictions[i]) # Ensure float

            # Get ensemble prediction using quarter system
            # Ensure predict_final_score handles potential missing keys in game_data_dict
            ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
                game_data_dict=game_data_dict,
                main_model_prediction=main_pred,
                weight_manager=quarter_system.weight_manager # Assuming weight_manager is accessible like this
            )

            # Estimate uncertainty
            current_quarter = int(game_data_dict.get('current_quarter', 0))
            # Use calculated score_differential if available, else calculate from scores
            score_diff_feat = game_data_dict.get('score_differential')
            home_score_live = float(game_data_dict.get('home_score', 0))
            away_score_live = float(game_data_dict.get('away_score', 0))
            score_margin = abs(float(score_diff_feat if pd.notna(score_diff_feat) else (home_score_live - away_score_live)))
            momentum = float(game_data_dict.get('cumulative_momentum', 0.0)) # Example context feature

            # Use dynamic interval adjustment if historic_accuracy was loaded
            lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
                prediction=float(ensemble_pred),
                current_quarter=current_quarter,
                historic_accuracy=historic_accuracy # Pass loaded historical coverage dict
            )
            # Fallback if dynamic adjustment isn't used/fails
            # lower_b, upper_b, _ = uncertainty_estimator.calculate_prediction_interval(
            #     prediction=float(ensemble_pred),
            #     current_quarter=current_quarter,
            #     score_margin=score_margin,
            #     momentum=momentum
            # )
            # conf_pct = max(0.0, min(100.0, 100.0 - ((upper_b-lower_b) / uncertainty_estimator.EXPECTED_RANGES.get(current_quarter, 25.0) * 75.0)))


            # Create SVG indicator
            svg_indicator = confidence_viz.create_confidence_svg(
                prediction=float(ensemble_pred),
                lower_bound=float(lower_b),
                upper_bound=float(upper_b),
                current_quarter=current_quarter,
                current_home_score=float(home_score_live) # Pass current score if live
            )

            # Compile result
            result_row = {
                'game_id': game_data_dict.get('game_id'),
                'home_team': game_data_dict.get('home_team'),
                'away_team': game_data_dict.get('away_team'),
                'game_date': game_data_dict.get('game_date'),
                'current_quarter': current_quarter,
                'home_score': home_score_live,
                'away_score': away_score_live,
                'main_model_pred': main_pred,
                'quarter_model_sum_pred': breakdown.get('quarter_model_sum'), # Changed key for clarity
                'ensemble_pred': float(ensemble_pred),
                'lower_bound': float(lower_b),
                'upper_bound': float(upper_b),
                'confidence_pct': float(conf_pct),
                'confidence_svg': svg_indicator,
                'main_weight': breakdown.get('weights', {}).get('main'),
                'quarter_weight': breakdown.get('weights', {}).get('quarter'),
                # Add predicted quarter scores if needed
                **{f'predicted_{k}': v for k, v in breakdown.get('quarter_predictions', {}).items()}
            }
            results.append(result_row)

        except Exception as e:
            if debug: print(f"Error processing game {game_id_log}: {e}")
            traceback.print_exc()
            # Append minimal error info - ensure keys match expected output structure
            results.append({
                'game_id': game_id_log,
                'home_team': game_row.get('home_team'),
                'away_team': game_row.get('away_team'),
                'game_date': game_row.get('game_date'),
                'error': str(e),
                # Add defaults for other columns to maintain structure
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

