# backend/mlb_score_prediction/ensemble.py

"""
ensemble.py - MLB Ensemble Weight Management Module

This module provides:
  - EnsembleWeightManager: A class to compute and adjust ensemble weights for MLB predictions
    based on error history and game context (innings, run differentials, etc.).
  - generate_enhanced_predictions: A function to generate enhanced predictions for live MLB games
    by combining a main model and inning-specific predictions, applying uncertainty estimation
    and ensemble weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)
# Ensure logger is configured if not done globally
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

NUM_INNINGS_REGULATION = 9

# -------------------- EnsembleWeightManager (for MLB) --------------------
class EnsembleWeightManager:
    """
    Manages ensemble weights for MLB predictions using standard and adaptive strategies.
    Adjusts weights based on game context (inning, run differential, momentum)
    and tracks error history.
    """
    def __init__(self, error_history: Optional[Dict[str, Dict[int, float]]] = None, debug: bool = False) -> None:
        # MLB-specific error history (example values in runs)
        self.error_history: Dict[str, Dict[int, float]] = error_history or {
            'main_model': {i: (2.0 - (i-1)*0.1) for i in range(1, NUM_INNINGS_REGULATION + 1)}, # e.g., Error for inning 1: 2.0, Inning 9: 1.2
            'inning_model': {i: (1.5 - (i-1)*0.05) for i in range(1, NUM_INNINGS_REGULATION + 1)} # e.g., Error for inning 1: 1.5, Inning 9: 1.1
        }
        self.weighting_strategies: Dict[str, Any] = {
            'standard': self._standard_weights,
            'adaptive': self._adaptive_weights,
        }
        self.weight_history: List[Dict[str, Any]] = []
        # These constants may need tuning for MLB
        self.BASE_WEIGHT_MAIN_MIN: float = 0.50 # Min weight for main model in adaptive
        self.BASE_WEIGHT_MAIN_MAX: float = 0.90 # Max weight for main model in adaptive
        
        self.HEURISTIC_MAIN_WEIGHT_MAX: float = 0.95 # Overall cap for main model weight
        self.HEURISTIC_INNING_WEIGHT_MAX: float = 0.50 # Overall cap for inning model weight (1 - HEURISTIC_MAIN_WEIGHT_MIN)

        self.HISTORIC_WEIGHT_SMOOTHING: float = 0.7
        self.debug: bool = debug
        self.log("MLB EnsembleWeightManager initialized.", level="DEBUG" if self.debug else "INFO")


    def log(self, message: str, level: str = "INFO") -> None:
        """Logs messages using the module logger."""
        log_func = getattr(logger, level.lower(), logger.info)
        if self.debug or level.upper() != "DEBUG": # Log non-DEBUG messages always, DEBUG only if self.debug
            log_func(f"[MLBWeightManager] {message}")

    def _standard_weights(self, inning: int, **kwargs) -> Tuple[float, float]:
        """Returns standard fixed weights for a given MLB inning."""
        # Example: Gradually increase main model weight. Tune for MLB.
        weights = {
            1: (0.60, 0.40), 2: (0.65, 0.35), 3: (0.70, 0.30),
            4: (0.75, 0.25), 5: (0.80, 0.20), 6: (0.85, 0.15),
            7: (0.90, 0.10), 8: (0.92, 0.08), 9: (0.95, 0.05)
        }
        return weights.get(inning, (0.80, 0.20)) # Default for out-of-range (e.g. extra innings if not specified)

    def _adaptive_weights(self, inning: int, **kwargs) -> Tuple[float, float]:
        """
        Computes adaptive weights based on error history for MLB.
        Returns (main_weight, inning_weight). Higher weight to model with lower error.
        """
        main_model_errors = self.error_history.get('main_model', {})
        inning_model_errors = self.error_history.get('inning_model', {})

        if not isinstance(main_model_errors, dict) or not isinstance(inning_model_errors, dict) or \
           inning not in main_model_errors or inning not in inning_model_errors:
            self.log(f"Insufficient error history for Inning {inning}. Using standard weights.", level="DEBUG")
            return self._standard_weights(inning)

        main_error = main_model_errors[inning]
        inning_error = inning_model_errors[inning]
        total_error = main_error + inning_error

        if total_error <= 1e-6: # Avoid division by zero or tiny errors
            self.log(f"Total error non-positive or too small for Inning {inning}. Using standard weights.", level="WARNING")
            return self._standard_weights(inning)

        main_weight = inning_error / total_error # Weight main model inversely to its error share
        main_weight = min(max(main_weight, self.BASE_WEIGHT_MAIN_MIN), self.BASE_WEIGHT_MAIN_MAX)
        inning_weight = 1.0 - main_weight
        
        self.log(f"Adaptive base weights for I{inning}: Main={main_weight:.3f}, Inning={inning_weight:.3f} (Errors: Main={main_error:.2f}, Inning={inning_error:.2f})", level="DEBUG")
        return main_weight, inning_weight

    def get_base_weights(self, inning: int, strategy: str = 'adaptive', **kwargs) -> Tuple[float, float]:
        """Retrieves base weights for MLB using the specified strategy."""
        strategy_func = self.weighting_strategies.get(strategy, self._adaptive_weights)
        return strategy_func(inning, **kwargs)

    def calculate_ensemble_mlb(
        self,
        main_pred_home: float,
        main_pred_away: float,
        inning_model_total_pred_home: float,
        inning_model_total_pred_away: float,
        current_inning: int, # Innings completed
        weighting_strategy: str = 'adaptive',
        run_differential: float = 0,
        momentum: float = 0, # MLB-specific momentum metric
        innings_remaining: Optional[int] = None, # Innings left in regulation
        # Optional uncertainty params, currently not used in adaptive logic but could be
        main_uncertainty: Optional[float] = None, 
        inning_uncertainty: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculates final ensemble run predictions for home and away MLB teams.
        Adjusts weights based on game context.

        Returns:
            Tuple of (final_ensemble_home, final_ensemble_away, final_main_weight, final_inning_weight).
        """
        if not (0 <= current_inning <= NUM_INNINGS_REGULATION + 5): # Allow for some extra innings if needed
            self.log(f"Invalid current_inning {current_inning} for calculate_ensemble_mlb. Using unweighted main predictions.", level="WARNING")
            return main_pred_home, main_pred_away, 1.0, 0.0
        
        # Use current_inning + 1 for prospective weights if current_inning is completed innings
        # Or just current_inning if it means "inning in progress" for which prediction is made
        # Assuming current_inning is "completed innings", so we look at weights for inning_in_progress
        inning_for_weights = min(current_inning + 1, NUM_INNINGS_REGULATION) if current_inning < NUM_INNINGS_REGULATION else NUM_INNINGS_REGULATION


        base_main_weight, base_inning_weight = self.get_base_weights(
            inning_for_weights, # Use the inning for which prediction context applies
            weighting_strategy,
            main_uncertainty=main_uncertainty, # Pass along if base strategies use them
            inning_uncertainty=inning_uncertainty
        )
        main_w, inning_w = base_main_weight, base_inning_weight
        self.log(f"Initial base weights for context of I{inning_for_weights}: Main={main_w:.3f}, Inning={inning_w:.3f}.", level="DEBUG")

        # --- Contextual Adjustments (tuned for MLB) ---
        # Game Progress adjustment: More weight to main model early, inning model can gain if it tracks well.
        # This is a bit different from NBA's time; here it's based on innings.
        if innings_remaining is not None and innings_remaining >= 0:
            progress = np.clip((NUM_INNINGS_REGULATION - innings_remaining) / float(NUM_INNINGS_REGULATION), 0.0, 1.0)
            # Sigmoid can make start/end game weights more pronounced.
            # Early game (low progress), sigmoid is low, less adjustment from inning model.
            # Late game (high progress), sigmoid is high, more trust in what has transpired.
            # If inning_w is high initially, this can temper it early on.
            sigmoid_progress = 1.0 / (1.0 + np.exp(-10 * (progress - 0.75))) # Shifted to emphasize later innings for inning model impact

            # Adjustment: if inning model has high base weight, reduce it slightly early, increase it late.
            # If main_w is high, this gives a bit more to inning_w as game progresses.
            adjustment_to_inning_w = (1.0 - inning_w) * sigmoid_progress * 0.2 # Modest adjustment factor
            
            inning_w += adjustment_to_inning_w
            main_w = 1.0 - inning_w
            self.log(f"Game Progress Adjustment (Progress: {progress:.2f}, I{inning_for_weights}): Weights -> Main={main_w:.3f}, Inning={inning_w:.3f}", level="DEBUG")

        # Close game adjustment: (e.g., run_differential <= 2 runs)
        # If game is close, inning-by-inning dynamics (captured by inning_model) might be more telling.
        close_game_threshold_runs = 2 
        if abs(run_differential) <= close_game_threshold_runs:
            # Increase inning_w more if game is very close. Max effect at 0 diff.
            close_adjust = 0.10 * (1.0 - abs(run_differential) / close_game_threshold_runs)
            inning_w = min(self.HEURISTIC_INNING_WEIGHT_MAX, inning_w + close_adjust)
            main_w = 1.0 - inning_w
            self.log(f"Close Game Adjustment (Diff: {run_differential}): Weights -> Main={main_w:.3f}, Inning={inning_w:.3f}", level="DEBUG")

        # Momentum adjustment (MLB specific, e.g., recent scoring streak, pitching changes)
        # This is a placeholder; actual MLB momentum metric would be complex.
        momentum_threshold_mlb = 0.2 # Example threshold for a normalized MLB momentum score
        if abs(momentum) > momentum_threshold_mlb:
            momentum_adjust = 0.05 * (abs(momentum) - momentum_threshold_mlb) / (1.0 - momentum_threshold_mlb + 1e-6)
            inning_w = min(self.HEURISTIC_INNING_WEIGHT_MAX, inning_w + momentum_adjust)
            main_w = 1.0 - inning_w
            self.log(f"Momentum Adjustment (Momentum: {momentum:.2f}): Weights -> Main={main_w:.3f}, Inning={inning_w:.3f}", level="DEBUG")

        # Prediction gap adjustment (for home team predictions, as an example)
        # If main model and inning-based model diverge significantly, trust main model more (often more stable).
        prediction_gap_home = abs(main_pred_home - inning_model_total_pred_home)
        gap_threshold_runs = 3.0 # e.g., 3 runs difference
        if prediction_gap_home > gap_threshold_runs:
            gap_adjust = min(0.2, 0.05 * (prediction_gap_home - gap_threshold_runs)) # Max 0.2 weight shift
            main_w = min(self.HEURISTIC_MAIN_WEIGHT_MAX, main_w + gap_adjust)
            inning_w = 1.0 - main_w
            self.log(f"Prediction Gap Adj. (Home Gap: {prediction_gap_home:.1f}): Weights -> Main={main_w:.3f}, Inning={inning_w:.3f}", level="DEBUG")

        # Ensure weights are capped and sum to 1
        main_w = np.clip(main_w, 1.0 - self.HEURISTIC_INNING_WEIGHT_MAX, self.HEURISTIC_MAIN_WEIGHT_MAX)
        inning_w = 1.0 - main_w
        self.log(f"Final Adjusted Weights for I{inning_for_weights}: Main={main_w:.3f}, Inning={inning_w:.3f}", level="DEBUG")

        final_ensemble_home = main_w * main_pred_home + inning_w * inning_model_total_pred_home
        final_ensemble_away = main_w * main_pred_away + inning_w * inning_model_total_pred_away
        
        return final_ensemble_home, final_ensemble_away, main_w, inning_w

    def track_weight_usage(self, game_id: str, inning: int, main_weight: float, inning_weight: float,
                             prediction_error_home: Optional[float] = None, # Track error for home/away separately if needed
                             prediction_error_away: Optional[float] = None) -> None:
        """Tracks weight usage and associated prediction error for analysis in MLB."""
        record = {
            'game_id': str(game_id),
            'inning': inning, # Inning context of the prediction
            'main_weight': main_weight,
            'inning_weight': inning_weight,
            'timestamp': pd.Timestamp.now(),
            'prediction_error_home': prediction_error_home,
            'prediction_error_away': prediction_error_away,
        }
        self.weight_history.append(record)
        self.log(f"Tracked weights for game {game_id}, I{inning} (Errors H/A: {prediction_error_home}/{prediction_error_away}).", level="DEBUG")

    def update_error_history_from_validation(self, validation_summary: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        Updates error history based on an MLB validation summary DataFrame.
        The DataFrame should include columns: 'inning', 'avg_abs_main_error', 'avg_abs_inning_error'.
        """
        self.log("Updating MLB error history from validation summary...", level="INFO")
        updated = False
        required_cols = ['inning', 'avg_abs_main_error', 'avg_abs_inning_error']
        if not all(col in validation_summary.columns for col in required_cols):
            self.log(f"Validation summary missing one or more required columns: {required_cols}", level="ERROR")
            return self.error_history

        for _, row in validation_summary.iterrows():
            try:
                inn = int(row['inning'])
                if not (1 <= inn <= NUM_INNINGS_REGULATION): continue

                avg_main_err = row.get('avg_abs_main_error')
                avg_inning_err = row.get('avg_abs_inning_error') # Changed from quarter_error

                current_main_errors = self.error_history.get('main_model', {})
                current_inning_errors = self.error_history.get('inning_model', {})

                if inn not in current_main_errors or inn not in current_inning_errors:
                    self.log(f"Inning {inn} missing in error history template. Skipping update for this inning.", level="WARNING")
                    continue

                if pd.notna(avg_main_err):
                    curr_main = current_main_errors[inn]
                    new_main = self.HISTORIC_WEIGHT_SMOOTHING * avg_main_err + (1 - self.HISTORIC_WEIGHT_SMOOTHING) * curr_main
                    current_main_errors[inn] = new_main
                    self.log(f"Updated main_model error for I{inn}: {curr_main:.2f} -> {new_main:.2f}", level="DEBUG")
                    updated = True
                
                if pd.notna(avg_inning_err):
                    curr_inning = current_inning_errors[inn] # Changed from quarter
                    new_inning = self.HISTORIC_WEIGHT_SMOOTHING * avg_inning_err + (1 - self.HISTORIC_WEIGHT_SMOOTHING) * curr_inning
                    current_inning_errors[inn] = new_inning
                    self.log(f"Updated inning_model error for I{inn}: {curr_inning:.2f} -> {new_inning:.2f}", level="DEBUG")
                    updated = True
            except Exception as e:
                self.log(f"Error processing validation row: {row}. Error: {e}", level="ERROR", exc_info=True)
        
        if updated:
            self.log("MLB Error history updated from validation summary.", level="INFO")
        else:
            self.log("No updates applied to MLB error history from validation summary.", level="DEBUG")
        return self.error_history

# -------------------- generate_enhanced_predictions (for MLB) --------------------
def generate_enhanced_predictions(
    live_games_df: pd.DataFrame, # DataFrame of current live games basic info
    model_payload_path: str,     # Path to the saved main model (e.g., XGBoostScorePredictor)
    feature_generator: Any,      # MLBFeatureGenerator instance
    inning_system: Any,          # InningSpecificModelSystem instance (from mlb_models.py)
    confidence_viz: Optional[Any] = None, # Visualization tool instance
    historical_games_df: Optional[pd.DataFrame] = None, # For feature generation
    team_stats_df: Optional[pd.DataFrame] = None,       # For feature generation
    debug: bool = False
) -> pd.DataFrame:
    """
    Generates enhanced predictions (home & away runs) for live MLB games.
    Combines a main model with inning-specific predictions using ensemble weighting.
    Optionally adds uncertainty bounds and confidence visualizations.
    """
    logger.info(f"Starting enhanced MLB predictions for {len(live_games_df)} games.")
    if live_games_df.empty:
        logger.warning("No live games data provided to generate_enhanced_predictions.")
        return pd.DataFrame()
        
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model'] # This is an instance of BaseScorePredictor (e.g. XGBoostScorePredictor)
        required_features = model_payload['features'] # List of feature names
        logger.info(f"Loaded main MLB model payload with {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        logger.error(f"MLB Model payload file not found: {model_payload_path}.")
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"MLB Model payload missing key: {e}. Requires 'model' and 'features'.")
        return pd.DataFrame()
    except Exception as e:
        logger.exception(f"Failed to load MLB model payload from {model_payload_path}: {e}")
        return pd.DataFrame()

    try:
        # Generate features for all live games
        features_df = feature_generator.generate_features_for_live_games( # Assuming this method name
            live_games_df,
            historical_games_df=historical_games_df, # Pass historical data if needed by generator
            team_stats_df=team_stats_df             # Pass team stats if needed
        )
        if features_df.empty or features_df.shape[0] != live_games_df.shape[0]:
            logger.error(f"MLB feature generation returned unexpected shape. Input: {live_games_df.shape}, Output: {features_df.shape}")
            return pd.DataFrame()
        logger.info(f"MLB feature generation complete. Shape: {features_df.shape}")
    except Exception as e:
        logger.exception(f"Exception during MLB feature generation: {e}")
        return pd.DataFrame()

    # Align features for the main model
    X_main = pd.DataFrame(index=features_df.index)
    for f in required_features:
        if f not in features_df.columns:
            X_main[f] = 0  # Fill missing with default 0, log warning
            logger.warning(f"Missing required feature '{f}' for main model, added with default 0.")
        else:
            X_main[f] = features_df[f]
    try:
        # Ensure correct order and handle any remaining NaNs after potential fill_value
        X_main = X_main.reindex(columns=required_features, fill_value=0) 
        if X_main.isnull().any().any():
            logger.warning("Null values detected in X_main after reindexing and fill_value. Imputing with 0.")
            X_main = X_main.fillna(0) # Final safety net for NaNs
    except Exception as e:
        logger.exception(f"Error selecting/reindexing required features for main model: {e}")
        return pd.DataFrame()

    # Get main model predictions (home_runs, away_runs)
    main_preds_df: Optional[pd.DataFrame] = None
    try:
        main_preds_df = main_model.predict(X_main) # Should return df with 'predicted_home_runs', 'predicted_away_runs'
        if main_preds_df is None or 'predicted_home_runs' not in main_preds_df.columns or 'predicted_away_runs' not in main_preds_df.columns:
            raise ValueError("Main model prediction did not return expected DataFrame with home/away runs.")
        logger.info(f"Main MLB model predictions generated for {len(main_preds_df)} samples.")
    except Exception as e:
        logger.exception(f"Error during main MLB model prediction: {e}")
        # Fallback: Use league average runs if feature_generator provides it
        avg_runs = feature_generator.league_averages.get('runs_per_game_per_team', 4.5) if hasattr(feature_generator, 'league_averages') else 4.5
        main_preds_df = pd.DataFrame({
            'predicted_home_runs': np.full(len(features_df), avg_runs),
            'predicted_away_runs': np.full(len(features_df), avg_runs)
        }, index=features_df.index)
        logger.info(f"Using fallback MLB predictions with average runs {avg_runs}.")

    results = []
    # Ensure features_df and main_preds_df can be merged or iterated together
    # Typically, they would share an index from live_games_df
    # If main_preds_df lost index, we assume row-order alignment.
    if features_df.index.name != main_preds_df.index.name and len(features_df) == len(main_preds_df):
        main_preds_df.index = features_df.index # Align indices if they differ but lengths match

    # Combine features with main predictions for iteration
    combined_df_for_iteration = features_df.copy()
    # Ensure correct merging of main predictions. If indices are aligned, direct assignment is fine.
    # If main_preds_df might not have all indices of features_df (e.g. if some games failed prediction)
    # use join or merge. For simplicity, assuming indices are compatible or lengths match.
    combined_df_for_iteration['main_pred_home_runs'] = main_preds_df['predicted_home_runs']
    combined_df_for_iteration['main_pred_away_runs'] = main_preds_df['predicted_away_runs']
    
    for i, game_row_series in combined_df_for_iteration.iterrows():
        game_id_log = game_row_series.get('game_id', f'index_{i}')
        try:
            game_data_dict = game_row_series.to_dict() # Contains all features + main model predictions
            main_pred_h = float(game_data_dict['main_pred_home_runs'])
            main_pred_a = float(game_data_dict['main_pred_away_runs'])

            # Call the MLB InningSpecificModelSystem's prediction method
            # It now takes both home & away main model predictions
            # Its weight_manager uses calculate_ensemble_mlb
            (ensemble_pred_h, ensemble_pred_a, 
             confidence, # Confidence from inning_system is more of a structural confidence/weight blend
             breakdown) = inning_system.predict_final_score(
                game_data=game_data_dict, # Pass the full feature set
                main_model_pred_home=main_pred_h,
                main_model_pred_away=main_pred_a,
                weight_manager=getattr(inning_system, "weight_manager", None)  # Safe lookup
            )
            
            current_inn_completed = int(game_data_dict.get('current_inning_completed', 0))
            live_home_runs = float(game_data_dict.get('home_score', 0)) # Actual current score from data
            live_away_runs = float(game_data_dict.get('away_score', 0))

            # Uncertainty and Visualization - focusing on home team prediction or total for simplicity first
            # This part might need more significant adaptation for dual (home/away) predictions
            final_pred_for_unc = ensemble_pred_h # Example: use home team's prediction for uncertainty

            svg_indicator = "<svg>N/A</svg>"
            if confidence_viz and hasattr(confidence_viz, 'create_confidence_svg_mlb'):
                svg_indicator = confidence_viz.create_confidence_svg_mlb(
                    prediction=float(final_pred_for_unc),
                    lower_bound=float(final_pred_for_unc) - 1,
                    upper_bound=float(final_pred_for_unc) + 1,
                    current_inning=current_inn_completed,
                    current_score_metric=live_home_runs
                )


            result_row = {
                'game_id': game_data_dict.get('game_id'),
                'home_team_name': game_data_dict.get('home_team_name', game_data_dict.get('home_team_id')), # Use names if available
                'away_team_name': game_data_dict.get('away_team_name', game_data_dict.get('away_team_id')),
                'game_date': game_data_dict.get('game_date_et', game_data_dict.get('game_date')), # Use ET if available
                'current_inning_completed': current_inn_completed,
                'live_home_runs': live_home_runs,
                'live_away_runs': live_away_runs,
                'main_model_pred_home': main_pred_h,
                'main_model_pred_away': main_pred_a,
                'inning_model_sum_home': breakdown.get('inning_model_sum_home'),
                'inning_model_sum_away': breakdown.get('inning_model_sum_away'),
                'ensemble_pred_home': float(ensemble_pred_h),
                'ensemble_pred_away': float(ensemble_pred_a),
                'confidence_svg': svg_indicator,
                'main_weight': breakdown.get('weights', {}).get('main_model', breakdown.get('weights', {}).get('main')),
                'inning_weight': breakdown.get('weights', {}).get('inning_model', breakdown.get('weights', {}).get('inning'))
            }
            results.append(result_row)
        except Exception as e:
            logger.exception(f"Error processing MLB game {game_id_log} for enhanced prediction: {e}")
            results.append({
                'game_id': game_id_log,
                'home_team_name': game_row_series.get('home_team_name', game_row_series.get('home_team_id')),
                'away_team_name': game_row_series.get('away_team_name', game_row_series.get('away_team_id')),
                'game_date': game_row_series.get('game_date_et', game_row_series.get('game_date')),
                'error': str(e),
                # Add other minimal fields for error rows
                'current_inning_completed': game_row_series.get('current_inning_completed', 0),
                'live_home_runs': game_row_series.get('home_score', 0),
                'live_away_runs': game_row_series.get('away_score', 0),
                'ensemble_pred_home': np.nan, 'ensemble_pred_away': np.nan,
            })

    return pd.DataFrame(results)