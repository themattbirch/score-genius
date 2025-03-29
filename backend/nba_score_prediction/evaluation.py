# backend/nba_score_prediction/evaluation.py

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
import joblib
from scipy import stats
from sklearn.metrics import r2_score


# Import the QuarterSpecificModelSystem class from the proper location
from nba_score_prediction.models import QuarterSpecificModelSystem

# -------------------- PredictionUncertaintyEstimator Class --------------------


class PredictionUncertaintyEstimator:
    """
    Estimates uncertainty (confidence intervals) for NBA score predictions
    based on historical error patterns and game context.
    """
    def __init__(self, debug=False):
        """Initialize the uncertainty estimator with default values."""
        self.debug = debug
        
        # Default mean absolute error by quarter (improves as game progresses)
        self.mae_by_quarter = {0: 8.5, 1: 7.0, 2: 6.0, 3: 5.0, 4: 0.0}
        
        # Default standard deviation by quarter
        self.std_by_quarter = {0: 4.5, 1: 3.8, 2: 3.2, 3: 2.6, 4: 0.0}
        
        # Adjustment factors for different game situations
        self.margin_adjustments = {'close': 1.2, 'moderate': 1.0, 'blowout': 0.8}
        self.momentum_effects = {'high': 1.2, 'moderate': 1.0, 'low': 0.8}
        
        # Storage for historical error tracking
        self.historical_errors = {q: [] for q in range(5)}
        self.interval_coverage = {q: {'inside': 0, 'total': 0} for q in range(5)}
    
    def _print_debug(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
             print(f"[{type(self).__name__}] {message}")

    def calculate_prediction_interval(
        self, 
        prediction: float, 
        current_quarter: int,
        score_margin: Optional[float] = None, 
        momentum: Optional[float] = None, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate a prediction interval based on quarter and game context.
        
        Args:
            prediction: The point prediction value
            current_quarter: Current quarter of the game (0-4)
            score_margin: Current score margin between teams
            momentum: Current momentum metric (positive or negative)
            confidence_level: Desired confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound, interval_width)
        """
        # Get base error metrics for this quarter
        mae = self.mae_by_quarter.get(current_quarter, 8.0)
        std = self.std_by_quarter.get(current_quarter, 4.0)
        
        # Adjust for score margin (close games are harder to predict)
        if score_margin is not None:
            if score_margin < 5:
                margin_factor = self.margin_adjustments['close']
            elif score_margin > 15:
                margin_factor = self.margin_adjustments['blowout']
            else:
                margin_factor = self.margin_adjustments['moderate']
            mae *= margin_factor
            std *= margin_factor
        
        # Adjust for momentum (high momentum can affect predictability)
        if momentum is not None:
            abs_momentum = abs(momentum)
            if abs_momentum > 0.6:
                momentum_factor = self.momentum_effects['high']
            elif abs_momentum > 0.3:
                momentum_factor = self.momentum_effects['moderate']
            else:
                momentum_factor = self.momentum_effects['low']
            mae *= momentum_factor
            std *= momentum_factor
        
        # Calculate z-score for desired confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate interval half-width based on combined error
        interval_half_width = z_score * np.sqrt(mae**2 + std**2)
        
        # Narrow interval as game progresses
        narrowing_factor = 1.0 - (current_quarter * 0.15)
        interval_half_width *= max(0.4, narrowing_factor)
        
        # Calculate bounds
        lower_bound = max(0, prediction - interval_half_width)  # Scores can't be negative
        upper_bound = prediction + interval_half_width
        
        return lower_bound, upper_bound, interval_half_width * 2

    def update_error_metrics(self, errors_by_quarter: Dict[int, List[float]]) -> None:
        """
        Update error metrics based on new observations.
        
        Args:
            errors_by_quarter: Dictionary mapping quarters to lists of prediction errors
        """
        for quarter, errors in errors_by_quarter.items():
            if errors:
                # Add new errors to history (keeping last 100 for each quarter)
                self.historical_errors[quarter].extend(errors)
                self.historical_errors[quarter] = self.historical_errors[quarter][-100:]
                
                # Recalculate MAE and STD
                mae = np.mean(np.abs(self.historical_errors[quarter]))
                std = np.std(self.historical_errors[quarter])
                
                # Update metrics
                self.mae_by_quarter[quarter] = mae
                self.std_by_quarter[quarter] = std
                
                self._print_debug(f"Updated Q{quarter} metrics: MAE={mae:.2f}, STD={std:.2f}")

    def record_interval_coverage(self, quarter: int, lower: float, upper: float, actual: float) -> None:
        """
        Record whether an actual value fell within a prediction interval.
        
        Args:
            quarter: Game quarter
            lower: Lower bound of interval
            upper: Upper bound of interval
            actual: Actual observed value
        """
        self.interval_coverage[quarter]['total'] += 1
        if lower <= actual <= upper:
            self.interval_coverage[quarter]['inside'] += 1
            
        if quarter in [0, 1, 2, 3, 4]:  # Valid quarters only
            coverage = self.interval_coverage[quarter]['inside'] / self.interval_coverage[quarter]['total'] * 100
            self._print_debug(f"Q{quarter} interval coverage: {coverage:.1f}% ({self.interval_coverage[quarter]['inside']}/{self.interval_coverage[quarter]['total']})")

    def get_coverage_stats(self) -> pd.DataFrame:
      """
      Get statistics on prediction interval coverage.

      Returns:
          DataFrame with coverage statistics by quarter
      """
      stats_list = []
      for quarter, data in self.interval_coverage.items():
          if data['total'] > 0:
              coverage_pct = (data['inside'] / data['total']) * 100
              stats_list.append({
                  'quarter': quarter,
                  'sample_size': data['total'],
                  'covered': data['inside'],
                  'coverage_pct': coverage_pct
              })
      return pd.DataFrame(stats_list)


    def generate_confidence_svg(self, prediction, lower_bound, upper_bound, current_quarter,
                                home_score=None, svg_min=0, svg_max=150, width=300, height=60,
                                expected_width=None) -> str:
        """
        Generate SVG visualization for prediction confidence.

        Args:
            prediction: Predicted score
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            current_quarter: Current game quarter
            home_score: (Optional) Actual current score to mark on the graph
            svg_min: Minimum value for score axis
            svg_max: Maximum value for score axis
            width: SVG width
            height: SVG height
            expected_width: Optional dict of expected widths per quarter

        Returns:
            str: SVG markup
        """
        score_range = svg_max - svg_min

        def to_svg_x(score):
            return (score - svg_min) / score_range * width

        pred_x = to_svg_x(prediction)
        lower_x = to_svg_x(lower_bound)
        upper_x = to_svg_x(upper_bound)

        quarter_colors = {
            0: "#d3d3d3",
            1: "#ffa07a",
            2: "#ff7f50",
            3: "#ff4500",
            4: "#8b0000"
        }
        color = quarter_colors.get(current_quarter, "#000000")

        interval_width = upper_bound - lower_bound
        expected = expected_width.get(current_quarter, 25) if expected_width else 25
        confidence = max(0, min(100, 100 - (interval_width / expected * 100)))

        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa" rx="5" ry="5" />
        <text x="10" y="15" font-family="Arial" font-size="12" fill="#555">Prediction Confidence (Q{current_quarter})</text>
        <rect x="{lower_x}" y="{height/2 - 5}" width="{upper_x - lower_x}" height="10" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="1" rx="2" ry="2" />
        <circle cx="{pred_x}" cy="{height/2}" r="6" fill="{color}" />
        <text x="{pred_x}" y="{height/2 - 10}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333" font-weight="bold">{prediction:.1f}</text>
        <text x="{lower_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{lower_bound:.1f}</text>
        <text x="{upper_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{upper_bound:.1f}</text>
        <text x="{width - 10}" y="{height - 10}" text-anchor="end" font-family="Arial" font-size="12" fill="#333" font-weight="bold">{confidence:.0f}% confidence</text>
        </svg>"""

        if home_score is not None and home_score > 0:
            score_x = to_svg_x(home_score)
            svg = svg.replace('</svg>', f"""  <!-- Current Score -->
        <circle cx="{score_x}" cy="{height/2}" r="5" fill="none" stroke="#28a745" stroke-width="2" />
        <text x="{score_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#28a745" font-weight="bold">{home_score}</text>
        </svg>""")

        return svg
      
    def dynamically_adjust_interval(
        self, 
        prediction: float, 
        current_quarter: int, 
        historic_accuracy: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Dynamically adjust prediction interval based on historical accuracy.
        
        Args:
            prediction: Predicted score
            current_quarter: Current game quarter (0-4)
            historic_accuracy: Dictionary with historical coverage statistics
            
        Returns:
            Tuple of (adjusted_lower_bound, adjusted_upper_bound, confidence_percentage)
        """
        # Get base prediction interval
        lower, upper, width = self.calculate_prediction_interval(prediction, current_quarter)
        
        # Exit early if no historical data available
        if not historic_accuracy or current_quarter not in historic_accuracy:
            expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}.get(current_quarter, 25)
            confidence = max(0, min(100, 100 - (width / expected_width * 100)))
            return lower, upper, confidence
        
        # Get historical coverage for this quarter
        accuracy = historic_accuracy[current_quarter]
        
        # Widen interval if coverage is too low
        if accuracy.get('coverage_pct', 95) < 90:
            # Calculate widening factor based on how far below target
            widening_factor = (95 - accuracy.get('coverage_pct', 95)) / 50
            width *= (1.0 + widening_factor)
            self._print_debug(f"Widening Q{current_quarter} interval by factor {1.0 + widening_factor:.2f} due to low coverage ({accuracy.get('coverage_pct', 95):.1f}%)")
            
        # Narrow interval if coverage is too high
        elif accuracy.get('coverage_pct', 95) > 98:
            # Calculate narrowing factor based on how far above target
            narrowing_factor = (accuracy.get('coverage_pct', 95) - 95) / 300
            width *= (1.0 - narrowing_factor)
            self._print_debug(f"Narrowing Q{current_quarter} interval by factor {1.0 - narrowing_factor:.2f} due to high coverage ({accuracy.get('coverage_pct', 95):.1f}%)")
        
        # Recalculate bounds with adjusted width
        lower = max(0, prediction - width/2)
        upper = prediction + width/2
        
        # Calculate confidence percentage
        expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}.get(current_quarter, 25)
        confidence = max(0, min(100, 100 - (width / expected_width * 100)))
        
        return lower, upper, confidence

# -------------------- Validation Framework --------------------
def validate_enhanced_predictions(
    model_payload_path: str,  # Path to the saved model payload
    feature_generator: Any,  # Instance of NBAFeatureEngine
    historical_df: Optional[pd.DataFrame] = None,  # Source of historical games for testing
    num_test_games: int = 20,
    debug: bool = False,
    # Add other necessary components if needed by feature_generator or quarter_system
    quarter_system: Optional[Any] = None,  # Pass initialized QuarterSystem if needed
    # Supabase/DB dependencies removed assuming historical_df is provided or loaded externally
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate enhanced prediction system on historical games using a loaded model payload.

    Args:
        model_payload_path: Path to the file containing the model and required features.
        feature_generator: Instance of NBAFeatureEngine.
        historical_df: DataFrame with historical games (MUST be provided for validation).
        num_test_games: Number of historical games to test from the provided df.
        debug: Whether to print debug messages.
        quarter_system: Pre-initialized instance of QuarterSpecificModelSystem.

    Returns:
        tuple: (DataFrame with detailed validation results per game/quarter,
                DataFrame with aggregated improvement metrics by quarter)
    """
    # --- Load Model and Required Features ---
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model']
        required_features = model_payload['features']
        if debug:
            print(f"[validate_enhanced_predictions] Loaded model and {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        print(f"ERROR: Model payload file not found at {model_payload_path}. Cannot run validation.")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames on critical error
    except KeyError as e:
        print(f"ERROR: Model payload at {model_payload_path} is missing key: {e}. Requires 'model' and 'features'.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to load model payload from {model_payload_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    # --- Input Validation and Test Data Selection ---
    if historical_df is None or historical_df.empty:
        print("ERROR: historical_df must be provided for validation.")
        return pd.DataFrame(), pd.DataFrame()

    if 'game_date' not in historical_df.columns:
        print("ERROR: historical_df missing 'game_date' column.")
        return pd.DataFrame(), pd.DataFrame()
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])

    # Use most recent games from provided DataFrame
    test_games = historical_df.sort_values('game_date', ascending=False).head(num_test_games)
    if debug:
        print(f"Selected {len(test_games)} most recent games for validation.")

    if len(test_games) == 0:
        print("Warning: No test games selected.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Initialize Quarter System (if not passed) ---
    if quarter_system is None:
        if debug:
            print("Initializing QuarterSpecificModelSystem for validation...")
        quarter_system = QuarterSpecificModelSystem(feature_generator, debug=debug)
        quarter_system.load_models()  # Load quarter-specific models

    # Check if quarter_system requires weight_manager and if it's present
    if not hasattr(quarter_system, 'weight_manager'):
        print("ERROR: QuarterSpecificModelSystem instance does not have 'weight_manager'. Ensemble calculation might fail.")
        # Depending on implementation, you might need to initialize it here or ensure it's part of QuarterSystem init
        # Example: from backend.score_prediction.feature_engineering import EnsembleWeightManager
        # quarter_system.weight_manager = EnsembleWeightManager(debug=debug)

    # --- Process Each Test Game ---
    validation_results = []
    required_base_cols = ['game_id', 'home_team', 'away_team', 'home_score', 'away_score']  # Check these exist in test_games
    if not all(col in test_games.columns for col in required_base_cols):
        print(f"ERROR: Test games missing one or more required columns: {required_base_cols}")
        return pd.DataFrame(), pd.DataFrame()

    for _, game in test_games.iterrows():
        actual_home_score = float(game['home_score'])
        game_id_log = game['game_id']

        # Test predictions for each quarter state (0 = pregame, 1-4 = end of quarter)
        for test_quarter in range(0, 5):
            if debug and test_quarter == 0:
                print(f"\n--- Testing Game: {game_id_log} ({game['home_team']} vs {game['away_team']}) ---")
            if debug:
                print(f"Simulating state at end of Q{test_quarter}...")

            # Create a simulated game state DataFrame (single row)
            sim_data = {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game['game_date'],  # Already datetime
                'current_quarter': test_quarter,
                # Include other base columns if feature generator needs them
            }

            # Add known quarter scores up to the simulated point
            current_home_q_score = 0
            current_away_q_score = 0
            for q in range(1, test_quarter + 1):
                home_q_col = f'home_q{q}'
                away_q_col = f'away_q{q}'
                sim_data[home_q_col] = game.get(home_q_col, 0)
                sim_data[away_q_col] = game.get(away_q_col, 0)
                current_home_q_score += sim_data[home_q_col]
                current_away_q_score += sim_data[away_q_col]

            # Add live score based on simulated state (needed for some features/context)
            sim_data['home_score'] = current_home_q_score
            sim_data['away_score'] = current_away_q_score
            # sim_data['score_differential'] = current_home_q_score - current_away_q_score  # This will be recalculated by feature eng

            sim_game_df = pd.DataFrame([sim_data])

            # Generate features for this simulated state
            try:
                # Use the wrapper; pass historical_df for context like rest days
                features_df = feature_generator.generate_all_features(
                    sim_game_df,
                    historical_games_df=historical_df  # Pass full history for context
                    # team_stats_df can be added if needed
                )

                if features_df.empty:
                    raise ValueError("Feature generation returned empty DataFrame.")

                # Select features for main model using loaded required_features
                missing_features_check = []
                for f in required_features:
                    if f not in features_df.columns:
                        features_df[f] = 0  # Default fill
                        missing_features_check.append(f)
                if missing_features_check and debug:
                    print(f"  Q{test_quarter}: Warning - Added {len(missing_features_check)} missing main model features: {missing_features_check[:5]}...")

                # Select features IN ORDER
                X_main_sim = features_df.reindex(columns=required_features, fill_value=0)

                # Get main model prediction
                main_pred = float(main_model.predict(X_main_sim)[0])

                # Get ensemble prediction using quarter system
                # Pass the feature-rich row as a dict
                ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
                    game_data_dict=features_df.iloc[0].to_dict(),
                    main_model_prediction=main_pred,
                    weight_manager=quarter_system.weight_manager  # Pass weight manager
                )
                ensemble_pred = float(ensemble_pred)  # Ensure float

                # Calculate prediction errors
                main_error = main_pred - actual_home_score
                ensemble_error = ensemble_pred - actual_home_score

                # Record results (include individual model errors for later analysis if needed)
                # Also include quarter_model_sum_pred from breakdown
                validation_results.append({
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_date': game['game_date'],
                    'current_quarter': test_quarter,  # State being simulated (0-4)
                    'actual_home_score': actual_home_score,
                    'main_prediction': main_pred,
                    'quarter_model_sum_prediction': breakdown.get('quarter_model_sum'),  # Store this
                    'ensemble_prediction': ensemble_pred,
                    'main_error': main_error,
                    'quarter_model_sum_error': breakdown.get('quarter_model_sum', np.nan) - actual_home_score,  # Store this error too!
                    'ensemble_error': ensemble_error,
                    'main_abs_error': abs(main_error),
                    'ensemble_abs_error': abs(ensemble_error),
                    'main_squared_error': main_error**2,
                    'ensemble_squared_error': ensemble_error**2,
                    'confidence': confidence,  # From predict_final_score
                    'main_weight': breakdown.get('weights', {}).get('main'),
                    'quarter_weight': breakdown.get('weights', {}).get('quarter')
                })
                if debug:
                    print(f"  Q{test_quarter}: Main={main_pred:.1f}, Ens={ensemble_pred:.1f} (Actual={actual_home_score}) -> MAE Ens={abs(ensemble_error):.1f}")

            except Exception as e:
                if debug:
                    print(f"  Q{test_quarter}: Error processing game {game_id_log}: {e}")
                    # Optionally print traceback for detailed debugging:
                    # traceback.print_exc()
                # Record error state
                validation_results.append({
                    'game_id': game['game_id'], 'home_team': game['home_team'], 'away_team': game['away_team'],
                    'game_date': game['game_date'], 'current_quarter': test_quarter,
                    'actual_home_score': actual_home_score, 'main_prediction': np.nan,
                    'quarter_model_sum_prediction': np.nan, 'ensemble_prediction': np.nan,
                    'main_error': np.nan, 'quarter_model_sum_error': np.nan, 'ensemble_error': np.nan,
                    'main_abs_error': np.nan, 'ensemble_abs_error': np.nan,
                    'main_squared_error': np.nan, 'ensemble_squared_error': np.nan,
                    'confidence': np.nan, 'main_weight': np.nan, 'quarter_weight': np.nan,
                    'error_flag': 1  # Add flag to indicate error
                })

    # --- Aggregate and Return Results ---
    validation_df = pd.DataFrame(validation_results)

    if validation_df.empty:
        if debug:
            print("No validation results generated.")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate aggregated error metrics by quarter
    # Use .agg named aggregation for cleaner output columns in Pandas >= 0.25
    error_metrics = validation_df.groupby('current_quarter').agg(
        count=('game_id', 'size'),
        main_mae=('main_abs_error', 'mean'),
        main_mae_std=('main_abs_error', 'std'),
        main_rmse=('main_squared_error', lambda x: np.sqrt(x.mean())),  # Calculate RMSE from mean squared error
        qsum_mae=('quarter_model_sum_error', lambda x: np.abs(x).mean()),  # Add metrics for quarter sum model
        qsum_rmse=('quarter_model_sum_error', lambda x: np.sqrt((x**2).mean())),
        ensemble_mae=('ensemble_abs_error', 'mean'),
        ensemble_mae_std=('ensemble_abs_error', 'std'),
        ensemble_rmse=('ensemble_squared_error', lambda x: np.sqrt(x.mean()))
    ).reset_index()

    if debug:
        print("\n--- Validation Error Metrics by Quarter ---")
        print(error_metrics.round(2))

    # Calculate improvement metrics and R²
    improvements = []
    # Need r2_score if calculating R²
    try:
        from sklearn.metrics import r2_score
    except ImportError:
        r2_score = None
        print("Warning: sklearn.metrics not found, cannot calculate R2.")

    for quarter in range(0, 5):
        quarter_data = validation_df[(validation_df['current_quarter'] == quarter) & validation_df['ensemble_prediction'].notna()].copy()
        if not quarter_data.empty:
            metrics = {
                'quarter': quarter,
                'sample_size': len(quarter_data)
            }

            # Calculate Mean Errors (MAE, RMSE) directly from aggregated metrics if preferred
            agg_row = error_metrics[error_metrics['current_quarter'] == quarter].iloc[0]
            metrics.update({
                'main_mae': agg_row['main_mae'], 'ensemble_mae': agg_row['ensemble_mae'],
                'main_rmse': agg_row['main_rmse'], 'ensemble_rmse': agg_row['ensemble_rmse'],
            })

            # Calculate % Improvements
            metrics['mae_improvement_pct'] = ((metrics['main_mae'] - metrics['ensemble_mae']) / metrics['main_mae'] * 100) if metrics['main_mae'] else 0
            metrics['rmse_improvement_pct'] = ((metrics['main_rmse'] - metrics['ensemble_rmse']) / metrics['main_rmse'] * 100) if metrics['main_rmse'] else 0

            # Calculate R² if possible
            if r2_score:
                y_true = quarter_data['actual_home_score']
                metrics['main_r2'] = r2_score(y_true, quarter_data['main_prediction'])
                metrics['ensemble_r2'] = r2_score(y_true, quarter_data['ensemble_prediction'])
                metrics['r2_improvement'] = metrics['ensemble_r2'] - metrics['main_r2']
            else:
                metrics['main_r2'], metrics['ensemble_r2'], metrics['r2_improvement'] = np.nan, np.nan, np.nan

            improvements.append(metrics)

    improvement_df = pd.DataFrame(improvements)

    if debug:
        print("\n--- Validation Improvement by Quarter ---")
        print(improvement_df.round(2))

    # Return both detailed results and aggregated improvements
    # The detailed validation_df now includes 'quarter_model_sum_prediction' and 'quarter_model_sum_error'
    # which can be used to update the EnsembleWeightManager's error history correctly.
    return validation_df, improvement_df


def get_recommended_model_params(quarter, model_type=None):
    """
    Returns optimized hyperparameters for specific quarter models.
    
    Args:
        quarter: Quarter number (1-4)
        model_type: Model type to override default recommendation (RandomForest, XGBoost, Ridge)
        
    Returns:
        dict: Hyperparameters for the recommended model type
    """
    # Return hyperparameters for specific model type if requested
    if model_type == "RandomForest":
        return {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        }
    elif model_type == "XGBoost":
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    elif model_type == "Ridge":
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
    
    # Return quarter-specific optimal model configuration
    if quarter == 1:
        # For Q1, use XGBoost with parameters optimized for early game prediction
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 3,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    elif quarter == 2:
        # For Q2, use XGBoost with parameters that incorporate Q1 information
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    elif quarter == 3:
        # For Q3, use XGBoost with parameters that balance complexity and robustness
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 4,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    else:  # quarter == 4 or any other value
        # For Q4, use Ridge regression for stability and robustness
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
