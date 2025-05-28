# backend/nba_score_prediction/simulation.py

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
from functools import wraps
import scipy.stats as scipy_stats 
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Union, Any
import logging # Added logging

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ PREDICTION UNCERTAINTY ESTIMATOR DEFINITION ADDED HERE +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictionUncertaintyEstimator:
    """
    Estimates prediction intervals and confidence for NBA score predictions based
    on game state and historical performance.
    """

    # --- Default Expected Ranges (can be overridden or tuned) ---
    # Represents typical +/- error expected around the prediction per quarter
    # Based loosely on typical RMSE values seen during model development
    DEFAULT_EXPECTED_RANGES = {
        0: 18.0, # Pregame (wider) - Represents uncertainty over full game
        1: 15.0, # End of Q1
        2: 12.0, # End of Q2
        3: 9.0,  # End of Q3
        4: 7.0   # End of Q4 (narrower) - Represents uncertainty remaining
    }

    # --- Default Coverage Targets (can be overridden) ---
    DEFAULT_COVERAGE_TARGETS = {
        0: 0.70, # 70% coverage pregame
        1: 0.75, # 75% coverage Q1
        2: 0.80, # 80% coverage Q2
        3: 0.85, # 85% coverage Q3
        4: 0.90  # 90% coverage Q4
    }

    def __init__(self,
                 expected_ranges: Optional[Dict[int, float]] = None,
                 coverage_targets: Optional[Dict[int, float]] = None,
                 historical_coverage_stats: Optional[pd.DataFrame] = None,
                 debug: bool = False):
        """
        Initialize the uncertainty estimator.

        Args:
            expected_ranges: Dict mapping quarter (0-4) to expected prediction range (+/- points).
            coverage_targets: Dict mapping quarter (0-4) to target coverage percentage (0.0-1.0).
            historical_coverage_stats: DataFrame with past coverage performance, maybe columns like
                                       ['quarter', 'actual_coverage', 'avg_interval_width'].
                                       Must be set externally or loaded.
            debug: Enable debug logging.
        """
        self.expected_ranges = expected_ranges or self.DEFAULT_EXPECTED_RANGES
        self.coverage_targets = coverage_targets or self.DEFAULT_COVERAGE_TARGETS
        # Store historical stats if provided, can be updated later
        self.coverage_stats = historical_coverage_stats
        self.debug = debug
        self.log("PredictionUncertaintyEstimator initialized.", level="DEBUG")
        if self.coverage_stats is not None:
            self.log(f"Initialized with historical coverage stats for {len(self.coverage_stats)} quarters.", level="DEBUG")


    def log(self, message, level="INFO"):
        """Log messages based on debug flag."""
        # Use logger.<level> methods directly
        if self.debug or level.upper() != "DEBUG":
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(f"[UncertaintyEstimator] {message}")

    def set_historical_coverage(self, coverage_stats_df: pd.DataFrame):
        """Allows updating the historical coverage stats after initialization."""
        if isinstance(coverage_stats_df, pd.DataFrame) and not coverage_stats_df.empty:
             self.coverage_stats = coverage_stats_df
             self.log(f"Historical coverage stats updated for {len(self.coverage_stats)} quarters.", level="INFO")
        else:
             self.log("Attempted to update historical coverage with invalid data.", level="WARNING")


    def get_coverage_stats(self) -> Optional[pd.DataFrame]:
         """Returns the stored historical coverage statistics DataFrame."""
         if self.coverage_stats is not None and not self.coverage_stats.empty:
              self.log("Returning stored historical coverage stats.", level="DEBUG")
              # Return a copy to prevent external modification
              return self.coverage_stats.copy()
         else:
              self.log("No historical coverage stats available to return.", level="DEBUG")
              return None

    def _calculate_base_interval(self, prediction: float, quarter: int) -> Tuple[float, float]:
        """Calculates the base prediction interval based on expected ranges for the quarter."""
        # Ensure quarter is valid, default to pregame (0) if not
        if not isinstance(quarter, int) or quarter < 0 or quarter > 4:
             self.log(f"Invalid quarter ({quarter}) provided, using pregame (Q0) range.", level="WARNING")
             quarter = 0

        # Get the expected half-range (+/-) for this quarter
        half_range = self.expected_ranges.get(quarter, self.DEFAULT_EXPECTED_RANGES[0]) # Fallback to Q0 default

        lower = prediction - half_range
        upper = prediction + half_range
        self.log(f"Base interval for Q{quarter} (Range: +/- {half_range:.1f}): [{lower:.1f}, {upper:.1f}]", level="DEBUG")
        return lower, upper

    def calculate_prediction_interval(self,
                                      prediction: float,
                                      current_quarter: int,
                                      score_margin: Optional[float] = None, 
                                      momentum: Optional[float] = None 
                                      ) -> Tuple[float, float, float]:
        """
        Calculates a prediction interval, potentially adjusting the base width
        based on game context like score margin or momentum.

        Args:
            prediction: The point score prediction.
            current_quarter: The current quarter (0 for pregame, 1-4 during game).
            score_margin: Optional absolute score difference.
            momentum: Optional momentum metric.

        Returns:
            Tuple: (lower_bound, upper_bound, base_confidence_percentage)
        """
        # 1. Get Base Interval
        lower_b, upper_b = self._calculate_base_interval(prediction, current_quarter)
        base_width = upper_b - lower_b

        if base_width <= 0:
            self.log("Base interval width is zero or negative.", level="WARNING")
            return lower_b, upper_b, 50.0 

        # 2. Apply Contextual Adjustments (Optional)
        adjustment_factor = 1.0
        if current_quarter > 0: 
            # Example: Widen interval slightly for very close games
            if score_margin is not None and score_margin < 5:
                # Increase range up to 5% for closer games (0 diff = +5%, 5 diff = +0%)
                adjustment_factor += 0.05 * (1.0 - score_margin / 5.0)
                self.log(f"Applying close game adjustment (margin={score_margin:.1f}). Factor: {adjustment_factor:.3f}", level="DEBUG")

            # Example: Widen interval slightly for high momentum swings
            if momentum is not None and abs(momentum) > 0.6:
                 # Increase range up to 5% for high momentum (0.6 = +0%, 1.0 = +5%)
                 adj = 0.05 * min(1.0, (abs(momentum) - 0.6) / 0.4)
                 adjustment_factor += adj
                 self.log(f"Applying high momentum adjustment (mom={momentum:.2f}). Factor: {adjustment_factor:.3f}", level="DEBUG")

        # Apply the adjustment factor to the interval width
        adj_lower_b, adj_upper_b = lower_b, upper_b
        if adjustment_factor > 1.0: 
             center = (lower_b + upper_b) / 2.0
             new_half_range = (base_width / 2.0) * adjustment_factor
             adj_lower_b = center - new_half_range
             adj_upper_b = center + new_half_range
             self.log(f"Context adjusted interval: [{adj_lower_b:.1f}, {adj_upper_b:.1f}]", level="DEBUG")

        # 3. Calculate Base Confidence Score
        # Confidence inversely proportional to the interval width relative to expected default range.
        final_width = adj_upper_b - adj_lower_b
        expected_width_for_q = self.expected_ranges.get(current_quarter, self.DEFAULT_EXPECTED_RANGES[0]) * 2.0

        if expected_width_for_q > 1e-6:
         # Suggestion: Define a target confidence for when width matches expected,
            # and scale deviation from there. Example:
            target_conf_at_expected_width = 80.0 # e.g., 80% confidence if width is as expected
            width_ratio = final_width / expected_width_for_q if expected_width_for_q > 1e-6 else 1.0
            # Example scaling: Lower confidence if wider (ratio > 1), higher if narrower (ratio < 1)
            confidence = target_conf_at_expected_width - (width_ratio - 1.0) * 50.0 # Adjust the '50.0' sensitivity factor
            confidence = max(5.0, min(95.0, confidence)) # Apply caps (avoiding 0 or 100 maybe)
        else:
            confidence = 50.0 

        self.log(f"Calculated base interval: [{adj_lower_b:.1f}, {adj_upper_b:.1f}], Confidence: {confidence:.1f}%", level="DEBUG")
        return adj_lower_b, adj_upper_b, confidence


    def dynamically_adjust_interval(self,
                                    prediction: float,
                                    current_quarter: int,
                                    historic_accuracy: Optional[Dict] = None 
                                    ) -> Tuple[float, float, float]:
        """
        Calculates a base prediction interval and then adjusts its width based on
        provided historical coverage performance for that quarter compared to targets.

        Args:
            prediction: The point prediction score.
            current_quarter: The current quarter (0-4).
            historic_accuracy: Dict mapping quarters to dicts containing keys like
                               'actual_coverage'. Example: { 1: {'actual_coverage': 0.78}, ... }

        Returns:
            Tuple: (adjusted_lower_bound, adjusted_upper_bound, confidence_percentage)
                   Confidence here is the base confidence before dynamic adjustment.
                   The adjustment modifies the *bounds* to meet historical coverage.
        """
        # 1. Calculate Base Interval & Confidence using the other method
        lower_b, upper_b, base_confidence = self.calculate_prediction_interval(
            prediction=prediction,
            current_quarter=current_quarter,
            score_margin=None,
            momentum=None
        )
        base_width = upper_b - lower_b

        if base_width <= 0: 
            self.log("Base interval width is zero or negative. Cannot adjust.", level="WARNING")
            return lower_b, upper_b, base_confidence

        adj_lower, adj_upper = lower_b, upper_b 

        # 2. Adjust Interval Width based on Historical Coverage (if available)
        adjustment_factor = 1.0
        q_stats = None
        target_cov = None

        if historic_accuracy and isinstance(historic_accuracy, dict) and current_quarter in historic_accuracy:
            q_stats = historic_accuracy[current_quarter]
            actual_cov = q_stats.get('actual_coverage') 
            target_cov = self.coverage_targets.get(current_quarter) 

            if actual_cov is not None and target_cov is not None and target_cov > 0 and actual_cov > 0:
                # If actual coverage < target, need wider interval (factor > 1)
                # If actual coverage > target, need narrower interval (factor < 1)
                # Adjustment aims to scale the width to achieve the target coverage
                # Assumes a roughly linear relationship near the target, can be refined
                coverage_ratio = actual_cov / target_cov
                adjustment_factor = 1.0 / coverage_ratio 

                # Dampen adjustment: Apply only a fraction of the suggested change
                damping = 0.5
                adjustment_factor = 1.0 + (adjustment_factor - 1.0) * damping

                # Bound adjustment: Prevent extreme widening/narrowing
                min_factor, max_factor = 0.7, 1.5
                adjustment_factor = max(min_factor, min(max_factor, adjustment_factor))

                center = (lower_b + upper_b) / 2.0
                adj_half_width = (base_width / 2.0) * adjustment_factor
                adj_lower = center - adj_half_width
                adj_upper = center + adj_half_width

                self.log(f"Dynamically adjusted interval for Q{current_quarter}. Target Cov: {target_cov:.1%}, Actual Cov: {actual_cov:.1%}. Adjustment Factor: {adjustment_factor:.3f}. New Interval: [{adj_lower:.1f}, {adj_upper:.1f}]", level="DEBUG")

            else:
                 self.log(f"Missing valid actual ({actual_cov}) or target ({target_cov}) coverage for Q{current_quarter}. No dynamic width adjustment.", level="DEBUG")
        else:
             self.log(f"No historical accuracy data provided or quarter {current_quarter} missing. Using base interval width.", level="DEBUG")


        # NOTE: The confidence returned is still the 'base_confidence' associated with
        # the initially calculated interval. The dynamic adjustment primarily modifies
        # the bounds to align with historical performance, not necessarily recalculating
        # confidence based on the *new* width in this implementation.
        final_confidence = max(0.0, min(100.0, base_confidence))

        return adj_lower, adj_upper, final_confidence


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ END PREDICTION UNCERTAINTY ESTIMATOR DEFINITION +++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# --- Example Usage (Optional, for testing this module) ---
if __name__ == '__main__':
    # Ensure basic logging is set up if running directly
    if not logger.handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    logger.info("--- Running simulation.py example (PredictionUncertaintyEstimator) ---")

    # --- Test Case 1: Basic Initialization & Base Calculation ---
    print("\n--- Test Case 1: Basic Usage ---")
    estimator_basic = PredictionUncertaintyEstimator(debug=True)
    pred = 220.0
    for q_test in range(5): 
        print(f"\nTesting Quarter {q_test}...")
        l1, u1, c1 = estimator_basic.calculate_prediction_interval(pred, q_test, score_margin=15, momentum=0.2)
        print(f"  Base Interval (margin=15, mom=0.2): [{l1:.1f}, {u1:.1f}], Confidence: {c1:.1f}%")
        l2, u2, c2 = estimator_basic.calculate_prediction_interval(pred, q_test, score_margin=2, momentum=0.8)
        print(f"  Base Interval (margin=2, mom=0.8):  [{l2:.1f}, {u2:.1f}], Confidence: {c2:.1f}%")

    # --- Test Case 2: Initialization with Historical Stats & Dynamic Adjustment ---
    print("\n--- Test Case 2: Dynamic Adjustment Usage ---")
    # Create dummy historical stats DataFrame
    hist_stats_data = {
        'quarter': [0, 1, 2, 3, 4],
        # Simulate different coverage scenarios
        'actual_coverage': [0.65, 0.70, 0.85, 0.82, 0.95], 
        'avg_interval_width': [35, 30, 22, 18, 12]
    }
    hist_stats_df = pd.DataFrame(hist_stats_data)

    # Initialize estimator WITH historical stats
    estimator_dynamic = PredictionUncertaintyEstimator(
        historical_coverage_stats=hist_stats_df,
        debug=True
    )

    # Prepare the dictionary format needed by dynamically_adjust_interval
    hist_dict_for_func = None
    retrieved_stats = estimator_dynamic.get_coverage_stats()
    if retrieved_stats is not None:
        hist_dict_for_func = retrieved_stats.set_index('quarter').to_dict('index')
        print("\nHistorical accuracy stats loaded:")
        print(hist_dict_for_func)

    if hist_dict_for_func:
        for q_test in range(5):
            print(f"\nTesting Dynamic Adjustment for Quarter {q_test}...")
            target_cov_disp = estimator_dynamic.coverage_targets.get(q_test, 'N/A')
            actual_cov_disp = hist_dict_for_func.get(q_test, {}).get('actual_coverage', 'N/A')
            print(f"  Target Coverage: {target_cov_disp:.1%}, Actual Historical: {actual_cov_disp:.1%}")

            # Call the dynamic adjustment function
            l_adj, u_adj, c_adj = estimator_dynamic.dynamically_adjust_interval(
                prediction=pred,
                current_quarter=q_test,
                historic_accuracy=hist_dict_for_func
            )
            print(f"  => Adjusted Interval: [{l_adj:.1f}, {u_adj:.1f}], Confidence: {c_adj:.1f}%")
    else:
        print("\nCould not retrieve historical stats to test dynamic adjustment.")


    logger.info("--- simulation.py example finished ---")