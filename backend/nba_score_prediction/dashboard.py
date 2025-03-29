# backend/nba_score_prediction/dashboard.py

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
from functools import wraps
import scipy.stats as stats
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Union, Any

# -------------------- ConfidenceVisualizer Function --------------------
class ConfidenceVisualizer:
    """
    Generates visual confidence indicators (SVG format) for predictions,
    incorporating prediction intervals and game context.
    """

    def __init__(self):
        """Initialize the confidence visualizer."""
        # Base colors
        self.colors = {
            'background': '#f8f9fa', # Light grey background
            'text_primary': '#333333', # Dark text
            'text_secondary': '#555555', # Lighter text
            'current_score_marker': '#28a745', # Green for current score
            # Quarter-specific colors for the interval/prediction point
            # (Adjust these colors as desired)
            'q0': "#d3d3d3",  # Gray for pregame
            'q1': "#ffa07a",  # Light salmon
            'q2': "#ff7f50",  # Coral
            'q3': "#ff4500",  # OrangeRed
            'q4': "#dc3545",  # Red (Slightly less dark than darkred)
            'default_q': "#6c757d" # Secondary grey for fallback
        }
        # Expected interval width by quarter (used for confidence % calculation)
        # These values should ideally be derived from your PredictionUncertaintyEstimator's analysis
        # or tuned based on observed performance. Taken from the old estimator code.
        self.EXPECTED_RANGES = {0: 30.0, 1: 26.0, 2: 22.0, 3: 18.0, 4: 14.0}

    def _get_quarter_color(self, quarter: int) -> str:
        """Get the color associated with a specific quarter."""
        return self.colors.get(f'q{quarter}', self.colors['default_q'])

    def create_confidence_svg(
        self,
        prediction: float,
        lower_bound: float,
        upper_bound: float,
        current_quarter: int = 0,
        current_home_score: Optional[float] = None, # Optional current score marker
        width: int = 300,
        height: int = 80 # Adjusted height for better spacing
    ) -> str:
        """
        Create an SVG visualization of prediction confidence interval.

        Args:
            prediction: The point prediction value.
            lower_bound: The lower bound of the confidence interval.
            upper_bound: The upper bound of the confidence interval.
            current_quarter: Current game quarter (0-4).
            current_home_score: Optional current home score to display a marker.
            width: SVG width in pixels.
            height: SVG height in pixels.

        Returns:
            SVG markup as a string.
        """
        # --- Calculations ---
        interval_width = upper_bound - lower_bound
        color = self._get_quarter_color(current_quarter)

        # Define the visible score range in the SVG (add padding)
        padding = 10 # Min padding around the interval
        svg_min_score = max(0, lower_bound - padding - (interval_width * 0.1)) # Add a bit more space left
        svg_max_score = upper_bound + padding + (interval_width * 0.1) # Add a bit more space right
        # Ensure range isn't too small if interval is tiny
        if (svg_max_score - svg_min_score) < 10.0:
            mid_point = (lower_bound + upper_bound) / 2.0
            svg_min_score = mid_point - 5.0
            svg_max_score = mid_point + 5.0

        score_range_svg = svg_max_score - svg_min_score

        # Function to scale score to SVG x-coordinate (within drawing area)
        content_width = width # Use full width for scaling
        def to_svg_x(score):
            # Handle edge case where score_range_svg is zero or very small
            if score_range_svg <= 1e-6:
                return content_width / 2
            # Clamp score to visible range before scaling to avoid extreme coordinates
            clamped_score = max(svg_min_score, min(score, svg_max_score))
            return ((clamped_score - svg_min_score) / score_range_svg) * content_width


        # Calculate positions
        pred_x = to_svg_x(prediction)
        lower_x = to_svg_x(lower_bound)
        upper_x = to_svg_x(upper_bound)

        # Calculate confidence percentage (inverse relationship with interval width vs expected)
        # Use expected range for the current quarter, default if quarter > 4
        expected_width = self.EXPECTED_RANGES.get(current_quarter, self.EXPECTED_RANGES[4]) # Default to Q4 expected width if invalid Q
        confidence_pct = 0.0
        if expected_width > 0: # Avoid division by zero
             # Confidence decreases as interval_width exceeds expected_width
             # This formula ensures 100% at width=0, decreasing linearly based on ratio. Capped at 0.
             confidence_pct = max(0.0, min(100.0, 100.0 - (interval_width / expected_width * 75.0))) # Adjusted scaling factor

        # --- SVG Generation ---
        # Using f-string with multi-line capability
        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="font-family: Arial, sans-serif;">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{self.colors['background']}" rx="5" ry="5" />

  <text x="10" y="18" font-size="12px" fill="{self.colors['text_secondary']}">Prediction Range (Q{current_quarter})</text>

  <rect x="{lower_x}" y="{height/2 - 6}" width="{max(0, upper_x - lower_x)}" height="12" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="1" rx="3" ry="3" />

  <circle cx="{pred_x}" cy="{height/2}" r="5" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>
  <text x="{pred_x}" y="{height/2 - 10}" text-anchor="middle" font-size="13px" fill="{self.colors['text_primary']}" font-weight="bold">{prediction:.1f}</text>

  <text x="{lower_x}" y="{height/2 + 18}" text-anchor="{ 'start' if lower_x < 15 else 'middle' }" font-size="11px" fill="{self.colors['text_secondary']}" >{lower_bound:.1f}</text>
  <text x="{upper_x}" y="{height/2 + 18}" text-anchor="{ 'end' if upper_x > width - 15 else 'middle' }" font-size="11px" fill="{self.colors['text_secondary']}">{upper_bound:.1f}</text>
"""
        # Add Current Score Marker (Optional)
        if current_home_score is not None and current_home_score >= 0:
            # Only draw if within the visible range to avoid clutter
            if svg_min_score <= current_home_score <= svg_max_score:
                score_x = to_svg_x(current_home_score)
                svg += f"""
  <line x1="{score_x}" y1="{height/2 - 12}" x2="{score_x}" y2="{height/2 + 12}" stroke="{self.colors['current_score_marker']}" stroke-width="1.5" stroke-dasharray="2,2" />
  <text x="{score_x}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="10px" fill="{self.colors['current_score_marker']}" font-weight="bold">Cur: {current_home_score:.0f}</text>
"""
        # Add Confidence Percentage Label
        svg += f"""
  <text x="{width - 10}" y="18" text-anchor="end" font-size="12px" fill="{self.colors['text_primary']}" font-weight="bold">{confidence_pct:.0f}% Conf.</text>
</svg>"""

        return svg