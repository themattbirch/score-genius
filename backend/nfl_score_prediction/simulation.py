# backend/nfl_score_prediction/simulation.py
"""
simulation.py - NFL Monte Carlo Simulation Module

This module provides a simulator to generate probabilistic outcomes from the
pre-game point margin and total points predictions. By running thousands of
simulations based on the model's historical error (RMSE), it can calculate:
  - Win probabilities for each team.
  - Prediction intervals for final scores.
  - Probabilities for various betting market outcomes (e.g., covering the spread).
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class NFLMonteCarloSimulator:
    """
    Runs Monte Carlo simulations to generate probabilistic NFL game outcomes.
    """
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Initializes the simulator.

        Args:
            n_simulations: The number of game simulations to run.
            random_seed: A seed for the random number generator for reproducibility.
        """
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed=random_seed)
        self.margin_rmse: Optional[float] = None
        self.total_rmse: Optional[float] = None
        logger.info(f"NFLMonteCarloSimulator initialized with {n_simulations} simulations.")

    def set_error_distribution(self, margin_rmse: float, total_rmse: float):
        """
        Sets the historical prediction errors (RMSE) for the models.
        This is crucial for defining the simulation's variance.

        Args:
            margin_rmse: The Root Mean Squared Error of the margin prediction model.
            total_rmse: The Root Mean Squared Error of the total points model.
        """
        if margin_rmse <= 0 or total_rmse <= 0:
            raise ValueError("RMSE values must be positive.")
        self.margin_rmse = margin_rmse
        self.total_rmse = total_rmse
        logger.info(f"Error distribution set: Margin RMSE={margin_rmse:.2f}, Total RMSE={total_rmse:.2f}")

    def run_simulation(self, pred_margin: float, pred_total: float) -> pd.DataFrame:
        """
        Runs the Monte Carlo simulation for a single game.

        Args:
            pred_margin: The model's single-point prediction for the point margin.
            pred_total: The model's single-point prediction for the total points.

        Returns:
            A pandas DataFrame containing the raw results of all simulations.
        """
        if self.margin_rmse is None or self.total_rmse is None:
            raise RuntimeError("Error distribution must be set via set_error_distribution() before running.")

        # Simulate thousands of potential outcomes based on a normal distribution
        # centered at the prediction with a scale of the model's historical error (RMSE).
        sim_margins = self.rng.normal(loc=pred_margin, scale=self.margin_rmse, size=self.n_simulations)
        sim_totals = self.rng.normal(loc=pred_total, scale=self.total_rmse, size=self.n_simulations)

        # Derive home and away scores from the simulated margins and totals
        sim_home_scores = (sim_totals + sim_margins) / 2
        sim_away_scores = (sim_totals - sim_margins) / 2

        # Final scores are discrete integers, so we round the results.
        # Scores also cannot be negative.
        sim_home_scores = np.maximum(0, np.round(sim_home_scores)).astype(int)
        sim_away_scores = np.maximum(0, np.round(sim_away_scores)).astype(int)

        return pd.DataFrame({
            'sim_home_score': sim_home_scores,
            'sim_away_score': sim_away_scores
        })

    def summarize_simulation(
        self,
        sim_results: pd.DataFrame,
        home_spread: Optional[float] = None,
        game_total_line: Optional[float] = None
    ) -> Dict:
        """
        Summarizes the raw simulation results into key probabilistic metrics.

        Args:
            sim_results: The DataFrame of raw simulation results from run_simulation().
            home_spread: The betting point spread for the home team (e.g., -6.5).
            game_total_line: The betting total points line (e.g., 48.5).

        Returns:
            A dictionary containing win probabilities, score intervals, and betting market probabilities.
        """
        # Calculate derived columns for analysis
        sim_results['sim_margin'] = sim_results['sim_home_score'] - sim_results['sim_away_score']
        sim_results['sim_total'] = sim_results['sim_home_score'] + sim_results['sim_away_score']

        summary = {}

        # 1. Win Probability
        summary['home_win_prob'] = (sim_results['sim_margin'] > 0).mean()
        summary['away_win_prob'] = (sim_results['sim_margin'] < 0).mean()
        summary['tie_prob'] = (sim_results['sim_margin'] == 0).mean()

        # 2. Score and Margin Intervals (80% confidence interval)
        quantiles = [0.10, 0.50, 0.90] # 10th, 50th (median), 90th percentile
        summary['home_score_dist'] = sim_results['sim_home_score'].quantile(quantiles).to_dict()
        summary['away_score_dist'] = sim_results['sim_away_score'].quantile(quantiles).to_dict()
        summary['margin_dist'] = sim_results['sim_margin'].quantile(quantiles).to_dict()
        summary['total_dist'] = sim_results['sim_total'].quantile(quantiles).to_dict()

        # 3. Betting Market Probabilities
        if home_spread is not None:
            # Home team covers if their margin is greater than their spread
            # e.g., if spread is -6.5, margin must be > -6.5 (i.e., win by 7+)
            summary['home_cover_prob'] = (sim_results['sim_margin'] > -home_spread).mean()

        if game_total_line is not None:
            summary['over_prob'] = (sim_results['sim_total'] > game_total_line).mean()
            summary['under_prob'] = (sim_results['sim_total'] < game_total_line).mean()

        return summary

if __name__ == '__main__':
    logger.info("--- Running simulation.py example ---")

    # 1. Initialize the simulator
    simulator = NFLMonteCarloSimulator(n_simulations=50000)

    # 2. Set the model's historical error (these values would come from evaluating the
    #    trained models on a test set during the training pipeline).
    # Example: Our margin model is off by ~13 points on average, total by ~10 points.
    simulator.set_error_distribution(margin_rmse=13.0, total_rmse=10.5)

    # 3. Define a hypothetical pre-game prediction and betting lines
    # Model predicts home team wins by 3, with a total of 45.
    predicted_margin = 3.0
    predicted_total = 45.0
    # Betting market lines for the game
    market_home_spread = -2.5  # Home team is a 2.5 point favorite
    market_total_line = 47.5   # Over/Under is 47.5

    # 4. Run the simulation
    raw_results = simulator.run_simulation(predicted_margin, predicted_total)

    # 5. Summarize the results
    summary = simulator.summarize_simulation(
        sim_results=raw_results,
        home_spread=market_home_spread,
        game_total_line=market_total_line
    )

    # 6. Print the summary
    print("\n--- Simulation Summary ---")
    print(f"Prediction: Margin={predicted_margin:.1f}, Total={predicted_total:.1f}")
    print(f"Betting Lines: Spread={market_home_spread:.1f}, Total={market_total_line:.1f}\n")
    print(f"Home Win Probability: {summary['home_win_prob']:.1%}")
    print(f"Away Win Probability: {summary['away_win_prob']:.1%}\n")

    print(f"80% Interval for Home Score: {summary['home_score_dist'][0.10]:.0f} - {summary['home_score_dist'][0.90]:.0f} (Median: {summary['home_score_dist'][0.50]:.0f})")
    print(f"80% Interval for Away Score: {summary['away_score_dist'][0.10]:.0f} - {summary['away_score_dist'][0.90]:.0f} (Median: {summary['away_score_dist'][0.50]:.0f})\n")

    print(f"Probability Home Covers {-market_home_spread:.1f}: {summary['home_cover_prob']:.1%}")
    print(f"Probability Over {market_total_line:.1f}: {summary['over_prob']:.1%}")
    print(f"Probability Under {market_total_line:.1f}: {summary['under_prob']:.1%}")