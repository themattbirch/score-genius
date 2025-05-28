# backend/nba_score_prediction/dummy_modules.py

import numpy as np
import pandas as pd
import logging # Import logging to use logger

# Configure a logger for the dummy module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config if not set elsewhere

# --- DUMMY Feature Pipeline Function ---
def run_feature_pipeline(*args, **kwargs) -> pd.DataFrame:
    """Dummy implementation of the feature engineering pipeline."""
    logger.warning("--- Using DUMMY run_feature_pipeline ---")
    # Print received arguments for debugging if needed
    # logger.debug(f"Dummy run_feature_pipeline received args: {args}")
    # logger.debug(f"Dummy run_feature_pipeline received kwargs: {kwargs}")
    # Return an empty DataFrame as a placeholder
    return pd.DataFrame()

# --- REMOVED Dummy FeatureEngine Class ---
# class FeatureEngine:
#     def __init__(self, *args, **kwargs): pass
#     def generate_all_features(self, **kwargs): return pd.DataFrame()

# --- Other Dummy Classes/Functions (Keep as they were) ---
class SVRScorePredictor:
    def __init__(self, *args, **kwargs):
        logger.warning("--- Using DUMMY SVRScorePredictor ---")
        pass
    def train(self, **kwargs):
        pass
    def predict(self, X):
        logger.warning("--- DUMMY SVRScorePredictor: predict called ---")
        # Return empty DataFrame with expected columns if possible, else just empty
        return pd.DataFrame(columns=['predicted_home_score', 'predicted_away_score'])
    def load_model(self, *args, **kwargs): # Add dummy load_model if used in load_trained_models
         logger.warning("--- DUMMY SVRScorePredictor: load_model called ---")
         pass


class RidgeScorePredictor:
    def __init__(self, *args, **kwargs):
        logger.warning("--- Using DUMMY RidgeScorePredictor ---")
        pass
    def train(self, **kwargs):
        pass
    def predict(self, X):
        logger.warning("--- DUMMY RidgeScorePredictor: predict called ---")
        return pd.DataFrame(columns=['predicted_home_score', 'predicted_away_score'])
    def load_model(self, *args, **kwargs): # Add dummy load_model
         logger.warning("--- DUMMY RidgeScorePredictor: load_model called ---")
         pass


def compute_recency_weights(*args, **kwargs):
    logger.warning("--- Using DUMMY compute_recency_weights ---")
    return np.ones(10)  # simple example

# Dummy plotting functions
def plot_feature_importances(*args, **kwargs): logger.warning("--- DUMMY plot_feature_importances called ---"); pass
def plot_actual_vs_predicted(*args, **kwargs): logger.warning("--- DUMMY plot_actual_vs_predicted called ---"); pass
def plot_residuals_analysis_detailed(*args, **kwargs): logger.warning("--- DUMMY plot_residuals_analysis_detailed called ---"); pass
def plot_conditional_bias(*args, **kwargs): logger.warning("--- DUMMY plot_conditional_bias called ---"); pass
def plot_temporal_bias(*args, **kwargs): logger.warning("--- DUMMY plot_temporal_bias called ---"); pass

# Dummy utils class/functions (if needed)
class utils: # If you were importing a utils class
    @staticmethod
    def remove_duplicate_columns(df):
        logger.warning("--- Using DUMMY utils.remove_duplicate_columns ---")
        return df

# Add any other dummy functions/classes needed by train_models.py or prediction.py
# For example, if helper functions from prediction.py were moved to a utils file:
def get_supabase_client(): logger.warning("--- Using DUMMY get_supabase_client ---"); return None
def load_recent_historical_data(*args, **kwargs): logger.warning("--- Using DUMMY load_recent_historical_data ---"); return pd.DataFrame()
def load_team_stats_data(*args, **kwargs): logger.warning("--- Using DUMMY load_team_stats_data ---"); return pd.DataFrame()
def fetch_upcoming_games_data(*args, **kwargs): logger.warning("--- Using DUMMY fetch_upcoming_games_data ---"); return pd.DataFrame()
def load_trained_models(*args, **kwargs): logger.warning("--- Using DUMMY load_trained_models ---"); return {}, []
def fetch_and_parse_betting_odds(*args, **kwargs): logger.warning("--- Using DUMMY fetch_and_parse_betting_odds ---"); return {}
def calibrate_prediction_with_odds(pred, odds, factor): logger.warning("--- Using DUMMY calibrate_prediction_with_odds ---"); return pred # Passthrough
def display_prediction_summary(preds): logger.warning("--- Using DUMMY display_prediction_summary ---"); pass
def upsert_score_predictions(preds): logger.warning("--- Using DUMMY upsert_score_predictions ---"); pass

# Dummy simulation class if needed
class PredictionUncertaintyEstimator:
        def __init__(self, *args, **kwargs): logger.warning("--- Using DUMMY PredictionUncertaintyEstimator ---"); pass
        def add_prediction_intervals(self, df): logger.warning("--- DUMMY PredictionUncertaintyEstimator: add_prediction_intervals called ---"); return df # Passthrough