# backend/models/player_projection.py

import os
import pandas as pd
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# For XGBoost quantile regression
from xgboost import XGBRegressor
from config import MODEL_PATH

def load_player_data(player_id):
    """
    Loads historical player stats data for a specific player.
    Assumes a CSV export from Supabase at '../../data/historical_player_stats.csv'
    with columns including 'game_date', 'player_id', and various performance metrics.
    """
    data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_player_stats.csv')
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    # Filter data for the given player_id
    df_player = df[df['player_id'] == player_id].copy()
    df_player.sort_values('game_date', inplace=True)
    return df_player

def preprocess_player_data(df_player, target_metrics=None):
    """
    Preprocess player data for multiple metrics.
    
    Parameters:
        df_player (DataFrame): The player's historical stats.
        target_metrics (list, optional): List of metrics to forecast.
            Defaults to ['points', 'rebounds', 'assists', 'steals', 'blocks'].
    
    Returns:
        dict: A dictionary mapping each metric to its time series (pandas Series).
    """
    if target_metrics is None:
        target_metrics = ['points', 'rebounds', 'assists', 'steals', 'blocks']
    df_player = df_player.copy()
    df_player.set_index('game_date', inplace=True)
    time_series = {metric: df_player[metric] for metric in target_metrics if metric in df_player.columns}
    return time_series

def create_features(ts, lag=3):
    """
    Create lag features for a time series to be used in regression.
    
    Parameters:
        ts (pandas Series): Time series data.
        lag (int): Number of lag features (default 3).
    
    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
    """
    df = pd.DataFrame(ts)
    df.columns = [ts.name]  # Ensure the series has a name
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df[ts.name].shift(i)
    df.dropna(inplace=True)
    X = df[[f'lag_{i}' for i in range(1, lag + 1)]]
    y = df[ts.name]
    return X, y

def train_arima_model(ts, order=(1, 1, 1)):
    """
    Trains an ARIMA model on the given time series data.
    Returns the fitted model.
    """
    model = ARIMA(ts, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def train_xgboost_with_quantiles(X, y):
    """
    Train multiple XGBoost models for different quantiles.
    
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        
    Returns:
        dict: Dictionary with keys 'median', 'lower', and 'upper' mapping to trained models.
    """
    models = {}
    # Train model for median (q=0.5)
    model_median = XGBRegressor()
    model_median.fit(X, y)
    models['median'] = model_median

    # Train model for lower bound (q=0.1)
    model_lower = XGBRegressor(objective='reg:quantile', alpha=0.1)
    model_lower.fit(X, y)
    models['lower'] = model_lower

    # Train model for upper bound (q=0.9)
    model_upper = XGBRegressor(objective='reg:quantile', alpha=0.9)
    model_upper.fit(X, y)
    models['upper'] = model_upper

    return models

def train_model(time_series, player_id, method="xgboost"):
    """
    Train models for each metric using the specified method.
    
    Parameters:
        time_series (dict): Mapping from metric names to pandas Series.
        player_id: Player identifier.
        method (str): "arima" or "xgboost". Defaults to "xgboost".
    
    Returns:
        dict: Mapping from metric to its trained model(s). For XGBoost, each value is a dict of quantile models.
    """
    models = {}
    if method == "arima":
        for metric, ts in time_series.items():
            model_fit = train_arima_model(ts, order=(1, 1, 1))
            models[metric] = model_fit
    elif method == "xgboost":
        for metric, ts in time_series.items():
            X, y = create_features(ts, lag=3)
            models[metric] = train_xgboost_with_quantiles(X, y)
    else:
        raise ValueError("Unsupported method. Choose 'arima' or 'xgboost'.")
    return models

def is_back_to_back(player_df, game_date):
    """Determine if a game is the second of a back-to-back set."""
    if player_df.empty or 'game_date' not in player_df:
        return False
        
    # Convert to datetime if needed
    dates = pd.to_datetime(player_df['game_date'])
    game_date = pd.to_datetime(game_date)
    
    # Find previous game date
    prev_games = dates[dates < game_date]
    if prev_games.empty:
        return False
        
    latest_prev = prev_games.max()
    days_between = (game_date - latest_prev).days
    
    return days_between == 1

def detect_streak(player_df, metric='points', games=5, threshold=1.2):
    """
    Detect if player is on a hot or cold streak compared to season average.
    
    Returns a tuple (streak_label, multiplier) where:
      - streak_label: 'hot', 'cold', or 'neutral'
      - multiplier: an adjustment factor (capped at 1.5 for hot, floored at 0.7 for cold)
    """
    if player_df.empty or metric not in player_df:
        return 'neutral', 1.0
    
    # Get season average for the metric
    season_avg = player_df[metric].mean()
    
    # Get recent average (last n games)
    recent = player_df.tail(games)
    if len(recent) < 3:  # Need at least 3 games to detect a streak
        return 'neutral', 1.0
        
    recent_avg = recent[metric].mean()
    
    # Calculate ratio compared to season average
    ratio = recent_avg / season_avg if season_avg > 0 else 1.0
    
    if ratio > threshold:
        return 'hot', min(ratio, 1.5)  # Cap at 1.5x multiplier for hot streaks
    elif ratio < (1/threshold):
        return 'cold', max(ratio, 0.7)  # Floor at 0.7x multiplier for cold streaks
    else:
        return 'neutral', 1.0

def save_models(models, player_id):
    """
    Saves the trained models dictionary to disk using a player-specific filename.
    """
    filepath = MODEL_PATH.format(player_id)
    joblib.dump(models, filepath)
    print(f"Models for player {player_id} saved to {filepath}")

def load_models(player_id):
    """
    Loads the trained models for a given player_id if they exist.
    
    Returns:
        dict: Mapping from metric to model(s), or None if not found.
    """
    filepath = MODEL_PATH.format(player_id)
    if os.path.exists(filepath):
        models = joblib.load(filepath)
        print(f"Models for player {player_id} loaded successfully.")
        return models
    else:
        print("No models found for player", player_id, ". Please train the models first.")
        return None

def get_defensive_rating(opponent_id):
    """
    Retrieves the defensive rating for the given opponent.
    For demonstration, this function returns a dummy value.
    In production, implement a query or calculation based on historical team stats.
    """
    # Example: A defensive rating of 1.0 is average.
    return 1.0

def incorporate_matchup(base_prediction, opponent_id, defensive_ratings=None):
    """
    Adjust prediction based on opponent's defensive strength.
    
    Parameters:
        base_prediction (float): The base prediction.
        opponent_id: Identifier for the opponent.
        defensive_ratings (dict, optional): Mapping from opponent_id to defensive multiplier.
    
    Returns:
        float: Adjusted prediction.
    """
    if defensive_ratings is None:
        defensive_rating = get_defensive_rating(opponent_id)
    else:
        defensive_rating = defensive_ratings.get(opponent_id, 1.0)
    return base_prediction * defensive_rating

def adjust_for_minutes(prediction, current_minutes, avg_minutes):
    """
    Adjust prediction based on expected minutes vs. historical average.
    """
    if avg_minutes > 0:
        return prediction * (current_minutes / avg_minutes)
    return prediction

def adjust_live_prediction(base_prediction, current_stats, minutes_played, total_minutes=48):
    """
    Blend pre-game prediction with current performance for in-game updates.
    
    Parameters:
        base_prediction (float): Pre-game prediction.
        current_stats (float): Current accumulated stat.
        minutes_played (float): Minutes played so far.
        total_minutes (float): Expected total minutes (default 48).
    
    Returns:
        float: Adjusted prediction.
    """
    if minutes_played >= total_minutes:
        return current_stats
    weight = minutes_played / total_minutes
    remaining_projection = base_prediction * (1 - weight)
    return current_stats + remaining_projection

def predict_player_performance(models, steps=1, method="xgboost", opponent_id=None, current_live_data=None):
    """
    Forecasts the next 'steps' performance for each metric.
    
    For XGBoost quantile models, returns a dictionary with median prediction and confidence intervals.
    For ARIMA models, returns the forecast.
    
    Optionally applies matchup adjustments if opponent_id is provided and in-game adjustments if current_live_data is provided.
    
    Parameters:
        models (dict): Mapping from metric to trained model(s).
        steps (int): Number of future steps to forecast.
        method (str): "xgboost" or "arima".
        opponent_id: Opponent identifier for matchup adjustment.
        current_live_data (dict, optional): Current live stats to adjust predictions.
    
    Returns:
        dict: Mapping from metric to predictions. For XGBoost, each value is a dict with 'median', 'lower', 'upper'.
    """
    forecasts = {}
    for metric, model in models.items():
        if method == "arima":
            # ARIMA model: use forecast
            forecast = model.forecast(steps=steps)
            pred = forecast[0]
            # Optionally adjust for matchup
            if opponent_id is not None:
                pred = incorporate_matchup(pred, opponent_id)
            forecasts[metric] = {'median': pred}
        elif method == "xgboost":
            # XGBoost: assume model is a dict with keys 'median', 'lower', 'upper'
            # Here, we need to create proper feature vector from live data.
            # For demonstration, we use a placeholder vector.
            last_features = np.array([0] * 3)  # TODO: Replace with actual lag features from live data.
            median_pred = model['median'].predict([last_features])[0]
            lower_pred = model['lower'].predict([last_features])[0]
            upper_pred = model['upper'].predict([last_features])[0]
            # Apply matchup adjustment if opponent_id is provided.
            if opponent_id is not None:
                median_pred = incorporate_matchup(median_pred, opponent_id)
                lower_pred = incorporate_matchup(lower_pred, opponent_id)
                upper_pred = incorporate_matchup(upper_pred, opponent_id)
            # Optionally adjust live prediction using current_live_data if provided.
            if current_live_data and metric in current_live_data:
                median_pred = adjust_live_prediction(median_pred, current_live_data[metric], current_live_data.get("minutes_played", 0))
            forecasts[metric] = {
                'median': median_pred,
                'lower': lower_pred,
                'upper': upper_pred
            }
        else:
            raise ValueError("Unsupported method. Choose 'arima' or 'xgboost'.")
    return forecasts

if __name__ == '__main__':
    # Example usage for a specific player_id, e.g., player_id = 23
    player_id = 23
    df_player = load_player_data(player_id)
    if df_player.empty:
        print("No data found for player", player_id)
    else:
        # Preprocess data for multiple metrics
        time_series = preprocess_player_data(df_player)
        
        # Train models using XGBoost quantile regression
        models = train_model(time_series, player_id, method="xgboost")
        save_models(models, player_id)
        
        # Forecast the next game performance for each metric
        forecasts = predict_player_performance(models, steps=1, method="xgboost", opponent_id="opponent_001", current_live_data={"points": 20, "minutes_played": 30})
        print(f"Forecasts for player {player_id}:\n", forecasts)
