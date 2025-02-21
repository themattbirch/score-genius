import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Template path to save/load a model for each player based on their player_id.
MODEL_PATH_TEMPLATE = os.path.join(os.path.dirname(__file__), 'player_projection_model_{}.pkl')

def load_player_data(player_id):
    """
    Loads historical player stats data for a specific player.
    Assumes a CSV export from Supabase at '../../data/historical_player_stats.csv'
    with columns including 'game_date', 'player_id', and 'points'.
    """
    data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_player_stats.csv')
    df = pd.read_csv(data_path, parse_dates=['game_date'])
    # Filter data for the given player_id
    df_player = df[df['player_id'] == player_id].copy()
    df_player.sort_values('game_date', inplace=True)
    return df_player

def preprocess_player_data(df_player):
    """
    Preprocess the player data for ARIMA model training.
    Uses 'points' as the target metric.
    """
    # Set 'game_date' as the index
    df_player.set_index('game_date', inplace=True)
    # Extract the time series for points
    ts = df_player['points']
    return ts

def train_arima_model(ts, order=(1, 1, 1)):
    """
    Trains an ARIMA model on the given time series data.
    Returns the fitted model.
    """
    model = ARIMA(ts, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def save_model(model_fit, player_id):
    """
    Saves the trained model to disk using a player-specific filename.
    """
    filepath = MODEL_PATH_TEMPLATE.format(player_id)
    joblib.dump(model_fit, filepath)
    print(f"Model for player {player_id} saved to {filepath}")

def load_model(player_id):
    """
    Loads the trained model for a given player_id if it exists.
    """
    filepath = MODEL_PATH_TEMPLATE.format(player_id)
    if os.path.exists(filepath):
        model_fit = joblib.load(filepath)
        print(f"Model for player {player_id} loaded successfully.")
        return model_fit
    else:
        print("No model found for player", player_id, ". Please train the model first.")
        return None

def predict_player_performance(model_fit, steps=1):
    """
    Forecasts the next 'steps' performance metrics (e.g., points) for the player.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

if __name__ == '__main__':
    # Example usage for a specific player_id, e.g., player_id = 23
    player_id = 23
    df_player = load_player_data(player_id)
    if df_player.empty:
        print("No data found for player", player_id)
    else:
        ts = preprocess_player_data(df_player)
        model_fit = train_arima_model(ts)
        save_model(model_fit, player_id)
        # Forecast the next game performance (points)
        forecast = predict_player_performance(model_fit, steps=1)
        print(f"Forecast for player {player_id}: {forecast}")
