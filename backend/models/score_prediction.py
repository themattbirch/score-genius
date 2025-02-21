# backend/models/score_prediction.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'score_prediction_model.pkl')

def load_training_data():
    """
    Loads historical game data.
    In production, this could be replaced with a Supabase query.
    For now, we'll assume a CSV export from our notebook.
    """
    data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_games.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Extract and engineer features for score prediction.
    For example, use current scores, quarter breakdowns, etc.
    Adjust feature selection based on available columns.
    """
    # Example features: using current home and away scores, quarter averages, etc.
    feature_cols = ['home_score', 'away_score', 'home_q1', 'home_q2', 'home_q3', 'home_q4']
    # Assume final_home_score is our target (or final combined game score)
    target_col = 'final_home_score'  # Adjust as needed
    # Drop rows with missing values for simplicity
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_model(X, y):
    """
    Trains a linear regression model for final score prediction.
    """
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Training MSE: {mse:.2f}")
    return model

def save_model(model, filepath=MODEL_PATH):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath=MODEL_PATH):
    """
    Loads the trained model from disk.
    """
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print("Model loaded successfully.")
        return model
    else:
        print("No model found. Please train the model first.")
        return None

def predict_final_score(model, current_features):
    """
    Uses the trained model to predict the final score.
    current_features should be a list or array matching the training features.
    """
    prediction = model.predict([current_features])
    return prediction[0]

if __name__ == '__main__':
    # Example training routine:
    df = load_training_data()
    X, y = preprocess_data(df)
    model = train_model(X, y)
    save_model(model)
    
    # Example prediction:
    # Replace the following with actual current game features:
    example_features = [110, 108, 28, 27, 29, 26]
    predicted_score = predict_final_score(model, example_features)
    print(f"Predicted final home score: {predicted_score:.2f}")
