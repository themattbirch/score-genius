# backend/models/score_prediction.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

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
    Extracts and engineers features for score prediction.
    Adjustments include:
      - Using 'home_score' as the target.
      - Computing a score ratio (home_score / (home_score + away_score)).
      - Computing rolling averages of home and away scores (previous 5 games per team).
    """
    # Ensure game_date is a datetime and sort for rolling calculations
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Compute rolling averages for home and away teams (shifted to exclude current game)
    df.sort_values(by=['home_team', 'game_date'], inplace=True)
    df['rolling_home_score'] = df.groupby('home_team')['home_score'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    df.sort_values(by=['away_team', 'game_date'], inplace=True)
    df['rolling_away_score'] = df.groupby('away_team')['away_score'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    
    # Fill NaN values with overall averages if needed
    df['rolling_home_score'].fillna(df['home_score'].mean(), inplace=True)
    df['rolling_away_score'].fillna(df['away_score'].mean(), inplace=True)
    
    # Compute relative stat: score_ratio
    df['score_ratio'] = df['home_score'] / (df['home_score'] + df['away_score'])
    
    # Define feature columns. We use quarter scores and the new engineered features.
    feature_cols = [
        'home_q1', 'home_q2', 'home_q3', 'home_q4',
        'score_ratio', 'rolling_home_score', 'rolling_away_score'
    ]
    
    # Target is now the final home_score
    target_col = 'home_score'
    
    # Drop rows with missing values in features or target
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_model(X, y):
    """
    Trains a GradientBoostingRegressor for predicting the final home score.
    Uses a train-test split to report test MSE.
    Also prints feature importance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on training and test data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("Feature Importance:\n", feature_importance)
    
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

def adjust_live_features(features):
    """
    Adjusts input features for live game predictions.
    If certain quarter data is missing (None), fill with default estimates.
    
    Expected feature order:
      [home_q1, home_q2, home_q3, home_q4, score_ratio, rolling_home_score, rolling_away_score]
    
    For live games during Q2, for example, home_q3 and home_q4 might be missing.
    We fill them with the average of available quarters or historical averages.
    """
    # Example defaults (you may compute these from historical data)
    default_q3 = 28  # Example average for Q3
    default_q4 = 27  # Example average for Q4
    
    # If home_q3 or home_q4 are None or not provided, fill with default values.
    if features[2] is None:
        features[2] = default_q3
    if features[3] is None:
        features[3] = default_q4
    return features

def predict_final_score(model, current_features):
    """
    Uses the trained model to predict the final score.
    Validates that the input features match the expected count.
    
    Parameters:
        model: Trained model with an attribute `feature_names_in_` (set during fit).
        current_features (list or array): Input features for prediction.
        
    Returns:
        float: Predicted final home score.
    """
    # Adjust live features if necessary
    current_features = adjust_live_features(current_features)
    
    expected_features = len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected_features is not None and len(current_features) != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {len(current_features)}")
    prediction = model.predict([current_features])
    return prediction[0]

if __name__ == '__main__':
    # Example training routine:
    df = load_training_data()
    X, y = preprocess_data(df)
    model = train_model(X, y)
    save_model(model)
    
    # Example prediction:
    # For live game predictions, if Q3 and Q4 data are not yet available, pass None for those.
    # For example: [home_q1, home_q2, home_q3, home_q4, score_ratio, rolling_home_score, rolling_away_score]
    example_features = [28, 27, None, None, 0.51, 110, 108]
    predicted_score = predict_final_score(model, example_features)
    print(f"Predicted final home score: {predicted_score:.2f}")
