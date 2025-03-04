# backend/models/score_prediction.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy import create_engine

from caching.supabase_client import supabase
from config import MODEL_PATH

def load_training_data():
    """
    Loads historical game data from Supabase.
    Returns a Pandas DataFrame containing the data from the 'nba_historical_game_stats' table.
    """
    response = supabase.table("nba_historical_game_stats").select("*").execute()
    data = response.data  # 'data' is expected to be a list of dictionaries
    if not data:
        raise ValueError("No data returned from Supabase. Check your table and connection.")
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Extracts and engineers features for score prediction.
    Adjustments include:
      - Using 'home_score' as the target.
      - Computing a score ratio (home_score / (home_score + away_score)).
      - Computing rolling averages of home and away scores (previous 5 games per team).
      - **New:** Computing historical matchup difference between home and away teams.
    """
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Compute rolling averages for home and away teams
    df.sort_values(by=['home_team', 'game_date'], inplace=True)
    df['rolling_home_score'] = df.groupby('home_team')['home_score'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    df.sort_values(by=['away_team', 'game_date'], inplace=True)
    df['rolling_away_score'] = df.groupby('away_team')['away_score'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    # Fill missing rolling averages with overall means
    df['rolling_home_score'] = df['rolling_home_score'].fillna(df['home_score'].mean())
    df['rolling_away_score'] = df['rolling_away_score'].fillna(df['away_score'].mean())

    # Compute score ratio
    df['score_ratio'] = df['home_score'] / (df['home_score'] + df['away_score'])

    # **Compute Historical Matchup Difference:**
    # For each game, we calculate the average difference (home_score - away_score)
    # from previous encounters between the same teams.
    def compute_matchup_diff(row):
        home = row['home_team']
        away = row['away_team']
        current_date = row['game_date']
        # Filter past games where these teams have played
        past_games = df[(df['home_team'] == home) & (df['away_team'] == away) & (df['game_date'] < current_date)]
        if not past_games.empty:
            return (past_games['home_score'] - past_games['away_score']).mean()
        else:
            return 0  # or a default value

    df['prev_matchup_diff'] = df.apply(compute_matchup_diff, axis=1)

    # Define feature columns
    feature_cols = [
        'home_q1', 'home_q2', 'home_q3', 'home_q4',
        'score_ratio', 'rolling_home_score', 'rolling_away_score',
        'prev_matchup_diff'  # New feature
    ]

    target_col = 'home_score'
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

def predict_final_score(model, game_id, home_team, away_team, db_connection):
    """
    Uses precomputed features to predict the final home score.
    
    Parameters:
        model: Trained prediction model.
        game_id: ID of the game to predict.
        home_team: Home team identifier.
        away_team: Away team identifier.
        db_connection: Database connection string.
        
    Returns:
        Predicted final score.
    """
    # Connect to the database
    conn = create_engine(db_connection)
    
    # Retrieve precomputed features for the given game and home team
    query = f"""
    SELECT * FROM nba_precomputed_features 
    WHERE game_id = '{game_id}' AND team = '{home_team}'
    """
    features_df = pd.read_sql(query, conn)
    if features_df.empty:
        raise ValueError(f"No precomputed features found for game_id {game_id} and team {home_team}")
    
    # Retrieve game-level data (for quarter scores, etc.)
    query_game = f"SELECT * FROM nba_historical_game_stats WHERE game_id = '{game_id}'"
    game_df = pd.read_sql(query_game, conn)
    if game_df.empty:
        raise ValueError(f"Game data not found for game_id {game_id}")
    
    # Construct the feature vector
    # (Ensure the order here matches the order used during model training)
    feature_vector = [
        game_df.iloc[0]['home_q1'],
        game_df.iloc[0]['home_q2'],
        game_df.iloc[0]['home_q3'],
        game_df.iloc[0]['home_q4'],
        features_df.iloc[0]['rolling_pts_5'],
        # Add additional features as needed (e.g., efficiency, prev_matchup_diff)
        features_df.iloc[0]['efficiency'],
        features_df.iloc[0]['prev_matchup_diff']
    ]
    
    # Make and return the prediction
    prediction = model.predict([feature_vector])
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
