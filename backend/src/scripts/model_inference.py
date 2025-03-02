# File: backend/src/scripts/model_inference.py

import os
import joblib
import pandas as pd
from models.score_prediction import load_training_data, preprocess_data, train_model, save_model
from caching.supabase_client import supabase
import datetime

# Define the model file path. In a notebook or script, __file__ may not be available so we use os.getcwd()
MODEL_PATH = os.path.join(os.getcwd(), 'score_prediction_model.pkl')

def fetch_new_features():
    """
    Fetches the latest engineered features from Supabase.
    Assumes the features are stored in a table called "nba_precomputed_features".
    Returns a DataFrame containing the features.
    """
    response = supabase.table("nba_precomputed_features").select("*").execute()
    data = response.data
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        print("No new features found.")
        return None

def generate_predictions(model, X):
    """
    Generates predictions using the provided model and feature DataFrame X.
    If X contains an identifier column like 'unique_game_id', it is dropped.
    Returns the predictions as a NumPy array.
    """
    # Drop identifier columns if present
    if 'unique_game_id' in X.columns:
        X_features = X.drop(columns=['unique_game_id'])
    else:
        X_features = X.copy()
    predictions = model.predict(X_features)
    return predictions

def retrain_model():
    """
    Retrains the model using the entire updated historical dataset.
    Loads historical data, runs feature engineering, trains a new model, and saves it.
    Returns the newly trained model.
    """
    print("Retraining the model with updated historical data...")
    df = load_training_data()
    # Optionally, filter or trim the historical data here if needed.
    X, y = preprocess_data(df)
    model = train_model(X, y)
    save_model(model)
    print("Model retraining complete.")
    return model

def run_model_inference():
    """
    Runs model inference and retraining.
    Steps:
      - Load the latest engineered features from Supabase.
      - Load the existing trained model (or retrain if not found).
      - Generate predictions on the new data.
      - Log the predictions.
      - Retrain the model with updated historical data.
    """
    print("Running model inference...")

    # Step 1: Load new engineered features from Supabase
    new_features_df = fetch_new_features()
    if new_features_df is None or new_features_df.empty:
        print("No new feature data available. Skipping inference.")
        return

    # Step 2: Load the existing trained model, or retrain if not found
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Loaded existing model from:", MODEL_PATH)
    else:
        print("No trained model found. Retraining a new model.")
        model = retrain_model()

    # Step 3: Generate predictions on new features
    predictions = generate_predictions(model, new_features_df)
    new_features_df['predicted_home_score'] = predictions

    # Log the predictions. Here we print out the game IDs and predictions.
    if 'unique_game_id' in new_features_df.columns:
        print("Predictions for new games:")
        print(new_features_df[['unique_game_id', 'predicted_home_score']].head())
    else:
        print("Predictions:")
        print(new_features_df[['predicted_home_score']].head())

    # Step 4: Retrain the model with updated historical data
    updated_model = retrain_model()

    print("Model inference and retraining complete.")

if __name__ == "__main__":
    run_model_inference()
