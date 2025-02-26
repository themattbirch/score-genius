# backend/models/ensemble_model.py

import numpy as np
import joblib

# Existing ensemble functions (ensemble_predictions, calculate_dynamic_weights, ensemble_with_confidence,
# get_context_weights, detect_outliers) would be defined here.

# --- Performance Tracking System ---
def update_historical_errors(historical_errors, model_predictions, actual_outcome):
    """
    Updates the historical error records after a game completes.
    
    Parameters:
        historical_errors (dict): Current error records.
        model_predictions (dict): Predictions made by each model.
        actual_outcome (float): The actual final value.
        
    Returns:
        dict: Updated historical errors.
    """
    # Learning rate (how quickly to update historical errors)
    alpha = 0.1
    
    for model, prediction in model_predictions.items():
        error = (prediction - actual_outcome) ** 2
        if model in historical_errors:
            # Exponential moving average of errors
            historical_errors[model] = (1 - alpha) * historical_errors[model] + alpha * error
        else:
            historical_errors[model] = error
            
    return historical_errors

# --- Expanded Game Contexts ---
def determine_game_context(current_score, quarter, time_remaining):
    """
    Determines the current game context based on game state.
    
    Parameters:
        current_score (tuple): (home_score, away_score).
        quarter (int): Current quarter (1-4, 5+ for OT).
        time_remaining (float): Minutes remaining in the quarter.
        
    Returns:
        str: Game context identifier.
    """
    home, away = current_score
    point_diff = abs(home - away)
    
    if quarter == 0:  # Pre-game
        return 'pre_game'
    elif quarter <= 2:
        return 'first_half'
    elif quarter == 4 and time_remaining < 5 and point_diff <= 10:
        return 'clutch_time'
    elif point_diff >= 20:
        return 'blowout'
    elif quarter >= 5:
        return 'overtime'
    elif point_diff <= 5:
        return 'close_game'
    else:
        return 'standard'

# --- Persistence Functionality ---
def save_ensemble_state(historical_errors, filepath='ensemble_state.pkl'):
    """Save the ensemble's learning state to disk."""
    joblib.dump(historical_errors, filepath)
    print(f"Ensemble state saved to {filepath}")

def load_ensemble_state(filepath='ensemble_state.pkl'):
    """Load the ensemble's learning state from disk."""
    try:
        state = joblib.load(filepath)
        print(f"Loaded ensemble state from {filepath}")
        return state
    except FileNotFoundError:
        print("No ensemble state file found. Returning empty state.")
        return {}
