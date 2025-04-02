# backend/src/scripts/run_predictions.py

import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback
from datetime import datetime, timedelta
import importlib.util

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

import config
from caching.supabase_client import supabase

# Dynamically import the module by file path (will work at runtime)
module_path = os.path.join(os.path.dirname(__file__), 'nba_stats_live_handler.py')
spec = importlib.util.spec_from_file_location("nba_stats_live_handler", module_path)
nba_stats_live_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nba_stats_live_handler)
get_games_for_prediction = nba_stats_live_handler.get_games_for_predictio

def load_prediction_model():
    """Load the trained score prediction model."""
    try:
        model_path = config.MODEL_PATH
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"Model file not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def get_team_rolling_averages():
    """Get rolling scoring averages for all teams."""
    try:
        # Get recent games from the last 60 days
        threshold_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        response = supabase.table("nba_historical_game_stats").select("*").gte("game_date", threshold_date).execute()
        
        if not response.data:
            print("No historical data found for team averages")
            return {}
        
        # Calculate team averages
        team_avgs = {}
        df = pd.DataFrame(response.data)
        
        # Get unique teams
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = home_teams.union(away_teams)
        
        for team in all_teams:
            # Get home and away games
            home_games = df[df['home_team'] == team]['home_score']
            away_games = df[df['away_team'] == team]['away_score']
            
            # Combine and calculate average
            all_scores = pd.concat([home_games, away_games])
            if len(all_scores) > 0:
                team_avgs[team] = all_scores.mean()
            else:
                team_avgs[team] = 105.0  # Default NBA average
        
        return team_avgs
    
    except Exception as e:
        print(f"Error getting team averages: {e}")
        traceback.print_exc()
        return {}

def get_previous_matchup_diff(home_team, away_team):
    """Get point differential from previous matchups between two teams."""
    try:
        # Query for games where these teams played each other
        response_home = supabase.table("nba_historical_game_stats").select("*")\
            .eq("home_team", home_team)\
            .eq("away_team", away_team)\
            .order('game_date', desc=True)\
            .limit(5).execute()
            
        response_away = supabase.table("nba_historical_game_stats").select("*")\
            .eq("home_team", away_team)\
            .eq("away_team", home_team)\
            .order('game_date', desc=True)\
            .limit(5).execute()
        
        # Combine results
        matchups = response_home.data + response_away.data
        
        if not matchups:
            return 0
        
        # Calculate point differential from home team perspective
        diffs = []
        for game in matchups:
            if game['home_team'] == home_team:
                # Home team was home in this game
                diff = game['home_score'] - game['away_score']
            else:
                # Home team was away in this game
                diff = game['away_score'] - game['home_score']
            diffs.append(diff)
        
        # Return average differential
        return sum(diffs) / len(diffs) if diffs else 0
    
    except Exception as e:
        print(f"Error getting matchup difference: {e}")
        return 0

def prepare_features(games_df, model=None):
    """Prepare features for prediction based on model type."""
    if games_df.empty:
        return pd.DataFrame()
    
    # Determine if we have the enhanced model
    is_enhanced_model = False
    if model is not None and hasattr(model, 'feature_importances_'):
        feature_count = len(model.feature_importances_)
        is_enhanced_model = (feature_count > 8)
    
    # Define feature lists
    original_features = [
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 
        'score_ratio', 'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff'
    ]
    
    enhanced_features = [
        'home_q1', 'home_q2', 'home_q3', 'home_q4', 
        'score_ratio', 'prev_matchup_diff',
        'rest_days_home', 'rest_days_away', 'rest_advantage',
        'is_back_to_back_home', 'is_back_to_back_away',
        'q1_to_q2_momentum', 'q2_to_q3_momentum', 'q3_to_q4_momentum', 'cumulative_momentum'
    ]
    
    # Choose which features to use
    expected_features = enhanced_features if is_enhanced_model else original_features
    print(f"Using {'enhanced' if is_enhanced_model else 'original'} feature set")
    
    # Get team averages
    team_avgs = get_team_rolling_averages()
    
    # Prepare feature DataFrame
    features = []
    
    for idx, game in games_df.iterrows():
        # Get basic game data
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        current_quarter = int(game['current_quarter'])
        
        # Quarter scores
        home_q1 = float(game['home_q1'])
        home_q2 = float(game['home_q2'])
        home_q3 = float(game['home_q3'])
        home_q4 = float(game['home_q4'])
        
        away_q1 = float(game['away_q1'])
        away_q2 = float(game['away_q2'])
        away_q3 = float(game['away_q3'])
        away_q4 = float(game['away_q4'])
        
        # Get matchup history
        prev_matchup_diff = get_previous_matchup_diff(home_team, away_team)
        
        # Calculate score ratio
        score_ratio = game['score_ratio']
        
        # Create feature dictionary
        feature_dict = {
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_q1': home_q1,
            'home_q2': home_q2,
            'home_q3': home_q3,
            'home_q4': home_q4,
            'score_ratio': score_ratio,
            'prev_matchup_diff': prev_matchup_diff
        }
        
        # Add rolling averages for original model
        if not is_enhanced_model:
            feature_dict['rolling_home_score'] = team_avgs.get(home_team, 105.0)
            feature_dict['rolling_away_score'] = team_avgs.get(away_team, 105.0)
        
        # Add enhanced features if needed
        if is_enhanced_model:
            # Default rest features
            feature_dict['rest_days_home'] = 2
            feature_dict['rest_days_away'] = 2
            feature_dict['rest_advantage'] = 0
            feature_dict['is_back_to_back_home'] = 0
            feature_dict['is_back_to_back_away'] = 0
            
            # Calculate momentum features
            feature_dict['q1_to_q2_momentum'] = 0
            feature_dict['q2_to_q3_momentum'] = 0
            feature_dict['q3_to_q4_momentum'] = 0
            feature_dict['cumulative_momentum'] = 0
            
            if current_quarter >= 2:
                feature_dict['q1_to_q2_momentum'] = (home_q2 - home_q1) - (away_q2 - away_q1)
                
            if current_quarter >= 3:
                feature_dict['q2_to_q3_momentum'] = (home_q3 - home_q2) - (away_q3 - away_q2)
                
            if current_quarter >= 4:
                feature_dict['q3_to_q4_momentum'] = (home_q4 - home_q3) - (away_q4 - away_q3)
            
            # Calculate cumulative momentum
            weights = [0.2, 0.3, 0.5]  # Weights for each quarter transition
            
            if current_quarter == 2:
                feature_dict['cumulative_momentum'] = feature_dict['q1_to_q2_momentum']
            elif current_quarter == 3:
                feature_dict['cumulative_momentum'] = (
                    feature_dict['q1_to_q2_momentum'] * weights[0] + 
                    feature_dict['q2_to_q3_momentum'] * weights[1]
                ) / (weights[0] + weights[1])
            elif current_quarter >= 4:
                feature_dict['cumulative_momentum'] = (
                    feature_dict['q1_to_q2_momentum'] * weights[0] + 
                    feature_dict['q2_to_q3_momentum'] * weights[1] + 
                    feature_dict['q3_to_q4_momentum'] * weights[2]
                ) / sum(weights)
            
            # Normalize momentum to [-1, 1]
            if feature_dict['cumulative_momentum'] != 0:
                feature_dict['cumulative_momentum'] = max(min(
                    feature_dict['cumulative_momentum'] / 15.0, 1.0), -1.0)
        
        features.append(feature_dict)
    
    # Create DataFrame
    features_df = pd.DataFrame(features)
    
    # Ensure all expected features exist with proper types
    for feature in expected_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
        features_df[feature] = pd.to_numeric(features_df[feature], errors='coerce').fillna(0)
    
    return features_df

def calculate_win_probability(home_score, away_score, quarter):
    """Calculate win probability based on score difference and game progress."""
    score_diff = home_score - away_score
    game_progress = min(quarter / 4.0, 1.0)
    k_factor = 0.05 + (game_progress * 0.15)
    win_prob = 1.0 / (1.0 + np.exp(-k_factor * score_diff))
    return win_prob

def calculate_confidence(quarter):
    """Calculate prediction confidence based on quarter."""
    confidence_map = {0: 30, 1: 45, 2: 65, 3: 80, 4: 95}
    return confidence_map.get(quarter, 30)

def run_predictions():
    """Main function to run in-game predictions."""
    # Step 1: Load the model
    model = load_prediction_model()
    if model is None:
        print("Failed to load prediction model")
        return pd.DataFrame()
    
    # Step 2: Get games for prediction
    games_df = get_games_for_prediction()
    if games_df.empty:
        print("No games available for prediction")
        return pd.DataFrame()
    
    print(f"Running predictions for {len(games_df)} games...")
    
    # Step 3: Prepare features
    features_df = prepare_features(games_df, model)
    if features_df.empty:
        print("Failed to prepare features")
        return pd.DataFrame()
    
    # Step 4: Get the right feature columns for prediction
    if hasattr(model, 'feature_importances_'):
        feature_count = len(model.feature_importances_)
        is_enhanced = (feature_count > 8)
        
        if is_enhanced:
            model_features = [
                'home_q1', 'home_q2', 'home_q3', 'home_q4', 
                'score_ratio', 'prev_matchup_diff',
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away',
                'q1_to_q2_momentum', 'q2_to_q3_momentum', 'q3_to_q4_momentum', 'cumulative_momentum'
            ]
        else:
            model_features = [
                'home_q1', 'home_q2', 'home_q3', 'home_q4', 
                'score_ratio', 'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff'
            ]
    else:
        # Default to original features
        model_features = [
            'home_q1', 'home_q2', 'home_q3', 'home_q4', 
            'score_ratio', 'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff'
        ]
    
    # Step 5: Make predictions
    X_pred = features_df[model_features]
    predictions = model.predict(X_pred)
    
    # Step 6: Create results DataFrame with predictions
    results = []
    
    for i, (idx, game) in enumerate(games_df.iterrows()):
        # Get game info
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        current_quarter = int(game['current_quarter'])
        home_score = float(game['home_score'])
        away_score = float(game['away_score'])
        
        # Get prediction
        predicted_home_score = predictions[i]
        
        # Calculate away score prediction
        prev_matchup_diff = features_df.loc[i, 'prev_matchup_diff']
        
        # Scale effect based on game progress
        diff_weight = min(0.3 + (0.1 * current_quarter), 0.6)
        
        # Factor in momentum if available
        momentum_adj = 0
        if 'cumulative_momentum' in features_df.columns:
            momentum = features_df.loc[i, 'cumulative_momentum']
            momentum_adj = momentum * 3.0  # Scale to points impact
        
        predicted_away_score = predicted_home_score - (prev_matchup_diff * diff_weight) - momentum_adj
        
        # Ensure predictions are at least current scores
        predicted_home_score = max(predicted_home_score, home_score)
        predicted_away_score = max(predicted_away_score, away_score)
        
        # Calculate metrics
        win_probability = calculate_win_probability(predicted_home_score, predicted_away_score, current_quarter)
        confidence = calculate_confidence(current_quarter)
        
        # Create result dictionary
        result = {
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'current_quarter': current_quarter,
            'current_home_score': home_score,
            'current_away_score': away_score,
            'predicted_home_final': predicted_home_score,
            'predicted_away_final': predicted_away_score,
            'remaining_home_points': predicted_home_score - home_score,
            'remaining_away_points': predicted_away_score - away_score,
            'win_probability': win_probability,
            'confidence': confidence,
            'momentum_shift': features_df.loc[i, 'cumulative_momentum'] if 'cumulative_momentum' in features_df.columns else 0,
            'projected_margin': predicted_home_score - predicted_away_score,
            'total_projected_score': predicted_home_score + predicted_away_score,
            'time_remaining': 12 * (4 - current_quarter) if current_quarter <= 4 else 0
        }
        
        # Add actual finals if available (for historical testing)
        if 'actual_home_final' in game:
            result['actual_home_final'] = game['actual_home_final']
            result['actual_away_final'] = game['actual_away_final']
            result['home_prediction_error'] = predicted_home_score - game['actual_home_final']
            result['away_prediction_error'] = predicted_away_score - game['actual_away_final']
        
        results.append(result)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Run predictions
    predictions = run_predictions()
    
    if not predictions.empty:
        print(f"\nPredictions generated for {len(predictions)} games:")
        for _, game in predictions.iterrows():
            print(f"\n{game['away_team']} @ {game['home_team']} - Quarter {game['current_quarter']}")
            print(f"Current Score: {game['away_team']} {game['current_away_score']} - {game['home_team']} {game['current_home_score']}")
            print(f"Predicted Final: {game['away_team']} {game['predicted_away_final']:.1f} - {game['home_team']} {game['predicted_home_final']:.1f}")
            print(f"Win Probability: {game['win_probability']:.1%}, Confidence: {game['confidence']}%")
            
            if 'actual_home_final' in game:
                print(f"Actual Final: {game['away_team']} {game['actual_away_final']} - {game['home_team']} {game['actual_home_final']}")
                print(f"Prediction Error: {game['away_team']} {game['away_prediction_error']:.1f}, {game['home_team']} {game['home_prediction_error']:.1f}")
    else:
        print("No predictions generated")