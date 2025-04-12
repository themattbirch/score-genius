# backend/src/scripts/integrate_predictions.py

import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback
from datetime import datetime

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

import config
from caching.supabase_client import supabase

# Use relative imports with direct function calls 
from .run_predictions import run_predictions  # Use dot-prefix for relative import

def ensure_model_loaded():
    """
    Ensures that the prediction model is loaded.
    """
    try:
        # First check if model is in globals
        if 'model' in globals() and globals()['model'] is not None:
            return globals()['model']
        
        # Try to load from config path
        model_path = config.MODEL_PATH
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Store in globals for future use
            globals()['model'] = model
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"Model file not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def get_team_rolling_averages(days_lookback=60):
    """
    Retrieves the rolling scoring average for each team from historical data.
    
    Args:
        days_lookback: Number of days to look back for calculating the average
        
    Returns:
        Dictionary mapping team names to their rolling scoring average
    """
    from datetime import datetime, timedelta
    
    # Calculate the date threshold
    threshold_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    
    try:
        # Fetch recent historical game data
        response = supabase.table("nba_historical_game_stats").select("*").gte("game_date", threshold_date).execute()
        historical_data = response.data
        
        if not historical_data:
            print(f"No historical game data available from the last {days_lookback} days.")
            return {}
        
        df = pd.DataFrame(historical_data)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values('game_date')
        
        # Initialize dictionary for team averages
        team_avgs = {}
        
        # Get unique teams
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in all_teams:
            # Get home games where team is home
            home_games = df[df['home_team'] == team][['game_date', 'home_score']].rename(
                columns={'home_score': 'score'})
            
            # Get away games where team is away
            away_games = df[df['away_team'] == team][['game_date', 'away_score']].rename(
                columns={'away_score': 'score'})
            
            # Combine all games
            team_games = pd.concat([home_games, away_games]).sort_values('game_date')
            
            if not team_games.empty:
                # Calculate recent average (last 5 games if available)
                recent_games = team_games.tail(5)
                team_avgs[team] = recent_games['score'].mean()
            else:
                # Fallback to a reasonable default
                team_avgs[team] = 105.0  # NBA average is approximately 100-110 points per game
        
        return team_avgs
    except Exception as e:
        print(f"Error getting team rolling averages: {e}")
        traceback.print_exc()
        return {}

def get_previous_matchup_diff(home_team, away_team, max_lookback=5):
    """Gets the point differential from previous matchups between two teams."""
    try:
        # Use separate queries for home and away configurations to avoid syntax issues
        response_home = supabase.table("nba_historical_game_stats").select("*")\
            .eq("home_team", home_team)\
            .eq("away_team", away_team)\
            .order('game_date', desc=True)\
            .limit(max_lookback).execute()
            
        response_away = supabase.table("nba_historical_game_stats").select("*")\
            .eq("home_team", away_team)\
            .eq("away_team", home_team)\
            .order('game_date', desc=True)\
            .limit(max_lookback).execute()
        
        # Combine results
        home_matchups = response_home.data
        away_matchups = response_away.data
        matchups = home_matchups + away_matchups
        
        # Sort by date (most recent first)
        if matchups:
            matchups.sort(key=lambda x: x['game_date'], reverse=True)
            matchups = matchups[:max_lookback]
        
        if not matchups:
            return 0
        
        # Calculate point differential from home team perspective
        differentials = []
        for game in matchups:
            if game['home_team'] == home_team and game['away_team'] == away_team:
                diff = game['home_score'] - game['away_score']
            elif game['home_team'] == away_team and game['away_team'] == home_team:
                diff = game['away_score'] - game['home_score']
            else:
                continue
            differentials.append(diff)
        
        return sum(differentials) / len(differentials) if differentials else 0
    except Exception as e:
        print(f"Error getting previous matchups: {e}")
        return 0

def normalize_team_name(name):
    """
    Normalize a team name for consistent comparison.
    """
    if not name:
        return ""
    
    # Common name variations and abbreviations
    name_map = {
        "sixers": "philadelphia 76ers",
        "76ers": "philadelphia 76ers",
        "blazers": "portland trail blazers",
        "trailblazers": "portland trail blazers",
        "cavs": "cleveland cavaliers",
        "mavs": "dallas mavericks",
        "knicks": "new york knicks", 
        "nets": "brooklyn nets",
        "lakers": "los angeles lakers",
        "clippers": "los angeles clippers",
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "warriors": "golden state warriors",
        "timberwolves": "minnesota timberwolves",
        "t-wolves": "minnesota timberwolves",
        "nuggets": "denver nuggets"
    }
    
    # Convert to lowercase for case-insensitive matching
    name_lower = name.lower()
    
    # Direct mapping for known variations
    for key, value in name_map.items():
        if key in name_lower:
            return value
    
    # Remove common suffixes
    import re
    cleaned = re.sub(r'\b(the|basketball|club|team)\b', '', name_lower, flags=re.IGNORECASE)
    
    # Remove extra spaces and trim
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def run_live_inference():
    """
    Main function to run in-game prediction.
    """
    print("Running prediction pipeline...")
    
    # Run the prediction pipeline using the new unified script
    predictions = run_predictions()
    
    if isinstance(predictions, pd.DataFrame) and predictions.empty:
        print("No predictions generated.")
        return None
    
    print(f"Generated predictions for {len(predictions)} games.")
    return predictions

if __name__ == "__main__":
    # Test the prediction integration
    print("Testing prediction integration...")
    
    # Generate predictions
    predictions = run_live_inference()
    
    if predictions is not None and not predictions.empty:
        print(f"\nPredictions generated for {len(predictions)} games:")
        for _, game in predictions.iterrows():
            print(f"\n{game['away_team']} vs {game['home_team']} - Quarter {game['current_quarter']}")
            print(f"Current Score: {game['away_team']} {game['current_away_score']} - {game['home_team']} {game['current_home_score']}")
            print(f"Predicted Final: {game['away_team']} {game['predicted_away_final']:.1f} - {game['home_team']} {game['predicted_home_final']:.1f}")
            print(f"Win Probability: {game['win_probability']:.1%}")
            
            if 'momentum_shift' in game:
                print(f"Momentum Shift: {game['momentum_shift']:.2f}")
    else:
        print("No predictions generated.")