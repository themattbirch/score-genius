# File: backend/src/scripts/precompute_features.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Integer, Float, String
from datetime import datetime
import time
from sqlalchemy.exc import OperationalError

def convert_time_to_minutes(time_str):
    """
    Converts a time string of the form "MM:SS" into a float representing total minutes.
    For example, "36:25" becomes 36 + 25/60.
    """
    if pd.isna(time_str) or ":" not in str(time_str):
        return None
    try:
        minutes, seconds = str(time_str).split(":")
        return float(minutes) + float(seconds) / 60.0
    except Exception as e:
        print("Error converting time:", e)
        return None

def precompute_features(db_connection_string):
    try:
        print("Connecting to database...")
        engine = create_engine(db_connection_string)
    
        # Load raw historical game data
        query = "SELECT * FROM nba_historical_game_stats ORDER BY game_date"
        df = pd.read_sql(query, engine)
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        print(f"Loaded {len(df)} historical games")
        
        # Convert 'minutes' column to a numeric value and drop the raw 'minutes' column
        if 'minutes' in df.columns:
            df['minutes_numeric'] = df['minutes'].apply(convert_time_to_minutes)
            df = df.drop(columns=['minutes'])
        
        # Sort the DataFrame by game_date for accurate rolling calculations
        df = df.sort_values('game_date')
        
        # Compute rolling averages for home and away teams (using previous 5 games)
        df['rolling_home_score'] = df.groupby('home_team')['home_score'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        df['rolling_away_score'] = df.groupby('away_team')['away_score'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        
        # Fill NaN values in rolling averages using team averages or overall averages
        overall_home_avg = df['home_score'].mean()
        overall_away_avg = df['away_score'].mean()
        home_team_avgs = df.groupby('home_team')['home_score'].mean().to_dict()
        away_team_avgs = df.groupby('away_team')['away_score'].mean().to_dict()
        
        df['rolling_home_score'] = df.apply(
            lambda row: row['rolling_home_score'] if not pd.isna(row['rolling_home_score'])
            else home_team_avgs.get(row['home_team'], overall_home_avg),
            axis=1
        )
        df['rolling_away_score'] = df.apply(
            lambda row: row['rolling_away_score'] if not pd.isna(row['rolling_away_score'])
            else away_team_avgs.get(row['away_team'], overall_away_avg),
            axis=1
        )
        
        # Compute score ratio as home_score / (home_score + away_score)
        df['score_ratio'] = df.apply(
            lambda row: row['home_score'] / (row['home_score'] + row['away_score'])
            if (row['home_score'] + row['away_score']) > 0 else 0.5,
            axis=1
        )
        
        # Calculate previous matchup differentials
        def get_matchup_diff(row):
            teams = tuple(sorted([row['home_team'], row['away_team']]))
            curr_date = row['game_date']
            home_team = row['home_team']
            # Look for previous games between these two teams before the current game
            prev_matchups = df[
                (df['game_date'] < curr_date) &
                (
                    ((df['home_team'] == teams[0]) & (df['away_team'] == teams[1])) |
                    ((df['home_team'] == teams[1]) & (df['away_team'] == teams[0]))
                )
            ].sort_values('game_date', ascending=False)
            
            if len(prev_matchups) > 0:
                recent = prev_matchups.iloc[0]
                if recent['home_team'] == home_team:
                    return recent['home_score'] - recent['away_score']
                else:
                    return recent['away_score'] - recent['home_score']
            return 0
        
        df['prev_matchup_diff'] = df.apply(get_matchup_diff, axis=1)
        
        # Print some diagnostic information about matchup differentials
        print("Unique prev_matchup_diff values:", df['prev_matchup_diff'].unique())
        print("Number of non-zero values:", (df['prev_matchup_diff'] != 0).sum())
        
        # Select the features to be used for the model
        feature_cols = [
            'game_id', 'home_q1', 'home_q2', 'home_q3', 'home_q4',
            'score_ratio', 'rolling_home_score', 'rolling_away_score',
            'prev_matchup_diff'
        ]
        features_df = df[feature_cols].copy()
        
        # Save the features DataFrame to the database
        features_df.to_sql(
            'nba_precomputed_features', 
            engine, 
            if_exists='replace',
            index=False,
            schema='public',
            dtype={
                'prev_matchup_diff': Float,
                'score_ratio': Float,
                'rolling_home_score': Float,
                'rolling_away_score': Float
            }
        )
        
        print("Feature precomputation completed successfully")
        return features_df
    except Exception as e:
        print(f"Error during feature precomputation: {e}")
        raise  # Re-raise to ensure the scheduler knows the job failed

if __name__ == "__main__":
    # Replace with your actual connection string
    DB_CONNECTION = "your_connection_string_here"
    precompute_features(DB_CONNECTION)