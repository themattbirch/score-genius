# File: backend/src/scripts/precompute_features.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Integer, Float, String
from datetime import datetime, timedelta
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

def calculate_rest_days(df, team_column, date_column='game_date'):
    """
    Calculate rest days between games for each team
    
    Args:
        df: DataFrame with game data
        team_column: Column with team identifiers
        date_column: Column with game dates
        
    Returns:
        Series with rest days
    """
    # Sort by team and date
    tmp = df.sort_values([team_column, date_column])
    
    # Calculate days between consecutive games for the same team
    tmp['prev_game_date'] = tmp.groupby(team_column)[date_column].shift(1)
    tmp['rest_days'] = (tmp[date_column] - tmp['prev_game_date']).dt.days
    
    # Fill first game of season with average rest (7 days is reasonable for season start)
    tmp['rest_days'] = tmp['rest_days'].fillna(7)
    
    # Constrain to reasonable values (1-15 days)
    tmp['rest_days'] = tmp['rest_days'].clip(1, 15)
    
    return tmp['rest_days']

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
        
        # Calculate rest days for home and away teams
        print("Calculating rest days...")
        home_rest_days = calculate_rest_days(df, 'home_team')
        away_rest_days = calculate_rest_days(df, 'away_team')
        
        df['home_rest_days'] = home_rest_days
        df['away_rest_days'] = away_rest_days
        
        # Calculate win percentages for each team (last 10 games)
        print("Calculating win percentages...")
        
        # Create team results for both home and away games
        home_results = df[['game_date', 'home_team', 'home_score', 'away_score']].copy()
        home_results['team'] = home_results['home_team']
        home_results['win'] = (home_results['home_score'] > home_results['away_score']).astype(int)
        
        away_results = df[['game_date', 'away_team', 'home_score', 'away_score']].copy()
        away_results['team'] = away_results['away_team']
        away_results['win'] = (away_results['away_score'] > away_results['home_score']).astype(int)
        
        # Combine and sort by date
        team_results = pd.concat([home_results, away_results], ignore_index=True)
        team_results = team_results.sort_values(['team', 'game_date'])
        
        # Calculate rolling win percentage (last 10 games)
        team_results['win_pct'] = team_results.groupby('team')['win'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean())
        
        # Map win percentages back to original dataframe
        win_pct_map = team_results.set_index(['team', 'game_date'])['win_pct'].to_dict()
        
        def get_win_pct(row, team_col):
            team = row[team_col]
            date = row['game_date']
            return win_pct_map.get((team, date), 0.5)  # Default to 0.5 if no data
        
        df['home_win_pct'] = df.apply(lambda row: get_win_pct(row, 'home_team'), axis=1)
        df['away_win_pct'] = df.apply(lambda row: get_win_pct(row, 'away_team'), axis=1)
        
        # Calculate points per game (PPG) metrics
        print("Calculating points per game metrics...")
        
        # Home team PPG (last 10 games)
        df['home_ppg'] = df.groupby('home_team')['home_score'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        
        # Away team PPG (last 10 games)
        df['away_ppg'] = df.groupby('away_team')['away_score'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        
        # Calculate points allowed per game (PAPG)
        df['home_papg'] = df.groupby('home_team')['away_score'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        
        df['away_papg'] = df.groupby('away_team')['home_score'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        
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
        
        # Fill NaN values for all computed metrics
        for metric in ['rolling_home_score', 'rolling_away_score', 'home_ppg', 'away_ppg', 
                       'home_papg', 'away_papg', 'home_win_pct', 'away_win_pct']:
            if metric in df.columns:
                df[metric] = df[metric].fillna(df[metric].mean())
        
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
        
        # List of ALL features needed for the model
        feature_cols = [
            # Original features
            'game_id', 'home_q1', 'home_q2', 'home_q3', 'home_q4',
            'score_ratio', 'rolling_home_score', 'rolling_away_score',
            'prev_matchup_diff',
            # Added features required by the model
            'home_rest_days', 'away_rest_days', 
            'home_win_pct', 'away_win_pct',
            'home_ppg', 'away_ppg', 
            'home_papg', 'away_papg'
        ]
        
        # Ensure all features are present in the dataframe
        for col in feature_cols:
            if col not in df.columns:
                print(f"Warning: Feature '{col}' not found in dataframe. Using placeholder values.")
                df[col] = 0
        
        # Select the features to be used for the model
        features_df = df[feature_cols].copy()
        
        # Print feature statistics
        print("\nFeature statistics:")
        for col in feature_cols:
            if col != 'game_id':  # Skip non-numeric columns
                try:
                    print(f"{col}: mean={features_df[col].mean():.2f}, std={features_df[col].std():.2f}")
                except:
                    print(f"{col}: non-numeric or missing")
        
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
                'rolling_away_score': Float,
                'home_rest_days': Float,
                'away_rest_days': Float,
                'home_win_pct': Float,
                'away_win_pct': Float,
                'home_ppg': Float,
                'away_ppg': Float,
                'home_papg': Float,
                'away_papg': Float
            }
        )
        
        print("Feature precomputation completed successfully")
        print(f"Generated {len(features_df)} rows with {len(feature_cols)} features")
        return features_df
    except Exception as e:
        print(f"Error during feature precomputation: {e}")
        raise  # Re-raise to ensure the scheduler knows the job failed

if __name__ == "__main__":
    # Load DATABASE_URL from .env file
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the DATABASE_URL from environment variables
    DB_CONNECTION = os.getenv("DATABASE_URL")
    
    if not DB_CONNECTION:
        print("Error: DATABASE_URL environment variable not found. Please check your .env file.")
        exit(1)
    
    print(f"Using database connection from .env file")
    precompute_features(DB_CONNECTION)