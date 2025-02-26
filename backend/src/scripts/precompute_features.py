# backend/src/scripts/precompute_features.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime

def precompute_features(db_connection_string):
    """
    Precomputes and stores all derived features used by prediction models.
    This should be run after new game data is ingested.
    """
    # Connect to database
    engine = create_engine(db_connection_string)
    
    # Load raw game data
    query = "SELECT * FROM nba_historical_game_stats ORDER BY game_date"
    df = pd.read_sql(query, engine)
    
    # Ensure game_date is datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Compute rolling averages and other metrics per team
    print("Computing rolling averages...")
    team_list = list(set(df['home_team'].tolist() + df['away_team'].tolist()))
    
    feature_data = []
    
    for team in team_list:
        # Get all games where the team played (home or away)
        home_games = df[df['home_team'] == team].sort_values('game_date')
        away_games = df[df['away_team'] == team].sort_values('game_date')
        
        # Process home games
        for i, row in home_games.iterrows():
            # Get previous 5 games (home or away)
            prev_home = home_games[home_games['game_date'] < row['game_date']].tail(5)
            prev_away = away_games[away_games['game_date'] < row['game_date']].tail(5)
            prev_games = pd.concat([prev_home, prev_away]).sort_values('game_date').tail(5)
            
            if not prev_games.empty:
                rolling_pts = prev_games.apply(
                    lambda x: x['home_score'] if x['home_team'] == team else x['away_score'], 
                    axis=1
                ).mean()
                
                # Calculate efficiency (points per possession estimated)
                possessions = row['home_score'] + row['home_turnovers'] - row['home_off_reb']
                efficiency = row['home_score'] / possessions if possessions > 0 else 0
                
                # Find previous matchup with opponent
                opponent = row['away_team']
                prev_matchup = df[
                    (df['game_date'] < row['game_date']) & 
                    (((df['home_team'] == team) & (df['away_team'] == opponent)) |
                     ((df['home_team'] == opponent) & (df['away_team'] == team)))
                ].sort_values('game_date', ascending=False)
                
                matchup_diff = 0
                if not prev_matchup.empty:
                    recent = prev_matchup.iloc[0]
                    if recent['home_team'] == team:
                        matchup_diff = recent['home_score'] - recent['away_score']
                    else:
                        matchup_diff = recent['away_score'] - recent['home_score']
                
                feature_data.append({
                    'game_id': row['game_id'],
                    'team': team,
                    'is_home': 1,
                    'rolling_pts_5': rolling_pts,
                    'efficiency': efficiency,
                    'prev_matchup_diff': matchup_diff,
                    'updated_at': datetime.now()
                })
        
        # Similar processing for away games could be added here if needed.
    
    # Create DataFrame from computed features and store in a new table
    features_df = pd.DataFrame(feature_data)
    features_df.to_sql('nba_precomputed_features', engine, if_exists='replace', index=False)
    print(f"Saved {len(features_df)} precomputed feature records to database")

if __name__ == "__main__":
    DB_CONNECTION = "your_connection_string_here"  # Replace with your actual connection string
    precompute_features(DB_CONNECTION)
