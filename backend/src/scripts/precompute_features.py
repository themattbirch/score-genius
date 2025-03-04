# File: backend/src/scripts/precompute_features.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import time
from sqlalchemy.exc import OperationalError

def precompute_features(db_connection_string):
    print("Received connection string:", db_connection_string)
    max_retries = 5
    delay = 10  # seconds

    # Try connecting with retries
    for attempt in range(max_retries):
        try:
            engine = create_engine(db_connection_string)
            with engine.connect() as conn:
                print("Connection successful on attempt", attempt + 1)
            break
        except OperationalError as e:
            print(f"Connection attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    else:
        raise OperationalError("Failed to connect after several attempts.", params={}, orig=None)

    # Load raw historical game data
    query = "SELECT * FROM nba_historical_game_stats ORDER BY game_date"
    df = pd.read_sql(query, engine)
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Optional: convert 'minutes' if needed
    if 'minutes' in df.columns:
        def convert_time_to_minutes(time_str):
            if pd.isna(time_str):
                return None
            try:
                minutes, seconds = time_str.split(':')
                return float(minutes) + float(seconds) / 60.0
            except Exception as e:
                print("Error converting time:", e)
                return None
        df['minutes_numeric'] = df['minutes'].apply(convert_time_to_minutes)

    # Compute rolling scores for home and away teams
    df['rolling_home_score'] = df.groupby('home_team')['home_score'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).median())
    df['rolling_away_score'] = df.groupby('away_team')['away_score'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).median())
    df['rolling_home_score'] = df['rolling_home_score'].fillna(df['home_score'].median())
    df['rolling_away_score'] = df['rolling_away_score'].fillna(df['away_score'].median())
    df['score_ratio'] = df.apply(lambda row: row['home_score'] / row['away_score'] if row['away_score'] != 0 else 0, axis=1)

    # Initialize a list to collect feature dictionaries
    feature_data = []

    # Compute features for each game from the home perspective
    home_df = df[df['home_team'].notnull()].copy()
    # (Assuming you have quarter scores 'home_q1', etc., already in df;
    # if not, you’ll need to compute them.)

    # Loop over the home games to compute additional features like prev_matchup_diff
    for idx, row in home_df.iterrows():
        # For example, find previous matchup difference:
        opponent = row['away_team']
        prev_matchup = df[
            (df['game_date'] < row['game_date']) & 
            (((df['home_team'] == row['home_team']) & (df['away_team'] == opponent)) |
             ((df['home_team'] == opponent) & (df['away_team'] == row['home_team'])))
        ].sort_values('game_date', ascending=False)
        if not prev_matchup.empty:
            recent = prev_matchup.iloc[0]
            if recent['home_team'] == row['home_team']:
                matchup_diff = recent['home_score'] - recent['away_score']
            else:
                matchup_diff = recent['away_score'] - recent['home_score']
        else:
            matchup_diff = 0

        # Append a dictionary for this game with all expected features.
        feature_data.append({
            'game_id': row['game_id'],
            # For quarter scores, if they exist; otherwise, default to 0.
            'home_q1': row.get('home_q1', 0),
            'home_q2': row.get('home_q2', 0),
            'home_q3': row.get('home_q3', 0),
            'home_q4': row.get('home_q4', 0),
            'rolling_home_score': row['rolling_home_score'],
            'rolling_away_score': row['rolling_away_score'],
            'score_ratio': row['score_ratio'],
            'prev_matchup_diff': matchup_diff
        })

    # Create a DataFrame from the computed feature data
    features_df = pd.DataFrame(feature_data)

    # Optionally, check for any missing expected columns and fill with 0.
    expected_features = ['home_q1', 'home_q2', 'home_q3', 'home_q4', 
                         'rolling_home_score', 'rolling_away_score', 'score_ratio', 'prev_matchup_diff']
    for col in expected_features:
        if col not in features_df.columns:
            features_df[col] = 0

    # Save the features DataFrame to the database
    features_df.to_sql('nba_precomputed_features', engine, if_exists='replace', index=False, schema='public')
    print("Saved {} precomputed feature records to database".format(len(features_df)))

    return features_df

