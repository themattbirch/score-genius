# backend/models/quarter_analysis.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def load_quarter_data():
    """
    Loads historical game data with quarter statistics.
    """
    data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_games.csv')
    df = pd.read_csv(data_path)
    return df

def create_team_quarter_profiles(df):
    """
    Create quarter scoring profiles for each team, incorporating home, away, and overall splits.
    
    Returns:
        dict: Mapping team names to their quarter tendencies.
    """
    team_profiles = {}
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    for team in teams:
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]
        
        home_q_avgs = {
            'q1_avg': home_games['home_q1'].mean(),
            'q2_avg': home_games['home_q2'].mean(),
            'q3_avg': home_games['home_q3'].mean(),
            'q4_avg': home_games['home_q4'].mean(),
            'q3_vs_q1_diff': home_games['home_q3'].mean() - home_games['home_q1'].mean(),
            'q4_vs_q3_diff': home_games['home_q4'].mean() - home_games['home_q3'].mean(),
        }
        
        away_q_avgs = {
            'q1_avg': away_games['away_q1'].mean(),
            'q2_avg': away_games['away_q2'].mean(),
            'q3_avg': away_games['away_q3'].mean(),
            'q4_avg': away_games['away_q4'].mean(),
            'q3_vs_q1_diff': away_games['away_q3'].mean() - away_games['away_q1'].mean(),
            'q4_vs_q3_diff': away_games['away_q4'].mean() - away_games['away_q3'].mean(),
        }
        
        overall_q1 = pd.concat([home_games['home_q1'], away_games['away_q1']]).mean()
        overall_q2 = pd.concat([home_games['home_q2'], away_games['away_q2']]).mean()
        overall_q3 = pd.concat([home_games['home_q3'], away_games['away_q3']]).mean()
        overall_q4 = pd.concat([home_games['home_q4'], away_games['away_q4']]).mean()
        overall = {
            'q1_avg': overall_q1,
            'q2_avg': overall_q2,
            'q3_avg': overall_q3,
            'q4_avg': overall_q4,
            'home_advantage_q1': home_games['home_q1'].mean() - away_games['away_q1'].mean() if not away_games.empty else None,
            'home_advantage_q2': home_games['home_q2'].mean() - away_games['away_q2'].mean() if not away_games.empty else None,
            'home_advantage_q3': home_games['home_q3'].mean() - away_games['away_q3'].mean() if not away_games.empty else None,
            'home_advantage_q4': home_games['home_q4'].mean() - away_games['away_q4'].mean() if not away_games.empty else None,
        }
        
        team_profiles[team] = {
            'home': home_q_avgs,
            'away': away_q_avgs,
            'overall': overall
        }
    
    return team_profiles

def analyze_rest_impact(df):
    """
    Analyze how rest days affect quarter performance.
    
    Returns:
        dict: Quarter differentials and a refined fatigue factor grouped by rest type.
    """
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    df['rest_days'] = df['game_date'].diff().dt.days
    
    rest_groups = {
        'back_to_back': df[df['rest_days'] == 1],
        'one_day_rest': df[(df['rest_days'] > 1) & (df['rest_days'] <= 2)],
        'multiple_days_rest': df[df['rest_days'] > 2]
    }
    
    rest_impact = {}
    for rest_type, games in rest_groups.items():
        q4_diff = games['home_q4'].mean() - games['away_q4'].mean()
        q1_diff = games['home_q1'].mean() - games['away_q1'].mean()
        
        if q1_diff != 0:
            fatigue_change = (q4_diff - q1_diff) / abs(q1_diff)
            # Cap the fatigue change between -0.5 and 0.5
            fatigue_change = max(min(fatigue_change, 0.5), -0.5)
            fatigue_factor = 1 + fatigue_change
        else:
            fatigue_factor = 1.0
        
        rest_impact[rest_type] = {
            'q4_vs_q1_diff': q4_diff - q1_diff,
            'q4_avg': games['home_q4'].mean(),
            'fatigue_factor': fatigue_factor
        }
    
    return rest_impact

def analyze_quarter_differences(df):
    """
    Computes the average scoring difference for each quarter across the dataset.
    """
    quarters = ['q1', 'q2', 'q3', 'q4']
    quarter_diff = {}
    for q in quarters:
        home_avg = df[f'home_{q}'].mean()
        away_avg = df[f'away_{q}'].mean()
        quarter_diff[q] = home_avg - away_avg
    return quarter_diff

def train_momentum_models(df):
    """
    Train models for various momentum indicators.
    """
    models = {}
    
    # Halftime adjustment model: predict Q3 differential from first half differential
    df['first_half_diff'] = (df['home_q1'] + df['home_q2']) - (df['away_q1'] + df['away_q2'])
    df['third_q_diff'] = df['home_q3'] - df['away_q3']
    X = df[['first_half_diff']]
    y = df['third_q_diff']
    models['halftime_adjustment'] = LinearRegression().fit(X, y)
    
    # Closing momentum model: predict Q4 differential from pre-Q4 differential
    df['pre_q4_diff'] = (df['home_q1'] + df['home_q2'] + df['home_q3']) - (df['away_q1'] + df['away_q2'] + df['away_q3'])
    df['q4_diff'] = df['home_q4'] - df['away_q4']
    X = df[['pre_q4_diff']]
    y = df['q4_diff']
    models['closing_momentum'] = LinearRegression().fit(X, y)
    
    # Q3 momentum model: predict Q3 differential using away Q3 score
    X = df[['away_q3']]
    y = df['home_q3'] - df['away_q3']
    models['q3_momentum'] = LinearRegression().fit(X, y)
    
    return models

def identify_momentum_shifts(df):
    """
    Detect game situations that typically lead to momentum shifts.
    """
    # Calculate quarter-to-quarter shifts
    for i in range(1, 4):
        df[f'home_q{i}_to_q{i+1}_shift'] = df[f'home_q{i+1}'] - df[f'home_q{i}']
        df[f'away_q{i}_to_q{i+1}_shift'] = df[f'away_q{i+1}'] - df[f'away_q{i}']
    
    # Ensure required columns exist
    if 'first_half_diff' not in df.columns:
        df['first_half_diff'] = (df['home_q1'] + df['home_q2']) - (df['away_q1'] + df['away_q2'])
    if 'pre_q4_diff' not in df.columns:
        df['pre_q4_diff'] = (df['home_q1'] + df['home_q2'] + df['home_q3']) - (df['away_q1'] + df['away_q2'] + df['away_q3'])
    
    shifts = {
        'halftime_boost': df.groupby('home_team')['home_q2_to_q3_shift'].mean(),
        'comeback_quarters': df[df['pre_q4_diff'] < -5]['q4_diff'].mean(),
        'blowout_expansion': df[df['first_half_diff'] > 15]['third_q_diff'].mean()
    }
    return shifts

def predict_remaining_quarters(current_quarter, quarter_scores, team_stats):
    """
    Predict scores for the remaining quarters of a game.
    """
    predictions = {}
    for q in range(current_quarter + 1, 5):
        # Base prediction: weighted combination of team offensive rating and historical quarter average
        home_q_pred = team_stats.get('home_off_rating', 25) * 0.7 + quarter_scores.get('home_avg', 25) * 0.3
        away_q_pred = team_stats.get('away_off_rating', 25) * 0.7 + quarter_scores.get('away_avg', 25) * 0.3
        
        # Quarter-specific adjustments
        if q == 3:
            home_q_pred += 0.5
        if q == 4:
            home_total = sum(quarter_scores.get(f'home_q{i}', 0) for i in range(1, current_quarter+1))
            away_total = sum(quarter_scores.get(f'away_q{i}', 0) for i in range(1, current_quarter+1))
            score_diff = abs(home_total - away_total)
            if score_diff < 10:
                if team_stats.get('is_home', True):
                    home_q_pred += 2
                else:
                    away_q_pred += 1
        predictions[f'home_q{q}'] = home_q_pred
        predictions[f'away_q{q}'] = away_q_pred
    return predictions

def predict_q3_momentum(model, current_away_q3):
    """
    Predict the Q3 momentum shift using the trained model.
    """
    prediction = model.predict([[current_away_q3]])
    return prediction[0]

if __name__ == "__main__":
    # Load data
    df = load_quarter_data()
    
    # Analyze quarter differences
    quarter_differences = analyze_quarter_differences(df)
    print("Average Quarter Differences:", quarter_differences)
    
    # Create team quarter profiles with home/away splits
    team_profiles = create_team_quarter_profiles(df)
    print("Team Quarter Profiles:")
    print(team_profiles)
    
    # Analyze rest impact with refined fatigue factor
    rest_impact = analyze_rest_impact(df)
    print("Rest Impact Analysis:")
    print(rest_impact)
    
    # Train momentum models
    momentum_models = train_momentum_models(df)
    
    # Identify momentum shifts
    shifts = identify_momentum_shifts(df)
    print("Momentum Shifts:", shifts)
    
    # Example: Predict Q3 momentum
    predicted_q3_diff = predict_q3_momentum(momentum_models['q3_momentum'], 28)
    print(f"Predicted Q3 momentum for away score 28: {predicted_q3_diff:.2f}")
    
    # Example: Predict remaining quarters
    current_quarter = 2
    quarter_scores = {
        'home_q1': 28, 'home_q2': 27,
        'away_q1': 27, 'away_q2': 26,
        'home_avg': 28, 'away_avg': 27
    }
    team_stats = {
        'home_off_rating': 30,
        'away_off_rating': 29,
        'is_home': True
    }
    predictions = predict_remaining_quarters(current_quarter, quarter_scores, team_stats)
    print("Predicted remaining quarters:", predictions)