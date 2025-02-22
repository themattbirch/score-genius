import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def load_quarter_data():
   """
   Loads historical game data with quarter statistics.
   """
   data_path = os.path.join(os.path.dirname(__file__), '../../data/historical_games.csv')
   df = pd.read_csv(data_path)
   return df

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
   
   # 1. Halftime adjustment model
   df['first_half_diff'] = (df['home_q1'] + df['home_q2']) - (df['away_q1'] + df['away_q2'])
   df['third_q_diff'] = df['home_q3'] - df['away_q3']
   X = df[['first_half_diff']]
   y = df['third_q_diff']
   models['halftime_adjustment'] = LinearRegression().fit(X, y)
   
   # 2. Closing momentum model
   df['pre_q4_diff'] = (df['home_q1'] + df['home_q2'] + df['home_q3']) - (df['away_q1'] + df['away_q2'] + df['away_q3'])
   df['q4_diff'] = df['home_q4'] - df['away_q4']
   X = df[['pre_q4_diff']]
   y = df['q4_diff']
   models['closing_momentum'] = LinearRegression().fit(X, y)
   
   # 3. Original Q3 momentum model
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
   
   # Ensure required precomputed columns exist
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
       # Base prediction combines team offensive rating and average quarter score
       home_q_pred = team_stats.get('home_off_rating', 25) * 0.7 + quarter_scores.get('home_avg', 25) * 0.3
       away_q_pred = team_stats.get('away_off_rating', 25) * 0.7 + quarter_scores.get('away_avg', 25) * 0.3
       
       # Quarter-specific adjustments
       if q == 3:
           # Home teams often perform better after halftime
           home_q_pred += 0.5
       if q == 4:
           # For fourth quarter, if the game is close, both teams may score more
           home_total = sum(quarter_scores.get(f'home_q{i}', 0) for i in range(1, current_quarter+1))
           away_total = sum(quarter_scores.get(f'away_q{i}', 0) for i in range(1, current_quarter+1))
           score_diff = abs(home_total - away_total)
           if score_diff < 10:
               # Apply home/away split
               if team_stats.get('is_home', True):
                   home_q_pred += 2   # Home team bonus for close game
               else:
                   away_q_pred += 1   # Away team gets a smaller boost
       
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