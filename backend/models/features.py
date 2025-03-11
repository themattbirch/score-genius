# backend/models/features.py

"""
NBAFeatureGenerator - Unified module for NBA score prediction feature engineering.
Centralizes all feature generation previously spread across multiple notebook cells.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import time
from functools import wraps

def profile_time(func):
    """Decorator to profile function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class NBAFeatureGenerator:
    """
    Unified class for generating all features required by NBA score prediction models.
    Centralizes calculations previously spread across multiple notebook cells.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the feature generator.
        
        Args:
            debug (bool): Whether to output detailed logs.
        """
        self.debug = debug
        
    def log(self, message):
        """Simple logging function."""
        if self.debug:
            print(f"[NBAFeatureGenerator] {message}")
    
    @profile_time
    def generate_all_features(self, df, skip_rest_calc=False):
        """
        Master function that generates all features needed by any model in the system.
        
        Args:
            df (DataFrame): Input raw game data.
            skip_rest_calc (bool): Whether to skip rest calculations (for testing purposes).
            
        Returns:
            DataFrame: DataFrame with all computed features.
        """
        try:
            if df is None or df.empty:
                self.log("Empty DataFrame provided")
                return pd.DataFrame()
                
            self.log(f"Generating features for {len(df)} games")
            result_df = df.copy()
            
            # 1. Ensure basic columns exist and are properly typed.
            result_df = self.ensure_data_types(result_df)
            
            # 2. Compute basic features (current scores, score ratio, rolling averages).
            result_df = self.compute_basic_features(result_df)
            
            # 3. Calculate rest-related features.
            if not skip_rest_calc:
                result_df = self.calculate_rest_features(result_df)
            else:
                self.log("Skipping rest calculations")
                result_df['rest_days_home'] = 2
                result_df['rest_days_away'] = 2
                result_df['is_back_to_back_home'] = 0
                result_df['is_back_to_back_away'] = 0
                result_df['rest_advantage'] = 0
            
            # 4. Compute momentum-related features.
            result_df = self.compute_momentum_features(result_df)
            
            # 5. Compute matchup history features.
            result_df = self.compute_matchup_features(result_df)
            
            # 6. Compute quarter-specific features.
            result_df = self.compute_quarter_specific_features(result_df)
            
            # 7. Validate all features.
            result_df = self.validate_features(result_df)
            
            # 8. Add time remaining and interaction features
            result_df = self.add_advanced_features(result_df)
            
            self.log(f"Feature generation complete: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df
            
        except Exception as e:
            self.log(f"Error in generate_all_features: {e}")
            traceback.print_exc()
            return df
    
    def ensure_data_types(self, df):
        """
        Ensure that critical columns exist and have correct data types.
        """
        result_df = df.copy()
        
        # Convert game_date to datetime.
        if 'game_date' in result_df.columns:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
        
        # Ensure quarter columns are numeric.
        for team in ['home', 'away']:
            for q in range(1, 5):
                col = f'{team}_q{q}'
                if col in result_df.columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
        # Ensure current_quarter is an integer.
        if 'current_quarter' in result_df.columns:
            result_df['current_quarter'] = pd.to_numeric(result_df['current_quarter'], errors='coerce').fillna(0).astype(int)
        else:
            result_df['current_quarter'] = 0
            for idx, row in result_df.iterrows():
                for q in range(4, 0, -1):
                    if (float(row.get(f'home_q{q}', 0)) > 0) or (float(row.get(f'away_q{q}', 0)) > 0):
                        result_df.at[idx, 'current_quarter'] = q
                        break
        
        # Ensure team names are strings.
        for col in ['home_team', 'away_team']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
        
        return result_df
    
    def compute_basic_features(self, df):
        """
        Compute basic game state features such as current scores, score ratio, and rolling averages.
        """
        result_df = df.copy()
        
        # Calculate current scores.
        if 'home_score' not in result_df.columns or result_df['home_score'].isna().any():
            result_df['home_score'] = 0
            for q in range(1, 5):
                col = f'home_q{q}'
                if col in result_df.columns:
                    result_df['home_score'] += result_df[col].fillna(0)
        if 'away_score' not in result_df.columns or result_df['away_score'].isna().any():
            result_df['away_score'] = 0
            for q in range(1, 5):
                col = f'away_q{q}'
                if col in result_df.columns:
                    result_df['away_score'] += result_df[col].fillna(0)
        
        # Compute score ratio.
        result_df['score_ratio'] = 0.5
        mask = (result_df['home_score'] + result_df['away_score']) > 0
        if mask.any():
            result_df.loc[mask, 'score_ratio'] = (
                result_df.loc[mask, 'home_score'] / 
                (result_df.loc[mask, 'home_score'] + result_df.loc[mask, 'away_score'])
            )
        
        # Compute rolling averages if not already present.
        if 'rolling_home_score' not in result_df.columns or 'rolling_away_score' not in result_df.columns:
            result_df = self.compute_rolling_averages(result_df)
        
        return result_df
    
    def compute_rolling_averages(self, df, window_size=5):
        """
        Compute rolling averages for home and away scores.
        """
        result_df = df.copy()
        
        if 'game_date' in result_df.columns:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
            result_df = result_df.sort_values('game_date')
        
        # Default league average.
        result_df['rolling_home_score'] = 105.0
        result_df['rolling_away_score'] = 105.0
        
        try:
            self.log(f"Computing rolling averages with window={window_size}")
            home_avgs = result_df.groupby('home_team')['home_score'].transform(
                lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
            )
            away_avgs = result_df.groupby('away_team')['away_score'].transform(
                lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
            )
            mask_home = ~home_avgs.isna()
            mask_away = ~away_avgs.isna()
            if mask_home.any():
                result_df.loc[mask_home, 'rolling_home_score'] = home_avgs[mask_home]
            if mask_away.any():
                result_df.loc[mask_away, 'rolling_away_score'] = away_avgs[mask_away]
            league_avg = result_df[['home_score', 'away_score']].stack().mean()
            self.log(f"League average score: {league_avg:.1f}")
            result_df['rolling_home_score'] = result_df['rolling_home_score'].fillna(league_avg)
            result_df['rolling_away_score'] = result_df['rolling_away_score'].fillna(league_avg)
        except Exception as e:
            self.log(f"Error computing rolling averages: {e}")
            result_df['rolling_home_score'] = 105.0
            result_df['rolling_away_score'] = 105.0
        
        return result_df
    
    @profile_time
    def calculate_rest_features(self, df, max_lookback_days=30):
        """
        Calculate rest day features using vectorized operations.
        """
        self.log(f"Calculating rest features with {max_lookback_days}-day lookback")
        result_df = df.copy()
        
        if 'game_date' in result_df.columns:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
        else:
            self.log("Warning: 'game_date' column not found")
            return result_df
        
        try:
            # Sort once for efficiency
            result_df = result_df.sort_values('game_date')
            
            # Compute previous game dates for home and away using vectorized operations
            result_df['prev_game_home'] = result_df.groupby('home_team')['game_date'].shift(1)
            result_df['prev_game_away'] = result_df.groupby('away_team')['game_date'].shift(1)
            
            # Calculate days between games
            result_df['rest_days_home'] = (result_df['game_date'] - result_df['prev_game_home']).dt.days
            result_df['rest_days_away'] = (result_df['game_date'] - result_df['prev_game_away']).dt.days
            
            # Handle missing values and clip to reasonable range
            result_df['rest_days_home'] = result_df['rest_days_home'].fillna(3).clip(lower=1, upper=10)
            result_df['rest_days_away'] = result_df['rest_days_away'].fillna(3).clip(lower=1, upper=10)
            
            # Calculate rest advantage and back-to-back indicators
            result_df['rest_advantage'] = (result_df['rest_days_home'] - result_df['rest_days_away']).clip(-5, 5)
            result_df['is_back_to_back_home'] = (result_df['rest_days_home'] <= 1).astype(int)
            result_df['is_back_to_back_away'] = (result_df['rest_days_away'] <= 1).astype(int)
            
            # Log distribution statistics
            if self.debug:
                b2b_home_pct = result_df['is_back_to_back_home'].mean() * 100
                b2b_away_pct = result_df['is_back_to_back_away'].mean() * 100
                self.log(f"Home teams on back-to-back: {b2b_home_pct:.1f}%")
                self.log(f"Away teams on back-to-back: {b2b_away_pct:.1f}%")
            
            self.log("Rest features calculated (vectorized)")
        except Exception as e:
            self.log(f"Error calculating rest features: {e}")
            traceback.print_exc()
            # Set default values
            result_df['rest_days_home'] = 2
            result_df['rest_days_away'] = 2
            result_df['rest_advantage'] = 0
            result_df['is_back_to_back_home'] = 0
            result_df['is_back_to_back_away'] = 0
        
        return result_df
    
    def compute_momentum_features(self, df):
        """
        Compute momentum features based on quarter-to-quarter scoring differences.
        """
        result_df = df.copy()
        
        try:
            # Initialize momentum features with zeros
            result_df['q1_to_q2_momentum'] = 0
            result_df['q2_to_q3_momentum'] = 0
            result_df['q3_to_q4_momentum'] = 0
            result_df['cumulative_momentum'] = 0
            
            # Calculate quarter-to-quarter shifts using vectorized operations
            for i in range(1, 4):
                home_col = f'home_q{i}'
                away_col = f'away_q{i}'
                next_home_col = f'home_q{i+1}'
                next_away_col = f'away_q{i+1}'
                momentum_col = f'q{i}_to_q{i+1}_momentum'
                
                if all(col in result_df.columns for col in [home_col, away_col, next_home_col, next_away_col]):
                    # Calculate quarter-to-quarter momentum (home perspective)
                    home_shift = result_df[next_home_col] - result_df[home_col]
                    away_shift = result_df[next_away_col] - result_df[away_col]
                    result_df[momentum_col] = (home_shift - away_shift).clip(-20, 20)
            
            # Calculate weighted cumulative momentum based on current quarter
            for idx, row in result_df.iterrows():
                current_quarter = int(row.get('current_quarter', 0))
                weights = [0.2, 0.3, 0.5]  # Weight for each quarter transition
                
                if current_quarter >= 2:
                    if current_quarter == 2:
                        # Only Q1->Q2 momentum available
                        result_df.at[idx, 'cumulative_momentum'] = row['q1_to_q2_momentum']
                    elif current_quarter == 3:
                        # Q1->Q2 and Q2->Q3 momentum available
                        result_df.at[idx, 'cumulative_momentum'] = (
                            row['q1_to_q2_momentum'] * weights[0] +
                            row['q2_to_q3_momentum'] * weights[1]
                        ) / (weights[0] + weights[1])
                    elif current_quarter >= 4:
                        # All momentum available
                        result_df.at[idx, 'cumulative_momentum'] = (
                            row['q1_to_q2_momentum'] * weights[0] +
                            row['q2_to_q3_momentum'] * weights[1] +
                            row['q3_to_q4_momentum'] * weights[2]
                        ) / sum(weights)
            
            # Normalize cumulative momentum to [-1, 1] range
            result_df['cumulative_momentum'] = result_df['cumulative_momentum'] / 15.0
            result_df['cumulative_momentum'] = result_df['cumulative_momentum'].clip(-1, 1)
            
        except Exception as e:
            self.log(f"Error computing momentum features: {e}")
            traceback.print_exc()
            # Ensure momentum features exist even after error
            result_df['q1_to_q2_momentum'] = 0
            result_df['q2_to_q3_momentum'] = 0
            result_df['q3_to_q4_momentum'] = 0
            result_df['cumulative_momentum'] = 0
        
        return result_df
    
    def compute_matchup_features(self, df, max_lookback=5):
        """
        Compute previous matchup differentials.
        """
        result_df = df.copy()
        result_df['prev_matchup_diff'] = 0.0
        
        try:
            self.log(f"Computing matchup features (max_lookback={max_lookback})")
            
            # Ensure required columns exist
            if 'game_date' not in result_df.columns:
                self.log("Warning: 'game_date' column missing for matchup calculation")
                return result_df
                
            if 'home_team' not in result_df.columns or 'away_team' not in result_df.columns:
                self.log("Warning: Team columns missing for matchup calculation")
                return result_df
                
            # Create unique team pair identifier (sorted to handle home/away reversals)
            result_df['team_pair'] = result_df.apply(
                lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), 
                axis=1
            )
            
            # Convert to datetime and sort
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
            result_df = result_df.sort_values('game_date')
            
            # Process each unique team pair
            matchup_diffs = {}
            team_pairs = result_df['team_pair'].unique()
            self.log(f"Processing matchup history for {len(team_pairs)} team pairs")
            
            for team_pair in team_pairs:
                # Get all games between these teams
                pair_games = result_df[result_df['team_pair'] == team_pair].copy()
                if len(pair_games) <= 1:
                    continue
                    
                # Calculate point differential from home team perspective
                first_home_team = pair_games.iloc[0]['home_team']
                pair_games['matchup_diff'] = pair_games.apply(
                    lambda row: row['home_score'] - row['away_score'] if row['home_team'] == first_home_team
                    else row['away_score'] - row['home_score'], 
                    axis=1
                )
                
                # For each game, store the mean differential of previous matchups
                for i in range(len(pair_games)):
                    game_id = pair_games.iloc[i]['game_id']
                    if i == 0:
                        matchup_diffs[game_id] = 0  # No previous matchups
                    else:
                        # Average of previous matchups (limited by max_lookback)
                        lookback = min(i, max_lookback)
                        prev_diffs = pair_games.iloc[i-lookback:i]['matchup_diff'].values
                        matchup_diffs[game_id] = np.mean(prev_diffs)
            
            # Add the calculated differentials to the dataframe
            result_df['prev_matchup_diff'] = result_df['game_id'].map(matchup_diffs).fillna(0)
            
            # Clip extreme values
            result_df['prev_matchup_diff'] = result_df['prev_matchup_diff'].clip(-15, 15)
            
        except Exception as e:
            self.log(f"Error computing matchup features: {e}")
            traceback.print_exc()
            result_df['prev_matchup_diff'] = 0.0
        
        return result_df
    
    def compute_quarter_specific_features(self, df):
        """
        Compute quarter-specific features (e.g., first_half_diff and pre_q4_diff).
        """
        result_df = df.copy()
        
        try:
            # Calculate first half differential (home - away)
            result_df['first_half_diff'] = (
                result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) -
                result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0)
            )
            
            # Calculate pre-Q4 differential (all quarters except Q4)
            result_df['pre_q4_diff'] = (
                result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) + result_df['home_q3'].fillna(0) -
                result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0) - result_df['away_q3'].fillna(0)
            )
            
            # Add current score differential
            result_df['score_differential'] = result_df['home_score'] - result_df['away_score']
            
        except Exception as e:
            self.log(f"Error computing quarter-specific features: {e}")
            traceback.print_exc()
            # Set default values
            result_df['first_half_diff'] = 0
            result_df['pre_q4_diff'] = 0
            result_df['score_differential'] = 0
        
        return result_df
    
    def add_advanced_features(self, df):
        """
        Add advanced features like time remaining and interaction terms.
        """
        result_df = df.copy()
        
        try:
            # Calculate time remaining (in normalized units)
            result_df['time_remaining_mins'] = result_df['current_quarter'].apply(
                lambda q: max(0, 48 - ((q - 1) * 12 + 6))  # Approximate minutes left
            )
            
            # Normalize to [0, 1] range
            result_df['time_remaining_norm'] = result_df['time_remaining_mins'] / 48.0
            
            # Create momentum indicator (consolidation of momentum features)
            if 'cumulative_momentum' in result_df.columns:
                result_df['momentum_indicator'] = result_df['cumulative_momentum']
            elif 'q1_to_q2_momentum' in result_df.columns:
                # If no cumulative momentum, use the most recent quarter transition
                result_df['momentum_indicator'] = result_df.apply(
                    lambda row: (
                        row['q3_to_q4_momentum'] / 15.0 if row['current_quarter'] >= 4 and pd.notna(row['q3_to_q4_momentum']) 
                        else row['q2_to_q3_momentum'] / 15.0 if row['current_quarter'] >= 3 and pd.notna(row['q2_to_q3_momentum'])
                        else row['q1_to_q2_momentum'] / 15.0 if row['current_quarter'] >= 2 and pd.notna(row['q1_to_q2_momentum'])
                        else 0
                    ),
                    axis=1
                )
            else:
                result_df['momentum_indicator'] = 0
            
            # Score-momentum interaction (useful for predicting comebacks)
            if 'score_ratio' in result_df.columns and 'momentum_indicator' in result_df.columns:
                # Interaction term captures when momentum contradicts current score
                # (e.g., trailing team building momentum for comeback)
                result_df['score_momentum_interaction'] = (result_df['score_ratio'] - 0.5) * result_df['momentum_indicator']
            
        except Exception as e:
            self.log(f"Error adding advanced features: {e}")
            traceback.print_exc()
            # Ensure features exist with defaults
            result_df['time_remaining_mins'] = 24  # Default to half-game
            result_df['time_remaining_norm'] = 0.5
            result_df['momentum_indicator'] = 0
            result_df['score_momentum_interaction'] = 0
        
        return result_df
    
    def validate_features(self, df):
        """
        Validate and clean all critical features.
        """
        validated_df = df.copy()
        
        # Recalculate key derived features if missing
        if 'first_half_diff' not in validated_df.columns or validated_df['first_half_diff'].isna().any():
            self.log("Recalculating first_half_diff")
            validated_df['first_half_diff'] = (
                validated_df['home_q1'].fillna(0) + validated_df['home_q2'].fillna(0) -
                validated_df['away_q1'].fillna(0) - validated_df['away_q2'].fillna(0)
            )
            
        if 'pre_q4_diff' not in validated_df.columns or validated_df['pre_q4_diff'].isna().any():
            self.log("Recalculating pre_q4_diff")
            validated_df['pre_q4_diff'] = (
                validated_df['home_q1'].fillna(0) + validated_df['home_q2'].fillna(0) + validated_df['home_q3'].fillna(0) -
                validated_df['away_q1'].fillna(0) - validated_df['away_q2'].fillna(0) - validated_df['away_q3'].fillna(0)
            )
            
        # Clip extreme values for key features
        if 'prev_matchup_diff' in validated_df.columns:
            validated_df['prev_matchup_diff'] = validated_df['prev_matchup_diff'].clip(-15, 15).fillna(0)
        else:
            validated_df['prev_matchup_diff'] = 0
            
        # Ensure all required numeric features exist and are valid
        numeric_features = [
            'home_q1', 'home_q2', 'home_q3', 'home_q4',
            'away_q1', 'away_q2', 'away_q3', 'away_q4',
            'home_score', 'away_score', 'score_ratio',
            'rolling_home_score', 'rolling_away_score',
            'rest_days_home', 'rest_days_away', 'rest_advantage',
            'is_back_to_back_home', 'is_back_to_back_away',
            'q1_to_q2_momentum', 'q2_to_q3_momentum', 'q3_to_q4_momentum',
            'cumulative_momentum', 'first_half_diff', 'pre_q4_diff', 'prev_matchup_diff'
        ]
        
        for feature in numeric_features:
            if feature in validated_df.columns:
                validated_df[feature] = pd.to_numeric(validated_df[feature], errors='coerce').fillna(0)
            else:
                self.log(f"Adding missing feature: {feature}")
                validated_df[feature] = 0
                
        # Apply reasonable constraints to values
        validated_df['score_ratio'] = validated_df['score_ratio'].clip(0, 1)
        validated_df['rolling_home_score'] = validated_df['rolling_home_score'].clip(80, 140)
        validated_df['rolling_away_score'] = validated_df['rolling_away_score'].clip(80, 140)
        
        # Final validation for quarter-specific features
        for q in range(1, 5):
            validated_df[f'home_q{q}'] = validated_df[f'home_q{q}'].clip(0, 60)
            validated_df[f'away_q{q}'] = validated_df[f'away_q{q}'].clip(0, 60)
            
        return validated_df
    
    def get_quarter_feature_sets(self):
        """
        Returns optimized feature sets for each quarter model.
        Based on feature importance analysis.
        """
        return {
            'q1': [
                'rest_advantage', 'is_back_to_back_home', 'is_back_to_back_away', 
                'prev_matchup_diff', 'rolling_home_score', 'rolling_away_score',
                'time_remaining_norm'
            ],
            
            'q2': [
                'home_q1', 'away_q1', 'rest_advantage', 'prev_matchup_diff', 
                'rolling_home_score', 'rolling_away_score', 'q1_to_q2_momentum',
                'score_ratio', 'momentum_indicator', 'score_momentum_interaction'
            ],
            
            'q3': [
                'home_q1', 'home_q2', 'away_q1', 'away_q2', 
                'first_half_diff', 'q1_to_q2_momentum', 'q2_to_q3_momentum',
                'cumulative_momentum', 'score_ratio', 'rest_advantage'
            ],
            
            'q4': [
                'home_q3', 'away_q3', 'pre_q4_diff', 'score_ratio', 
                'cumulative_momentum', 'score_differential',
                'time_remaining_norm', 'momentum_indicator'
            ]
        }
    
    def get_expected_features(self, enhanced=True):
        """
        Returns the list of expected feature names.
        """
        if enhanced:
            return [
                'home_q1', 'home_q2', 'home_q3', 'home_q4',
                'score_ratio', 'prev_matchup_diff',
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away',
                'q1_to_q2_momentum', 'q2_to_q3_momentum', 'q3_to_q4_momentum', 'cumulative_momentum'
            ]
        else:
            return [
                'home_q1', 'home_q2', 'home_q3', 'home_q4',
                'score_ratio', 'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff'
            ]


def ensemble_quarter_predictions(main_prediction, quarter_sum_prediction, current_quarter):
    """
    Combine the main model's prediction with quarter-specific predictions using an optimized weighting scheme.
    
    Args:
        main_prediction: Full-game score prediction from the main model
        quarter_sum_prediction: Sum of played + predicted quarters from quarter models
        current_quarter: Current quarter of the game (1-4)
        
    Returns:
        tuple: (ensemble_prediction, confidence, weight_main, weight_quarter)
    """
    # Define weights and confidence by quarter - increased main model influence
    if current_quarter == 1:
        weight_main, weight_quarter, confidence = 0.7, 0.3, 0.40
    elif current_quarter == 2:
        weight_main, weight_quarter, confidence = 0.8, 0.2, 0.60
    elif current_quarter == 3:
        weight_main, weight_quarter, confidence = 0.9, 0.1, 0.80
    else:
        weight_main, weight_quarter, confidence = 1.0, 0.0, 0.95
    
    # Safety check for large discrepancies between models
    if abs(main_prediction - quarter_sum_prediction) > 20:
        print(f"Warning: Large gap detected between models ({main_prediction:.1f} vs {quarter_sum_prediction:.1f})")
        # Fall back to main model with small quarter adjustment
        weight_main, weight_quarter = 0.9, 0.1
    
    # Blend predictions based on weights
    ensemble_prediction = weight_main * main_prediction + weight_quarter * quarter_sum_prediction
    
    return ensemble_prediction, confidence, weight_main, weight_quarter


def get_recommended_model_params(quarter, model_type=None):
    """
    Returns optimized hyperparameters for specific quarter models.
    
    Args:
        quarter: Quarter number (1-4)
        
    Returns:
        dict: Hyperparameters for the recommended model type
    """

        # If model type is explicitly specified, use those params regardless of quarter
    if model_type == "RandomForest":
        return {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 100,     # Reduced from default
                'max_depth': 5,          # Limit tree depth
                'min_samples_split': 10, # Require more samples per split
                'min_samples_leaf': 4,   # Require more samples per leaf
                'max_features': 'sqrt',  # Use feature subsampling
                'bootstrap': True,       # Enable bootstrapping
                'random_state': 42
            }
        }
    # Quarter-specific recommended models
   
    if quarter == 1:
        # Q1: GradientBoosting with strong regularization
        return {
            'model_type': 'GradientBoosting',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 3,
                'min_samples_split': 10,
                'subsample': 0.8,
                'random_state': 42
            }
        }
    elif quarter == 2:
        # Q2: GradientBoosting with moderate regularization
        return {
            'model_type': 'GradientBoosting',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 5,
                'subsample': 0.8,
                'random_state': 42
            }
        }
    elif quarter == 3:
        # Q3: GradientBoosting with lighter regularization
        return {
            'model_type': 'GradientBoosting',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 4,
                'min_samples_split': 5,
                'subsample': 0.8,
                'random_state': 42
            }
        }
    else:  # Q4
        # Q4: Ridge regression (optimal for final quarter)
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
    
    