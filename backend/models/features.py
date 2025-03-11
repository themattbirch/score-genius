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
        
        result_df = result_df.sort_values('game_date')
        # Compute previous game dates for home and away
        result_df['prev_game_home'] = result_df.sort_values('game_date').groupby('home_team')['game_date'].shift(1)
        result_df['prev_game_away'] = result_df.sort_values('game_date').groupby('away_team')['game_date'].shift(1)
        
        result_df['rest_days_home'] = (result_df['game_date'] - result_df['prev_game_home']).dt.days.fillna(3).clip(lower=1, upper=10)
        result_df['rest_days_away'] = (result_df['game_date'] - result_df['prev_game_away']).dt.days.fillna(3).clip(lower=1, upper=10)
        result_df['rest_advantage'] = result_df['rest_days_home'] - result_df['rest_days_away']
        result_df['is_back_to_back_home'] = (result_df['rest_days_home'] <= 1).astype(int)
        result_df['is_back_to_back_away'] = (result_df['rest_days_away'] <= 1).astype(int)
        self.log("Rest features calculated (vectorized).")
        return result_df
    
    def compute_momentum_features(self, df):
        """
        Compute momentum features based on quarter-to-quarter scoring differences.
        """
        result_df = df.copy()
        for i in range(1, 4):
            result_df[f'home_q{i}_to_q{i+1}_shift'] = result_df[f'home_q{i+1}'] - result_df[f'home_q{i}']
            result_df[f'away_q{i}_to_q{i+1}_shift'] = result_df[f'away_q{i+1}'] - result_df[f'away_q{i}']
        result_df['q1_to_q2_momentum'] = result_df['home_q1_to_q2_shift'] - result_df['away_q1_to_q2_shift']
        result_df['q2_to_q3_momentum'] = result_df['home_q2_to_q3_shift'] - result_df['away_q2_to_q3_shift']
        result_df['q3_to_q4_momentum'] = result_df['home_q3_to_q4_shift'] - result_df['away_q3_to_q4_shift']
        result_df['cumulative_momentum'] = (result_df['q1_to_q2_momentum'] +
                                            result_df['q2_to_q3_momentum'] +
                                            result_df['q3_to_q4_momentum'])
        result_df['cumulative_momentum'] = result_df['cumulative_momentum'].clip(-15, 15) / 15.0
        return result_df
    
    def compute_matchup_features(self, df, max_lookback=5):
        """
        Compute previous matchup differentials.
        """
        result_df = df.copy()
        if 'prev_matchup_diff' not in result_df.columns:
            result_df['prev_matchup_diff'] = 0.0
        try:
            self.log(f"Computing matchup features (max_lookback={max_lookback})")
            matchup_diffs = {}
            # Create unique team pair identifier
            result_df['team_pair'] = result_df.apply(lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1)
            for team_pair in result_df['team_pair'].unique():
                pair_games = result_df[result_df['team_pair'] == team_pair].sort_values('game_date')
                # Calculate matchup differential from home perspective
                pair_games['matchup_diff'] = pair_games.apply(
                    lambda row: row['home_score'] - row['away_score'] if row['home_team'] == pair_games.iloc[0]['home_team']
                    else row['away_score'] - row['home_score'], axis=1
                )
                for i in range(len(pair_games)):
                    game_id = pair_games.iloc[i]['game_id']
                    if i == 0:
                        matchup_diffs[game_id] = 0
                    else:
                        prev_diffs = pair_games.iloc[:i]['matchup_diff'].values
                        matchup_diffs[game_id] = np.mean(prev_diffs)
            result_df['prev_matchup_diff'] = result_df['game_id'].map(matchup_diffs).fillna(0)
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
        result_df['first_half_diff'] = (
            result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) -
            result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0)
        )
        result_df['pre_q4_diff'] = (
            result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) + result_df['home_q3'].fillna(0) -
            result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0) - result_df['away_q3'].fillna(0)
        )
        return result_df
    
    def validate_features(self, df):
        """
        Validate and clean all critical features.
        """
        validated_df = df.copy()
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
        if 'prev_matchup_diff' in validated_df.columns:
            validated_df['prev_matchup_diff'] = validated_df['prev_matchup_diff'].clip(-15, 15).fillna(0)
        else:
            validated_df['prev_matchup_diff'] = 0
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
        missing_features = [f for f in numeric_features if f not in validated_df.columns]
        if missing_features:
            self.log(f"Warning: Missing features after validation: {missing_features}")
        return validated_df
    
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
