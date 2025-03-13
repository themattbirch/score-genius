"""
NBAFeatureGenerator - Unified module for NBA score prediction feature engineering.
This module now includes:
  - Enhanced dynamic ensemble weighting (dynamic_ensemble_predictions)
  - A robust quarter-specific model system with fallbacks (QuarterSpecificModelSystem)
  - An uncertainty estimation module (PredictionUncertaintyEstimator)
  - Enhanced final prediction generation (generate_enhanced_predictions)
  - A validation framework (validate_enhanced_predictions)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import time
import math
from functools import wraps
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------- Helper Decorators & Simple Logging --------------------
def profile_time(func):
    """Decorator to profile function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[PROFILE] {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# A simple logging function for this module.
def simple_log(message, level="INFO", debug=False):
    if debug or level != "DEBUG":
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}")

# -------------------- NBAFeatureGenerator Class --------------------
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
        # Define league averages for fallback scenarios
        self.league_averages = {
            'score': 105.0,
            'quarter_scores': {1: 27.5, 2: 26.8, 3: 26.2, 4: 25.5}
        }

    def log(self, message, level="INFO"):
        simple_log(f"[NBAFeatureGenerator] {message}", level, self.debug)

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
                self.log("Empty DataFrame provided", "WARNING")
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
            # 8. Add advanced features.
            result_df = self.add_advanced_features(result_df)
            self.log(f"Feature generation complete: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df
        except Exception as e:
            self.log(f"Error in generate_all_features: {e}", "ERROR")
            traceback.print_exc()
            return df

    def ensure_data_types(self, df):
        """
        Ensure that critical columns exist and have correct data types.
        """
        result_df = df.copy()
        if 'game_date' in result_df.columns:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
        # Convert quarter scores to numeric
        for team in ['home', 'away']:
            for q in range(1, 5):
                col = f'{team}_q{q}'
                if col in result_df.columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        # Determine current quarter
        if 'current_quarter' in result_df.columns:
            result_df['current_quarter'] = pd.to_numeric(result_df['current_quarter'], errors='coerce').fillna(0).astype(int)
        else:
            result_df['current_quarter'] = 0
            for q in range(4, 0, -1):
                mask = ((result_df.get(f'home_q{q}', 0) > 0) | (result_df.get(f'away_q{q}', 0) > 0)) & (result_df['current_quarter'] == 0)
                result_df.loc[mask, 'current_quarter'] = q
        # Ensure team names are strings
        for col in ['home_team', 'away_team']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
        return result_df

    def compute_basic_features(self, df):
        """
        Compute basic game state features such as current scores, score ratio, and rolling averages.
        """
        result_df = df.copy()
        # Calculate total scores if missing
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
        # Compute score ratio
        result_df['score_ratio'] = 0.5
        mask = (result_df['home_score'] + result_df['away_score']) > 0
        if mask.any():
            result_df.loc[mask, 'score_ratio'] = (
                result_df.loc[mask, 'home_score'] /
                (result_df.loc[mask, 'home_score'] + result_df.loc[mask, 'away_score'])
            )
        # Compute rolling averages if needed
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
        result_df['rolling_home_score'] = self.league_averages['score']
        result_df['rolling_away_score'] = self.league_averages['score']
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
            self.log(f"Error computing rolling averages: {e}", "ERROR")
            result_df['rolling_home_score'] = self.league_averages['score']
            result_df['rolling_away_score'] = self.league_averages['score']
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
            self.log("Warning: 'game_date' column not found", "WARNING")
            return result_df
        try:
            result_df = result_df.sort_values('game_date')
            result_df['prev_game_home'] = result_df.groupby('home_team')['game_date'].shift(1)
            result_df['prev_game_away'] = result_df.groupby('away_team')['game_date'].shift(1)
            result_df['rest_days_home'] = (result_df['game_date'] - result_df['prev_game_home']).dt.days
            result_df['rest_days_away'] = (result_df['game_date'] - result_df['prev_game_away']).dt.days
            result_df['rest_days_home'] = result_df['rest_days_home'].fillna(3).clip(lower=1, upper=10)
            result_df['rest_days_away'] = result_df['rest_days_away'].fillna(3).clip(lower=1, upper=10)
            result_df['rest_advantage'] = (result_df['rest_days_home'] - result_df['rest_days_away']).clip(-5, 5)
            result_df['is_back_to_back_home'] = (result_df['rest_days_home'] <= 1).astype(int)
            result_df['is_back_to_back_away'] = (result_df['rest_days_away'] <= 1).astype(int)
            if self.debug:
                b2b_home_pct = result_df['is_back_to_back_home'].mean() * 100
                b2b_away_pct = result_df['is_back_to_back_away'].mean() * 100
                self.log(f"Home teams on back-to-back: {b2b_home_pct:.1f}%")
                self.log(f"Away teams on back-to-back: {b2b_away_pct:.1f}%")
            self.log("Rest features calculated (vectorized)")
        except Exception as e:
            self.log(f"Error calculating rest features: {e}", "ERROR")
            traceback.print_exc()
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
            result_df['q1_to_q2_momentum'] = 0
            result_df['q2_to_q3_momentum'] = 0
            result_df['q3_to_q4_momentum'] = 0
            result_df['cumulative_momentum'] = 0
            for i in range(1, 4):
                home_col = f'home_q{i}'
                away_col = f'away_q{i}'
                next_home_col = f'home_q{i+1}'
                next_away_col = f'away_q{i+1}'
                momentum_col = f'q{i}_to_q{i+1}_momentum'
                if all(col in result_df.columns for col in [home_col, away_col, next_home_col, next_away_col]):
                    home_shift = result_df[next_home_col] - result_df[home_col]
                    away_shift = result_df[next_away_col] - result_df[away_col]
                    result_df[momentum_col] = (home_shift - away_shift).clip(-20, 20)
            weights = np.array([0.2, 0.3, 0.5])
            for idx, row in result_df.iterrows():
                current_quarter = int(row.get('current_quarter', 0))
                if current_quarter < 2:
                    continue
                if current_quarter == 2:
                    result_df.at[idx, 'cumulative_momentum'] = row['q1_to_q2_momentum']
                elif current_quarter == 3:
                    result_df.at[idx, 'cumulative_momentum'] = (
                        row['q1_to_q2_momentum'] * weights[0] +
                        row['q2_to_q3_momentum'] * weights[1]
                    ) / (weights[0] + weights[1])
                elif current_quarter >= 4:
                    result_df.at[idx, 'cumulative_momentum'] = (
                        row['q1_to_q2_momentum'] * weights[0] +
                        row['q2_to_q3_momentum'] * weights[1] +
                        row['q3_to_q4_momentum'] * weights[2]
                    ) / sum(weights)
            result_df['cumulative_momentum'] = result_df['cumulative_momentum'] / 15.0
            result_df['cumulative_momentum'] = result_df['cumulative_momentum'].clip(-1, 1)
        except Exception as e:
            self.log(f"Error computing momentum features: {e}", "ERROR")
            traceback.print_exc()
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
            if 'game_date' not in result_df.columns:
                self.log("Warning: 'game_date' column missing for matchup calculation", "WARNING")
                return result_df
            if 'home_team' not in result_df.columns or 'away_team' not in result_df.columns:
                self.log("Warning: Team columns missing for matchup calculation", "WARNING")
                return result_df
            result_df['team_pair'] = result_df.apply(
                lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1
            )
            result_df['game_date'] = pd.to_datetime(result_df['game_date'], errors='coerce')
            result_df = result_df.sort_values('game_date')
            matchup_diffs = {}
            team_pairs = result_df['team_pair'].unique()
            self.log(f"Processing matchup history for {len(team_pairs)} team pairs")
            for team_pair in team_pairs:
                pair_games = result_df[result_df['team_pair'] == team_pair].copy()
                if len(pair_games) <= 1:
                    continue
                first_home_team = pair_games.iloc[0]['home_team']
                pair_games['matchup_diff'] = pair_games.apply(
                    lambda row: row['home_score'] - row['away_score'] if row['home_team'] == first_home_team
                    else row['away_score'] - row['home_score'], axis=1
                )
                for i in range(len(pair_games)):
                    game_id = pair_games.iloc[i]['game_id']
                    if i == 0:
                        matchup_diffs[game_id] = 0
                    else:
                        lookback = min(i, max_lookback)
                        prev_diffs = pair_games.iloc[i-lookback:i]['matchup_diff'].values
                        matchup_diffs[game_id] = np.mean(prev_diffs)
            result_df['prev_matchup_diff'] = result_df['game_id'].map(matchup_diffs).fillna(0)
            result_df['prev_matchup_diff'] = result_df['prev_matchup_diff'].clip(-15, 15)
        except Exception as e:
            self.log(f"Error computing matchup features: {e}", "ERROR")
            traceback.print_exc()
            result_df['prev_matchup_diff'] = 0.0
        return result_df

    def compute_quarter_specific_features(self, df):
        """
        Compute quarter-specific features (e.g., first_half_diff and pre_q4_diff).
        """
        result_df = df.copy()
        try:
            result_df['first_half_diff'] = (
                result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) -
                result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0)
            )
            result_df['pre_q4_diff'] = (
                result_df['home_q1'].fillna(0) + result_df['home_q2'].fillna(0) + result_df['home_q3'].fillna(0) -
                result_df['away_q1'].fillna(0) - result_df['away_q2'].fillna(0) - result_df['away_q3'].fillna(0)
            )
            result_df['score_differential'] = result_df['home_score'] - result_df['away_score']
        except Exception as e:
            self.log(f"Error computing quarter-specific features: {e}", "ERROR")
            traceback.print_exc()
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
            result_df['time_remaining_mins'] = result_df['current_quarter'].apply(
                lambda q: max(0, 48 - ((q - 1) * 12 + 6))
            )
            result_df['time_remaining_norm'] = result_df['time_remaining_mins'] / 48.0
            if 'cumulative_momentum' in result_df.columns:
                result_df['momentum_indicator'] = result_df['cumulative_momentum']
            elif 'q1_to_q2_momentum' in result_df.columns:
                result_df['momentum_indicator'] = result_df.apply(
                    lambda row: (row['q3_to_q4_momentum'] if (row.get('q3_to_q4_momentum') and row['current_quarter'] >= 4)
                                 else row['q2_to_q3_momentum'] if (row.get('q2_to_q3_momentum') and row['current_quarter'] >= 3)
                                 else row['q1_to_q2_momentum'] if (row.get('q1_to_q2_momentum') and row['current_quarter'] >= 2)
                                 else 0), axis=1
                )
            else:
                result_df['momentum_indicator'] = 0
            if 'score_ratio' in result_df.columns and 'momentum_indicator' in result_df.columns:
                result_df['score_momentum_interaction'] = (result_df['score_ratio'] - 0.5) * result_df['momentum_indicator']
        except Exception as e:
            self.log(f"Error adding advanced features: {e}", "ERROR")
            traceback.print_exc()
            result_df['time_remaining_mins'] = 24
            result_df['time_remaining_norm'] = 0.5
            result_df['momentum_indicator'] = 0
            result_df['score_momentum_interaction'] = 0
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
            else:
                self.log(f"Adding missing feature: {feature}")
                validated_df[feature] = 0
        validated_df['score_ratio'] = validated_df['score_ratio'].clip(0, 1)
        validated_df['rolling_home_score'] = validated_df['rolling_home_score'].clip(80, 140)
        validated_df['rolling_away_score'] = validated_df['rolling_away_score'].clip(80, 140)
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
                'first_half_diff', 'q1_to_q2_momentum',
                'q2_to_q3_momentum', 'cumulative_momentum', 'score_ratio', 'rest_advantage'
            ],
            'q4': [
                'home_q1', 'home_q2', 'home_q3', 
                'away_q1', 'away_q2', 'away_q3',
                'pre_q4_diff', 'score_ratio', 
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
                'away_q1', 'away_q2', 'away_q3', 'away_q4',
                'score_ratio', 'prev_matchup_diff',
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away',
                'q1_to_q2_momentum', 'q2_to_q3_momentum', 'q3_to_q4_momentum', 'cumulative_momentum',
                'first_half_diff', 'pre_q4_diff', 'score_differential',
                'time_remaining_norm', 'momentum_indicator', 'score_momentum_interaction'
            ]
        else:
            return [
                'home_q1', 'home_q2', 'home_q3', 'home_q4',
                'away_q1', 'away_q2', 'away_q3', 'away_q4',
                'score_ratio', 'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff',
                'score_differential', 'rest_advantage'
            ]

# -------------------- Ensemble Weight Visualization & Tuning --------------------
class EnsembleWeightVisualizer:
    """
    Tools for visualizing and tuning ensemble weights with different strategies.
    """
    def __init__(self, error_history=None):
        self.error_history = error_history or {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.weighting_strategies = {
            'standard': self._standard_weights,
            'conservative': self._conservative_weights,
            'aggressive': self._aggressive_weights,
            'adaptive': self._adaptive_weights,
            'uncertainty_based': self._uncertainty_based_weights
        }
        self.weight_history = []
        
    def _standard_weights(self, quarter, **kwargs):
        if quarter == 1:
            return 0.80, 0.20
        elif quarter == 2:
            return 0.85, 0.15
        elif quarter == 3:
            return 0.90, 0.10
        else:
            return 0.95, 0.05
            
    def _conservative_weights(self, quarter, **kwargs):
        if quarter == 1:
            return 0.90, 0.10
        elif quarter == 2:
            return 0.92, 0.08
        elif quarter == 3:
            return 0.95, 0.05
        else:
            return 0.97, 0.03
    
    def _aggressive_weights(self, quarter, **kwargs):
        if quarter == 1:
            return 0.70, 0.30
        elif quarter == 2:
            return 0.75, 0.25
        elif quarter == 3:
            return 0.80, 0.20
        else:
            return 0.85, 0.15
            
    def _adaptive_weights(self, quarter, **kwargs):
        if not self.error_history:
            return self._standard_weights(quarter)
        main_error = self.error_history.get('main_model', {}).get(quarter, 7.0)
        quarter_error = self.error_history.get('quarter_model', {}).get(quarter, 7.0)
        total_error = main_error + quarter_error
        main_weight = quarter_error / total_error if total_error > 0 else 0.8
        main_weight = min(max(main_weight, 0.6), 0.95)
        quarter_weight = 1.0 - main_weight
        return main_weight, quarter_weight
        
    def _uncertainty_based_weights(self, quarter, **kwargs):
        main_uncertainty = kwargs.get('main_uncertainty', 7.0)
        quarter_uncertainty = kwargs.get('quarter_uncertainty', 7.0)
        total_uncertainty = main_uncertainty + quarter_uncertainty
        if total_uncertainty > 0:
            main_weight = quarter_uncertainty / total_uncertainty
            quarter_weight = main_uncertainty / total_uncertainty
        else:
            main_weight, quarter_weight = self._standard_weights(quarter)
        main_weight = min(max(main_weight, 0.6), 0.95)
        quarter_weight = 1.0 - main_weight
        return main_weight, quarter_weight
    
    def compare_strategies(self, current_quarter, score_differential=0, momentum=0, 
                           main_uncertainty=None, quarter_uncertainty=None):
        results = []
        kwargs = {
            'score_differential': score_differential,
            'momentum': momentum,
            'main_uncertainty': main_uncertainty,
            'quarter_uncertainty': quarter_uncertainty
        }
        for name, strategy_func in self.weighting_strategies.items():
            main_weight, quarter_weight = strategy_func(current_quarter, **kwargs)
            results.append({
                'strategy': name,
                'main_weight': main_weight,
                'quarter_weight': quarter_weight
            })
        return pd.DataFrame(results)
    
    def visualize_weight_progression(self, strategy='standard'):
        strategy_func = self.weighting_strategies.get(strategy, self._standard_weights)
        quarters = [1, 2, 3, 4]
        main_weights = []
        quarter_weights = []
        for q in quarters:
            main_w, quarter_w = strategy_func(q)
            main_weights.append(main_w)
            quarter_weights.append(quarter_w)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(quarters))
        width = 0.35
        ax.bar(x - width/2, main_weights, width, label='Main Model', color='#3498db')
        ax.bar(x + width/2, quarter_weights, width, label='Quarter Model', color='#e74c3c')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Weight')
        ax.set_title(f'Ensemble Weight Progression - {strategy.capitalize()} Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.legend()
        for i, v in enumerate(main_weights):
            ax.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(quarter_weights):
            ax.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig
    
    def visualize_all_strategies(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, strategy in enumerate(self.weighting_strategies.keys()):
            if i >= len(axes):
                break
            quarters = [1, 2, 3, 4]
            main_weights = []
            quarter_weights = []
            strategy_func = self.weighting_strategies[strategy]
            for q in quarters:
                main_w, quarter_w = strategy_func(q)
                main_weights.append(main_w)
                quarter_weights.append(quarter_w)
            ax = axes[i]
            x = range(len(quarters))
            ax.plot(x, main_weights, 'o-', label='Main Model', linewidth=3, markersize=8, color='#3498db')
            ax.plot(x, quarter_weights, 'o-', label='Quarter Model', linewidth=3, markersize=8, color='#e74c3c')
            ax.set_title(f'{strategy.capitalize()} Strategy', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(quarters)
            ax.set_xlabel('Quarter')
            ax.set_ylabel('Weight')
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.0)
        plt.tight_layout()
        fig.suptitle('Ensemble Weight Strategies Comparison', fontsize=16)
        plt.subplots_adjust(top=0.93)
        return fig
    
    def track_weight_usage(self, game_id, quarter, main_weight, quarter_weight, prediction_error=None):
        self.weight_history.append({
            'game_id': game_id,
            'quarter': quarter,
            'main_weight': main_weight,
            'quarter_weight': quarter_weight,
            'timestamp': pd.Timestamp.now(),
            'prediction_error': prediction_error
        })
    
    def update_weights_from_feedback(self, recent_window=10):
        if len(self.weight_history) < recent_window:
            return self.error_history
        recent_errors = self.weight_history[-recent_window:]
        recent_df = pd.DataFrame(recent_errors)
        if 'prediction_error' not in recent_df.columns or recent_df['prediction_error'].isna().all():
            return self.error_history
        quarter_errors = recent_df.groupby('quarter')['prediction_error'].mean().to_dict()
        for q, error in quarter_errors.items():
            if q in self.error_history['main_model']:
                self.error_history['main_model'][q] = 0.7 * error + 0.3 * self.error_history['main_model'][q]
        return self.error_history
    
    def analyze_weight_performance(self):
        if not self.weight_history:
            return None
        history_df = pd.DataFrame(self.weight_history)
        if 'prediction_error' not in history_df.columns or history_df['prediction_error'].isna().all():
            return None
        history_df['abs_error'] = history_df['prediction_error'].abs()
        results = []
        for quarter, quarter_df in history_df.groupby('quarter'):
            main_corr = quarter_df['main_weight'].corr(quarter_df['abs_error'])
            quarter_corr = quarter_df['quarter_weight'].corr(quarter_df['abs_error'])
            min_error_idx = quarter_df['abs_error'].idxmin()
            if not pd.isna(min_error_idx):
                best_main = quarter_df.loc[min_error_idx, 'main_weight']
                best_quarter = quarter_df.loc[min_error_idx, 'quarter_weight']
            else:
                best_main, best_quarter = None, None
            avg_error = quarter_df['abs_error'].mean()
            results.append({
                'quarter': quarter,
                'sample_size': len(quarter_df),
                'avg_error': avg_error,
                'main_weight_correlation': main_corr,
                'quarter_weight_correlation': quarter_corr,
                'best_main_weight': best_main,
                'best_quarter_weight': best_quarter
            })
        return pd.DataFrame(results).sort_values('quarter')

    def create_confidence_indicator(self, prediction, lower_bound, upper_bound, 
                                    current_quarter, home_score=None, width=100, height=30):
        def to_svg_x(score, svg_min, score_range):
            return (score - svg_min) / score_range * width
        interval_width = upper_bound - lower_bound
        expected_range = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}
        svg_min = max(0, lower_bound - 10)
        score_range = max(expected_range.get(current_quarter, 25), upper_bound + 10 - svg_min)
        pred_x = to_svg_x(prediction, svg_min, score_range)
        lower_x = to_svg_x(lower_bound, svg_min, score_range)
        upper_x = to_svg_x(upper_bound, svg_min, score_range)
        quarter_colors = {0: "#d3d3d3", 1: "#ffa07a", 2: "#ff7f50", 3: "#ff4500", 4: "#8b0000"}
        color = quarter_colors.get(current_quarter, "#000000")
        confidence = max(0, min(100, 100 - (interval_width / expected_range.get(current_quarter, 25) * 100)))
        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="{lower_x}" y="{height/2 - 5}" width="{upper_x - lower_x}" height="10" fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="1" />
  <circle cx="{pred_x}" cy="{height/2}" r="5" fill="{color}" />
  <text x="{pred_x}" y="{height/2 - 8}" text-anchor="middle" font-size="10">{prediction:.1f}</text>
  <text x="{lower_x}" y="{height - 5}" text-anchor="middle" font-size="8">{lower_bound:.1f}</text>
  <text x="{upper_x}" y="{height - 5}" text-anchor="middle" font-size="8">{upper_bound:.1f}</text>
  <text x="{width - 5}" y="{height/2 + 4}" text-anchor="end" font-size="10" font-weight="bold">{confidence:.0f}%</text>
</svg>"""
        if home_score is not None and home_score > 0:
            score_x = to_svg_x(home_score, svg_min, score_range)
            svg += f"""<circle cx="{score_x}" cy="{height/2}" r="4" fill="none" stroke="green" stroke-width="2" />"""
        return svg
    
    def dynamically_adjust_interval(self, prediction, current_quarter, historic_accuracy):
        if not historic_accuracy or current_quarter not in historic_accuracy:
            return self.calculate_prediction_interval(prediction, current_quarter)
        lower, upper, width = self.calculate_prediction_interval(prediction, current_quarter)
        accuracy = historic_accuracy[current_quarter]
        if accuracy.get('coverage_pct', 95) < 90:
            widening_factor = (95 - accuracy.get('coverage_pct', 95)) / 50
            width *= (1.0 + widening_factor)
            lower = prediction - width/2
            upper = prediction + width/2
        elif accuracy.get('coverage_pct', 95) > 98:
            narrowing_factor = (accuracy.get('coverage_pct', 95) - 95) / 300
            width *= (1.0 - narrowing_factor)
            lower = prediction - width/2
            upper = prediction + width/2
        expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}
        confidence = max(0, min(100, 100 - (width / expected_width.get(current_quarter, 25) * 100)))
        return lower, upper, confidence

# -------------------- Dynamic Ensemble Weighting --------------------
def dynamic_ensemble_predictions(main_prediction, quarter_prediction, current_quarter,
                                 score_differential=0, momentum=0, time_remaining=None,
                                 error_history=None, weighting_strategy='adaptive',
                                 main_uncertainty=None, quarter_uncertainty=None):
    """
    Enhanced ensemble method with dynamic weighting based on multiple factors.
    Args:
        main_prediction: Prediction from the main model.
        quarter_prediction: Prediction from quarter-specific models.
        current_quarter: Current quarter of the game (1-4).
        score_differential: Point difference between teams (home - away).
        momentum: Momentum value (-1 to 1, positive = home team momentum).
        time_remaining: Minutes remaining in the game (optional).
        error_history: Dict with historical error metrics by quarter (optional).
        weighting_strategy: Strategy to use ('standard', 'conservative', 'aggressive', 'adaptive', 'uncertainty_based').
        main_uncertainty: Uncertainty estimate for main model prediction.
        quarter_uncertainty: Uncertainty estimate for quarter model prediction.
    Returns:
        tuple: (ensemble_prediction, confidence, weight_main, weight_quarter)
    """
    weight_viz = EnsembleWeightVisualizer(error_history)
    strategy_func = weight_viz.weighting_strategies.get(weighting_strategy, weight_viz._standard_weights)
    kwargs = {
        'score_differential': score_differential,
        'momentum': momentum,
        'main_uncertainty': main_uncertainty,
        'quarter_uncertainty': quarter_uncertainty
    }
    weight_main, weight_quarter = strategy_func(current_quarter, **kwargs)
    if time_remaining is not None:
        total_minutes = 48.0
        elapsed = total_minutes - time_remaining
        progress = min(1.0, max(0.0, elapsed / total_minutes))
        sigmoid_progress = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
        weight_main = weight_main + (1.0 - weight_main) * sigmoid_progress
        weight_quarter = 1.0 - weight_main
    if abs(score_differential) < 8:
        close_game_adjustment = 0.05 * (1.0 - abs(score_differential) / 8.0)
        weight_quarter = min(0.35, weight_quarter + close_game_adjustment)
        weight_main = 1.0 - weight_quarter
    if abs(momentum) > 0.3:
        momentum_adjustment = 0.05 * (abs(momentum) - 0.3) / 0.7
        weight_quarter = min(0.3, weight_quarter + momentum_adjustment)
        weight_main = 1.0 - weight_quarter
    prediction_gap = abs(main_prediction - quarter_prediction)
    if prediction_gap > 15:
        discrepancy_adjustment = min(0.3, 0.01 * prediction_gap)
        weight_main = min(0.95, weight_main + discrepancy_adjustment)
        weight_quarter = 1.0 - weight_main
    if current_quarter == 1:
        base_confidence = 0.40
    elif current_quarter == 2:
        base_confidence = 0.60
    elif current_quarter == 3:
        base_confidence = 0.80
    else:
        base_confidence = 0.95
    confidence = base_confidence
    if prediction_gap > 15:
        confidence = max(0.3, confidence - (prediction_gap / 100.0))
    if abs(momentum) > 0.5:
        confidence = max(0.3, confidence - 0.1)
    ensemble_prediction = weight_main * main_prediction + weight_quarter * quarter_prediction
    return ensemble_prediction, confidence, weight_main, weight_quarter

# -------------------- Quarter-Specific Model System --------------------
class QuarterSpecificModelSystem:
    """
    A robust system for managing quarter-specific prediction models with automatic
    fallbacks and ensemble integration.
    """
    def __init__(self, feature_generator=None):
        """
        Initialize the quarter model system.
        Args:
            feature_generator: An instance of NBAFeatureGenerator (optional).
        """
        self.models = {}
        self.fallback_models = {}
        self.feature_generator = feature_generator or NBAFeatureGenerator()
        self.quarter_feature_sets = self.feature_generator.get_quarter_feature_sets()
        self.error_history = {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.prediction_cache = {}

    def load_models(self, model_dir='models'):
        """
        Load all quarter-specific models from disk.
        Args:
            model_dir: Directory containing the model files.
        """
        import joblib
        import os
        for quarter in range(1, 5):
            model_path = os.path.join(model_dir, f'q{quarter}_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[quarter] = joblib.load(model_path)
                    print(f"Loaded Q{quarter} model from {model_path}")
                except Exception as e:
                    print(f"Error loading Q{quarter} model: {e}")
                    self.models[quarter] = None
        self._create_fallback_models()

    def _create_fallback_models(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        for quarter in range(1, 5):
            if quarter <= 3:
                model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            else:
                model = Ridge(alpha=1.0, random_state=42)
            self.fallback_models[quarter] = model

    def train_fallback_models(self, X, y, current_quarter):
        if current_quarter <= 0 or current_quarter > 4:
            return
        features = self.quarter_feature_sets.get(f'q{current_quarter}', [])
        available_features = [f for f in features if f in X.columns]
        if not available_features:
            return
        model = self.fallback_models.get(current_quarter)
        if model is not None:
            try:
                model.fit(X[available_features], y)
                print(f"Trained fallback model for Q{current_quarter} with {len(available_features)} features")
            except Exception as e:
                print(f"Error training fallback model for Q{current_quarter}: {e}")

    def predict_quarter(self, X, quarter):
        cache_key = f"{hash(str(X.values.tobytes()))}_q{quarter}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        if quarter < 1 or quarter > 4:
            return 0.0
        features = self.quarter_feature_sets.get(f'q{quarter}', [])
        available_features = [f for f in features if f in X.columns]
        if not available_features:
            print(f"No features available for Q{quarter} prediction. Using default.")
            return self.feature_generator.league_averages['quarter_scores'].get(quarter, 25.0)
        model = self.models.get(quarter)
        prediction = None
        if model is not None:
            try:
                if hasattr(model, 'feature_names_in_'):
                    if all(f in model.feature_names_in_ for f in available_features):
                        prediction = model.predict(X[available_features])[0]
                    else:
                        print(f"Q{quarter} model feature mismatch. Using fallback.")
                else:
                    prediction = model.predict(X[available_features])[0]
            except Exception as e:
                print(f"Error using Q{quarter} model: {e}")
        if prediction is None:
            fallback = self.fallback_models.get(quarter)
            if fallback is not None:
                try:
                    prediction = fallback.predict(X[available_features])[0]
                except Exception as e:
                    print(f"Error using Q{quarter} fallback model: {e}")
        if prediction is None:
            prediction = self.feature_generator.league_averages['quarter_scores'].get(quarter, 25.0)
        self.prediction_cache[cache_key] = prediction
        return prediction

    def predict_remaining_quarters(self, game_data, current_quarter):
        if current_quarter >= 4:
            return {}
        X = pd.DataFrame([game_data])
        if not hasattr(self.feature_generator, 'add_advanced_features'):
            X = self.feature_generator.generate_all_features(X)
        else:
            X = self.feature_generator.add_advanced_features(X)
        results = {}
        for q in range(current_quarter + 1, 5):
            pred_score = self.predict_quarter(X, q)
            results[f'q{q}'] = pred_score
            X[f'home_q{q}'] = pred_score
            if q == 2:
                X['first_half_diff'] = (X['home_q1'].values[0] + pred_score -
                                        X['away_q1'].values[0] - X.get('away_q2', pd.Series([0])).values[0])
                X['q1_to_q2_momentum'] = (pred_score - X['home_q1'].values[0] -
                                          (X.get('away_q2', pd.Series([0])).values[0] - X['away_q1'].values[0])
                )
            elif q == 3:
                X['pre_q4_diff'] = (X['home_q1'].values[0] + X['home_q2'].values[0] + pred_score -
                                    X['away_q1'].values[0] - X['away_q2'].values[0] - X.get('away_q3', pd.Series([0])).values[0])
                X['q2_to_q3_momentum'] = (pred_score - X['home_q2'].values[0] -
                                          (X.get('away_q3', pd.Series([0])).values[0] - X['away_q2'].values[0])
                )
            elif q == 4:
                X['q3_to_q4_momentum'] = (pred_score - X['home_q3'].values[0] -
                                          (X.get('away_q4', pd.Series([0])).values[0] - X['away_q3'].values[0])
                )
        return results

    def predict_final_score(self, game_data, main_model_prediction=None):
        current_quarter = int(game_data.get('current_quarter', 0))
        home_score = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, min(current_quarter + 1, 5))])
        if main_model_prediction is None:
            if current_quarter > 0:
                main_model_prediction = home_score * (4.0 / current_quarter)
            else:
                main_model_prediction = self.feature_generator.league_averages['score']
        if current_quarter <= 0:
            return main_model_prediction, 0.4, {
                'main_model': main_model_prediction,
                'quarter_model': 0,
                'weights': {'main': 1.0, 'quarter': 0.0}
            }
        remaining_quarters = self.predict_remaining_quarters(game_data, current_quarter)
        quarter_sum_prediction = home_score + sum(remaining_quarters.values())
        score_differential = float(game_data.get('score_differential', 0))
        momentum = float(game_data.get('cumulative_momentum', 0))
        if current_quarter <= 4:
            time_remaining = 12 * (4 - current_quarter) + 6
        else:
            time_remaining = 0
        ensemble_pred, confidence, weight_main, weight_quarter = dynamic_ensemble_predictions(
            main_model_prediction,
            quarter_sum_prediction,
            current_quarter,
            score_differential,
            momentum,
            time_remaining,
            self.error_history
        )
        breakdown = {
            'main_model': main_model_prediction,
            'quarter_model': quarter_sum_prediction,
            'weights': {'main': weight_main, 'quarter': weight_quarter},
            'quarter_predictions': remaining_quarters,
            'current_score': home_score,
            'score_differential': score_differential,
            'momentum': momentum,
            'current_quarter': current_quarter,
            'time_remaining': time_remaining
        }
        return ensemble_pred, confidence, breakdown

# -------------------- Prediction Uncertainty Estimator --------------------
class PredictionUncertaintyEstimator:
    """
    Estimates uncertainty (confidence intervals) for NBA score predictions
    based on historical error patterns and game context.
    """
    def __init__(self):
        self.mae_by_quarter = {0: 8.5, 1: 7.0, 2: 6.0, 3: 5.0, 4: 0.0}
        self.std_by_quarter = {0: 4.5, 1: 3.8, 2: 3.2, 3: 2.6, 4: 0.0}
        self.margin_adjustments = {'close': 1.2, 'moderate': 1.0, 'blowout': 0.8}
        self.momentum_effects = {'high': 1.2, 'moderate': 1.0, 'low': 0.8}
        self.historical_errors = {q: [] for q in range(5)}
        self.interval_coverage = {q: {'inside': 0, 'total': 0} for q in range(5)}

    def calculate_prediction_interval(self, prediction, current_quarter,
                                      score_margin=None, momentum=None, 
                                      confidence_level=0.95):
        mae = self.mae_by_quarter.get(current_quarter, 8.0)
        std = self.std_by_quarter.get(current_quarter, 4.0)
        if score_margin is not None:
            if score_margin < 5:
                margin_factor = self.margin_adjustments['close']
            elif score_margin > 15:
                margin_factor = self.margin_adjustments['blowout']
            else:
                margin_factor = self.margin_adjustments['moderate']
            mae *= margin_factor
            std *= margin_factor
        if momentum is not None:
            abs_momentum = abs(momentum)
            if abs_momentum > 0.6:
                momentum_factor = self.momentum_effects['high']
            elif abs_momentum > 0.3:
                momentum_factor = self.momentum_effects['moderate']
            else:
                momentum_factor = self.momentum_effects['low']
            mae *= momentum_factor
            std *= momentum_factor
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        interval_half_width = z_score * np.sqrt(mae**2 + std**2)
        narrowing_factor = 1.0 - (current_quarter * 0.15)
        interval_half_width *= max(0.4, narrowing_factor)
        lower_bound = prediction - interval_half_width
        upper_bound = prediction + interval_half_width
        return lower_bound, upper_bound, interval_half_width * 2

    def update_error_metrics(self, errors_by_quarter):
        for quarter, errors in errors_by_quarter.items():
            if errors:
                self.historical_errors[quarter].extend(errors)
                self.historical_errors[quarter] = self.historical_errors[quarter][-100:]
                mae = np.mean(np.abs(self.historical_errors[quarter]))
                std = np.std(self.historical_errors[quarter])
                self.mae_by_quarter[quarter] = mae
                self.std_by_quarter[quarter] = std

    def record_interval_coverage(self, quarter, lower, upper, actual):
        self.interval_coverage[quarter]['total'] += 1
        if lower <= actual <= upper:
            self.interval_coverage[quarter]['inside'] += 1

    def get_coverage_stats(self):
        stats_list = []
        for quarter, data in self.interval_coverage.items():
            if data['total'] > 0:
                coverage_pct = (data['inside'] / data['total']) * 100
                stats_list.append({
                    'quarter': quarter,
                    'sample_size': data['total'],
                    'covered': data['inside'],
                    'coverage_pct': coverage_pct
                })
        return pd.DataFrame(stats_list)
    
    def visualize_confidence_intervals(self, predictions_df):
        n_games = len(predictions_df)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = {0: '#d3d3d3', 1: '#ffa07a', 2: '#ff7f50', 3: '#ff4500', 4: '#8b0000'}
        for i, (_, game) in enumerate(predictions_df.iterrows()):
            quarter = int(game.get('current_quarter', 0))
            prediction = float(game.get('predicted_home_score', 105))
            score_margin = abs(float(game.get('score_differential', 0)))
            momentum = float(game.get('cumulative_momentum', 0)) if 'cumulative_momentum' in game else None
            lower, upper, _ = self.calculate_prediction_interval(prediction, quarter, score_margin, momentum)
            ax.scatter(i, prediction, color=colors.get(quarter, '#000000'), s=100,
                       label=f'Q{quarter}' if quarter not in plt.gca().get_legend_handles_labels()[1] else "")
            ax.plot([i, i], [lower, upper], color=colors.get(quarter, '#000000'),
                    linewidth=3, alpha=0.7)
            ax.text(i, upper + 2, f"{game.get('home_team', '')}", rotation=90, ha='center', va='bottom', fontsize=10)
            ax.add_patch(Rectangle((i-0.2, lower), 0.4, upper-lower, alpha=0.2, color=colors.get(quarter, '#000000')))
            current_score = game.get('home_score', None)
            if current_score is not None and current_score > 0:
                ax.scatter(i, current_score, marker='o', s=80, facecolors='none', edgecolors='green', linewidth=2, alpha=0.8)
        ax.set_xlabel('Game', fontsize=12)
        ax.set_ylabel('Predicted Home Score', fontsize=12)
        ax.set_title('Score Predictions with Confidence Intervals', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Quarter')
        ax.set_xticks(range(n_games))
        plt.tight_layout()
        return fig
    
    def create_confidence_indicator(self, prediction, lower_bound, upper_bound, 
                                    current_quarter, home_score=None, width=100, height=30):
        interval_width = upper_bound - lower_bound
        expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}
        svg_min = max(0, lower_bound - 10)
        score_range = max(expected_width.get(current_quarter, 25), upper_bound + 10 - svg_min)
        def to_svg_x(score):
            return (score - svg_min) / score_range * width
        pred_x = to_svg_x(prediction)
        lower_x = to_svg_x(lower_bound)
        upper_x = to_svg_x(upper_bound)
        quarter_colors = {0: "#d3d3d3", 1: "#ffa07a", 2: "#ff7f50", 3: "#ff4500", 4: "#8b0000"}
        color = quarter_colors.get(current_quarter, "#000000")
        confidence = max(0, min(100, 100 - (interval_width / expected_width.get(current_quarter, 25) * 100)))
        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="{lower_x}" y="{height/2 - 5}" width="{upper_x - lower_x}" height="10" fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="1" />
  <circle cx="{pred_x}" cy="{height/2}" r="5" fill="{color}" />
  <text x="{pred_x}" y="{height/2 - 8}" text-anchor="middle" font-size="10">{prediction:.1f}</text>
  <text x="{lower_x}" y="{height - 5}" text-anchor="middle" font-size="8">{lower_bound:.1f}</text>
  <text x="{upper_x}" y="{height - 5}" text-anchor="middle" font-size="8">{upper_bound:.1f}</text>
  <text x="{width - 5}" y="{height/2 + 4}" text-anchor="end" font-size="10" font-weight="bold">{confidence:.0f}%</text>
</svg>"""
        if home_score is not None and home_score > 0:
            score_x = to_svg_x(home_score)
            svg += f"""<circle cx="{score_x}" cy="{height/2}" r="4" fill="none" stroke="green" stroke-width="2" />"""
        return svg
    
    def dynamically_adjust_interval(self, prediction, current_quarter, historic_accuracy):
        if not historic_accuracy or current_quarter not in historic_accuracy:
            return self.calculate_prediction_interval(prediction, current_quarter)
        lower, upper, width = self.calculate_prediction_interval(prediction, current_quarter)
        accuracy = historic_accuracy[current_quarter]
        if accuracy.get('coverage_pct', 95) < 90:
            widening_factor = (95 - accuracy.get('coverage_pct', 95)) / 50
            width *= (1.0 + widening_factor)
            lower = prediction - width/2
            upper = prediction + width/2
        elif accuracy.get('coverage_pct', 95) > 98:
            narrowing_factor = (accuracy.get('coverage_pct', 95) - 95) / 300
            width *= (1.0 - narrowing_factor)
            lower = prediction - width/2
            upper = prediction + width/2
        expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}
        confidence = max(0, min(100, 100 - (width / expected_width.get(current_quarter, 25) * 100)))
        return lower, upper, confidence

# -------------------- Enhanced Final Predictions --------------------
def generate_enhanced_predictions(live_games_df, main_model):
    """
    Generate enhanced predictions with quarter-specific models and uncertainty estimates.
    Args:
        live_games_df: DataFrame with live game data.
        main_model: The main prediction model.
    Returns:
        DataFrame with enhanced predictions.
    """
    feature_generator = NBAFeatureGenerator(debug=True)
    quarter_system = QuarterSpecificModelSystem(feature_generator)
    quarter_system.load_models()
    uncertainty_estimator = PredictionUncertaintyEstimator()
    features_df = feature_generator.generate_all_features(live_games_df)
    expected_features = feature_generator.get_expected_features(
        enhanced=hasattr(main_model, 'feature_importances_') and len(main_model.feature_importances_) > 8
    )
    missing_features = [f for f in expected_features if f not in features_df.columns]
    if missing_features:
        print(f"Warning: Missing features for main model prediction: {missing_features}")
        available_features = [f for f in expected_features if f in features_df.columns]
        if len(available_features) >= 8:
            expected_features = available_features
        else:
            expected_features = feature_generator.get_expected_features(enhanced=False)
    main_predictions = main_model.predict(features_df[expected_features])
    historic_accuracy = None
    if hasattr(uncertainty_estimator, 'get_coverage_stats'):
        historic_accuracy = uncertainty_estimator.get_coverage_stats()
        if not historic_accuracy.empty:
            historic_accuracy = {r['quarter']: r for _, r in historic_accuracy.iterrows()}
    results = []
    for i, (idx, game) in enumerate(features_df.iterrows()):
        main_pred = main_predictions[i]
        ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(game, main_pred)
        current_quarter = int(game.get('current_quarter', 0))
        score_margin = abs(float(game.get('score_differential', 0)))
        momentum = float(game.get('cumulative_momentum', 0))
        lower_bound, upper_bound, interval_width = uncertainty_estimator.calculate_prediction_interval(
            ensemble_pred, current_quarter, score_margin, momentum
        )
        if hasattr(uncertainty_estimator, 'dynamically_adjust_interval'):
            dynamic_lower, dynamic_upper, dynamic_confidence = uncertainty_estimator.dynamically_adjust_interval(
                ensemble_pred, current_quarter, historic_accuracy
            )
        else:
            dynamic_lower, dynamic_upper, dynamic_confidence = lower_bound, upper_bound, confidence * 100
        confidence_indicator = None
        if hasattr(uncertainty_estimator, 'create_confidence_indicator'):
            confidence_indicator = uncertainty_estimator.create_confidence_indicator(
                ensemble_pred, dynamic_lower, dynamic_upper, current_quarter, 
                home_score=game.get('home_score', 0)
            )
        result = {
            'game_id': game.get('game_id'),
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'current_quarter': current_quarter,
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            'score_differential': game.get('score_differential', 0),
            'main_model_prediction': main_pred,
            'quarter_model_prediction': breakdown['quarter_model'],
            'ensemble_prediction': ensemble_pred,
            'prediction_confidence': confidence,
            'lower_bound': dynamic_lower,
            'upper_bound': dynamic_upper,
            'interval_width': dynamic_upper - dynamic_lower,
            'confidence_pct': dynamic_confidence,
            'main_weight': breakdown['weights']['main'],
            'quarter_weight': breakdown['weights']['quarter'],
            'confidence_indicator': confidence_indicator
        }
        if 'quarter_predictions' in breakdown:
            for q, score in breakdown['quarter_predictions'].items():
                result[f'predicted_{q}'] = score
        results.append(result)
    return pd.DataFrame(results)

# -------------------- Validation Framework --------------------
def validate_enhanced_predictions(num_test_games=20):
    """
    Validate enhanced prediction system on historical games.
    Args:
        num_test_games: Number of historical games to test.
    Returns:
        tuple: (DataFrame with validation results, DataFrame with improvement metrics)
    """
    from sqlalchemy import create_engine
    import config
    engine = create_engine(config.DATABASE_URL)
    query = f"""
    SELECT * FROM nba_historical_game_stats
    WHERE game_date >= CURRENT_DATE - INTERVAL '60 days'
    ORDER BY game_date DESC
    LIMIT {num_test_games}
    """
    test_games = pd.read_sql(query, engine)
    import joblib
    main_model = joblib.load(config.MODEL_PATH)
    feature_generator = NBAFeatureGenerator(debug=True)
    quarter_system = QuarterSpecificModelSystem(feature_generator)
    quarter_system.load_models()
    validation_results = []
    for _, game in test_games.iterrows():
        actual_home_score = game['home_score']
        for test_quarter in range(0, 5):
            sim_game = {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game['game_date'],
                'current_quarter': test_quarter
            }
            for q in range(1, 5):
                if q <= test_quarter:
                    sim_game[f'home_q{q}'] = game.get(f'home_q{q}', 0)
                    sim_game[f'away_q{q}'] = game.get(f'away_q{q}', 0)
                else:
                    sim_game[f'home_q{q}'] = 0
                    sim_game[f'away_q{q}'] = 0
            sim_game['home_score'] = sum([sim_game.get(f'home_q{q}', 0) for q in range(1, test_quarter + 1)])
            sim_game['away_score'] = sum([sim_game.get(f'away_q{q}', 0) for q in range(1, test_quarter + 1)])
            sim_game['score_differential'] = sim_game['home_score'] - sim_game['away_score']
            features_df = feature_generator.generate_all_features(pd.DataFrame([sim_game]))
            expected_features = feature_generator.get_expected_features(
                enhanced=hasattr(main_model, 'feature_importances_') and len(main_model.feature_importances_) > 8
            )
            missing_features = [f for f in expected_features if f not in features_df.columns]
            if missing_features:
                available_features = [f for f in expected_features if f in features_df.columns]
                if len(available_features) >= 8:
                    expected_features = available_features
                else:
                    expected_features = feature_generator.get_expected_features(enhanced=False)
            main_pred = main_model.predict(features_df[expected_features])[0]
            ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(features_df.iloc[0], main_pred)
            main_error = main_pred - actual_home_score
            ensemble_error = ensemble_pred - actual_home_score
            validation_results.append({
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'current_quarter': test_quarter,
                'actual_home_score': actual_home_score,
                'main_prediction': main_pred,
                'ensemble_prediction': ensemble_pred,
                'main_error': main_error,
                'ensemble_error': ensemble_error,
                'main_abs_error': abs(main_error),
                'ensemble_abs_error': abs(ensemble_error),
                'confidence': confidence,
                'main_weight': breakdown['weights']['main'],
                'quarter_weight': breakdown['weights']['quarter']
            })
    validation_df = pd.DataFrame(validation_results)
    error_metrics = validation_df.groupby('current_quarter').agg({
        'main_abs_error': ['mean', 'std', 'min', 'max', 'count'],
        'ensemble_abs_error': ['mean', 'std', 'min', 'max', 'count']
    })
    print("Enhanced Model Validation Results:")
    print(error_metrics)
    improvements = []
    for quarter in range(0, 5):
        quarter_data = validation_df[validation_df['current_quarter'] == quarter]
        if not quarter_data.empty:
            main_mae = quarter_data['main_abs_error'].mean()
            ensemble_mae = quarter_data['ensemble_abs_error'].mean()
            pct_improvement = ((main_mae - ensemble_mae) / main_mae) * 100 if main_mae > 0 else 0
            improvements.append({
                'quarter': quarter,
                'main_mae': main_mae,
                'ensemble_mae': ensemble_mae,
                'pct_improvement': pct_improvement,
                'sample_size': len(quarter_data)
            })
    print("\nImprovement by Quarter:")
    improvement_df = pd.DataFrame(improvements)
    print(improvement_df)
    return validation_df, improvement_df

# -------------------- Recommended Model Parameters --------------------
def get_recommended_model_params(quarter, model_type=None):
    """
    Returns optimized hyperparameters for specific quarter models.
    Args:
        quarter: Quarter number (1-4).
        model_type: Optional override for model type.
    Returns:
        dict: Hyperparameters for the recommended model type.
    """
    if model_type == "RandomForest":
        return {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        }
    elif model_type == "XGBoost":
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    if quarter == 1:
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
    else:
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }

# -------------------- Early Quarter Model Optimizer --------------------
class EarlyQuarterModelOptimizer:
    """
    Specialized optimization for early quarter predictions (Q1 and Q2)
    where traditional models struggle with limited information.
    """
    def __init__(self, feature_generator=None):
        self.feature_generator = feature_generator or NBAFeatureGenerator(debug=False)
        self.models = {'q1': {}, 'q2': {}}
        self.q1_feature_sets = {
            'basic': [
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'rolling_home_score', 'rolling_away_score'
            ],
            'matchup': [
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'rolling_home_score', 'rolling_away_score',
                'prev_matchup_diff'
            ],
            'advanced': [
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away',
                'rolling_home_score', 'rolling_away_score',
                'prev_matchup_diff', 'time_remaining_norm'
            ]
        }
        self.q2_feature_sets = {
            'basic': [
                'home_q1', 'away_q1', 'score_ratio', 
                'rest_advantage'
            ],
            'momentum': [
                'home_q1', 'away_q1', 'score_ratio', 
                'rest_advantage', 'q1_to_q2_momentum'
            ],
            'advanced': [
                'home_q1', 'away_q1', 'score_ratio', 
                'rest_advantage', 'q1_to_q2_momentum', 
                'rolling_home_score', 'rolling_away_score',
                'prev_matchup_diff', 'time_remaining_norm',
                'momentum_indicator', 'score_momentum_interaction'
            ]
        }
        
    def initialize_xgboost_models(self):
        try:
            import xgboost as xgb
            self.models['q1']['xgb_basic'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05, 
                max_depth=3,
                gamma=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            self.models['q1']['xgb_tuned'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.03, 
                max_depth=4,
                min_child_weight=2,
                gamma=1,
                subsample=0.75,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=42
            )
            self.models['q2']['xgb_basic'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1, 
                max_depth=3,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            self.models['q2']['xgb_tuned'] = xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.05, 
                max_depth=4,
                min_child_weight=1,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=42
            )
            print("XGBoost models initialized successfully")
            return True
        except ImportError:
            print("XGBoost not available. Install with: pip install xgboost")
            return False
            
    def train_models(self, training_data, target_col='home_score'):
        results = {'q1': {}, 'q2': {}}
        if not isinstance(training_data, pd.DataFrame) or training_data.empty:
            print("No training data provided")
            return results
        if 'momentum_indicator' not in training_data.columns:
            print("Adding advanced features to training data")
            training_data = self.feature_generator.add_advanced_features(training_data)
        q1_target = f"{target_col[:-5]}q1" if target_col.endswith('score') else 'home_q1'
        if q1_target in training_data.columns:
            print(f"Training Q1 models with target: {q1_target}")
            for feature_set_name, features in self.q1_feature_sets.items():
                valid_features = [f for f in features if f in training_data.columns]
                if len(valid_features) < len(features):
                    print(f"Warning: Missing {len(features)-len(valid_features)} features for Q1 {feature_set_name} set")
                if not valid_features:
                    continue
                X = training_data[valid_features].copy()
                y = training_data[q1_target]
                for model_name, model in self.models['q1'].items():
                    try:
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        mse = np.mean((y - y_pred) ** 2)
                        mae = np.mean(np.abs(y - y_pred))
                        results['q1'][f"{model_name}_{feature_set_name}"] = {
                            'model': model,
                            'features': valid_features,
                            'train_mse': mse,
                            'train_mae': mae,
                            'feature_importance': dict(zip(valid_features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                        }
                        print(f"Trained Q1 {model_name} with {feature_set_name} features: MSE={mse:.2f}, MAE={mae:.2f}")
                    except Exception as e:
                        print(f"Error training Q1 {model_name} with {feature_set_name} features: {e}")
        else:
            print(f"Target column {q1_target} not found for Q1 models")
        q2_target = f"{target_col[:-5]}q2" if target_col.endswith('score') else 'home_q2'
        if q2_target in training_data.columns:
            print(f"Training Q2 models with target: {q2_target}")
            for feature_set_name, features in self.q2_feature_sets.items():
                valid_features = [f for f in features if f in training_data.columns]
                if len(valid_features) < len(features):
                    print(f"Warning: Missing {len(features)-len(valid_features)} features for Q2 {feature_set_name} set")
                if not valid_features:
                    continue
                X = training_data[valid_features].copy()
                y = training_data[q2_target]
                for model_name, model in self.models['q2'].items():
                    try:
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        mse = np.mean((y - y_pred) ** 2)
                        mae = np.mean(np.abs(y - y_pred))
                        results['q2'][f"{model_name}_{feature_set_name}"] = {
                            'model': model,
                            'features': valid_features,
                            'train_mse': mse,
                            'train_mae': mae,
                            'feature_importance': dict(zip(valid_features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                        }
                        print(f"Trained Q2 {model_name} with {feature_set_name} features: MSE={mse:.2f}, MAE={mae:.2f}")
                    except Exception as e:
                        print(f"Error training Q2 {model_name} with {feature_set_name} features: {e}")
        else:
            print(f"Target column {q2_target} not found for Q2 models")
        return results
    
    def evaluate_models(self, test_data, target_col='home_score'):
        if not isinstance(test_data, pd.DataFrame) or test_data.empty:
            print("No test data provided")
            return pd.DataFrame()
        if 'momentum_indicator' not in test_data.columns:
            print("Adding advanced features to test data")
            test_data = self.feature_generator.add_advanced_features(test_data)
        eval_results = []
        q1_target = f"{target_col[:-5]}q1" if target_col.endswith('score') else 'home_q1'
        if q1_target in test_data.columns:
            for model_key, model_info in self.models['q1'].items():
                for feature_set_name, features in self.q1_feature_sets.items():
                    model_id = f"{model_key}_{feature_set_name}"
                    if model_id in self.models['q1']:
                        model_data = self.models['q1'][model_id]
                        model = model_data['model']
                        features = model_data['features']
                        if all(f in test_data.columns for f in features):
                            X_test = test_data[features]
                            y_test = test_data[q1_target]
                            y_pred = model.predict(X_test)
                            mse = np.mean((y_test - y_pred) ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(y_test - y_pred))
                            eval_results.append({
                                'quarter': 'Q1',
                                'model': model_key,
                                'feature_set': feature_set_name,
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'sample_size': len(X_test)
                            })
                            print(f"Q1 {model_key} with {feature_set_name} features: RMSE={rmse:.2f}, MAE={mae:.2f}")
        if q1_target == '':
            print("No evaluation results generated")
            return pd.DataFrame()
        return pd.DataFrame(eval_results)
    
    def compare_to_baseline(self, eval_df, baseline_rmse):
        if eval_df.empty:
            return pd.DataFrame()
        comparison = eval_df.copy()
        comparison['baseline_rmse'] = comparison['quarter'].apply(
            lambda q: baseline_rmse.get(q[1], 7.5)
        )
        comparison['rmse_improvement'] = comparison['baseline_rmse'] - comparison['rmse']
        comparison['pct_improvement'] = (comparison['rmse_improvement'] / comparison['baseline_rmse']) * 100
        return comparison.sort_values(['quarter', 'pct_improvement'], ascending=[True, False])

# -------------------- Enhanced Model Validation --------------------
def validate_enhanced_predictions(num_test_games=20):
    """
    Validate the enhanced prediction system on historical games.
    Args:
        num_test_games: Number of historical games to test.
    Returns:
        tuple: (DataFrame with validation results, DataFrame with improvement metrics)
    """
    from sqlalchemy import create_engine
    import config
    engine = create_engine(config.DATABASE_URL)
    query = f"""
    SELECT * FROM nba_historical_game_stats
    WHERE game_date >= CURRENT_DATE - INTERVAL '60 days'
    ORDER BY game_date DESC
    LIMIT {num_test_games}
    """
    test_games = pd.read_sql(query, engine)
    import joblib
    main_model = joblib.load(config.MODEL_PATH)
    feature_generator = NBAFeatureGenerator(debug=True)
    quarter_system = QuarterSpecificModelSystem(feature_generator)
    quarter_system.load_models()
    validation_results = []
    for _, game in test_games.iterrows():
        actual_home_score = game['home_score']
        for test_quarter in range(0, 5):
            sim_game = {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game['game_date'],
                'current_quarter': test_quarter
            }
            for q in range(1, 5):
                if q <= test_quarter:
                    sim_game[f'home_q{q}'] = game.get(f'home_q{q}', 0)
                    sim_game[f'away_q{q}'] = game.get(f'away_q{q}', 0)
                else:
                    sim_game[f'home_q{q}'] = 0
                    sim_game[f'away_q{q}'] = 0
            sim_game['home_score'] = sum([sim_game.get(f'home_q{q}', 0) for q in range(1, test_quarter + 1)])
            sim_game['away_score'] = sum([sim_game.get(f'away_q{q}', 0) for q in range(1, test_quarter + 1)])
            sim_game['score_differential'] = sim_game['home_score'] - sim_game['away_score']
            features_df = feature_generator.generate_all_features(pd.DataFrame([sim_game]))
            expected_features = feature_generator.get_expected_features(
                enhanced=hasattr(main_model, 'feature_importances_') and len(main_model.feature_importances_) > 8
            )
            missing_features = [f for f in expected_features if f not in features_df.columns]
            if missing_features:
                available_features = [f for f in expected_features if f in features_df.columns]
                if len(available_features) >= 8:
                    expected_features = available_features
                else:
                    expected_features = feature_generator.get_expected_features(enhanced=False)
            main_pred = main_model.predict(features_df[expected_features])[0]
            ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(features_df.iloc[0], main_pred)
            main_error = main_pred - actual_home_score
            ensemble_error = ensemble_pred - actual_home_score
            validation_results.append({
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'current_quarter': test_quarter,
                'actual_home_score': actual_home_score,
                'main_prediction': main_pred,
                'ensemble_prediction': ensemble_pred,
                'main_error': main_error,
                'ensemble_error': ensemble_error,
                'main_abs_error': abs(main_error),
                'ensemble_abs_error': abs(ensemble_error),
                'confidence': confidence,
                'main_weight': breakdown['weights']['main'],
                'quarter_weight': breakdown['weights']['quarter']
            })
    validation_df = pd.DataFrame(validation_results)
    error_metrics = validation_df.groupby('current_quarter').agg({
        'main_abs_error': ['mean', 'std', 'min', 'max', 'count'],
        'ensemble_abs_error': ['mean', 'std', 'min', 'max', 'count']
    })
    print("Enhanced Model Validation Results:")
    print(error_metrics)
    improvements = []
    for quarter in range(0, 5):
        quarter_data = validation_df[validation_df['current_quarter'] == quarter]
        if not quarter_data.empty:
            main_mae = quarter_data['main_abs_error'].mean()
            ensemble_mae = quarter_data['ensemble_abs_error'].mean()
            pct_improvement = ((main_mae - ensemble_mae) / main_mae) * 100 if main_mae > 0 else 0
            improvements.append({
                'quarter': quarter,
                'main_mae': main_mae,
                'ensemble_mae': ensemble_mae,
                'pct_improvement': pct_improvement,
                'sample_size': len(quarter_data)
            })
    print("\nImprovement by Quarter:")
    improvement_df = pd.DataFrame(improvements)
    print(improvement_df)
    return validation_df, improvement_df

# -------------------- Recommended Model Parameters --------------------
def get_recommended_model_params(quarter, model_type=None):
    """
    Returns optimized hyperparameters for specific quarter models.
    Args:
        quarter: Quarter number (1-4).
        model_type: Optional override for model type.
    Returns:
        dict: Hyperparameters for the recommended model type.
    """
    if model_type == "RandomForest":
        return {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        }
    elif model_type == "XGBoost":
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    if quarter == 1:
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
    else:
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
