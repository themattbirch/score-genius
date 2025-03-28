# feature_engineering.py - Unified Feature Engineering for NBA Score Prediction

"""
NBAFeatureEngine - Unified module for NBA score prediction feature engineering.
This module includes:
  - Comprehensive feature engineering for NBA games
  - Enhanced dynamic ensemble weighting
  - A robust quarter-specific model system with fallbacks
  - An uncertainty estimation module
  - Enhanced final prediction generation
  - A validation framework
"""

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
from functools import wraps
import scipy.stats as stats
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Union, Any

# -------------------- Simple Logging Function --------------------
def log(message, level="INFO", debug=False):
    """Simple logging function with timestamps."""
    if debug or level != "DEBUG":
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")

# -------------------- NBAFeatureEngine Class --------------------
class NBAFeatureEngine:
    """Core feature engineering for NBA score prediction models."""
    
    # Modify the __init__ method
    def __init__(self, supabase_client: Optional[Any] = None, debug: bool = False): # Make client optional if not always needed
        """
        Initialize the feature engine.

        Args:
            supabase_client: Optional configured Supabase client instance.
                             Required if using methods that interact with Supabase.
            debug: Whether to print debug messages
        """
        self.debug = debug
        self.supabase_client = supabase_client # Store the passed client

        # You might want to add a check if certain operations require the client
        # if self.supabase_client is None:
        #    self.log("Supabase client not provided. Some features relying on DB access may fail.", "WARNING")

        
        # NBA average values for fallback situations
        self.defaults = {
            'win_pct': 0.5,
            'offensive_rating': 110.0,
            'defensive_rating': 110.0,
            'pace': 98.5,
            'home_advantage': 3.5,
            'quarter_points': 25.0,
            'estimated_possessions': 198,   
            'avg_pts_for': 110.0            
        }
        
        # League-wide averages
        self.league_averages = {
            'score': 110.0,
            'quarter_scores': {1: 27.5, 2: 27.5, 3: 27.0, 4: 28.0}
        }
        
        # Quarter-specific feature sets
        self.quarter_feature_sets = self._get_optimized_feature_sets()
    
    def log(self, message, level="INFO"):
        """Print log messages if debug mode is on."""
        if self.debug or level != "DEBUG":
            print(f"[NBA Engine] {message}")

    def generate_all_features(self, df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None, # If used by any sub-method
                              rolling_window: int = 10):
        """
        Applies all feature engineering steps in sequence to the input DataFrame.

        Args:
            df: DataFrame containing game data (can be live or historical).
                Requires base columns like game_id, game_date, home_team, away_team.
                May require score columns (home_score, away_score) if available
                and needed for history/rolling features on historical runs.
            historical_games_df: DataFrame of historical games needed for rest
                                 and potentially history features.
            team_stats_df: DataFrame of team stats, if needed by any feature step.
            rolling_window: Window size for rolling features.

        Returns:
            DataFrame with all features added, or original DataFrame on critical error.
        """
        if df is None or df.empty:
            self.log("Input DataFrame is empty for generate_all_features.", "WARNING")
            return df

        self.log("Starting comprehensive feature generation pipeline...", level="INFO")
        features_df = df.copy()

        try:
            # --- Sequence of Feature Engineering Steps ---

            # 1. Add basic rolling averages (using the existing method or a vectorized one)
            # Assuming 'add_rolling_features' is suitable or replace with vectorized version name
            features_df = self.add_rolling_features(features_df, window_sizes=[rolling_window])
            self.log("Step 1/4: Rolling features added.", level="DEBUG")

            # 2. Add team history (rolling scores & prev matchup diff - use vectorized)
            # Ensure home_score, away_score are present if using historical data
            if 'home_score' not in features_df.columns and historical_games_df is not None:
                # If predicting future games, these might not exist yet. History relies on past scores.
                 self.log("Home/Away scores missing, needed for history calc. Merging from historical if possible.", level="DEBUG")
                 # This merge might be complex depending on keys. Simplification: History features might only work reliably on historical data.
                 pass # Or attempt merge: features_df = pd.merge(features_df, historical_games_df[['game_id', 'home_score', 'away_score']], on='game_id', how='left')

            features_df = self.add_team_history_features_vectorized(features_df, rolling_window=rolling_window)
            self.log("Step 2/4: Team history features (rolling, matchup diff) added.", level="DEBUG")

            # 3. Add rest days and back-to-back features (use vectorized)
            # Requires historical_games_df for accurate calculation unless df itself contains enough history
            features_df = self.add_rest_features_vectorized(features_df, historical_df=historical_games_df)
            self.log("Step 3/4: Rest features added.", level="DEBUG")

            # 4. Add advanced metrics (eFG%, Pace, Possessions, Efficiency)
            # Requires basic box score stats (FGM/A, 3PM/A, FTM/A, REB, TOV, etc.)
            # Ensure these columns exist in features_df before calling
            features_df = self.integrate_advanced_features(features_df)
            self.log("Step 4/4: Advanced features (shooting, pace, efficiency) integrated.", level="DEBUG")

            # --- Add any other feature generation steps here ---
            # e.g., self.add_momentum_features(features_df)
            # e.g., self.add_betting_line_features(features_df)

            self.log("Comprehensive feature generation pipeline complete.", level="INFO")
            return features_df

        except Exception as e:
            self.log(f"Error during feature generation pipeline: {e}", "ERROR")
            traceback.print_exc()
            # Return the DataFrame as it was before the error, or the original df?
            # Returning original df might be safer if partial features are problematic.
            return df

    @staticmethod
    def profile_time(func=None, debug_mode=None):
        """
        Decorator to profile the execution time of a function.
        Can be used with or without parameters.
        
        Args:
            func: The function to decorate (None if used with parameters)
            debug_mode: Override debug mode setting (None to use class debug attribute)
            
        Examples:
            @profile_time
            def my_func(): ...
            
            @profile_time(debug_mode=True)
            def my_other_func(): ...
        """
        # Handle case when decorator is used without parameters
        if func is not None:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Record start time
                start_time = time.time()
                
                # Run the function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Format the message
                message = f"{func.__name__} executed in {execution_time:.4f} seconds"
                
                # Determine how to output the message
                if args and hasattr(args[0], '_print_debug'):
                    # If class has debug method, use it
                    args[0]._print_debug(message)
                elif args and hasattr(args[0], 'debug') and args[0].debug:
                    # If class has debug flag set to True
                    print(message)
                elif debug_mode:
                    # If debug_mode parameter is True
                    print(message)
                
                return result
            return wrapper
        
        # Handle case when decorator is used with parameters
        else:
            def decorator(f):
                @functools.wraps(f)
                def wrapper(*args, **kwargs):
                    # Record start time
                    start_time = time.time()
                    
                    # Run the function
                    result = f(*args, **kwargs)
                    
                    # Calculate execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Format the message
                    message = f"{f.__name__} executed in {execution_time:.4f} seconds"
                    
                    # Determine how to output the message
                    if args and hasattr(args[0], '_print_debug'):
                        # If class has debug method, use it
                        args[0]._print_debug(message)
                    elif args and hasattr(args[0], 'debug') and args[0].debug:
                        # If class has debug flag set to True
                        print(message)
                    elif debug_mode:
                        # If debug_mode parameter is True
                        print(message)
                    
                    return result
                return wrapper
            return decorator

    @profile_time
    def integrate_advanced_features(self, df):
        """
        Add advanced shooting, possession, pace, and efficiency metrics using available game stats.

        Args:
            df: DataFrame with game data including FGM/A, 3PM/A, FTM/A, OReb, DReb, TOV, Score.

        Returns:
            DataFrame with added advanced features.
        """
        self.log("Adding advanced features (Shooting, Poss, Pace, Efficiency)...")
        result_df = df.copy()

        # Define required columns for calculation checks
        shooting_cols = ['home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted']
        three_cols = ['home_3pm', 'home_3pa', 'away_3pm', 'away_3pa']
        ft_cols = ['home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted']
        reb_cols = ['home_off_reb', 'home_def_reb', 'away_off_reb', 'away_def_reb']
        turnover_cols = ['home_turnovers', 'away_turnovers']
        score_cols = ['home_score', 'away_score'] # Needed for efficiency

        all_req_cols = shooting_cols + three_cols + ft_cols + reb_cols + turnover_cols + score_cols

        # Check if required columns exist
        missing_cols = [col for col in all_req_cols if col not in result_df.columns]
        if missing_cols:
            self.log(f"Missing columns required for advanced features: {missing_cols}. Skipping some calculations.", "WARNING")
            # Fill missing numeric columns with 0 to prevent errors later
            for col in missing_cols:
                 if col in all_req_cols and col not in result_df.columns: # Avoid adding if only some exist
                     if any(kw in col for kw in ['made', 'attempted', 'pm', 'pa', 'reb', 'turnovers', 'score']):
                          result_df[col] = 0

        # Convert required columns to numeric, filling NaNs with 0
        for col in all_req_cols:
             if col in result_df.columns:
                  result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # --- Calculate Shooting Metrics ---
        # FG% (Vectorized)
        result_df['home_fg_pct'] = (result_df['home_fg_made'] / result_df['home_fg_attempted'].replace(0, 1)).fillna(0)
        result_df['away_fg_pct'] = (result_df['away_fg_made'] / result_df['away_fg_attempted'].replace(0, 1)).fillna(0)
        result_df['fg_pct_diff'] = result_df['home_fg_pct'] - result_df['away_fg_pct']

        # 3P% (Vectorized)
        result_df['home_3p_pct'] = (result_df['home_3pm'] / result_df['home_3pa'].replace(0, 1)).fillna(0)
        result_df['away_3p_pct'] = (result_df['away_3pm'] / result_df['away_3pa'].replace(0, 1)).fillna(0)

        # eFG% (Vectorized)
        result_df['home_efg_pct'] = ((result_df['home_fg_made'] + 0.5 * result_df['home_3pm']) / result_df['home_fg_attempted'].replace(0, 1)).fillna(0)
        result_df['away_efg_pct'] = ((result_df['away_fg_made'] + 0.5 * result_df['away_3pm']) / result_df['away_fg_attempted'].replace(0, 1)).fillna(0)
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']

        # FT% and FT Rate (Vectorized)
        result_df['home_ft_pct'] = (result_df['home_ft_made'] / result_df['home_ft_attempted'].replace(0, 1)).fillna(0)
        result_df['away_ft_pct'] = (result_df['away_ft_made'] / result_df['away_ft_attempted'].replace(0, 1)).fillna(0)
        result_df['home_ft_rate'] = (result_df['home_ft_attempted'] / result_df['home_fg_attempted'].replace(0, 1)).fillna(0) # FTAs per FGA
        result_df['away_ft_rate'] = (result_df['away_ft_attempted'] / result_df['away_fg_attempted'].replace(0, 1)).fillna(0)
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']


        # --- Calculate Possessions, Pace, and Efficiency ---
        # Using the formula: Poss ≈ FGA + 0.44*FTA - 1.07*(OREB% * MissedFG) + TOV
        # Where OREB% = OREB / (OREB + Opp DREB)
        # Note: 0.44 factor is more common than 0.4 used in original code.

        # Calculate OREB% safely (avoid division by zero)
        home_oreb_pct = (result_df['home_off_reb'] / (result_df['home_off_reb'] + result_df['away_def_reb']).replace(0, 1)).fillna(0)
        away_oreb_pct = (result_df['away_off_reb'] / (result_df['away_off_reb'] + result_df['home_def_reb']).replace(0, 1)).fillna(0)

        # Calculate missed FGs
        home_missed_fg = result_df['home_fg_attempted'] - result_df['home_fg_made']
        away_missed_fg = result_df['away_fg_attempted'] - result_df['away_fg_made']

        # Calculate possessions
        result_df['home_possessions'] = (result_df['home_fg_attempted'] +
                                       0.44 * result_df['home_ft_attempted'] -
                                       1.07 * home_oreb_pct * home_missed_fg +
                                       result_df['home_turnovers']).fillna(self.defaults['estimated_possessions'])
        result_df['away_possessions'] = (result_df['away_fg_attempted'] +
                                       0.44 * result_df['away_ft_attempted'] -
                                       1.07 * away_oreb_pct * away_missed_fg +
                                       result_df['away_turnovers']).fillna(self.defaults['estimated_possessions'])

        # --- Correct Pace Calculation ---
        # Assuming 48 minutes regulation. Handling OT robustly is complex, omit for MVP.
        # Avoid division by zero if possessions are zero.
        minutes_played = 48.0 # Assume regulation for MVP
        result_df['home_pace'] = (result_df['home_possessions'] / minutes_played * 48.0).fillna(self.defaults.get('pace', 98.0))
        result_df['away_pace'] = (result_df['away_possessions'] / minutes_played * 48.0).fillna(self.defaults.get('pace', 98.0))
        result_df['game_pace'] = (result_df['home_pace'] + result_df['away_pace']) / 2 # Simple average game pace
        result_df['pace_differential'] = result_df['home_pace'] - result_df['away_pace']


        # --- Calculate Offensive & Defensive Efficiency ---
        # Pts per 100 Possessions (estimated)
        result_df['home_offensive_efficiency'] = (result_df['home_score'] * 100 / result_df['home_possessions'].replace(0, 1)).fillna(self.defaults['avg_pts_for']) # Use default if poss=0
        result_df['away_offensive_efficiency'] = (result_df['away_score'] * 100 / result_df['away_possessions'].replace(0, 1)).fillna(self.defaults['avg_pts_for'])

        # Defensive efficiency is opponent's offensive efficiency
        result_df['home_defensive_efficiency'] = result_df['away_offensive_efficiency']
        result_df['away_defensive_efficiency'] = result_df['home_offensive_efficiency']
        result_df['efficiency_differential'] = result_df['home_offensive_efficiency'] - result_df['home_defensive_efficiency'] # Home Off Eff - Home Def Eff

        # --- Score Convenience Features ---
        if 'total_score' not in result_df.columns:
            result_df['total_score'] = result_df['home_score'] + result_df['away_score']
        if 'point_diff' not in result_df.columns:
            result_df['point_diff'] = result_df['home_score'] - result_df['away_score']

        return result_df
    
    def _get_optimized_feature_sets(self) -> Dict[int, List[str]]:
        """
        Get optimized feature sets for each quarter.
        
        Returns:
            Dictionary mapping quarter to list of optimal features
        """
        # These feature sets focus on the most predictive features by quarter
        return {
            1: [
                'rolling_home_score', 'rolling_away_score', 'prev_matchup_diff',
                'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away'
            ],
            2: [
                'home_q1', 'away_q1', 
                'rolling_home_score', 'rolling_away_score',
                'q1_to_q2_momentum', 'score_differential'
            ],
            3: [
                'home_q1', 'home_q2', 'away_q1', 'away_q2',
                'first_half_diff', 'score_differential',
                'q2_to_q3_momentum', 'cumulative_momentum'
            ],
            4: [
                'home_q1', 'home_q2', 'home_q3', 'away_q1', 'away_q2', 'away_q3',
                'pre_q4_diff', 'score_differential',
                'cumulative_momentum', 'is_clutch_time'
            ]
        }
    
    def normalize_team_name(self, team_name):
        """
        Normalize team name for consistent lookup.
        
        Args:
            team_name: Raw team name string
            
        Returns:
            Normalized team name
        """
        if not isinstance(team_name, str):
            return ""
        
        # Common name variations
        mapping = {
            "sixers": "76ers",
            "phila": "76ers",
            "philadelphia": "76ers",
            "trail blazers": "blazers",
            "clippers": "la clippers",
            "los angeles clippers": "la clippers",
            "lakers": "la lakers", 
            "los angeles lakers": "la lakers",
            "new york": "knicks",
            "golden state": "warriors",
            "san antonio": "spurs"
        }
        
        team_lower = team_name.lower()
        for key, value in mapping.items():
            if key in team_lower:
                return value
        
        return team_name
    
    def add_rolling_features(self, df, window_sizes=[5, 10]):
        """
        Add rolling average features for scoring and defense.
        
        Args:
            df: DataFrame with game data
            window_sizes: List of window sizes to calculate rolling averages for
        
        Returns:
            DataFrame with added rolling features
        """
        try:
            if df is None or df.empty:
                self.log("No data provided for rolling features", "WARNING")
                return df
            
            result_df = df.copy()
            
            # Ensure date column is datetime and sort
            if 'game_date' in result_df.columns:
                result_df['game_date'] = pd.to_datetime(result_df['game_date'])
                result_df = result_df.sort_values('game_date')
            
            # Get team and score columns to make sure they exist
            req_cols = ['home_team', 'away_team', 'home_score', 'away_score']
            missing_cols = [col for col in req_cols if col not in result_df.columns]
            
            if missing_cols:
                self.log(f"Missing required columns: {missing_cols}", "WARNING")
                # Fill in missing columns with defaults
                for col in missing_cols:
                    if 'score' in col:
                        result_df[col] = self.league_averages['score']
                    else:
                        result_df[col] = 'Unknown'
            
            # Use vectorized operations for better performance
            for window in window_sizes:
                # Home team stats
                result_df[f'home_last{window}_avg'] = result_df.groupby('home_team')['home_score'].transform(
                    lambda x: x.shift(1).rolling(window=min(window, len(x)), min_periods=1).mean()
                )
                
                result_df[f'home_last{window}_allowed'] = result_df.groupby('home_team')['away_score'].transform(
                    lambda x: x.shift(1).rolling(window=min(window, len(x)), min_periods=1).mean()
                )
                
                # Away team stats
                result_df[f'away_last{window}_avg'] = result_df.groupby('away_team')['away_score'].transform(
                    lambda x: x.shift(1).rolling(window=min(window, len(x)), min_periods=1).mean()
                )
                
                result_df[f'away_last{window}_allowed'] = result_df.groupby('away_team')['home_score'].transform(
                    lambda x: x.shift(1).rolling(window=min(window, len(x)), min_periods=1).mean()
                )
            
            # Fill NaN values with overall means
            for col in result_df.columns:
                if 'last' in col and result_df[col].isna().any():
                    if 'avg' in col:
                        result_df[col] = result_df[col].fillna(self.league_averages['score'])
                    elif 'allowed' in col:
                        result_df[col] = result_df[col].fillna(self.league_averages['score'])
            
            # Add standardized column names for consistency
            if 'home_last5_avg' in result_df.columns:
                result_df['home_rolling_score'] = result_df['home_last5_avg']
                result_df['away_rolling_score'] = result_df['away_last5_avg']
                result_df['home_rolling_opp_score'] = result_df['home_last5_allowed']
                result_df['away_rolling_opp_score'] = result_df['away_last5_allowed']
            
            return result_df
            
        except Exception as e:
            self.log(f"Error adding rolling features: {str(e)}", "ERROR")
            traceback.print_exc()
            return df
    
    def add_team_history_features_vectorized(self, df, rolling_window=10):
        """
        [Vectorized] Add team history features like rolling averages and previous matchup results.

        Args:
            df: DataFrame with team and game data, MUST include 'game_id', 'game_date',
                'home_team', 'away_team', 'home_score', 'away_score'.
            rolling_window: The window size for calculating rolling score averages.

        Returns:
            DataFrame with added team history features ('rolling_home_score',
            'rolling_away_score', 'prev_matchup_diff'). Returns original df on critical error.
        """
        if df is None or df.empty:
            self.log("No data provided for team history features", "WARNING")
            return df

        self.log("Adding team history features (Vectorized)...")
        result_df = df.copy()

        # --- Check for required columns ---
        required_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score']
        missing_cols = [col for col in required_cols if col not in result_df.columns]

        if missing_cols:
            self.log(f"Missing required columns for team history (vectorized): {missing_cols}. Cannot proceed.", "ERROR")
            # Add empty columns to avoid breaking downstream code, but return early?
            # Or return original df? Returning original for safety.
            # You might want to raise an error instead.
            # for col in ['rolling_home_score', 'rolling_away_score', 'prev_matchup_diff']:
            #    if col not in result_df.columns:
            #        result_df[col] = 0
            return df # Return original DataFrame if critical columns are missing

        # --- Data Preparation ---
        try:
            result_df['game_date'] = pd.to_datetime(result_df['game_date'])
            # Sort by date for rolling calculations and shift logic
            result_df = result_df.sort_values('game_date').reset_index(drop=True)
        except Exception as e:
            self.log(f"Error converting game_date to datetime or sorting: {e}", "ERROR")
            return df # Return original df on critical error

        # --- Calculate Rolling Scores (Vectorized) ---
        self.log(f"Calculating rolling scores with window={rolling_window}...", level="DEBUG")
        # Create a temporary structure for calculating team-specific rolling averages
        # Melt home and away games into a single team-centric view
        home_scores_df = result_df[['game_date', 'home_team', 'home_score']].rename(
            columns={'home_team': 'team', 'home_score': 'score_for'}
        )
        away_scores_df = result_df[['game_date', 'away_team', 'away_score']].rename(
            columns={'away_team': 'team', 'away_score': 'score_for'}
        )
        team_scores_df = pd.concat([home_scores_df, away_scores_df], ignore_index=True)
        team_scores_df = team_scores_df.sort_values(['team', 'game_date'])

        # Calculate rolling average of PAST scores for each team
        # shift(1) ensures we don't include the current game's score
        # min_periods=1 handles the start of a team's history
        team_scores_df[f'rolling_score_avg'] = team_scores_df.groupby('team')['score_for']\
            .transform(lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).mean())

        # Fill initial NaNs (first games for each team) with league average score
        league_avg_score = self.league_averages.get('score', 110.0)
        team_scores_df[f'rolling_score_avg'] = team_scores_df[f'rolling_score_avg'].fillna(league_avg_score)

        # Merge rolling averages back into the main DataFrame
        # Merge for home team
        result_df = pd.merge(
            result_df,
            team_scores_df[['game_date', 'team', 'rolling_score_avg']],
            how='left',
            left_on=['game_date', 'home_team'],
            right_on=['game_date', 'team']
        ).rename(columns={'rolling_score_avg': 'rolling_home_score'}).drop(columns='team')

        # Merge for away team
        result_df = pd.merge(
            result_df,
            team_scores_df[['game_date', 'team', 'rolling_score_avg']],
            how='left',
            left_on=['game_date', 'away_team'],
            right_on=['game_date', 'team']
        ).rename(columns={'rolling_score_avg': 'rolling_away_score'}).drop(columns='team')

        # --- Calculate Previous Matchup Differential (Vectorized) ---
        self.log("Calculating previous matchup differential...", level="DEBUG")

        # Create a unique, order-invariant matchup key
        result_df['matchup_key'] = result_df.apply(
            lambda row: tuple(sorted((row['home_team'], row['away_team']))),
            axis=1
        )

        # Sort by matchup key and date to prepare for shift
        result_df = result_df.sort_values(['matchup_key', 'game_date'])

        # Get previous game's details within the same matchup group using shift
        cols_to_shift = ['home_team', 'away_team', 'home_score', 'away_score']
        shifted_cols = result_df.groupby('matchup_key')[cols_to_shift].shift(1)
        shifted_cols.columns = [f'prev_{col}' for col in cols_to_shift] # Rename shifted columns

        # Join shifted data back - using join on index since we sorted
        result_df = result_df.join(shifted_cols)

        # Fill NaNs in previous scores with 0 before calculation to avoid errors
        result_df['prev_home_score'] = result_df['prev_home_score'].fillna(0)
        result_df['prev_away_score'] = result_df['prev_away_score'].fillna(0)

        # Calculate diff using np.select for vectorized conditional logic
        conditions = [
            result_df['prev_home_team'].isna(),             # Cond 1: No previous matchup recorded for this pair
            result_df['home_team'] == result_df['prev_home_team'] # Cond 2: Current home team was also home team in the previous matchup
            # Cond 3: Current home team was away team in previous matchup (implicit else)
        ]
        choices = [
            0,                                                  # Choice 1: Diff is 0 if no prior matchup
            result_df['prev_home_score'] - result_df['prev_away_score'], # Choice 2: Use previous game's home-away diff
            result_df['prev_away_score'] - result_df['prev_home_score']  # Choice 3: Use previous game's away-home diff
        ]

        result_df['prev_matchup_diff'] = np.select(conditions, choices, default=0)

        # --- Final Cleanup ---
        # Ensure numeric types and fill any remaining NaNs in result columns
        numeric_cols = ['rolling_home_score', 'rolling_away_score', 'prev_matchup_diff']
        for col in numeric_cols:
            if col in result_df.columns:
                # Fill NaNs that might have occurred during merges (though should be minimal with how='left')
                # Use appropriate defaults (e.g., league avg for scores, 0 for diff)
                fill_val = league_avg_score if 'score' in col else 0
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(fill_val)
            else:
                # If a column wasn't created due to error, add it with default
                 fill_val = league_avg_score if 'score' in col else 0
                 result_df[col] = fill_val
                 self.log(f"Column '{col}' was missing, added with default value {fill_val}.", "WARNING")


        # Remove temporary columns
        result_df = result_df.drop(columns=['matchup_key', 'prev_home_team', 'prev_away_team',
                                             'prev_home_score', 'prev_away_score'], errors='ignore')

        # Restore original sort order? Optional, depends if downstream code expects it.
        # If df had an index before reset_index, you might want to restore it.
        # For now, it remains sorted by game_date from the initial step.

        self.log("Finished adding team history features (Vectorized).")
        return result_df

    # Example (Conceptual) for add_rest_features vectorization
    def add_rest_features_vectorized(self, df, historical_df=None):
        # ... (Initial checks and date conversion)

        # Combine data if necessary and sort
        combined_df = pd.concat([historical_df, df], ignore_index=True).drop_duplicates(subset=['game_id']) # Ensure unique games
        combined_df['game_date'] = pd.to_datetime(combined_df['game_date'])
        combined_df = combined_df.sort_values('game_date')

        # Melt to get one row per team per game
        home_melt = combined_df[['game_id', 'game_date', 'home_team']].rename(columns={'home_team': 'team'})
        away_melt = combined_df[['game_id', 'game_date', 'away_team']].rename(columns={'away_team': 'team'})
        team_dates = pd.concat([home_melt, away_melt]).sort_values(['team', 'game_date'])

        # Calculate last game date for each team
        team_dates['last_game_date'] = team_dates.groupby('team')['game_date'].shift(1)

        # Merge back into the original df structure (focusing on the games in the input 'df')
        result_df = df.copy()
        result_df['game_date'] = pd.to_datetime(result_df['game_date'])

        # Merge last game date for home team
        result_df = pd.merge(
            result_df,
            team_dates[['game_id', 'team', 'last_game_date']],
            how='left',
            left_on=['game_id', 'home_team'],
            right_on=['game_id', 'team']
        ).rename(columns={'last_game_date': 'home_last_game_date'}).drop(columns='team')

        # Merge last game date for away team
        result_df = pd.merge(
            result_df,
            team_dates[['game_id', 'team', 'last_game_date']],
            how='left',
            left_on=['game_id', 'away_team'],
            right_on=['game_id', 'team']
        ).rename(columns={'last_game_date': 'away_last_game_date'}).drop(columns='team')

        # Calculate rest days (handle NaT for first games)
        default_rest = 2
        result_df['rest_days_home'] = (result_df['game_date'] - result_df['home_last_game_date']).dt.days.fillna(default_rest).clip(upper=4)
        result_df['rest_days_away'] = (result_df['game_date'] - result_df['away_last_game_date']).dt.days.fillna(default_rest).clip(upper=4)

        result_df['is_back_to_back_home'] = (result_df['rest_days_home'] <= 1).astype(int)
        result_df['is_back_to_back_away'] = (result_df['rest_days_away'] <= 1).astype(int)
        result_df['rest_advantage'] = result_df['rest_days_home'] - result_df['rest_days_away']

        # Drop temporary merge columns
        result_df = result_df.drop(columns=['home_last_game_date', 'away_last_game_date'])

        # Ensure numeric types
        rest_columns = ['rest_days_home', 'rest_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage']
        for col in rest_columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        self.log("Added rest features (vectorized).")
        return result_df

    def _calculate_team_rest(self, team, game_date, team_last_game, historical_df=None):
        """
        Helper function to calculate rest days for a team.
        
        Args:
            team: Team name
            game_date: Date of current game
            team_last_game: Dictionary tracking team's last game
            historical_df: Optional historical dataframe
            
        Returns:
            Number of rest days
        """
        default_rest = 2  # Default rest days if no previous game
        
        # Try to find team's last game from historical data if provided
        if historical_df is not None and not historical_df.empty:
            prev_games = historical_df[
                ((historical_df['home_team'] == team) | (historical_df['away_team'] == team)) &
                (historical_df['game_date'] < game_date)
            ].sort_values('game_date', ascending=False)
            
            if not prev_games.empty:
                last_game_date = prev_games.iloc[0]['game_date']
                rest_days = (game_date - last_game_date).days
                return min(rest_days, 4)  # Cap at realistic maximum
        
        # Otherwise use tracking dictionary
        if team in team_last_game:
            rest_days = (game_date - team_last_game[team]).days
            return min(rest_days, 4)  # Cap at realistic maximum
        
        return default_rest

from caching.supabase_client import supabase

# -------------------- Data Fetching Class --------------------
class SupabaseDataFetcher:
    """ Handles data fetching from Supabase for NBA prediction models. """
    def __init__(self, supabase_client: Any, debug: bool = False): # Expects client
        """ Initialize with a Supabase client. """
        if supabase_client is None:
             raise ValueError("Supabase client must be provided to SupabaseDataFetcher.")
        self.supabase = supabase_client # Use the passed client
        self.debug = debug

    def _print_debug(self, message):
        if self.debug:
            print(f"[{type(self).__name__}] {message}")

    def fetch_historical_games(self, days_lookback=365):
        """ Load historical games for the lookback period with efficient pagination. """
        threshold_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
        self._print_debug(f"Loading historical game data since {threshold_date}")

        all_data = []
        page_size = 1000
        start_index = 0
        try:
            while True:
                response = self.supabase.table("nba_historical_game_stats") \
                    .select("*") \
                    .gte("game_date", threshold_date) \
                    .order('game_date') \
                    .range(start_index, start_index + page_size - 1) \
                    .execute()

                batch_data = response.data
                batch_size = len(batch_data)
                all_data.extend(batch_data)
                self._print_debug(f"Retrieved batch of {batch_size} records, total: {len(all_data)}")

                if batch_size < page_size:
                    break # Last page
                start_index += page_size

            if not all_data:
                self._print_debug(f"No historical game data found since {threshold_date}.")
                return pd.DataFrame() # Return empty df, let caller handle fallback

            df = pd.DataFrame(all_data)
            df['game_date'] = pd.to_datetime(df['game_date'])
            # Convert known numeric columns
            numeric_cols = [ # List all expected numeric cols from schema...
                 'home_score', 'away_score', 'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot',
                 'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_ot', 'home_assists', 'away_assists',
                 # ... include all numeric columns listed in schema
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            return df.sort_values('game_date').reset_index(drop=True)

        except Exception as e:
            self._print_debug(f"Error loading historical games: {e}")
            traceback.print_exc()
            return pd.DataFrame() # Return empty df on error


    def fetch_team_stats(self):
        """ Fetch team performance stats from nba_historical_team_stats. """
        self._print_debug("Fetching team stats...")
        try:
            # Adjust table name if different (e.g., 'nba_historical_team_stats')
            response = self.supabase.table("nba_historical_team_stats").select("*").execute()
            data = response.data
            if data:
                self._print_debug(f"Fetched {len(data)} team stat records")
                return pd.DataFrame(data)
            else:
                self._print_debug("No team stats found")
                return pd.DataFrame()
        except Exception as e:
            self._print_debug(f"Error fetching team stats: {e}")
            return pd.DataFrame()

    def fetch_upcoming_games(self, days_ahead=7):
        """ Fetch upcoming games from Supabase. """
        self._print_debug(f"Fetching upcoming games for next {days_ahead} days...")
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

            response = self.supabase.table("nba_upcoming_games") \
                .select("*, home_team:home_team_id(*), away_team:away_team_id(*)") \
                .gte("game_date", today) \
                .lte("game_date", future_date) \
                .execute()

            data = response.data
            if data:
                self._print_debug(f"Fetched {len(data)} upcoming games")
                df = pd.DataFrame(data)
                # Flatten nested team data if necessary (adjust based on actual response structure)
                if 'home_team' in df.columns and isinstance(df['home_team'].iloc[0], dict):
                    df['home_team_name'] = df['home_team'].apply(lambda x: x.get('team_name', 'Unknown') if isinstance(x, dict) else 'Unknown')
                    df['away_team_name'] = df['away_team'].apply(lambda x: x.get('team_name', 'Unknown') if isinstance(x, dict) else 'Unknown')
                    df = df.drop(columns=['home_team', 'away_team']) # Drop original nested columns
                    # Rename columns to match expected 'home_team', 'away_team'
                    df = df.rename(columns={'home_team_name': 'home_team', 'away_team_name': 'away_team'})

                df['game_date'] = pd.to_datetime(df['game_date'])
                return df
            else:
                self._print_debug("No upcoming games found")
                return pd.DataFrame()
        except Exception as e:
            self._print_debug(f"Error fetching upcoming games: {e}")
            return pd.DataFrame()


# -------------------- Ensemble Weight Manager --------------------
class EnsembleWeightManager:
    """
    Manages ensemble weights, including base strategies and dynamic adjustments,
    to calculate the final ensemble prediction.
    """
    def __init__(self, error_history: Optional[Dict[str, Dict[int, float]]] = None, debug: bool = False):
        self.error_history = error_history or {
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.weighting_strategies = {
            'standard': self._standard_weights,
            'adaptive': self._adaptive_weights,
        }
        self.weight_history: List[Dict[str, Any]] = []
        # Constants for adaptive weighting and heuristic adjustments
        self.BASE_WEIGHT_MIN = 0.6  # Min weight for main model in adaptive strategy
        self.BASE_WEIGHT_MAX = 0.95 # Max weight for main model in adaptive strategy
        self.HEURISTIC_MAIN_WEIGHT_MAX = 0.95 # Overall cap after heuristics
        self.HEURISTIC_QUARTER_WEIGHT_MAX = 0.4 # Cap for quarter weight after heuristics (implicitly caps main at 0.6)

        self.HISTORIC_WEIGHT_SMOOTHING = 0.7 # Smoothing factor for error history updates
        self.debug = debug

    def log(self, message, level="INFO"):
         """Log messages based on debug flag."""
         if self.debug or level != "DEBUG":
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [WeightManager] {level}: {message}")

    def _standard_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """ Standard weighting strategy. """
        weights = { 1: (0.80, 0.20), 2: (0.85, 0.15), 3: (0.90, 0.10), 4: (0.95, 0.05) }
        return weights.get(quarter, (0.90, 0.10)) # Default for Q0 or invalid

    def _adaptive_weights(self, quarter: int, **kwargs) -> Tuple[float, float]:
        """ Adaptive weighting based on historical error rates. """
        if not self.error_history or quarter not in self.error_history.get('main_model', {}) or quarter not in self.error_history.get('quarter_model', {}):
            self.log(f"Insufficient error history for Q{quarter}. Using standard weights.", level="DEBUG")
            return self._standard_weights(quarter)

        main_error = self.error_history['main_model'][quarter]
        quarter_error = self.error_history['quarter_model'][quarter]

        total_error = main_error + quarter_error
        if total_error <= 0: # Avoid division by zero
             self.log(f"Total error is zero or negative for Q{quarter}. Using standard weights.", level="WARNING")
             return self._standard_weights(quarter)

        # Higher weight to model with lower error (weight = other_error / total_error)
        main_weight = quarter_error / total_error

        # Constrain base weights
        main_weight = min(max(main_weight, self.BASE_WEIGHT_MIN), self.BASE_WEIGHT_MAX)
        quarter_weight = 1.0 - main_weight
        self.log(f"Adaptive base weights for Q{quarter}: Main={main_weight:.3f}, Quarter={quarter_weight:.3f} (Errors M:{main_error:.2f}, Q:{quarter_error:.2f})", level="DEBUG")
        return main_weight, quarter_weight

    def get_base_weights(self, quarter: int, strategy: str = 'adaptive', **kwargs) -> Tuple[float, float]:
        """ Get base ensemble weights using the specified strategy. """
        strategy_func = self.weighting_strategies.get(strategy, self._adaptive_weights)
        return strategy_func(quarter, **kwargs)

    def calculate_ensemble(
        self,
        main_prediction: float,
        quarter_prediction: float, # This is the quarter model's prediction for the *full game* (current + remaining)
        current_quarter: int,
        weighting_strategy: str = 'adaptive',
        score_differential: float = 0,
        momentum: float = 0,
        time_remaining: Optional[float] = None, # Assume minutes
        main_uncertainty: Optional[float] = None, # Pass through if needed by base strategy
        quarter_uncertainty: Optional[float] = None # Pass through if needed by base strategy
    ) -> Tuple[float, float, float]:
        """
        Calculates the final ensemble prediction after applying heuristic adjustments
        to the base weights.

        Args:
            main_prediction: Prediction from the main full-game model.
            quarter_prediction: Full-game prediction derived from quarter models (current score + predicted remaining).
            current_quarter: Current quarter of the game (1-4, assume 0 pre-game handled before calling).
            weighting_strategy: Strategy for getting base weights ('adaptive', 'standard').
            score_differential: Current score difference (home - away).
            momentum: Current game momentum metric.
            time_remaining: Estimated time remaining in minutes.
            main_uncertainty: Optional uncertainty measure for main model.
            quarter_uncertainty: Optional uncertainty measure for quarter model.

        Returns:
            Tuple: (final_ensemble_prediction, final_main_weight, final_quarter_weight)
        """
        if not (1 <= current_quarter <= 4):
             self.log(f"calculate_ensemble called with invalid quarter {current_quarter}. Returning main prediction.", level="WARNING")
             return main_prediction, 1.0, 0.0

        # 1. Get Base Weights
        base_main_weight, base_quarter_weight = self.get_base_weights(
            current_quarter,
            weighting_strategy,
            main_uncertainty=main_uncertainty,
            quarter_uncertainty=quarter_uncertainty
        )
        main_w, quarter_w = base_main_weight, base_quarter_weight # Start with base weights

        # 2. Apply Heuristic Adjustments (Based on global function logic)
        self.log(f"Applying heuristics to base weights (M:{main_w:.3f}, Q:{quarter_w:.3f}). Context: Diff={score_differential:.1f}, Mom={momentum:.2f}, TimeRem={time_remaining}", level="DEBUG")

        # Adjust based on time remaining (Increase main weight as game progresses)
        if time_remaining is not None and time_remaining >= 0:
            total_minutes = 48.0
            elapsed = total_minutes - time_remaining
            progress = min(1.0, max(0.0, elapsed / total_minutes))
            # Sigmoid function to smoothly increase main model weight as game progresses
            sigmoid_progress = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5))) # Range [~0, ~1]
            adjustment = (1.0 - main_w) * sigmoid_progress # How much closer to 1.0 can we get?
            main_w = main_w + adjustment
            quarter_w = 1.0 - main_w
            self.log(f"  Time Adjustment (Progress {progress:.2f}): Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if game is close (Give quarter model slightly more weight)
        if abs(score_differential) < 8:
            # Scale adjustment from 0.05 (at diff 0) down to 0 (at diff 8)
            close_game_adjustment = 0.05 * (1.0 - abs(score_differential) / 8.0)
            # Apply adjustment but cap quarter weight
            quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + close_game_adjustment)
            main_w = 1.0 - quarter_w
            self.log(f"  Close Game Adjustment: Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if high momentum (Give quarter model slightly more weight)
        momentum_threshold = 0.3
        if abs(momentum) > momentum_threshold:
             # Scale adjustment based on momentum magnitude beyond threshold
             momentum_adjustment = 0.05 * (abs(momentum) - momentum_threshold) / (1.0 - momentum_threshold) # Normalize effect range
             # Apply adjustment but cap quarter weight
             quarter_w = min(self.HEURISTIC_QUARTER_WEIGHT_MAX, quarter_w + momentum_adjustment)
             main_w = 1.0 - quarter_w
             self.log(f"  Momentum Adjustment: Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # Adjust if predictions diverge significantly (Trust main model more)
        prediction_gap = abs(main_prediction - quarter_prediction)
        gap_threshold = 15.0
        if prediction_gap > gap_threshold:
            # Scale adjustment based on how large the gap is
            discrepancy_adjustment = min(0.3, 0.01 * (prediction_gap - gap_threshold)) # Apply adjustment only beyond threshold
            # Apply adjustment but cap main weight
            main_w = min(self.HEURISTIC_MAIN_WEIGHT_MAX, main_w + discrepancy_adjustment)
            quarter_w = 1.0 - main_w
            self.log(f"  Prediction Gap Adjustment (Gap {prediction_gap:.1f}): Weights -> M={main_w:.3f}, Q={quarter_w:.3f}", level="DEBUG")

        # --- Final Check on Weights ---
        # Ensure weights sum to 1 and are within reasonable bounds after all adjustments
        main_w = min(max(main_w, 1.0 - self.HEURISTIC_QUARTER_WEIGHT_MAX), self.HEURISTIC_MAIN_WEIGHT_MAX)
        quarter_w = 1.0 - main_w
        self.log(f"Final Adjusted Weights: Main={main_w:.3f}, Quarter={quarter_w:.3f}", level="DEBUG")

        # 3. Calculate Final Weighted Prediction
        final_ensemble_prediction = main_w * main_prediction + quarter_w * quarter_prediction

        # 4. Return prediction and final weights (Confidence calculation removed)
        return final_ensemble_prediction, main_w, quarter_w


    def track_weight_usage(
        self,
        game_id: str,
        quarter: int,
        main_weight: float,
        quarter_weight: float,
        prediction_error: Optional[float] = None
    ) -> None:
        """ Tracks the weights used and their performance for later analysis. """
        # (Implementation remains the same as provided in Code Block 1)
        if not isinstance(game_id, str): game_id = str(game_id) # Ensure string game_id
        self.weight_history.append({
            'game_id': game_id,
            'quarter': quarter,
            'main_weight': main_weight,
            'quarter_weight': quarter_weight,
            'timestamp': pd.Timestamp.now(),
            'prediction_error': prediction_error
        })
        self.log(f"Tracked weights for game {game_id}, Q{quarter}. Error={prediction_error}", level="DEBUG")


    def update_weights_from_feedback(
        self,
        recent_window: int = 10
    ) -> Dict[str, Dict[int, float]]:
        """ Updates error history based on recent prediction performance tracked. """
        # (Implementation remains the same as provided in Code Block 1)
        if len(self.weight_history) < recent_window:
             self.log(f"Not enough history ({len(self.weight_history)}/{recent_window}) to update error metrics.", level="DEBUG")
             return self.error_history

        recent_data = self.weight_history[-recent_window:]
        recent_df = pd.DataFrame(recent_data)

        if 'prediction_error' not in recent_df.columns or recent_df['prediction_error'].isna().all():
            self.log("No valid prediction errors found in recent history. Cannot update.", level="DEBUG")
            return self.error_history

        # Calculate average *absolute* error by quarter might be more stable for weighting
        recent_df['abs_error'] = recent_df['prediction_error'].abs()
        quarter_avg_abs_errors = recent_df.groupby('quarter')['abs_error'].mean().to_dict()
        self.log(f"Recent avg absolute errors by quarter: {quarter_avg_abs_errors}", level="DEBUG")

        updated = False
        # Update error history using exponential moving average approach
        for q, avg_abs_error in quarter_avg_abs_errors.items():
             if q in self.error_history.get('main_model', {}) and q in self.error_history.get('quarter_model', {}):
                  # How to update? Assume the 'prediction_error' tracked was for the *ensemble*.
                  # We need errors for main and quarter models separately to properly update adaptive weights.
                  # This current update logic assumes the ensemble error reflects 'main_model' error, which is incorrect.
                  # TODO: Requires tracking separate errors or a different update strategy for adaptive weights.
                  # Placeholder: Update 'main_model' error history based on ensemble error for now.
                  current_error = self.error_history['main_model'][q]
                  new_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_error +
                               (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_error)
                  self.error_history['main_model'][q] = new_error
                  self.log(f"Updated main_model error history for Q{q}: {current_error:.2f} -> {new_error:.2f}", level="DEBUG")
                  updated = True
             else:
                  self.log(f"Quarter {q} not found in error history structure. Skipping update.", level="WARNING")

        if updated:
             self.log("Error history updated based on feedback.", level="INFO")
        else:
             self.log("Error history not updated.", level="DEBUG")

        return self.error_history
    
    # Modify the update function signature to accept processed validation data
    def update_error_history_from_validation(self, validation_summary: pd.DataFrame):
        """
        Updates error history based on aggregated validation results.

        Args:
            validation_summary: A DataFrame with columns like
                                'quarter', 'avg_abs_main_error', 'avg_abs_quarter_error'.
                                (This needs to be calculated from your validation run)
        """
        self.log("Updating error history from validation summary...")
        updated = False
        for _, row in validation_summary.iterrows():
            q = int(row['quarter'])
            avg_abs_main_err = row.get('avg_abs_main_error')
            avg_abs_quarter_err = row.get('avg_abs_quarter_error')

            if q not in self.error_history.get('main_model', {}) or \
            q not in self.error_history.get('quarter_model', {}):
                self.log(f"Quarter {q} not found in error history structure. Skipping.", level="WARNING")
                continue

            if pd.notna(avg_abs_main_err):
                current_main_error = self.error_history['main_model'][q]
                # Use smoothing (EMA - Exponential Moving Average) or just replace
                new_main_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_main_err +
                                (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_main_error)
                self.error_history['main_model'][q] = new_main_error
                self.log(f"Updated main_model error history for Q{q}: {current_main_error:.2f} -> {new_main_error:.2f}", level="DEBUG")
                updated = True

            if pd.notna(avg_abs_quarter_err):
                current_quarter_error = self.error_history['quarter_model'][q]
                new_quarter_error = (self.HISTORIC_WEIGHT_SMOOTHING * avg_abs_quarter_err +
                                    (1.0 - self.HISTORIC_WEIGHT_SMOOTHING) * current_quarter_error)
                self.error_history['quarter_model'][q] = new_quarter_error
                self.log(f"Updated quarter_model error history for Q{q}: {current_quarter_error:.2f} -> {new_quarter_error:.2f}", level="DEBUG")
                updated = True

        if updated:
            self.log("Error history updated based on validation feedback.", level="INFO")
        else:
            self.log("Error history not updated (no valid data?).", level="DEBUG")
        return self.error_history

    # Note: The old 'update_weights_from_feedback' based on live ensemble error
    # should probably be removed or significantly rethought if you want truly adaptive live weights.
    # Relying on periodic updates from validation is more robust.

# -------------------- ConfidenceVisualizer Function --------------------
class ConfidenceVisualizer:
    """
    Generates visual confidence indicators (SVG format) for predictions,
    incorporating prediction intervals and game context.
    """

    def __init__(self):
        """Initialize the confidence visualizer."""
        # Base colors
        self.colors = {
            'background': '#f8f9fa', # Light grey background
            'text_primary': '#333333', # Dark text
            'text_secondary': '#555555', # Lighter text
            'current_score_marker': '#28a745', # Green for current score
            # Quarter-specific colors for the interval/prediction point
            # (Adjust these colors as desired)
            'q0': "#d3d3d3",  # Gray for pregame
            'q1': "#ffa07a",  # Light salmon
            'q2': "#ff7f50",  # Coral
            'q3': "#ff4500",  # OrangeRed
            'q4': "#dc3545",  # Red (Slightly less dark than darkred)
            'default_q': "#6c757d" # Secondary grey for fallback
        }
        # Expected interval width by quarter (used for confidence % calculation)
        # These values should ideally be derived from your PredictionUncertaintyEstimator's analysis
        # or tuned based on observed performance. Taken from the old estimator code.
        self.EXPECTED_RANGES = {0: 30.0, 1: 26.0, 2: 22.0, 3: 18.0, 4: 14.0}

    def _get_quarter_color(self, quarter: int) -> str:
        """Get the color associated with a specific quarter."""
        return self.colors.get(f'q{quarter}', self.colors['default_q'])

    def create_confidence_svg(
        self,
        prediction: float,
        lower_bound: float,
        upper_bound: float,
        current_quarter: int = 0,
        current_home_score: Optional[float] = None, # Optional current score marker
        width: int = 300,
        height: int = 80 # Adjusted height for better spacing
    ) -> str:
        """
        Create an SVG visualization of prediction confidence interval.

        Args:
            prediction: The point prediction value.
            lower_bound: The lower bound of the confidence interval.
            upper_bound: The upper bound of the confidence interval.
            current_quarter: Current game quarter (0-4).
            current_home_score: Optional current home score to display a marker.
            width: SVG width in pixels.
            height: SVG height in pixels.

        Returns:
            SVG markup as a string.
        """
        # --- Calculations ---
        interval_width = upper_bound - lower_bound
        color = self._get_quarter_color(current_quarter)

        # Define the visible score range in the SVG (add padding)
        padding = 10 # Min padding around the interval
        svg_min_score = max(0, lower_bound - padding - (interval_width * 0.1)) # Add a bit more space left
        svg_max_score = upper_bound + padding + (interval_width * 0.1) # Add a bit more space right
        # Ensure range isn't too small if interval is tiny
        if (svg_max_score - svg_min_score) < 10.0:
            mid_point = (lower_bound + upper_bound) / 2.0
            svg_min_score = mid_point - 5.0
            svg_max_score = mid_point + 5.0

        score_range_svg = svg_max_score - svg_min_score

        # Function to scale score to SVG x-coordinate (within drawing area)
        content_width = width # Use full width for scaling
        def to_svg_x(score):
            # Handle edge case where score_range_svg is zero or very small
            if score_range_svg <= 1e-6:
                return content_width / 2
            # Clamp score to visible range before scaling to avoid extreme coordinates
            clamped_score = max(svg_min_score, min(score, svg_max_score))
            return ((clamped_score - svg_min_score) / score_range_svg) * content_width


        # Calculate positions
        pred_x = to_svg_x(prediction)
        lower_x = to_svg_x(lower_bound)
        upper_x = to_svg_x(upper_bound)

        # Calculate confidence percentage (inverse relationship with interval width vs expected)
        # Use expected range for the current quarter, default if quarter > 4
        expected_width = self.EXPECTED_RANGES.get(current_quarter, self.EXPECTED_RANGES[4]) # Default to Q4 expected width if invalid Q
        confidence_pct = 0.0
        if expected_width > 0: # Avoid division by zero
             # Confidence decreases as interval_width exceeds expected_width
             # This formula ensures 100% at width=0, decreasing linearly based on ratio. Capped at 0.
             confidence_pct = max(0.0, min(100.0, 100.0 - (interval_width / expected_width * 75.0))) # Adjusted scaling factor

        # --- SVG Generation ---
        # Using f-string with multi-line capability
        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="font-family: Arial, sans-serif;">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{self.colors['background']}" rx="5" ry="5" />

  <text x="10" y="18" font-size="12px" fill="{self.colors['text_secondary']}">Prediction Range (Q{current_quarter})</text>

  <rect x="{lower_x}" y="{height/2 - 6}" width="{max(0, upper_x - lower_x)}" height="12" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="1" rx="3" ry="3" />

  <circle cx="{pred_x}" cy="{height/2}" r="5" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>
  <text x="{pred_x}" y="{height/2 - 10}" text-anchor="middle" font-size="13px" fill="{self.colors['text_primary']}" font-weight="bold">{prediction:.1f}</text>

  <text x="{lower_x}" y="{height/2 + 18}" text-anchor="{ 'start' if lower_x < 15 else 'middle' }" font-size="11px" fill="{self.colors['text_secondary']}" >{lower_bound:.1f}</text>
  <text x="{upper_x}" y="{height/2 + 18}" text-anchor="{ 'end' if upper_x > width - 15 else 'middle' }" font-size="11px" fill="{self.colors['text_secondary']}">{upper_bound:.1f}</text>
"""
        # Add Current Score Marker (Optional)
        if current_home_score is not None and current_home_score >= 0:
            # Only draw if within the visible range to avoid clutter
            if svg_min_score <= current_home_score <= svg_max_score:
                score_x = to_svg_x(current_home_score)
                svg += f"""
  <line x1="{score_x}" y1="{height/2 - 12}" x2="{score_x}" y2="{height/2 + 12}" stroke="{self.colors['current_score_marker']}" stroke-width="1.5" stroke-dasharray="2,2" />
  <text x="{score_x}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="10px" fill="{self.colors['current_score_marker']}" font-weight="bold">Cur: {current_home_score:.0f}</text>
"""
        # Add Confidence Percentage Label
        svg += f"""
  <text x="{width - 10}" y="18" text-anchor="end" font-size="12px" fill="{self.colors['text_primary']}" font-weight="bold">{confidence_pct:.0f}% Conf.</text>
</svg>"""

        return svg
    
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
import joblib # Added
import os # Added
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import Ridge
# Need NBAFeatureEngine and EnsembleWeightManager definitions from the previous code.
# Make sure xgboost is installed: pip install xgboost
try:
    import xgboost as xgb
except ImportError:
    xgb = None # Handle cases where xgboost might not be installed

# -------------------- Quarter-Specific Model System --------------------
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
import joblib # Added
import os # Added
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import Ridge
# Need NBAFeatureEngine and EnsembleWeightManager definitions from the previous code.
# Make sure xgboost is installed: pip install xgboost
try:
    import xgboost as xgb
except ImportError:
    xgb = None # Handle cases where xgboost might not be installed

# Assuming NBAFeatureEngine and EnsembleWeightManager classes are defined above this point

# -------------------- Quarter-Specific Model System --------------------
class QuarterSpecificModelSystem:
    """
    Manages quarter-specific models for prediction, including fallbacks.
    Also incorporates methods for initializing, training, and evaluating
    specific models, particularly for early quarters (Q1, Q2).
    """
    def __init__(self, feature_generator: NBAFeatureEngine, debug: bool = False):
        """
        Initialize the system.

        Args:
            feature_generator: Instance of NBAFeatureEngine.
            debug: Verbosity flag.
        """
        self.feature_generator = feature_generator
        self.debug = debug

        # --- Model Storage ---
        # Stores models used for PREDICTION (loaded from disk or fallbacks)
        self.models: Dict[int, Any] = {}
        # Stores fallback models (Ridge)
        self.fallback_models: Dict[int, Any] = {}
        # Stores models initialized for TRAINING purposes
        self.trainable_models: Dict[int, Dict[str, Any]] = {1: {}, 2: {}, 3: {}, 4: {}} # e.g., {1: {'xgb_tuned': model_instance}}

        # --- Feature Set Definitions ---
        # Specific feature sets desired for Q1/Q2 (from EarlyQuarterModelOptimizer)
        self.q1_specific_feature_sets = {
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
                'prev_matchup_diff', #'time_remaining_norm' # Feature might not be relevant pre-game/per quarter training
            ]
        }
        self.q2_specific_feature_sets = {
            'basic': [
                'home_q1', 'away_q1', #'score_ratio', # Needs careful definition
                'rest_advantage'
            ],
            'momentum': [
                'home_q1', 'away_q1', #'score_ratio',
                'rest_advantage', 'q1_to_q2_momentum'
            ],
            'advanced': [
                'home_q1', 'away_q1', #'score_ratio',
                'rest_advantage', 'q1_to_q2_momentum',
                'rolling_home_score', 'rolling_away_score',
                'prev_matchup_diff', #'time_remaining_norm',
                #'momentum_indicator', # Need definition
                #'score_momentum_interaction' # Need definition
            ]
            # NOTE: Commented out features like score_ratio, time_remaining_norm,
            # momentum_indicator, score_momentum_interaction need to be reliably
            # generated by your feature_generator if you intend to use them.
        }

        # Consolidated feature sets used for PREDICTION
        # Prioritizes specific Q1/Q2 sets, uses generator's sets otherwise
        self.quarter_feature_sets = self._get_consolidated_feature_sets()

        # --- Other Attributes ---
        self.error_history = { # Example structure - should be loaded/updated
            'main_model': {1: 7.0, 2: 6.2, 3: 5.5, 4: 4.7},
            'quarter_model': {1: 8.5, 2: 7.0, 3: 5.8, 4: 4.5}
        }
        self.prediction_cache: Dict[str, float] = {} # Simple in-memory cache

        # Initialize fallback models upon instantiation
        self._create_fallback_models()
        self.log("QuarterSpecificModelSystem initialized.", level="DEBUG")


    def log(self, message, level="INFO"):
        """Log messages based on debug flag."""
        if self.debug or level != "DEBUG":
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [QuarterSystem] {level}: {message}")

    def _get_consolidated_feature_sets(self) -> Dict[int, List[str]]:
        """
        Returns the primary feature set to use for prediction for each quarter.
        Uses specific Q1/Q2 sets if defined, otherwise falls back to generator's sets.
        """
        base_sets = self.feature_generator._get_optimized_feature_sets() # Access generator's definition
        # Use the 'advanced' set from the specific Q1/Q2 definitions if available
        final_sets = {
            1: self.q1_specific_feature_sets.get('advanced', base_sets.get(1, [])),
            2: self.q2_specific_feature_sets.get('advanced', base_sets.get(2, [])),
            3: base_sets.get(3, []),
            4: base_sets.get(4, [])
        }
        self.log(f"Consolidated prediction feature sets: {final_sets}", level="DEBUG")
        return final_sets

    # --- Model Loading and Fallback ---

    def load_models(self, model_dir='models'):
        """
        Load pre-trained quarter-specific models from disk for PREDICTION.
        These are the primary models used by predict_quarter.

        Args:
            model_dir: Directory containing the model files (e.g., q1_model.pkl).
        """
        self.log(f"Loading prediction models from {model_dir}...")
        for quarter in range(1, 5):
            # Expecting simple names like q1_model.pkl, q2_model.pkl etc.
            model_path = os.path.join(model_dir, f'q{quarter}_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[quarter] = joblib.load(model_path)
                    self.log(f"Loaded Q{quarter} prediction model from {model_path}")
                except Exception as e:
                    self.log(f"Error loading Q{quarter} prediction model: {e}", level="ERROR")
                    self.models[quarter] = None # Ensure it's None if loading fails
            else:
                self.log(f"Prediction model for Q{quarter} not found at {model_path}. Will use fallback.", level="WARNING")
                self.models[quarter] = None

        # Ensure fallback models are ready if primary models failed to load
        self._create_fallback_models()


    def _create_fallback_models(self):
        """Create simple Ridge regression models as fallbacks for prediction."""
        self.log("Creating fallback Ridge models...", level="DEBUG")
        for quarter in range(1, 5):
            # Use Ridge for all quarters as a simple, robust fallback
            # Only create if not already present (or potentially overwrite)
            # if quarter not in self.fallback_models:
            model = Ridge(alpha=1.0, random_state=42)
            self.fallback_models[quarter] = model
            self.log(f"Fallback model created for Q{quarter}.", level="DEBUG")

    # --- Initialization, Training, and Evaluation Methods (Integrated from EarlyQuarterModelOptimizer) ---

    def initialize_trainable_models(self, quarters: List[int] = [1, 2], model_type='xgboost'):
        """
        Initialize models specifically for training purposes, particularly for Q1/Q2.
        Stores initialized models in `self.trainable_models`.

        Args:
            quarters: List of quarters to initialize models for (default: [1, 2]).
            model_type: Type of model to initialize ('xgboost' supported currently).
        """
        self.log(f"Initializing trainable models for quarters {quarters} (type: {model_type})...")
        if model_type.lower() == 'xgboost':
            if xgb is None:
                self.log("XGBoost library not found. Cannot initialize XGBoost models.", level="ERROR")
                return False
            try:
                # Using parameters similar to EarlyQuarterModelOptimizer's 'tuned' versions
                params_q1 = {
                    'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 4,
                    'min_child_weight': 2, 'gamma': 1, 'subsample': 0.75,
                    'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
                    'objective': 'reg:squarederror', 'random_state': 42
                }
                params_q2 = {
                    'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4,
                    'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.8,
                    'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                    'objective': 'reg:squarederror', 'random_state': 42
                }
                # Could also use get_recommended_model_params here for consistency

                if 1 in quarters:
                    self.trainable_models[1]['xgb_tuned'] = xgb.XGBRegressor(**params_q1)
                    self.log("Initialized trainable XGBoost model for Q1.")
                if 2 in quarters:
                     self.trainable_models[2]['xgb_tuned'] = xgb.XGBRegressor(**params_q2)
                     self.log("Initialized trainable XGBoost model for Q2.")
                # Add initialization for Q3/Q4 if needed
                return True
            except Exception as e:
                 self.log(f"Error initializing XGBoost models: {e}", level="ERROR")
                 return False
        else:
            self.log(f"Model type '{model_type}' not currently supported for initialization.", level="WARNING")
            return False

    def train_quarter_models(self, training_data: pd.DataFrame, target_col_prefix: str = 'home', quarters: List[int] = [1, 2]):
        """
        Train the initialized models using provided data.

        Args:
            training_data: DataFrame containing features and target variables.
            target_col_prefix: Prefix for the target quarter score columns (e.g., 'home', 'away').
            quarters: List of quarters for which to train models.

        Returns:
            Dictionary containing training results (metrics, feature importance).
        """
        self.log(f"Starting training for quarters {quarters} with target prefix '{target_col_prefix}'...")
        results = {q: {} for q in quarters}
        if not isinstance(training_data, pd.DataFrame) or training_data.empty:
            self.log("No training data provided.", level="ERROR")
            return results

        # Ensure features are present (example check)
        # Ideally, feature generation happens before calling train
        # if 'momentum_indicator' not in training_data.columns:
        #     self.log("Adding advanced features to training data (assuming necessary)", level="DEBUG")
        #     # This assumes feature_generator has the necessary method
        #     # training_data = self.feature_generator.add_advanced_features(training_data) # Be careful modifying input data

        feature_sets_map = {
            1: self.q1_specific_feature_sets,
            2: self.q2_specific_feature_sets,
            # Add mappings for Q3, Q4 if training them here using specific sets
        }

        for q in quarters:
            if q not in self.trainable_models or not self.trainable_models[q]:
                 self.log(f"No trainable models initialized for Q{q}. Skipping training.", level="WARNING")
                 continue

            target_col = f"{target_col_prefix}_q{q}"
            if target_col not in training_data.columns:
                 self.log(f"Target column '{target_col}' not found for Q{q}. Skipping training.", level="ERROR")
                 continue

            self.log(f"Training Q{q} models with target: {target_col}")
            q_feature_sets = feature_sets_map.get(q)
            if not q_feature_sets:
                 self.log(f"No specific feature sets defined for training Q{q}. Using default prediction set.", level="WARNING")
                 # Use the prediction set as fallback for training if no specific training sets defined
                 q_feature_sets = {f'prediction_set_q{q}': self.quarter_feature_sets.get(q, [])}


            y = training_data[target_col]

            for feature_set_name, features in q_feature_sets.items():
                valid_features = [f for f in features if f in training_data.columns]
                missing_count = len(features) - len(valid_features)
                if missing_count > 0:
                    self.log(f"Warning: Missing {missing_count} features for Q{q} '{feature_set_name}' set.", level="WARNING")
                if not valid_features:
                    self.log(f"No valid features found for Q{q} '{feature_set_name}'. Skipping.", level="WARNING")
                    continue

                X = training_data[valid_features].copy()

                # Train each model initialized for this quarter
                for model_name, model in self.trainable_models[q].items():
                    try:
                        start_time = time.time()
                        model.fit(X, y)
                        train_time = time.time() - start_time

                        # Optional: Evaluate on training set (can indicate overfitting)
                        y_pred_train = model.predict(X)
                        mse_train = np.mean((y - y_pred_train) ** 2)
                        mae_train = np.mean(np.abs(y - y_pred_train))

                        trained_model_key = f"{model_name}_{feature_set_name}"
                        results[q][trained_model_key] = {
                            'model': model, # Store the trained model instance itself
                            'features': valid_features,
                            'train_mse': mse_train,
                            'train_mae': mae_train,
                            'train_time_sec': train_time,
                            'feature_importance': dict(zip(valid_features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                        }
                        self.log(f"Trained Q{q} {model_name} with '{feature_set_name}' features: MAE={mae_train:.3f} (Train Time: {train_time:.2f}s)")

                        # Optional: Overwrite the main prediction model for this quarter
                        # self.models[q] = model
                        # self.log(f"Updated main prediction model for Q{q} with newly trained '{trained_model_key}'.")

                    except Exception as e:
                        self.log(f"Error training Q{q} {model_name} with '{feature_set_name}' features: {e}", level="ERROR")
                        traceback.print_exc() # Print detailed traceback for debugging

        return results

    def evaluate_trained_models(self, test_data: pd.DataFrame, trained_models_results: Dict, target_col_prefix: str = 'home'):
        """
        Evaluate models that were trained (using the results from train_quarter_models).

        Args:
            test_data: DataFrame containing test features and target variables.
            trained_models_results: The dictionary returned by train_quarter_models.
            target_col_prefix: Prefix for the target quarter score columns.

        Returns:
            DataFrame with evaluation metrics (MSE, RMSE, MAE) for each trained model variation.
        """
        self.log("Evaluating trained models...")
        eval_results = []
        if not isinstance(test_data, pd.DataFrame) or test_data.empty:
            self.log("No test data provided for evaluation.", level="ERROR")
            return pd.DataFrame()

        # Ensure features are present if needed
        # test_data = self.feature_generator.add_advanced_features(test_data) # If needed

        for quarter, trained_variations in trained_models_results.items():
            target_col = f"{target_col_prefix}_q{quarter}"
            if target_col not in test_data.columns:
                self.log(f"Target column '{target_col}' not found in test data for Q{quarter}. Skipping evaluation.", level="WARNING")
                continue

            y_test = test_data[target_col]

            for model_variation_key, model_info in trained_variations.items():
                 model = model_info.get('model')
                 features = model_info.get('features')
                 model_base_name, feature_set_name = model_variation_key.rsplit('_', 1) # Assumes key format 'xgb_tuned_advanced'

                 if not model or not features:
                     self.log(f"Model or features missing for '{model_variation_key}'. Skipping.", level="WARNING")
                     continue

                 if all(f in test_data.columns for f in features):
                     X_test = test_data[features]
                     try:
                         y_pred_test = model.predict(X_test)
                         mse = np.mean((y_test - y_pred_test) ** 2)
                         rmse = np.sqrt(mse)
                         mae = np.mean(np.abs(y_test - y_pred_test))

                         eval_results.append({
                             'quarter': f'Q{quarter}',
                             'model': model_base_name,
                             'feature_set': feature_set_name,
                             'mse': mse,
                             'rmse': rmse,
                             'mae': mae,
                             'sample_size': len(X_test)
                         })
                         self.log(f"Evaluated Q{quarter} {model_base_name} ('{feature_set_name}'): RMSE={rmse:.3f}, MAE={mae:.3f}")
                     except Exception as e:
                          self.log(f"Error evaluating Q{quarter} {model_base_name} ('{feature_set_name}'): {e}", level="ERROR")
                 else:
                     missing_eval_features = [f for f in features if f not in test_data.columns]
                     self.log(f"Missing features in test data for Q{quarter} {model_base_name} ('{feature_set_name}'): {missing_eval_features}. Skipping evaluation.", level="WARNING")

        if not eval_results:
            self.log("No evaluation results generated.", level="WARNING")
            return pd.DataFrame()

        return pd.DataFrame(eval_results)

    def compare_evaluation_to_baseline(self, eval_df: pd.DataFrame, baseline_rmse: Dict[int, float]):
        """
        Compare evaluation results (RMSE) against baseline values.

        Args:
            eval_df: DataFrame returned by evaluate_trained_models.
            baseline_rmse: Dictionary mapping quarter number (int) to baseline RMSE value. e.g., {1: 8.0, 2: 7.5}

        Returns:
            DataFrame with added baseline comparison columns (improvement, pct_improvement).
        """
        if not isinstance(eval_df, pd.DataFrame) or eval_df.empty:
            self.log("Evaluation DataFrame is empty. Cannot compare to baseline.", level="WARNING")
            return pd.DataFrame()
        if not baseline_rmse:
             self.log("Baseline RMSE dictionary is empty. Cannot compare.", level="WARNING")
             return eval_df

        comparison = eval_df.copy()
        # Extract quarter number safely
        comparison['q_num'] = comparison['quarter'].str.extract('(\d+)').astype(int)
        comparison['baseline_rmse'] = comparison['q_num'].apply(
            lambda q: baseline_rmse.get(q, np.nan) # Use np.nan if baseline for a quarter is missing
        )
        comparison = comparison.dropna(subset=['baseline_rmse']) # Drop rows where baseline is missing

        if comparison.empty:
             self.log("No matching baseline RMSE values found for evaluated quarters.", level="WARNING")
             return pd.DataFrame()

        comparison['rmse_improvement'] = comparison['baseline_rmse'] - comparison['rmse']
        comparison['pct_improvement'] = (comparison['rmse_improvement'] / comparison['baseline_rmse']) * 100

        self.log("Baseline comparison complete.", level="DEBUG")
        return comparison.sort_values(['q_num', 'pct_improvement'], ascending=[True, False]).drop(columns=['q_num'])

    # --- Core Prediction Logic ---

    def predict_quarter(self, X: pd.DataFrame, quarter: int) -> float:
        """
        Predict score for a specific quarter using the loaded model or fallback.
        Uses the consolidated feature set for prediction.

        Args:
            X: DataFrame containing features for the game (should be a single row).
            quarter: The quarter number (1-4) to predict.

        Returns:
            Predicted score for the quarter.
        """
        cache_key = f"{hash(str(X.values.tobytes()))}_q{quarter}" # Simple cache key
        if cache_key in self.prediction_cache:
            self.log(f"Using cached prediction for Q{quarter}", level="DEBUG")
            return self.prediction_cache[cache_key]

        if not (1 <= quarter <= 4):
            self.log(f"Invalid quarter ({quarter}) requested for prediction.", level="ERROR")
            return 0.0

        # Use the consolidated feature set defined in __init__
        features = self.quarter_feature_sets.get(quarter, [])
        available_features = [f for f in features if f in X.columns]

        if not available_features:
            default_score = self.feature_generator.league_averages['quarter_scores'].get(quarter, 25.0)
            self.log(f"No features available for Q{quarter} prediction. Using default: {default_score}", level="WARNING")
            self.prediction_cache[cache_key] = default_score
            return default_score

        # Prioritize loaded model, then fallback
        model_to_use = self.models.get(quarter)
        model_source = "primary"

        if model_to_use is None:
            model_to_use = self.fallback_models.get(quarter)
            model_source = "fallback"
            if model_to_use is None:
                 default_score = self.feature_generator.league_averages['quarter_scores'].get(quarter, 25.0)
                 self.log(f"No primary or fallback model available for Q{quarter}. Using default: {default_score}", level="ERROR")
                 self.prediction_cache[cache_key] = default_score
                 return default_score

        self.log(f"Predicting Q{quarter} using {model_source} model with features: {available_features}", level="DEBUG")
        prediction = None
        try:
            # Check if model expects specific features (e.g., scikit-learn >= 1.0)
            required_model_features = []
            if hasattr(model_to_use, 'feature_names_in_'):
                required_model_features = list(model_to_use.feature_names_in_)
                # Ensure available features match model's expectations
                if not all(f in required_model_features for f in available_features):
                    # This case indicates a potential mismatch between training and prediction features
                    self.log(f"Feature mismatch for Q{quarter} {model_source} model. Available: {available_features}, Expected: {required_model_features}. Attempting with available.", level="WARNING")
                    # Try predicting with available features, might fail if model strictly requires others
                    prediction = model_to_use.predict(X[available_features])[0]
                else:
                     # Ensure columns are in the order the model expects
                     prediction = model_to_use.predict(X[required_model_features])[0] # Use model's order
            else:
                 # Older models or non-sklearn models might not have feature_names_in_
                 prediction = model_to_use.predict(X[available_features])[0] # Use available features

        except Exception as e:
            self.log(f"Error during Q{quarter} prediction with {model_source} model: {e}", level="ERROR")
            # If primary model fails, try fallback (if not already using it)
            if model_source == "primary":
                 self.log(f"Attempting Q{quarter} prediction with fallback model...", level="WARNING")
                 fallback_model = self.fallback_models.get(quarter)
                 if fallback_model:
                     try:
                         # Assuming fallback (Ridge) is less sensitive to exact feature set
                         prediction = fallback_model.predict(X[available_features])[0]
                         model_source = "fallback (after primary error)"
                     except Exception as fb_e:
                          self.log(f"Error during Q{quarter} prediction with fallback model: {fb_e}", level="ERROR")

        # Final fallback to default score if all predictions fail
        if prediction is None:
            prediction = self.feature_generator.league_averages['quarter_scores'].get(quarter, 25.0)
            model_source = "default (after errors)"
            self.log(f"Prediction failed for Q{quarter}. Using default score: {prediction}", level="ERROR")

        self.log(f"Q{quarter} Prediction ({model_source}): {prediction:.3f}", level="DEBUG")
        self.prediction_cache[cache_key] = prediction
        return prediction


    def predict_remaining_quarters(self, game_data: Dict, current_quarter: int) -> Dict[str, float]:
        """
        Predict scores for all quarters remaining in the game.

        Args:
            game_data: Dictionary representing the current state of the game.
            current_quarter: The current quarter number (0 if pre-game, 1-4 during game).

        Returns:
            Dictionary mapping remaining quarter numbers (str) to predicted scores.
            e.g., {'q3': 26.5, 'q4': 27.1}
        """
        if current_quarter >= 4:
            self.log("Game is in Q4 or later, no remaining quarters to predict.", level="DEBUG")
            return {}

        # Create a DataFrame for prediction (needs consistent features)
        # Feature generation should happen *outside* this loop ideally,
        # but we regenerate based on predictions for subsequent quarters here.
        X = pd.DataFrame([game_data])
        try:
            # Use the primary feature generation method (ensure it's correctly defined in NBAFeatureEngine)
            # Assuming 'generate_features_for_prediction' is the main method
            if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                 X = self.feature_generator.generate_features_for_prediction(X) # Generate initial features
            elif hasattr(self.feature_generator, 'integrate_advanced_features'): # Fallback
                 X = self.feature_generator.integrate_advanced_features(X)
            else:
                 self.log("Suitable feature generation method not found in feature_generator.", level="ERROR")
                 return {} # Cannot proceed without features
        except Exception as e:
            self.log(f"Error generating initial features for remaining quarters: {e}", level="ERROR")
            return {}


        results = {}
        # Predict sequentially for remaining quarters
        for q in range(max(1, current_quarter + 1), 5): # Start from Q1 or next quarter
            try:
                pred_score = self.predict_quarter(X.iloc[[0]], q) # Pass single row DataFrame
                results[f'q{q}'] = pred_score

                # Update the DataFrame `X` with the predicted score for the next iteration's feature calculation
                # NOTE: This assumes features for Q(n+1) depend on the predicted score of Q(n).
                # This iterative update can be complex and depends heavily on feature definitions.
                if f'home_q{q}' in X.columns:
                    X[f'home_q{q}'] = pred_score
                else:
                    # Add column if it doesn't exist (might happen on first prediction)
                    X = X.assign(**{f'home_q{q}': pred_score})

                # Re-calculate any features that depend on the newly predicted quarter score
                # This step is CRUCIAL and depends entirely on your feature engineering logic.
                # Example: If 'cumulative_momentum' depends on latest quarter score, recalculate it here.
                # For simplicity, we might need to regenerate all features based on the updated X.
                # This is inefficient but safer if feature dependencies are complex.

                # Option 1: Selective Update (more efficient, requires knowing dependencies)
                # if q == 2: X['q1_to_q2_momentum'] = calculate_q1_q2_momentum(X['home_q1'], pred_score, ...) etc.

                # Option 2: Full Regeneration (safer but slower)
                if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                    X = self.feature_generator.generate_features_for_prediction(X)
                elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                    X = self.feature_generator.integrate_advanced_features(X)


            except Exception as e:
                 self.log(f"Error predicting or updating features for Q{q}: {e}", level="ERROR")
                 # If one quarter fails, maybe stop or return partial results?
                 results[f'q{q}'] = self.feature_generator.league_averages['quarter_scores'].get(q, 25.0) # Use default on error
                 # Continue to next quarter? Or break? Let's continue for now.

        self.log(f"Predicted remaining quarters ({current_quarter+1}-4): {results}", level="DEBUG")
        return results

    def predict_final_score(self, game_data: Dict, main_model_prediction: Optional[float] = None, weight_manager: Optional[EnsembleWeightManager] = None) -> Tuple[float, float, Dict]:
        """
        Predict the final score using an ensemble managed by EnsembleWeightManager.

        Args:
            game_data: Dictionary representing the current state of the game. Should contain known quarter scores
                       (e.g., home_q1, away_q1) and other features needed by the feature generator.
            main_model_prediction: Optional pre-calculated prediction from the main full-game model.
            weight_manager: Instance of EnsembleWeightManager to calculate weights and ensemble. REQUIRED.

        Returns:
            Tuple: (ensemble_prediction, confidence, breakdown_dict)
                   Returns (fallback_prediction, 0.0, {'error': ...}) on critical errors.
        """
        self.log("Predicting final score using WeightManager...", level="DEBUG")
        fallback_pred = main_model_prediction or self.feature_generator.league_averages.get('score', 110.0)

        # --- Input Validation ---
        if not isinstance(game_data, dict):
            self.log("Input game_data must be a dictionary.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Invalid game_data type'}
        if weight_manager is None:
            # *** CRITICAL: Ensure weight_manager is ALWAYS provided ***
            self.log("WeightManager instance IS REQUIRED for predict_final_score.", level="ERROR")
            # Decide how to handle: raise error or return fallback? Returning fallback here.
            return fallback_pred, 0.0, {'error': 'Missing WeightManager'}
        # Ensure weight_manager has the required method
        if not hasattr(weight_manager, 'calculate_dynamic_ensemble'):
            self.log("Provided weight_manager is missing 'calculate_dynamic_ensemble' method.", level="ERROR")
            return fallback_pred, 0.0, {'error': 'Invalid WeightManager instance'}


        current_quarter = int(game_data.get('current_quarter', 0))

        # Calculate score accumulated so far safely
        home_score_so_far = sum([float(game_data.get(f'home_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])
        away_score_so_far = sum([float(game_data.get(f'away_q{q}', 0) or 0) for q in range(1, current_quarter + 1)])

        # --- Get Main Model Prediction (Handle if missing) ---
        if main_model_prediction is None:
            if current_quarter > 0:
                main_model_prediction = home_score_so_far * (4.0 / max(1, current_quarter)) # Avoid division by zero
                self.log(f"Main model prediction not provided. Extrapolating based on current score: {main_model_prediction:.2f}", level="WARNING")
            else:
                main_model_prediction = self.feature_generator.league_averages.get('score', 110.0)
                self.log(f"Main model prediction not provided pregame. Using league average: {main_model_prediction:.2f}", level="WARNING")
        # Ensure it's a float after potential calculation/retrieval
        try:
            main_model_prediction = float(main_model_prediction)
        except (ValueError, TypeError):
            self.log(f"Main model prediction ('{main_model_prediction}') is not numeric. Using league average.", level="ERROR")
            main_model_prediction = self.feature_generator.league_averages.get('score', 110.0)


        # --- Handle Pre-game Scenario ---
        if current_quarter <= 0:
            self.log("Pregame prediction. Using main model prediction only.", level="DEBUG")
            confidence = 0.4 # Assign typical pregame confidence
            breakdown = {
                'main_model': main_model_prediction,
                'quarter_model_sum': 0,
                'weights': {'main': 1.0, 'quarter': 0.0}, # Base weights for pregame
                'quarter_predictions': {},
                'current_score': 0,
                'score_differential': 0,
                'momentum': 0,
                'current_quarter': 0,
                'time_remaining_minutes_est': 48.0 # Full game minutes
            }
            return main_model_prediction, confidence, breakdown

        # --- Get Quarter Model Prediction Component ---
        remaining_quarters_pred = self.predict_remaining_quarters(game_data, current_quarter)
        # Check if prediction failed critically inside predict_remaining_quarters
        if not isinstance(remaining_quarters_pred, dict):
             self.log("predict_remaining_quarters failed. Cannot calculate quarter sum.", level="ERROR")
             # Fallback: Use main prediction only
             confidence = 0.3 # Lower confidence due to failure
             breakdown = {'error': 'Failed to predict remaining quarters', 'main_model': main_model_prediction}
             return main_model_prediction, confidence, breakdown

        predicted_score_remaining = sum(remaining_quarters_pred.values())
        quarter_sum_prediction = home_score_so_far + predicted_score_remaining

        # --- Gather Context Features ---
        # Generate features based on the *current* game_data state to get context like momentum, diff
        try:
             if hasattr(self.feature_generator, 'generate_features_for_prediction'):
                  current_features_df = self.feature_generator.generate_features_for_prediction(pd.DataFrame([game_data]))
             elif hasattr(self.feature_generator, 'integrate_advanced_features'):
                  current_features_df = self.feature_generator.integrate_advanced_features(pd.DataFrame([game_data]))
             else: current_features_df = pd.DataFrame([game_data]) # Fallback to original data if no generator method

             # Safely extract features, providing defaults if missing
             score_differential = float(current_features_df.get('score_differential', home_score_so_far - away_score_so_far).iloc[0])
             momentum = float(current_features_df.get('cumulative_momentum', 0.0).iloc[0])

        except Exception as e:
             self.log(f"Error getting score_differential/momentum from features: {e}. Using defaults.", level="WARNING")
             score_differential = home_score_so_far - away_score_so_far
             momentum = 0.0

        # Estimate time remaining (example: minutes)
        if current_quarter <= 4:
            time_remaining_minutes = 12.0 * (4 - current_quarter) # Simple estimate: Full quarters remaining
        else: # Overtime?
            time_remaining_minutes = 0.0 # Assume end of regulation for simplicity if OT not handled

        # --- Calculate Ensemble Prediction using WeightManager ---
        try:
             # *** CALL THE NEW METHOD ON THE weight_manager INSTANCE ***
             ensemble_pred, confidence, final_weight_main, final_weight_quarter = weight_manager.calculate_ensemble(
                 main_prediction=float(main_model_prediction),
                 quarter_prediction=float(quarter_sum_prediction),
                 current_quarter=int(current_quarter),
                 # weighting_strategy='adaptive', # Pass strategy if needed
                 score_differential=float(score_differential),
                 momentum=float(momentum),
                 time_remaining_minutes=float(time_remaining_minutes) if time_remaining_minutes is not None else None
             )
        except Exception as e:
             self.log(f"Error calling weight_manager.calculate_dynamic_ensemble: {e}. Using main prediction fallback.", level="ERROR")
             traceback.print_exc()
             # Fallback if ensemble calculation fails
             ensemble_pred = main_model_prediction
             confidence = 0.3 # Low confidence on error
             final_weight_main = 1.0
             final_weight_quarter = 0.0

        # --- Compile Breakdown ---
        breakdown = {
            'main_model': main_model_prediction,
            'quarter_model_sum': quarter_sum_prediction, # Sum of current + predicted remaining
            'weights': {'main': final_weight_main, 'quarter': final_weight_quarter}, # Store final adjusted weights
            'quarter_predictions': remaining_quarters_pred, # Predictions for future quarters only
            'current_score': home_score_so_far,
            'score_differential': score_differential,
            'momentum': momentum,
            'current_quarter': current_quarter,
            'time_remaining_minutes_est': time_remaining_minutes
        }
        self.log(f"Final Ensemble Prediction: {ensemble_pred:.2f} (Confidence: {confidence:.2f})", level="DEBUG")

        # Final type check before returning
        try:
            final_ensemble_pred = float(ensemble_pred)
            final_confidence = float(confidence)
        except (ValueError, TypeError):
           self.log("Ensemble prediction or confidence not numeric. Returning fallback.", level="ERROR")
           return fallback_pred, 0.0, {'error': 'Non-numeric ensemble result'}

        return final_ensemble_pred, final_confidence, breakdown

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
import joblib
from scipy import stats

# -------------------- PredictionUncertaintyEstimator Class --------------------


class PredictionUncertaintyEstimator:
    """
    Estimates uncertainty (confidence intervals) for NBA score predictions
    based on historical error patterns and game context.
    """
    def __init__(self, debug=False):
        """Initialize the uncertainty estimator with default values."""
        self.debug = debug
        
        # Default mean absolute error by quarter (improves as game progresses)
        self.mae_by_quarter = {0: 8.5, 1: 7.0, 2: 6.0, 3: 5.0, 4: 0.0}
        
        # Default standard deviation by quarter
        self.std_by_quarter = {0: 4.5, 1: 3.8, 2: 3.2, 3: 2.6, 4: 0.0}
        
        # Adjustment factors for different game situations
        self.margin_adjustments = {'close': 1.2, 'moderate': 1.0, 'blowout': 0.8}
        self.momentum_effects = {'high': 1.2, 'moderate': 1.0, 'low': 0.8}
        
        # Storage for historical error tracking
        self.historical_errors = {q: [] for q in range(5)}
        self.interval_coverage = {q: {'inside': 0, 'total': 0} for q in range(5)}
    
    def _print_debug(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
             print(f"[{type(self).__name__}] {message}")

    def calculate_prediction_interval(
        self, 
        prediction: float, 
        current_quarter: int,
        score_margin: Optional[float] = None, 
        momentum: Optional[float] = None, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate a prediction interval based on quarter and game context.
        
        Args:
            prediction: The point prediction value
            current_quarter: Current quarter of the game (0-4)
            score_margin: Current score margin between teams
            momentum: Current momentum metric (positive or negative)
            confidence_level: Desired confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound, interval_width)
        """
        # Get base error metrics for this quarter
        mae = self.mae_by_quarter.get(current_quarter, 8.0)
        std = self.std_by_quarter.get(current_quarter, 4.0)
        
        # Adjust for score margin (close games are harder to predict)
        if score_margin is not None:
            if score_margin < 5:
                margin_factor = self.margin_adjustments['close']
            elif score_margin > 15:
                margin_factor = self.margin_adjustments['blowout']
            else:
                margin_factor = self.margin_adjustments['moderate']
            mae *= margin_factor
            std *= margin_factor
        
        # Adjust for momentum (high momentum can affect predictability)
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
        
        # Calculate z-score for desired confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate interval half-width based on combined error
        interval_half_width = z_score * np.sqrt(mae**2 + std**2)
        
        # Narrow interval as game progresses
        narrowing_factor = 1.0 - (current_quarter * 0.15)
        interval_half_width *= max(0.4, narrowing_factor)
        
        # Calculate bounds
        lower_bound = max(0, prediction - interval_half_width)  # Scores can't be negative
        upper_bound = prediction + interval_half_width
        
        return lower_bound, upper_bound, interval_half_width * 2

    def update_error_metrics(self, errors_by_quarter: Dict[int, List[float]]) -> None:
        """
        Update error metrics based on new observations.
        
        Args:
            errors_by_quarter: Dictionary mapping quarters to lists of prediction errors
        """
        for quarter, errors in errors_by_quarter.items():
            if errors:
                # Add new errors to history (keeping last 100 for each quarter)
                self.historical_errors[quarter].extend(errors)
                self.historical_errors[quarter] = self.historical_errors[quarter][-100:]
                
                # Recalculate MAE and STD
                mae = np.mean(np.abs(self.historical_errors[quarter]))
                std = np.std(self.historical_errors[quarter])
                
                # Update metrics
                self.mae_by_quarter[quarter] = mae
                self.std_by_quarter[quarter] = std
                
                self._print_debug(f"Updated Q{quarter} metrics: MAE={mae:.2f}, STD={std:.2f}")

    def record_interval_coverage(self, quarter: int, lower: float, upper: float, actual: float) -> None:
        """
        Record whether an actual value fell within a prediction interval.
        
        Args:
            quarter: Game quarter
            lower: Lower bound of interval
            upper: Upper bound of interval
            actual: Actual observed value
        """
        self.interval_coverage[quarter]['total'] += 1
        if lower <= actual <= upper:
            self.interval_coverage[quarter]['inside'] += 1
            
        if quarter in [0, 1, 2, 3, 4]:  # Valid quarters only
            coverage = self.interval_coverage[quarter]['inside'] / self.interval_coverage[quarter]['total'] * 100
            self._print_debug(f"Q{quarter} interval coverage: {coverage:.1f}% ({self.interval_coverage[quarter]['inside']}/{self.interval_coverage[quarter]['total']})")

    def get_coverage_stats(self) -> pd.DataFrame:
        """
        Get statistics on prediction interval coverage.
        
        Returns:
            DataFrame with coverage statistics by quarter
        """
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
    
        # Function to convert score to SVG x-coordinate
        def to_svg_x(score):
            return (score - svg_min) / score_range * width
        
        # Calculate positions
        pred_x = to_svg_x(prediction)
        lower_x = to_svg_x(lower_bound)
        upper_x = to_svg_x(upper_bound)
        
        # Quarter-specific colors (from light to dark red as game progresses)
        quarter_colors = {
            0: "#d3d3d3",  # Gray for pregame
            1: "#ffa07a",  # Light salmon for Q1
            2: "#ff7f50",  # Coral for Q2
            3: "#ff4500",  # OrangeRed for Q3
            4: "#8b0000"   # DarkRed for Q4
        }
        color = quarter_colors.get(current_quarter, "#000000")
        
        # Calculate confidence percentage (inverse relationship with interval width)
        confidence = max(0, min(100, 100 - (interval_width / expected_width.get(current_quarter, 25) * 100)))
        
        # Create SVG
        svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa" rx="5" ry="5" />
  
  <!-- Title -->
  <text x="10" y="15" font-family="Arial" font-size="12" fill="#555">Prediction Confidence (Q{current_quarter})</text>
  
  <!-- Confidence Range Bar -->
  <rect x="{lower_x}" y="{height/2 - 5}" width="{upper_x - lower_x}" height="10" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="1" rx="2" ry="2" />
  
  <!-- Prediction Point -->
  <circle cx="{pred_x}" cy="{height/2}" r="6" fill="{color}" />
  <text x="{pred_x}" y="{height/2 - 10}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333" font-weight="bold">{prediction:.1f}</text>
  
  <!-- Range Labels -->
  <text x="{lower_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{lower_bound:.1f}</text>
  <text x="{upper_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">{upper_bound:.1f}</text>
  
  <!-- Confidence Percentage -->
  <text x="{width - 10}" y="{height - 10}" text-anchor="end" font-family="Arial" font-size="12" fill="#333" font-weight="bold">{confidence:.0f}% confidence</text>
</svg>"""
        
        # Add current score marker if available
        if home_score is not None and home_score > 0:
            score_x = to_svg_x(home_score)
            svg = svg.replace('</svg>', f"""  <!-- Current Score -->
  <circle cx="{score_x}" cy="{height/2}" r="5" fill="none" stroke="#28a745" stroke-width="2" />
  <text x="{score_x}" y="{height/2 + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#28a745" font-weight="bold">{home_score}</text>
</svg>""")
        
        return svg
    
    def dynamically_adjust_interval(
        self, 
        prediction: float, 
        current_quarter: int, 
        historic_accuracy: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Dynamically adjust prediction interval based on historical accuracy.
        
        Args:
            prediction: Predicted score
            current_quarter: Current game quarter (0-4)
            historic_accuracy: Dictionary with historical coverage statistics
            
        Returns:
            Tuple of (adjusted_lower_bound, adjusted_upper_bound, confidence_percentage)
        """
        # Get base prediction interval
        lower, upper, width = self.calculate_prediction_interval(prediction, current_quarter)
        
        # Exit early if no historical data available
        if not historic_accuracy or current_quarter not in historic_accuracy:
            expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}.get(current_quarter, 25)
            confidence = max(0, min(100, 100 - (width / expected_width * 100)))
            return lower, upper, confidence
        
        # Get historical coverage for this quarter
        accuracy = historic_accuracy[current_quarter]
        
        # Widen interval if coverage is too low
        if accuracy.get('coverage_pct', 95) < 90:
            # Calculate widening factor based on how far below target
            widening_factor = (95 - accuracy.get('coverage_pct', 95)) / 50
            width *= (1.0 + widening_factor)
            self._print_debug(f"Widening Q{current_quarter} interval by factor {1.0 + widening_factor:.2f} due to low coverage ({accuracy.get('coverage_pct', 95):.1f}%)")
            
        # Narrow interval if coverage is too high
        elif accuracy.get('coverage_pct', 95) > 98:
            # Calculate narrowing factor based on how far above target
            narrowing_factor = (accuracy.get('coverage_pct', 95) - 95) / 300
            width *= (1.0 - narrowing_factor)
            self._print_debug(f"Narrowing Q{current_quarter} interval by factor {1.0 - narrowing_factor:.2f} due to high coverage ({accuracy.get('coverage_pct', 95):.1f}%)")
        
        # Recalculate bounds with adjusted width
        lower = max(0, prediction - width/2)
        upper = prediction + width/2
        
        # Calculate confidence percentage
        expected_width = {0: 30, 1: 26, 2: 22, 3: 18, 4: 14}.get(current_quarter, 25)
        confidence = max(0, min(100, 100 - (width / expected_width * 100)))
        
        return lower, upper, confidence


import joblib
import pandas as pd
import numpy as np
import traceback
from datetime import datetime # Make sure datetime is imported if used within validate
from typing import Optional, Dict, Any, List, Tuple # Import necessary types

# Assuming these classes are defined in your feature_engineering module or elsewhere
# from feature_engineering import (NBAFeatureEngine, QuarterSpecificModelSystem,
#                                  PredictionUncertaintyEstimator, ConfidenceVisualizer)
# from sklearn.metrics import r2_score # Needed for validate_enhanced_predictions


def generate_enhanced_predictions(
    live_games_df: pd.DataFrame,
    model_payload_path: str, # Path to the saved model payload
    feature_generator: Any, # Instance of NBAFeatureEngine
    quarter_system: Any, # Instance of QuarterSpecificModelSystem
    uncertainty_estimator: Any, # Instance of PredictionUncertaintyEstimator
    confidence_viz: Any, # Instance of ConfidenceVisualizer
    historical_games_df: Optional[pd.DataFrame] = None,
    team_stats_df: Optional[pd.DataFrame] = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generates enhanced predictions for live games using a loaded model payload.

    Args:
        live_games_df: DataFrame of games to predict.
        model_payload_path: Path to the file containing the model and required features.
        feature_generator: Instance of NBAFeatureEngine.
        quarter_system: Instance of QuarterSpecificModelSystem.
        uncertainty_estimator: Instance of PredictionUncertaintyEstimator.
        confidence_viz: Instance of ConfidenceVisualizer.
        historical_games_df: Optional DataFrame of historical games for feature generation.
        team_stats_df: Optional DataFrame of team stats for feature generation.
        debug: Whether to print debug messages.

    Returns:
        DataFrame with predictions, uncertainty bounds, and confidence SVG.
    """
    # --- Load Model and Required Features ---
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model']
        required_features = model_payload['features']
        if debug: print(f"[generate_enhanced_predictions] Loaded model and {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        print(f"ERROR: Model payload file not found at {model_payload_path}. Cannot generate predictions.")
        return pd.DataFrame() # Return empty DataFrame on critical error
    except KeyError as e:
         print(f"ERROR: Model payload at {model_payload_path} is missing key: {e}. Requires 'model' and 'features'.")
         return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to load model payload from {model_payload_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    # --- Generate Features ---
    # Use the generate_all_features wrapper method
    # Ensure it generates all features needed by both main_model and quarter_system
    try:
        features_df = feature_generator.generate_all_features(
            live_games_df,
            historical_games_df=historical_games_df,
            team_stats_df=team_stats_df
        )
        if features_df.empty or features_df.shape[0] != live_games_df.shape[0]:
             print(f"ERROR: Feature generation failed or returned unexpected shape. Input: {live_games_df.shape}, Output: {features_df.shape}")
             return pd.DataFrame()
        if debug: print(f"[generate_enhanced_predictions] Feature generation complete. Shape: {features_df.shape}")
    except Exception as e:
        print(f"ERROR: Exception during feature generation: {e}")
        traceback.print_exc()
        return pd.DataFrame()


    # --- Main Model Prediction ---
    if not required_features: # Should have been caught during loading, but double-check
         print("ERROR: No required features list available for the main model. Cannot predict.")
         return pd.DataFrame()

    # Ensure all required features are present in the generated features_df
    missing_features_check = []
    for f in required_features:
        if f not in features_df.columns:
            features_df[f] = 0 # Use 0 as default, or implement better default logic
            missing_features_check.append(f)
    if missing_features_check:
        if debug: print(f"[generate_enhanced_predictions] Warning: Added {len(missing_features_check)} missing columns with default 0 for main model: {missing_features_check[:5]}...")
    else:
         if debug: print(f"[generate_enhanced_predictions] All {len(required_features)} required features found.")

    # Select ONLY required features IN THE CORRECT ORDER using reindex
    try:
        # Use fill_value=0 for any columns that might still be missing after the check (shouldn't happen often)
        X_main = features_df.reindex(columns=required_features, fill_value=0)
        if X_main.isnull().any().any():
             print("[generate_enhanced_predictions] Warning: Null values detected in feature matrix X_main after reindex. Check defaults/data.")
             # Optionally fill NaNs again if needed: X_main = X_main.fillna(0)
    except Exception as e:
         print(f"[generate_enhanced_predictions] CRITICAL Error selecting/reindexing required features: {e}. Cannot predict.")
         traceback.print_exc()
         return pd.DataFrame()

    # --- Predict ---
    try:
        main_predictions = main_model.predict(X_main)
        if debug: print(f"[generate_enhanced_predictions] Main model predictions generated ({len(main_predictions)}).")
    except Exception as e:
        print(f"[generate_enhanced_predictions] Error during main model prediction: {e}")
        traceback.print_exc()
        # Fallback prediction using feature_generator defaults
        avg_score = feature_generator.defaults.get('avg_pts_for', 110.0)
        main_predictions = np.full(len(features_df), avg_score)
        if debug: print(f"[generate_enhanced_predictions] Using fallback predictions ({avg_score}).")


    # --- Process Each Game for Ensemble & Uncertainty ---
    results = []
    # Placeholder for historical accuracy - load real stats if available for dynamic intervals
    historic_accuracy = uncertainty_estimator.get_coverage_stats().set_index('quarter').to_dict('index') if hasattr(uncertainty_estimator, 'get_coverage_stats') else None

    # Use iterrows carefully on the potentially large features_df. Consider optimization if this loop is slow.
    # Ensure index aligns with main_predictions
    features_df = features_df.reset_index() # Ensure index is 0, 1, 2... matching main_predictions array

    for i, game_row in features_df.iterrows():
        game_id_log = game_row.get('game_id', f'index_{i}') # For logging
        try:
            # Convert row to dict for quarter_system processing
            # Ensure all necessary columns (like quarter scores, current_quarter) are present
            game_data_dict = game_row.to_dict()
            main_pred = float(main_predictions[i]) # Ensure float

            # Get ensemble prediction using quarter system
            # Ensure predict_final_score handles potential missing keys in game_data_dict
            ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
                game_data_dict=game_data_dict,
                main_model_prediction=main_pred,
                weight_manager=quarter_system.weight_manager # Assuming weight_manager is accessible like this
            )

            # Estimate uncertainty
            current_quarter = int(game_data_dict.get('current_quarter', 0))
            # Use calculated score_differential if available, else calculate from scores
            score_diff_feat = game_data_dict.get('score_differential')
            home_score_live = float(game_data_dict.get('home_score', 0))
            away_score_live = float(game_data_dict.get('away_score', 0))
            score_margin = abs(float(score_diff_feat if pd.notna(score_diff_feat) else (home_score_live - away_score_live)))
            momentum = float(game_data_dict.get('cumulative_momentum', 0.0)) # Example context feature

            # Use dynamic interval adjustment if historic_accuracy was loaded
            lower_b, upper_b, conf_pct = uncertainty_estimator.dynamically_adjust_interval(
                prediction=float(ensemble_pred),
                current_quarter=current_quarter,
                historic_accuracy=historic_accuracy # Pass loaded historical coverage dict
            )
            # Fallback if dynamic adjustment isn't used/fails
            # lower_b, upper_b, _ = uncertainty_estimator.calculate_prediction_interval(
            #     prediction=float(ensemble_pred),
            #     current_quarter=current_quarter,
            #     score_margin=score_margin,
            #     momentum=momentum
            # )
            # conf_pct = max(0.0, min(100.0, 100.0 - ((upper_b-lower_b) / uncertainty_estimator.EXPECTED_RANGES.get(current_quarter, 25.0) * 75.0)))


            # Create SVG indicator
            svg_indicator = confidence_viz.create_confidence_svg(
                prediction=float(ensemble_pred),
                lower_bound=float(lower_b),
                upper_bound=float(upper_b),
                current_quarter=current_quarter,
                current_home_score=float(home_score_live) # Pass current score if live
            )

            # Compile result
            result_row = {
                'game_id': game_data_dict.get('game_id'),
                'home_team': game_data_dict.get('home_team'),
                'away_team': game_data_dict.get('away_team'),
                'game_date': game_data_dict.get('game_date'),
                'current_quarter': current_quarter,
                'home_score': home_score_live,
                'away_score': away_score_live,
                'main_model_pred': main_pred,
                'quarter_model_sum_pred': breakdown.get('quarter_model_sum'), # Changed key for clarity
                'ensemble_pred': float(ensemble_pred),
                'lower_bound': float(lower_b),
                'upper_bound': float(upper_b),
                'confidence_pct': float(conf_pct),
                'confidence_svg': svg_indicator,
                'main_weight': breakdown.get('weights', {}).get('main'),
                'quarter_weight': breakdown.get('weights', {}).get('quarter'),
                # Add predicted quarter scores if needed
                **{f'predicted_{k}': v for k, v in breakdown.get('quarter_predictions', {}).items()}
            }
            results.append(result_row)

        except Exception as e:
            if debug: print(f"Error processing game {game_id_log}: {e}")
            traceback.print_exc()
            # Append minimal error info - ensure keys match expected output structure
            results.append({
                'game_id': game_id_log,
                'home_team': game_row.get('home_team'),
                'away_team': game_row.get('away_team'),
                'game_date': game_row.get('game_date'),
                'error': str(e),
                # Add defaults for other columns to maintain structure
                'current_quarter': game_row.get('current_quarter', 0),
                'home_score': game_row.get('home_score', 0),
                'away_score': game_row.get('away_score', 0),
                'main_model_pred': np.nan,
                'quarter_model_sum_pred': np.nan,
                'ensemble_pred': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'confidence_pct': np.nan,
                'confidence_svg': '<svg>Error</svg>',
                'main_weight': np.nan,
                'quarter_weight': np.nan,
            })

    return pd.DataFrame(results)

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import joblib
from datetime import datetime, timedelta
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- Validation Framework --------------------
def validate_enhanced_predictions(
    model_payload_path: str, # Path to the saved model payload
    feature_generator: Any, # Instance of NBAFeatureEngine
    historical_df: Optional[pd.DataFrame] = None, # Source of historical games for testing
    num_test_games: int = 20,
    debug: bool = False,
    # Add other necessary components if needed by feature_generator or quarter_system
    quarter_system: Optional[Any] = None, # Pass initialized QuarterSystem if needed
    # Supabase/DB dependencies removed assuming historical_df is provided or loaded externally
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate enhanced prediction system on historical games using a loaded model payload.

    Args:
        model_payload_path: Path to the file containing the model and required features.
        feature_generator: Instance of NBAFeatureEngine.
        historical_df: DataFrame with historical games (MUST be provided for validation).
        num_test_games: Number of historical games to test from the provided df.
        debug: Whether to print debug messages.
        quarter_system: Pre-initialized instance of QuarterSpecificModelSystem.

    Returns:
        tuple: (DataFrame with detailed validation results per game/quarter,
                DataFrame with aggregated improvement metrics by quarter)
    """
    # --- Load Model and Required Features ---
    try:
        model_payload = joblib.load(model_payload_path)
        main_model = model_payload['model']
        required_features = model_payload['features']
        if debug: print(f"[validate_enhanced_predictions] Loaded model and {len(required_features)} features from {model_payload_path}")
    except FileNotFoundError:
        print(f"ERROR: Model payload file not found at {model_payload_path}. Cannot run validation.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on critical error
    except KeyError as e:
         print(f"ERROR: Model payload at {model_payload_path} is missing key: {e}. Requires 'model' and 'features'.")
         return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to load model payload from {model_payload_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    # --- Input Validation and Test Data Selection ---
    if historical_df is None or historical_df.empty:
         print("ERROR: historical_df must be provided for validation.")
         return pd.DataFrame(), pd.DataFrame()

    if 'game_date' not in historical_df.columns:
         print("ERROR: historical_df missing 'game_date' column.")
         return pd.DataFrame(), pd.DataFrame()
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])

    # Use most recent games from provided DataFrame
    test_games = historical_df.sort_values('game_date', ascending=False).head(num_test_games)
    if debug: print(f"Selected {len(test_games)} most recent games for validation.")

    if len(test_games) == 0:
        print("Warning: No test games selected.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Initialize Quarter System (if not passed) ---
    if quarter_system is None:
        if debug: print("Initializing QuarterSpecificModelSystem for validation...")
        quarter_system = QuarterSpecificModelSystem(feature_generator, debug=debug)
        quarter_system.load_models() # Load quarter-specific models

    # Check if quarter_system requires weight_manager and if it's present
    if not hasattr(quarter_system, 'weight_manager'):
        print("ERROR: QuarterSpecificModelSystem instance does not have 'weight_manager'. Ensemble calculation might fail.")
        # Depending on implementation, you might need to initialize it here or ensure it's part of QuarterSystem init
        # Example: from feature_engineering import EnsembleWeightManager
        # quarter_system.weight_manager = EnsembleWeightManager(debug=debug)


    # --- Process Each Test Game ---
    validation_results = []
    required_base_cols = ['game_id', 'home_team', 'away_team', 'home_score', 'away_score'] # Check these exist in test_games
    if not all(col in test_games.columns for col in required_base_cols):
        print(f"ERROR: Test games missing one or more required columns: {required_base_cols}")
        return pd.DataFrame(), pd.DataFrame()


    for _, game in test_games.iterrows():
        actual_home_score = float(game['home_score'])
        game_id_log = game['game_id']

        # Test predictions for each quarter state (0 = pregame, 1-4 = end of quarter)
        for test_quarter in range(0, 5):
            if debug and test_quarter == 0:
                print(f"\n--- Testing Game: {game_id_log} ({game['home_team']} vs {game['away_team']}) ---")
            if debug: print(f"Simulating state at end of Q{test_quarter}...")

            # Create a simulated game state DataFrame (single row)
            sim_data = {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game['game_date'], # Already datetime
                'current_quarter': test_quarter,
                # Include other base columns if feature generator needs them
            }

            # Add known quarter scores up to the simulated point
            current_home_q_score = 0
            current_away_q_score = 0
            for q in range(1, test_quarter + 1):
                 home_q_col = f'home_q{q}'
                 away_q_col = f'away_q{q}'
                 sim_data[home_q_col] = game.get(home_q_col, 0)
                 sim_data[away_q_col] = game.get(away_q_col, 0)
                 current_home_q_score += sim_data[home_q_col]
                 current_away_q_score += sim_data[away_q_col]

            # Add live score based on simulated state (needed for some features/context)
            sim_data['home_score'] = current_home_q_score
            sim_data['away_score'] = current_away_q_score
            # sim_data['score_differential'] = current_home_q_score - current_away_q_score # This will be recalculated by feature eng

            sim_game_df = pd.DataFrame([sim_data])

            # Generate features for this simulated state
            try:
                # Use the wrapper; pass historical_df for context like rest days
                features_df = feature_generator.generate_all_features(
                    sim_game_df,
                    historical_games_df=historical_df # Pass full history for context
                    # team_stats_df can be added if needed
                )

                if features_df.empty:
                    raise ValueError("Feature generation returned empty DataFrame.")

                # Select features for main model using loaded required_features
                missing_features_check = []
                for f in required_features:
                    if f not in features_df.columns:
                        features_df[f] = 0 # Default fill
                        missing_features_check.append(f)
                if missing_features_check and debug:
                     print(f"  Q{test_quarter}: Warning - Added {len(missing_features_check)} missing main model features: {missing_features_check[:5]}...")

                # Select features IN ORDER
                X_main_sim = features_df.reindex(columns=required_features, fill_value=0)

                # Get main model prediction
                main_pred = float(main_model.predict(X_main_sim)[0])

                # Get ensemble prediction using quarter system
                # Pass the feature-rich row as a dict
                ensemble_pred, confidence, breakdown = quarter_system.predict_final_score(
                     game_data_dict=features_df.iloc[0].to_dict(),
                     main_model_prediction=main_pred,
                     weight_manager=quarter_system.weight_manager # Pass weight manager
                )
                ensemble_pred = float(ensemble_pred) # Ensure float

                # Calculate prediction errors
                main_error = main_pred - actual_home_score
                ensemble_error = ensemble_pred - actual_home_score

                # Record results (include individual model errors for later analysis if needed)
                # Also include quarter_model_sum_pred from breakdown
                validation_results.append({
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_date': game['game_date'],
                    'current_quarter': test_quarter, # State being simulated (0-4)
                    'actual_home_score': actual_home_score,
                    'main_prediction': main_pred,
                    'quarter_model_sum_prediction': breakdown.get('quarter_model_sum'), # Store this
                    'ensemble_prediction': ensemble_pred,
                    'main_error': main_error,
                    'quarter_model_sum_error': breakdown.get('quarter_model_sum', np.nan) - actual_home_score, # Store this error too!
                    'ensemble_error': ensemble_error,
                    'main_abs_error': abs(main_error),
                    'ensemble_abs_error': abs(ensemble_error),
                    'main_squared_error': main_error**2,
                    'ensemble_squared_error': ensemble_error**2,
                    'confidence': confidence, # From predict_final_score
                    'main_weight': breakdown.get('weights', {}).get('main'),
                    'quarter_weight': breakdown.get('weights', {}).get('quarter')
                })
                if debug: print(f"  Q{test_quarter}: Main={main_pred:.1f}, Ens={ensemble_pred:.1f} (Actual={actual_home_score}) -> MAE Ens={abs(ensemble_error):.1f}")


            except Exception as e:
                if debug:
                    print(f"  Q{test_quarter}: Error processing game {game_id_log}: {e}")
                    # Optionally print traceback for detailed debugging:
                    # traceback.print_exc()
                # Record error state
                validation_results.append({
                    'game_id': game['game_id'], 'home_team': game['home_team'], 'away_team': game['away_team'],
                    'game_date': game['game_date'], 'current_quarter': test_quarter,
                    'actual_home_score': actual_home_score, 'main_prediction': np.nan,
                    'quarter_model_sum_prediction': np.nan, 'ensemble_prediction': np.nan,
                    'main_error': np.nan, 'quarter_model_sum_error': np.nan, 'ensemble_error': np.nan,
                    'main_abs_error': np.nan, 'ensemble_abs_error': np.nan,
                    'main_squared_error': np.nan, 'ensemble_squared_error': np.nan,
                    'confidence': np.nan, 'main_weight': np.nan, 'quarter_weight': np.nan,
                    'error_flag': 1 # Add flag to indicate error
                })


    # --- Aggregate and Return Results ---
    validation_df = pd.DataFrame(validation_results)

    if validation_df.empty:
        if debug: print("No validation results generated.")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate aggregated error metrics by quarter
    # Use .agg named aggregation for cleaner output columns in Pandas >= 0.25
    error_metrics = validation_df.groupby('current_quarter').agg(
        count=('game_id', 'size'),
        main_mae=('main_abs_error', 'mean'),
        main_mae_std=('main_abs_error', 'std'),
        main_rmse=('main_squared_error', lambda x: np.sqrt(x.mean())), # Calculate RMSE from mean squared error
        qsum_mae=('quarter_model_sum_error', lambda x: np.abs(x).mean()), # Add metrics for quarter sum model
        qsum_rmse=('quarter_model_sum_error', lambda x: np.sqrt((x**2).mean())),
        ensemble_mae=('ensemble_abs_error', 'mean'),
        ensemble_mae_std=('ensemble_abs_error', 'std'),
        ensemble_rmse=('ensemble_squared_error', lambda x: np.sqrt(x.mean()))
    ).reset_index()


    if debug:
        print("\n--- Validation Error Metrics by Quarter ---")
        print(error_metrics.round(2))

    # Calculate improvement metrics and R²
    improvements = []
    # Need r2_score if calculating R²
    try: from sklearn.metrics import r2_score
    except ImportError: r2_score = None; print("Warning: sklearn.metrics not found, cannot calculate R2.")


    for quarter in range(0, 5):
        quarter_data = validation_df[(validation_df['current_quarter'] == quarter) & validation_df['ensemble_prediction'].notna()].copy()
        if not quarter_data.empty:
            metrics = {
                'quarter': quarter,
                'sample_size': len(quarter_data)
            }

            # Calculate Mean Errors (MAE, RMSE) directly from aggregated metrics if preferred
            agg_row = error_metrics[error_metrics['current_quarter'] == quarter].iloc[0]
            metrics.update({
                'main_mae': agg_row['main_mae'], 'ensemble_mae': agg_row['ensemble_mae'],
                'main_rmse': agg_row['main_rmse'], 'ensemble_rmse': agg_row['ensemble_rmse'],
            })

            # Calculate % Improvements
            metrics['mae_improvement_pct'] = ((metrics['main_mae'] - metrics['ensemble_mae']) / metrics['main_mae'] * 100) if metrics['main_mae'] else 0
            metrics['rmse_improvement_pct'] = ((metrics['main_rmse'] - metrics['ensemble_rmse']) / metrics['main_rmse'] * 100) if metrics['main_rmse'] else 0

            # Calculate R² if possible
            if r2_score:
                y_true = quarter_data['actual_home_score']
                metrics['main_r2'] = r2_score(y_true, quarter_data['main_prediction'])
                metrics['ensemble_r2'] = r2_score(y_true, quarter_data['ensemble_prediction'])
                metrics['r2_improvement'] = metrics['ensemble_r2'] - metrics['main_r2']
            else:
                 metrics['main_r2'], metrics['ensemble_r2'], metrics['r2_improvement'] = np.nan, np.nan, np.nan

            improvements.append(metrics)

    improvement_df = pd.DataFrame(improvements)

    if debug:
        print("\n--- Validation Improvement by Quarter ---")
        print(improvement_df.round(2))

    # Return both detailed results and aggregated improvements
    # The detailed validation_df now includes 'quarter_model_sum_prediction' and 'quarter_model_sum_error'
    # which can be used to update the EnsembleWeightManager's error history correctly.
    return validation_df, improvement_df


def get_recommended_model_params(quarter, model_type=None):
    """
    Returns optimized hyperparameters for specific quarter models.
    
    Args:
        quarter: Quarter number (1-4)
        model_type: Model type to override default recommendation (RandomForest, XGBoost, Ridge)
        
    Returns:
        dict: Hyperparameters for the recommended model type
    """
    # Return hyperparameters for specific model type if requested
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
    elif model_type == "Ridge":
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
    
    # Return quarter-specific optimal model configuration
    if quarter == 1:
        # For Q1, use XGBoost with parameters optimized for early game prediction
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 3,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    elif quarter == 2:
        # For Q2, use XGBoost with parameters that incorporate Q1 information
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        }
    elif quarter == 3:
        # For Q3, use XGBoost with parameters that balance complexity and robustness
        return {
            'model_type': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
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
    else:  # quarter == 4 or any other value
        # For Q4, use Ridge regression for stability and robustness
        return {
            'model_type': 'Ridge',
            'params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'solver': 'auto',
                'random_state': 42
            }
        }
