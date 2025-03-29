# backend/nba_score_prediction/feature_engineering.py

# feature_engineering.py - Unified Feature Engineering for NBA Score Prediction

"""
NBAFeatureEngine - Unified module for NBA score prediction feature engineering.
This module includes:
  - Comprehensive feature engineering based on historical game stats
  - Integration of season-level team stats (Win %, Avg Pts For/Against)
  - Standard calculations for Pace, Possessions, Efficiency, Rebound Rate, Turnover Rate
  - Vectorized calculations for rolling features, team history, and rest days
  - Team name normalization utility
  - Basic logging and profiling capabilities
"""

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
# import math # Removed as math functions handled by numpy/pandas
from functools import wraps
import functools
from typing import Dict, List, Tuple, Optional, Union, Any

# -------------------- Simple Logging Function --------------------
def log(message, level="INFO", debug=False):
    """Simple logging function with timestamps."""
    if debug or level != "DEBUG":
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")

# -------------------- NBAFeatureEngine Class --------------------
class NBAFeatureEngine:
    """Core feature engineering for NBA score prediction models."""

    def __init__(self, supabase_client: Optional[Any] = None, debug: bool = False):
        """
        Initialize the feature engine.

        Args:
            supabase_client: Optional configured Supabase client instance.
                             (Currently unused directly in calculations).
            debug: Whether to print debug messages.
        """
        self.debug = debug
        self.supabase_client = supabase_client
        # if self.supabase_client is None:
        #    self.log("Supabase client not provided.", "WARNING")

        # NBA average values for fallback situations / missing data
        # Consider updating these periodically based on recent league trends
        self.defaults = {
            'win_pct': 0.5,
            'offensive_rating': 115.0, # Pts per 100 possessions (updated example)
            'defensive_rating': 115.0, # Pts allowed per 100 possessions (updated example)
            'pace': 100.0,             # Possessions per 48 minutes (updated example)
            'home_advantage': 3.0,     # Typical point advantage (can vary)
            'quarter_points': 28.5,    # Average points per quarter (updated example)
            'estimated_possessions': 100, # Fallback per team per game
            'avg_pts_for': 115.0,      # Updated example
            'avg_pts_against': 115.0,  # Updated example
            'oreb_pct': 0.23,          # Example avg OReb%
            'trb_pct': 0.5,            # Example avg TRB%
            'tov_rate': 0.13,          # Example avg TOV rate
        }

        # League-wide averages (can be refined based on actual data)
        # Could be loaded dynamically or updated based on historical_df analysis
        self.league_averages = {
            'score': self.defaults['avg_pts_for'],
            'quarter_scores': {1: 28.5, 2: 28.5, 3: 28.0, 4: 29.0} # Example values
        }

        # Quarter-specific feature sets (For potential future quarter-based models)
        # Not directly used in the main generate_all_features flow for pre-game prediction
        self.quarter_feature_sets = self._get_optimized_feature_sets()

    def log(self, message, level="INFO"):
        """Print log messages if debug mode is on."""
        if self.debug or level != "DEBUG":
            # Use a consistent prefix for engine logs
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [NBA FeatureEngine] {level}: {message}")

    def _determine_season(self, game_date: pd.Timestamp) -> str:
        """
        Determines the NBA season string (e.g., '2023-24') based on the game date.
        NBA season typically starts in October.
        """
        if pd.isna(game_date):
             # Handle cases where date might be missing, perhaps return a default season or raise error
             self.log("Missing game_date in _determine_season, cannot determine season.", "WARNING")
             return "Unknown_Season" # Or another placeholder

        year = game_date.year
        month = game_date.month
        if month >= 10: # October, November, December belong to the season starting that year
            return f"{year}-{str(year+1)[-2:]}"
        else: # January to September belong to the season that started the previous year
            return f"{year-1}-{str(year)[-2:]}"

    # @NBAFeatureEngine.profile_time(debug_mode=True) # Optional profiling
    def add_season_context_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges season-level statistics onto the game data DataFrame.

        Args:
            df: DataFrame with game data, must include 'game_date', 'home_team', 'away_team'.
            team_stats_df: DataFrame with season stats (e.g., from nba_historical_team_stats),
                           must include 'team_name', 'season', and the stats columns.

        Returns:
            DataFrame with added season context features, or original df if merge fails.
        """
        self.log("Adding season context features...", level="INFO")
        if team_stats_df is None or team_stats_df.empty:
            self.log("`team_stats_df` is empty or not provided. Skipping season context features.", "WARNING")
            # Add placeholder columns to df to avoid downstream errors if needed
            placeholder_cols = [
                 'home_season_win_pct', 'away_season_win_pct', 'home_season_avg_pts_for',
                 'away_season_avg_pts_for', 'home_season_avg_pts_against', 'away_season_avg_pts_against',
                 'season_win_pct_diff', 'season_pts_for_diff', 'season_pts_against_diff'
            ]
            for col in placeholder_cols:
                 if col not in df.columns:
                      default_val = self.defaults['win_pct'] if 'win_pct' in col else \
                                    self.defaults['avg_pts_for'] if 'pts_for' in col else \
                                    self.defaults['avg_pts_against'] if 'pts_against' in col else 0
                      df[col] = default_val
            return df
        if 'game_date' not in df.columns:
             self.log("`game_date` column missing from input df. Cannot determine season for merging stats.", "ERROR")
             return df # Cannot proceed without game date

        result_df = df.copy()

        # --- Ensure Required Columns in team_stats_df ---
        req_team_stats_cols = ['team_name', 'season', 'wins_all_percentage',
                               'points_for_avg_all', 'points_against_avg_all']
                               # Add 'current_form' if using it: 'current_form']
        missing_req = [col for col in req_team_stats_cols if col not in team_stats_df.columns]
        if missing_req:
            self.log(f"Missing required columns in `team_stats_df`: {missing_req}. Cannot add season features.", "ERROR")
            return df # Or add placeholders as above? Returning original df for safety.

        # --- Prepare for Merge ---
        # 1. Determine season for each game in result_df
        try:
            # Convert game_date to datetime just in case it's not already
            result_df['game_date'] = pd.to_datetime(result_df['game_date'])
            result_df['season'] = result_df['game_date'].apply(self._determine_season)
        except Exception as e:
            self.log(f"Error determining season from game_date: {e}. Cannot merge season stats.", "ERROR")
            traceback.print_exc()
            return df

        # 2. Select and rename columns in team_stats_df for clarity before merging
        stats_to_merge = team_stats_df[req_team_stats_cols].copy()

        # 3. *** Activate Normalization BEFORE Merge for Robustness ***
        self.log("Normalizing team names for season stats merge...", level="DEBUG")
        # Check if normalization function exists and apply it
        if hasattr(self, 'normalize_team_name'):
             result_df['home_team_norm'] = result_df['home_team'].apply(self.normalize_team_name)
             result_df['away_team_norm'] = result_df['away_team'].apply(self.normalize_team_name)
             stats_to_merge['team_name_norm'] = stats_to_merge['team_name'].apply(self.normalize_team_name)
             merge_on_cols_left = ['home_team_norm', 'season']
             merge_on_cols_right = ['team_name_norm', 'season']
             merge_on_cols_left_away = ['away_team_norm', 'season'] # Use normalized away team
        else:
             self.log("normalize_team_name method not found, merging on raw team names.", "WARNING")
             merge_on_cols_left = ['home_team', 'season']
             merge_on_cols_right = ['team_name', 'season']
             merge_on_cols_left_away = ['away_team', 'season'] # Use raw away team

        # --- Merge for Home Team ---
        self.log("Merging season stats for home teams...", level="DEBUG")
        home_stats_to_merge = stats_to_merge.rename(columns={
            'wins_all_percentage': 'home_season_win_pct',
            'points_for_avg_all': 'home_season_avg_pts_for',
            'points_against_avg_all': 'home_season_avg_pts_against',
            # 'current_form': 'home_current_form' # If using
        })
        result_df = pd.merge(
            result_df,
            home_stats_to_merge,
            how='left',
            left_on=merge_on_cols_left,
            right_on=merge_on_cols_right
        )
        # Drop the potentially duplicated merge columns from the right side
        if merge_on_cols_right[0] != merge_on_cols_left[0]: result_df = result_df.drop(columns=[merge_on_cols_right[0]], errors='ignore')
        if merge_on_cols_right[1] != merge_on_cols_left[1]: result_df = result_df.drop(columns=[merge_on_cols_right[1]], errors='ignore')


        # --- Merge for Away Team ---
        self.log("Merging season stats for away teams...", level="DEBUG")
        away_stats_to_merge = stats_to_merge.rename(columns={
            'wins_all_percentage': 'away_season_win_pct',
            'points_for_avg_all': 'away_season_avg_pts_for',
            'points_against_avg_all': 'away_season_avg_pts_against',
            # 'current_form': 'away_current_form' # If using
        })
        result_df = pd.merge(
            result_df,
            away_stats_to_merge,
            how='left',
            left_on=merge_on_cols_left_away, # Use appropriate left key (normalized or raw)
            right_on=merge_on_cols_right
        )
        if merge_on_cols_right[0] != merge_on_cols_left_away[0]: result_df = result_df.drop(columns=[merge_on_cols_right[0]], errors='ignore')
        if merge_on_cols_right[1] != merge_on_cols_left_away[1]: result_df = result_df.drop(columns=[merge_on_cols_right[1]], errors='ignore')

        # --- Fill NaNs for Missing Season Data ---
        # Use defaults or league averages for teams/seasons not found in team_stats_df
        self.log("Filling potential NaNs from season stat merge...", level="DEBUG")
        result_df['home_season_win_pct'] = result_df['home_season_win_pct'].fillna(self.defaults['win_pct'])
        result_df['away_season_win_pct'] = result_df['away_season_win_pct'].fillna(self.defaults['win_pct'])
        result_df['home_season_avg_pts_for'] = result_df['home_season_avg_pts_for'].fillna(self.defaults['avg_pts_for'])
        result_df['away_season_avg_pts_for'] = result_df['away_season_avg_pts_for'].fillna(self.defaults['avg_pts_for'])
        result_df['home_season_avg_pts_against'] = result_df['home_season_avg_pts_against'].fillna(self.defaults['avg_pts_against'])
        result_df['away_season_avg_pts_against'] = result_df['away_season_avg_pts_against'].fillna(self.defaults['avg_pts_against'])
        # result_df['home_current_form'] = result_df['home_current_form'].fillna('N/A') # Or a numeric encoding
        # result_df['away_current_form'] = result_df['away_current_form'].fillna('N/A')

        # --- Derived Season Features ---
        result_df['season_win_pct_diff'] = result_df['home_season_win_pct'] - result_df['away_season_win_pct']
        result_df['season_pts_for_diff'] = result_df['home_season_avg_pts_for'] - result_df['away_season_avg_pts_for']
        result_df['season_pts_against_diff'] = result_df['home_season_avg_pts_against'] - result_df['away_season_avg_pts_against']
        # Net points average diff
        result_df['season_net_pts_diff'] = (result_df['home_season_avg_pts_for'] - result_df['home_season_avg_pts_against']) - \
                                           (result_df['away_season_avg_pts_for'] - result_df['away_season_avg_pts_against'])


        # Drop intermediate columns if not needed later
        result_df = result_df.drop(columns=['season', 'home_team_norm', 'away_team_norm'], errors='ignore')

        self.log("Finished adding season context features.", level="INFO")
        return result_df

    # @NBAFeatureEngine.profile_time(debug_mode=True) # Optional profiling
    def generate_all_features(self, df: pd.DataFrame,
                              historical_games_df: Optional[pd.DataFrame] = None,
                              team_stats_df: Optional[pd.DataFrame] = None,
                              rolling_window: int = 10) -> pd.DataFrame:
        """
        Applies all feature engineering steps in sequence to the input DataFrame.
        Includes intra-game momentum calculation and rolling features for scores & momentum.
        Args:
            # ... (args remain the same)
        Returns:
            # ... (return remains the same)
        """
        # --- (Initial checks and data combination logic remains the same) ---
        if df is None or df.empty: return pd.DataFrame()
        if historical_games_df is None or historical_games_df.empty:
             self.log("`historical_games_df` is empty or not provided...", "WARNING")
             historical_games_df = pd.DataFrame(columns=df.columns)
        self.log(f"Starting comprehensive feature generation for {len(df)} games...", level="INFO")
        features_df = df.copy()
        self.log("Step 1/7: Preprocessing and combining data...", level="DEBUG")
        # ... (date conversion, combine df + historical_games_df into calc_df, sort) ...
        # Assume calc_df is created and sorted by game_date, game_id here

        try:
            # --- Sequence of Feature Engineering Steps ---

            # 2. Add Intra-Game Momentum features (needed for rolling momentum)
            self.log("Step 2/7: Adding intra-game momentum features...", level="DEBUG")
            calc_df = self.add_intra_game_momentum(calc_df) # Calculate on combined data

            # 3. Add Rolling features (Scores + Momentum)
            self.log("Step 3/7: Adding rolling features (scores & momentum)...", level="DEBUG")
            # This now uses the output of step 2
            calc_df = self.add_rolling_features(calc_df, window_sizes=[rolling_window])

            # 4. Add team history (previous matchup diff)
            self.log("Step 4/7: Adding team history features...", level="DEBUG")
            calc_df = self.add_team_history_features_vectorized(calc_df, rolling_window=rolling_window)

            # 5. Add rest days and back-to-back features
            self.log("Step 5/7: Adding rest features...", level="DEBUG")
            calc_df = self.add_rest_features_vectorized(calc_df)

            # 6. Add advanced metrics (eFG%, Pace, Poss, Eff, TRB%, TOV%)
            self.log("Step 6/7: Integrating advanced metrics...", level="DEBUG")
            calc_df = self.integrate_advanced_features(calc_df)

            # 7. Add Season Context Features (If team_stats_df provided)
            self.log("Step 7/7: Adding season context features...", level="DEBUG")
            if team_stats_df is not None and not team_stats_df.empty:
                 calc_df = self.add_season_context_features(calc_df, team_stats_df)
            else:
                 self.log("Skipping season context features (no data provided).", level="DEBUG")
                 # Add placeholder columns if necessary (as implemented before)
                 # ...

            # --- Filter back to the original games requested in `df` ---
            # ... (Filtering logic remains the same) ...
            original_game_ids = df['game_id'].unique()
            result_df = calc_df[calc_df['game_id'].isin(original_game_ids)].copy()

            # ... (Row count check remains the same) ...

            self.log(f"Comprehensive feature generation pipeline complete for {len(result_df)} games.", level="INFO")
            return result_df

        except Exception as e:
            self.log(f"Error during feature generation pipeline: {e}", "ERROR")
            traceback.print_exc()
            return df # Return original df on error

    @staticmethod
    def profile_time(func=None, debug_mode=None):
        """ Decorator to profile the execution time of a function. """
        # (Implementation unchanged)
        if func is not None:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                message = f"{func.__name__} executed in {execution_time:.4f} seconds"
                is_debug = False
                if args and hasattr(args[0], 'debug'): is_debug = args[0].debug
                if debug_mode is not None: is_debug = debug_mode
                if is_debug: print(message)
                return result
            return wrapper
        else:
            def decorator(f):
                @functools.wraps(f)
                def wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = f(*args, **kwargs)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    message = f"{f.__name__} executed in {execution_time:.4f} seconds"
                    is_debug = False
                    if args and hasattr(args[0], 'debug'): is_debug = args[0].debug
                    if debug_mode is not None: is_debug = debug_mode
                    if is_debug: print(message)
                    return result
                return wrapper
            return decorator

    # @profile_time(debug_mode=True) # Optional profiling
    def integrate_advanced_features(self, df):
        """
        Calculates and adds advanced metrics: eFG%, FT Rate, Possessions, Pace,
        Offensive/Defensive Efficiency, Rebound Rate (TRB%), Turnover Rate (TOV%), OReb%.

        Relies on box score columns being present in `df`. Handles missing columns/NaNs.

        Args:
            df: DataFrame with game data including necessary box score stats.

        Returns:
            DataFrame with added advanced features.
        """
        self.log("Integrating advanced features (Shooting, Poss, Pace, Eff, Reb%, TOV%)...", level="DEBUG")
        result_df = df.copy()

        # Define required columns from nba_historical_game_stats
        base_cols = ['home_score', 'away_score']
        fg_cols = ['home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted']
        tp_cols = ['home_3pm', 'home_3pa', 'away_3pm', 'away_3pa']
        ft_cols = ['home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted']
        # Include total rebounds explicitly for TRB%
        reb_cols = ['home_off_reb', 'home_def_reb', 'home_total_reb', 'away_off_reb', 'away_def_reb', 'away_total_reb']
        misc_cols = ['home_turnovers', 'away_turnovers', 'home_assists', 'away_assists'] # Assists might be used later

        all_req_cols = base_cols + fg_cols + tp_cols + ft_cols + reb_cols + misc_cols

        # Check for missing columns and fill numeric ones with 0 or NaN initially
        missing_cols = [col for col in all_req_cols if col not in result_df.columns]
        if missing_cols:
            self.log(f"Missing columns for advanced features: {missing_cols}. Filling with NaN.", "WARNING")
            for col in missing_cols:
                 if col not in result_df.columns:
                      result_df[col] = np.nan # Use NaN, then fill appropriately

        # Convert required columns to numeric, coercing errors to NaN, then fill NaNs with 0
        # (Filling with 0 is generally safe for counts/stats before calculations)
        for col in all_req_cols:
             if col in result_df.columns:
                  result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # --- Calculate Shooting Metrics (Vectorized) ---
        self.log("Calculating shooting percentages...", level="DEBUG")
        # FG%
        result_df['home_fg_pct'] = (result_df['home_fg_made'] / result_df['home_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['away_fg_pct'] = (result_df['away_fg_made'] / result_df['away_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['fg_pct_diff'] = result_df['home_fg_pct'] - result_df['away_fg_pct']

        # 3P%
        result_df['home_3p_pct'] = (result_df['home_3pm'] / result_df['home_3pa'].replace(0, np.nan)).fillna(0)
        result_df['away_3p_pct'] = (result_df['away_3pm'] / result_df['away_3pa'].replace(0, np.nan)).fillna(0)

        # eFG% (Effective Field Goal Percentage)
        result_df['home_efg_pct'] = ((result_df['home_fg_made'] + 0.5 * result_df['home_3pm']) / result_df['home_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['away_efg_pct'] = ((result_df['away_fg_made'] + 0.5 * result_df['away_3pm']) / result_df['away_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']

        # FT% and FT Rate (FTA / FGA)
        result_df['home_ft_pct'] = (result_df['home_ft_made'] / result_df['home_ft_attempted'].replace(0, np.nan)).fillna(0)
        result_df['away_ft_pct'] = (result_df['away_ft_made'] / result_df['away_ft_attempted'].replace(0, np.nan)).fillna(0)
        result_df['home_ft_rate'] = (result_df['home_ft_attempted'] / result_df['home_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['away_ft_rate'] = (result_df['away_ft_attempted'] / result_df['away_fg_attempted'].replace(0, np.nan)).fillna(0)
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']

        # --- Calculate Offensive Rebound Percentage (OReb%) ---
        # OReb% = Team OReb / (Team OReb + Opponent DReb)
        self.log("Calculating OReb%...", level="DEBUG")
        home_oreb_chance_denom = (result_df['home_off_reb'] + result_df['away_def_reb']).replace(0, np.nan)
        result_df['home_oreb_pct'] = (result_df['home_off_reb'] / home_oreb_chance_denom).fillna(self.defaults['oreb_pct']) # Use default if calculation fails

        away_oreb_chance_denom = (result_df['away_off_reb'] + result_df['home_def_reb']).replace(0, np.nan)
        result_df['away_oreb_pct'] = (result_df['away_off_reb'] / away_oreb_chance_denom).fillna(self.defaults['oreb_pct']) # Use default

        # --- Calculate Possessions (Using OReb% in formula) ---
        # Poss ~ FGA + 0.44*FTA - 1.07 * OReb% * MissedFGA + TOV
        self.log("Calculating possessions using standard formula...", level="DEBUG")

        # Calculate missed FGs
        home_missed_fg = result_df['home_fg_attempted'] - result_df['home_fg_made']
        away_missed_fg = result_df['away_fg_attempted'] - result_df['away_fg_made']

        # Calculate possessions using the formula involving OReb%
        home_poss = (result_df['home_fg_attempted'] +
                       0.44 * result_df['home_ft_attempted'] -
                       1.07 * result_df['home_oreb_pct'] * home_missed_fg +
                       result_df['home_turnovers'])
        away_poss = (result_df['away_fg_attempted'] +
                       0.44 * result_df['away_ft_attempted'] -
                       1.07 * result_df['away_oreb_pct'] * away_missed_fg +
                       result_df['away_turnovers'])

        # Handle potential negative/invalid values and fill NaNs using default
        # Ensure possessions are at least 1 to avoid division by zero in Pace/Efficiency
        result_df['home_possessions'] = home_poss.clip(lower=1).fillna(self.defaults['estimated_possessions'])
        result_df['away_possessions'] = away_poss.clip(lower=1).fillna(self.defaults['estimated_possessions'])

        # --- Calculate Pace (Standard Formula - Vectorized) ---
        # Pace = Team Possessions per 48 minutes
        self.log("Calculating pace (assuming 48 min games)...", level="DEBUG")
        minutes_played = 48.0 # Base assumption

        # --- Placeholder for Overtime Adjustment ---
        # if 'num_ot_periods' in result_df.columns: # Check if OT period data exists
        #     # Ensure 'num_ot_periods' is numeric and filled
        #     result_df['num_ot_periods'] = pd.to_numeric(result_df['num_ot_periods'], errors='coerce').fillna(0)
        #     minutes_played = 48.0 + result_df['num_ot_periods'] * 5.0
        # elif 'home_ot' in result_df.columns and 'away_ot' in result_df.columns: # Alternative check using OT scores
        #     # Assuming OT score columns exist and are numeric/filled
        #     # This logic depends on how OT is represented (e.g., total OT score, score per OT period)
        #     # Simple check: if either OT score > 0, assume at least 1 OT period. Refine if more detail available.
        #     is_ot = (result_df['home_ot'] > 0) | (result_df['away_ot'] > 0)
        #     # This is a rough estimate, actual OT periods might vary. Needs better data schema ideally.
        #     minutes_played = np.where(is_ot, 48.0 + 5.0, 48.0) # Adds 5 mins if any OT score > 0

        result_df['home_pace'] = (result_df['home_possessions'] * 48.0 / minutes_played).fillna(self.defaults['pace'])
        result_df['away_pace'] = (result_df['away_possessions'] * 48.0 / minutes_played).fillna(self.defaults['pace'])
        # Game pace can be average of team paces, or calculated from total possessions/minutes
        # total_poss = result_df['home_possessions'] + result_df['away_possessions']
        # result_df['game_pace'] = (total_poss * 48.0 / (minutes_played * 2)).fillna(self.defaults['pace']) # Avg pace this way
        result_df['game_pace'] = (result_df['home_pace'] + result_df['away_pace']) / 2 # Simpler average
        result_df['pace_differential'] = result_df['home_pace'] - result_df['away_pace']


        # --- Calculate Offensive & Defensive Efficiency (Ratings - Vectorized) ---
        # Efficiency = Points per 100 Possessions
        self.log("Calculating offensive/defensive efficiency (ratings)...", level="DEBUG")
        result_df['home_offensive_rating'] = (result_df['home_score'] * 100 / result_df['home_possessions'].replace(0, np.nan)).fillna(self.defaults['offensive_rating'])
        result_df['away_offensive_rating'] = (result_df['away_score'] * 100 / result_df['away_possessions'].replace(0, np.nan)).fillna(self.defaults['offensive_rating'])

        # Defensive rating = Opponent's Offensive Rating
        result_df['home_defensive_rating'] = result_df['away_offensive_rating']
        result_df['away_defensive_rating'] = result_df['home_offensive_rating']

        # Net Rating (Offensive Rating - Defensive Rating)
        result_df['home_net_rating'] = result_df['home_offensive_rating'] - result_df['home_defensive_rating']
        result_df['away_net_rating'] = result_df['away_offensive_rating'] - result_df['away_defensive_rating']
        result_df['efficiency_differential'] = result_df['home_net_rating'] - result_df['away_net_rating'] # Diff in net ratings


        # --- Calculate Total Rebound Percentage (TRB%) ---
        # TRB% = Team Total Reb / (Team Total Reb + Opponent Total Reb)
        self.log("Calculating TRB%...", level="DEBUG")
        total_rebounds_game = (result_df['home_total_reb'] + result_df['away_total_reb']).replace(0, np.nan) # Avoid division by zero if somehow 0 total rebs
        result_df['home_trb_pct'] = (result_df['home_total_reb'] / total_rebounds_game).fillna(self.defaults['trb_pct'])
        result_df['away_trb_pct'] = (result_df['away_total_reb'] / total_rebounds_game).fillna(self.defaults['trb_pct'])
        result_df['trb_pct_diff'] = result_df['home_trb_pct'] - result_df['away_trb_pct']


        # --- Calculate Turnover Rate (TOV%) ---
        # TOV% = Turnovers / Possessions (per 100 possessions is also common)
        # Let's calculate as TOV per 100 Possessions for consistency with ratings
        self.log("Calculating TOV% (per 100 Poss)...", level="DEBUG")
        result_df['home_tov_rate'] = (result_df['home_turnovers'] * 100 / result_df['home_possessions'].replace(0, np.nan)).fillna(self.defaults['tov_rate'] * 100) # Adjust default if needed
        result_df['away_tov_rate'] = (result_df['away_turnovers'] * 100 / result_df['away_possessions'].replace(0, np.nan)).fillna(self.defaults['tov_rate'] * 100)
        # Lower TOV rate is better. Diff (away - home): positive favors home.
        result_df['tov_rate_diff'] = result_df['away_tov_rate'] - result_df['home_tov_rate']


        # --- Basic Score Convenience Features ---
        if 'home_score' in result_df.columns and 'away_score' in result_df.columns:
            if 'total_score' not in result_df.columns:
                result_df['total_score'] = result_df['home_score'] + result_df['away_score']
            if 'point_diff' not in result_df.columns:
                result_df['point_diff'] = result_df['home_score'] - result_df['away_score']
        else:
            # Ensure columns exist even if scores were missing (filled with 0 earlier)
            if 'total_score' not in result_df.columns: result_df['total_score'] = result_df['home_score'] + result_df['away_score']
            if 'point_diff' not in result_df.columns: result_df['point_diff'] = result_df['home_score'] - result_df['away_score']


        self.log("Finished integrating advanced features.", level="DEBUG")
        return result_df

    def _get_optimized_feature_sets(self) -> Dict[int, List[str]]:
        """
        Defines static feature sets potentially useful for quarter-specific models.
        NOTE: These are examples and not used by the main `generate_all_features` which
              focuses on pre-game features. Generating live/intra-game features
              would require a different setup processing live data feeds.
        """
        # (Implementation unchanged, comment added for clarity)
        return {
            1: [ # Pre-game / Q1 Features (Examples)
                'home_rolling_score_10', 'away_rolling_score_10', # Using specific window
                'home_rolling_opp_score_10', 'away_rolling_opp_score_10',
                'rolling_margin_diff', # Added rolling margin diff
                'prev_matchup_diff', 'rest_days_home', 'rest_days_away', 'rest_advantage',
                'is_back_to_back_home', 'is_back_to_back_away',
                'home_season_win_pct', 'away_season_win_pct', 'season_win_pct_diff', # Season context
                'season_net_pts_diff', # Added season net points diff
                # Use rolling averages of advanced stats if calculated, otherwise season averages or defaults
                # Example: 'home_rolling_eff_diff', 'away_rolling_eff_diff' (if calculated in rolling step)
                'home_offensive_rating', 'away_offensive_rating', # Historical/Season Averages
                'home_defensive_rating', 'away_defensive_rating',
                'pace_differential', 'efficiency_differential',
                'trb_pct_diff', 'tov_rate_diff' # Added new diffs
            ],
             # Q2, Q3, Q4 sets would likely include features derived from previous quarters' performance
             2: [], 3: [], 4: [] # Placeholder for future quarter-specific feature definition
        }

    def normalize_team_name(self, team_name):
        """ Normalizes team name for consistent lookup. Add more mappings as needed. """
        # (Using the improved version from user)
        if not isinstance(team_name, str):
            return ""
        team_lower = team_name.lower().strip()
        # Comprehensive mapping
        mapping = {
            "76ers": "76ers", "philadelphia": "76ers", "phila": "76ers", "sixers": "76ers",
            "bucks": "bucks", "milwaukee": "bucks",
            "bulls": "bulls", "chicago": "bulls",
            "cavaliers": "cavaliers", "cleveland": "cavaliers", "cle": "cavaliers",
            "celtics": "celtics", "boston": "celtics",
            "clippers": "clippers", "la clippers": "clippers", "los angeles clippers": "clippers",
            "grizzlies": "grizzlies", "memphis": "grizzlies",
            "hawks": "hawks", "atlanta": "hawks",
            "heat": "heat", "miami": "heat",
            "hornets": "hornets", "charlotte": "hornets",
            "jazz": "jazz", "utah": "jazz",
            "kings": "kings", "sacramento": "kings",
            "knicks": "knicks", "new york": "knicks", "ny knicks": "knicks",
            "lakers": "lakers", "la lakers": "lakers", "los angeles lakers": "lakers",
            "magic": "magic", "orlando": "magic",
            "mavericks": "mavericks", "dallas": "mavericks", "mavs": "mavericks",
            "nets": "nets", "brooklyn": "nets", "bkn": "nets",
            "nuggets": "nuggets", "denver": "nuggets",
            "pacers": "pacers", "indiana": "pacers",
            "pelicans": "pelicans", "new orleans": "pelicans", "no pelicans": "pelicans",
            "pistons": "pistons", "detroit": "pistons",
            "raptors": "raptors", "toronto": "raptors",
            "rockets": "rockets", "houston": "rockets",
            "spurs": "spurs", "san antonio": "spurs",
            "suns": "suns", "phoenix": "suns", "phx": "suns",
            "thunder": "thunder", "oklahoma city": "thunder", "okc": "thunder",
            "timberwolves": "timberwolves", "minnesota": "timberwolves", "twolves": "timberwolves",
            "blazers": "blazers", "portland": "blazers", "trail blazers": "blazers",
            "warriors": "warriors", "golden state": "warriors", "gs warriors": "warriors", "gsw": "warriors",
            "wizards": "wizards", "washington": "wizards",
        }
        # Direct match first (most common case)
        if team_lower in mapping:
            return mapping[team_lower]

        # Check parts for city/mascot match (be careful with short names)
        for key, value in mapping.items():
             # Check if the key is a significant part of the input name
             if len(key) > 3 and (f" {key} " in f" {team_lower} " or team_lower.startswith(key + " ") or team_lower.endswith(" " + key)):
                 return value

        # Fallback: basic cleanup (remove non-alphanumeric)
        cleaned = ''.join(filter(str.isalnum, team_lower))
        # If cleaned version is a standard name (value in mapping), return it
        if cleaned in mapping.values(): return cleaned
        # Otherwise return the cleaned version or original if cleaning is empty
        self.log(f"Team name '{team_name}' normalized to '{cleaned if cleaned else team_lower}' (potential fallback).", "DEBUG")
        return cleaned if cleaned else team_lower


    # @profile_time(debug_mode=True) # Optional profiling
    def add_rolling_features(self, df, window_sizes=[10]):
        """
        [Refactored] Adds rolling average features for scores, margins, AND momentum
        based on past games, calculated from a team-centric view.

        Requires df to be sorted by 'game_date' and have necessary columns, including
        outputs from add_intra_game_momentum if rolling momentum is desired.

        Args:
            df: DataFrame sorted by 'game_date', containing game data & intra-game momentum scores.
            window_sizes: List of window sizes for rolling averages (e.g., [5, 10]).

        Returns:
            DataFrame with added rolling features (scores, margins, momentum).
        """
        self.log(f"Adding rolling features for windows: {window_sizes} (Scores & Momentum)...", level="DEBUG")
        if df is None or df.empty:
            self.log("Input DataFrame is empty for rolling features.", "WARNING")
            return df

        result_df = df.copy() # Work on a copy

        # --- Ensure requirements ---
        req_score_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score']
        # Add required momentum cols (outputs from previous step)
        req_momentum_cols = ['momentum_score_ewma_q4', 'end_q3_diff', 'q3_margin_change', 'q4_margin_change'] # Add others if needed
        req_cols = req_score_cols + req_momentum_cols

        missing_cols = [col for col in req_cols if col not in result_df.columns]
        if missing_cols:
            # Only raise error for core score columns, warn and skip momentum for others
            core_missing = [c for c in missing_cols if c in req_score_cols]
            momentum_missing = [c for c in missing_cols if c in req_momentum_cols]
            if core_missing:
                self.log(f"Missing required score columns for rolling features: {core_missing}. Cannot proceed accurately.", "ERROR")
                return result_df # Return without adding features if core cols missing
            if momentum_missing:
                self.log(f"Missing intra-game momentum columns for rolling: {momentum_missing}. Skipping rolling momentum.", "WARNING")
                # Remove missing momentum cols from the list to be processed
                req_momentum_cols = [c for c in req_momentum_cols if c not in momentum_missing]


        # Convert score columns to numeric, fill NaNs with default
        result_df['home_score'] = pd.to_numeric(result_df['home_score'], errors='coerce').fillna(self.defaults['avg_pts_for'])
        result_df['away_score'] = pd.to_numeric(result_df['away_score'], errors='coerce').fillna(self.defaults['avg_pts_against'])

        # Convert momentum columns to numeric, fill NaNs with 0 (neutral momentum)
        for col in req_momentum_cols:
             if col in result_df.columns:
                 result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # --- Vectorized Calculation using Team-Centric View ---
        self.log("Creating team-centric view for rolling calculations...", level="DEBUG")
        # Base columns needed for merge keys and sorting
        base_view_cols = ['game_id', 'game_date', 'team', 'location']
        # Score columns for rolling scores
        score_view_cols = ['score_for', 'score_against']
        # Momentum columns available for rolling
        momentum_view_cols = req_momentum_cols # Use the list of valid momentum columns

        # Create home and away views with consistent column names
        home_view = result_df[['game_id', 'game_date', 'home_team', 'home_score', 'away_score'] + req_momentum_cols].rename(
            columns={'home_team': 'team', 'home_score': 'score_for', 'away_score': 'score_against'}
        )
        home_view['location'] = 'home'

        away_view = result_df[['game_id', 'game_date', 'away_team', 'home_score', 'away_score'] + req_momentum_cols].rename(
            columns={'away_team': 'team', 'away_score': 'score_for', 'home_score': 'score_against'}
        )
        away_view['location'] = 'away'

        # Combine and sort by team, then date (crucial for shift+rolling)
        team_view = pd.concat([home_view, away_view], ignore_index=True)
        team_view = team_view.sort_values(['team', 'game_date', 'game_id']) # Use game_id as tiebreaker

        # --- Calculate Rolling Averages ---
        for window in window_sizes:
            self.log(f"Calculating rolling window: {window}", level="DEBUG")
            min_p = max(1, window // 2) # Require at least half the window periods

            # Columns to perform rolling calculations on
            cols_to_roll = score_view_cols + momentum_view_cols

            for col in cols_to_roll:
                 # Calculate rolling average (shifted to exclude current game)
                 rolling_col_name = f'rolling_{col}_{window}'
                 team_view[rolling_col_name] = team_view.groupby('team')[col].transform(
                      lambda x: x.shift(1).rolling(window=window, min_periods=min_p).mean()
                 )
                 # Fill NaNs resulting from rolling (e.g., early games) with appropriate defaults
                 default_val = 0 # Default for momentum, changes, diffs
                 if col == 'score_for': default_val = self.defaults['avg_pts_for']
                 if col == 'score_against': default_val = self.defaults['avg_pts_against']
                 team_view[rolling_col_name] = team_view[rolling_col_name].fillna(default_val)


        # --- Merge these general rolling stats back into result_df ---
        self.log("Merging team-centric rolling stats back to game view...", level="DEBUG")
        team_view['merge_key_rolling'] = team_view['game_id'].astype(str) + "_" + team_view['team']
        result_df['merge_key_home'] = result_df['game_id'].astype(str) + "_" + result_df['home_team']
        result_df['merge_key_away'] = result_df['game_id'].astype(str) + "_" + result_df['away_team']

        # Define all columns created by rolling
        rolling_cols_generated = [f'rolling_{col}_{w}' for col in cols_to_roll for w in window_sizes]
        cols_to_merge = ['merge_key_rolling'] + rolling_cols_generated
        merge_data = team_view[cols_to_merge].drop_duplicates(subset=['merge_key_rolling'], keep='last')

        # Function to generate rename dictionary
        def get_rename_dict(prefix, columns, windows):
            rd = {}
            for col_base in columns: # e.g., 'score_for', 'momentum_score_ewma_q4'
                for w in windows:
                     rd[f'rolling_{col_base}_{w}'] = f'{prefix}_rolling_{col_base}_{w}' # e.g., home_rolling_score_for_10
            return rd

        # Merge for home team
        rename_dict_home = get_rename_dict('home', cols_to_roll, window_sizes)
        result_df = pd.merge(result_df, merge_data, how='left', left_on='merge_key_home', right_on='merge_key_rolling')
        result_df = result_df.rename(columns=rename_dict_home)
        result_df = result_df.drop(columns=['merge_key_rolling'], errors='ignore')

        # Merge for away team
        rename_dict_away = get_rename_dict('away', cols_to_roll, window_sizes)
        result_df = pd.merge(result_df, merge_data, how='left', left_on='merge_key_away', right_on='merge_key_rolling')
        result_df = result_df.rename(columns=rename_dict_away)
        result_df = result_df.drop(columns=['merge_key_rolling'], errors='ignore')

        # Drop temporary merge keys from result_df
        result_df = result_df.drop(columns=['merge_key_home', 'merge_key_away'], errors='ignore')

        # Final check for NaNs in added rolling columns (robustness)
        final_rolling_cols = list(rename_dict_home.values()) + list(rename_dict_away.values())
        for col in final_rolling_cols:
             if col in result_df.columns:
                  # Determine default based on base column name
                  default_val = 0 # Default for momentum
                  if 'score_for' in col: default_val = self.defaults['avg_pts_for']
                  if 'score_against' in col: default_val = self.defaults['avg_pts_against']
                  result_df[col] = result_df[col].fillna(default_val)

        # --- Create derived rolling features (margins, diffs) for the primary window ---
        primary_window = window_sizes[-1] # Use the last window as primary

        # Rolling Scores & Margins
        home_roll_score_col = f'home_rolling_score_for_{primary_window}'
        away_roll_score_col = f'away_rolling_score_for_{primary_window}'
        home_roll_opp_col = f'home_rolling_score_against_{primary_window}'
        away_roll_opp_col = f'away_rolling_score_against_{primary_window}'

        if all(c in result_df.columns for c in [home_roll_score_col, away_roll_score_col, home_roll_opp_col, away_roll_opp_col]):
             result_df['rolling_home_score'] = result_df[home_roll_score_col] # Standardized name
             result_df['rolling_away_score'] = result_df[away_roll_score_col] # Standardized name
             result_df['rolling_home_opp_score'] = result_df[home_roll_opp_col] # Standardized name
             result_df['rolling_away_opp_score'] = result_df[away_roll_opp_col] # Standardized name
             result_df['rolling_home_margin'] = result_df['rolling_home_score'] - result_df['rolling_home_opp_score']
             result_df['rolling_away_margin'] = result_df['rolling_away_score'] - result_df['rolling_away_opp_score']
             result_df['rolling_margin_diff'] = result_df['rolling_home_margin'] - result_df['rolling_away_margin']

        # Rolling Momentum Difference (Example: using momentum_score_ewma_q4)
        home_roll_mom_col = f'home_rolling_momentum_score_ewma_q4_{primary_window}'
        away_roll_mom_col = f'away_rolling_momentum_score_ewma_q4_{primary_window}'
        if home_roll_mom_col in result_df.columns and away_roll_mom_col in result_df.columns:
             result_df['rolling_momentum_ewma_diff'] = result_df[home_roll_mom_col] - result_df[away_roll_mom_col]
             # Add standardized names if desired
             result_df['rolling_home_momentum'] = result_df[home_roll_mom_col]
             result_df['rolling_away_momentum'] = result_df[away_roll_mom_col]


        self.log("Finished adding rolling features (Scores & Momentum).", level="DEBUG")
        return result_df

    # @profile_time(debug_mode=True) # Optional profiling
    def add_team_history_features_vectorized(self, df, rolling_window=10):
        """
        [Vectorized] Adds previous matchup score differential. Rolling scores are assumed
        to be calculated by add_rolling_features already.

        Requires df sorted by 'game_date' and necessary columns.

        Args:
            df: DataFrame with game data, sorted by date.
            rolling_window: The window size used for rolling scores (used by dependent features).

        Returns:
            DataFrame with added 'prev_matchup_diff'.
        """
        if df is None or df.empty:
            self.log("No data provided for team history features", "WARNING")
            return df

        self.log("Adding team history features (Vectorized - Prev Matchup Diff)...", level="DEBUG")
        result_df = df.copy()

        # --- Check for required columns ---
        required_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            self.log(f"Missing required columns for team history (vectorized): {missing_cols}. Cannot calculate prev_matchup_diff.", "ERROR")
            result_df['prev_matchup_diff'] = 0 # Add placeholder
            return result_df

        # Ensure scores are numeric for diff calculation, keep NaNs for now
        result_df['home_score'] = pd.to_numeric(result_df['home_score'], errors='coerce')
        result_df['away_score'] = pd.to_numeric(result_df['away_score'], errors='coerce')


        # --- Calculate Previous Matchup Differential (Vectorized) ---
        self.log("Calculating previous matchup differential...", level="DEBUG")

        # Create a unique, order-invariant matchup key
        result_df['matchup_key'] = result_df.apply(
            lambda row: "_vs_".join(sorted([str(row['home_team']), str(row['away_team'])])),
            axis=1
        )

        # Sort by matchup key and date to prepare for shift (Crucial!)
        result_df = result_df.sort_values(['matchup_key', 'game_date', 'game_id']) # Stable sort

        # Get previous game's details within the same matchup group using shift(1)
        cols_to_shift = ['home_team', 'away_team', 'home_score', 'away_score']
        shifted_cols = result_df.groupby('matchup_key', observed=True)[cols_to_shift].shift(1) # Use observed=True if matchup_key is categorical
        shifted_cols.columns = [f'prev_{col}' for col in cols_to_shift] # Rename

        # Join shifted data back
        result_df = result_df.join(shifted_cols)

        # Calculate diff using np.select
        conditions = [
            result_df['prev_home_team'].isna(), # Condition 1: No previous matchup
            result_df['home_team'] == result_df['prev_home_team'] # Condition 2: Current home was prev home
        ]
        # Calculate potential differences, handling NaNs in previous scores
        prev_diff_home_win = result_df['prev_home_score'] - result_df['prev_away_score']
        prev_diff_away_win = result_df['prev_away_score'] - result_df['prev_home_score']

        choices = [
            0,                  # Choice 1: Diff is 0 if no prior matchup
            prev_diff_home_win, # Choice 2: Use previous H-A diff
            prev_diff_away_win  # Choice 3 (Default): Use previous A-H diff
        ]
        result_df['prev_matchup_diff'] = np.select(conditions, choices, default=0)

        # Fill NaNs in the final diff column (if prev scores were NaN) with 0
        result_df['prev_matchup_diff'] = result_df['prev_matchup_diff'].fillna(0).astype(float)


        # --- Final Cleanup ---
        result_df = result_df.drop(columns=['matchup_key', 'prev_home_team', 'prev_away_team',
                                             'prev_home_score', 'prev_away_score'], errors='ignore')

        self.log("Finished adding team history features (Vectorized - Prev Matchup Diff).")
        return result_df

    # @profile_time(debug_mode=True) # Optional profiling
    def add_rest_features_vectorized(self, df):
        """
        [Vectorized] Adds rest days, back-to-back status, and rest advantage.

        Requires df sorted by 'game_date'. Needs 'game_id', 'game_date', 'home_team', 'away_team'.

        Args:
            df: DataFrame containing game data, sorted by 'game_date'.

        Returns:
            DataFrame with added rest features.
        """
        self.log("Adding rest features (Vectorized)...", level="DEBUG")
        if df is None or df.empty:
            self.log("Input DataFrame is empty for rest features.", "WARNING")
            return df

        result_df = df.copy() # Work on a copy

        # --- Ensure requirements ---
        req_cols = ['game_id', 'game_date', 'home_team', 'away_team']
        missing_cols = [col for col in req_cols if col not in result_df.columns]
        if missing_cols:
            self.log(f"Missing required columns for rest features: {missing_cols}. Cannot proceed.", "ERROR")
            # Add placeholder columns with default values
            default_rest = 3
            result_df['rest_days_home'] = default_rest; result_df['rest_days_away'] = default_rest
            result_df['is_back_to_back_home'] = 0; result_df['is_back_to_back_away'] = 0
            result_df['rest_advantage'] = 0
            return result_df

        # Assume df is already sorted by game_date and date column is datetime

        # --- Vectorized Calculation using Team-Centric View ---
        self.log("Melting data for team-centric rest calculation...", level="DEBUG")
        home_melt = result_df[['game_id', 'game_date', 'home_team']].rename(columns={'home_team': 'team'})
        away_melt = result_df[['game_id', 'game_date', 'away_team']].rename(columns={'away_team': 'team'})
        team_dates = pd.concat([home_melt, away_melt], ignore_index=True)
        team_dates = team_dates.sort_values(['team', 'game_date', 'game_id']) # Stable sort

        self.log("Calculating last game date per team using shift...", level="DEBUG")
        team_dates['last_game_date'] = team_dates.groupby('team')['game_date'].shift(1)

        # Merge last_game_date back into the original result_df structure
        self.log("Merging last game dates back...", level="DEBUG")
        team_dates['merge_key_rest'] = team_dates['game_id'].astype(str) + "_" + team_dates['team']
        result_df['merge_key_home_rest'] = result_df['game_id'].astype(str) + "_" + result_df['home_team']
        result_df['merge_key_away_rest'] = result_df['game_id'].astype(str) + "_" + result_df['away_team']

        merge_data_rest = team_dates[['merge_key_rest', 'last_game_date']].drop_duplicates(subset=['merge_key_rest'], keep='last') # Keep last

        result_df = pd.merge(result_df, merge_data_rest, how='left', left_on='merge_key_home_rest', right_on='merge_key_rest')
        result_df = result_df.rename(columns={'last_game_date': 'home_last_game_date'}).drop(columns=['merge_key_rest'])

        result_df = pd.merge(result_df, merge_data_rest, how='left', left_on='merge_key_away_rest', right_on='merge_key_rest')
        result_df = result_df.rename(columns={'last_game_date': 'away_last_game_date'}).drop(columns=['merge_key_rest'])

        result_df = result_df.drop(columns=['merge_key_home_rest', 'merge_key_away_rest'], errors='ignore')

        # --- Calculate Rest Days ---
        self.log("Calculating rest days and B2B status...", level="DEBUG")
        default_rest_days = 3
        max_rest_cap = 7

        # Ensure dates are datetime objects before subtraction
        result_df['game_date'] = pd.to_datetime(result_df['game_date'])
        result_df['home_last_game_date'] = pd.to_datetime(result_df['home_last_game_date'])
        result_df['away_last_game_date'] = pd.to_datetime(result_df['away_last_game_date'])

        result_df['rest_days_home'] = (result_df['game_date'] - result_df['home_last_game_date']).dt.days
        result_df['rest_days_away'] = (result_df['game_date'] - result_df['away_last_game_date']).dt.days

        # Fill NaNs (first game) and apply cap. Ensure >= 0 days.
        result_df['rest_days_home'] = result_df['rest_days_home'].fillna(default_rest_days).clip(lower=0, upper=max_rest_cap)
        result_df['rest_days_away'] = result_df['rest_days_away'].fillna(default_rest_days).clip(lower=0, upper=max_rest_cap)

        # --- Calculate Back-to-Back (B2B) Status ---
        result_df['is_back_to_back_home'] = (result_df['rest_days_home'] == 1).astype(int)
        result_df['is_back_to_back_away'] = (result_df['rest_days_away'] == 1).astype(int)

        # --- Calculate Rest Advantage ---
        result_df['rest_advantage'] = result_df['rest_days_home'] - result_df['rest_days_away']

        # --- Final Cleanup ---
        result_df = result_df.drop(columns=['home_last_game_date', 'away_last_game_date'], errors='ignore')

        # Ensure numeric types
        rest_columns = ['rest_days_home', 'rest_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage']
        for col in rest_columns:
            if col in result_df.columns:
                 dtype = int if 'is_back_to_back' in col else float # Use float for days/advantage
                 result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(dtype)

        self.log("Finished adding rest features (Vectorized).")
        return result_df


# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    print("NBAFeatureEngine Script - Example Usage")

    # --- Dummy Historical Game Stats ---
    historical_data = {
        'game_id': [1, 2, 3, 4, 5, 6, 9, 10], # Added more games for rolling history
        'game_date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-03', '2024-01-04', '2024-01-02', '2024-01-04']),
        'home_team': ['Lakers', 'Clippers', 'Lakers', 'Warriors', 'Knicks', 'Lakers', 'Warriors', 'Knicks'],
        'away_team': ['Warriors', 'Knicks', 'Clippers', 'Clippers', 'Warriors', 'Knicks', 'Knicks', 'Clippers'],
        'home_score': [110, 105, 115, 120, 100, 108, 112, 95 ],
        'away_score': [100, 102, 111, 118, 98, 105, 101, 99 ],
        'home_fg_made': [40, 38, 42, 45, 35, 39, 41, 33 ], 'home_fg_attempted': [85, 82, 88, 90, 80, 84, 86, 78 ],
        'away_fg_made': [36, 37, 40, 44, 34, 38, 37, 35 ], 'away_fg_attempted': [80, 81, 85, 88, 78, 83, 82, 80 ],
        'home_3pm': [10, 9, 12, 14, 8, 11, 13, 7 ], 'home_3pa': [30, 28, 35, 38, 25, 33, 36, 24 ],
        'away_3pm': [8, 10, 11, 13, 7, 9, 9, 10 ], 'away_3pa': [25, 32, 33, 36, 24, 30, 28, 31 ],
        'home_ft_made': [20, 20, 19, 16, 22, 19, 17, 22 ], 'home_ft_attempted': [25, 24, 22, 20, 26, 23, 21, 26 ],
        'away_ft_made': [20, 18, 20, 17, 23, 20, 18, 19 ], 'away_ft_attempted': [24, 22, 23, 21, 27, 24, 22, 23 ],
        'home_off_reb': [10, 9, 11, 12, 8, 10, 11, 7 ], 'home_def_reb': [30, 28, 32, 35, 27, 31, 33, 25 ],
        'away_off_reb': [8, 9, 10, 11, 7, 9, 8, 9 ], 'away_def_reb': [28, 27, 30, 33, 26, 29, 29, 28 ],
        'home_turnovers': [12, 14, 11, 10, 15, 13, 10, 16 ], 'away_turnovers': [13, 15, 12, 11, 16, 14, 12, 13 ],
        'home_assists': [25, 22, 28, 30, 20, 26, 27, 19 ], 'away_assists': [20, 21, 24, 28, 18, 23, 22, 20 ],
        # Calculate total rebounds from off/def if not present
        'home_total_reb': [40, 37, 43, 47, 35, 41, 44, 32 ], 'away_total_reb': [36, 36, 40, 44, 33, 38, 37, 37 ],
    }
    historical_df = pd.DataFrame(historical_data)
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date']) # Ensure datetime

    # --- Dummy Season Team Stats ---
    team_stats_data = {
        'team_name': ['Lakers', 'Clippers', 'Warriors', 'Knicks', 'LA Lakers'], # Added variation for normalization test
        'season': ['2023-24'] * 5, # Assuming all games are in this season
        'wins_all_percentage': [0.550, 0.600, 0.650, 0.500, 0.550], # Dupe Lakers stats for LA Lakers
        'points_for_avg_all': [115.5, 116.0, 118.0, 112.0, 115.5],
        'points_against_avg_all': [114.0, 112.5, 115.0, 113.0, 114.0],
    }
    team_stats_df = pd.DataFrame(team_stats_data)

    # --- Data for Games to Predict Features For ---
    predict_data = {
        'game_id': [7, 8, 11], # Added game 11
        'game_date': pd.to_datetime(['2024-01-05', '2024-01-05', '2024-01-06']),
        'home_team': ['Warriors', 'Clippers', 'Lakers'], # Game 11 Lakers Home
        'away_team': ['Lakers', 'Knicks', 'Warriors'], # Game 11 Warriors Away
         # Scores and box stats are NaN for future games
         'home_score': [np.nan, np.nan, np.nan], 'away_score': [np.nan, np.nan, np.nan],
         # Include all columns expected by 'integrate_advanced_features', even if NaN
          'home_fg_made': [np.nan, np.nan, np.nan], 'home_fg_attempted': [np.nan, np.nan, np.nan],
          'away_fg_made': [np.nan, np.nan, np.nan], 'away_fg_attempted': [np.nan, np.nan, np.nan],
          'home_3pm': [np.nan, np.nan, np.nan], 'home_3pa': [np.nan, np.nan, np.nan],
          'away_3pm': [np.nan, np.nan, np.nan], 'away_3pa': [np.nan, np.nan, np.nan],
          'home_ft_made': [np.nan, np.nan, np.nan], 'home_ft_attempted': [np.nan, np.nan, np.nan],
          'away_ft_made': [np.nan, np.nan, np.nan], 'away_ft_attempted': [np.nan, np.nan, np.nan],
          'home_off_reb': [np.nan, np.nan, np.nan], 'home_def_reb': [np.nan, np.nan, np.nan], 'home_total_reb': [np.nan, np.nan, np.nan],
          'away_off_reb': [np.nan, np.nan, np.nan], 'away_def_reb': [np.nan, np.nan, np.nan], 'away_total_reb': [np.nan, np.nan, np.nan],
          'home_turnovers': [np.nan, np.nan, np.nan], 'away_turnovers': [np.nan, np.nan, np.nan],
          'home_assists': [np.nan, np.nan, np.nan], 'away_assists': [np.nan, np.nan, np.nan]
    }
    predict_df = pd.DataFrame(predict_data)
    predict_df['game_date'] = pd.to_datetime(predict_df['game_date']) # Ensure datetime

    # --- Initialize and Run Engine ---
    feature_engine = NBAFeatureEngine(debug=True) # Enable debug logging
    rolling_win = 3 # Use smaller window for small dummy dataset

    features_result_df = feature_engine.generate_all_features(
        df=predict_df,
        historical_games_df=historical_df,
        team_stats_df=team_stats_df, # Pass season stats
        rolling_window=rolling_win
    )

    print("\n--- Feature Generation Results ---")
    # Display more columns or use options for better visibility
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120) # Adjust width as needed
    print(features_result_df)

    print("\n--- Generated Columns (Sorted) ---")
    print(sorted(features_result_df.columns.tolist())) # Sort for easier reading

    # --- Display Selected Key Features ---
    cols_to_show = [
        'game_id', 'game_date', 'home_team', 'away_team',
        # Rolling (using primary names if generated)
        'rolling_home_score', 'rolling_away_score', 'rolling_margin_diff',
        # History
        'prev_matchup_diff',
        # Rest
        'rest_days_home', 'rest_days_away', 'is_back_to_back_home', 'is_back_to_back_away', 'rest_advantage',
        # Advanced (pace/eff use defaults for future games, others NaN or 0)
        'home_pace', 'game_pace', 'home_offensive_rating', 'home_defensive_rating', 'home_net_rating',
        'efficiency_differential', 'home_efg_pct', 'away_efg_pct', 'home_oreb_pct', 'away_oreb_pct', 'home_trb_pct', 'away_trb_pct', 'home_tov_rate', 'away_tov_rate',
        'trb_pct_diff', 'tov_rate_diff', # Advanced Diffs
        # Season Context
        'home_season_win_pct', 'away_season_win_pct', 'season_win_pct_diff',
        'home_season_avg_pts_for', 'away_season_avg_pts_for', 'season_pts_for_diff',
        'season_net_pts_diff' # Added Season Net Pts Diff
    ]
    # Filter cols_to_show to only those present in the final DataFrame
    cols_to_show_present = [col for col in cols_to_show if col in features_result_df.columns]

    print("\n--- Selected Features ---")
    # Select rows corresponding to the original predict_df game_ids for cleaner output
    predict_game_ids = predict_df['game_id'].unique()
    print(features_result_df[features_result_df['game_id'].isin(predict_game_ids)][cols_to_show_present])

        # @NBAFeatureEngine.profile_time(debug_mode=True) # Optional profiling
    def add_intra_game_momentum(self, df):
        """
        [Vectorized] Adds intra-game momentum features based on quarter scores.

        Calculates quarter margins, changes in margins, cumulative differences,
        and an EWMA-based momentum score based on quarter margin trends within
        each completed game.

        Requires df to have quarter score columns (home_q1, away_q1, etc.).

        Args:
            df: DataFrame containing game data with quarter scores.

        Returns:
            DataFrame with added intra-game momentum features (e.g., q3_margin_change, end_q3_diff, momentum_score_ewma_q4).
        """
        self.log("Adding intra-game momentum features (Vectorized)...", level="DEBUG")
        if df is None or df.empty:
            self.log("Input DataFrame is empty for momentum features.", "WARNING")
            return df

        result_df = df.copy()

        # --- Ensure requirements & Preprocessing ---
        # Define required quarter score columns
        qtr_cols = [f'{loc}_q{i}' for loc in ['home', 'away'] for i in range(1, 5)]
        # Optional: include OT if you want to factor it into final momentum state
        # ot_cols = ['home_ot', 'away_ot']
        # all_qtr_cols = qtr_cols + ot_cols
        all_qtr_cols = qtr_cols # Sticking to regulation for now

        missing_cols = [col for col in all_qtr_cols if col not in result_df.columns]
        if missing_cols:
            self.log(f"Missing required quarter score columns for momentum: {missing_cols}. Filling with 0.", "WARNING")
            for col in missing_cols:
                 if col not in result_df.columns:
                      result_df[col] = 0 # Fill missing quarter scores with 0

        # Convert quarter columns to numeric, fill NaNs with 0
        for col in all_qtr_cols:
             if col in result_df.columns:
                 result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # --- Calculate Quarter Margins (Home Team Perspective) ---
        self.log("Calculating quarter margins...", level="DEBUG")
        for i in range(1, 5):
            result_df[f'q{i}_margin'] = result_df[f'home_q{i}'] - result_df[f'away_q{i}']

        # --- Calculate Cumulative End-of-Quarter Differentials ---
        self.log("Calculating end-of-quarter differentials...", level="DEBUG")
        result_df['end_q1_diff'] = result_df['q1_margin']
        result_df['end_q2_diff'] = result_df.get('end_q1_diff', 0) + result_df['q2_margin'] # Use .get for safety
        result_df['end_q3_diff'] = result_df.get('end_q2_diff', 0) + result_df['q3_margin']
        # Final regulation diff (useful for checking against point_diff if no OT)
        result_df['end_q4_reg_diff'] = result_df.get('end_q3_diff', 0) + result_df['q4_margin']

        # --- Calculate Quarter-over-Quarter Margin Changes ---
        # How much did the home team's advantage change from one quarter to the next?
        self.log("Calculating quarter margin changes...", level="DEBUG")
        result_df['q2_margin_change'] = result_df['q2_margin'] - result_df.get('q1_margin', 0)
        result_df['q3_margin_change'] = result_df['q3_margin'] - result_df.get('q2_margin', 0)
        result_df['q4_margin_change'] = result_df['q4_margin'] - result_df.get('q3_margin', 0)

        # --- Calculate EWMA Momentum Score ---
        # Apply EWMA row-wise across the quarter margins to get a score reflecting recent trend
        self.log("Calculating EWMA momentum score based on quarter margins...", level="DEBUG")
        quarter_margins = result_df[[f'q{i}_margin' for i in range(1, 5)]]

        # Calculate EWMA ending after Q4 (reflects end-of-game momentum)
        # Span=3 gives reasonable weight to recent quarters in a 4-quarter sequence.
        # adjust=False gives more weight to recent points faster.
        ewma_momentum_q4 = quarter_margins.apply(
            lambda row: row.ewm(span=3, adjust=False).mean().iloc[-1] if row.notna().any() else 0, axis=1
        )
        result_df['momentum_score_ewma_q4'] = ewma_momentum_q4.fillna(0)

        # Optional: Calculate EWMA ending after Q3 (momentum entering Q4)
        ewma_momentum_q3 = quarter_margins[['q1_margin', 'q2_margin', 'q3_margin']].apply(
            lambda row: row.ewm(span=2, adjust=False).mean().iloc[-1] if row.notna().any() else 0, axis=1 # Shorter span
        )
        result_df['momentum_score_ewma_q3'] = ewma_momentum_q3.fillna(0)

        # --- Other Momentum Indicators ---
        # Lead Change Indicator (Did lead sign change significantly?)
        self.log("Calculating lead change indicators...", level="DEBUG")
        q1_sign = np.sign(result_df['end_q1_diff'])
        q2_sign = np.sign(result_df['end_q2_diff'])
        q3_sign = np.sign(result_df['end_q3_diff'])
        q4_sign = np.sign(result_df['end_q4_reg_diff'])

        # Change if signs are opposite (and neither is zero, to be strict)
        result_df['lead_change_q1_to_q2'] = ((q1_sign * q2_sign) < 0).astype(int)
        result_df['lead_change_q2_to_q3'] = ((q2_sign * q3_sign) < 0).astype(int)
        result_df['lead_change_q3_to_q4'] = ((q3_sign * q4_sign) < 0).astype(int)

        # Indicator for large margin swings (e.g., margin change > X points)
        margin_swing_threshold = 10 # Example: 10 point swing in a quarter
        result_df['large_swing_q2'] = (result_df['q2_margin_change'].abs() >= margin_swing_threshold).astype(int)
        result_df['large_swing_q3'] = (result_df['q3_margin_change'].abs() >= margin_swing_threshold).astype(int)
        result_df['large_swing_q4'] = (result_df['q4_margin_change'].abs() >= margin_swing_threshold).astype(int)
        result_df['num_large_swings'] = result_df[['large_swing_q2', 'large_swing_q3', 'large_swing_q4']].sum(axis=1)

        # --- Final Cleanup ---
        # Ensure numeric types for new features
        momentum_cols = [
            'q1_margin', 'q2_margin', 'q3_margin', 'q4_margin',
            'end_q1_diff', 'end_q2_diff', 'end_q3_diff', 'end_q4_reg_diff',
            'q2_margin_change', 'q3_margin_change', 'q4_margin_change',
            'momentum_score_ewma_q4', 'momentum_score_ewma_q3',
            'lead_change_q1_to_q2', 'lead_change_q2_to_q3', 'lead_change_q3_to_q4',
            'large_swing_q2', 'large_swing_q3', 'large_swing_q4', 'num_large_swings'
        ]
        for col in momentum_cols:
             if col in result_df.columns:
                  # Flags are int, others float
                  dtype = int if ('lead_change' in col or 'large_swing' in col or 'num_large' in col) else float
                  result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(dtype)

        self.log("Finished adding intra-game momentum features (Vectorized).")
        return result_df