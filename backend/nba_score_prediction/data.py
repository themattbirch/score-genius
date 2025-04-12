# backend/nba_score_prediction/data.py

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# -------------------- Data Fetching Class --------------------
class SupabaseDataFetcher:
    """ Handles data fetching from Supabase for NBA prediction models. """

    def __init__(self, supabase_client: Any, debug: bool = False):
        """
        Initialize with a configured Supabase client instance.

        Args:
            supabase_client: The Supabase client instance.
            debug: If True, print debug messages during execution.
        """
        if supabase_client is None:
             raise ValueError("Supabase client must be provided to SupabaseDataFetcher.")
        self.supabase = supabase_client # Use the passed client
        self.debug = debug

    def _print_debug(self, message: str):
        """ Prints debug messages if debug mode is enabled. """
        if self.debug:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{type(self).__name__}] {message}")

    def fetch_historical_games(self, days_lookback: int = 365) -> pd.DataFrame:
        """
        Fetches historical game stats from the 'nba_historical_game_stats' table
        for the specified lookback period using efficient pagination.

        Args:
            days_lookback: Number of days back from today to fetch data for.

        Returns:
            pd.DataFrame: Contains historical game data, sorted by game_date.
                          Columns are explicitly selected and numeric types are cleaned.
                          Returns an empty DataFrame on error or if no data is found.
        """
        threshold_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
        self._print_debug(f"Fetching historical game data since {threshold_date}")

        # Explicitly list all required columns from the nba_historical_game_stats table
        required_columns = [
            'id', 'game_id', 'home_team', 'away_team', 'game_date',
            'home_score', 'away_score', 'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot',
            'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_ot',
            'home_assists', 'home_steals', 'home_blocks', 'home_turnovers', 'home_fouls',
            'away_assists', 'away_steals', 'away_blocks', 'away_turnovers', 'away_fouls',
            'home_off_reb', 'home_def_reb', 'home_total_reb',
            'away_off_reb', 'away_def_reb', 'away_total_reb',
            'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
            'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
            'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted'
        ]
        # Columns that need conversion to numeric (excluding IDs treated as objects/strings)
        numeric_cols = [col for col in required_columns if col not in
                        ['id', 'game_id', 'home_team', 'away_team', 'game_date']]

        all_data = []
        page_size = 1000 
        start_index = 0
        try:
            while True:
                response = self.supabase.table("nba_historical_game_stats") \
                    .select(", ".join(required_columns)) \
                    .gte("game_date", threshold_date) \
                    .order('game_date') \
                    .range(start_index, start_index + page_size - 1) \
                    .execute()

                # Handle potential API errors if response structure is unexpected
                if not hasattr(response, 'data'):
                     raise ValueError(f"Supabase response missing 'data' attribute. Response: {response}")

                batch_data = response.data
                batch_size = len(batch_data)
                all_data.extend(batch_data)
                self._print_debug(f"Retrieved batch of {batch_size} records, total: {len(all_data)}")

                if batch_size < page_size:
                    break 
                start_index += page_size

            if not all_data:
                self._print_debug(f"No historical game data found since {threshold_date}.")
                return pd.DataFrame() 

            df = pd.DataFrame(all_data)

            # --- Data Cleaning ---
            # Convert date column
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            df = df.dropna(subset=['game_date']) # Drop rows where date conversion failed

            # Convert all expected numeric columns, coercing errors and filling NaNs with 0
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    self._print_debug(f"Warning: Expected numeric column '{col}' missing. Adding with zeros.")
                    df[col] = 0

            return df.sort_values('game_date').reset_index(drop=True)

        except Exception as e:
            self._print_debug(f"Error fetching or processing historical games: {e}")
            traceback.print_exc()
            return pd.DataFrame() 

    def fetch_team_stats(self) -> pd.DataFrame:
        """
        Fetches team performance stats from the 'nba_historical_team_stats' table.

        Returns:
            pd.DataFrame: Contains team season stats.
                          Returns an empty DataFrame on error or if no data is found.
        """
        self._print_debug("Fetching team stats...")

        # Explicitly define expected columns based on schema provided earlier
        required_columns = [
             'id', 'team_id', 'team_name', 'season', 'league_id',
             'games_played_home', 'games_played_away', 'games_played_all',
             'wins_home_total', 'wins_home_percentage', 'wins_away_total', 'wins_away_percentage',
             'wins_all_total', 'wins_all_percentage', 'losses_home_total', 'losses_home_percentage',
             'losses_away_total', 'losses_away_percentage', 'losses_all_total', 'losses_all_percentage',
             'points_for_total_home', 'points_for_total_away', 'points_for_total_all',
             'points_for_avg_home', 'points_for_avg_away', 'points_for_avg_all',
             'points_against_total_home', 'points_against_total_away', 'points_against_total_all',
             'points_against_avg_home', 'points_against_avg_away', 'points_against_avg_all',
             'updated_at', 'current_form'
        ]

        try:
            response = self.supabase.table("nba_historical_team_stats") \
                .select(", ".join(required_columns)) \
                .execute()

            if not hasattr(response, 'data'):
                 raise ValueError(f"Supabase response missing 'data' attribute. Response: {response}")

            data = response.data
            if data:
                self._print_debug(f"Fetched {len(data)} team stat records")
                # Basic conversion - can add more specific numeric conversions if needed
                df = pd.DataFrame(data)
                # Example numeric conversions (adjust list as needed)
                numeric_team_cols = [
                    'wins_all_percentage', 'points_for_avg_all', 'points_against_avg_all'
                    # Add other relevant numeric columns from the list above
                ]
                for col in numeric_team_cols:
                     if col in df.columns:
                         df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                self._print_debug("No team stats found")
                return pd.DataFrame()
        except Exception as e:
            self._print_debug(f"Error fetching team stats: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def fetch_upcoming_games(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetches upcoming games within the specified timeframe from the
        'nba_upcoming_games' table, including related team names.

        Args:
            days_ahead: Number of days into the future to fetch games for.

        Returns:
            pd.DataFrame: Contains upcoming game schedules with 'game_id', 'game_date',
                          'home_team', 'away_team'.
                          Returns an empty DataFrame on error or if no games are found.
        """
        self._print_debug(f"Fetching upcoming games for next {days_ahead} days...")
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

            select_query = "game_id, game_date, home_team:home_team_id(team_name), away_team:away_team_id(team_name)"

            response = self.supabase.table("nba_upcoming_games") \
                .select(select_query) \
                .gte("game_date", today) \
                .lte("game_date", future_date) \
                .order('game_date') \
                .execute()

            if not hasattr(response, 'data'):
                 raise ValueError(f"Supabase response missing 'data' attribute. Response: {response}")

            data = response.data
            if data:
                self._print_debug(f"Fetched {len(data)} upcoming games")
                df = pd.DataFrame(data)

                # --- Flatten nested team data more robustly ---
                # Handle home team
                if 'home_team' in df.columns:
                    df['home_team_name_extracted'] = df['home_team'].apply(
                        lambda x: x.get('team_name', 'Unknown') if isinstance(x, dict) else 'Unknown'
                    )
                else:
                     df['home_team_name_extracted'] = 'Unknown'

                # Handle away team
                if 'away_team' in df.columns:
                     df['away_team_name_extracted'] = df['away_team'].apply(
                         lambda x: x.get('team_name', 'Unknown') if isinstance(x, dict) else 'Unknown'
                     )
                else:
                     df['away_team_name_extracted'] = 'Unknown'

                # Drop original nested columns if they exist
                df = df.drop(columns=['home_team', 'away_team'], errors='ignore')

                # Rename columns to match expected 'home_team', 'away_team' format
                df = df.rename(columns={'home_team_name_extracted': 'home_team',
                                        'away_team_name_extracted': 'away_team'})

                # Select and order final columns
                final_cols = ['game_id', 'game_date', 'home_team', 'away_team']
                df = df[final_cols]

                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                df = df.dropna(subset=['game_date']) # Drop games with invalid dates

                return df.sort_values('game_date').reset_index(drop=True)
            else:
                self._print_debug("No upcoming games found")
                return pd.DataFrame()
        except Exception as e:
            self._print_debug(f"Error fetching upcoming games: {e}")
            traceback.print_exc()
            return pd.DataFrame()
