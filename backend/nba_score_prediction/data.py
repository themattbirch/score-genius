# backend/nba_score_prediction/data.py
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

