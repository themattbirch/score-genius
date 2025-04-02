"""
Shared utilities for NBA score prediction model.
Contains common functions used across multiple notebooks.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def convert_time_to_minutes(time_str: str) -> float:
    """
    Convert a time string "MM:SS" to a float (minutes + seconds/60).
    Returns None if conversion fails.
    
    Args:
        time_str: Time string in "MM:SS" format
    Returns:
        float: Time in minutes
    """
    if pd.isna(time_str) or ":" not in str(time_str):
        return None
    try:
        minutes, seconds = str(time_str).split(":")
        return float(minutes) + float(seconds) / 60.0
    except Exception as e:
        print(f"Error converting time: {e}")
        return None

def ensure_numeric_features(df, feature_columns, flag_critical=True):
    """
    Ensures all feature columns are numeric, replacing NaN/None values with appropriate defaults.
    Also flags critical missing values for review.
    
    Args:
        df (DataFrame): Input DataFrame
        feature_columns (list): List of column names to process
        flag_critical (bool): Whether to flag critical missing values
    Returns:
        DataFrame: DataFrame with ensured numeric features
        dict: Dictionary of flags for critical missing values (if flag_critical=True)
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default values based on column type
    default_map = {
        'home_q1': 0, 'home_q2': 0, 'home_q3': 0, 'home_q4': 0,
        'away_q1': 0, 'away_q2': 0, 'away_q3': 0, 'away_q4': 0,
        'home_score': 0, 'away_score': 0,
        'rolling_home_score': 105.0, 'rolling_away_score': 105.0,
        'score_ratio': 0.5,
        'prev_matchup_diff': 0,
        'rest_days_home': 2, 'rest_days_away': 2, 'rest_advantage': 0,
        'is_back_to_back_home': 0, 'is_back_to_back_away': 0,
        'q1_to_q2_momentum': 0, 'q2_to_q3_momentum': 0, 'q3_to_q4_momentum': 0,
        'cumulative_momentum': 0
    }
    
    # Critical columns where missing values might significantly impact predictions
    critical_columns = ['home_q1', 'home_q2', 'home_q3', 'home_q4',
                       'away_q1', 'away_q2', 'away_q3', 'away_q4',
                       'score_ratio', 'cumulative_momentum']
    
    # For any column not in default_map, use 0 as default
    for col in feature_columns:
        if col not in default_map:
            default_map[col] = 0
    
    # Dictionary to track critical missing values
    missing_critical = {}
    
    # Process each column
    for col in feature_columns:
        if col in result_df.columns:
            # Store original NaN count for critical columns
            if flag_critical and col in critical_columns:
                nan_count = result_df[col].isna().sum()
                if nan_count > 0:
                    missing_critical[col] = nan_count
            
            # Convert to numeric, forcing errors to NaN
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # Replace NaN values with appropriate defaults
            result_df[col] = result_df[col].fillna(default_map.get(col, 0))
        else:
            # If column doesn't exist, add it with default values
            result_df[col] = default_map.get(col, 0)
    
    # Print summary of critical missing values if requested
    if flag_critical and missing_critical:
        print("WARNING: Critical missing values detected:")
        for col, count in missing_critical.items():
            print(f" • {col}: {count} missing values")
    
    if flag_critical:
        return result_df, missing_critical
    else:
        return result_df

def get_nba_season(date):
    """
    Extract NBA season from date (NBA seasons start in October and end in June)
    
    Args:
        date: datetime or string date
    Returns:
        str: NBA season in format "YYYY-YYYY"
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    year = date.year
    month = date.month
    
    # For October through December, the season starts in the current year
    if month >= 10:
        return f"{year}-{year+1}"
    # For January through June, the season started in the previous year
    elif month <= 6:
        return f"{year-1}-{year}"
    # For July through September, use the upcoming season
    else:
        return f"{year}-{year+1}"

def fetch_pacific_time():
    """Gets current time in Pacific timezone"""
    pacific_tz = pytz.timezone("America/Los_Angeles")
    return datetime.now(pacific_tz)