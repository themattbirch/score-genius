# /notebooks/basketball_metrics.py

import pandas as pd
import numpy as np

def convert_time_to_minutes(time_str):
    """
    Convert time string in MM:SS format to numeric minutes.
    
    Args:
        time_str: String in format 'MM:SS' (e.g., '24:35')
    Returns:
        Float representing total minutes (e.g., 24.583)
    """
    if not time_str or not isinstance(time_str, str):
        return 0.0
    
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + (seconds / 60.0)
        else:
            return float(time_str)  # Try direct conversion if not in MM:SS format
    except (ValueError, TypeError):
        print(f"Warning: Could not convert time value '{time_str}' to minutes")
        return 0.0

def integrated_basketball_analytics(df):
    """
    Integrated function to calculate all basketball analytics metrics in the proper sequence.
    This combines the shooting metrics, free throw metrics, advanced metrics,
    and defensive metrics calculations into a single pipeline.
    
    Args:
        df: DataFrame with raw game data
    Returns:
        DataFrame with all basketball analytics metrics added
    """
    print("Calculating comprehensive basketball analytics metrics...")
    
    # 1. Calculate shooting metrics
    df = calculate_shooting_metrics(df)
    
    # 2. Calculate free throw metrics
    df = calculate_free_throw_metrics(df)
    
    # 3. Calculate advanced metrics
    df = calculate_advanced_metrics(df)
    
    # 4. Calculate defensive metrics
    df = calculate_defensive_metrics(df)
    
    # 5. Calculate interaction metrics
    if all(col in df.columns for col in ['cumulative_momentum', 'home_off_efficiency']):
        # Momentum-efficiency interaction (high momentum + high efficiency is extra strong)
        df['momentum_efficiency'] = df['cumulative_momentum'] * (
            df['home_off_efficiency'] - df['away_off_efficiency']) / 100
    
    if all(col in df.columns for col in ['score_differential', 'game_pace']):
        # Pace-adjusted score differential
        # (Same point differential has different meaning in slow vs fast-paced games)
        df['pace_adj_diff'] = df['score_differential'] * (100 / df['game_pace'])
    
    print("Basketball analytics metrics calculated successfully")
    return df

def add_basketball_metrics_to_features(feature_sets):
    """
    Enhance the existing feature sets with basketball analytics metrics and team form.
    This function integrates shooting, free throw, advanced metrics,
    and defensive metrics into the quarter-specific feature sets.

    Args:
        feature_sets: Dictionary with feature lists for each quarter
    Returns:
        Updated feature sets with basketball analytics metrics
    """
    # These are the basketball analytics metrics to potentially add
    shooting_metrics = ['home_fg_pct', 'away_fg_pct', 'fg_pct_diff', 'home_efg_pct', 'away_efg_pct']
    free_throw_metrics = ['home_ft_pct', 'away_ft_pct', 'ft_pct_diff', 'home_ft_rate', 'away_ft_rate']
    advanced_metrics = ['home_possessions', 'away_possessions', 'game_pace',
                       'home_off_efficiency', 'away_off_efficiency', 'efficiency_diff',
                       'momentum_efficiency', 'pace_adj_diff']
    form_metrics = ['home_form_score', 'form_score_diff', 'home_streak', 'streak_diff', 'total_momentum']
    defensive_metrics = ['steal_diff', 'block_diff', 'def_reb_diff', 'def_efficiency_diff', 'turnover_rate_diff']
    
    # Enhanced feature sets with basketball metrics
    enhanced_sets = {}
    for quarter, features in feature_sets.items():
        # Start with existing features
        enhanced_features = features.copy()
        
        # Add appropriate metrics based on quarter
        if quarter == 'q1':
            # For pre-game, team form is especially important
            enhanced_features.extend([
                'fg_pct_diff',
                'ft_pct_diff',
                'efficiency_diff',
                'home_form_score',
                'streak_diff', # Form is key for pre-game prediction
                'def_reb_diff'  # Add defensive rebounding as a key pregame indicator
            ])
        elif quarter == 'q2':
            # In Q2, current form still matters but in-game stats start to take over
            enhanced_features.extend([
                'fg_pct_diff',
                'ft_pct_diff',
                'game_pace',
                'efficiency_diff',
                'home_form_score',
                'total_momentum', # Combined momentum and form
                'steal_diff',     # Add defensive activity metrics
                'block_diff'
            ])
        elif quarter == 'q3':
            # In Q3, form becomes less important as game stats take over
            enhanced_features.extend([
                'fg_pct_diff',
                'ft_pct_diff',
                'game_pace',
                'efficiency_diff',
                'momentum_efficiency',
                'total_momentum',
                'def_efficiency_diff',  # Add defensive efficiency as games progress
                'turnover_rate_diff'
            ])
        elif quarter == 'q4':
            # In Q4, use all available metrics
            enhanced_features.extend([
                'fg_pct_diff',
                'ft_pct_diff',
                'game_pace',
                'efficiency_diff',
                'momentum_efficiency',
                'pace_adj_diff',
                'total_momentum', # Still include form in late game scenarios
                'steal_diff',
                'block_diff',
                'def_efficiency_diff',
                'turnover_rate_diff'
            ])
        
        enhanced_sets[quarter] = enhanced_features
    
    return enhanced_sets

def ensure_numeric_features(df, columns=None):
    """
    Ensure specified columns are numeric, converting them if necessary.
    
    Args:
        df: DataFrame with columns to convert
        columns: List of column names to convert (if None, tries to convert all)
    Returns:
        DataFrame with converted numeric columns
    """
    result_df = df.copy()
    
    # If no columns specified, try to convert all
    if columns is None:
        columns = result_df.columns
    
    for col in columns:
        if col in result_df.columns:
            # Try to convert to numeric, coercing errors to NaN
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # Fill NaN values with 0
            if result_df[col].isna().any():
                print(f"Warning: Column '{col}' contains non-numeric values that were converted to 0")
                result_df[col] = result_df[col].fillna(0)
    
    return result_df

def calculate_shooting_metrics(df):
    """
    Calculate shooting metrics like FG%, eFG%, and related statistics.
    Args:
        df: DataFrame with raw game data including field goal and free throw stats
    Returns:
        DataFrame with added shooting metrics
    """
    result_df = df.copy()
    
    # Check if field goal stats are available
    fg_cols = ['home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted']
    if not all(col in result_df.columns for col in fg_cols):
        print("Field goal data not available. Skipping shooting metrics calculation.")
        return result_df
    
    # Ensure data is numeric
    result_df = ensure_numeric_features(result_df, fg_cols)
    
    # Calculate field goal percentages (handle division by zero)
    result_df['home_fg_pct'] = np.divide(
        result_df['home_fg_made'], 
        result_df['home_fg_attempted'],
        out=np.zeros_like(result_df['home_fg_made'], dtype=float),
        where=result_df['home_fg_attempted'] > 0
    )
    
    result_df['away_fg_pct'] = np.divide(
        result_df['away_fg_made'], 
        result_df['away_fg_attempted'],
        out=np.zeros_like(result_df['away_fg_made'], dtype=float),
        where=result_df['away_fg_attempted'] > 0
    )
    
    # Calculate field goal percentage differential
    result_df['fg_pct_diff'] = result_df['home_fg_pct'] - result_df['away_fg_pct']
    
    # Add 3-point metrics if available
    three_pt_cols = ['home_3pm', 'home_3pa', 'away_3pm', 'away_3pa']
    if all(col in result_df.columns for col in three_pt_cols):
        result_df = ensure_numeric_features(result_df, three_pt_cols)
        
        # Calculate 3-point percentages
        result_df['home_3pt_pct'] = np.divide(
            result_df['home_3pm'], 
            result_df['home_3pa'],
            out=np.zeros_like(result_df['home_3pm'], dtype=float),
            where=result_df['home_3pa'] > 0
        )
        
        result_df['away_3pt_pct'] = np.divide(
            result_df['away_3pm'], 
            result_df['away_3pa'],
            out=np.zeros_like(result_df['away_3pm'], dtype=float),
            where=result_df['away_3pa'] > 0
        )
        
        # Calculate 3-point percentage differential
        result_df['3pt_pct_diff'] = result_df['home_3pt_pct'] - result_df['away_3pt_pct']
        
        # Calculate effective field goal percentage (eFG%)
        # eFG% = (FG + 0.5 * 3P) / FGA
        result_df['home_efg_pct'] = np.divide(
            (result_df['home_fg_made'] + 0.5 * result_df['home_3pm']),
            result_df['home_fg_attempted'],
            out=np.zeros_like(result_df['home_fg_made'], dtype=float),
            where=result_df['home_fg_attempted'] > 0
        )
        
        result_df['away_efg_pct'] = np.divide(
            (result_df['away_fg_made'] + 0.5 * result_df['away_3pm']),
            result_df['away_fg_attempted'],
            out=np.zeros_like(result_df['away_fg_made'], dtype=float),
            where=result_df['away_fg_attempted'] > 0
        )
        
        # Calculate eFG% differential
        result_df['efg_pct_diff'] = result_df['home_efg_pct'] - result_df['away_efg_pct']
        
        print("Added 3-point and effective field goal percentage metrics")
    
    # Calculate game-state adjusted shooting metrics if available
    if 'current_quarter' in result_df.columns:
        # Identify key shooting metrics by quarter
        quarter_fg_cols = [f'home_fg_made_q{q}' for q in range(1, 5)] + [f'away_fg_made_q{q}' for q in range(1, 5)]
        if any(col in result_df.columns for col in quarter_fg_cols):
            # Calculate quarter-by-quarter shooting metrics
            for q in range(1, 5):
                home_q = f'home_fg_made_q{q}'
                away_q = f'away_fg_made_q{q}'
                home_att_q = f'home_fg_attempted_q{q}'
                away_att_q = f'away_fg_attempted_q{q}'
                
                # If we have the data for this quarter
                if all(col in result_df.columns for col in [home_q, away_q, home_att_q, away_att_q]):
                    result_df[f'home_fg_pct_q{q}'] = np.divide(
                        result_df[home_q], 
                        result_df[home_att_q],
                        out=np.zeros_like(result_df[home_q], dtype=float),
                        where=result_df[home_att_q] > 0
                    )
                    
                    result_df[f'away_fg_pct_q{q}'] = np.divide(
                        result_df[away_q], 
                        result_df[away_att_q],
                        out=np.zeros_like(result_df[away_q], dtype=float),
                        where=result_df[away_att_q] > 0
                    )
                    
                    print(f"Added quarter {q} shooting percentages")
    
    # Example validation
    if 'home_fg_pct' in result_df.columns:
        # Check for unrealistic values
        if (result_df['home_fg_pct'] > 1).any():
            print("Warning: Found home FG% values greater than 100%")
        
    if 'away_fg_pct' in result_df.columns:
        # Check for unrealistic values
        if (result_df['away_fg_pct'] > 1).any():
            print("Warning: Found away FG% values greater than 100%")
    
    print("Added shooting metrics: home_fg_pct, away_fg_pct, fg_pct_diff")
    return result_df

def calculate_free_throw_metrics(df):
    """
    Calculate free throw related metrics.
    
    Args:
        df: DataFrame with game data
    Returns:
        DataFrame with added free throw metrics
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if free throw stats are available
    ft_cols = ['home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted']
    if not all(col in result_df.columns for col in ft_cols):
        print("Free throw data not available. Skipping free throw metrics calculation.")
        return result_df
    
    # Ensure data is numeric
    result_df = ensure_numeric_features(result_df, ft_cols)
    
    # Calculate free throw percentages (handle division by zero)
    result_df['home_ft_pct'] = np.divide(
        result_df['home_ft_made'], 
        result_df['home_ft_attempted'],
        out=np.zeros_like(result_df['home_ft_made'], dtype=float),
        where=result_df['home_ft_attempted'] > 0
    )
    
    result_df['away_ft_pct'] = np.divide(
        result_df['away_ft_made'], 
        result_df['away_ft_attempted'],
        out=np.zeros_like(result_df['away_ft_made'], dtype=float),
        where=result_df['away_ft_attempted'] > 0
    )
    
    # Calculate free throw percentage differential
    result_df['ft_pct_diff'] = result_df['home_ft_pct'] - result_df['away_ft_pct']
    
    # Calculate free throw rate (FTA/FGA) if field goal data is available
    if all(col in result_df.columns for col in ['home_fg_attempted', 'away_fg_attempted']):
        result_df['home_ft_rate'] = np.divide(
            result_df['home_ft_attempted'], 
            result_df['home_fg_attempted'],
            out=np.zeros_like(result_df['home_ft_attempted'], dtype=float),
            where=result_df['home_fg_attempted'] > 0
        )
        
        result_df['away_ft_rate'] = np.divide(
            result_df['away_ft_attempted'], 
            result_df['away_fg_attempted'],
            out=np.zeros_like(result_df['away_ft_attempted'], dtype=float),
            where=result_df['away_fg_attempted'] > 0
        )
        
        # Free throw rate differential
        result_df['ft_rate_diff'] = result_df['home_ft_rate'] - result_df['away_ft_rate']
        print("Added free throw rate metrics: home_ft_rate, away_ft_rate, ft_rate_diff")
    
    # Example validation
    if 'home_ft_pct' in result_df.columns:
        # Check for unrealistic values
        if (result_df['home_ft_pct'] > 1).any():
            print("Warning: Found home FT% values greater than 100%")
        
    if 'away_ft_pct' in result_df.columns:
        # Check for unrealistic values
        if (result_df['away_ft_pct'] > 1).any():
            print("Warning: Found away FT% values greater than 100%")
    
    print("Added free throw metrics: home_ft_pct, away_ft_pct, ft_pct_diff")
    return result_df

def calculate_advanced_metrics(df):
    """
    Calculate advanced metrics like possessions, pace, and efficiency ratings.
    Args:
        df: DataFrame with raw game data
    Returns:
        DataFrame with added advanced metrics
    """
    result_df = df.copy()
    
    # Check for required columns
    required_cols = ['home_fg_attempted', 'away_fg_attempted', 'home_ft_attempted', 'away_ft_attempted']
    if not all(col in result_df.columns for col in required_cols):
        print("Required data for possession calculation not available. Skipping advanced metrics.")
        return result_df
    
    # For full possession calculation we need ORB (offensive rebounds) and TO (turnovers)
    # If not available, we'll use an estimated formula
    has_orb_to = all(col in result_df.columns for col in 
                     ['home_off_reb', 'away_off_reb', 'home_turnovers', 'away_turnovers'])
    
    # Calculate possessions
    # Formula: Possessions = FGA - ORB + TO + 0.44 * FTA
    # If ORB and TO not available, use simplified formula: Possessions ≈ FGA + 0.44 * FTA
    
    # Home possessions
    if has_orb_to:
        result_df['home_possessions'] = result_df['home_fg_attempted'] - result_df['home_off_reb'] + \
                                       result_df['home_turnovers'] + 0.44 * result_df['home_ft_attempted']
        result_df['away_possessions'] = result_df['away_fg_attempted'] - result_df['away_off_reb'] + \
                                       result_df['away_turnovers'] + 0.44 * result_df['away_ft_attempted']
    else:
        # Simplified formula
        result_df['home_possessions'] = result_df['home_fg_attempted'] + 0.44 * result_df['home_ft_attempted']
        result_df['away_possessions'] = result_df['away_fg_attempted'] + 0.44 * result_df['away_ft_attempted']
    
    # Calculate pace (possessions per 48 minutes)
    if 'minutes_numeric' in result_df.columns:
        # If we have actual minutes played
        result_df['home_pace'] = np.divide(
            result_df['home_possessions'] * 48,
            result_df['minutes_numeric'],
            out=np.zeros_like(result_df['home_possessions'], dtype=float),
            where=result_df['minutes_numeric'] > 0
        )
        
        result_df['away_pace'] = np.divide(
            result_df['away_possessions'] * 48,
            result_df['minutes_numeric'],
            out=np.zeros_like(result_df['away_possessions'], dtype=float),
            where=result_df['minutes_numeric'] > 0
        )
    elif 'current_quarter' in result_df.columns:
        # Estimate minutes based on current quarter
        result_df['estimated_minutes'] = result_df['current_quarter'].apply(
            lambda q: min((q - 1) * 12 + 6, 48) # Assuming middle of current quarter
        )
        
        result_df['home_pace'] = np.divide(
            result_df['home_possessions'] * 48,
            result_df['estimated_minutes'],
            out=np.zeros_like(result_df['home_possessions'], dtype=float),
            where=result_df['estimated_minutes'] > 0
        )
        
        result_df['away_pace'] = np.divide(
            result_df['away_possessions'] * 48,
            result_df['estimated_minutes'],
            out=np.zeros_like(result_df['away_possessions'], dtype=float),
            where=result_df['estimated_minutes'] > 0
        )
    
    # Calculate game pace (average of home and away)
    result_df['game_pace'] = (result_df['home_pace'] + result_df['away_pace']) / 2
    
    # Calculate offensive efficiency (points per 100 possessions)
    if all(col in result_df.columns for col in ['home_score', 'away_score']):
        result_df['home_off_efficiency'] = np.divide(
            result_df['home_score'] * 100,
            result_df['home_possessions'],
            out=np.zeros_like(result_df['home_score'], dtype=float),
            where=result_df['home_possessions'] > 0
        )
        
        result_df['away_off_efficiency'] = np.divide(
            result_df['away_score'] * 100,
            result_df['away_possessions'],
            out=np.zeros_like(result_df['away_score'], dtype=float),
            where=result_df['away_possessions'] > 0
        )
        
        # Calculate efficiency differential
        result_df['efficiency_diff'] = result_df['home_off_efficiency'] - result_df['away_off_efficiency']
    
    print("Added advanced metrics: possessions, pace, and offensive efficiency")
    return result_df

def calculate_defensive_metrics(df):
    """
    Calculate defensive metrics like steals, blocks, defensive efficiency.
    Args:
        df: DataFrame with raw game data including defensive stats
    Returns:
        DataFrame with added defensive metrics
    """
    result_df = df.copy()
    
    # Check if defensive stats are available
    defense_cols = ['home_steals', 'away_steals', 'home_blocks', 'away_blocks', 
                    'home_def_reb', 'away_def_reb', 'home_turnovers', 'away_turnovers']
    
    if not any(col in result_df.columns for col in defense_cols):
        print("Defensive data not available. Skipping defensive metrics calculation.")
        return result_df
    
    # Ensure available columns are numeric
    available_cols = [col for col in defense_cols if col in result_df.columns]
    if available_cols:
        result_df = ensure_numeric_features(result_df, available_cols)
    
    # Calculate steal differential if available
    if all(col in result_df.columns for col in ['home_steals', 'away_steals']):
        result_df['steal_diff'] = result_df['home_steals'] - result_df['away_steals']
        print("Added steal differential")
    
    # Calculate block differential if available
    if all(col in result_df.columns for col in ['home_blocks', 'away_blocks']):
        result_df['block_diff'] = result_df['home_blocks'] - result_df['away_blocks']
        print("Added block differential")
    
    # Calculate defensive rebound differential if available
    if all(col in result_df.columns for col in ['home_def_reb', 'away_def_reb']):
        result_df['def_reb_diff'] = result_df['home_def_reb'] - result_df['away_def_reb']
        print("Added defensive rebound differential")
    
    # Calculate defensive efficiency if possessions and opponent score are available
    if all(col in result_df.columns for col in ['home_possessions', 'away_possessions', 'home_score', 'away_score']):
        # Home defensive efficiency (points allowed per 100 possessions)
        result_df['home_def_efficiency'] = np.divide(
            result_df['away_score'] * 100, 
            result_df['home_possessions'],
            out=np.zeros_like(result_df['away_score'], dtype=float),
            where=result_df['home_possessions'] > 0
        )
        
        # Away defensive efficiency
        result_df['away_def_efficiency'] = np.divide(
            result_df['home_score'] * 100, 
            result_df['away_possessions'],
            out=np.zeros_like(result_df['home_score'], dtype=float),
            where=result_df['away_possessions'] > 0
        )
        
        # Defensive efficiency differential (lower is better for defense, so flip the sign)
        result_df['def_efficiency_diff'] = result_df['away_def_efficiency'] - result_df['home_def_efficiency']
        print("Added defensive efficiency metrics")
    
    # Calculate turnover rate if turnovers and possessions are available
    if all(col in result_df.columns for col in ['home_turnovers', 'away_turnovers', 'home_possessions', 'away_possessions']):
        # Home turnover rate (turnovers per 100 possessions)
        result_df['home_turnover_rate'] = np.divide(
            result_df['home_turnovers'] * 100, 
            result_df['home_possessions'],
            out=np.zeros_like(result_df['home_turnovers'], dtype=float),
            where=result_df['home_possessions'] > 0
        )
        
        # Away turnover rate
        result_df['away_turnover_rate'] = np.divide(
            result_df['away_turnovers'] * 100, 
            result_df['away_possessions'],
            out=np.zeros_like(result_df['away_turnovers'], dtype=float),
            where=result_df['away_possessions'] > 0
        )
        
        # Turnover rate differential
        result_df['turnover_rate_diff'] = result_df['away_turnover_rate'] - result_df['home_turnover_rate']
        print("Added turnover rate metrics")
    
    return result_df

# Example usage:
quarter_feature_sets = {
    'q1': ['prev_matchup_diff', 'rest_advantage'],
    'q2': ['home_q1', 'away_q1', 'score_ratio'],
    'q3': ['home_q1', 'home_q2', 'away_q1', 'away_q2', 'cumulative_momentum'],
    'q4': ['home_q1', 'home_q2', 'home_q3', 'away_q1', 'away_q2', 'away_q3', 'score_differential']
}

# Enhance features with basketball metrics
enhanced_feature_sets = add_basketball_metrics_to_features(quarter_feature_sets)

# This code will only run if this file is executed directly (not when imported)
if __name__ == "__main__":
    # Display the enhanced feature sets
    for quarter, features in enhanced_feature_sets.items():
        print(f"\n{quarter.upper()} Enhanced Features:")
        for feature in features:
            print(f" • {feature}")

    # Example code that uses features_df - only runs when module is executed directly
    import pandas as pd
    # Create a simple example dataframe for demonstration
    example_features_df = pd.DataFrame({
        'home_score': [100], 'away_score': [95],
        'home_fg_made': [40], 'home_fg_attempted': [85],
        'away_fg_made': [38], 'away_fg_attempted': [90],
        'home_ft_made': [20], 'home_ft_attempted': [25],
        'away_ft_made': [19], 'away_ft_attempted': [22],
        'home_steals': [8], 'away_steals': [6],
        'home_blocks': [5], 'away_blocks': [3],
        'home_def_reb': [30], 'away_def_reb': [25],
        'home_turnovers': [12], 'away_turnovers': [15]
    })
    
    print("\nApplying basketball analytics to example features dataframe...")
    enhanced_df = integrated_basketball_analytics(example_features_df)
    print(f"Added {len(enhanced_df.columns) - len(example_features_df.columns)} new metrics columns")