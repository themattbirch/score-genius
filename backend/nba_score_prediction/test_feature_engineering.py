import pandas as pd
from backend.nba_score_prediction.feature_engineering import NBAFeatureEngine

# --- Sample Data Setup ---
sample_games_data = {
    'game_id': ['DET_MEM_2024', 'DET_BOS_2024', 'MEM_LAL_2024'],
    'game_date': pd.to_datetime(['2024-11-10', '2024-11-12', '2024-11-11']),
    'home_team': ['Detroit Pistons', 'Detroit Pistons', 'Memphis Grizzlies'],
    'away_team': ['Memphis Grizzlies', 'Boston Celtics', 'Los Angeles Lakers'],
    # Add other necessary base columns with dummy values if needed
}
sample_games_df = pd.DataFrame(sample_games_data)

sample_historical_data = {  # For H2H testing
    'game_id': ['DET_MEM_2023'],
    'game_date': pd.to_datetime(['2023-12-15']),
    'home_team': ['Detroit Pistons'],
    'away_team': ['Memphis Grizzlies'],
    'home_score': [105],
    'away_score': [110],
    # Include additional historical columns as required
}
sample_historical_df = pd.DataFrame(sample_historical_data)

sample_team_stats_data = {  # For season context testing
    'team_name': ['Detroit Pistons', 'Memphis Grizzlies', 'Boston Celtics', 'Los Angeles Lakers'],
    'season': ['2024-2025'] * 4,
    'wins_all_percentage': [0.25, 0.45, 0.70, 0.55],
    'points_for_avg_all': [108.5, 112.0, 118.0, 115.0],
    'points_against_avg_all': [116.0, 113.0, 110.0, 114.0],
    'current_form': ['LLWLL', 'WLWLL', 'WWWLW', 'LWWWL'],
}
sample_team_stats_df = pd.DataFrame(sample_team_stats_data)

# --- Test Function Example ---
def test_rest_features_pistons_grizzlies():
    feature_engineer = NBAFeatureEngine(debug=False)
    
    # Run the rest features integration on the sample games data
    result_df = feature_engineer.add_rest_features_vectorized(sample_games_df.copy())

    # Example assertions (adjust these based on expected values from your logic)
    # For DET_BOS_2024 (Detroit Pistons playing an away game against Boston Celtics),
    # assume their last game (DET_MEM_2024) was on 2024-11-10.
    pistons_row = result_df[result_df['game_id'] == 'DET_BOS_2024'].iloc[0]
    # Example: Assert that rest_days_home is 2 (11-12 minus 11-10)
    assert pistons_row['rest_days_home'] == 2, f"Expected rest_days_home of 2, got {pistons_row['rest_days_home']}"
    # And that they are not on a back-to-back (0 indicates false)
    assert pistons_row['is_back_to_back_home'] == 0, "Expected is_back_to_back_home to be 0"

    # For MEM_LAL_2024, assume Memphis played their previous game on 2024-11-10.
    grizzlies_row = result_df[result_df['game_id'] == 'MEM_LAL_2024'].iloc[0]
    # Example: Assert that rest_days_home is 1 (11-11 minus 11-10)
    assert grizzlies_row['rest_days_home'] == 1, f"Expected rest_days_home of 1, got {grizzlies_row['rest_days_home']}"
    # And that they are flagged as back-to-back (1 indicates true)
    assert grizzlies_row['is_back_to_back_home'] == 1, "Expected is_back_to_back_home to be 1"

    print("test_rest_features_pistons_grizzlies PASSED")

# --- Run the tests ---
if __name__ == '__main__':
    test_rest_features_pistons_grizzlies()
    # Additional test functions could be called here
