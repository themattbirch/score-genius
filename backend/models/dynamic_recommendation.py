# backend/models/dynamic_recommendation.py

import numpy as np

def generate_recommendations(model_outputs: dict, player_projection: dict = None) -> dict:
    """
    Generates dynamic recommendations based on outputs from various models,
    with game state awareness and additional over/under and fantasy recommendations.

    Parameters:
        model_outputs (dict): A dictionary containing outputs from various models,
                              e.g., win probability, momentum shift, projected margin,
                              total projected score, quarter, time remaining, etc.
        player_projection (dict, optional): A dictionary mapping player names to their 
                              current performance stats (e.g., current_points, current_rebounds).
    
    Returns:
        dict: A dictionary of recommendations, such as betting tips, momentum advice,
              spread tips, over/under recommendations, fantasy tips, clutch tips, and game flow tips.
    """
    recommendations = {}
    
    # Extract basic values
    win_prob = model_outputs.get("win_probability", 0.5)
    momentum = model_outputs.get("momentum_shift", 0)
    projected_margin = model_outputs.get("projected_margin", 0)
    total_projected = model_outputs.get("total_projected_score", 0)
    
    # Extract game state information
    quarter = model_outputs.get("quarter", 0)
    time_remaining = model_outputs.get("time_remaining", 48)  # Default: full game minutes
    
    # Adjust betting tip based on game state
    if quarter >= 3 and win_prob > 0.9:
        recommendations["betting_tip"] = "Late game, high confidence in home team win."
    else:
        if win_prob > 0.75:
            recommendations["betting_tip"] = "Strong home advantage indicates high probability of winning."
        elif win_prob < 0.5:
            recommendations["betting_tip"] = "Home team seems unlikely to win."
        else:
            recommendations["betting_tip"] = "Game appears competitive; consider hedging."
    
    # Additional recommendations based on game state:
    if quarter == 4 and time_remaining < 5 and abs(projected_margin) < 6:
        recommendations["clutch_tip"] = "Close game in final minutes - consider wagering on final possession outcomes."
    
    if quarter < 2:
        recommendations["game_flow_tip"] = "Early game - monitor pace and shooting percentages before wagering."
    
    # Momentum advice
    if momentum > 0.3:
        recommendations["momentum_advice"] = "Momentum is shifting strongly - monitor game pace for opportunities."
    elif momentum < -0.3:
        recommendations["momentum_advice"] = "Opposing momentum detected - consider conservative play."
    else:
        recommendations["momentum_advice"] = "Momentum appears balanced."
    
    # Spread tip based on projected margin
    if projected_margin >= 10:
        recommendations["spread_tip"] = "High projected margin - consider the spread."
    elif projected_margin <= -10:
        recommendations["spread_tip"] = "Low projected margin - consider reverse spread."
    else:
        recommendations["spread_tip"] = "Projected margin is narrow - use caution with spread action."
    
    # Over/Under Recommendation
    if total_projected > 0:
        over_under = "over" if total_projected > 220 else "under"
        recommendations["over_under_tip"] = f"Projected total score of {total_projected} suggests {over_under} wager."
    
    # Enhanced Fantasy Recommendations
    if player_projection and quarter >= 3:
        high_performing_players = {
            player: stats for player, stats in player_projection.items()
            if stats.get("current_points", 0) > 15 or stats.get("current_rebounds", 0) > 8
        }
        if high_performing_players:
            players_list = ", ".join(high_performing_players.keys())
            recommendations["fantasy_tip"] = f"Players having strong games: {players_list}. Consider for DFS lineups."
        else:
            recommendations["fantasy_tip"] = "No standout player performance detected for fantasy play."
    else:
        recommendations["fantasy_tip"] = "Monitor player performance for fantasy opportunities."
    
    return recommendations

if __name__ == "__main__":
    # Example usage:
    example_model_outputs = {
        "win_probability": 0.92,
        "momentum_shift": 0.35,
        "projected_margin": 4,
        "total_projected_score": 225,
        "quarter": 4,
        "time_remaining": 4  # minutes remaining in the game
    }
    # Example player projection data
    example_player_projection = {
        "Player X": {"current_points": 18, "current_rebounds": 9},
        "Player Y": {"current_points": 14, "current_rebounds": 7}
    }
    
    recommendations = generate_recommendations(example_model_outputs, example_player_projection)
    for key, tip in recommendations.items():
        print(f"{key}: {tip}")
