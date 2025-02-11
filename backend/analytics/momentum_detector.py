# backend/analytics/momentum_detector.py
def detect_momentum(game_data: dict) -> dict:
    """
    Detect momentum shifts in game data using placeholder logic.
    Returns a dictionary with momentum trend details.
    """
    try:
        scores = game_data.get('scores', {})
        home_q1 = scores.get('home', {}).get('quarter_1', 0)
        away_q1 = scores.get('away', {}).get('quarter_1', 0)
        if home_q1 - away_q1 > 10:
            trend = "Home momentum"
        elif away_q1 - home_q1 > 10:
            trend = "Away momentum"
        else:
            trend = "Balanced"
        return {"trend": trend}
    except Exception:
        return {"trend": "Unknown"}

if __name__ == "__main__":
    dummy_game_data = {
        "scores": {
            "home": {"quarter_1": 35},
            "away": {"quarter_1": 20}
        }
    }
    momentum = detect_momentum(dummy_game_data)
    print(f"Detected momentum: {momentum}")
