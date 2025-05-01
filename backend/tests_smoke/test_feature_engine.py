# backend/tests_smoke/test_feature_engine.py
import pandas as pd
from backend.features.legacy.feature_engineering import FeatureEngine

def test_generate_minimal_features():
    fe = FeatureEngine(debug=False)
    df = pd.DataFrame([{
        "game_id": "9999",
        "game_date": "2024-01-01",
        "home_team": "Lakers",
        "away_team": "Celtics",
        # minimal boxes so functions don't crash
        "home_score": 100,
        "away_score": 98,
    }])
    out = fe.generate_all_features(df)
    # sanity checks
    assert not out.empty
    assert "predicted_point_diff" not in out.columns          # still raw features
    assert (out["game_id"] == "9999").all()
