# backend/utils/preprocess_data.py
import pandas as pd

def clean_game_data(raw_data: dict) -> pd.DataFrame:
    """
    Conavert raw game JSON data into a cleaned pandas DataFrame.
    """
    games = raw_data.get("response", [])
    df = pd.json_normalize(games)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate additional features from the game data.
    """
    if "scores.home.total" in df.columns and "scores.away.total" in df.columns:
        df["score_diff"] = df["scores.home.total"] - df["scores.away.total"]
    return df

if __name__ == "__main__":
    sample_data = {
        "response": [
            {
                "id": 1912,
                "scores": {"home": {"total": 115}, "away": {"total": 104}},
                "status": {"short": "FT"}
            }
        ]
    }
    df = clean_game_data(sample_data)
    df = engineer_features(df)
    print(df)
