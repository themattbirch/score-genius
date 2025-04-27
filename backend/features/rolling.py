import pandas as pd
from .utils import generate_rolling_column_name, DEFAULTS
from .base_windows import fetch_rolling
from .legacy.rolling import add_rolling_features as add_rolling_features_legacy

def add_rolling_features(df: pd.DataFrame,
                         conn=None) -> pd.DataFrame:
    if conn is None:                     # no DB → pure Python path
        return add_rolling_features_legacy(df, window_sizes=[5, 10, 20])

    rolled = fetch_rolling(conn, df['game_id'].astype(str).tolist())
    if rolled.empty:
        return add_rolling_features_legacy(df, window_sizes=[5, 10, 20])

    out = df.merge(rolled,
                   left_on=['game_id', 'home_team_norm'],
                   right_on=['game_id', 'team_norm'],
                   how='left',
                   suffixes=('', '_tmp'))

    # fill NaNs with defaults for new cols
    for col in rolled.columns:
        if col in ('game_id', 'team_norm', 'game_date'):
            continue
        if col not in out.columns:
            continue
        base = col.split('_')[1]  # crude parse, okay for MVP
        out[col] = out[col].fillna(DEFAULTS.get(base, 0.0))

    return out.drop(columns=['team_norm'])