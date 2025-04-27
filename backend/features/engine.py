from __future__ import annotations
import pandas as pd
from .utils import DEFAULTS
from .momentum import add_intra_game_momentum
from .advanced import add_advanced_features            # temporary pass-through
from .rolling import add_rolling_features
from .rest import add_rest_features
from .h2h import add_matchup_features
from .season import add_season_features
from .form import add_form_features

class FeatureEngine:
    """
    Thin orchestrator that calls the modular feature steps.
    Public API matches the original: `generate_all_features(...)`.
    """
    def __init__(self, supabase_client=None):
        self.conn = supabase_client        # optional for SQL fetch

    def generate_all_features(
        self,
        df: pd.DataFrame,
        historical_games_df: pd.DataFrame | None = None,
        team_stats_df: pd.DataFrame | None = None,
        rolling_windows: list[int] = [5, 10, 20],
        h2h_window: int = 7
    ) -> pd.DataFrame:

        if df.empty:
            return df

        out = df.copy()
        # ---- Step sequence (unchanged order) ----
        out = add_intra_game_momentum(out)
        out = add_advanced_features(out)
        out = add_rolling_features(out, conn=self.conn)
        out = add_rest_features(out)
        out = add_matchup_features(out, historical_games_df, max_games=h2h_window)
        out = add_season_features(out, team_stats_df)
        out = add_form_features(out)
        return out
