# backend/features/engine.py

from __future__ import annotations

import pandas as pd
from typing import Optional, Sequence

from .momentum    import transform as momentum_transform
from .advanced    import transform as advanced_transform
from .rolling     import transform as rolling_transform
from .rest        import transform as rest_transform
from .h2h         import transform as h2h_transform
from .season      import transform as season_transform
from .form        import transform as form_transform


class FeatureEngine:
    """
    Orchestrates the modular pipeline for feature engineering.
    Public API matches the original: `generate_all_features(...)`.
    """

    def __init__(self,
                 supabase_client: Optional[object] = None,
                 debug: bool = False):
        self.conn = supabase_client
        self.debug = debug

    def generate_all_features(
        self,
        df: pd.DataFrame,
        historical_games_df: Optional[pd.DataFrame] = None,
        team_stats_df: Optional[pd.DataFrame] = None,
        rolling_windows: Sequence[int] = (5, 10, 20),
        h2h_window: int = 7
    ) -> pd.DataFrame:
        """
        Applies each modular transform in sequence:

          1. momentum_transform
          2. advanced_transform
          3. rolling_transform
          4. rest_transform
          5. h2h_transform
          6. season_transform
          7. form_transform

        Returns a **new** DataFrame.
        """
        if df is None or df.empty:
            return df

        out = df.copy()

        out = momentum_transform(out, debug=self.debug)
        out = advanced_transform(out, debug=self.debug)
        out = rolling_transform(
            out,
            conn=self.conn,
            window_sizes=list(rolling_windows),
            debug=self.debug
        )
        out = rest_transform(out, debug=self.debug)
        out = h2h_transform(
            out,
            historical_df=historical_games_df,
            window=h2h_window,
            debug=self.debug
        )
        out = season_transform(
            out,
            team_stats_df=team_stats_df,
            debug=self.debug
        )
        out = form_transform(out, debug=self.debug)

        return out
