# backend/nba_features/advanced.py
"""
Attaches pre-calculated historical/seasonal advanced statistical splits to game data.

This module takes an input DataFrame of games and another DataFrame containing
pre-calculated seasonal advanced stats (with home/away splits, often from an RPC
or a processed data source like 'nba_team_seasonal_advanced_splits').
It looks up the relevant season's home performance stats for the home team and
away performance stats for the away team, and attaches them to each game.
Differentials between these historical stats are also calculated.
"""

from __future__ import annotations
import logging
from typing import Mapping, Sequence, Dict # Added Dict

import numpy as np # Not strictly needed here now, but pandas uses it
import pandas as pd

# Assuming profile_time is for performance measurement, keep if used
from .utils import normalize_team_name, DEFAULTS, profile_time 

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# Expected base advanced stat names (these should align with keys in DEFAULTS
# and the base names in the columns of stats_df, e.g., 'pace' for 'pace_home')
EXPECTED_STATS: Sequence[str] = [
    "pace", "off_rtg", "def_rtg", "net_rtg", "efg_pct", "tov_pct", "oreb_pct", "ft_rate"
]

_STAT_MAP = {
    "pace":               "pace",
    "off_rtg":            "offensive_rating",
    "def_rtg":            "defensive_rating",
    "net_rtg":            "net_rating",
    "efg_pct":            "efg_pct",
    "tov_pct":            "tov_rate",
    "oreb_pct":           "oreb_pct",
    "ft_rate":            "ft_rate",
}

@profile_time
def transform(
    df: pd.DataFrame,
    *,
    stats_df: pd.DataFrame,
    season: int,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    # 1) normalize team names
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)

    # 2) filter and prepare seasonal splits
    seasonal = stats_df[stats_df["season"] == season].copy()
    if "team_norm" not in seasonal.columns:
        seasonal["team_norm"] = seasonal["team_name"].map(normalize_team_name)

    # 3) for each side + stat, pull via .map() so we always create the column
    for side, prefix in (("home", "h"), ("away", "a")):
        lookup = seasonal.set_index("team_norm")  # indexable by normalize_team_name
        for stat in EXPECTED_STATS:
            src_col = f"{stat}_{side}"
            tgt_col = f"{prefix}_{stat}_{side}"
            impute_col = f"{tgt_col}_imputed"

            if src_col in lookup.columns:
                out[tgt_col] = out[f"{side}_norm"].map(lookup[src_col])
            else:
                # missing entirely → all NaN
                out[tgt_col] = pd.NA

            # imputation flag & fill
            default = DEFAULTS.get(stat, 0.0)
            if flag_imputations:
                out[impute_col] = out[tgt_col].isna()
            out[tgt_col] = (
                pd.to_numeric(out[tgt_col], errors="coerce")
                .fillna(default)
                .astype(float)
            )
            if flag_imputations:
                out[impute_col] = out[impute_col].astype(bool)

    # 4) compute split diffs
    for stat in EXPECTED_STATS:
        out[f"hist_{stat}_split_diff"] = (
            out[f"h_{stat}_home"] - out[f"a_{stat}_away"]
        )
    # Mirror the specific rating columns into the names rolling expects:
    out["home_offensive_rating"]  = out["h_off_rtg_home"]
    out["away_offensive_rating"]  = out["a_off_rtg_away"]
    out["home_defensive_rating"]  = out["h_def_rtg_home"]
    out["away_defensive_rating"]  = out["a_def_rtg_away"]
    out["home_net_rating"]        = out["h_net_rtg_home"]
    out["away_net_rating"]        = out["a_net_rtg_away"]

    # 5) cleanup
    return out.drop(columns=["home_norm", "away_norm"], errors="ignore")
