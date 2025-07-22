# backend/nba_features/advanced.py
"""
Attach previous-season advanced splits to each game row.

Revision highlights
-------------------
* Skips team_id merge if data is unavailable and uses team_norm as the key.
* Copies merged stats into canonical `h_*_home` / `a_*_away` columns before imputation.
* Coverage logging after each stage to catch gaps.
* All helper columns and intermediate keys are dropped at the end.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name, profile_time

logger = logging.getLogger(__name__)
__all__: Sequence[str] = ["transform"]

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

EXPECTED_STATS: Sequence[str] = (
    "pace",
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "ft_rate",
)

# Collapse any string to lowercase alphanumerics – "Team Alpha" → "teamalpha"
_alphanum_key = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _norm_season_key(val: Any) -> Optional[int]:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        m = re.search(r"\d{4}", val)
        if m:
            return int(m.group(0))
    try:
        return int(float(val))
    except Exception:
        return np.nan


def _coverage_report(df: pd.DataFrame, stage: str) -> None:
    total = len(df)
    home_cols = [f"h_{s}_home" for s in EXPECTED_STATS if f"h_{s}_home" in df]
    away_cols = [f"a_{s}_away" for s in EXPECTED_STATS if f"a_{s}_away" in df]
    home_real = df[home_cols].notna().all(axis=1).sum() if home_cols else 0
    away_real = df[away_cols].notna().all(axis=1).sum() if away_cols else 0
    home_pct = 100 * home_real / total if total else 0.0
    away_pct = 100 * away_real / total if total else 0.0
    logger.info(
        f"Coverage Report ({stage}): Home={home_pct:.1f}% ({home_real}/{total}), "
        f"Away={away_pct:.1f}% ({away_real}/{total})"
    )
    if stage == "Final" and max(home_pct, away_pct) < 80:
        logger.warning("Final coverage below 80% – investigate ingestion.")


# --------------------------------------------------------------------------- #
# Core transform                                                               #
# --------------------------------------------------------------------------- #

@profile_time
def transform(
    df: pd.DataFrame,
    *,
    all_historical_splits_df: pd.DataFrame,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Attach previous‑season advanced splits to each game row."""

    out = df.copy()
    if "adv_stats_lookup_season" not in out:
        logger.error("Missing 'adv_stats_lookup_season'; skipping transform.")
        return out

    # -------------------------------------------------------------------
    # 1) Prepare game‑side keys                                           
    # -------------------------------------------------------------------
    out["season_join"] = (
        out["adv_stats_lookup_season"].apply(_norm_season_key).astype("Int64")
    )
    out["home_norm"] = out["home_team"].map(normalize_team_name).map(_alphanum_key)
    out["away_norm"] = out["away_team"].map(normalize_team_name).map(_alphanum_key)

    # -------------------------------------------------------------------
    # 2) Prepare splits table                                             
    # -------------------------------------------------------------------
    splits = all_historical_splits_df.copy()
    if splits.empty:
        logger.warning("Empty splits table; will impute all defaults.")
    else:
        if "season" in splits and "season_join" not in splits:
            splits["season_join"] = splits["season"].apply(_norm_season_key).astype("Int64")
        if "team_name" in splits:
            splits["team_norm"] = (
                splits["team_name"].map(normalize_team_name).map(_alphanum_key)
            )

    # -------------------------------------------------------------------
    # 3) Merge home & away stats                                          
    # -------------------------------------------------------------------
    merged = out.copy()
    if not splits.empty and "team_norm" in splits:
        # Home‑side stats
        home_cols = ["team_norm", "season_join"] + [f"{s}_home" for s in EXPECTED_STATS]
        home_stats = splits.loc[:, splits.columns.intersection(home_cols)].copy()

        # Away‑side stats
        away_cols = ["team_norm", "season_join"] + [f"{s}_away" for s in EXPECTED_STATS]
        away_stats = splits.loc[:, splits.columns.intersection(away_cols)].copy()

        merged = merged.merge(
            home_stats,
            left_on=["home_norm", "season_join"],
            right_on=["team_norm", "season_join"],
            how="left",
        )
        merged = merged.merge(
            away_stats,
            left_on=["away_norm", "season_join"],
            right_on=["team_norm", "season_join"],
            how="left",
            suffixes=("_home", "_away"),
        )
        _coverage_report(merged, "Team‑norm merge")

    # -------------------------------------------------------------------
    # 4) Copy merged columns into canonical h_* / a_* names               
    # -------------------------------------------------------------------
    for stat in EXPECTED_STATS:
        src_h = f"{stat}_home"
        dest_h = f"h_{stat}_home"
        if src_h in merged and dest_h not in merged:
            merged[dest_h] = merged[src_h]

        src_a = f"{stat}_away"
        dest_a = f"a_{stat}_away"
        if src_a in merged and dest_a not in merged:
            merged[dest_a] = merged[src_a]

    # -------------------------------------------------------------------
    # 5) Impute defaults & flag                                           
    # -------------------------------------------------------------------
    for prefix, side in (("h", "home"), ("a", "away")):
        for stat in EXPECTED_STATS:
            col = f"{prefix}_{stat}_{side}"
            flag_col = f"{col}_imputed"
            default_val = DEFAULTS.get(stat, 0.0)

            if col not in merged:
                merged[col] = np.nan  # create column so imputation is recorded
            if flag_imputations:
                merged[flag_col] = merged[col].isna()
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default_val)

    # -------------------------------------------------------------------
    # 6) Compute differentials                                            
    # -------------------------------------------------------------------
    for stat in EXPECTED_STATS:
        h_col = f"h_{stat}_home"
        a_col = f"a_{stat}_away"
        if h_col in merged and a_col in merged:
            merged[f"hist_{stat}_split_diff"] = merged[h_col] - merged[a_col]

    # -------------------------------------------------------------------
    # 7) Diagnostics & cleanup                                            
    # -------------------------------------------------------------------
    _coverage_report(merged, "Final")
    merged.drop(
        columns=[
            "season_join",
            "home_norm",
            "away_norm",
            "team_norm_home",
            "team_norm_away",
        ],
        errors="ignore",
        inplace=True,
    )

    return merged