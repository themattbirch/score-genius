# backend/mlb_features/handedness_for_display.py

from __future__ import annotations
import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name
except ImportError:
    MLB_DEFAULTS: dict = {}
    def normalize_team_name(val: Any) -> str:
        return str(val).strip().lower() if pd.notna(val) else "Unknown"

# ---------------------------------------------------------------------------
# Shared normaliser
# ---------------------------------------------------------------------------

def _safe_norm(val: Any) -> str:
    if pd.isna(val):
        return "unknown"
    mapped = normalize_team_name(val)
    if isinstance(mapped, str) and not mapped.lower().startswith("unknown"):
        base = mapped
    else:
        base = str(val)
    return base.strip().lower().replace(" ", "")

# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def _attach_vs_hand(
    df_to_modify: pd.DataFrame,
    team_col_name_in_df: str,
    opp_hand_col_name_in_df: str,
    hist_lookup_df: pd.DataFrame,
    new_col_name_base: str,
    hist_stat_vs_lhp: str,
    hist_stat_vs_rhp: str,
    defaults_key: str,
    fallback_value: float,
    flag_imputations: bool,
    **_ignore_extra: Any,
) -> pd.DataFrame:
    """Attach offence vs. pitcher-hand splits."""
    df = df_to_modify.copy()

    if (
        hist_lookup_df.empty
        or hist_stat_vs_lhp not in hist_lookup_df.columns
        or hist_stat_vs_rhp not in hist_lookup_df.columns
    ):
        default_val = MLB_DEFAULTS.get(defaults_key, fallback_value)
        df[new_col_name_base] = default_val
        if flag_imputations:
            df[f"{new_col_name_base}_imputed"] = 1
        return df

    df["_merge_key"] = df[team_col_name_in_df].apply(_safe_norm)
    df["_opp_hand"] = (
        df[opp_hand_col_name_in_df]
        .astype(str)
        .str.strip()
        .str.upper()
        .fillna("U")
        .replace({"": "U", "NONE": "U", "NULL": "U"})
        .str[0]
    )

    hist = (
        hist_lookup_df[[hist_stat_vs_lhp, hist_stat_vs_rhp, "team_norm"]]
        .drop_duplicates("team_norm")
        .set_index("team_norm")
    )

    df = df.merge(hist, left_on="_merge_key", right_index=True, how="left")

    lhp_vals = pd.to_numeric(df[hist_stat_vs_lhp], errors="coerce")
    rhp_vals = pd.to_numeric(df[hist_stat_vs_rhp], errors="coerce")

    selected = np.where(
        df["_opp_hand"] == "L", lhp_vals,
        np.where(df["_opp_hand"] == "R", rhp_vals, np.nan)
    )

    default_val = MLB_DEFAULTS.get(defaults_key, fallback_value)
    df[new_col_name_base] = np.where(np.isnan(selected), default_val, selected)
    if flag_imputations:
        df[f"{new_col_name_base}_imputed"] = np.isnan(selected).astype(int)

    return df.drop(columns=["_merge_key", "_opp_hand", hist_stat_vs_lhp, hist_stat_vs_rhp], errors="ignore")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform(
    df: pd.DataFrame,
    mlb_pitcher_splits_df: pd.DataFrame | None = None,
    *,
    season_to_lookup: int | None = None,
    home_team_col_param: str = "home_team_id",
    away_team_col_param: str = "away_team_id",
    home_pitcher_hand_col: str = "home_probable_pitcher_handedness",
    away_pitcher_hand_col: str = "away_probable_pitcher_handedness",
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Add offence vs. pitcher-hand splits to DataFrame."""
    if mlb_pitcher_splits_df is None or mlb_pitcher_splits_df.empty:
        fallback = MLB_DEFAULTS.get("mlb_avg_runs_vs_hand", 4.3)
        out = df.copy()
        for base in ("h_team_off_avg_runs_vs_opp_hand", "a_team_off_avg_runs_vs_opp_hand"):
            out[base] = fallback
            if flag_imputations:
                out[f"{base}_imputed"] = 1
        return out

    lookup = mlb_pitcher_splits_df.copy()
    if season_to_lookup is not None and "season" in lookup.columns:
        lookup = lookup[lookup["season"] == season_to_lookup].copy()
    if "team_norm" not in lookup.columns:
        key = "team_id" if "team_id" in lookup.columns else (
            "team_name" if "team_name" in lookup.columns else None)
        if key:
            lookup["team_norm"] = lookup[key].apply(_safe_norm)
        else:
            lookup["team_norm"] = lookup.index.to_series().apply(_safe_norm)

    fallback = MLB_DEFAULTS.get("mlb_avg_runs_vs_hand", 4.3)

    out = _attach_vs_hand(
        df_to_modify=df,
        team_col_name_in_df=home_team_col_param,
        opp_hand_col_name_in_df=away_pitcher_hand_col,
        hist_lookup_df=lookup,
        new_col_name_base="h_team_off_avg_runs_vs_opp_hand",
        hist_stat_vs_lhp="season_avg_runs_vs_lhp",
        hist_stat_vs_rhp="season_avg_runs_vs_rhp",
        defaults_key="mlb_avg_runs_vs_hand",
        fallback_value=fallback,
        flag_imputations=flag_imputations,
    )
    out = _attach_vs_hand(
        df_to_modify=out,
        team_col_name_in_df=away_team_col_param,
        opp_hand_col_name_in_df=home_pitcher_hand_col,
        hist_lookup_df=lookup,
        new_col_name_base="a_team_off_avg_runs_vs_opp_hand",
        hist_stat_vs_lhp="season_avg_runs_vs_lhp",
        hist_stat_vs_rhp="season_avg_runs_vs_rhp",
        defaults_key="mlb_avg_runs_vs_hand",
        fallback_value=fallback,
        flag_imputations=flag_imputations,
    )
    return out