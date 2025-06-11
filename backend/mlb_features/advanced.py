# backend/mlb_features/advanced.py

from __future__ import annotations
import logging
from typing import Any, Dict as TypingDict, Tuple

import numpy as np
import pandas as pd

from . import handedness_for_display      # <- keeps vs-hand logic isolated

logger = logging.getLogger(__name__)

try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name
except ImportError:               # unit-test context / isolated run
    MLB_DEFAULTS: dict = {}
    def normalize_team_name(v: Any) -> str:           # noqa: N802
        return str(v).strip().lower() if pd.notna(v) else "Unknown"


# --------------------------------------------------------------------------- #
#  Public globals – the test-suite references these attributes directly
# --------------------------------------------------------------------------- #
home_team_col: str = "home_team_id"
away_team_col: str = "away_team_id"


# --------------------------------------------------------------------------- #
#  Shared normaliser – identical copy lives in handedness_for_display
# --------------------------------------------------------------------------- #
def _safe_norm(val: Any) -> str:
    """
    Turn any team identifier into a compact merge key:

    1. Try `normalize_team_name`; if it returns something OTHER than an
       'unknown' sentinel, use that.
    2. Otherwise fall back to the raw value, stripped/lower-cased with
       spaces removed.  NaN → 'unknown'.
    """
    if pd.isna(val):
        return "unknown"

    mapped = normalize_team_name(val)
    if isinstance(mapped, str) and not mapped.lower().startswith("unknown"):
        base = mapped
    else:
        base = str(val)

    return base.strip().lower().replace(" ", "")


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _attach_split(
    df_to_mod: pd.DataFrame,
    team_col: str,
    lookup_df: pd.DataFrame,
    mappings: TypingDict[str, Tuple[str, str, float]],
    flag_imp: bool,
) -> pd.DataFrame:
    """
    Add home/away historical splits (win-%, runs for/against) to *df_to_mod*.
    `mappings` maps new-col → (hist-col, defaults-key, fallback).
    """
    if team_col not in df_to_mod.columns:
        out = df_to_mod.copy()
        for new_c, (_, k, fb) in mappings.items():
            out[new_c] = MLB_DEFAULTS.get(k, fb)
            if flag_imp:
                out[f"{new_c}_imputed"] = 1
        return out

    df = df_to_mod.copy()
    df["_merge_key"] = df[team_col].apply(_safe_norm)

    if lookup_df.empty:
        for new_c, (_, k, fb) in mappings.items():
            df[new_c] = MLB_DEFAULTS.get(k, fb)
            if flag_imp:
                df[f"{new_c}_imputed"] = 1
        return df.drop(columns=["_merge_key"])

    needed = {h for h, _, _ in mappings.values()}
    avail  = [c for c in needed if c in lookup_df.columns]

    hist = (lookup_df[avail + ["team_norm"]]
            .drop_duplicates("team_norm")
            .set_index("team_norm"))

    df = df.merge(hist, left_on="_merge_key", right_index=True, how="left")

    for new_c, (hist_c, k, fb) in mappings.items():
        default_val = MLB_DEFAULTS.get(k, fb)
        if hist_c in df.columns:
            df[new_c] = df[hist_c].fillna(default_val)
            if flag_imp:
                df[f"{new_c}_imputed"] = df[hist_c].isna().astype(int)
        else:
            df[new_c] = default_val
            if flag_imp:
                df[f"{new_c}_imputed"] = 1

    return df.drop(columns=["_merge_key"] + list(needed & set(df.columns)), errors="ignore")


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def transform(
    df: pd.DataFrame,
    historical_team_stats_df: pd.DataFrame,
    *,
    flag_imputations: bool = True,
    season_to_lookup: int | None = None,
    home_team_col_param: str = "home_team_id",
    away_team_col_param: str = "away_team_id",
    home_hand_col_param: str = "home_probable_pitcher_handedness",
    away_hand_col_param: str = "away_probable_pitcher_handedness",
) -> pd.DataFrame:
    """
    Build advanced features:
    ▸ Home/Away historical splits  
    ▸ Offence vs opponent pitcher handedness
    """
    logger.info("ENGINE: advanced.transform starting (debug mode)")

    # short-circuit on empty input
    if df.empty:
        return df.copy()

    # bind globals for the unit-tests
    global home_team_col, away_team_col
    home_team_col = home_team_col_param
    away_team_col = away_team_col_param

    # runtime defaults
    def_win = MLB_DEFAULTS.get("mlb_hist_win_pct", 0.50)
    def_run = MLB_DEFAULTS.get("mlb_hist_runs_avg", 4.50)

    out = df.copy()

    # ------------------------------------------------------------------
    # Build & normalize lookup table
    # ------------------------------------------------------------------
    if historical_team_stats_df is not None and not historical_team_stats_df.empty:
        lookup = historical_team_stats_df.copy()
        if season_to_lookup is not None and "season" in lookup.columns:
            lookup = lookup[lookup["season"] == season_to_lookup].copy()
        if "team_norm" not in lookup.columns:
            key_col = "team_id" if "team_id" in lookup.columns else (
                "team_name" if "team_name" in lookup.columns else None
            )
            if key_col:
                lookup["team_norm"] = lookup[key_col].apply(_safe_norm)
            else:
                lookup["team_norm"] = lookup.index.to_series().apply(_safe_norm)
    else:
        lookup = pd.DataFrame()

    # ------------------------------------------------------------------
    # Home/Away split features
    # ------------------------------------------------------------------
    split_map = {
        "h_team_hist_HA_win_pct":        ("wins_home_percentage",    "mlb_hist_win_pct", def_win),
        "h_team_hist_HA_runs_for_avg":   ("runs_for_avg_home",       "mlb_hist_runs_avg", def_run),
        "h_team_hist_HA_runs_against_avg": ("runs_against_avg_home","mlb_hist_runs_avg", def_run),
        "a_team_hist_HA_win_pct":        ("wins_away_percentage",    "mlb_hist_win_pct", def_win),
        "a_team_hist_HA_runs_for_avg":   ("runs_for_avg_away",       "mlb_hist_runs_avg", def_run),
        "a_team_hist_HA_runs_against_avg": ("runs_against_avg_away","mlb_hist_runs_avg", def_run),
    }
    out = _attach_split(
        out, home_team_col_param, lookup,
        {k: v for k, v in split_map.items() if k.startswith("h_")},
        flag_imputations
    )
    out = _attach_split(
        out, away_team_col_param, lookup,
        {k: v for k, v in split_map.items() if k.startswith("a_")},
        flag_imputations
    )

    # ensure the handedness module uses the same MLB_DEFAULTS dict
    handedness_for_display.MLB_DEFAULTS = MLB_DEFAULTS

    # ------------------------------------------------------------------
    # Offence vs opponent pitcher handedness (delegated)
    # ------------------------------------------------------------------
    out = handedness_for_display.transform(
        df                    = out,
        mlb_pitcher_splits_df = lookup,
        season_to_lookup      = season_to_lookup,
        home_team_col_param   = home_team_col_param,
        away_team_col_param   = away_team_col_param,
        home_pitcher_hand_col = home_hand_col_param,
        away_pitcher_hand_col = away_hand_col_param,
        flag_imputations      = flag_imputations,
    )

    # ------------------------------------------------------------------
    # Final cleanup – drop raw IDs / handedness cols
    # ------------------------------------------------------------------
    drop_cols = [
        home_team_col_param,
        away_team_col_param,
        home_hand_col_param,
        away_hand_col_param,
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    return out