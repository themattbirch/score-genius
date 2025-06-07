# backend/nba_features/season.py

"""nba_features.season
---------------------------------
Attach **previous‑season summary stats** (win‑%, average points for/against, etc.)
for both home and away teams to every game row.

Key upgrades
~~~~~~~~~~~~
1.  **Season‑key normalisation** – `_norm_season_key()` converts any season
    representation ("2022‑23", "2022/23", 2022.0 …) to the canonical `int`
    ``2022`` so merges never fail due to format drift.
2.  **Single‑pass merge** – prepares a keyed lookup table and joins once per
    side; no per‑row `.loc` access.
3.  **Granular imputation flags** – every filled‑default cell gets a
    ``*_imputed`` boolean for quick quality scans.
4.  **Slimmer memory footprint** – only the columns required for joining and
    modelling are materialised.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULTS, determine_season, normalize_team_name

logger = logging.getLogger(__name__)
__all__: Sequence[str] = ["transform"]

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
SEASON_STATS_SRC_COLS = (
    "wins_all_percentage",
    "points_for_avg_all",
    "points_against_avg_all",
    "current_form",
)


# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def _norm_season_key(val: Any) -> Optional[int]:
    """Convert season strings / floats to the season *start* year ``int``.

    Examples
    --------
    * ``"2022-23"`` ➜ ``2022``
    * ``2022.0``     ➜ ``2022``
    * ``np.nan``     ➜ ``np.nan``
    """
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

def _fill_and_flag(df: pd.DataFrame, flag_imputations: bool) -> None:
    """Coerce numeric, fill NaNs with defaults, and set imputation flags."""
    # Define columns and their defaults
    stat_map = {
        "season_win_pct": DEFAULTS.get("win_pct", 0.5),
        "season_avg_pts_for": DEFAULTS.get("avg_pts_for", 110.0),
        "season_avg_pts_against": DEFAULTS.get("avg_pts_against", 110.0),
        "current_form": DEFAULTS.get("current_form", "N/A"),
    }

    for side in ("home", "away"):
        for base_col, default_val in stat_map.items():
            col = f"{side}_{base_col}"
            imp_col = f"{col}_imputed"

            if col not in df.columns:
                df[col] = default_val
                if flag_imputations:
                    df[imp_col] = True
            else:
                # Use the pre-calculated imputation flag if it exists
                if flag_imputations and imp_col not in df.columns:
                     df[imp_col] = df[col].isna()

                if isinstance(default_val, str):
                    df[col] = df[col].fillna(default_val)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)

            # Final type enforcement
            df[col] = df[col].astype(str if isinstance(default_val, str) else float)
            if flag_imputations:
                df[imp_col] = df[imp_col].astype(bool)

def _previous_season_int(season_int: Optional[int]) -> Optional[int]:
    """Return the previous season's *start* year (int)."""
    if season_int is None or pd.isna(season_int):
        return np.nan
    return int(season_int) - 1


# ────────────────────────────────────────────────────────────────────────────────
# Main transform
# ────────────────────────────────────────────────────────────────────────────────

def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Attach previous‑season context features to each game row."""

    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    out = df.copy()

    # Sanity‑check essential columns
    required_cols = {"game_date", "home_team", "away_team"}
    missing = required_cols - set(out.columns)
    if missing:
        logger.error("SEASON.PY: Missing required columns %s – returning original df.", missing)
        if debug:
            logger.setLevel(orig_level)
        return df

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out.dropna(subset=["game_date"], inplace=True)
    if out.empty:
        logger.warning("SEASON.PY: All rows dropped after invalid dates.")
        if debug:
            logger.setLevel(orig_level)
        return out

    # ─── Create season keys ─────────────────────────────────────────────────
    out["season_int"] = out["game_date"].apply(determine_season).apply(_norm_season_key)
    out["lookup_season_int"] = out["season_int"].apply(_previous_season_int)

    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)

    # ─── Prepare team‑stats lookup table ────────────────────────────────────
    if team_stats_df is None or team_stats_df.empty:
        logger.warning("SEASON.PY: team_stats_df empty – everything will be imputed.")
        _impute_all(out, flag_imputations)
        _log_imputation(out)
        if debug:
            logger.setLevel(orig_level)
        return out

    ts = team_stats_df.copy()
    ts["team_norm"] = ts["team_name"].map(normalize_team_name)
    ts["season_norm"] = ts["season"].apply(_norm_season_key)
    ts.dropna(subset=["team_norm", "season_norm"], inplace=True)

    # keep only last record per (team, season) pair
    ts.sort_values("season_norm", inplace=True)
    ts = ts.drop_duplicates(subset=["team_norm", "season_norm"], keep="last")

    ts = ts.set_index(["team_norm", "season_norm"])
    ts = ts[list(SEASON_STATS_SRC_COLS)]  # slim to required cols

    # ─── Merge helper (closure) ─────────────────────────────────────────────
    def _merge(side: str) -> None:
        nonlocal out  # Tell the inner function to modify the 'out' from the outer scope
        prefix = f"{side}_"
        lookup_key_cols = [f"{side}_norm", "lookup_season_int"]
        sub = ts.copy().rename(
            columns={
                "wins_all_percentage": f"{prefix}season_win_pct",
                "points_for_avg_all": f"{prefix}season_avg_pts_for",
                "points_against_avg_all": f"{prefix}season_avg_pts_against",
                "current_form": f"{prefix}current_form",
            }
        )
        # Re-assign the result of the merge back to the 'out' DataFrame
        out = out.merge(
            sub,
            how="left",
            left_on=lookup_key_cols,
            right_index=True,
        )
        if flag_imputations:
            for col in [
                f"{prefix}season_win_pct",
                f"{prefix}season_avg_pts_for",
                f"{prefix}season_avg_pts_against",
                f"{prefix}current_form",
            ]:
                if col in out.columns:
                    out[f"{col}_imputed"] = out[col].isna()

    _merge("home")
    _merge("away")

    # ─── Fill defaults & enforce dtypes ─────────────────────────────────────
    _fill_and_flag(out, flag_imputations)

    # ─── Derived metrics ────────────────────────────────────────────────────
    out["season_win_pct_diff"] = out["home_season_win_pct"] - out["away_season_win_pct"]
    out["season_pts_for_diff"] = out["home_season_avg_pts_for"] - out["away_season_avg_pts_for"]
    out["season_pts_against_diff"] = (
        out["home_season_avg_pts_against"] - out["away_season_avg_pts_against"]
    )

    out["home_season_net_rating"] = (
        out["home_season_avg_pts_for"] - out["home_season_avg_pts_against"]
    )
    out["away_season_net_rating"] = (
        out["away_season_avg_pts_for"] - out["away_season_avg_pts_against"]
    )
    out["season_net_rating_diff"] = out["home_season_net_rating"] - out["away_season_net_rating"]

    # Ensure numeric dtype
    num_cols = [
        "season_win_pct_diff",
        "season_pts_for_diff",
        "season_pts_against_diff",
        "home_season_net_rating",
        "away_season_net_rating",
        "season_net_rating_diff",
    ]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    # ─── Diagnostics ────────────────────────────────────────────────────────
    _log_imputation(out)

    # ─── Cleanup helper columns ─────────────────────────────────────────────
    out.drop(
        columns=[c for c in ("season_int", "lookup_season_int", "home_norm", "away_norm") if c in out.columns],
        inplace=True,
        errors="ignore",
    )

    if debug:
        logger.setLevel(orig_level)
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Utility sub‑routines
# ────────────────────────────────────────────────────────────────────────────────

def _impute_all(df: pd.DataFrame, flag: bool) -> None:
    """Fill every expected column with defaults (no stats available)."""
    defaults_map = {
        "home_season_win_pct": DEFAULTS.get("win_pct", 0.5),
        "away_season_win_pct": DEFAULTS.get("win_pct", 0.5),
        "home_season_avg_pts_for": DEFAULTS.get("avg_pts_for", 110.0),
        "away_season_avg_pts_for": DEFAULTS.get("avg_pts_for", 110.0),
        "home_season_avg_pts_against": DEFAULTS.get("avg_pts_against", 110.0),
        "away_season_avg_pts_against": DEFAULTS.get("avg_pts_against", 110.0),
        "home_current_form": DEFAULTS.get("current_form", "N/A"),
        "away_current_form": DEFAULTS.get("current_form", "N/A"),
    }
    for col, val in defaults_map.items():
        df[col] = val
        if flag:
            df[f"{col}_imputed"] = True

    # zeros for derived metrics
    zeros = [
        "season_win_pct_diff",
        "season_pts_for_diff",
        "season_pts_against_diff",
        "home_season_net_rating",
        "away_season_net_rating",
        "season_net_rating_diff",
    ]
    for z in zeros:
        df[z] = 0.0


def _log_imputation(df: pd.DataFrame) -> None:
    if "home_season_win_pct_imputed" not in df.columns:
        return
    total = len(df)
    imputed = df["home_season_win_pct_imputed"].sum()
    pct = 100.0 * imputed / total if total else 0
    logger.info(
        "SEASON.PY: %s/%s games (%.1f%%) used default season stats.", imputed, total, pct
    )
    if pct > 20.0:
        logger.warning("SEASON.PY: High imputation rate – verify season keys or data coverage.")
