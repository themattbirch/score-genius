# backend/nba_features/advanced.py
"""
Attach prior‑season advanced splits to each game row.

Key improvements over the earlier version
-----------------------------------------
1.  **Season‑key normalisation** – `_norm_season_key()` converts every season
    representation ("2022‑23", "2022/23", 2022.0 …) to the canonical start
    year `int(2022)`.  Both source and target DataFrames are normalised so
    merges succeed even when formats differ.
2.  **Guard‑rails & logging** – early warnings if merge hit‑rate is low;
    imputation percentages are logged.
3.  **Memory‑friendly merges** – only the columns required for joining are
    carried through, reducing copy overhead.

The function still returns a fully numeric, type‑stable `DataFrame` ready for
modelling and leaves `adv_stats_lookup_season` intact for downstream modules.
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

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _norm_season_key(val: Any) -> Optional[int]:
    """Return the *start* year of a season as an ``int``.

    Handles common formats:
    * ``"2022-23"`` / ``"2022/23"`` / ``"2022‑2023"`` ➜ ``2022``
    * ``2022`` / ``2022.0`` ➜ ``2022``
    * unparsable / NA ➜ ``np.nan``
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        match = re.search(r"\d{4}", val)
        if match:
            return int(match.group(0))
    try:
        return int(float(val))
    except Exception:
        return np.nan


# ────────────────────────────────────────────────────────────────────────────────
# Main transform
# ────────────────────────────────────────────────────────────────────────────────

@profile_time
def transform(
    df: pd.DataFrame,
    *,
    all_historical_splits_df: pd.DataFrame,
    flag_imputations: bool = True,
    debug: bool = False,  # kept for interface completeness
) -> pd.DataFrame:
    """Attach prior‑season advanced splits and compute diffs.

    Parameters
    ----------
    df
        Must contain ``home_team``, ``away_team`` and
        ``adv_stats_lookup_season`` columns.
    all_historical_splits_df
        Multi‑season splits with columns ``season`` and either ``team_norm``
        or ``team_name`` plus ``*_home`` / ``*_away`` stats.
    """

    out = df.copy()

    # ─── Preconditions ───────────────────────────────────────────────────────
    if "adv_stats_lookup_season" not in out.columns:
        logger.error("'adv_stats_lookup_season' column missing – skipping advanced split attachment.")
        return out

    # Team normalisation (once only)
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)

    # Season‑key normalisation for *games*
    out["lookup_season_norm"] = out["adv_stats_lookup_season"].apply(_norm_season_key)

    # ─── Prepare historical splits ───────────────────────────────────────────
    if all_historical_splits_df.empty:
        logger.warning("Received empty historical splits DF – all values will be default‑imputed.")
        seasonal_stats_processed = pd.DataFrame()
    else:
        seasonal_stats_processed = all_historical_splits_df.copy()

        if "team_norm" not in seasonal_stats_processed.columns:
            if "team_name" in seasonal_stats_processed.columns:
                seasonal_stats_processed["team_norm"] = seasonal_stats_processed["team_name"].map(
                    normalize_team_name
                )
            else:
                logger.error("Historical splits DF missing 'team_norm' and 'team_name'.")
                seasonal_stats_processed = pd.DataFrame()

        if "season" not in seasonal_stats_processed.columns:
            logger.error("Historical splits DF missing 'season'.")
            seasonal_stats_processed = pd.DataFrame()

    # If still no usable data, fall back to imputation path
    if seasonal_stats_processed.empty:
        _add_default_columns(out, flag_imputations)
        _log_imputation_rate(out)
        return out

    # Season‑key normalisation for *stats*
    seasonal_stats_processed["season_norm"] = seasonal_stats_processed["season"].apply(_norm_season_key)
    seasonal_stats_processed.dropna(subset=["season_norm"], inplace=True)
    seasonal_stats_processed["season_norm"] = seasonal_stats_processed["season_norm"].astype(int)

    # ─── Merge HOME splits ───────────────────────────────────────────────────
    home_stats = seasonal_stats_processed[[
        "team_norm",
        "season_norm",
        *[f"{s}_home" for s in EXPECTED_STATS],
    ]].rename(columns={f"{s}_home": f"h_{s}_home" for s in EXPECTED_STATS})

    out = out.merge(
        home_stats,
        left_on=["home_norm", "lookup_season_norm"],
        right_on=["team_norm", "season_norm"],
        how="left",
        suffixes=("", "_dup"),
    ).drop(columns=[c for c in ("team_norm", "season_norm") if c in out.columns])

    # ─── Merge AWAY splits ───────────────────────────────────────────────────
    away_stats = seasonal_stats_processed[[
        "team_norm",
        "season_norm",
        *[f"{s}_away" for s in EXPECTED_STATS],
    ]].rename(columns={f"{s}_away": f"a_{s}_away" for s in EXPECTED_STATS})

    out = out.merge(
        away_stats,
        left_on=["away_norm", "lookup_season_norm"],
        right_on=["team_norm", "season_norm"],
        how="left",
        suffixes=("", "_dup"),
    ).drop(columns=[c for c in ("team_norm", "season_norm") if c in out.columns])

    # ─── Imputation & type enforcement ───────────────────────────────────────
    _fill_and_flag(out, flag_imputations)

    # ─── Diff columns ────────────────────────────────────────────────────────
    for stat in EXPECTED_STATS:
        out[f"hist_{stat}_split_diff"] = out[f"h_{stat}_home"] - out[f"a_{stat}_away"]

    # ─── Rating mirrors (for legacy downstream code) ─────────────────────────
    ratings_map = {
        "home_offensive_rating": "h_off_rtg_home",
        "away_offensive_rating": "a_off_rtg_away",
        "home_defensive_rating": "h_def_rtg_home",
        "away_defensive_rating": "a_def_rtg_away",
        "home_net_rating": "h_net_rtg_home",
        "away_net_rating": "a_net_rtg_away",
    }
    for new_col, src_col in ratings_map.items():
        out[new_col] = out[src_col]

    # ─── Diagnostics ─────────────────────────────────────────────────────────
    _log_imputation_rate(out)

    # ─── Cleanup helper columns ──────────────────────────────────────────────
    out.drop(columns=[c for c in ("home_norm", "away_norm", "lookup_season_norm") if c in out.columns],
             inplace=True, errors="ignore")

    return out


# ────────────────────────────────────────────────────────────────────────────────
# Utility sub‑routines (kept outside transform for readability)
# ────────────────────────────────────────────────────────────────────────────────

def _add_default_columns(df: pd.DataFrame, flag_imputations: bool) -> None:
    """Create expected columns filled with defaults when no stats are available."""
    for side_label, prefix in (("home", "h"), ("away", "a")):
        for stat in EXPECTED_STATS:
            tgt = f"{prefix}_{stat}_{side_label}"
            df[tgt] = DEFAULTS.get(stat, 0.0)
            if flag_imputations:
                df[f"{tgt}_imputed"] = True


def _fill_and_flag(df: pd.DataFrame, flag_imputations: bool) -> None:
    """Coerce numeric, fill NaNs with defaults, set *_imputed flags."""
    for side_label, prefix in (("home", "h"), ("away", "a")):
        for stat in EXPECTED_STATS:
            col = f"{prefix}_{stat}_{side_label}"
            imp_col = f"{col}_imputed"
            default = DEFAULTS.get(stat, 0.0)

            if col not in df.columns:
                df[col] = default
                if flag_imputations:
                    df[imp_col] = True
            else:
                if flag_imputations:
                    df[imp_col] = pd.to_numeric(df[col], errors="coerce").isna()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
            if flag_imputations and imp_col not in df.columns:
                df[imp_col] = False
            if flag_imputations:
                df[imp_col] = df[imp_col].astype(bool)


def _log_imputation_rate(df: pd.DataFrame) -> None:
    """Emit a summary of imputation rates for quick diagnostics."""
    if "h_off_rtg_home_imputed" not in df.columns:
        return

    total = len(df)
    home_imputed = df["h_off_rtg_home_imputed"].sum()
    away_imputed = df["a_off_rtg_away_imputed"].sum()

    home_pct = 100.0 * home_imputed / total if total else 0
    away_pct = 100.0 * away_imputed / total if total else 0

    logger.info(
        "ADVANCED.PY: Home stats imputed for %s/%s games (%.1f%%).", home_imputed, total, home_pct
    )
    logger.info(
        "ADVANCED.PY: Away stats imputed for %s/%s games (%.1f%%).", away_imputed, total, away_pct
    )
    if max(home_pct, away_pct) > 20.0:
        logger.warning("ADVANCED.PY: High imputation rate detected – check season key normalisation or data coverage.")
