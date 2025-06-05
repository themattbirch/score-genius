# backend/mlb_features/rolling.py

"""
Calculates leakage‑free rolling mean and standard deviation features
for MLB games based on team‑level game statistics.

Key points
-----------
* No‑look‑ahead — each game’s features use only *previous* games.
* Handles multiple games by the same team on a single date (dupes).
* Optional imputation (and flags) with sensible league defaults.
* Back‑compat: accepts either ``flag_imputations`` **or** the older
  ``flag_imputation`` kwarg used in some legacy code/tests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ───────────────────── Logger ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────── Defaults & helpers ─────────────
try:
    from .utils import DEFAULTS, normalize_team_name
except ImportError:  # fallback for tests / standalone runs
    DEFAULTS: Dict[str, Any] = {}
    def normalize_team_name(x: Any) -> str:  # type: ignore[override]
        return str(x).strip().lower() if pd.notna(x) else "unknown"

# Base column names
_HS, _AS = "home_score", "away_score"
_HH, _AH = "home_hits", "away_hits"
_HE, _AE = "home_errors", "away_errors"

# Map generic stat → (home_raw_col, away_raw_col)
stat_map: Dict[str, tuple[str, str]] = {
    "runs_scored":       (_HS, _AS),
    "runs_allowed":      (_AS, _HS),
    "hits_for":          (_HH, _AH),
    "hits_allowed":      (_AH, _HH),
    "errors_committed":  (_HE, _AE),
    "errors_by_opponent":(_AE, _HE),
}

# ────────────────── Core rolling helper ─────────────

def _lagged_rolling_stat(s: pd.Series, window: int, min_periods: int, stat: str) -> pd.Series:
    """Leakage‑free rolling mean/std with same‑day duplicate exclusion."""
    if s.empty:
        return pd.Series([], dtype=float, index=s.index)

    shifted = s.shift(1)
    dates = pd.Series(s.index, index=s.index)
    shifted.loc[dates == dates.shift(1)] = np.nan  # remove previous same‑day value

    if stat == "mean":
        primary = shifted.rolling(window, min_periods=min_periods).mean()
        fallback = shifted.rolling(window, min_periods=1).mean()
    elif stat == "std":
        primary = shifted.rolling(window, min_periods=min_periods).std()
        fallback = shifted.rolling(window, min_periods=1).std()
    else:
        raise ValueError(stat)

    return primary.fillna(fallback)

# ──────────────── Public transform ─────────────────

def transform(
    df: pd.DataFrame,
    window_sizes: List[int] | tuple[int, ...] = (5,), # Using (5,) as a more typical default
    *,
    flag_imputations: bool = True,
    flag_imputation: Optional[bool] = None,  # legacy alias
    debug: bool = False,
) -> pd.DataFrame:
    """Attach rolling features."""
    if flag_imputation is not None:
        flag_imputations = flag_imputation

    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    if df is None or df.empty:
        logger.warning("Empty input to rolling.transform; returning copy.")
        logger.setLevel(orig_level) # Ensure logger level is reset
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    required_outcome_cols = [_HS, _AS, _HH, _AH, _HE, _AE]
    for col in required_outcome_cols:
        if col not in out.columns:
            logger.info(f"Rolling.transform: Input df missing '{col}'. Adding as NaN placeholder.")
            out[col] = np.nan
    # --- MODIFICATION END ---

    # Validate mandatory columns (now outcome cols should exist, possibly as NaN)
    req = {"game_id", "game_date_et", "home_team_id", "away_team_id", _HS, _AS, _HH, _AH, _HE, _AE}
    missing = req.difference(out.columns)
    if missing:
        # This error should ideally not be hit now for the outcome columns.
        # If it is, it means other fundamental columns like 'game_id' are missing from input.
        logger.error(f"Rolling.transform: Missing required columns: {sorted(missing)}")
        logger.setLevel(orig_level)
        return df # Return original df on error

    # Clean/parse dates
    out["_gdate"] = pd.to_datetime(out["game_date_et"], errors="coerce").dt.tz_localize(None)
    out.dropna(subset=["_gdate"], inplace=True)
    if out.empty:
        logger.warning("Rolling.transform: All rows dropped due to bad dates in 'game_date_et'.")
        # out.drop(columns=["_gdate"], inplace=True, errors="ignore") # _gdate already dropped implicitly if out is empty
        logger.setLevel(orig_level)
        return pd.DataFrame(columns=df.columns) # Return empty df with original columns

    # Normalise team ids
    out["home_norm"] = out["home_team_id"].apply(normalize_team_name)
    out["away_norm"] = out["away_team_id"].apply(normalize_team_name)

    # Build long table (team, stat, value)
    recs: list[dict[str, Any]] = []
    for _, r in out.iterrows(): # 'out' now has _HS, _AS etc. (as NaN for upcoming games)
        for side, team_norm_col in (("home", "home_norm"), ("away", "away_norm")):
            tnorm = r[team_norm_col]
            for gstat, (hcol, acol) in stat_map.items():
                raw_value = r[hcol] if side == "home" else r[acol]
                # pd.to_numeric(np.nan, errors="coerce") is np.nan
                value = pd.to_numeric(raw_value, errors="coerce") 
                recs.append({
                    "game_id": r["game_id"],
                    "team_norm": tnorm,
                    "game_date": r["_gdate"], # Using the cleaned _gdate
                    "stat": gstat,
                    "value": value,
                })
    
    long_df = pd.DataFrame.from_records(recs)
    # This .dropna will remove all rows corresponding to upcoming_df games
    # if their 'value' (from scores, hits, errors) became NaN.
    long_df.dropna(subset=["value"], inplace=True) 
    
    if long_df.empty:
        logger.warning("Rolling.transform: No valid stat rows after attempting to build long_df (likely due to input df having no historical outcomes or all NaNs for outcome columns). Returning input df without rolling features.")
        out.drop(columns=["_gdate", "home_norm", "away_norm"], inplace=True, errors="ignore")
        logger.setLevel(orig_level)
        return out # Return 'out' which is df + potentially NaN outcome cols + no new rolling features

    # Sort for correct rolling calculation (critical for shift logic)
    long_df.sort_values(["team_norm", "stat", "game_date", "game_id"], inplace=True, kind="mergesort", ignore_index=True)

    # Compute rolling stats
    for w in window_sizes:
        min_p = max(1, w // 2)
        # Groupby and transform using _lagged_rolling_stat
        # Ensure 'observed=True' if pandas version supports/requires for new behavior with categoricals
        long_df[f"mean_{w}"] = long_df.groupby(["team_norm", "stat"], observed=True)["value"].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, "mean"))
        long_df[f"std_{w}"] = long_df.groupby(["team_norm", "stat"], observed=True)["value"].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, "std"))

        if flag_imputations:
            for typ in ("mean", "std"):
                col = f"{typ}_{w}"
                imp_col = f"{col}_imputed"
                long_df[imp_col] = long_df[col].isna()
                def _fill(r_long_df_row): # Renamed variable for clarity
                    # Ensure 'stat' column is used for lookup as in original
                    key = r_long_df_row["stat"] if typ == "mean" else f"{r_long_df_row['stat']}_std"
                    default_val = DEFAULTS.get(key, 0.0) # Use a sensible default, 0.0 is common
                    # Make sure r_long_df_row[col] is the value from the current row being applied
                    return r_long_df_row[col] if pd.notna(r_long_df_row[col]) else default_val
                long_df[col] = long_df.apply(_fill, axis=1)


    # Pivot wide per game/team
    pivots: list[pd.DataFrame] = []
    for w in window_sizes:
        cols: list[pd.DataFrame] = []
        for typ in ("mean", "std"):
            if f"{typ}_{w}" in long_df.columns:
                p = long_df.pivot_table(index=["game_id", "team_norm"], columns="stat", values=f"{typ}_{w}")
                p.columns = [f"rolling_{s}_{typ}_{w}" for s in p.columns]
                cols.append(p)
            if flag_imputations and f"{typ}_{w}_imputed" in long_df.columns:
                p = long_df.pivot_table(index=["game_id", "team_norm"], columns="stat", values=f"{typ}_{w}_imputed")
                p.columns = [f"rolling_{s}_{typ}_{w}_imputed" for s in p.columns]
                cols.append(p)
        if cols:
            pivots.append(pd.concat(cols, axis=1))

    if pivots:
        wide = pd.concat(pivots, axis=1).reset_index()
        wide["_mkey"] = wide["game_id"].astype(str) + "_" + wide["team_norm"]
        out["home_mkey"] = out["game_id"].astype(str) + "_" + out["home_norm"]
        out["away_mkey"] = out["game_id"].astype(str) + "_" + out["away_norm"]

        rcols = [c for c in wide.columns if c.startswith("rolling_")]
        if rcols:
            # merge home
            hm = wide[["_mkey"] + rcols].rename(columns={c: f"home_{c}" for c in rcols})
            out = out.merge(hm, how="left", left_on="home_mkey", right_on="_mkey")
            # merge away
            am = wide[["_mkey"] + rcols].rename(columns={c: f"away_{c}" for c in rcols})
            out = out.merge(am, how="left", left_on="away_mkey", right_on="_mkey", suffixes=("", "_dup"))
            out.drop(columns=[c for c in ["_mkey", "_mkey_dup"] if c in out.columns], inplace=True, errors="ignore")
    else:
        logger.warning("No rolling stats generated; proceeding without new columns.")

    # Cleanup temp cols
    out.drop(columns=[c for c in ["home_norm", "away_norm", "home_mkey", "away_mkey", "_gdate"] if c in out.columns], inplace=True, errors="ignore")

    # Cast imputed cols to bool
    for col in [c for c in out.columns if c.endswith("_imputed")]:
        # cast to genuine Python bool (not numpy.bool_) so identity checks in tests pass
        out[col] = pd.Series([bool(v) for v in out[col].to_numpy()], index=out.index, dtype="object")

    if debug:
        logger.setLevel(orig_level)
        logger.debug(f"rolling.transform completed; output shape {out.shape}")

    return out
