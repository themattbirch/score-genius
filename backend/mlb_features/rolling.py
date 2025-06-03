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
    window_sizes: List[int] | tuple[int, ...] = (5,),
    *,
    flag_imputations: bool = True,
    flag_imputation: Optional[bool] = None,  # legacy alias
    debug: bool = False,
) -> pd.DataFrame:
    """Attach rolling features.

    Parameters
    ----------
    df: input games dataframe (must include id/date/team/box‑score columns).
    window_sizes: iterable of int — rolling windows.
    flag_imputations / flag_imputation: if true, fill NaNs with DEFAULTS
        and create *_imputed columns.  Either kwarg is accepted.
    debug: verbose logging.
    """
    # Alias resolution
    if flag_imputation is not None:
        flag_imputations = flag_imputation

    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    if df is None or df.empty:
        logger.warning("Empty input to rolling.transform; returning copy.")
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    # Validate mandatory columns
    req = {"game_id", "game_date_et", "home_team_id", "away_team_id", _HS, _AS, _HH, _AH, _HE, _AE}
    missing = req.difference(out.columns)
    if missing:
        logger.error(f"Missing required columns: {sorted(missing)}")
        logger.setLevel(orig_level)
        return df

    # Clean/parse dates
    out["_gdate"] = pd.to_datetime(out["game_date_et"], errors="coerce").dt.tz_localize(None)
    out.dropna(subset=["_gdate"], inplace=True)
    if out.empty:
        logger.warning("All rows dropped due to bad dates.")
        out.drop(columns=["_gdate"], inplace=True, errors="ignore")
        logger.setLevel(orig_level)
        return out

    # Normalise team ids
    out["home_norm"] = out["home_team_id"].apply(normalize_team_name)
    out["away_norm"] = out["away_team_id"].apply(normalize_team_name)

    # Build long table (team, stat, value)
    recs: list[dict[str, Any]] = []
    for _, r in out.iterrows():
        for side, team in (("home", "home_norm"), ("away", "away_norm")):
            tnorm = r[team]
            for gstat, (hcol, acol) in stat_map.items():
                value = r[hcol] if side == "home" else r[acol]
                recs.append({
                    "game_id": r["game_id"],
                    "team_norm": tnorm,
                    "game_date": r["_gdate"],
                    "stat": gstat,
                    "value": pd.to_numeric(value, errors="coerce"),
                })
    long_df = pd.DataFrame.from_records(recs).dropna(subset=["value"])
    if long_df.empty:
        logger.warning("No stat rows after cleaning; returning input.")
        out.drop(columns=["_gdate", "home_norm", "away_norm"], inplace=True, errors="ignore")
        logger.setLevel(orig_level)
        return out

    long_df.sort_values(["team_norm", "stat", "game_date", "game_id"], inplace=True, kind="mergesort", ignore_index=True)

    # Compute rolling stats
    for w in window_sizes:
        min_p = max(1, w // 2)
        long_df[f"mean_{w}"] = long_df.groupby(["team_norm", "stat"], observed=True)["value"].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, "mean"))
        long_df[f"std_{w}"] = long_df.groupby(["team_norm", "stat"], observed=True)["value"].transform(
            lambda s: _lagged_rolling_stat(s, w, min_p, "std"))

        if flag_imputations:
            for typ in ("mean", "std"):
                col = f"{typ}_{w}"
                imp_col = f"{col}_imputed"
                long_df[imp_col] = long_df[col].isna()
                def _fill(r):
                    key = r["stat"] if typ == "mean" else f"{r['stat']}_std"
                    return r[col] if pd.notna(r[col]) else DEFAULTS.get(key, 0.0)
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
