# ---------------------------------------------------------------------
# backend/nba_features/rolling.py - Using robust helper functions
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

# Assuming utils.py contains DEFAULTS dict and normalize_team_name function
from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
__all__ = ["transform"]

# ------------------------------------------------------------------ #
# Helper Functions for Rolling Calculations
# ------------------------------------------------------------------ #

def _lagged_rolling_stat(
    s: pd.Series,
    w: int,
    min_p: int,
    stat_func: str
) -> pd.Series:
    """
    Leakage‑free rolling statistic (mean or std).

    • Excludes *all* games played on the same calendar date as the current row.  
    • Falls back to the mean/std of the available history (>=1 game) when the
      group has fewer than `min_periods` rows – this covers the common
      “second‑game” edge case without touching `min_periods` itself.
    """
    if s.empty:
        return s

    ordered = s.sort_index()                       # index == game_date
    dates   = pd.Series(ordered.index, index=ordered.index)

    # --- 1. shift(1) to drop the current game ------------------------------
    shifted_vals = ordered.shift(1)

    # --- 2. wipe out any same‑day rows (idx == previous idx) ---------------
    same_day_mask = dates == dates.shift(1)
    shifted_vals[same_day_mask] = np.nan

    # --- 3. rolling calc ---------------------------------------------------
    _roller = shifted_vals.rolling(window=w, min_periods=min_p)
    if stat_func == "mean":
        rolled = _roller.mean()
        fallback = shifted_vals.rolling(window=w, min_periods=1).mean()
    elif stat_func == "std":
        rolled = _roller.std()
        fallback = shifted_vals.rolling(window=w, min_periods=1).std()
    else:
        raise ValueError(f"Unsupported stat_func: {stat_func}")

    # If we failed the min_periods test but still have ≥1 prior game,
    # use the fallback value (covers the “one‑game history” scenario).
    rolled = rolled.where(~rolled.isna(), fallback)

    # Re‑align to the original slice order expected by .transform
    return rolled.reindex(s.index)

# ------------------------------------------------------------------ #
# Main Transform Function
# ------------------------------------------------------------------ #

def _mk(side: str, base: str, kind: str, w: int) -> str:
    """home+away column‑name factory"""
    return f"{side}_rolling_{base}_{kind}_{w}"


def transform(
    df: pd.DataFrame,
    *,
    window_sizes: List[int] = (5, 10, 20),
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add leakage‑free rolling mean/std features for each team & stat using
    robust helper functions.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG) # Ensure logging is configured
        logger.debug(f"Input df shape: {df.shape}")

    if df.empty:
        logger.debug("Input DataFrame is empty, returning copy.")
        return df.copy()

    out = df.copy()

    # --- Data Preparation ---
    # Convert game_date and normalize team names
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.tz_localize(None)
    out["home_norm"] = out["home_team"].map(normalize_team_name)
    out["away_norm"] = out["away_team"].map(normalize_team_name)
    logger.debug("Converted game_date and normalized team names.")

    # --- Create Long Format ---
    mapping = {
        ("home_score", "away_score"): "score_for",
        ("home_offensive_rating", "away_offensive_rating"): "off_rating",
        ("home_defensive_rating", "away_defensive_rating"): "def_rating",
        ("home_net_rating", "away_net_rating"): "net_rating",
    }

    recs: list[dict] = []
    required_cols = {col for pair in mapping for col in pair}
    missing_cols = required_cols - set(out.columns)
    if missing_cols:
        logger.warning(f"Missing required columns for rolling features: {missing_cols}. Skipping.")
        # Return original frame minus added normalization columns if no features can be made
        return out.drop(columns=["home_norm", "away_norm"], errors="ignore")

    logger.debug("Creating long format DataFrame...")
    for (home_col, away_col), generic in mapping.items():
        # Check if specific columns for this stat exist (redundant if checked above, but safe)
        if home_col not in out.columns or away_col not in out.columns:
            logger.warning(f"Skipping stat '{generic}' due to missing columns: {home_col} or {away_col}")
            continue
        # Iterate using vectorized operations where possible, but iterrows is sometimes clearer for complex reshaping
        for idx, r in out.iterrows(): # Using index idx might be useful if needed later
            recs.append(
                {
                    "original_index": idx, # Keep track of original row if needed
                    "game_id": r["game_id"],
                    "game_date": r["game_date"],
                    "team_norm": r["home_norm"],
                    "stat": generic,
                    "val": pd.to_numeric(r[home_col], errors="coerce"),
                }
            )
            recs.append(
                {
                    "original_index": idx,
                    "game_id": r["game_id"],
                    "game_date": r["game_date"],
                    "team_norm": r["away_norm"],
                    "stat": generic,
                    "val": pd.to_numeric(r[away_col], errors="coerce"),
                }
            )

    if not recs:
        logger.warning("No records generated for long format. Check input columns and mapping.")
        return out.drop(columns=["home_norm", "away_norm"], errors="ignore")

    # Create long_df and set game_date as index *temporarily* for sorting robustness in helper
    long_df = pd.DataFrame.from_records(recs)
    long_df = long_df.set_index('game_date') # Use game_date as index for sort_index() in helper
    long_df = long_df.sort_values(["team_norm", "stat", "game_date"], kind="mergesort") # Sort primarily by team/stat, then date index
    logger.debug(f"Long format df created. Shape: {long_df.shape}. Index: {long_df.index.name}")


    # --- Calculate Rolling Features ---
    pieces = []
    logger.debug(f"Calculating rolling features for window sizes: {window_sizes}")
    for w in window_sizes:
        min_p = max(1, w // 2)
        logger.debug(f"Processing window size: {w}, min_periods: {min_p}")

        # Group by team and stat. Ensure group_keys=False for transform compatibility.
        # The pre-sorting ensures shift(1) works correctly within each group.
        grp = long_df.groupby(["team_norm", "stat"], group_keys=False, observed=True)

        # Calculate rolling mean using the robust helper function via transform
        mean_col_name = f"mean_{w}"
        long_df[mean_col_name] = grp["val"].transform(
            lambda s: _lagged_rolling_stat(s, w=w, min_p=min_p, stat_func='mean')
        )
        logger.debug(f"Calculated {mean_col_name}")

        # Calculate rolling std using the robust helper function via transform
        std_col_name = f"std_{w}"
        long_df[std_col_name] = grp["val"].transform(
            lambda s: _lagged_rolling_stat(s, w=w, min_p=min_p, stat_func='std')
        )
        logger.debug(f"Calculated {std_col_name}")


        # --- Fill NaNs with Defaults & Clip Std ---
        # Use vectorized fillna based on the 'stat' column
        # Create mapping Series for defaults to use with fillna
        default_means = long_df['stat'].map(lambda s: DEFAULTS.get(s, 0.0))
        default_stds = long_df['stat'].map(lambda s: DEFAULTS.get(f"{s}_std", 0.0))

        long_df[mean_col_name] = long_df[mean_col_name].fillna(default_means)
        long_df[std_col_name] = long_df[std_col_name].fillna(default_stds).clip(lower=0)
        logger.debug(f"Filled NaNs and clipped std for window {w}")

        # --- Pivot to Wide Format ---
        # Reset index *before* pivoting if game_date was the index
        long_df_reset = long_df.reset_index() # Bring game_date back as column

        # Pivot mean
        wide_mean = long_df_reset.pivot_table(
            index=["game_id", "team_norm"], # Use game_id and team_norm from columns
            columns="stat",
            values=mean_col_name,
            aggfunc="first", # Use first non-null value found for a game_id/team_norm combo
        )
        wide_mean.columns = [f"rolling_{s}_mean_{w}" for s in wide_mean.columns]
        logger.debug(f"Pivoted mean features for window {w}. Shape: {wide_mean.shape}")

        # Pivot std
        wide_std = long_df_reset.pivot_table(
            index=["game_id", "team_norm"],
            columns="stat",
            values=std_col_name,
            aggfunc="first",
        )
        wide_std.columns = [f"rolling_{s}_std_{w}" for s in wide_std.columns]
        logger.debug(f"Pivoted std features for window {w}. Shape: {wide_std.shape}")


        pieces.append(pd.concat([wide_mean, wide_std], axis=1))
        logger.debug(f"Appended pivoted features for window {w}")

    # --- Consolidate and Merge Back ---
    if not pieces:
        logger.warning("No rolling stats generated (pieces list is empty); returning original frame.")
        # Drop added columns before returning
        return out.drop(columns=["home_norm", "away_norm"], errors="ignore")

    # Concatenate all wide pieces for different windows
    tidy = pd.concat(pieces, axis=1).reset_index()  # index is [game_id, team_norm]
    logger.debug(f"Concatenated all features. Tidy shape: {tidy.shape}")

    # Create merge keys for joining back to the original 'out' DataFrame
    tidy["merge_key"] = tidy["game_id"].astype(str) + "_" + tidy["team_norm"]
    out["merge_key_home"] = out["game_id"].astype(str) + "_" + out["home_norm"]
    out["merge_key_away"] = out["game_id"].astype(str) + "_" + out["away_norm"]
    logger.debug("Created merge keys.")

    # Prepare column renaming maps
    roll_cols = [c for c in tidy.columns if c.startswith("rolling_")]
    home_map = {
        c: _mk("home", "_".join(c.split("_")[1:-2]), c.split("_")[-2], int(c.split("_")[-1]))
        for c in roll_cols
    }
    away_map = {
        c: _mk("away", "_".join(c.split("_")[1:-2]), c.split("_")[-2], int(c.split("_")[-1]))
        for c in roll_cols
    }

    # Merge home team features
    out = out.merge(
        tidy[["merge_key", *roll_cols]].rename(columns=home_map),
        how="left",
        left_on="merge_key_home",
        right_on="merge_key",
        suffixes=('', '_drop_home') # Add suffix to avoid duplicate merge_key
    )
    logger.debug("Merged home features.")

    # Merge away team features
    out = out.merge(
        tidy[["merge_key", *roll_cols]].rename(columns=away_map),
        how="left",
        left_on="merge_key_away",
        right_on="merge_key",
        suffixes=('', '_drop_away') # Add suffix to avoid duplicate merge_key
    )
    logger.debug("Merged away features.")


    # --- Clean Up ---
    # Define columns to drop, including temporary merge keys and suffixed columns
    cols_to_drop = [
        "home_norm", "away_norm",
        "merge_key_home", "merge_key_away",
        "merge_key", # Original merge key from tidy after merge
        "merge_key_drop_home", "merge_key_drop_away", # Suffixed merge keys
    ]
    # Also drop any other suffixed columns created during merge conflicts if necessary
    suffixed_cols = [c for c in out.columns if '_drop_home' in c or '_drop_away' in c]
    cols_to_drop.extend(suffixed_cols)

    # Use set for unique columns and list for drop function
    out.drop(columns=list(set(cols_to_drop)), inplace=True, errors="ignore")
    logger.debug(f"Cleaned up temporary columns. Final df shape: {out.shape}")

    return out