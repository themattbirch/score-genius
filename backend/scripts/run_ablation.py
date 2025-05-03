# backend/scripts/run_ablation.py
"""
Faster Ablation‐study runner with:
  • dependency‐aware combos
  • memoized transforms
  • parallel CV
"""

from __future__ import annotations
import sys
from pathlib import Path

# 1) ensure project root on PYTHONPATH so `import backend…` works
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 2) auto‐load .env for Supabase keys, etc.
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "backend" / ".env")

import argparse
import itertools
import json
from typing import Iterable, Mapping
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from supabase import create_client

# your modular thin‐proxies
from backend.nba_features import advanced, momentum, rolling, rest, h2h, season, form
from backend import config

import logging

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# canonical execution order & explicit deps
# ---------------------------------------------------------------------------- #
EXECUTION_ORDER = ["advanced", "momentum", "rest", "h2h", "season", "form"]
MODULES: Mapping[str, callable] = {
    "advanced": advanced.transform,
    "momentum": momentum.transform,
    #"rolling":  rolling.transform,
    "rest":     rest.transform,
    "h2h":      h2h.transform,
    "season":   season.transform,
    "form":     form.transform,
}
REQUIRES = {
    #"rolling": ("advanced", "momentum"),
    "h2h":     ("rolling",),
    "season":  ("advanced",),
    "form":    ("season",),
}

def _closure(blocks: Iterable[str]) -> list[str]:
    """Expand blocks with transitive deps, return in canonical order."""
    need = set(blocks)
    changed = True
    while changed:
        changed = False
        for b in list(need):
            for dep in REQUIRES.get(b, ()):
                if dep not in need:
                    need.add(dep)
                    changed = True
    return [b for b in EXECUTION_ORDER if b in need]

def apply_blocks(
    df: pd.DataFrame,
    blocks: Iterable[str],
    *,
    hist_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    rolling_windows: list[int],
    h2h_window: int, # Keep this parameter name for the function signature
) -> pd.DataFrame:
    """
    Apply each block in EXECUTION_ORDER (after expansion) to a fresh df copy.
    """
    out = df.copy()
    active = set(blocks)
    # Use a temporary debug flag based on script args if needed, or pass it down
    # debug_mode = args.debug # Assuming args is accessible or passed down

    for name in EXECUTION_ORDER:
        if name not in active:
            continue

        fn = MODULES[name]
        logger.debug(f"Applying block: {name}") # Added debug log

        # Prepare kwargs for the specific module
        kwargs = {'debug': False} # Default debug to False unless passed down

        if name == "rolling":
            kwargs['window_sizes'] = rolling_windows
            out = fn(out, **kwargs) # Pass kwargs
        elif name == "h2h":
            # *** FIX: Use 'max_games' instead of 'window' ***
            kwargs['historical_df'] = hist_df
            kwargs['max_games'] = h2h_window # Use the correct keyword argument
            out = fn(out, **kwargs) # Pass kwargs
        elif name == "season":
            kwargs['team_stats_df'] = team_stats_df
            out = fn(out, **kwargs) # Pass kwargs
        else:
             # For modules like momentum, advanced, rest, form that only take df and debug
             out = fn(out, **kwargs) # Pass kwargs

    return out

def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns with at least one non‐NA value."""
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any()]

def drop_target_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns directly related to or calculated from the final game outcome
    to prevent data leakage during cross-validation or training.
    """
    # Original direct target columns
    direct_targets = [
        'home_score',
        'away_score',
        'total_score', # Calculated from home_score + away_score
        'point_diff',  # Calculated from home_score - away_score
    ]

    # Columns often calculated *using* final scores (potential leakage if present in input)
    # Add any other columns from your raw data that are post-game results
    derived_targets = [
        'home_offensive_rating', 'away_offensive_rating',
        'home_defensive_rating', 'away_defensive_rating', # Derived from opponent's offensive rating
        'home_net_rating', 'away_net_rating',             # Derived from off/def ratings
        'efficiency_differential',                       # Derived from net ratings
        # Add others if they exist in your raw data and depend on final scores:
        # 'home_efg_pct', 'away_efg_pct', # Sometimes calculated post-game
        # 'home_tov_rate', 'away_tov_rate', # Sometimes calculated post-game
        # etc.
    ]

    # Also remove identifiers that shouldn't be features
    identifiers = ['game_id', 'game_date'] # game_date might be okay if handled carefully, but safer to remove here

    # Combine all columns to drop
    columns_to_drop = direct_targets + derived_targets + identifiers

    # Drop columns, ignoring errors if a column doesn't exist
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    # Optional: Log which columns were actually dropped
    dropped_cols = [col for col in columns_to_drop if col in df.columns]
    if dropped_cols:
         logger.debug(f"drop_target_leak removed columns: {dropped_cols}")
    else:
         logger.debug("drop_target_leak: No target-related columns found to remove.")


    return df_cleaned

def whitelist_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Optional) further restrict to a safe‐only whitelist if you want.
    For now we just use numeric_features + drop_target_leak.
    """
    return df

# ---------------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Dependency-aware, memoized ablation CV")
    p.add_argument("--sample",    type=int,    default=5_000, help="0=full")
    p.add_argument("--folds",     type=int,    default=3,     help="CV folds")
    p.add_argument("--jobs",      type=int,    default=-1,    help="n_jobs for CV")
    p.add_argument("--min-blocks",type=int,    default=1,     help="skip combos smaller than this")
    p.add_argument("--max-blocks",type=int,    default=7,     help="skip combos larger than this")
    p.add_argument("--out",       type=Path,   default=Path("reports/ablation_results.json"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- load history.parquet ------------------------------------------------ #
    for cand in (Path("data/history.parquet"), Path("backend/data/history.parquet")):
        if cand.exists():
            raw = pd.read_parquet(cand)
            break
    else:
        sys.exit("ERROR: history.parquet not found")

    if args.sample and len(raw) > args.sample:
        raw = raw.sample(args.sample, random_state=42).reset_index(drop=True)

    # --- pull team stats once (for season block) ----------------------------- #
    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    resp = supa.table("nba_historical_team_stats").select("*").execute()
    team_stats_df = pd.DataFrame(resp.data or [])

    # --- target & baseline -------------------------------------------------- #
    y = raw["home_score"] + raw["away_score"]
    dummy = DummyRegressor(strategy="mean")
    baseline_rmse = -cross_val_score(
        dummy, np.zeros((len(y),1)), y,
        cv=args.folds, scoring="neg_root_mean_squared_error", n_jobs=args.jobs
    ).mean()
    print(f"Baseline (mean) RMSE: {baseline_rmse:.3f}")

    ridge = Ridge(alpha=1.0, random_state=42)
    results: dict[str, float] = {"baseline": baseline_rmse}

    rolling_windows = [5,10,20]
    h2h_window     = 7
    blocks         = list(MODULES)

    # --- memo cache for feature‐blocks outputs ------------------------------ #
    cache: dict[str, pd.DataFrame] = {}

    # --- ablation ----------------------------------------------------------- #
    for L in range(args.min_blocks, min(args.max_blocks, len(blocks)) + 1):
            for combo in itertools.combinations(blocks, L):
              full_combo = _closure(combo)
              combo_key  = " + ".join(combo)
              key        = " + ".join(full_combo)

            # memoize
            if key not in cache:
                cache[key] = apply_blocks(
                    raw,
                    full_combo,
                    hist_df=raw,
                    team_stats_df=team_stats_df,
                    rolling_windows=rolling_windows,
                    h2h_window=h2h_window
                )
            feats = cache[key]

            # drop any direct‐leak cols, keep only numeric
            X = numeric_features(drop_target_leak(feats))
            X = whitelist_pregame_features(X)

            # **DEBUG INSPECTION**
            print(f">>> combo = {combo_key}")
            print("  Columns:", X.columns.tolist())
            print("  Feature count:", len(X.columns))

            rmse = -cross_val_score(
                ridge, X, y,
                cv=args.folds,
                scoring="neg_root_mean_squared_error",
                n_jobs=args.jobs
            ).mean()

            delta = baseline_rmse - rmse
            results[combo_key] = rmse
            print(f"{combo_key:45}  RMSE {rmse:.3f}  Δ {delta:+.3f}")

    # --- write out ---------------------------------------------------------- #
    with args.out.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✅ Ablation results written to {args.out}")

if __name__ == "__main__":
    main()
