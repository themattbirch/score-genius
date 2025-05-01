# backend/scripts/run_ablation.py
"""
Ablation-study runner

 * Reads the big `history.parquet`
 * Applies selected feature-blocks (thin proxies around legacy FeatureEngine)
 * 5-fold CV RMSE with sklearn Ridge
 * Writes JSON to <out_path> (default reports/ablation_results.json)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# --------------------------------------------------------------------------- #
#  Project imports – thin-proxy modules
# --------------------------------------------------------------------------- #
from backend.features import advanced, momentum, rolling, rest, h2h, season, form
from backend import config
from supabase import create_client

# canonical order of dependencies
EXECUTION_ORDER: list[str] = [
    "advanced",  # box-score → efficiencies, pace, etc.
    "momentum",  # requires quarter scores
    "rolling",   # rolling window stats
    "rest",      # rest days, back-to-back
    "h2h",       # head-to-head vs historical
    "season",    # season context via team_stats
    "form",      # form strings last
]

MODULES: Mapping[str, callable] = {
    "advanced": advanced.transform,
    "momentum": momentum.transform,
    "rolling":  rolling.transform,
    "rest":     rest.transform,
    "h2h":      h2h.transform,
    "season":   season.transform,
    "form":     form.transform,
}


def apply_blocks(
    df: pd.DataFrame,
    blocks: Iterable[str],
    *,
    hist_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    rolling_windows: list[int],
    h2h_window: int,
) -> pd.DataFrame:
    """
    Apply *blocks* in canonical dependency order; return a fresh DF each time.
    """
    out = df.copy()
    active = set(blocks)

    for name in EXECUTION_ORDER:
        if name not in active:
            continue
        fn = MODULES[name]

        if name == "rolling":
            # pure-Python path: no DB conn here
            out = fn(out, window_sizes=rolling_windows)
        elif name == "h2h":
            # correctly pass `window=…`
            out = fn(out, historical_df=hist_df, window=h2h_window)
        elif name == "season":
            out = fn(out, team_stats_df=team_stats_df)
        else:
            # advanced, momentum, rest, form
            out = fn(out)
    return out


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only copy (drop all-NA columns)."""
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablation CV with Ridge on feature blocks")
    ap.add_argument("--sample", type=int, default=5000,
                    help="Random sample size for quick smoke (0 = full)")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/ablation_results.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- load full history parquet ---
    for p in (Path("data/history.parquet"), Path("backend/data/history.parquet")):
        if p.exists():
            raw = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError("Could not locate history.parquet")

    # sample down if requested
    if args.sample and len(raw) > args.sample:
        raw = raw.sample(args.sample, random_state=42).reset_index(drop=True)

    # --- pull team-stats for season context ---
    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    resp_ts = supa.table("nba_historical_team_stats").select("*").execute()
    team_stats_df = pd.DataFrame(resp_ts.data or [])

    # --- targets & baseline ---
    y = raw["home_score"] + raw["away_score"]
    base_X = numeric_features(raw[["home_score", "away_score"]])
    ridge = Ridge(alpha=1.0, random_state=42)
    baseline_rmse = -cross_val_score(
        ridge, base_X, y,
        cv=args.folds,
        scoring="neg_root_mean_squared_error",
    ).mean()
    print(f"Baseline RMSE: {baseline_rmse:.3f}")

    results: dict[str, float] = {"baseline": baseline_rmse}
    rolling_windows = [5, 10, 20]
    h2h_window     = 7
    blocks = list(MODULES.keys())

    # --- ablation loop ---
    for L in range(1, len(blocks) + 1):
        for combo in itertools.combinations(blocks, L):
            feats = apply_blocks(
                raw,
                combo,
                hist_df=raw,
                team_stats_df=team_stats_df,
                rolling_windows=rolling_windows,
                h2h_window=h2h_window,
            )
            X = numeric_features(
                feats.drop(columns=["home_score", "away_score", "game_id"], errors="ignore")
            )
            rmse = -cross_val_score(
                ridge, X, y,
                cv=args.folds,
                scoring="neg_root_mean_squared_error",
            ).mean()

            key   = " + ".join(combo)
            delta = baseline_rmse - rmse
            results[key] = rmse
            print(f"{key:55} RMSE {rmse:.3f}  Δ {delta:+.3f}")

    # --- write results ---
    with args.out.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✅ Ablation results written to {args.out}")


if __name__ == "__main__":
    main()
