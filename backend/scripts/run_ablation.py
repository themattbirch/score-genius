#!/usr/bin/env python3
"""
Ablation-study runner
─────────────────────
* Loads history.parquet
* Applies feature blocks (thin proxies around legacy FeatureEngine)
* 5-fold CV RMSE with sklearn Ridge
* Writes JSON to reports/ablation_results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from backend.features import advanced, momentum, rolling, rest, h2h, season, form
from backend import config
from supabase import create_client

# --------------------------------------------------------------------------- #
#  canonical execution order & explicit deps
# --------------------------------------------------------------------------- #
EXECUTION_ORDER: list[str] = [
    "advanced",
    "momentum",
    "rolling",
    "rest",
    "h2h",
    "season",
    "form",
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

REQUIRES: dict[str, tuple[str, ...]] = {
    "rolling": ("advanced", "momentum"),
    "h2h":     ("rolling",),
    "season":  ("advanced",),
    "form":    ("season",),
}

# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _closure(blocks: Iterable[str]) -> list[str]:
    """Expand *blocks* with transitive deps, keep canonical order."""
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
    h2h_window: int,
) -> pd.DataFrame:
    out = df.copy()
    active = set(blocks)
    for name in EXECUTION_ORDER:
        if name not in active:
            continue
        fn = MODULES[name]
        if name == "rolling":
            out = fn(out, window_sizes=rolling_windows)
        elif name == "h2h":
            out = fn(out, historical_df=hist_df, window=h2h_window)
        elif name == "season":
            out = fn(out, team_stats_df=team_stats_df)
        else:
            out = fn(out)
    return out


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any()]


def drop_target_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any columns that directly contain the target or an obvious
    linear transform of it.
    """
    leak_cols = [c for c in df.columns if c.endswith("_score")]
    leak_cols += ["total_score", "point_diff"]
    return df.drop(columns=set(leak_cols) & set(df.columns), errors="ignore")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5_000,
                    help="Random sample size (0 = full data)")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/ablation_results.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # load history.parquet -------------------------------------------------- #
    for p in (Path("data/history.parquet"), Path("backend/data/history.parquet")):
        if p.exists():
            raw = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError("history.parquet not found")

    if args.sample and len(raw) > args.sample:
        raw = raw.sample(args.sample, random_state=42).reset_index(drop=True)

    # team-stats for season block ------------------------------------------ #
    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    tstats = supa.table("nba_historical_team_stats").select("*").execute()
    team_stats_df = pd.DataFrame(tstats.data or [])

    # targets -------------------------------------------------------------- #
    y = raw["home_score"] + raw["away_score"]

    # fair baseline: always predict historical mean ----------------------- #
    dummy = DummyRegressor(strategy="mean")
    baseline_rmse = -cross_val_score(
        dummy, np.zeros((len(y), 1)), y,        # features are a constant 0 column
        cv=args.folds,
        scoring="neg_root_mean_squared_error",
    ).mean()
    print(f"Baseline (mean) RMSE: {baseline_rmse:.3f}")

    ridge = Ridge(alpha=1.0, random_state=42)
    results: dict[str, float] = {"baseline": baseline_rmse}

    rolling_windows = [5, 10, 20]
    h2h_window = 7
    blocks = list(MODULES.keys())

    # ablation ------------------------------------------------------------- #
    for L in range(1, len(blocks) + 1):
        for combo in itertools.combinations(blocks, L):
            full_combo = _closure(combo)

            feats = apply_blocks(
                raw,
                full_combo,
                hist_df=raw,
                team_stats_df=team_stats_df,
                rolling_windows=rolling_windows,
                h2h_window=h2h_window,
            )

            X = numeric_features(drop_target_leak(feats))
            rmse = -cross_val_score(
                ridge, X, y,
                cv=args.folds,
                scoring="neg_root_mean_squared_error",
            ).mean()

            key = " + ".join(combo)
            delta = baseline_rmse - rmse
            results[key] = rmse
            print(f"{key:55} RMSE {rmse:.3f}  Δ {delta:+.3f}")

    # write results -------------------------------------------------------- #
    with args.out.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✅ Ablation results written to {args.out}")


if __name__ == "__main__":
    main()
