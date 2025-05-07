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

# 1) ensure project root on PYTHONPATH so import backend… works
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 2) auto‐load .env for Supabase keys, etc.
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "backend" / ".env")

import argparse
import itertools
import json
from typing import Iterable
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
EXECUTION_ORDER = ["rest", "h2h", "season", "form"]
MODULES = {
    "rest":   rest.transform,
    "h2h":    h2h.transform,
    "season": season.transform,
    "form":   form.transform,
}
REQUIRES = {
    "rest":   (),
    "h2h":    (),
    "season": (),
    "form":   ("season",),
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
    h2h_window: int,
) -> pd.DataFrame:
    """
    Apply each block in EXECUTION_ORDER (after expansion) to a fresh df copy.
    """
    out = df.copy()
    active = set(blocks)

    for name in EXECUTION_ORDER:
        if name not in active:
            continue

        fn = MODULES[name]
        logger.debug(f"Applying block: {name}")

        kwargs: dict[str, object] = {'debug': False}
        if name == "h2h":
            kwargs['historical_df'] = hist_df
            kwargs['max_games']     = h2h_window
        elif name == "season":
            kwargs['team_stats_df'] = team_stats_df

        out = fn(out, **kwargs)

    return out

def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any()]

def drop_target_leak(df: pd.DataFrame) -> pd.DataFrame:
    # every column from the raw stat table that leaks the answer:
    raw_post_game_stats = [
        'home_score','away_score',
        'home_q1','home_q2','home_q3','home_q4','home_ot',
        'away_q1','away_q2','away_q3','away_q4','away_ot',
        'home_fg_made','home_fg_attempted','away_fg_made','away_fg_attempted',
        'home_3pm','home_3pa','away_3pm','away_3pa',
        'home_ft_made','home_ft_attempted','away_ft_made','away_ft_attempted',
        'home_assists','home_steals','home_blocks','home_turnovers','home_fouls',
        'away_assists','away_steals','away_blocks','away_turnovers','away_fouls',
        'home_off_reb','home_def_reb','home_total_reb',
        'away_off_reb','away_def_reb','away_total_reb',
    ]
    identifiers = ['game_id', 'game_date']
    to_drop = set(raw_post_game_stats + identifiers)
    return df.drop(columns=to_drop, errors='ignore')

def whitelist_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # modify if you ever want a stricter whitelist

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

        # load the *raw* box‐score stats from Supabase (no leakage)
    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    resp = (
        supa
        .table("nba_historical_game_stats")
        .select("*")
        .execute()
    )
    raw = pd.DataFrame(resp.data or [])
    if raw.empty:
        sys.exit("ERROR: failed to load raw historical stats from Supabase")


    if args.sample and len(raw) > args.sample:
        raw = raw.sample(args.sample, random_state=42).reset_index(drop=True)

    # pull team stats once
    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    resp = supa.table("nba_historical_team_stats").select("*").execute()
    team_stats_df = pd.DataFrame(resp.data or [])

    # target & baseline
    y = raw["home_score"] + raw["away_score"]
    dummy = DummyRegressor(strategy="mean")
    baseline_rmse = -cross_val_score(
        dummy, np.zeros((len(y),1)), y,
        cv=args.folds, scoring="neg_root_mean_squared_error", n_jobs=args.jobs
    ).mean()
    print(f"Baseline (mean) RMSE: {baseline_rmse:.3f}")

    ridge = Ridge(alpha=1.0, random_state=42)
    results: dict[str, float] = {"baseline": baseline_rmse}

    h2h_window = 7
    blocks = list(MODULES.keys())

    # cache for feature‐blocks outputs
    cache: dict[str, pd.DataFrame] = {}

    # ablation loops
    for L in range(args.min_blocks, min(args.max_blocks, len(blocks)) + 1):
        for combo in itertools.combinations(blocks, L):
            full_combo = _closure(combo)
            combo_key  = " + ".join(combo)
            key        = " + ".join(full_combo)

            if key not in cache:
                cache[key] = apply_blocks(
                    raw,
                    full_combo,
                    hist_df=raw,
                    team_stats_df=team_stats_df,
                    h2h_window=h2h_window
                )
            feats = cache[key]

            X = numeric_features(drop_target_leak(feats))
            X = whitelist_pregame_features(X)

            # impute missing
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

            rmse = -cross_val_score(
                ridge, X, y,
                cv=args.folds,
                scoring="neg_root_mean_squared_error",
                n_jobs=args.jobs
            ).mean()

            delta = baseline_rmse - rmse
            results[combo_key] = rmse
            print(f"{combo_key:45}  RMSE {rmse:.3f}  Δ {delta:+.3f}")

    # write results
    with args.out.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n✅ Ablation results written to {args.out}")

if __name__ == "__main__":
    main()
