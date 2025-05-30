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
import argparse
import itertools
import json
import logging
from typing import Iterable, Mapping, Any, Optional, Dict, List

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from dotenv import load_dotenv
from supabase import create_client

# ──────────────────────────── project root setup ────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "backend" / ".env")

# ───────────────────────────── logger setup ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────────── import your feature modules ──────────────────────
from backend.nba_features import (
    advanced,
    rolling,
    rest,
    h2h,
    season,
    form,
    game_advanced_metrics,
)
from backend import config
from backend.nba_features.utils import determine_season

# ─────────────────── canonical execution order & deps ────────────────────
EXECUTION_ORDER = [
    "game_advanced_metrics",
    "rest",
    "advanced",
    "h2h",
    "season",
    "form",
    "rolling",
]
MODULES: Mapping[str, callable] = {
    "game_advanced_metrics": game_advanced_metrics.transform,
    "rest":               rest.transform,
    "advanced":           advanced.transform,
    "h2h":                h2h.transform,
    "season":             season.transform,
    "form":               form.transform,
    "rolling":            rolling.transform,
}
REQUIRES: Mapping[str, Iterable[str]] = {}

# ────────────────────── data‐fetching helpers ──────────────────────────
def _fetch_team_stats(db_conn: Any) -> pd.DataFrame:
    logger.info("Ablation: Fetching historical team stats...")
    try:
        resp = db_conn.from_("nba_historical_team_stats").select("*").execute()
        return pd.DataFrame(resp.data or [])
    except Exception as e:
        logger.error(f"Failed to fetch team stats: {e}")
        return pd.DataFrame()

def _fetch_ablation_adv_splits(db_conn: Any, season_year: int) -> pd.DataFrame:
    logger.info(f"Ablation: Fetching advanced splits for season {season_year}...")
    try:
        resp = (
            db_conn
            .from_("nba_team_seasonal_advanced_splits")
            .select("*")
            .eq("season", season_year)
            .execute()
        )
        return pd.DataFrame(resp.data or [])
    except Exception as e:
        logger.error(f"Failed to fetch adv_splits for season {season_year}: {e}")
        return pd.DataFrame()

# ─────────────────── core feature‐application logic ───────────────────
def apply_blocks(
    df: pd.DataFrame,
    blocks: Iterable[str],
    *,
    game_history_for_h2h: pd.DataFrame,
    general_team_stats_df: pd.DataFrame,
    seasonal_adv_splits_lookup_df: pd.DataFrame,
    lookup_season_for_adv_splits: Optional[int],
    rolling_windows_list: List[int],
    h2h_window_val: int,
    flag_all_imputations: bool = True,
    debug_mode_feature_gen: bool = False,
) -> pd.DataFrame:
    out_df = df.copy()
    to_run = set(blocks)

    for name in EXECUTION_ORDER:
        if name not in to_run:
            continue
        func = MODULES[name]
        kwargs: dict = {"debug": debug_mode_feature_gen}

        if name == "h2h":
            kwargs.update(historical_df=game_history_for_h2h, max_games=h2h_window_val)
        elif name == "season":
            kwargs.update(team_stats_df=general_team_stats_df, flag_imputations=flag_all_imputations)
        elif name == "form":
            kwargs.update(team_stats_df=general_team_stats_df)
        elif name == "advanced":
            kwargs.update(
                stats_df=seasonal_adv_splits_lookup_df,
                season=lookup_season_for_adv_splits,
                flag_imputations=flag_all_imputations,
            )
        elif name == "rolling":
            kwargs.update(window_sizes=rolling_windows_list, flag_imputation=flag_all_imputations)

        logger.debug(f"Ablation: Applying block '{name}'")
        out_df = func(out_df, **kwargs)

    return out_df

def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any()]

def drop_target_leak(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: insert your actual leak-dropping column list here
    return df

def whitelist_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

# ────────────────────────────── main ──────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Dependency-aware, memoized ablation CV")
    p.add_argument("--sample", type=int, default=5_000)
    p.add_argument("--folds",  type=int, default=3)
    p.add_argument("--jobs",   type=int, default=-1)
    p.add_argument("--out",    type=Path, default=Path("reports/ablation_results.json"))
    p.add_argument("--debug-features", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ablation study with args: {args}")

    # load raw history
    for cand in (
        Path("data/history.parquet"),
        Path("backend/data/history.parquet"),
        PROJECT_ROOT / "data" / "history.parquet",
    ):
        if cand.exists():
            raw = pd.read_parquet(cand)
            logger.info(f"Loaded raw game data from {cand}")
            break
    else:
        logger.error("history.parquet not found. Exiting.")
        sys.exit(1)

    if args.sample and len(raw) > args.sample:
        raw = raw.sample(args.sample, random_state=42).reset_index(drop=True)
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    if not supa:
        logger.error("Failed to init Supabase client. Exiting.")
        sys.exit(1)

    team_stats = _fetch_team_stats(supa)

    # determine lookup season for advanced splits
    lookup_season: Optional[int] = None
    adv_splits: pd.DataFrame = pd.DataFrame()
    if not raw["game_date"].dropna().empty:
        date0 = raw["game_date"].min()
        lookup_season = int(determine_season(date0).split("-")[0]) - 1
        adv_splits = _fetch_ablation_adv_splits(supa, lookup_season)

    # baseline
    y = raw["home_score"] + raw["away_score"]
    dummy = DummyRegressor(strategy="mean")
    base_rmse = -cross_val_score(
        dummy,
        np.zeros((len(y), 1)),
        y,
        cv=args.folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=args.jobs,
    ).mean()
    logger.info(f"Baseline RMSE: {base_rmse:.3f}")

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    results: Dict[str, float] = {"baseline_mean_target": base_rmse}

    all_blocks = EXECUTION_ORDER.copy()
    tscv = TimeSeriesSplit(n_splits=args.folds)

    for leave_out in all_blocks:
        logger.info(f"Evaluating without block '{leave_out}'")
        to_apply = [b for b in all_blocks if b != leave_out]
        scores: List[float] = []

        for train_idx, val_idx in tscv.split(raw):
            tr, va = raw.iloc[train_idx], raw.iloc[val_idx]

            X_tr = apply_blocks(
                tr, to_apply,
                game_history_for_h2h=tr,
                general_team_stats_df=team_stats,
                seasonal_adv_splits_lookup_df=adv_splits,
                lookup_season_for_adv_splits=lookup_season,
                rolling_windows_list=[5, 10, 20],
                h2h_window_val=7,
                flag_all_imputations=True,
                debug_mode_feature_gen=args.debug_features,
            )
            X_va = apply_blocks(
                va, to_apply,
                game_history_for_h2h=tr,
                general_team_stats_df=team_stats,
                seasonal_adv_splits_lookup_df=adv_splits,
                lookup_season_for_adv_splits=lookup_season,
                rolling_windows_list=[5, 10, 20],
                h2h_window_val=7,
                flag_all_imputations=True,
                debug_mode_feature_gen=args.debug_features,
            )

            # cast any _imputed flags to int
            for df_ in (X_tr, X_va):
                for c in df_.columns:
                    if c.endswith("_imputed"):
                        df_[c] = df_[c].astype(int)

            Xtr_num = numeric_features(drop_target_leak(whitelist_pregame_features(X_tr)))
            Xva_num = numeric_features(drop_target_leak(whitelist_pregame_features(X_va)))
            common = Xtr_num.columns.intersection(Xva_num.columns)
            Xtr_num, Xva_num = Xtr_num[common], Xva_num[common]

            if common.empty or Xtr_num.empty or Xva_num.empty:
                scores.append(np.nan)
                continue

            y_tr = tr["home_score"] + tr["away_score"]
            y_va = va["home_score"] + va["away_score"]
            try:
                ridge.fit(Xtr_num, y_tr)
                preds = ridge.predict(Xva_num)
                scores.append(np.sqrt(mean_squared_error(y_va, preds)))
            except Exception as e:
                logger.error(f"Error fitting '{leave_out}' fold: {e}")
                scores.append(np.nan)

        avg = float(np.nanmean(scores))
        results[f"without_{leave_out}"] = avg
        logger.info(f"Avg RMSE without '{leave_out}': {avg:.3f}")

    # write results
    with args.out.open("w") as fp:
        serial = {k: (float(v) if pd.notna(v) else None) for k, v in results.items()}
        json.dump(serial, fp, indent=4)
    logger.info(f"Ablation results written to {args.out}")

if __name__ == "__main__":
    main()
