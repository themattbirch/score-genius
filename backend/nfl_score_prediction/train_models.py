# backend/nfl_score_prediction/train_models.py
"""
NFL Score Prediction – Model Training Pipeline (with rich DEBUG logging)

Flow:
1. Load historical data (games + team box scores) via NFLDataFetcher.
2. Build features with run_nfl_feature_pipeline().
3. Create targets: margin (home - away) and total (home + away).
4. Feature selection (LassoCV) per target.
5. Chronological split → train / val / test.
6. Train individual models (with optional tuning), get val preds.
7. Optimize ensemble weights on val.
8. Retrain models on train+val (no leakage) and save.
9. Evaluate ensemble on test, save reports & plots.

Extra debug instrumentation:
- DataFrame profiling (rows/cols/nulls/constants) at key steps.
- Feature selection summaries (kept/dropped, null counts, variance).
- Model training timings, best params, weight distributions.
- Ensemble weight diagnostics, test-set error breakdowns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from supabase import Client, create_client
from collections import Counter

# ---- Project imports ----
try:
    from backend import config
    from backend.nfl_features.engine import NFLFeatureEngine
    from backend.nfl_score_prediction.data import NFLDataFetcher
    from backend.nfl_score_prediction.models import (
        BaseNFLPredictor,
        RidgeMarginPredictor,
        SVRMarginPredictor,
        XGBoostTotalPredictor,
        RFTotalPredictor,
        derive_scores_from_predictions,
        compute_recency_weights,
    )
    from backend.nfl_score_prediction.evaluation import (
        plot_actual_vs_predicted,
        generate_evaluation_report,
    )
except ImportError as e:
    print(f"Import error: {e}. Ensure PYTHONPATH includes project root.")
    sys.exit(1)

# ---- Paths & logging ----
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports" / "nfl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
DEFAULT_CV_FOLDS = 4

NFL_SAFE_FEATURE_PREFIXES = (
    "home_", "away_",
    "h_", "a_",
    "rolling_", "season_", "h2h_", "momentum_", "rest_", "form_",
    "adv_", "drive_", "situational_", "total_", "map_",
)
LABEL_BLOCKLIST = {"home_score", "away_score", "margin", "total"}
NFL_SAFE_EXACT_FEATURE_NAMES = (
    "day_of_week",
    "week",
    "is_division_game",
    "is_conference_game",
)

MODEL_CLASS_MAP: Dict[str, Type[BaseNFLPredictor]] = {
    "nfl_ridge_margin_predictor": RidgeMarginPredictor,
    "nfl_svr_margin_predictor": SVRMarginPredictor,
    "nfl_xgb_total_predictor": XGBoostTotalPredictor,
    "nfl_rf_total_predictor": RFTotalPredictor,
}

# =============================================================================
# Debug / profiling helpers
# =============================================================================

def _df_profile(df: pd.DataFrame, name: str, top_n: int = 12) -> Dict[str, Any]:
    if df.empty:
        return {"name": name, "rows": 0, "cols": 0}
    nulls = df.isna().sum().sort_values(ascending=False)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    num_cols = df.select_dtypes(include=np.number).columns
    zeros = (df[num_cols] == 0).sum().sort_values(ascending=False) if len(num_cols) else pd.Series(dtype=int)
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": nulls.head(top_n).to_dict(),
        "constant_cols": const_cols[:top_n],
        "zero_top": zeros.head(top_n).to_dict(),
    }

def _log_df(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    prof = _df_profile(df, name)
    logger.debug(
        "[DF] %-25s rows=%-6d cols=%-4d | null_top=%s | const=%s | zero_top=%s",
        prof["name"], prof["rows"], prof["cols"],
        prof.get("null_top"), prof.get("constant_cols"), prof.get("zero_top"),
    )

def _log_feature_list(selected: List[str], candidates: List[str], target: str, debug: bool):
    if not debug:
        return
    dropped = list(set(candidates) - set(selected))
    logger.debug("[FS] %s: kept=%d dropped=%d", target, len(selected), len(dropped))
    if dropped[:20]:
        logger.debug("[FS] %s dropped sample: %s", target, dropped[:20])

# =============================================================================
# Utilities
# =============================================================================

def get_supabase_client() -> Optional[Client]:
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Supabase client init failed: {e}")
        return None

def add_total_composites(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        h, a = f"home_{k}", f"away_{k}"
        if h in out.columns and a in out.columns:
            out[f"total_{k}"] = out[h] + out[a]
    return out

def chronological_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    date_col: str = "game_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]

def build_target_specific_candidates(features: pd.DataFrame) -> dict:
    num = set(features.select_dtypes(include=np.number).columns)

    # global include & exclude
    ALLOW_PREFIX = (
        "home_", "away_", "rolling_", "season_", "h2h_", "momentum_",
        "rest_", "form_", "adv_", "drive_", "situational_", "total_",
        "h_", "a_", "map_",
        "prev_season_",  # <-- ensure prior-season *_diff make it into 'base'
    )
    BLOCK = {"home_score", "away_score", "margin", "total"}  # no leakage
    # keep imputation flags for now (they sometimes help early-season)
    # To drop later, add: if c.endswith("_imputed"): continue

    base = [c for c in num if c.startswith(ALLOW_PREFIX) and c not in BLOCK]

    # buckets / helpers
    is_diff   = lambda c: c.endswith("_diff") or "_advantage" in c
    is_points = lambda c: ("points_for" in c) or ("points_against" in c) or ("total_points" in c) or c.startswith("total_")
    is_rest   = lambda c: ("rest_days" in c) or ("short_week" in c) or ("off_bye" in c)
    is_form   = lambda c: ("form_win_pct" in c) or ("current_streak" in c) or ("momentum" in c)
    is_h2h    = lambda c: c.startswith("h2h_")

    # --- helpers specific to season priors ---
    is_season = lambda c: (
        "prev_season_" in c
        or c.startswith("home_prev_season_")
        or c.startswith("away_prev_season_")
    )
    is_season_diff = lambda c: is_season(c) and c.endswith("_diff")
    is_side_specific_season = lambda c: (
        c.startswith("home_prev_season_") or c.startswith("away_prev_season_")
    )

    # margin → differentials & win-proxy signals (include prior-season diffs explicitly)
    margin_candidates = [
        c for c in base if (
            is_diff(c)                               # general diffs
            or is_season_diff(c)                     # prior-season diffs (e.g., prev_season_win_pct_diff)
            or is_rest(c)
            or is_form(c)
            or ("turnover_differential" in c)
            or (is_h2h(c) and ("home_win_pct" in c))
        )
    ]
    # keep raw rolling diffs if present
    margin_candidates += [c for c in base if "point_differential" in c]

    # Drop side-specific prior-season raw columns to avoid duplication/leakage-like redundancy
    margin_candidates = [c for c in margin_candidates if not is_side_specific_season(c)]

    # totals → scoring volume + pace/volume + rest + h2h totals
    is_volume = lambda c: any(k in c for k in (
        "yards_per_play", "plays_per_game", "seconds_per_play",
        "neutral_pass_rate", "red_zone", "explosive_play_rate",
        "drive_success_rate", "drive_plays", "drive_yards",
        # "epa_per_play", "pace"  # uncomment if these exist
    ))
    total_candidates = [
        c for c in base if (
            is_points(c)
            or is_volume(c)
            or is_rest(c)
            or (is_h2h(c) and ("avg_total_points" in c))
        )
    ]
    # allow a couple of momentum/form signals for total, too
    total_candidates += [c for c in base if is_form(c) and ("form_win_pct" in c)]
    # If you want prior-season diffs to also inform totals, uncomment:
    # total_candidates += [c for c in base if is_season_diff(c)]

    # de-dup & stable order
    margin_candidates = [c for c in base if c in set(margin_candidates)]
    total_candidates  = [c for c in base if c in set(total_candidates)]
    return {"margin": margin_candidates, "total": total_candidates}


def _prune_near_duplicates(X: pd.DataFrame, y: pd.Series, thresh: float = 0.98) -> list[str]:
    """
    Remove features that are extremely collinear with each other
    (absolute correlation >= thresh) to avoid duplicate signal.

    Keeps the first occurrence, drops subsequent near-duplicates.
    """
    if X.shape[1] <= 1:
        return list(X.columns)

    corr_matrix = X.corr().abs()
    to_drop = set()
    kept = []

    for col in corr_matrix.columns:
        if col in to_drop:
            continue
        kept.append(col)
        # Mark other cols with very high correlation to this one
        duplicates = corr_matrix.index[(corr_matrix[col] >= thresh) & (corr_matrix.index != col)]
        to_drop.update(duplicates)

    return kept

# ---------- helpers ----------

def _abs_corr_topk(X: pd.DataFrame, y: pd.Series, pool: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    cols = [c for c in pool if c in X.columns]
    if not cols:
        return []
    corrs = X[cols].apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
    return list(corrs.head(k).index)

def _prune_near_duplicates(X: pd.DataFrame, y: pd.Series | None = None, thresh: float = 0.98) -> list[str]:
    """Keep first feature in any near-duplicate (> thresh abs corr) group."""
    if X.shape[1] <= 1:
        return list(X.columns)
    cm = X.corr().abs()
    keep, dropped = [], set()
    for c in cm.columns:
        if c in dropped:
            continue
        keep.append(c)
        dupes = cm.index[(cm[c] >= thresh) & (cm.index != c)]
        dropped.update(dupes)
    return keep

def _stability_select(
    X: pd.DataFrame,
    y: pd.Series,
    base_selected: list[str],
    seeds=(42, 99, 123),
    floor: float = 2/3,
) -> list[str]:
    if not base_selected:
        return []
    hits = pd.Series(0, index=base_selected, dtype=int)
    for s in seeds:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X[base_selected].fillna(0.0))
        model = LassoCV(alphas=np.logspace(-4, -1, 40), cv=TimeSeriesSplit(n_splits=DEFAULT_CV_FOLDS), random_state=s, n_jobs=-1, max_iter=10000).fit(Xs, y)
        keep = [c for c, coef in zip(base_selected, model.coef_) if abs(coef) > 1e-5]
        hits.loc[keep] += 1
    stable = list(hits[hits / len(seeds) >= floor].index)
    return stable if stable else base_selected

def _bucket_for(col: str) -> str:
    c = col.lower()

    # Explicit prev-season routing (do this FIRST so it doesn't get caught by home_/away_)
    if c.startswith("prev_season_") or "_prev_season_" in c \
       or c.startswith("home_prev_season_") or c.startswith("away_prev_season_"):
        return "season"

    if c.startswith("h2h_"): return "h2h"
    if c.startswith("rolling_") or "rolling_" in c: return "rolling"
    if c.startswith("rest_") or "rest_days" in c or "short_week" in c or "off_bye" in c: return "rest"
    if c.startswith("form_") or "form_win_pct" in c or "momentum" in c: return "form/momentum"
    if c.startswith("drive_"): return "drive"
    if c.startswith("adv_"): return "advanced"
    if c.startswith("season_"): return "season"
    if c.startswith("situational_") or c in ("week","day_of_week","is_division_game","is_conference_game"): return "situational"
    if c.startswith("total_"): return "engineered_total"
    if c.startswith("home_") or c.startswith("away_"): return "raw_home_away"
    if c.startswith("map_"): return "map"
    return "other"

def _summarize_candidates(name: str, cols: list[str]) -> None:
    ct = Counter(_bucket_for(c) for c in cols)
    logger.info("CANDIDATES[%s] count=%d | by bucket=%s", name, len(cols), dict(ct))
    logger.debug("CANDIDATES[%s] sample=%s", name, cols[:10])

def _summarize_selected(name: str, cols: list[str]) -> None:
    ct = Counter(_bucket_for(c) for c in cols)
    logger.info("SELECTED[%s] count=%d | by bucket=%s", name, len(cols), dict(ct))
    logger.debug("SELECTED[%s] = %s", name, cols)


def _ensure_bucket_mi(
    X: pd.DataFrame,
    y: pd.Series,
    selected: list[str],
    target_name: str,
    min_per_bucket: int = 1,
    topk_per_bucket: int = 2,
) -> list[str]:
    """Ensure each bucket contributes at least `min_per_bucket` features; add via MI if needed."""
    sel_set = set(selected)
    buckets: dict[str, list[str]] = {}
    for c in X.columns:
        buckets.setdefault(_bucket_for(c), []).append(c)

    for bucket, cols in buckets.items():
        have = [c for c in selected if _bucket_for(c) == bucket]
        if len(have) >= min_per_bucket:
            continue

        # viable = non-constant, not already selected
        cols = [c for c in cols if c not in sel_set and X[c].nunique(dropna=False) > 1]
        if not cols:
            continue

        Xb = X[cols].fillna(0.0)
        # MI can fail if all-zero variance; filter again just in case
        cols = [c for c in cols if Xb[c].std(ddof=0) > 0]
        if not cols:
            continue

        mi = mutual_info_regression(Xb.values, y.values, random_state=SEED)
        order = np.argsort(mi)[::-1]
        add = [cols[i] for i in order[:topk_per_bucket]]
        for a in add:
            sel_set.add(a)

    return [c for c in selected] + [c for c in sel_set if c not in selected]

# ---------- main ----------

def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_candidates: list[str],
    target_name: str,
    save_dir: Path = MODELS_DIR,
    debug: bool = False,
) -> list[str]:
    logger.info("Lasso/EN feature selection → %s", target_name)

    # Safety: empty candidate list → fallback by |corr|
    if not feature_candidates:
        num = X.select_dtypes(include=np.number)
        corrs = num.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
        selected = list(corrs.head(20).index)
        with open(save_dir / f"nfl_{target_name}_selected_features.json", "w") as f:
            json.dump(selected, f, indent=4)
        return selected

    # ---------- 0) Build working matrix (numeric only) ----------
    cands = [c for c in feature_candidates if c in X.columns]
    X_sel = X[cands].select_dtypes(include=np.number).copy()
    X_sel = X_sel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # drop constants
    nonconst = [c for c in X_sel.columns if X_sel[c].nunique(dropna=False) > 1]
    X_sel = X_sel[nonconst]

    if X_sel.empty:
        num = X.select_dtypes(include=np.number)
        corrs = num.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
        selected = list(corrs.head(20).index)
        with open(save_dir / f"nfl_{target_name}_selected_features.json", "w") as f:
            json.dump(selected, f, indent=4)
        return selected

    # pre-prune near duplicates BEFORE selector
    CORR_PRUNE_THRESH = 0.9995 if target_name == "total" else 0.995
    keep_pre = _prune_near_duplicates(X_sel, y, thresh=CORR_PRUNE_THRESH)
    X_sel = X_sel[keep_pre]
    _log_df(X_sel, f"X_sel_{target_name}_before_selector", debug)

    # ---------- 1) Selector (ElasticNet for total, Lasso for margin) ----------
    tscv = TimeSeriesSplit(n_splits=DEFAULT_CV_FOLDS)
    use_en = (target_name == "total")
    if use_en:
        selector = ElasticNetCV(
            l1_ratio=[0.3, 0.5, 0.7],
            alphas=np.logspace(-5, -2, 40),
            cv=tscv, random_state=SEED, n_jobs=-1, max_iter=20000
        )
    else:
        selector = LassoCV(
            alphas=np.logspace(-5, -2, 50),
            cv=tscv, random_state=SEED, n_jobs=-1, max_iter=20000
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    selector.fit(X_scaled, y)
    coefs = selector.coef_
    selected = [c for c, b in zip(X_sel.columns, coefs) if float(b) != 0.0]

    # ---------- 2) Stability selection ----------
    if use_en:
        STABILITY_FLOOR = 0.0
        seeds = (42, 99, 123, 777)
    else:
        STABILITY_FLOOR = 1/3
        seeds = (42, 99, 123)
    selected = _stability_select(X_sel, y, selected, seeds=seeds, floor=STABILITY_FLOOR)

    # ---------- 3) Post-prune near-duplicates among selected ----------
    if selected:
        sel_keep = _prune_near_duplicates(X_sel[selected], y, thresh=CORR_PRUNE_THRESH)
        selected = sel_keep

    # ---------- 4) Ensure per-bucket MI coverage ----------
    selected = _ensure_bucket_mi(
        X_sel, y, selected, target_name,
        min_per_bucket=(2 if use_en else 1),
        topk_per_bucket=(3 if use_en else 2),
    )

    # ---------- 5) Enforce per-target floor via |corr| top-up ----------
    desired_keep_map = {"margin": 28, "total": 36}
    desired = desired_keep_map.get(target_name, 28)
    available = len(X_sel.columns)
    min_keep = min(desired, max(available, 0))

    if len(selected) < min_keep:
        rest = [c for c in X_sel.columns if c not in selected]
        need = max(0, min_keep - len(selected))
        top_up = _abs_corr_topk(X_sel, y, rest, need)
        selected += [c for c in top_up if c not in selected]


    # ---------- 6) Order by original candidate order for stability ----------
    selected = [c for c in feature_candidates if c in selected]

    # Save & log
    path = save_dir / f"nfl_{target_name}_selected_features.json"
    with open(path, "w") as f:
        json.dump(selected, f, indent=4)
    logger.info("Selected %d features for %s → %s", len(selected), target_name, path)
    _log_feature_list(selected, feature_candidates, target_name, debug)
    return selected

def optimize_ensemble_weights(
    validation_preds: Dict[str, pd.Series], y_val: pd.Series, debug: bool = False
) -> Dict[str, float]:
    errs = {name: mean_absolute_error(y_val, preds) for name, preds in validation_preds.items()}
    denom = sum(1.0 / (e + 1e-9) for e in errs.values())
    if denom == 0:
        k = len(errs)
        weights = {name: 1.0 / k for name in errs} if k else {}
    else:
        weights = {name: (1.0 / (e + 1e-9)) / denom for name, e in errs.items()}

    if debug:
        logger.debug("[ENSEMBLE] MAEs=%s", errs)
        logger.debug("[ENSEMBLE] Weights=%s (sum=%.4f)", weights, sum(weights.values()))
    return weights


def load_model_by_name(name: str) -> BaseNFLPredictor:
    cls = MODEL_CLASS_MAP[name]
    inst = cls()
    inst.load_model()
    return inst

# =============================================================================
# Model training helper
# =============================================================================

def train_and_get_val_preds(
    predictor_cls: Type[BaseNFLPredictor],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_dist: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    combined_weights: Optional[np.ndarray] = None,
    train_weights: Optional[np.ndarray] = None,
) -> Tuple[BaseNFLPredictor, pd.Series, Dict[str, Any]]:

    base = predictor_cls()
    model_name = base.model_name
    logger.info("=== Model: %s ===", model_name)

    best_params: Dict[str, Any] = {}

    # --- tuning on TRAIN only ---
    if not args.skip_tuning and param_dist:
        t0 = time.time()
        tscv = TimeSeriesSplit(n_splits=args.cv_splits)
        tuner = RandomizedSearchCV(
            estimator=base.build_pipeline(),
            param_distributions=param_dist,
            n_iter=args.tune_iterations,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            n_jobs=-1,
            random_state=SEED,
        )
        tuner.fit(X_train, y_train)
        best_params = tuner.best_params_
        logger.info("Best params (%s): %s (tuned in %.2fs)", model_name, best_params, time.time() - t0)

    # --- final model on TRAIN+VAL ---
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    final_model = predictor_cls()
    t1 = time.time()
    final_model.train(
        X_train=X_train_val,
        y_train=y_train_val,
        hyperparams=best_params,
        sample_weights=combined_weights,
    )
    logger.debug("%s final_model trained in %.2fs", model_name, time.time() - t1)
    final_model.save_model()

    # --- val-model on TRAIN only for unbiased VAL preds ---
    val_model = predictor_cls()
    t2 = time.time()
    val_model.train(
        X_train=X_train,
        y_train=y_train,
        hyperparams=best_params,
        sample_weights=train_weights,
    )
    logger.debug("%s val_model trained in %.2fs", model_name, time.time() - t2)
    val_preds = val_model.predict(X_val)

    return final_model, val_preds, best_params


# =============================================================================
# Main pipeline
# =============================================================================

def run_training_pipeline(args: argparse.Namespace) -> None:
    t0 = time.time()
    logger.info("--- NFL Training Pipeline ---")
    logger.info("Args: %s", vars(args))

    # 1. Data Loading (this part is correct)
    sb = get_supabase_client()
    if not sb:
        sys.exit(1)

    fetcher = NFLDataFetcher(sb)
    games_df = fetcher.fetch_historical_games(args.lookback_days)
    stats_df = fetcher.fetch_historical_team_game_stats(args.lookback_days)
    if games_df.empty or stats_df.empty:
        logger.critical("No historical data loaded.")
        sys.exit(1)

    _log_df(games_df, "games_df_raw", args.debug)
    _log_df(stats_df, "stats_df_raw", args.debug)

    # 2. Instantiate and run the refactored feature engine
    logger.info("Initializing and running the refactored NFLFeatureEngine...")
    feat_t0 = time.time()

    # The engine is now created as a class instance
    nfl_engine = NFLFeatureEngine(
        supabase_url=config.SUPABASE_URL,
        supabase_service_key=config.SUPABASE_SERVICE_KEY
    )

    # Call the build_features method
    features = nfl_engine.build_features(
        games_df=games_df,
        historical_games_df=games_df, # Pass the same df for history context
        historical_team_stats_df=stats_df,
        debug=args.debug
    )

    # Add sum-style composites for TOTAL modeling
    features = add_total_composites(features, keys=[
        "rolling_points_for_avg",
        "rolling_points_against_avg",
        "rolling_yards_per_play_avg",
        "rolling_turnover_differential_avg",
        "form_win_pct_5",
        "rest_days",
    ])


    logger.info("Feature pipeline complete in %.2fs | shape=%s", time.time() - feat_t0, features.shape)
    _log_df(features, "features_full", args.debug)
    _log_df(features, "features_full", args.debug)

    # Targets (safe; composites don’t touch labels)
    features = features.dropna(subset=["home_score", "away_score"])
    features["margin"] = features["home_score"] - features["away_score"]
    features["total"]  = features["home_score"] + features["away_score"]

    # Now build target-specific candidates
    cands = build_target_specific_candidates(features)
    logger.info(
        "Candidate features after filters: margin=%d | total=%d",
        len(cands["margin"]), len(cands["total"])
    )

    # visibility on bucket makeup
    _summarize_candidates("margin", cands["margin"])
    _summarize_candidates("total",  cands["total"])


    # 4. Chronological split
    train_df, val_df, test_df = chronological_split(
        features,
        test_size=args.test_size,
        val_size=args.val_size,
        date_col="game_date",
    )
    logger.info("Split sizes → train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    _log_df(train_df, "train_df", args.debug)
    _log_df(val_df,   "val_df",   args.debug)
    _log_df(test_df,  "test_df",  args.debug)

    # 5. Feature selection (per target)
    #feats_margin = run_feature_selection(
        #train_df, train_df["margin"], cands["margin"], "margin", debug=args.debug
   # )
    #feats_total = run_feature_selection(
        #train_df, train_df["total"], cands["total"], "total", debug=args.debug
    #)

    # 6. Recency weights
    recency_w_train    = compute_recency_weights(train_df["game_date"])
    combined_dates     = pd.concat([train_df, val_df])["game_date"]
    recency_w_combined = compute_recency_weights(combined_dates)

    if args.debug:
        logger.debug("Recency weights: train len=%s, combined len=%s",
                     len(recency_w_train) if recency_w_train is not None else None,
                     len(recency_w_combined) if recency_w_combined is not None else None)

    # 7. Model dictionaries & param grids
    margin_models: Dict[str, Type[BaseNFLPredictor]] = {
        "ridge": RidgeMarginPredictor,
        "svr": SVRMarginPredictor,
    }
    total_models: Dict[str, Type[BaseNFLPredictor]] = {
        "xgb": XGBoostTotalPredictor,
        "rf": RFTotalPredictor,
    }

    param_dists: Dict[str, Dict[str, Any]] = {
        "xgb": {
            "model__n_estimators":      randint(200, 600),       # [200, 600)
            "model__max_depth":         randint(2, 6),           # [2, 6)
            "model__learning_rate":     loguniform(1e-3, 5e-2),  # (0.001, 0.05)
            # use UNIFORM here: uniform(loc, scale) => [loc, loc+scale)
            "model__subsample":         uniform(0.6, 0.4),       # [0.6, 1.0)
            "model__colsample_bytree":  uniform(0.6, 0.4),       # [0.6, 1.0)
        },
        "rf": {
            "model__n_estimators":      randint(200, 600),
            "model__max_depth":         randint(4, 12),
            "model__min_samples_leaf":  randint(3, 20),
            "model__max_features":      randint(5, 40),
        },
    }

    # 8. Train loop
    results: Dict[str, Dict[str, Any]] = {}

    for target, model_map in [("margin", margin_models), ("total", total_models)]:
        # One selection pass per target
        feat_list = run_feature_selection(
            X=train_df,                  # ok to pass full frame; selector filters by cands
            y=train_df[target],
            feature_candidates=cands[target],
            target_name=target,
            debug=args.debug,
        )

        _summarize_selected(target, feat_list)

        logger.info("\n=== Target: %s ===", target.upper())
        X_train, y_train = train_df[feat_list], train_df[target]
        X_val,   y_val   = val_df[feat_list],   val_df[target]

        _log_df(X_train, f"X_train_{target}", args.debug)
        _log_df(X_val,   f"X_val_{target}",   args.debug)

        val_preds_dict: Dict[str, pd.Series] = {}

        for key, cls in model_map.items():
            t_model = time.time()
            _, val_preds, _ = train_and_get_val_preds(
                predictor_cls=cls,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                param_dist=param_dists.get(key),
                args=args,
                combined_weights=recency_w_combined,
                train_weights=recency_w_train,
            )
            logger.debug("Model %s (%s) total train+val cycle: %.2fs", cls.__name__, target, time.time() - t_model)
            val_preds_dict[cls().model_name] = val_preds

        weights = optimize_ensemble_weights(val_preds_dict, y_val, debug=args.debug)
        weights_path = MODELS_DIR / f"nfl_{target}_ensemble_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=4)
        logger.info("Ensemble weights saved → %s", weights_path)

        results[target] = {"weights": weights, "features": feat_list}

    # 9. Final test evaluation
    logger.info("\n=== Final Ensemble Evaluation (TEST) ===")
    final_preds: Dict[str, pd.Series] = {}
    for target, meta in results.items():
        feat_list = meta["features"]
        weights   = meta["weights"]

        X_test = test_df.reindex(columns=feat_list, fill_value=0.0)
        _log_df(X_test, f"X_test_{target}", args.debug)

        ensemble = pd.Series(0.0, index=test_df.index)
        for model_name, w in weights.items():
            model = load_model_by_name(model_name)
            preds = model.predict(X_test)
            ensemble += preds * w
        final_preds[target] = ensemble

    scores = derive_scores_from_predictions(final_preds["margin"], final_preds["total"])
    scores["actual_home_score"] = test_df["home_score"]
    scores["actual_away_score"] = test_df["away_score"]

    mae_home = mean_absolute_error(scores["actual_home_score"], scores["predicted_home_score"])
    mae_away = mean_absolute_error(scores["actual_away_score"], scores["predicted_away_score"])
    overall_mae = 0.5 * (mae_home + mae_away)
    logger.info("TEST MAE → overall=%.3f | home=%.3f | away=%.3f", overall_mae, mae_home, mae_away)

    if args.debug:
        logger.debug("Score head:\n%s", scores.head())

    # 10. Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"nfl_eval_report_{timestamp}.txt"
    generate_evaluation_report(scores, overall_mae, report_path)

    plot_actual_vs_predicted(
        scores["actual_home_score"],
        scores["predicted_home_score"],
        title="Actual vs Predicted Home Score (Test)",
        save_path=REPORTS_DIR / f"test_home_score_{timestamp}.png",
    )
    plot_actual_vs_predicted(
        scores["actual_away_score"],
        scores["predicted_away_score"],
        title="Actual vs Predicted Away Score (Test)",
        save_path=REPORTS_DIR / f"test_away_score_{timestamp}.png",
    )

    logger.info("Pipeline done in %.2fs", time.time() - t0)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NFL Score Prediction – Train Models")
    p.add_argument("--lookback-days", type=int, default=1825, help="~5 seasons of history")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--tune-iterations", type=int, default=50)
    p.add_argument("--cv-splits", type=int, default=DEFAULT_CV_FOLDS)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    run_training_pipeline(args)
