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
from scipy.stats import randint, loguniform
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from supabase import Client, create_client

# ---- Project imports ----
try:
    from backend import config
    from backend.nfl_features.engine import run_nfl_feature_pipeline
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
    "h_",
    "a_",
    "rolling_",
    "season_",
    "h2h_",
    "momentum_",
    "rest_",
    "form_",
    "adv_",
    "drive_",
    "situational_",
)
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


def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_candidates: List[str],
    target_name: str,
    save_dir: Path = MODELS_DIR,
    debug: bool = False,
) -> List[str]:
    logger.info("LassoCV feature selection → %s", target_name)
    X_sel = X[feature_candidates].copy().fillna(0.0)

    _log_df(X_sel, f"X_sel_{target_name}_before_FS", debug)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    lasso = LassoCV(
        cv=DEFAULT_CV_FOLDS,
        random_state=SEED,
        n_jobs=-1,
        max_iter=2000,
    ).fit(X_scaled, y)

    mask = np.abs(lasso.coef_) > 1e-5
    selected = list(pd.Index(feature_candidates)[mask])

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

    # 1. Data
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

    # 2. Features
    logger.info("Generating features...")
    feat_t0 = time.time()
    features = run_nfl_feature_pipeline(games_df, stats_df, debug=args.debug)
    logger.info("Feature pipeline complete in %.2fs | shape=%s", time.time() - feat_t0, features.shape)
    _log_df(features, "features_full", args.debug)

    # targets
    features = features.dropna(subset=["home_score", "away_score"])
    features["margin"] = features["home_score"] - features["away_score"]
    features["total"] = features["home_score"] + features["away_score"]
    _log_df(features[["home_score", "away_score", "margin", "total"]], "targets_added", args.debug)

    # 3. Candidate features
    numeric_cols = features.select_dtypes(include=np.number).columns
    candidates = [
        c for c in numeric_cols if c.startswith(NFL_SAFE_FEATURE_PREFIXES) or c in NFL_SAFE_EXACT_FEATURE_NAMES
    ]
    logger.debug("Candidate features (numeric & safe prefixes): %d", len(candidates))

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
    feats_margin = run_feature_selection(train_df, train_df["margin"], candidates, "margin", debug=args.debug)
    feats_total  = run_feature_selection(train_df, train_df["total"],  candidates, "total",  debug=args.debug)

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
            "model__n_estimators": randint(200, 600),
            "model__max_depth": randint(2, 6),
            "model__learning_rate": loguniform(1e-3, 5e-2),
            "model__subsample": loguniform(0.5, 0.5),
            "model__colsample_bytree": loguniform(0.5, 0.5),
        },
        "rf": {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": randint(4, 12),
            "model__min_samples_leaf": randint(3, 20),
            "model__max_features": randint(5, 40),
        },
    }

    # 8. Train loop
    results: Dict[str, Dict[str, Any]] = {}
    for target, model_map, feat_list in [
        ("margin", margin_models, feats_margin),
        ("total",  total_models,  feats_total),
    ]:
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
