# backend/nfl_score_prediction/train_models.py
"""
NFL Score Prediction – Model Training Pipeline (with rich DEBUG logging)

Key changes vs. prior:
- Mean imputation (no zeros) for feature selection, VAL, and TEST.
- Availability gating on TRAIN + VAL to avoid selecting brittle features.
- Conditional MI top-up freeze for DRIVE bucket on TOTAL.
- Small robust seeding of pace/points features for TOTAL when reliably present.
- Slightly widened model search spaces to recover upside tails (totals).
- Recency weighting sanity logs; missingness reports for VAL/TEST.
- Persist train feature means per target for inference-time imputation.

Flow:
1. Load historical data via NFLDataFetcher.
2. Build features with NFLFeatureEngine.
3. Create targets: margin (home - away) and total (home + away).
4. Feature selection (LassoCV/ElasticNetCV) per target with availability gating.
5. Chronological split → train / val / test.
6. Train individual models (with optional tuning), get val preds.
7. Optimize ensemble weights on val.
8. Retrain models on train+val and save.
9. Evaluate ensemble on test, save reports & plots, persist feature means.
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

def _drop_constant_columns(df: pd.DataFrame, *, exclude: tuple[str, ...] = ("game_id",), debug: bool = False) -> pd.DataFrame:
    """Drop columns that are all-NaN or have a single unique value."""
    if df.empty:
        return df
    cols = [c for c in df.columns if c not in exclude]
    nunq = {c: df[c].nunique(dropna=False) for c in cols}
    const_cols = [c for c, k in nunq.items() if k <= 1]
    if const_cols:
        if debug:
            logger.debug("TRAIN: dropping %d constant cols (sample=%s)", len(const_cols), const_cols[:18])
        else:
            logger.info("TRAIN: dropping %d constant cols", len(const_cols))
        df = df.drop(columns=const_cols, errors="ignore")
    return df

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
    # numeric universe
    num = set(features.select_dtypes(include=np.number).columns)

    # global include & exclude
    ALLOW_PREFIX = (
        "home_", "away_", "rolling_", "season_", "h2h_", "momentum_",
        "rest_", "form_", "adv_", "drive_", "situational_", "total_",
        "h_", "a_", "map_",
    )
    BLOCK = {"home_score", "away_score", "margin", "total"}

    # base = prefixed, numeric, non-blocked
    base = [c for c in num if any(c.startswith(p) for p in ALLOW_PREFIX) and c not in BLOCK]

    # --- bucket helpers ---
    is_diff   = lambda c: c.endswith("_diff") or "_advantage" in c
    is_points = lambda c: ("points_for" in c) or ("points_against" in c) or ("total_points" in c) or c.startswith("total_")
    is_rest   = lambda c: ("rest_days" in c) or ("short_week" in c) or ("off_bye" in c)
    is_form   = lambda c: ("form_win_pct" in c) or ("current_streak" in c) or ("momentum" in c)
    is_h2h    = lambda c: c.startswith("h2h_")

    # --- season priors helpers ---
    is_season = lambda c: (
        "prev_season_" in c
        or c.startswith("home_prev_season_")
        or c.startswith("away_prev_season_")
    )
    is_season_diff = lambda c: is_season(c) and c.endswith("_diff")
    is_side_specific_season = lambda c: (
        c.startswith("home_prev_season_") or c.startswith("away_prev_season_")
    )

    # pull season diffs from the full numeric set (NOT just base) so prev_season_* are included
    season_diffs_global = [c for c in num if is_season_diff(c)]

    # =========================
    # MARGIN candidates
    # =========================
    margin_core = [
        c for c in base if (
            is_diff(c) or is_rest(c) or is_form(c)
            or ("turnover_differential" in c)
            or (is_h2h(c) and ("home_win_pct" in c))
        )
    ]
    margin_core += [c for c in base if "point_differential" in c]
    margin_candidates = season_diffs_global + margin_core
    margin_candidates = [c for c in margin_candidates if not is_side_specific_season(c)]
    seen = set()
    margin_candidates = [c for c in margin_candidates if not (c in seen or seen.add(c))]

    # =========================
    # TOTAL candidates
    # =========================
    is_volume = lambda c: any(k in c for k in (
        "yards_per_play", "plays_per_game", "seconds_per_play",
        "neutral_pass_rate", "red_zone", "explosive_play_rate",
        "drive_success_rate", "drive_plays", "drive_yards",
    ))

    total_core = [
        c for c in base if (is_points(c) or is_volume(c) or is_rest(c) or (is_h2h(c) and ("avg_total_points" in c)))
    ]
    total_core += [c for c in base if is_form(c) and ("form_win_pct" in c)]
    total_candidates = season_diffs_global + total_core
    seen = set()
    total_candidates = [c for c in total_candidates if not (c in seen or seen.add(c))]

    return {"margin": margin_candidates, "total": total_candidates}


# ---------- helpers for selection & robustness ----------

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
        # Mean imputation here as well
        Xm = X[base_selected].copy()
        Xm = Xm.replace([np.inf, -np.inf], np.nan)
        Xm = Xm.fillna(Xm.mean(numeric_only=True))
        Xs = scaler.fit_transform(Xm)
        model = LassoCV(alphas=np.logspace(-4, -1, 40), cv=TimeSeriesSplit(n_splits=DEFAULT_CV_FOLDS), random_state=s, n_jobs=-1, max_iter=10000).fit(Xs, y)
        keep = [c for c, coef in zip(base_selected, model.coef_) if abs(coef) > 1e-5]
        hits.loc[keep] += 1
    stable = list(hits[hits / len(seeds) >= floor].index)
    return stable if stable else base_selected

def _bucket_for(col: str) -> str:
    c = col.lower()
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

def _feature_availability(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Non-null ratio per column."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(dtype=float)
    return 1.0 - df[present].isna().mean()

def _filter_by_availability(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidates: list[str],
    min_avail: float
) -> list[str]:
    """Keep only features that are sufficiently available on both TRAIN and VAL."""
    if not candidates:
        return candidates
    a_train = _feature_availability(train_df, candidates)
    a_val   = _feature_availability(val_df, candidates)
    keep = []
    for c in candidates:
        at = float(a_train.get(c, 0.0))
        av = float(a_val.get(c, 0.0))
        if at >= min_avail and av >= min_avail:
            keep.append(c)
    return keep

def _compute_feature_means(X: pd.DataFrame) -> Dict[str, float]:
    return {c: float(X[c].mean()) for c in X.columns}

def _apply_means_imputation(df: pd.DataFrame, feat_list: list[str], feat_means: Dict[str, float]) -> pd.DataFrame:
    """Reindex and fill NaNs using train means (no zeros)."""
    X = df.reindex(columns=feat_list)
    # pandas fillna accepts a dict mapping column -> value
    X = X.fillna(value=feat_means)
    # Any remaining (all-NaN columns not in dict) get filled by per-col mean=0 guard
    remain_nulls = X.columns[X.isna().any()].tolist()
    if remain_nulls:
        for c in remain_nulls:
            X[c] = X[c].fillna(feat_means.get(c, 0.0))
    return X

def _missingness_report(name: str, df: pd.DataFrame):
    if df.empty:
        return
    nn = 1.0 - df.isna().mean()
    worst = nn.sort_values().head(15)
    logger.info("MISSINGNESS[%s] worst-15 availability:\n%s", name, worst.to_string())


def _ensure_bucket_mi(
    X: pd.DataFrame,
    y: pd.Series,
    selected: list[str],
    target_name: str,
    min_per_bucket: int = 1,
    topk_per_bucket: int = 2,
    *,
    freeze_drive_bucket: bool = False,
) -> list[str]:
    """
    Ensure each bucket contributes at least `min_per_bucket` features; add via MI if needed.
    If `freeze_drive_bucket` is True, the 'drive' bucket will NOT be auto-topped up by MI.
    """
    sel_set = set(selected)
    buckets: dict[str, list[str]] = {}
    for c in X.columns:
        buckets.setdefault(_bucket_for(c), []).append(c)

    for bucket, cols in buckets.items():
        if freeze_drive_bucket and bucket == "drive":
            continue

        have = [c for c in selected if _bucket_for(c) == bucket]
        if len(have) >= min_per_bucket:
            continue

        # Mean impute here too
        cols = [c for c in cols if c not in sel_set and X[c].nunique(dropna=False) > 1]
        if not cols:
            continue

        Xb = X[cols].copy()
        Xb = Xb.replace([np.inf, -np.inf], np.nan).fillna(Xb.mean(numeric_only=True))
        cols = [c for c in cols if Xb[c].std(ddof=0) > 0]
        if not cols:
            continue

        mi = mutual_info_regression(Xb.values, y.values, random_state=SEED)
        order = np.argsort(mi)[::-1]
        add = [cols[i] for i in order[:topk_per_bucket]]
        for a in add:
            sel_set.add(a)

    out = [c for c in selected] + [c for c in sel_set if c not in selected]
    return out


# ---------- selection driver ----------

def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_candidates: list[str],
    target_name: str,
    save_dir: Path = MODELS_DIR,
    debug: bool = False,
    *,
    freeze_drive_bucket: bool = False,
) -> list[str]:
    logger.info("Lasso/EN feature selection → %s (freeze_drive_bucket=%s)", target_name, freeze_drive_bucket)

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
    X_sel = X_sel.replace([np.inf, -np.inf], np.nan)
    # MEAN IMPUTATION (TRAIN-SIDE)
    X_sel = X_sel.fillna(X_sel.mean(numeric_only=True))

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
        freeze_drive_bucket=freeze_drive_bucket,
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

    _summarize_selected(target_name, selected)

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

    # 1. Data Loading
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

    # 2. Feature engine
    logger.info("Initializing and running the refactored NFLFeatureEngine...")
    feat_t0 = time.time()

    nfl_engine = NFLFeatureEngine(
        supabase_url=config.SUPABASE_URL,
        supabase_service_key=config.SUPABASE_SERVICE_KEY,
    )

    features = nfl_engine.build_features(
        games_df=games_df,
        historical_games_df=games_df,
        historical_team_stats_df=stats_df,
        debug=args.debug,
        # DRIVE module controls
        window=args.drive_window,
        reset_by_season=args.drive_reset_by_season,
        min_prior_games=args.drive_min_prior_games,
        soft_fail=args.drive_soft_fail,
        disable_drive_stage=args.disable_drive_stage,
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

    # train-side constant dropper
    features = _drop_constant_columns(features, exclude=("game_id",), debug=args.debug)

    logger.info("Feature pipeline complete in %.2fs | shape=%s", time.time() - feat_t0, features.shape)
    _log_df(features, "features_full", args.debug)

    # Targets
    features = features.dropna(subset=["home_score", "away_score"])
    if features.empty:
        logger.critical("After dropping rows without labels, no data remains.")
        sys.exit(1)

    features["margin"] = features["home_score"] - features["away_score"]
    features["total"]  = features["home_score"] + features["away_score"]

    # Label sanity: simple stats
    label_stats = features[["home_score","away_score","total"]].describe().loc[["mean","50%","max","std"]]
    logger.info("LABELS stats (TRAIN+VAL+TEST overall snapshot):\n%s", label_stats.to_string())

    # Candidates (pre-filter)
    cands = build_target_specific_candidates(features)

    # Required seeds (margin priors always allowed)
    REQUIRED_MARGIN = ["prev_season_win_pct_diff", "prev_season_srs_lite_diff"]
    # TOTAL robust seeds: conditionally add if they meet availability threshold later
    TOTAL_SEED_CANDIDATES = [
        "rolling_points_for_avg_diff",
        "rolling_yards_per_play_avg_diff",
        "total_rolling_points_for_avg",
    ]

    def _seed_required(cands_list, required):
        req = [c for c in required if c in cands_list]
        seen = set(req)
        tail = [c for c in cands_list if c not in seen]
        return req + tail

    margin_candidates = _seed_required(cands["margin"], REQUIRED_MARGIN)
    total_candidates  = cands["total"][:]  # TOTAL seeds are determined after availability check

    _summarize_candidates("margin (pre-split)", margin_candidates)
    _summarize_candidates("total (pre-split)",  total_candidates)

    # 4) Split
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

    # 5) Availability gating on TRAIN + VAL
    min_avail = float(args.min_availability)
    margin_after_avail = _filter_by_availability(train_df, val_df, margin_candidates, min_avail=min_avail)
    total_after_avail  = _filter_by_availability(train_df, val_df, total_candidates,  min_avail=min_avail)

    # Add TOTAL robust seeds if they pass availability
    robust_total_seeds = [c for c in TOTAL_SEED_CANDIDATES if c in total_after_avail]
    REQUIRED_TOTAL = robust_total_seeds  # may be empty

    def _variance_filter(df, cols, min_var=1e-8, required=()):
        keep = []
        for c in cols:
            if c in required:
                keep.append(c); continue
            if c in df.columns and float(df[c].var()) > min_var:
                keep.append(c)
        return keep

    def _corr_filter(df, cols, max_corr=0.98, required=()):
        out = []
        for c in cols:
            if c in required:
                out.append(c); continue
            drop = any(
                abs(df[[c, k]].corr().iloc[0, 1]) >= max_corr
                for k in out
                if c in df.columns and k in df.columns
            )
            if not drop:
                out.append(c)
        return out

    margin_after_var = _variance_filter(train_df, margin_after_avail, required=REQUIRED_MARGIN)
    margin_after_cor = _corr_filter(train_df, margin_after_var,  required=REQUIRED_MARGIN)

    total_after_var  = _variance_filter(train_df, total_after_avail,  required=REQUIRED_TOTAL)
    total_after_cor  = _corr_filter(train_df, total_after_var,       required=REQUIRED_TOTAL)

    # Bucket makeup (filtered by availability)
    _summarize_candidates("margin (post-avail, pre-select)", margin_after_cor)
    _summarize_candidates("total (post-avail, pre-select)",  total_after_cor)

    # 6) Final feature selection on TRAIN only
    # Conditional DRIVE freeze for TOTAL: freeze if we have zero reliable DRIVE cols
    drive_in_total = [c for c in total_after_cor if _bucket_for(c) == "drive"]
    freeze_drive_total = (len(drive_in_total) == 0) or bool(args.freeze_drive_mi)

    margin_feat_list = run_feature_selection(
        X=train_df,
        y=train_df["margin"],
        feature_candidates=margin_after_cor,
        target_name="margin",
        debug=args.debug,
        freeze_drive_bucket=False,
    )

    # Ensure REQUIRED_MARGIN on top
    if REQUIRED_MARGIN:
        margin_feat_list = REQUIRED_MARGIN + [f for f in margin_feat_list if f not in REQUIRED_MARGIN]

    total_feat_list = run_feature_selection(
        X=train_df,
        y=train_df["total"],
        feature_candidates=total_after_cor,
        target_name="total",
        debug=args.debug,
        freeze_drive_bucket=freeze_drive_total,
    )

    # Ensure REQUIRED_TOTAL (robust pace/points) on top, if any
    if REQUIRED_TOTAL:
        total_feat_list = REQUIRED_TOTAL + [f for f in total_feat_list if f not in REQUIRED_TOTAL]

    _summarize_selected("margin", margin_feat_list)
    _summarize_selected("total",  total_feat_list)

    logger.info("Final MARGIN features (n=%d): %s", len(margin_feat_list), margin_feat_list[:15])
    logger.info("Final TOTAL  features (n=%d): %s", len(total_feat_list),  total_feat_list[:15])

    # 7) Recency weights + sanity logs
    recency_w_train    = compute_recency_weights(train_df["game_date"])
    combined_dates     = pd.concat([train_df, val_df])["game_date"]
    recency_w_combined = compute_recency_weights(combined_dates)
    if args.debug:
        logger.debug("Recency weights: train len=%s, combined len=%s",
                     len(recency_w_train) if recency_w_train is not None else None,
                     len(recency_w_combined) if recency_w_combined is not None else None)

    # Weighted vs unweighted TRAIN total mean (sanity)
    train_total_mean_unw = float(train_df["total"].mean())
    if recency_w_train is not None:
        w = np.asarray(recency_w_train, dtype=float)
        w = w / (w.sum() + 1e-12)
        train_total_mean_w = float(np.dot(train_df["total"].values, w))
        logger.info("TRAIN total mean (unweighted)=%.2f | (recency-weighted)=%.2f", train_total_mean_unw, train_total_mean_w)
    else:
        logger.info("TRAIN total mean (unweighted)=%.2f | (recency-weighted)=N/A", train_total_mean_unw)

    # 8. Model dictionaries & param grids (slightly widened)
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
            "model__n_estimators":      randint(200, 650),
            "model__max_depth":         randint(2, 8),            # was 2..6
            "model__learning_rate":     loguniform(1e-3, 8e-2),   # was up to 5e-2
            "model__subsample":         uniform(0.6, 0.4),
            "model__colsample_bytree":  uniform(0.6, 0.4),
        },
        "rf": {
            "model__n_estimators":      randint(200, 650),
            "model__max_depth":         randint(4, 13),
            "model__min_samples_leaf":  randint(2, 17),           # lower floor for more flexibility
            "model__max_features":      randint(5, 40),           # includes sqrt-ish for many p
        },
    }

    # 9. Train loop
    results: Dict[str, Dict[str, Any]] = {}

    for target, model_map in [("margin", margin_models), ("total", total_models)]:
        feat_list = margin_feat_list if target == "margin" else total_feat_list

        # Build TRAIN, VAL with mean imputation (no zeros)
        X_train = train_df.reindex(columns=feat_list)
        # Train-side mean imputation for selection consistency
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean(numeric_only=True))
        y_train = train_df[target]

        # Persistable means (used for VAL/TEST imputation and for inference)
        feat_means = _compute_feature_means(X_train)

        # VAL matrix using TRAIN means
        X_val = _apply_means_imputation(val_df, feat_list, feat_means)
        y_val = val_df[target]

        _log_df(X_train, f"X_train_{target}", args.debug)
        _log_df(X_val,   f"X_val_{target}",   args.debug)

        # Missingness reports
        _missingness_report(f"VAL[{target}]", X_val)

        # Train/val models, gather val preds
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

        # Persist feature list & means for inference-time parity
        with open(MODELS_DIR / f"nfl_{target}_selected_features.json", "w") as f:
            json.dump(feat_list, f, indent=4)
        with open(MODELS_DIR / f"nfl_{target}_feature_means.json", "w") as f:
            json.dump(feat_means, f, indent=4)

        results[target] = {"weights": weights, "features": feat_list, "means": feat_means}

    # 10. Final test evaluation (use TRAIN means per target)
    logger.info("\n=== Final Ensemble Evaluation (TEST) ===")
    final_preds: Dict[str, pd.Series] = {}

    for target, meta in results.items():
        feat_list = meta["features"]
        weights   = meta["weights"]
        feat_means = meta["means"]

        X_test = _apply_means_imputation(test_df, feat_list, feat_means)
        _log_df(X_test, f"X_test_{target}", args.debug)
        _missingness_report(f"TEST[{target}]", X_test)

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

    # 11. Reports
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

    # Availability & selection controls
    p.add_argument("--min-availability", type=float, default=0.80,
                   help="Minimum non-null ratio required on TRAIN and VAL for a feature to be eligible.")
    p.add_argument("--freeze-drive-mi", action="store_true",
                   help="Force MI to skip DRIVE bucket top-ups across targets (TOTAL also conditionally freezes).")

    # DRIVE knobs
    p.add_argument("--disable-drive-stage", action="store_true",
                   help="Skip the DRIVE feature module entirely for A/B tests.")
    p.add_argument("--drive-window", type=int, default=5,
                   help="Rolling window (games) for DRIVE pre-game averages.")
    p.add_argument("--drive-reset-by-season", dest="drive_reset_by_season", action="store_true",
                   help="Reset DRIVE rolling windows at season boundaries (default).")
    p.add_argument("--no-drive-reset-by-season", dest="drive_reset_by_season", action="store_false",
                   help="Do not reset DRIVE rolling windows at season boundaries.")
    p.set_defaults(drive_reset_by_season=True)
    p.add_argument("--drive-min-prior-games", type=int, default=0,
                   help="If >0, mask DRIVE *_avg to NaN when prior games < threshold.")
    p.add_argument("--drive-soft-fail", dest="drive_soft_fail", action="store_true",
                   help="Soft-fail DRIVE on error and keep pipeline running (default).")
    p.add_argument("--no-drive-soft-fail", dest="drive_soft_fail", action="store_false",
                   help="Raise on DRIVE errors instead of soft-failing.")
    p.set_defaults(drive_soft_fail=True)

    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    run_training_pipeline(args)