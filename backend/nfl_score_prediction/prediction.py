# backend/nfl_score_prediction/prediction.py
"""
prediction.py – NFL Score Prediction Generation Script (with rich DEBUG logging)

Pipeline:
1. Fetch upcoming games (window) + historical context (lookback).
2. Build features via NFLFeatureEngine().build_features().
3. Load artifacts (selected features + ensemble weights).
4. Load models with NFLEnsemble, predict margin & total.
5. Derive home/away scores, upsert to Supabase.
6. Optional: print human-readable summary.

Debug instrumentation (when --debug):
- CLI focus selectors: --focus-date, --focus-games (away@home), --focus-ids.
- Selected-features audit per target (counts, first ~20, missing).
- Frame audit per target (row count, null_top, non-numeric cols, order_crc).
- Focused-row feature snapshots (bucketed) before per-model predictions.
- Per-model raw predictions for focused rows.
- Ensemble weights per target; optional TOTAL clip pre/post with flags.
- Rolling freshness proxy via *_imputed flags or rolling-support counts if present.
- Priors sanity (prev season + advanced diffs).
- Optional CSV dump per focused game (selected vectors, per-model, ensemble, flags).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from supabase import Client, create_client

# Project imports
try:
    from backend import config
    from backend.nfl_features.engine import NFLFeatureEngine
    from backend.nfl_score_prediction.data import NFLDataFetcher
    from backend.nfl_score_prediction.ensemble import NFLEnsemble
    from backend.nfl_score_prediction.models import MODEL_DIR, derive_scores_from_predictions
except ImportError as e:
    print(f"Import error: {e}. Ensure PYTHONPATH includes project root.")
    sys.exit(1)

# ----------------------------------------------------------------------------- #
# Logging / paths
# ----------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ----------------------------------------------------------------------------- #
# Small diagnostics helpers
# ----------------------------------------------------------------------------- #
def _df_profile(df: pd.DataFrame, name: str, top_n: int = 10) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"name": name, "rows": 0, "cols": 0, "null_top": {}, "zero_top": {}}
    nulls = df.isna().sum().sort_values(ascending=False)
    num_cols = df.select_dtypes(include="number").columns
    zeros = (df[num_cols] == 0).sum().sort_values(ascending=False) if len(num_cols) else pd.Series(dtype=int)
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": {k: int(v) for k, v in nulls.head(top_n).to_dict().items() if v > 0},
        "zero_top": {k: int(v) for k, v in zeros.head(top_n).to_dict().items() if v > 0},
    }

def _log_df(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    prof = _df_profile(df, name)
    logger.debug(
        "[DF] %-22s rows=%-5d cols=%-4d | null_top=%s | zero_top=%s",
        prof["name"], prof["rows"], prof["cols"], prof.get("null_top"), prof.get("zero_top")
    )

def _order_crc(cols: Sequence[str]) -> str:
    m = hashlib.md5()
    m.update(",".join(list(cols)).encode("utf-8"))
    return m.hexdigest()[:8]

def _check_features(X: pd.DataFrame, needed: Sequence[str], label: str, debug: bool):
    if not debug:
        return
    missing = [c for c in needed if c not in X.columns]
    extra   = [c for c in X.columns if c not in needed]
    if missing:
        logger.debug("[%s][WARN] missing=%d: %s", label, len(missing), missing[:20])
    if extra:
        logger.debug("[%s] extra=%d (ignored in predict matrix)", label, len(extra))
    non_numeric = [c for c in needed if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        logger.debug("[%s][WARN] non-numeric columns: %s", label, non_numeric)

# ----------------------------------------------------------------------------- #
# Focus selection
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Focus:
    date_str: Optional[str]                 # 'YYYY-MM-DD'
    pairs: Set[Tuple[int, int]]             # {(away_id, home_id), ...}
    ids: Set[int]                           # {game_id, ...}

def _parse_focus_games(arg: Optional[str]) -> Set[Tuple[int, int]]:
    if not arg:
        return set()
    out: Set[Tuple[int, int]] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "@" not in tok:
            continue
        a, h = tok.split("@", 1)
        try:
            out.add((int(a), int(h)))
        except ValueError:
            continue
    return out

def _parse_focus_ids(arg: Optional[str]) -> Set[int]:
    if not arg:
        return set()
    out: Set[int] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out

def _normalize_date_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        # Accept YYYY-MM-DD or other parseable forms
        d = datetime.fromisoformat(s.strip())
        return d.date().isoformat()
    except Exception:
        # Try plain date
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date().isoformat()
        except Exception:
            return None

def _is_focused_row(
    row: Mapping[str, Any],
    focus: Focus
) -> bool:
    gid = int(row.get("game_id")) if "game_id" in row and pd.notna(row.get("game_id")) else None
    if focus.ids and gid is not None and gid in focus.ids:
        return True
    if focus.pairs:
        try:
            a = int(row.get("away_team_id"))
            h = int(row.get("home_team_id"))
            if (a, h) in focus.pairs:
                return True
        except Exception:
            pass
    if focus.date_str:
        gd = row.get("game_date")
        if isinstance(gd, (datetime, date)):
            gd_str = gd.date().isoformat() if isinstance(gd, datetime) else gd.isoformat()
        else:
            gd_str = str(gd)[:10] if gd else None
        if gd_str == focus.date_str:
            return True
    return False

def _ensure_season_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure season is populated (derived from game_date) and core IDs are numeric.
    NFL season rule: months 9–12 => that year, months 1–8 => previous year.
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    # Normalize dates
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce", utc=False)

        def _season_from_date(d: pd.Timestamp) -> Optional[int]:
            if pd.isna(d):
                return None
            y = d.year
            return y if d.month >= 9 else (y - 1)

        derived = out["game_date"].apply(_season_from_date)
        if "season" not in out.columns:
            out["season"] = derived
        else:
            # Fill only missing season values
            out["season"] = out["season"].where(out["season"].notna(), derived)

    # Coerce key IDs to numeric Int64
    for col in ("game_id", "home_team_id", "away_team_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_supabase_client() -> Optional[Client]:
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error("Supabase client init failed: %s", e)
        return None

def load_prediction_artifacts(model_dir: Path, debug: bool = False) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for target in ("margin", "total"):
        feats_path = model_dir / f"nfl_{target}_selected_features.json"
        wts_path   = model_dir / f"nfl_{target}_ensemble_weights.json"
        try:
            feats = json.loads(feats_path.read_text())
            wts   = json.loads(wts_path.read_text())
            artifacts[f"{target}_features"] = feats
            artifacts[f"{target}_weights"]  = wts
            if debug:
                logger.debug("PRED[SETUP] %s selected features (n=%d): %s",
                             target.upper(), len(feats), feats[:20])
        except FileNotFoundError as e:
            logger.critical("Missing artifact for %s: %s", target, e)
            return {}
    return artifacts

def add_total_composites(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """
    Match train-time composites: total_<k> = home_<k> + away_<k>.
    Safe if some columns are missing.
    """
    out = df.copy()
    for k in keys:
        h, a = f"home_{k}", f"away_{k}"
        if h in out.columns and a in out.columns:
            out[f"total_{k}"] = out[h] + out[a]
    return out

def chunked(lst: List[Dict[str, Any]], size: int = 500):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def upsert_score_predictions(preds: List[Dict[str, Any]], sb: Client, debug: bool = False) -> int:
    """
    Update existing schedule rows with predictions.
    """
    if not preds:
        return 0

    updated = 0
    for p in preds:
        gid = int(p["game_id"])
        payload = {
            "predicted_home_score": float(p["predicted_home_score"]),
            "predicted_away_score": float(p["predicted_away_score"]),
        }
        try:
            resp = (
                sb.table("nfl_game_schedule")
                  .update(payload)
                  .eq("game_id", gid)
                  .execute()
            )
            n = len(resp.data or [])
            updated += n
            if debug:
                logger.debug("Update game_id=%s -> modified=%d", gid, n)
        except Exception as e:
            logger.error("Update error for game_id=%s: %s", gid, e)

    logger.info("Updated predictions for %d games", updated)
    return updated

def display_prediction_summary(preds: List[Dict[str, Any]]) -> None:
    if not preds:
        return
    df = pd.DataFrame(preds).sort_values("game_date")
    print("-" * 80)
    print(f"{'DATE':<12}{'MATCHUP':<42}{'PREDICTED SCORE':<20}")
    print("-" * 80)
    for _, r in df.iterrows():
        d = datetime.fromisoformat(str(r["game_date"])).strftime("%Y-%m-%d")
        hn = r.get("home_team_name") or f"home_id {r.get('home_team_id')}"
        an = r.get("away_team_name") or f"away_id {r.get('away_team_id')}"
        matchup = f"{an} @ {hn}"
        score = f"{r['predicted_away_score']:.1f} - {r['predicted_home_score']:.1f}"
        print(f"{d:<12}{matchup:<42}{score:<20}")
    print("-" * 80)

def _clip_feature_columns(
    df: pd.DataFrame,
    bounds: Dict[str, tuple],
    debug: bool = False,
    focus_ids: Optional[set] = None,
) -> pd.DataFrame:
    """
    Clip designated feature columns to given (min, max) bounds.
    - Only clips columns that exist in df.
    - Logs how many values were clipped per column.
    - For focused games, logs pre→post per clipped feature.
    """
    out = df.copy()
    focus_ids = set(focus_ids or [])
    clipped_stats = {}

    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        s = out[col]
        pre = s.copy()

        # Clip
        out[col] = s.clip(lower=lo, upper=hi)

        # Count changes
        changed = (out[col] != pre) & pre.notna()
        n_changed = int(changed.sum())
        if n_changed > 0:
            clipped_stats[col] = n_changed

            if debug and focus_ids:
                for gid in focus_ids:
                    if "game_id" in out.columns:
                        row_mask = out["game_id"] == gid
                    elif out.index.name == "game_id":
                        row_mask = out.index == gid
                    else:
                        row_mask = pd.Series(False, index=out.index)

                    if row_mask.any():
                        pre_v = float(pre[row_mask].iloc[0]) if pd.notna(pre[row_mask].iloc[0]) else None
                        post_v = float(out.loc[row_mask, col].iloc[0]) if pd.notna(out.loc[row_mask, col].iloc[0]) else None
                        if pre_v != post_v:
                            logger.debug("PRED[CLIP] game_id=%s %s: %.3f → %.3f (lo=%.2f hi=%.2f)",
                                         gid, col, pre_v, post_v, lo, hi)

    if debug and clipped_stats:
        logger.debug("PRED[CLIP] counts=%s", clipped_stats)
    elif debug:
        logger.debug("PRED[CLIP] counts={} (no clipping applied)")

    return out

def _attenuate_total_diffs(X_total: pd.DataFrame, alpha: float = 0.6, debug: bool = False, focus_ids: Optional[Set[int]] = None) -> pd.DataFrame:
    """
    Scale down high-variance diff features that tend to over-suppress totals.
    Applied only to the design matrix for TOTAL (post-reindex), optionally only for focus_ids.
    """
    if X_total is None or X_total.empty:
        return X_total
    diffs = [
        "rolling_points_for_avg_diff",
        "rolling_points_against_avg_diff",
        "rolling_yards_per_play_avg_diff",
        "total_rolling_turnover_differential_avg",  # not a diff, but can over-pull
    ]
    cols = [c for c in diffs if c in X_total.columns]
    if not cols:
        return X_total
    X = X_total.copy()
    target_index = X.index.tolist() if not focus_ids else [i for i in X.index if i in focus_ids]
    if len(target_index) == 0:
        return X
    X.loc[target_index, cols] = X.loc[target_index, cols] * float(alpha)
    if debug:
        logger.debug("PRED[ATTN] TOTAL diffs scaled by alpha=%.2f on %d rows, cols=%s", alpha, len(target_index), cols)
    return X

def _shrink_h2h_features(
    df: pd.DataFrame,
    mu_total: float = 44.0,   # league-ish baseline total
    mu_margin: float = 0.0,
    k: float = 6.0,           # ~games to “trust” H2H fully
    debug: bool = False,
    focus_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Shrinks H2H features toward league priors with a weight w = clip(games/k,0,1).
    Works whether or not game_id is the index. Adds global summary logging.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure numeric types (coerce just in case)
    for col in ["h2h_games_played", "h2h_avg_total_points", "h2h_avg_point_diff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Weight based on games played
    if "h2h_games_played" in out.columns:
        w_raw = out["h2h_games_played"].astype(float)
    else:
        w_raw = pd.Series(0.0, index=out.index)

    w = (w_raw / float(k)).clip(lower=0.0, upper=1.0).fillna(0.0)

    def _shrink(series: pd.Series, mu: float) -> Tuple[pd.Series, pd.Series]:
        base = pd.to_numeric(series, errors="coerce").fillna(mu)
        shrunk = mu + w * (base - mu)
        delta = shrunk - base
        return shrunk, delta

    # Keep pre values for deltas
    before_total = out["h2h_avg_total_points"].copy() if "h2h_avg_total_points" in out.columns else None
    before_margin = out["h2h_avg_point_diff"].copy() if "h2h_avg_point_diff" in out.columns else None

    total_delta = None
    margin_delta = None

    if "h2h_avg_total_points" in out.columns:
        out["h2h_avg_total_points"], total_delta = _shrink(out["h2h_avg_total_points"], mu_total)

    if "h2h_avg_point_diff" in out.columns:
        out["h2h_avg_point_diff"], margin_delta = _shrink(out["h2h_avg_point_diff"], mu_margin)

    # Focused per-row logs (unchanged)
    if debug and focus_ids:
        game_id_series = (
            out["game_id"]
            if "game_id" in out.columns
            else pd.Series(out.index, index=out.index, name="game_id")
        )
        for gid in sorted(focus_ids):
            mask = game_id_series.eq(gid)
            if not mask.any():
                logger.debug("PRED[H2H] focus game_id=%s not present in this batch", gid)
                continue
            wv = float(w.loc[mask].iloc[0])
            if before_total is not None and "h2h_avg_total_points" in out.columns:
                old = float(before_total.loc[mask].iloc[0])
                new = float(out.loc[mask, "h2h_avg_total_points"].iloc[0])
                logger.debug(
                    "PRED[H2H] game_id=%s h2h_avg_total_points: %.2f → %.2f (w=%.2f, k=%.1f, mu=%.1f)",
                    gid, old, new, wv, k, mu_total
                )
            if before_margin is not None and "h2h_avg_point_diff" in out.columns:
                old = float(before_margin.loc[mask].iloc[0])
                new = float(out.loc[mask, "h2h_avg_point_diff"].iloc[0])
                logger.debug(
                    "PRED[H2H] game_id=%s h2h_avg_point_diff: %.2f → %.2f (w=%.2f, k=%.1f, mu=%.1f)",
                    gid, old, new, wv, k, mu_margin
                )

    # Global summary (touch counts + average magnitude), always when debug
    if debug:
        touched_rows = 0
        avg_abs_total = None
        avg_abs_margin = None

        if total_delta is not None:
            # A row is "touched" if any H2H column changed from its original value
            total_changed = (total_delta != 0).astype(int)
        else:
            total_changed = pd.Series(0, index=out.index)

        if margin_delta is not None:
            margin_changed = (margin_delta != 0).astype(int)
        else:
            margin_changed = pd.Series(0, index=out.index)

        any_changed = ((total_changed + margin_changed) > 0)
        touched_rows = int(any_changed.sum())

        if total_delta is not None:
            nz = total_delta[any_changed].abs()
            if len(nz) > 0:
                avg_abs_total = float(nz.mean())
        if margin_delta is not None:
            nz = margin_delta[any_changed].abs()
            if len(nz) > 0:
                avg_abs_margin = float(nz.mean())

        logger.debug(
            "PRED[H2H][SUMMARY] rows_touched=%d | avg_abs_delta_total=%s | avg_abs_delta_margin=%s",
            touched_rows,
            f"{avg_abs_total:.3f}" if avg_abs_total is not None else "n/a",
            f"{avg_abs_margin:.3f}" if avg_abs_margin is not None else "n/a",
        )

    return out

# ----------------------------------------------------------------------------- #
# Bucketing / snapshots
# ----------------------------------------------------------------------------- #
_BUCKET_RULES: List[Tuple[str, str]] = [
    (r"^rolling_", "rolling"),
    (r"(?:^season_|_season_)", "season"),
    (r"^(?:advanced_|adv_)", "advanced"),
    (r"^(?:form_|momentum_)", "form"),
    (r"^rest_", "rest"),
    (r"^h2h_", "h2h"),
    (r"^drive_", "drive"),
]

def _bucket_of(feature: str) -> str:
    for pat, bucket in _BUCKET_RULES:
        if re.search(pat, feature):
            return bucket
    # diffs that don't carry explicit prefix
    if feature.endswith("_diff"):
        return "diffs"
    return "other"

def _attenuate_margin_diffs(
    X: pd.DataFrame,
    alpha: float = 0.6,         # try 0.6 first; drop to 0.5 if still spiky
    debug: bool = False,
    focus_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Scale down high-variance diff features only for MARGIN model input."""
    if X is None or X.empty:
        return X

    cols = [
        "rolling_points_for_avg_diff",
        "rolling_points_against_avg_diff",
        "rolling_point_differential_avg_diff",
        "rolling_yards_per_play_avg_diff",
        "rolling_turnover_differential_avg_diff",
        "momentum_ewma_5_diff",
    ]
    existing = [c for c in cols if c in X.columns]
    if not existing:
        return X.copy()

    X2 = X.copy()
    target_idx = X2.index
    if focus_ids:
        target_idx = [gid for gid in X2.index if gid in focus_ids]
        if not target_idx:
            return X2

    # Attenuate
    X2.loc[target_idx, existing] = X2.loc[target_idx, existing] * float(alpha)

    # Tighter h2h cap for margin (these can be wild early season)
    if "h2h_avg_point_diff" in X2.columns:
        lo, hi = -7.0, 7.0
        X2.loc[target_idx, "h2h_avg_point_diff"] = X2.loc[target_idx, "h2h_avg_point_diff"].clip(lo, hi)

    if debug:
        logger.debug(
            "PRED[ATTN][MARGIN] diffs scaled by alpha=%.2f on %d rows, cols=%s",
            alpha, len(target_idx), existing
        )
    return X2


def _row_snapshot(
    gid: int,
    row_meta: Mapping[str, Any],
    X_row: pd.Series,
    selected_cols: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Return grouped dicts of feature values for a single row keyed by bucket.
    Also extracts impute flags and bucket counts.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    impute_flags: Dict[str, Any] = {}
    for col in selected_cols:
        if col not in X_row.index:
            # missing cols are zero-filled elsewhere; show explicitly as 0.0
            val = 0.0
        else:
            val = X_row[col]
        b = _bucket_of(col)
        grouped.setdefault(b, {})
        grouped[b][col] = float(val) if pd.notna(val) else None
        if col.endswith("_imputed") or "imputed" in col:
            impute_flags[col] = grouped[b][col]
    # Bucket counts
    counts = {b: len(cols) for b, cols in grouped.items()}
    meta = {
        "game_id": gid,
        "date": (row_meta.get("game_date").date().isoformat()
                 if isinstance(row_meta.get("game_date"), datetime)
                 else (row_meta.get("game_date").isoformat() if isinstance(row_meta.get("game_date"), date)
                       else str(row_meta.get("game_date"))[:10])),
        "away_id": int(row_meta.get("away_team_id")) if pd.notna(row_meta.get("away_team_id")) else None,
        "home_id": int(row_meta.get("home_team_id")) if pd.notna(row_meta.get("home_team_id")) else None,
        "key": f"{row_meta.get('away_team_id')}@{row_meta.get('home_team_id')}",
    }
    return {"meta": meta, "grouped": grouped, "impute_flags": impute_flags, "bucket_counts": counts}

def _log_snapshot(tag: str, snap: Dict[str, Dict[str, Any]]):
    m = snap["meta"]
    logger.debug(
        "PRED[ROW] %s game_id=%s, date=%s, away_id=%s, home_id=%s, key=%s",
        tag, m["game_id"], m["date"], m["away_id"], m["home_id"], m["key"]
    )
    # Compact print of a few buckets; keep everything but fold by bucket
    for bucket in ("season", "rolling", "advanced", "form", "rest", "h2h", "drive", "diffs", "other"):
        if bucket in snap["grouped"] and snap["grouped"][bucket]:
            logger.debug("  %s=%s", bucket, {k: snap["grouped"][bucket][k] for k in list(snap["grouped"][bucket].keys())[:12]})
    if snap["impute_flags"]:
        logger.debug("  impute_flags=%s", snap["impute_flags"])
    logger.debug("  bucket_counts=%s", snap["bucket_counts"])

# ----------------------------------------------------------------------------- #
# Per-model prediction utilities
# ----------------------------------------------------------------------------- #
def _per_model_predict(ensemble: NFLEnsemble, X_row_vec) -> Dict[str, float]:
    """
    Return {model_name: y_hat} for a single row vector.
    Falls back to {} if models aren't exposed.
    """
    preds: Dict[str, float] = {}
    models = getattr(ensemble, "models", None)
    if not isinstance(models, dict) or not models:
        return preds
    for name, est in models.items():
        try:
            y = float(est.predict(X_row_vec)[0])
            preds[name] = y
        except Exception:
            continue
    return preds

def _short_model_types(model_names: Iterable[str]) -> Dict[str, str]:
    """
    Map model key to a short label like 'Ridge' | 'SVR' | 'XGB' | 'RF' | 'GB' | 'OLS' | 'Other'.
    """
    out: Dict[str, str] = {}
    for n in model_names:
        ln = n.lower()
        if "ridge" in ln:
            out[n] = "Ridge"
        elif "svr" in ln:
            out[n] = "SVR"
        elif "xgb" in ln or "xgboost" in ln:
            out[n] = "XGB"
        elif "rf" in ln or "randomforest" in ln:
            out[n] = "RF"
        elif "gb" in ln or "gradient" in ln:
            out[n] = "GB"
        elif "ols" in ln or "linear" in ln:
            out[n] = "OLS"
        else:
            out[n] = "Other"
    return out

# ----------------------------------------------------------------------------- #
# Core
# ----------------------------------------------------------------------------- #
def generate_predictions(
    days_window: Optional[int] = 7,           # kept for signature compatibility (ignored)
    historical_lookback: int = 1825,
    debug_mode: bool = False,
    focus: Optional[Focus] = None,
    dump_dir: Optional[Path] = None,
    total_clip_min: Optional[float] = None,
    total_clip_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    End-to-end prediction pipeline (full schedule), with:
      - feature composites
      - H2H shrink (global, pre-model) with guard logs
      - advanced-feature mean imputation + _imputed flags (train means if available)
      - surgical clipping on volatile 'diff' features
      - TOTAL / MARGIN diff attenuation (pre-predict)
      - vectorized ensemble predictions
      - global mean calibration (TOTAL & MARGIN)
      - optional TOTAL clip band (post-calibration)
      - feasibility guard total ≥ |margin| + buffer
      - per-model breakdowns and CSV dumps for focused games
    """
    logger.info("--- NFL Prediction Pipeline ---")
    t0 = time.time()

    # Resolve focus set from caller (no hardcoding)
    FOCUS_IDS: Set[int] = set(focus.ids) if (focus and focus.ids) else set()

    # ------------------------------------------------------------------ #
    # Supabase + artifacts
    # ------------------------------------------------------------------ #
    sb = get_supabase_client()
    if not sb:
        logger.critical("No Supabase client. Abort.")
        return []

    artifacts = load_prediction_artifacts(MODEL_DIR, debug=debug_mode)
    if not artifacts:
        logger.critical("Missing artifacts. Abort.")
        return []

    # Optional: load extended training artifacts if present
    extra_artifacts_paths = [
        MODEL_DIR / "nfl_artifacts.json",
        MODEL_DIR / "nfl_train_artifacts.json",
        MODEL_DIR / "artifacts.json",
    ]
    extra: Dict[str, Any] = {}
    for p in extra_artifacts_paths:
        try:
            if p.exists():
                extra.update(json.loads(p.read_text()))
        except Exception as e:
            if debug_mode:
                logger.debug("PRED[SETUP] Skipping extra artifacts %s: %s", p, e)

    # Baselines / means fallbacks
    total_train_mean = float(extra.get("total_train_mean", artifacts.get("total_train_mean", 43.5)))
    margin_train_mean = float(extra.get("margin_train_mean", artifacts.get("margin_train_mean", 0.0)))
    total_p1 = extra.get("total_p1")  # may be None
    total_p99 = extra.get("total_p99")  # may be None
    feature_means: Dict[str, float] = extra.get("feature_means", {}) or {}

    # ------------------------------------------------------------------ #
    # Fetch schedule (ALL rows) + historical context
    # ------------------------------------------------------------------ #
    try:
        sched_resp = sb.table("nfl_game_schedule").select("*").execute()
        upcoming_df = pd.DataFrame(sched_resp.data or [])
    except Exception as e:
        logger.critical("Failed to fetch nfl_game_schedule: %s", e)
        return []

    upcoming_df = _ensure_season_and_ids(upcoming_df)
    if upcoming_df.empty:
        logger.info("No games found in nfl_game_schedule.")
        return []

    fetcher = NFLDataFetcher(sb)
    games_hist = fetcher.fetch_historical_games(historical_lookback)
    stats_hist = fetcher.fetch_historical_team_game_stats(historical_lookback)

    _log_df(upcoming_df, "upcoming_df", debug_mode)
    _log_df(games_hist,  "games_hist",  debug_mode)
    _log_df(stats_hist,  "stats_hist",  debug_mode)

    # ------------------------------------------------------------------ #
    # Feature pipeline
    # ------------------------------------------------------------------ #
    logger.info("Building features for %d games…", len(upcoming_df))
    feat_t = time.time()
    nfl_engine = NFLFeatureEngine(
        supabase_url=config.SUPABASE_URL,
        supabase_service_key=config.SUPABASE_SERVICE_KEY
    )
    features_df = nfl_engine.build_features(
        games_df=upcoming_df,
        historical_games_df=games_hist,
        historical_team_stats_df=stats_hist,
        debug=debug_mode,
    )
    logger.info("Feature pipeline done in %.2fs", time.time() - feat_t)
    _log_df(features_df, "features_df", debug_mode)

    if features_df.empty:
        logger.error("Feature DF empty. Abort.")
        return []

    # Add train-time composites BEFORE any shrink/clip
    features_df = add_total_composites(
        features_df,
        keys=[
            "rolling_points_for_avg",
            "rolling_points_against_avg",
            "rolling_yards_per_play_avg",
            "rolling_turnover_differential_avg",
            "form_win_pct_5",
            "rest_days",
        ],
    )

    # ------------------------------------------------------------------ #
    # Global H2H shrink (pre-model) with guard logs
    # ------------------------------------------------------------------ #
    touched_rows = 0
    if "h2h_games_played" in features_df.columns:
        try:
            touched_rows = int((pd.to_numeric(features_df["h2h_games_played"], errors="coerce").fillna(0.0) > 0).sum())
        except Exception:
            touched_rows = 0

    features_df = _shrink_h2h_features(
        features_df,
        mu_total=total_train_mean,
        mu_margin=margin_train_mean,
        k=float(extra.get("h2h_trust_k", 6.0)),
        debug=debug_mode,
        focus_ids=FOCUS_IDS,
    )

    if touched_rows == 0:
        logger.warning("PRED[H2H][WARN] No rows had h2h_games_played > 0; H2H shrink effectively no-op this batch.")

    # ------------------------------------------------------------------ #
    # Advanced feature mean imputation (+ _imputed flags)
    # - Use train-time feature_means if available
    # - Treat zeros as missing for adv_* / advanced_* (distribution fix)
    # ------------------------------------------------------------------ #
    # --- Advanced diffs observability (guarded: no warnings when none present) ---
    adv_cols = [c for c in features_df.columns if c.startswith(("adv_", "advanced_"))]
    if adv_cols:
        top_counts = []
        for c in adv_cols:
            zero_like = (features_df[c].fillna(0.0) == 0.0).sum()
            top_counts.append((c, int(zero_like)))
        fully_zero_like = sum(1 for c, cnt in top_counts if cnt == len(features_df))
        adv_imputed_ratio = 100.0 * fully_zero_like / max(1, len(adv_cols))
        if debug_mode:
            logger.debug(
                "PRED[IMPUTE] advanced_imputed_ratio=%.2f%%, top=%s",
                adv_imputed_ratio,
                sorted(top_counts, key=lambda x: x[1], reverse=True)[:5],
            )
    else:
        if debug_mode:
            logger.debug("PRED[IMPUTE] advanced_imputed_ratio=0%% (note: no advanced cols present)")


    # ------------------------------------------------------------------ #
    # Volatile diff clipping (global, pre-model)
    # ------------------------------------------------------------------ #
    CLIP_BOUNDS = {
        "rolling_points_for_avg_diff": (-12.0, 12.0),
        "rolling_points_against_avg_diff": (-12.0, 12.0),
        "rolling_point_differential_avg_diff": (-20.0, 20.0),
        "rolling_yards_per_play_avg_diff": (-1.5, 1.5),
        "rolling_turnover_differential_avg_diff": (-2.0, 2.0),
        "momentum_ewma_5_diff": (-25.0, 25.0),
    }
    features_df = _clip_feature_columns(features_df, CLIP_BOUNDS, debug=debug_mode, focus_ids=FOCUS_IDS)

    # ------------------------------------------------------------------ #
    # Model setup + matrices
    # ------------------------------------------------------------------ #
    for target in ("margin", "total"):
        selected = artifacts[f"{target}_features"]
        missing = [c for c in selected if c not in features_df.columns]
        logger.debug("PRED[SETUP] %s selected (n=%d): %s", target.upper(), len(selected), selected[:20])
        if missing:
            logger.debug("PRED[SETUP][WARN] %s missing-in-frame: %s", target.upper(), missing)

    margin_ensemble = NFLEnsemble(artifacts["margin_weights"], MODEL_DIR)
    total_ensemble  = NFLEnsemble(artifacts["total_weights"],  MODEL_DIR)
    margin_ensemble.load_models()
    total_ensemble.load_models()

    # Ensure consistent indexing by game_id
    if "game_id" in features_df.columns:
        features_df = features_df.set_index("game_id", drop=False)

    # Build matrices against selected feature lists (imputation already applied on features_df)
    X_margin = features_df.reindex(columns=artifacts["margin_features"], fill_value=0.0).copy()
    X_total  = features_df.reindex(columns=artifacts["total_features"],  fill_value=0.0).copy()

    # Frame audits
    if debug_mode:
        for label, X in (("MARGIN", X_margin), ("TOTAL", X_total)):
            prof = _df_profile(X, f"X_{label}")
            ord_crc = _order_crc(X.columns)
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            logger.debug("PRED[FRAME] %s rows=%d | null_top=%s | order_crc=%s", label, prof["rows"], prof["null_top"], ord_crc)
            if non_numeric:
                logger.debug("PRED[FRAME][WARN] %s non-numeric cols: %s", label, non_numeric)

    _check_features(X_margin, artifacts["margin_features"], "X_margin", debug_mode)
    _check_features(X_total,  artifacts["total_features"],  "X_total",  debug_mode)

    # ------------------------------------------------------------------ #
    # Pre-predict attenuation
    # ------------------------------------------------------------------ #
    # (Optional) baseline total per focus game before any attenuation – helpful in debugging
    if debug_mode and FOCUS_IDS and getattr(total_ensemble, "models", None):
        for gid in sorted([g for g in FOCUS_IDS if g in X_total.index]):
            try:
                _ = float(total_ensemble.predict(X_total.loc[[gid]])[0])
            except Exception as e:
                logger.debug("PRED[ATTN][BASE] skip game_id=%s (%s)", gid, e)

    # TOTAL attenuation (diffs)
    X_total  = _attenuate_total_diffs(X_total,  alpha=0.6, debug=debug_mode, focus_ids=None)
    X_margin = _attenuate_margin_diffs(X_margin, alpha=0.6, debug=debug_mode, focus_ids=None)


    # ------------------------------------------------------------------ #
    # Focused snapshots (per-model) BEFORE vectorized predict
    # ------------------------------------------------------------------ #
    per_model_totals: Dict[int, Dict[str, float]] = {}
    per_model_margins: Dict[int, Dict[str, float]] = {}

    if debug_mode and FOCUS_IDS:
        for gid in sorted([g for g in FOCUS_IDS if g in features_df.index]):
            # TOTAL per-model breakdown
            if getattr(total_ensemble, "models", None):
                pm_total: Dict[str, float] = {}
                pieces: List[str] = []
                xvec = X_total.loc[[gid]].values
                for mname, m in total_ensemble.models.items():
                    try:
                        y = float(m.predict(xvec)[0])
                        pm_total[mname] = y
                        tag = ("XGB" if "xgb" in mname.lower() else
                               "RF"  if "rf"  in mname.lower() else mname)
                        pieces.append(f"{tag}={y:.3f}")
                    except Exception:
                        pass
                per_model_totals[int(gid)] = pm_total
                if pieces:
                    logger.debug("PRED[MODEL] TOTAL game_id=%s | %s", gid, " | ".join(pieces))

            # MARGIN per-model breakdown
            if getattr(margin_ensemble, "models", None):
                pm_margin: Dict[str, float] = {}
                pieces = []
                xvec = X_margin.loc[[gid]].values
                for mname, m in margin_ensemble.models.items():
                    try:
                        y = float(m.predict(xvec)[0])
                        pm_margin[mname] = y
                        tag = ("Ridge" if "ridge" in mname.lower() else
                               "SVR"   if "svr"   in mname.lower() else mname)
                        pieces.append(f"{tag}={y:.3f}")
                    except Exception:
                        pass
                per_model_margins[int(gid)] = pm_margin
                if pieces:
                    logger.debug("PRED[MODEL] MARGIN game_id=%s | %s", gid, " | ".join(pieces))

            # Compact feature snapshots (first ~20 from each list)
            tfeat = artifacts["total_features"][:20]
            mfeat = artifacts["margin_features"][:20]
            snap_t = {c: float(features_df.at[gid, c]) if (c in features_df.columns and pd.notna(features_df.at[gid, c])) else 0.0 for c in tfeat}
            snap_m = {c: float(features_df.at[gid, c]) if (c in features_df.columns and pd.notna(features_df.at[gid, c])) else 0.0 for c in mfeat}
            logger.debug("PRED[ROW] TOTAL game_id=%s | %s", gid, snap_t)
            logger.debug("PRED[ROW] MARGIN game_id=%s | %s", gid, snap_m)

            # Also dump the full feature row to CSV
            outdir = PROJECT_ROOT / "debug" / "pred_inspect"
            outdir.mkdir(parents=True, exist_ok=True)
            features_df.loc[[gid]].to_csv(outdir / f"game_{gid}_features.csv", index=False)
            logger.debug("PRED[DUMP] wrote %s", outdir / f"game_{gid}_features.csv")

    # ------------------------------------------------------------------ #
    # Vectorized predict (ENSEMBLE)
    # ------------------------------------------------------------------ #
    pred_t = time.time()
    margin_preds = margin_ensemble.predict(X_margin)
    total_preds  = total_ensemble.predict(X_total)

    # Ensure Series (not numpy) with game_id index
    if isinstance(margin_preds, np.ndarray):
        margin_preds = pd.Series(margin_preds, index=features_df.index, name="pred_margin")
    if isinstance(total_preds, np.ndarray):
        total_preds = pd.Series(total_preds, index=features_df.index, name="pred_total")

    if debug_mode:
        logger.debug("PRED[ENS] MARGIN | weights=%s", artifacts.get("margin_weights", {}))
        logger.debug("PRED[ENS] TOTAL  | weights=%s", artifacts.get("total_weights", {}))
        logger.debug("Predict time: %.3fs", time.time() - pred_t)

    # ------------------------------------------------------------------ #
    # Global calibration (unconditional) → OPTIONAL clip → feasibility
    # ------------------------------------------------------------------ #
    # 1) Mean calibration for TOTAL
    week_mean_total = float(total_preds.mean()) if len(total_preds) else total_train_mean
    delta_total = total_train_mean - week_mean_total
    if abs(delta_total) >= 0.75:
        total_preds = (total_preds + delta_total).astype(float)
        if debug_mode:
            logger.debug(
                "PRED[CAL] TOTAL mean-calibration | week_mean=%.2f → target=%.2f (Δ=%.2f)",
                week_mean_total, total_train_mean, delta_total
            )

    # 2) Mean recenter for MARGIN (toward 0)
    week_mean_margin = float(margin_preds.mean()) if len(margin_preds) else 0.0
    if abs(week_mean_margin) >= 0.25:
        margin_preds = (margin_preds - week_mean_margin).astype(float)
        if debug_mode:
            logger.debug("PRED[CAL] MARGIN recenter | week_mean=%.2f → 0.00 (Δ=%.2f)", week_mean_margin, -week_mean_margin)

    # 3) Optional TOTAL clip band (after calibration)
    if (total_clip_min is not None) or (total_clip_max is not None):
        lo = float(total_clip_min) if total_clip_min is not None else float("-inf")
        hi = float(total_clip_max) if total_clip_max is not None else float("+inf")
        if debug_mode and FOCUS_IDS:
            for gid in [g for g in FOCUS_IDS if g in total_preds.index]:
                pre = float(total_preds.loc[gid])
                post = max(lo, min(hi, pre))
                logger.debug("PRED[ENS] TOTAL | post-cal clip: pre=%.3f → post=%.3f (game_id=%s)", pre, post, gid)
        total_preds = total_preds.clip(lower=lo, upper=hi)

    # 4) Feasibility floor: total must exceed |margin| by a buffer
    buffer_pts = 6.0
    pair = pd.DataFrame({"total": total_preds.astype(float), "margin": margin_preds.astype(float)}, index=total_preds.index)
    need_raise = (pair["total"] < pair["margin"].abs() + buffer_pts)
    if need_raise.any():
        min_allowed = pair["margin"].abs() + buffer_pts
        adjusted = pair["total"].where(~need_raise, min_allowed)
        if debug_mode and FOCUS_IDS:
            for gid in [g for g in FOCUS_IDS if g in adjusted.index and need_raise.loc[g]]:
                pre  = float(pair.at[gid, "total"])
                post = float(adjusted.at[gid])
                logger.debug("PRED[CAL][FEAS] gid=%s | total pre=%.2f → post=%.2f (|M|+buf=%.2f)", gid, pre, post, float(min_allowed.at[gid]))
        total_preds = adjusted.astype(float)

    # 5) Optional hard cap on margin to trim extremes
    margin_cap = 24.0
    over_cap = margin_preds.abs() > margin_cap
    if over_cap.any():
        margin_preds = margin_preds.clip(lower=-margin_cap, upper=margin_cap)
        if debug_mode and FOCUS_IDS:
            for gid in [g for g in FOCUS_IDS if g in margin_preds.index and over_cap.loc[g]]:
                logger.debug("PRED[CAL] MARGIN hard-cap applied | gid=%s cap=±%.1f", gid, margin_cap)

    # >>> Focus-only test shrink for close spreads (temporary) <<<
    if debug_mode and FOCUS_IDS:
        lam = 0.90
        thresh = 6.0
        for gid in [g for g in FOCUS_IDS if g in margin_preds.index]:
            pre = float(margin_preds.loc[gid])
            if abs(pre) <= thresh:
                margin_preds.loc[gid] = pre * lam
                logger.debug("PRED[SHRINK][MARGIN] gid=%s pre=%.3f → post=%.3f (λ=%.2f)", gid, pre, pre * lam, lam)

    # ------------------------------------------------------------------ #
    # Derive scores and assign back onto schedule
    # ------------------------------------------------------------------ #
    scores_df = derive_scores_from_predictions(margin_preds, total_preds)

    # Make sure scores_df is indexed by game_id and numeric
    if "game_id" in scores_df.columns:
        scores_df = scores_df.set_index("game_id", drop=False)
    scores_df["predicted_home_score"] = scores_df["predicted_home_score"].astype(float)
    scores_df["predicted_away_score"] = scores_df["predicted_away_score"].astype(float)
    _log_df(scores_df, "scores_df", debug_mode)

    # Assign onto upcoming_df by index = game_id
    final_df = upcoming_df.set_index("game_id", drop=False)
    final_df.loc[scores_df.index, "predicted_home_score"] = scores_df["predicted_home_score"]
    final_df.loc[scores_df.index, "predicted_away_score"] = scores_df["predicted_away_score"]

    # Compute totals/margins *after* assignment and *before* printing
    final_df["pred_total"]  = (final_df["predicted_home_score"] + final_df["predicted_away_score"]).astype(float)
    final_df["pred_margin"] = (final_df["predicted_home_score"] - final_df["predicted_away_score"]).astype(float)

    # Pretty-print (example)
    def _fmt(x): return f"{x:.2f}"
    for _, row in final_df.loc[scores_df.index].iterrows():
        date = str(row["game_date"])[:10]
        home = _fmt(row["predicted_home_score"])
        away = _fmt(row["predicted_away_score"])
        tot  = _fmt(row["pred_total"])
        mar  = _fmt(row["pred_margin"])
        print(f"{date}  away_id {int(row['away_team_id'])} @ home_id {int(row['home_team_id'])}   {away} - {home}   (T={tot}, M={mar})")

    if debug_mode:
        n_assigned = final_df["predicted_home_score"].notna().sum()
        logger.debug("PRED[JOIN] assigned predictions for %d games", n_assigned)

    # ------------------------------------------------------------------ #
    # Optional CSV dumps for focused games (post-ensemble & cal)
    # ------------------------------------------------------------------ #
    if debug_mode and FOCUS_IDS:
        base = dump_dir or (PROJECT_ROOT / "debug" / "pred_inspect")
        for gid in [g for g in FOCUS_IDS if g in final_df.index]:
            row = final_df.loc[gid]
            gd = row.get("game_date")
            date_str = gd.date().isoformat() if isinstance(gd, (pd.Timestamp, datetime)) else str(gd)[:10]

            subdir = Path(base) / date_str
            subdir.mkdir(parents=True, exist_ok=True)

            csv_row: Dict[str, Any] = {
                "game_id": int(gid),
                "date": date_str,
                "away_team_id": int(row["away_team_id"]),
                "home_team_id": int(row["home_team_id"]),
                "away_team_name": row.get("away_team_name"),
                "home_team_name": row.get("home_team_name"),
                "predicted_away_score": float(row.get("predicted_away_score")) if "predicted_away_score" in row else None,
                "predicted_home_score": float(row.get("predicted_home_score")) if "predicted_home_score" in row else None,
                "weights_total_json": json.dumps(artifacts.get("total_weights", {})),
                "weights_margin_json": json.dumps(artifacts.get("margin_weights", {})),
            }
            # Include per-model raw preds if we computed them
            for full, val in per_model_totals.get(int(gid), {}).items():
                csv_row[f"pred_total__{full}"] = float(val)
            for full, val in per_model_margins.get(int(gid), {}).items():
                csv_row[f"pred_margin__{full}"] = float(val)

            # Selected feature vectors (union of total + margin features)
            union_cols = list(dict.fromkeys(artifacts["total_features"] + artifacts["margin_features"]))
            for c in union_cols:
                csv_row[c] = float(features_df.at[gid, c]) if (c in features_df.columns and pd.notna(features_df.at[gid, c])) else 0.0

            # Impute flags explicitly
            impute_cols = [c for c in features_df.columns if "imput" in c.lower()]
            for c in impute_cols:
                key = f"flag__{c}"
                try:
                    csv_row[key] = float(features_df.at[gid, c]) if pd.notna(features_df.at[gid, c]) else 0.0
                except Exception:
                    csv_row[key] = 0.0

            out_path = subdir / f"{date_str}_{row['away_team_id']}@{row['home_team_id']}_{gid}.csv"
            pd.DataFrame([csv_row]).to_csv(out_path, index=False)
            logger.debug("PRED[DUMP] wrote %s", out_path)

    # ------------------------------------------------------------------ #
    # Build payload
    # ------------------------------------------------------------------ #
    payload: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for gid, row in final_df.iterrows():
        if pd.isna(row.get("predicted_home_score")):
            if debug_mode:
                logger.debug("Skipping game_id=%s due to NaN prediction", gid)
            continue
        payload.append(
            {
                "game_id":              int(gid),
                "game_date":            row["game_date"].isoformat() if hasattr(row["game_date"], "isoformat") else str(row["game_date"]),
                "home_team_id":         int(row["home_team_id"]),
                "away_team_id":         int(row["away_team_id"]),
                "home_team_name":       row.get("home_team_name"),
                "away_team_name":       row.get("away_team_name"),
                "predicted_home_score": round(float(row["predicted_home_score"]), 2),
                "predicted_away_score": round(float(row["predicted_away_score"]), 2),
                "prediction_utc":       now_iso,
            }
        )

    logger.info("Generated %d predictions in %.2fs", len(payload), time.time() - t0)
    return payload

# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate and Upsert NFL Score Predictions")
    parser.add_argument("--days", type=int, default=8, help="Days ahead to predict.")
    parser.add_argument("--lookback", type=int, default=1825, help="Historical days for features.")
    parser.add_argument("--no-upsert", action="store_true", help="Skip DB upsert.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")

    # Focus selectors
    parser.add_argument("--focus-date", type=str, default=None, help="Focus on a specific date (YYYY-MM-DD).")
    parser.add_argument("--focus-games", type=str, default=None, help="Comma-separated away@home pairs, e.g., 25@26,21@28.")
    parser.add_argument("--focus-ids", type=str, default=None, help="Comma-separated game_id list.")

    # Optional TOTAL clip band (debug/QA)
    parser.add_argument("--total-clip-min", type=float, default=None, help="Clip TOTAL predictions to lower bound.")
    parser.add_argument("--total-clip-max", type=float, default=None, help="Clip TOTAL predictions to upper bound.")

    # Dump dir
    parser.add_argument("--dump-dir", type=str, default=None, help="Directory to write per-focused-game CSVs.")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Build Focus from CLI (you already have these helpers in the file)
    focus = Focus(
        date_str=_normalize_date_str(args.focus_date),
        pairs=_parse_focus_games(args.focus_games),
        ids=_parse_focus_ids(args.focus_ids),
    )

    preds = generate_predictions(
        days_window=args.days,
        historical_lookback=args.lookback,
        debug_mode=args.debug,           # <- True/False from CLI
        focus=focus,                     # <- proper Focus object
        dump_dir=Path(args.dump_dir) if args.dump_dir else None,
        total_clip_min=args.total_clip_min,
        total_clip_max=args.total_clip_max,
    )


    if not preds:
        logger.info("No predictions produced.")
        sys.exit(0)

    display_prediction_summary(preds)

    if not args.no_upsert:
        sb = get_supabase_client()
        if sb:
            upsert_score_predictions(preds, sb, debug=args.debug)
        else:
            logger.error("Supabase unavailable; cannot upsert.")
    else:
        logger.info("--no-upsert specified. Skipping upsert.")

if __name__ == "__main__":
    main()
