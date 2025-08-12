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
    days_window: Optional[int] = 7,
    historical_lookback: int = 1825,
    debug_mode: bool = False,
    focus: Optional[Focus] = None,
    dump_dir: Optional[Path] = None,
    total_clip_min: Optional[float] = None,
    total_clip_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    logger.info("--- NFL Prediction Pipeline ---")
    t0 = time.time()

    sb = get_supabase_client()
    if not sb:
        logger.critical("No Supabase client. Abort.")
        return []

    artifacts = load_prediction_artifacts(MODEL_DIR, debug=debug_mode)
    if not artifacts:
        logger.critical("Missing artifacts. Abort.")
        return []

    fetcher = NFLDataFetcher(sb)
    upcoming_df = sb.table("nfl_game_schedule").select("*").execute()
    upcoming_df = pd.DataFrame(upcoming_df.data or [])
    upcoming_df = _ensure_season_and_ids(upcoming_df)
    if upcoming_df.empty:
        logger.info("No games found in nfl_game_schedule.")
        return []


    games_hist = fetcher.fetch_historical_games(historical_lookback)
    stats_hist = fetcher.fetch_historical_team_game_stats(historical_lookback)

    _log_df(upcoming_df, "upcoming_df", debug_mode)
    _log_df(games_hist,  "games_hist",  debug_mode)
    _log_df(stats_hist,  "stats_hist",  debug_mode)

    # --- Feature pipeline ---
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

    # Composites to align with train-time columns (safe no-op if missing)
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

    # Ensure index
    if "game_id" in features_df.columns:
        features_df = features_df.set_index("game_id", drop=False)

    # Selected features audit (missing-in-frame ideally empty)
    for target in ("margin", "total"):
        selected = artifacts[f"{target}_features"]
        missing_in_frame = [c for c in selected if c not in features_df.columns]
        logger.debug("PRED[SETUP] %s missing-in-frame: %s", target.upper(), missing_in_frame)

    # Ensembles
    margin_ensemble = NFLEnsemble(artifacts["margin_weights"], MODEL_DIR)
    total_ensemble  = NFLEnsemble(artifacts["total_weights"],  MODEL_DIR)
    margin_ensemble.load_models()
    total_ensemble.load_models()

    # Align matrices
    X_margin = features_df.reindex(columns=artifacts["margin_features"], fill_value=0.0).copy()
    X_total  = features_df.reindex(columns=artifacts["total_features"],  fill_value=0.0).copy()
    X_margin = X_margin.fillna(0.0)
    X_total  = X_total.fillna(0.0)

    # Frame audits
    if debug_mode:
        for label, X in (("MARGIN", X_margin), ("TOTAL", X_total)):
            prof = _df_profile(X, f"X_{label}")
            ord_crc = _order_crc(X.columns)
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            logger.debug(
                "PRED[FRAME] %s rows=%d | null_top=%s | order_crc=%s",
                label, prof["rows"], prof["null_top"], ord_crc
            )
            if non_numeric:
                logger.debug("PRED[FRAME][WARN] %s non-numeric cols: %s", label, non_numeric)

    _check_features(X_margin, artifacts["margin_features"], "X_margin", debug_mode)
    _check_features(X_total,  artifacts["total_features"],  "X_total",  debug_mode)

    # Focused-row snapshots + per-model predictions BEFORE ensemble
    focus = focus or Focus(None, set(), set())
    focused_gids: List[int] = []
    per_model_totals: Dict[int, Dict[str, float]] = {}
    per_model_margins: Dict[int, Dict[str, float]] = {}

    if debug_mode and (focus.date_str or focus.pairs or focus.ids):
        for gid, row in features_df.iterrows():
            if not _is_focused_row(row, focus):
                continue
            focused_gids.append(int(gid))

            # TOTAL snapshot
            xrow_total = X_total.loc[gid]
            snap_total = _row_snapshot(int(gid), row, xrow_total, artifacts["total_features"])
            _log_snapshot("TOTAL", snap_total)

            # MARGIN snapshot
            xrow_margin = X_margin.loc[gid]
            snap_margin = _row_snapshot(int(gid), row, xrow_margin, artifacts["margin_features"])
            _log_snapshot("MARGIN", snap_margin)

            # Per-model raw predictions for this row
            xvec_total  = xrow_total.values.reshape(1, -1)
            xvec_margin = xrow_margin.values.reshape(1, -1)

            pm_total  = _per_model_predict(total_ensemble, xvec_total)
            pm_margin = _per_model_predict(margin_ensemble, xvec_margin)
            per_model_totals[int(gid)]  = pm_total
            per_model_margins[int(gid)] = pm_margin

            if pm_total:
                short = _short_model_types(pm_total.keys())
                # Group by short type for compact line
                # If multiple of the same type exist, show first occurrence
                compact = {}
                for full, pred in pm_total.items():
                    label = short.get(full, "Other")
                    compact[label] = round(float(pred), 3)
                logger.debug("PRED[MODEL] TOTAL | " + " | ".join(f"{k}={v}" for k, v in compact.items()))
            else:
                logger.debug("PRED[MODEL] TOTAL | (per-model predictions unavailable)")

            if pm_margin:
                short = _short_model_types(pm_margin.keys())
                compact = {}
                for full, pred in pm_margin.items():
                    label = short.get(full, "Other")
                    compact[label] = round(float(pred), 3)
                logger.debug("PRED[MODEL] MARGIN | " + " | ".join(f"{k}={v}" for k, v in compact.items()))
            else:
                logger.debug("PRED[MODEL] MARGIN | (per-model predictions unavailable)")

            # Priors sanity (diffs) & rolling QC proxies
            priors = {}
            for c in ("prev_season_win_pct_diff", "prev_season_srs_lite_diff", "adv_red_zone_pct_diff"):
                if c in features_df.columns:
                    priors[c] = float(features_df.at[gid, c]) if pd.notna(features_df.at[gid, c]) else None
            # Try rolling support counts; else imputed flags summary
            roll_keys = [k for k in features_df.columns if re.search(r"(home|away)_rolling_.*(_games|_count)$", k)]
            if roll_keys:
                support = {rk: float(features_df.at[gid, rk]) if pd.notna(features_df.at[gid, rk]) else None for rk in roll_keys[:6]}
                logger.debug("PRED[QC] rolling_support_subset=%s", support)
            else:
                imputed_cols = [c for c in features_df.columns if "imput" in c.lower()]
                imputed_on = [c for c in imputed_cols if (pd.notna(features_df.at[gid, c]) and float(features_df.at[gid, c]) != 0.0)]
                logger.debug("PRED[QC] rolling_imputed_flags={count=%d, names=%s}", len(imputed_on), imputed_on[:10])
            logger.debug("PRED[QC] priors=%s", priors)

    # Predict (ENSEMBLE)
    pred_t = time.time()
    margin_preds = margin_ensemble.predict(X_margin)  # vector
    total_preds  = total_ensemble.predict(X_total)    # vector

    # Ensemble weights logs
    if debug_mode:
        logger.debug("PRED[ENS] MARGIN | weights=%s", artifacts.get("margin_weights", {}))
        logger.debug("PRED[ENS] TOTAL  | weights=%s", artifacts.get("total_weights", {}))

    # Optional TOTAL clip band
    if total_clip_min is not None or total_clip_max is not None:
        total_clip_min = float(total_clip_min) if total_clip_min is not None else float("-inf")
        total_clip_max = float(total_clip_max) if total_clip_max is not None else float("+inf")
        if debug_mode and focused_gids:
            for gid in focused_gids:
                pre = float(total_preds[features_df.index.get_loc(gid)])
                post = max(total_clip_min, min(total_clip_max, pre))
                logger.debug("PRED[ENS] TOTAL | preclip=%.3f → postclip=%.3f", pre, post)
        # Apply clip to full vector
        total_preds = total_preds.clip(min=total_clip_min, max=total_clip_max)

    if debug_mode:
        logger.debug("Predict time: %.3fs", time.time() - pred_t)

    # Derive scores
    scores_df = derive_scores_from_predictions(margin_preds, total_preds)
    # Ensure index is game_id (matches features_df/upcoming_df index)
    if "game_id" in scores_df.columns:
        scores_df = scores_df.set_index("game_id", drop=False)
    # Make sure numeric
    scores_df["predicted_home_score"] = scores_df["predicted_home_score"].astype(float)
    scores_df["predicted_away_score"] = scores_df["predicted_away_score"].astype(float)
    _log_df(scores_df, "scores_df", debug_mode)

    # Merge with schedule (direct assign avoids column-collision)
    upcoming_df = upcoming_df.set_index("game_id", drop=False)
    upcoming_df.loc[scores_df.index, "predicted_home_score"] = scores_df["predicted_home_score"]
    upcoming_df.loc[scores_df.index, "predicted_away_score"] = scores_df["predicted_away_score"]
    final_df = upcoming_df

    # Optional sanity log:
    if debug_mode:
        n = final_df["predicted_home_score"].notna().sum()
        logger.debug("PRED[JOIN] assigned predictions for %d games", n)


    # Dump per-focused-row CSVs (after ensemble)
    if debug_mode and (focus.date_str or focus.pairs or focus.ids):
        base = dump_dir or (PROJECT_ROOT / "debug" / "pred_inspect")
        for gid in focused_gids:
            if gid not in final_df.index:
                continue
            row = final_df.loc[gid]
            date_str = (row["game_date"].date().isoformat() if isinstance(row["game_date"], datetime)
                        else (row["game_date"].isoformat() if isinstance(row["game_date"], date) else str(row["game_date"])[:10]))
            subdir = Path(base) / date_str
            subdir.mkdir(parents=True, exist_ok=True)

            # Build CSV row
            csv_row: Dict[str, Any] = {
                "game_id": int(gid),
                "date": date_str,
                "away_team_id": int(row["away_team_id"]),
                "home_team_id": int(row["home_team_id"]),
                "away_team_name": row.get("away_team_name"),
                "home_team_name": row.get("home_team_name"),
                "pred_total_ensemble": float(row.get("pred_total")) if "pred_total" in row else None,
                "pred_margin_ensemble": float(row.get("pred_margin")) if "pred_margin" in row else None,
                "predicted_away_score": float(row.get("predicted_away_score")) if "predicted_away_score" in row else None,
                "predicted_home_score": float(row.get("predicted_home_score")) if "predicted_home_score" in row else None,
                "weights_total_json": json.dumps(getattr(total_ensemble, "weights", {})),
                "weights_margin_json": json.dumps(getattr(margin_ensemble, "weights", {})),
            }

            # Per-model predictions (if we computed them)
            for full, val in per_model_totals.get(int(gid), {}).items():
                csv_row[f"pred_total__{full}"] = float(val)
            for full, val in per_model_margins.get(int(gid), {}).items():
                csv_row[f"pred_margin__{full}"] = float(val)

            # Selected vectors (union of total + margin selected)
            union_cols = list(dict.fromkeys(artifacts["total_features"] + artifacts["margin_features"]))
            for c in union_cols:
                csv_row[c] = float(features_df.at[gid, c]) if (c in features_df.columns and pd.notna(features_df.at[gid, c])) else 0.0

            # Impute flags explicitly (some may already be in union)
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

    # Build payload
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

    focus = Focus(
        date_str=_normalize_date_str(args.focus_date),
        pairs=_parse_focus_games(args.focus_games),
        ids=_parse_focus_ids(args.focus_ids),
    )

    preds = generate_predictions(
        days_window=args.days,
        historical_lookback=args.lookback,
        debug_mode=args.debug,
        focus=focus,
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
