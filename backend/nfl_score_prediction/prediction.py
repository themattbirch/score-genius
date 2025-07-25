# backend/nfl_score_prediction/prediction.py
"""
prediction.py – NFL Score Prediction Generation Script (with rich DEBUG logging)

Pipeline:
1. Fetch upcoming games (window) + historical context (lookback).
2. Build features via run_nfl_feature_pipeline().
3. Load artifacts (selected features + ensemble weights).
4. Load all models once with NFLEnsemble, predict margin & total.
5. Derive home/away scores, upsert to Supabase.
6. Optional: print human-readable summary.

Debug additions:
- DataFrame profiling helpers (_df_profile / _log_df).
- Artifact sanity checks (missing features, zero-weight models).
- Feature alignment diagnostics (missing cols, NaN counts).
- Upsert batch sizes and response counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase import Client, create_client

# Project imports
try:
    from backend import config
    from backend.nfl_features.engine import run_nfl_feature_pipeline
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

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ----------------------------------------------------------------------------- #
# Small diagnostics helpers
# ----------------------------------------------------------------------------- #
def _df_profile(df: pd.DataFrame, name: str, top_n: int = 10) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"name": name, "rows": 0, "cols": 0}
    nulls = df.isna().sum().sort_values(ascending=False)
    num_cols = df.select_dtypes(include="number").columns
    zeros = (df[num_cols] == 0).sum().sort_values(ascending=False) if len(num_cols) else pd.Series(dtype=int)
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": nulls.head(top_n).to_dict(),
        "zero_top": zeros.head(top_n).to_dict(),
    }

def _log_df(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    prof = _df_profile(df, name)
    logger.debug(
        "[DF] %-22s rows=%-5d cols=%-4d | null_top=%s | zero_top=%s",
        prof["name"], prof["rows"], prof["cols"], prof.get("null_top"), prof.get("zero_top")
    )

def _check_features(X: pd.DataFrame, needed: List[str], label: str, debug: bool):
    if not debug:
        return
    missing = [c for c in needed if c not in X.columns]
    extra   = [c for c in X.columns if c not in needed]
    if missing:
        logger.debug("[%s] Missing %d cols: %s", label, len(missing), missing[:20])
    if extra:
        logger.debug("[%s] Extra %d cols ignored", label, len(extra))
    nan_ct = X.isna().sum().sum()
    if nan_ct:
        logger.debug("[%s] Total NaNs in matrix: %d", label, nan_ct)

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
                logger.debug("[ARTIFACT] %s feats=%d, weights=%d (sum=%.4f)",
                             target, len(feats), len(wts), sum(wts.values()))
        except FileNotFoundError as e:
            logger.critical("Missing artifact for %s: %s", target, e)
            return {}
    return artifacts

def chunked(lst: List[Dict[str, Any]], size: int = 500):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def upsert_score_predictions(preds: List[Dict[str, Any]], sb: Client, debug: bool = False) -> int:
    if not preds:
        return 0
    total = 0
    for batch in chunked(preds, 500):
        try:
            resp = sb.table("nfl_game_schedule").upsert(batch, on_conflict="game_id").execute()
            cnt = len(resp.data or [])
            total += cnt
            if debug:
                logger.debug("Upsert batch size=%d -> inserted/updated=%d", len(batch), cnt)
        except Exception as e:
            logger.error("Upsert error: %s", e)
    logger.info("Upserted %d predictions", total)
    return total

def display_prediction_summary(preds: List[Dict[str, Any]]) -> None:
    if not preds:
        return
    df = pd.DataFrame(preds).sort_values("game_date")
    print("-" * 80)
    print(f"{'DATE':<12}{'MATCHUP':<42}{'PREDICTED SCORE':<20}")
    print("-" * 80)
    for _, r in df.iterrows():
        d = datetime.fromisoformat(str(r["game_date"])).strftime("%Y-%m-%d")
        matchup = f"{r['away_team_name']} @ {r['home_team_name']}"
        score = f"{r['predicted_away_score']:.1f} - {r['predicted_home_score']:.1f}"
        print(f"{d:<12}{matchup:<42}{score:<20}")
    print("-" * 80)

# ----------------------------------------------------------------------------- #
# Core
# ----------------------------------------------------------------------------- #
def generate_predictions(
    days_window: int = 7,
    historical_lookback: int = 1825,
    debug_mode: bool = False,
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

    upcoming_df = fetcher.fetch_upcoming_games(days_window)
    if upcoming_df.empty:
        logger.info("No upcoming games in window.")
        return []

    games_hist = fetcher.fetch_historical_games(historical_lookback)
    stats_hist = fetcher.fetch_historical_team_game_stats(historical_lookback)

    _log_df(upcoming_df, "upcoming_df", debug_mode)
    _log_df(games_hist,  "games_hist",  debug_mode)
    _log_df(stats_hist,  "stats_hist",  debug_mode)

    # Features
    logger.info("Building features for %d games…", len(upcoming_df))
    feat_t = time.time()
    features_df = run_nfl_feature_pipeline(
        games_df=upcoming_df,
        historical_games_df=games_hist,
        historical_team_game_stats_df=stats_hist,
        debug=debug_mode,
    )
    logger.info("Feature pipeline done in %.2fs", time.time() - feat_t)
    _log_df(features_df, "features_df", debug_mode)

    if features_df.empty:
        logger.error("Feature DF empty. Abort.")
        return []

    # Ensure index
    if "game_id" in features_df.columns:
        features_df = features_df.set_index("game_id", drop=False)

    # Ensembles
    margin_ensemble = NFLEnsemble(artifacts["margin_weights"], MODEL_DIR)
    total_ensemble  = NFLEnsemble(artifacts["total_weights"],  MODEL_DIR)
    margin_ensemble.load_models()
    total_ensemble.load_models()

    # Align matrices
    X_margin = features_df.reindex(columns=artifacts["margin_features"], fill_value=0.0).fillna(0.0)
    X_total  = features_df.reindex(columns=artifacts["total_features"],  fill_value=0.0).fillna(0.0)

    _check_features(X_margin, artifacts["margin_features"], "X_margin", debug_mode)
    _check_features(X_total,  artifacts["total_features"],  "X_total",  debug_mode)

    # Predict
    pred_t = time.time()
    margin_preds = margin_ensemble.predict(X_margin)
    total_preds  = total_ensemble.predict(X_total)
    if debug_mode:
        logger.debug("Predict time: %.3fs", time.time() - pred_t)

    scores_df = derive_scores_from_predictions(margin_preds, total_preds)
    _log_df(scores_df, "scores_df", debug_mode)

    # Merge with upcoming metadata
    upcoming_df = upcoming_df.set_index("game_id")
    final_df = upcoming_df.join(scores_df, how="left")

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
                "home_team_name":       row["home_team_name"],
                "away_team_name":       row["away_team_name"],
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
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    preds = generate_predictions(
        days_window=args.days,
        historical_lookback=args.lookback,
        debug_mode=args.debug,
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
