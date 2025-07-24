# backend/nfl_score_prediction/prediction.py
"""
prediction.py - NFL Score Prediction Generation Script

This script generates pre-game score predictions for upcoming NFL games.
It orchestrates the entire inference pipeline:
1.  Loads data for upcoming games and recent historical context.
2.  Loads two sets of trained model artifacts: one for predicting the point
    margin and one for the total points.
3.  Runs the feature engineering pipeline on the upcoming games.
4.  Uses two separate ensembles to predict margin and total.
5.  Derives the final home/away scores from the ensemble outputs.
6.  Upserts the final predictions to the Supabase database.
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
    from backend.nfl_score_prediction.models import (
        MODEL_DIR,
        derive_scores_from_predictions,
    )
except ImportError as e:
    print(f"Import error: {e}. Ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# ----------------------------------------------------------------------------- #
# Paths & logging
# ----------------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def get_supabase_client() -> Optional[Client]:
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error("Supabase client init failed: %s", e)
        return None


def load_prediction_artifacts(model_dir: Path) -> Dict[str, Any]:
    """Load selected feature lists and ensemble weights for both targets."""
    artifacts: Dict[str, Any] = {}
    for target in ("margin", "total"):
        try:
            feats = json.loads((model_dir / f"nfl_{target}_selected_features.json").read_text())
            wts   = json.loads((model_dir / f"nfl_{target}_ensemble_weights.json").read_text())
            artifacts[f"{target}_features"] = feats
            artifacts[f"{target}_weights"]  = wts
            logger.info("Artifacts loaded for %s", target)
        except FileNotFoundError as e:
            logger.critical("Missing artifact for %s: %s", target, e)
            return {}
    return artifacts


def chunked(lst: List[Dict[str, Any]], size: int = 500):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def upsert_score_predictions(preds: List[Dict[str, Any]], sb: Client) -> int:
    if not preds:
        return 0

    total = 0
    for batch in chunked(preds, 500):
        try:
            resp = sb.table("nfl_game_schedule").upsert(batch, on_conflict="game_id").execute()
            total += len(resp.data or [])
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
        d = datetime.fromisoformat(r["game_date"]).strftime("%Y-%m-%d")
        matchup = f"{r['away_team_name']} @ {r['home_team_name']}"
        score = f"{r['predicted_away_score']:.1f} - {r['predicted_home_score']:.1f}"
        print(f"{d:<12}{matchup:<42}{score:<20}")
    print("-" * 80)


# ----------------------------------------------------------------------------- #
# Core pipeline
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

    artifacts = load_prediction_artifacts(MODEL_DIR)
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

    # Feature generation
    logger.info("Building features for %d games…", len(upcoming_df))
    features_df = run_nfl_feature_pipeline(
        games_df=upcoming_df,
        historical_games_df=games_hist,
        historical_team_game_stats_df=stats_hist,
        debug=debug_mode,
    )

    if features_df.empty:
        logger.error("Feature DF empty. Abort.")
        return []

    # Ensure game_id present and indexable
    if "game_id" in features_df.columns:
        features_df = features_df.set_index("game_id", drop=False)

    # Instantiate ensembles and load models once
    margin_ensemble = NFLEnsemble(artifacts["margin_weights"], MODEL_DIR)
    total_ensemble  = NFLEnsemble(artifacts["total_weights"],  MODEL_DIR)
    margin_ensemble.load_models()
    total_ensemble.load_models()

    # Align features
    X_margin = features_df.reindex(columns=artifacts["margin_features"], fill_value=0.0).fillna(0.0)
    X_total  = features_df.reindex(columns=artifacts["total_features"],  fill_value=0.0).fillna(0.0)

    # Predict
    margin_preds = margin_ensemble.predict(X_margin)
    total_preds  = total_ensemble.predict(X_total)

    scores_df = derive_scores_from_predictions(margin_preds, total_preds)

    # Merge with upcoming metadata
    upcoming_df = upcoming_df.set_index("game_id")
    final_df = upcoming_df.join(scores_df, how="left")

    payload: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for gid, row in final_df.iterrows():
        if pd.isna(row.get("predicted_home_score")):
            continue
        payload.append(
            {
                "game_id": int(gid),
                "game_date": row["game_date"].isoformat() if hasattr(row["game_date"], "isoformat") else str(row["game_date"]),
                "home_team_id": int(row["home_team_id"]),
                "away_team_id": int(row["away_team_id"]),
                "home_team_name": row["home_team_name"],
                "away_team_name": row["away_team_name"],
                "predicted_home_score": round(float(row["predicted_home_score"]), 2),
                "predicted_away_score": round(float(row["predicted_away_score"]), 2),
                "prediction_utc": now_iso,
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
    parser.add_argument("--debug", action="store_true", help="Debug logging.")
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
            upsert_score_predictions(preds, sb)
        else:
            logger.error("Supabase unavailable; cannot upsert.")
    else:
        logger.info("--no-upsert specified. Skipping upsert.")


if __name__ == "__main__":
    main()
