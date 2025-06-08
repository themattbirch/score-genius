#tmp/dump_features.py

"""
Dump the full feature matrix that the prediction step feeds to the models
so we can compare column names, order, and distributions.
"""
import os
import sys
import pathlib
from datetime import datetime, timezone

# 1) make sure your project root is on PYTHONPATH so imports work
ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# 1b) load your .env
from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

import pandas as pd
from pathlib import Path

# now this will see SUPABASE_URL and SUPABASE_SERVICE_KEY
from backend.caching.supabase_client import supabase

# 3) your feature-engine
from backend.nba_features.engine import run_feature_pipeline

# 4) your prediction helper (to inspect expected feature names)
from backend.nba_score_prediction.prediction import load_trained_models


def load_schedule_today() -> pd.DataFrame:
    """Grab today's games from Supabase schedule table."""
    today = datetime.now(timezone.utc).date().isoformat()
    rows  = (
        supabase
        .table("nba_game_schedule")
        .select("*")
        .eq("game_date", today)
        .execute()
        .data
    ) or []
    return pd.DataFrame(rows)


def load_team_season_stats() -> pd.DataFrame:
    """Grab the latest snapshot from your historical team-stats table."""
    rows = (
        supabase
        .table("nba_historical_team_stats")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
        .data
    ) or []
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ————————— CLI args —————————
    import argparse
    parser = argparse.ArgumentParser(
        description="Dump the full feature matrix for inspection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except actually write the CSV to disk"
    )
    args = parser.parse_args()

    # ————————— prepare inputs —————————
    games_df      = load_schedule_today()
    team_stats_df = load_team_season_stats()

    from backend.nba_features.engine import DEFAULT_EXECUTION_ORDER

    # filter out the two modules that need box‐scores
    preseason_order = [m for m in DEFAULT_EXECUTION_ORDER 
                      if m not in ("advanced")]

    # ————————— run your pipeline —————————
    feature_df = run_feature_pipeline(
        games_df,
        historical_games_df=None,  # not used pre-game
        team_stats_df=team_stats_df,
        execution_order=preseason_order,
        debug=False,
    )

    # ————————— optionally dump to /tmp so we don’t overwrite selected_features.json —————————
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dump_dir = pathlib.Path("/tmp/feature_dumps")
    out_path = dump_dir / f"feature_dump_{timestamp}.csv"

    if args.dry_run:
        print(f"[DRY RUN] Would dump feature matrix to: {out_path}  shape={feature_df.shape}")
    else:
        dump_dir.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(out_path, index=False)
        print(f"[OK] feature matrix dumped to: {out_path}  shape={feature_df.shape}")


    # ————————— peek at model expectations —————————
    # load both models and the selected feature list
    models, selected_features = load_trained_models(
        model_dir=Path("models/saved"),
        load_feature_list=True
    )

    print("\nSelected feature list saved with the model:")
    print(selected_features)

    # if you need the raw sklearn feature_names_in_
    # assuming your wrapper exposes `model` attribute:
    if "svr" in models:
        svr_est = getattr(models["svr"], 'model', None)
        if hasattr(svr_est, 'feature_names_in_'):
            print("\nSVR estimator.feature_names_in_:")
            print(list(svr_est.feature_names_in_))
    if "ridge" in models:
        ridge_est = getattr(models["ridge"], 'model', None)
        if hasattr(ridge_est, 'feature_names_in_'):
            print("\nRidge estimator.feature_names_in_:")
            print(list(ridge_est.feature_names_in_))