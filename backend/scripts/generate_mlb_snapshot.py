# backend/scripts/generate_snapshot.py

"""
Generate JSON snapshot cards by running your feature pipeline in-process.
"""

import os
import sys
import json
import argparse
import pandas as pd

# 1) make backend importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# 2) load credentials
try:
    from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
    print("Using config.py for credentials")
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    print("Using environment variables for credentials")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL and/or SUPABASE_SERVICE_KEY not found.")

# 3) init client
from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 4) import pipeline
from mlb_features.engine import run_feature_pipeline

def fetch_games(game_ids: list[int | str]) -> pd.DataFrame:
    """Fetch one or more games from mlb_game_schedule."""
    rows = []
    for gid in game_ids:
        try:
            val = int(gid)
        except ValueError:
            val = gid
        res = (
            supabase
            .table("mlb_game_schedule")
            .select("*")
            .eq("game_id", val)
            .limit(1)
            .execute()
        )
        data = res.data or []
        if not data:
            raise RuntimeError(f"No row for game_id {gid}")
        rows.append(data[0])
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser(
        description="Show selected features for upcoming games"
    )
    p.add_argument(
        "--game-ids", "-g", nargs="+", required=True,
        help="One or more game_id values"
    )
    p.add_argument(
        "--selected-features", "-s",
        default="selected_features.json",
        help="Path to your JSON file of selected features"
    )
    args = p.parse_args()

    # fetch just those rows
    df_sched = fetch_games(args.game_ids)

    # run full feature pipeline (you can pass other args if needed)
    full_df = run_feature_pipeline(
        df=df_sched,
        historical_games_df=df_sched,  # only these games, for H2H uses
        team_stats_df=None,            # or load real team_stats if desired
        rolling_windows=[5,10,20],
        h2h_window=5,
        debug=False
    )

    if full_df.empty:
        print("⚠ pipeline returned no features")
        sys.exit(1)

    # load your 44-feature list
    selected = json.load(open(args.selected_features))

    # pick out by game_id
    sub = (
        full_df
        .set_index("game_id")
        .loc[args.game_ids, selected]
        .T
    )

    print("\n=== Selected Features ===")
    print(sub.to_string())

if __name__ == "__main__":
    main()
