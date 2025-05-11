# backend/scripts/generate_snapshot.py

"""
Generate JSON snapshot cards by running your feature pipeline in-process.
"""

import os
import sys
import json
import argparse
import pandas as pd

# 1) Ensure backend root is discoverable (for config.py and nba_features package)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# 2) Load Supabase credentials via config.py or environment
try:
    from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
    print("Using config.py for credentials")
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    print("Using environment variables for credentials")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL and/or SUPABASE_SERVICE_KEY not found.")

# 3) Initialize Supabase client
from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 4) Import the pipeline runner
from nba_features.engine import run_feature_pipeline

# 5) Define which headline stats to include
HEADLINE_STATS = [
    "pace",
    "net_rating_diff",
    "h2h_edge",
    "rest_advantage",
    "injury_factor",
]


def fetch_raw_game(game_id: str) -> pd.DataFrame:
    """
    Fetch a single game from nba_game_schedule using limit(1).
    Returns a 1-row DataFrame or raises if not found.
    """
    try:
        game_id_val = int(game_id)
    except ValueError:
        game_id_val = game_id

    try:
        res = (
            supabase
            .table("nba_game_schedule")
            .select("*")
            .eq("game_id", game_id_val)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Error fetching schedule for {game_id}: {e}")

    data = res.data or []
    if not data:
        raise RuntimeError(f"No schedule row found for game_id {game_id}")

    return pd.DataFrame([data[0]])


def generate_snapshot(game_id: str, output_dir: str):
    """Run feature pipeline and write out the top stats for one game"""
    try:
        df = fetch_raw_game(game_id)
    except RuntimeError as e:
        print(f"✖ Skipping {game_id}: {e}")
        return

    full_df = run_feature_pipeline(df)
    if full_df.empty:
        print(f"✖ Pipeline returned empty for {game_id}")
        return

    row = full_df.iloc[0]
    snapshot = {stat: float(row.get(stat, 0.0)) for stat in HEADLINE_STATS}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{game_id}.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f)
    print(f"✔ Wrote snapshot for {game_id} to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate snapshot JSON for NBA games.")
    parser.add_argument(
        "--game-id", nargs="+", required=True,
        help="One or more game_id values to snapshot."
    )
    parser.add_argument(
        "--output-dir", default="reports/snapshots",
        help="Directory to write snapshot JSON files."
    )
    args = parser.parse_args()

    for gid in args.game_id:
        generate_snapshot(gid, args.output_dir)


if __name__ == "__main__":
    main()
