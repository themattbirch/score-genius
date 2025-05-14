#!/usr/bin/env python3
# backend/scripts/feature_check.py

from pathlib import Path
from dotenv import load_dotenv
import re

# 0) Load .env so SUPABASE_* are in os.environ
THIS_DIR    = Path(__file__).parent            # .../backend/scripts
BACKEND_DIR = THIS_DIR.parent                  # .../backend
load_dotenv(BACKEND_DIR / ".env")

import os, sys, json, argparse
import pandas as pd

# 1) Make sure we can import config.py and nba_features
sys.path.insert(0, str(BACKEND_DIR))
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from supabase import create_client, Client
from nba_features.engine import run_feature_pipeline

# 2) Instantiate supabase_client once
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    sys.exit("❌ Supabase credentials not found in config.py or .env")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def normalize_team_name(name: str) -> str:
    # 1) strip leading/trailing whitespace, lowercase
    clean = name.strip().lower()
    # 2) remove any non-alphanumeric characters (keeps spaces)
    clean = re.sub(r'[^a-z0-9 ]+', '', clean)
    # 3) collapse multiple spaces
    clean = ' '.join(clean.split())
    # 4) optional: map known aliases
    ALIASES = {
      "la lakers": "los angeles lakers",
      "ny knicks": "new york knicks",
      # add any others you need…
    }
    return ALIASES.get(clean, clean)

def fetch_games(game_ids: list[int]) -> pd.DataFrame:
    """
    Fetch schedule rows for the given game_ids.
    Warn and skip any IDs we don’t find.
    """
    rows = []
    missing = []
    for gid in game_ids:
        resp = (
            supabase_client
            .table("nba_game_schedule")
            .select("*")
            .eq("game_id", int(gid))
            .limit(1)
            .execute()
        )
        data = resp.data or []
        if data:
            rows.append(data[0])
        else:
            missing.append(gid)

    if missing:
        print(f"⚠️  Could not find schedule rows for game_id(s): {missing}.  Skipping those.")
    if not rows:
        sys.exit("❌ No valid schedule rows found for any of the supplied game_ids.")

    return pd.DataFrame(rows)


def fetch_historical_games(since: str="2022-01-01") -> pd.DataFrame:
    resp = (
        supabase_client
        .table("nba_historical_game_stats")
        .select("*")
        .gte("game_date", since)
        .order("game_date")
        .execute()
    )
    return pd.DataFrame(resp.data or [])

def fetch_team_stats(since_year: int=2022) -> pd.DataFrame:
    resp = (
        supabase_client
        .table("nba_historical_team_stats")
        .select("*")
        .gte("season", since_year)
        .order("team_name")
        .order("season")
        .execute()
    )
    df = pd.DataFrame(resp.data or [])
    if df.empty:
        return df

    # 1) rename for the season transform
    df = df.rename(columns={"season": "season_year"})

    # 2) ensure same team-normalization
    df["team_norm"] = df["team_name"].map(normalize_team_name)

    return df

def fetch_recent_ids(n: int) -> list[int]:
    resp = (
        supabase_client
        .table("nba_historical_game_stats")
        .select("game_id")
        .order("game_date", desc=True)
        .limit(n)
        .execute()
    )
    return [r["game_id"] for r in resp.data or []]

def main():
    p = argparse.ArgumentParser(
        description="Inspect selected-feature values for NBA games"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--game-ids", nargs="+", type=int,
        help="One or more specific game_id values to inspect."
    )
    group.add_argument(
        "--last-n", type=int,
        help="Instead of explicit IDs, fetch the last N historical games."
    )
    p.add_argument(
        "--features-path",
        default=str(BACKEND_DIR / "models" / "saved" / "selected_features.json"),
        help="Path to selected_features.json"
    )
    args = p.parse_args()

    # 1) Load selected_features.json
    sel_path = Path(args.features_path)
    if not sel_path.is_file():
        sys.exit(f"❌ Could not find selected_features.json at {sel_path}")
    selected = json.loads(sel_path.read_text())

    # 2) Fetch full history once
    all_hist = fetch_historical_games("1900-01-01")
    if all_hist.empty:
        sys.exit("❌ Could not fetch any historical games.")

    # 3) Build sched_df and game_ids
    if args.last_n:
        # take the last N rows by date
        recent = (
            all_hist
            .sort_values("game_date", ascending=False)
            .head(args.last_n)
        )
        game_ids = recent["game_id"].tolist()
        sched_df  = recent.copy()
        print(f"→ Inspecting the {args.last_n} most recent games: {game_ids[:5]}…")
    else:
        game_ids = args.game_ids
        sched_df = fetch_games(game_ids)

    # 4) Now feed the full history *and* schedule slice into the pipeline
    #    so that rolling/H2H can look back over all prior games.
    team_stats_df       = fetch_team_stats(2022)
    full_df = run_feature_pipeline(
        df=sched_df,
        historical_games_df=all_hist,
        team_stats_df=team_stats_df,
        rolling_windows=[5, 10, 20],
        h2h_window=5,
        debug=False
    )
    if full_df.empty:
        sys.exit("❌ pipeline returned nothing")

    # 5) Drop constant cols
    const_cols = []
    for c in full_df.columns:
        try:
            if full_df[c].nunique(dropna=False) <= 1:
                const_cols.append(c)
        except TypeError:
            continue
    if const_cols:
        print(f"Dropping {len(const_cols)} constant features:", const_cols)
        full_df = full_df.drop(columns=const_cols)

    # 6) Intersect with your selected list
    available = [f for f in selected if f in full_df.columns]
    missing   = [f for f in selected if f not in full_df.columns]
    if missing:
        print(f"⚠️  {len(missing)} selected features not generated by pipeline:")
        for f in missing:
            print("   -", f)

    # 7) Slice out and display
    sub = (
        full_df
        .set_index("game_id")
        .loc[game_ids, available]
        .T
    )
    print("\nFeature values (rows=features, cols=game_ids):\n")
    print(sub.to_markdown())

    # 8) Highlight any warnings
    const_feats = sub[sub.nunique(axis=1) == 1]
    if not const_feats.empty:
        print("\n⚠️  CONSTANT across chosen games:")
        print(const_feats.to_markdown())

    zero_feats = sub[(sub == 0).all(axis=1)]
    if not zero_feats.empty:
        print("\n⚠️  ALL ZERO (defaults) – pipeline isn’t filling these:")
        print(zero_feats.to_markdown())

if __name__ == "__main__":
    main()
