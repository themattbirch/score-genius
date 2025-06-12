#!/usr/bin/env python
import sys, pathlib
import logging

# 1) Insert project root so “backend” is on the import path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 2) Silence verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 3) Now import your snapshot function
from backend.mlb_features.make_mlb_snapshots import make_mlb_snapshot

# ← Replace this with an actual game_id from mlb_game_schedule
TEST_GAME_ID = "164288"

def main():
    try:
        print(f"→ Generating snapshot for MLB game {TEST_GAME_ID}")
        make_mlb_snapshot(TEST_GAME_ID)
        print("✅ Snapshot completed")
    except Exception as e:
        print("❌ Snapshot failed:", e)
        raise

if __name__ == "__main__":
    main()
