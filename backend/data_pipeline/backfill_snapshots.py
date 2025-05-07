"""
Back‑fill season‑final snapshots for 2014‑15 … 2023‑24.

Usage:
    $ python backfill_snapshots.py
"""

import sys
import os
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

# ── adjust PYTHONPATH so we can import the pipeline module
HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from data_pipeline.nba_team_stats_historical import snapshot_full_season

if __name__ == "__main__":
    start, end = 2018, 2023  # inclusive start, inclusive end‑year of start season
    for yr in range(start, end + 1):
        season_lbl = f"{yr}-{yr+1}"
        print(f"→ Back‑filling {season_lbl}")
        snapshot_full_season(season_lbl)
