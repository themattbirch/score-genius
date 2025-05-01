# backend/scripts/export_history.py
import os
from pathlib import Path

# ── Load backend/.env into os.environ ────────────────────────────────────────
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        # don’t overwrite anything already set
        os.environ.setdefault(key, val)

import pandas as pd
from caching.supabase_client import supabase

# where to write
out_path = Path("data") / "history.parquet"
out_path.parent.mkdir(exist_ok=True)

# the columns our FE pipeline needs
cols = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_fg_made", "home_fg_attempted",
    "away_fg_made", "away_fg_attempted",
    "home_3pm", "home_3pa", "away_3pm", "away_3pa",
    "home_ft_made", "home_ft_attempted", "away_ft_made", "away_ft_attempted",
    "home_off_reb", "home_def_reb", "home_total_reb",
    "away_off_reb", "away_def_reb", "away_total_reb",
    "home_turnovers", "away_turnovers",
    "home_ot", "away_ot"
]

resp = supabase.table("nba_historical_game_stats")\
               .select(",".join(cols))\
               .execute()
df = pd.DataFrame(resp.data or [])
df.to_parquet(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape}")
