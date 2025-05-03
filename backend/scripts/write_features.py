# backend/scripts/write_features.py
from __future__ import annotations
import sys, json
from pathlib import Path

import logging
logging.getLogger("backend.nba_features.rolling").setLevel(logging.ERROR)

# ─── Ensure project root is on PYTHONPATH ────────────────────────────────────
SCRIPT = Path(__file__).resolve()
ROOT   = SCRIPT.parents[2]
sys.path.insert(0, str(ROOT))

# ─── Load .env & Supabase client ────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

from supabase import create_client
from backend import config
supa = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)

# ─── Imports for feature pipeline and Lasso ─────────────────────────────────
from backend.nba_features.engine import run_feature_pipeline
from sklearn.linear_model      import LassoCV
from sklearn.feature_selection import SelectFromModel
import pandas as pd

# ─── Prepare output folder ───────────────────────────────────────────────────
MODELS_DIR = ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1) Fetch & sample historical games ──────────────────────────────────────
print("⏳ Loading historical games…")
resp = supa.table("nba_historical_game_stats").select("*").execute()
hist_df = pd.DataFrame(resp.data or [])
if hist_df.empty:
    print("❌ No historical games fetched—aborting.")
    sys.exit(1)
hist_df = hist_df.sample(min(len(hist_df), 1000), random_state=42).reset_index(drop=True)
print(f"✔ Sampled {len(hist_df)} rows for feature selection")

# ─── 2) Fetch team-stats ─────────────────────────────────────────────────────
print("⏳ Loading team stats…")
resp2 = supa.table("nba_historical_team_stats").select("*").execute()
team_stats_df = pd.DataFrame(resp2.data or [])

# ─── 3) Run your leak-proof feature pipeline ─────────────────────────────────
print("⏳ Running feature pipeline…")
features_df = run_feature_pipeline(
    df=hist_df.copy(),
    historical_games_df=hist_df,
    team_stats_df=team_stats_df,
    rolling_windows=[],
    h2h_window=7,
    debug=False,
)
if features_df is None or features_df.empty:
    print("❌ Feature pipeline returned no data—aborting.")
    sys.exit(1)

# ─── 4) Lasso‐based feature selection ────────────────────────────────────────
# Drop any target‐derived or leakage columns first
leakage_cols = ["home_score", "away_score", "total_score", "point_diff"]
X = (
    features_df
      .drop(columns=leakage_cols, errors="ignore")   # 🔥 drop targets 🔥
      .select_dtypes(include="number")               # keep only numeric
      .copy()
)
y_home = features_df["home_score"]
y_away = features_df["away_score"]

lasso_home = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X, y_home)
lasso_away = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X, y_away)

sel_home = SelectFromModel(lasso_home, prefit=True, threshold=1e-5)
sel_away = SelectFromModel(lasso_away, prefit=True, threshold=1e-5)
mask = sel_home.get_support() | sel_away.get_support()
final_features = list(X.columns[mask])

print(f"✔ Lasso selected {len(final_features)} features.")

# ─── 5) Write out JSON and exit ──────────────────────────────────────────────
out = MODELS_DIR / "selected_features.json"
with open(out, "w") as fp:
    json.dump(final_features, fp, indent=2)
print(f"✅ Wrote selected_features.json to {out}")
sys.exit(0)
