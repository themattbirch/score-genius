# scripts/run_ablation.py
import itertools, json, pickle, pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from backend.features.legacy import feature_engineering as fe_legacy
from backend.features import momentum, rolling, rest, h2h, season, form, advanced
from pathlib import Path
Path("reports").mkdir(exist_ok=True)

MODULES = {
    "momentum": momentum.transform,
    "rolling":  rolling.transform,
    "rest":     rest.transform,
    "h2h":      h2h.transform,
    "season":   season.transform,
    "form":     form.transform,
    "advanced": advanced.transform,
}

def apply_transforms(df, blocks):
    out = df.copy()
    for b in blocks:
        out = MODULES[b](out)
    return out

def run():
  # Determine the repo root: backend/scripts → backend → <project root>
  SCRIPT_DIR = Path(__file__).resolve().parent      # .../backend/scripts
  BACKEND_DIR = SCRIPT_DIR.parent                   # .../backend
  REPO_ROOT   = BACKEND_DIR.parent                  # project root

  # Try both possible locations:
  candidates = [
      REPO_ROOT / "data" / "history.parquet",
      BACKEND_DIR / "data" / "history.parquet"
  ]
  for candidate in candidates:
      if candidate.exists():
          raw_path = candidate
          break
  else:
      raise FileNotFoundError(
          f"Could not find history.parquet at any of: {candidates}"
      )

  raw = pd.read_parquet(raw_path)
  y   = raw["home_score"] + raw["away_score"]             # example target
  X0  = raw[["home_score","away_score"]]
  baseline = cross_val_score(fe_legacy.RidgeScorePredictor().model, X0, y, cv=5, scoring="neg_root_mean_squared_error").mean()
  
  results = {"baseline": -baseline}
  for L in range(1, len(MODULES)+1):
      for combo in itertools.combinations(MODULES.keys(), L):
          feats = apply_transforms(raw, combo)
          # drop target + any non-feature cols, then select only numeric columns
          X = (
              feats
              .drop(columns=["home_score", "away_score", "game_id"], errors="ignore")
              .select_dtypes(include=[np.number])
          )
          rmse = -cross_val_score(fe_legacy.RidgeScorePredictor().model, X, y,
                                    cv=5, scoring="neg_root_mean_squared_error").mean()
          results["+ ".join(combo)] = rmse
          print(f"{' + '.join(combo):50} ΔRMSE {baseline - rmse:+.3f}")
  json.dump(results, open("reports/ablation_results.json","w"), indent=2)

if __name__ == "__main__":
    run()
