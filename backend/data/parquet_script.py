import pandas as pd
from pathlib import Path

# Adjust the path if necessary, based on where you finally placed it
# (It seems the ablation script checks both 'data/' and 'backend/data/')
history_path = Path("backend/data/history.parquet")
# Or potentially: history_path = Path("backend/data/history.parquet")

if history_path.exists():
    df = pd.read_parquet(history_path)
    print("--- File Found! ---")
    print(f"Shape (rows, columns): {df.shape}")
    print("\n--- Columns: ---")
    print(df.columns.tolist())
    print("\n--- Data Types: ---")
    print(df.info())
    print("\n--- First 5 Rows: ---")
    print(df.head())
else:
    print(f"ERROR: Could not find file at {history_path}")