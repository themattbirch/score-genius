# analyze_meta.py

import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# Define path relative to the project root (score-genius)
# Assumes you run analyze_meta.py from the score-genius directory
MAIN_MODELS_DIR = Path("./models/saved") # Correct path: score-genius/models/saved

META_MODEL_FILENAME = "stacking_meta_model.joblib"
meta_model_load_path = MAIN_MODELS_DIR / META_MODEL_FILENAME
# --- END Configuration ---

print(f"--- Analyzing Meta-Model Coefficients from: {meta_model_load_path} ---")
print(f"Absolute path being checked: {meta_model_load_path.resolve()}") # Added for debugging

# Check if the directory exists
if not MAIN_MODELS_DIR.exists():
    print(f"ERROR: Models directory not found at {MAIN_MODELS_DIR.resolve()}")
# Check if the file exists
elif not meta_model_load_path.exists():
    print(f"ERROR: Meta-model file not found at '{meta_model_load_path.resolve()}'")
    print(f"Files in {MAIN_MODELS_DIR.resolve()}: {list(MAIN_MODELS_DIR.glob('*'))}") # List files if not found
else:
    try:
        # Load the saved meta-model payload
        meta_model_payload = joblib.load(meta_model_load_path)
        meta_model_home = meta_model_payload.get('meta_model_home')
        meta_model_away = meta_model_payload.get('meta_model_away')
        meta_feature_names = meta_model_payload.get('meta_feature_names') # Crucial: Names must match coef order
        base_models_used = meta_model_payload.get('base_models_used')
        timestamp = meta_model_payload.get('training_timestamp', 'N/A')

        print(f"\nMeta-model trained on: {timestamp}")
        print(f"Base models included in meta-features: {base_models_used}")
        print(f"Meta-features used for training: {meta_feature_names}")

        if meta_model_home is None or meta_model_away is None or meta_feature_names is None:
            print("\nERROR: Payload in meta-model file is incomplete or missing required keys.")
        else:
            # Get coefficients
            coefs_home = meta_model_home.coef_
            coefs_away = meta_model_away.coef_

            print(f"\nBest Alpha Found (Home): {getattr(meta_model_home, 'alpha_', 'N/A - Not RidgeCV?')}")
            print(f"Best Alpha Found (Away): {getattr(meta_model_away, 'alpha_', 'N/A - Not RidgeCV?')}")

            # --- Display Coefficients ---
            if len(coefs_home) == len(meta_feature_names) and len(coefs_away) == len(meta_feature_names):
                coef_df = pd.DataFrame({
                    'Feature': meta_feature_names,
                    'Home_Meta_Coef': coefs_home,
                    'Away_Meta_Coef': coefs_away
                })
                # Add absolute value for sorting by magnitude
                coef_df['Home_Abs_Coef'] = coef_df['Home_Meta_Coef'].abs()
                coef_df['Away_Abs_Coef'] = coef_df['Away_Meta_Coef'].abs()

                print("\n--- Home Meta-Model Coefficients (Sorted by Absolute Value) ---")
                print(coef_df[['Feature', 'Home_Meta_Coef']].iloc[coef_df['Home_Abs_Coef'].argsort()[::-1]].to_string(index=False))

                print("\n--- Away Meta-Model Coefficients (Sorted by Absolute Value) ---")
                print(coef_df[['Feature', 'Away_Meta_Coef']].iloc[coef_df['Away_Abs_Coef'].argsort()[::-1]].to_string(index=False))

                print("\n--- Interpretation Notes ---")
                print("* Coefficients close to 1.0 for a model's own prediction (e.g., 'ridge_pred_home' in Home model)")
                print("  and close to 0.0 for others suggest the meta-model relies heavily on that single base model.")
                print("* Coefficients that sum close to 1.0 might indicate a weighted average.")
                print("* Coefficients far from [0, 1] or large negative values might indicate the meta-model is making")
                print("  significant corrections or finds complex interactions between base model predictions.")

            else:
                print("\nERROR: Mismatch between number of coefficients and feature names in saved model.")
                print(f"  Feature names ({len(meta_feature_names)}): {meta_feature_names}")
                print(f"  Home Coefs ({len(coefs_home)}): {coefs_home}")
                print(f"  Away Coefs ({len(coefs_away)}): {coefs_away}")

    except Exception as e:
        print(f"\nERROR: Failed to load or analyze meta-model: {e}")
        import traceback
        traceback.print_exc()