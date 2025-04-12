# test_inference.py

import sys
import os
import time
from datetime import datetime

# Add the backend folder to path (adjust as needed)
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # FIXED: sys.path.insert not sys.insert

# Import the function
from src.scripts.model_inference import run_model_inference

print(f"=== TEST STARTED AT {datetime.now()} ===")
print("Attempting to run model inference directly...")

try:
    # Call the function with default parameters (no forced retraining)
    run_model_inference(force_retrain=False)
    print("Function completed successfully!")
except Exception as e:
    print(f"Error running model_inference: {e}")
    import traceback
    traceback.print_exc()

print(f"=== TEST COMPLETED AT {datetime.now()} ===")

# Add this to your test script
from pathlib import Path
import joblib

model_path = Path("./models/pregame_model.pkl")  # Adjust path as needed
if model_path.exists():
    model = joblib.load(model_path)
    print("Model contents:")
    if isinstance(model, dict):
        print(f"Keys: {model.keys()}")
        if 'features' in model:
            print(f"Features: {model['features'][:10]} ...")
        if 'models' in model:
            print(f"Model components: {list(model['models'].keys())}")
            for name, component in model['models'].items():
                if hasattr(component, 'feature_names_in_'):
                    print(f"{name} features: {component.feature_names_in_[:10]} ...")