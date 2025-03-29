# train_models.py (New or existing training script)

import pandas as pd
import numpy as np
import joblib
import os
import time
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split # Or use time-series split logic
import logging
from typing import List, Dict, Any, Optional

# Import necessary components
# Assuming models.py is in ../score_prediction/
from nba_score_prediction.models import QuarterSpecificModelSystem # Adjust import path as needed
from nba_score_prediction.feature_engineering import NBAFeatureEngine # Adjust import path
# Import specific model types if needed (e.g., XGBoost)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# --- Configuration ---
MODEL_SAVE_DIR = 'models/quarterly' # Directory to save trained quarter models
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
TARGET_COL_PREFIX = 'home' # Or 'total', 'away' depending on what quarter models predict
QUARTERS_TO_TRAIN = [1, 2] # Example: Focus on early quarters
BASELINE_RMSE = {1: 8.0, 2: 7.5} # Example baseline RMSE values for comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (Adapted from the original class) ---

def initialize_trainable_models(quarters: List[int] = [1, 2], model_type='xgboost') -> Dict[int, Dict[str, Any]]:
    """
    Initialize models specifically for training purposes.

    Args:
        quarters: List of quarters to initialize models for (default: [1, 2]).
        model_type: Type of model to initialize ('xgboost' supported currently).

    Returns:
        Dictionary mapping quarter number to a dict of initialized models.
        e.g., {1: {'xgb_tuned': model_instance}, 2: ... }
    """
    logger.info(f"Initializing trainable models for quarters {quarters} (type: {model_type})...")
    trainable_models = {q: {} for q in quarters} # Initialize structure

    if model_type.lower() == 'xgboost':
        if xgb is None:
            logger.error("XGBoost library not found. Cannot initialize XGBoost models.")
            return trainable_models # Return empty structure
        try:
            # Define parameters (could load from config)
            params_q1 = {
                'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 4,
                'min_child_weight': 2, 'gamma': 1, 'subsample': 0.75,
                'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
                'objective': 'reg:squarederror', 'random_state': 42
            }
            params_q2 = {
                'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4,
                'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.8,
                'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                'objective': 'reg:squarederror', 'random_state': 42
            }

            if 1 in quarters:
                trainable_models[1]['xgb_tuned'] = xgb.XGBRegressor(**params_q1)
                logger.info("Initialized trainable XGBoost model for Q1.")
            if 2 in quarters:
                trainable_models[2]['xgb_tuned'] = xgb.XGBRegressor(**params_q2)
                logger.info("Initialized trainable XGBoost model for Q2.")
            # Add initialization for Q3/Q4 if needed

        except Exception as e:
             logger.error(f"Error initializing XGBoost models: {e}")

    else:
        logger.warning(f"Model type '{model_type}' not currently supported for initialization.")

    return trainable_models


def train_quarter_models(training_data: pd.DataFrame,
                         trainable_models: Dict[int, Dict[str, Any]],
                         feature_sets: Dict[int, Dict[str, List[str]]], # e.g., {1: {'basic': [...], 'adv': [...]}}
                         target_col_prefix: str = 'home',
                         quarters_to_train: List[int] = [1, 2]) -> Dict:
    """
    Train the initialized models using provided data and feature sets.

    Args:
        training_data: DataFrame containing features and target variables.
        trainable_models: Dictionary of initialized models from `initialize_trainable_models`.
        feature_sets: Dictionary defining feature sets for each quarter to be trained.
                      e.g., {1: {'basic': [...], 'advanced': [...]}}
        target_col_prefix: Prefix for the target quarter score columns (e.g., 'home', 'away').
        quarters_to_train: List of quarters for which to train models.

    Returns:
        Dictionary containing training results (trained models, metrics, feature importance).
        Structure: { quarter_num: { model_variation_key: { 'model': ..., 'features': ..., ... } } }
    """
    logger.info(f"Starting training for quarters {quarters_to_train} with target prefix '{target_col_prefix}'...")
    results = {q: {} for q in quarters_to_train}
    if not isinstance(training_data, pd.DataFrame) or training_data.empty:
        logger.error("No training data provided.")
        return results

    for q in quarters_to_train:
        if q not in trainable_models or not trainable_models[q]:
             logger.warning(f"No trainable models initialized for Q{q}. Skipping training.")
             continue

        target_col = f"{target_col_prefix}_q{q}"
        if target_col not in training_data.columns:
             logger.error(f"Target column '{target_col}' not found for Q{q}. Skipping training.")
             continue

        logger.info(f"--- Training Q{q} models (Target: {target_col}) ---")
        q_feature_sets_to_train = feature_sets.get(q)
        if not q_feature_sets_to_train:
             logger.warning(f"No specific feature sets defined for training Q{q}. Skipping.")
             continue

        y = training_data[target_col]

        for feature_set_name, features in q_feature_sets_to_train.items():
            logger.info(f"Using feature set: '{feature_set_name}' ({len(features)} features)")
            valid_features = [f for f in features if f in training_data.columns]
            missing_count = len(features) - len(valid_features)
            if missing_count > 0:
                missing_fs = set(features) - set(valid_features)
                logger.warning(f"Missing {missing_count} features for Q{q} '{feature_set_name}' set: {missing_fs}")
            if not valid_features:
                logger.warning(f"No valid features found for Q{q} '{feature_set_name}'. Skipping.")
                continue

            X = training_data[valid_features].copy()
            # Optional: Handle missing values if not done in pipeline
            # X = X.fillna(X.median()) # Example

            # Train each model initialized for this quarter
            for model_name, model in trainable_models[q].items():
                logger.debug(f"Training model: {model_name}")
                try:
                    start_time = time.time()
                    # Important: Clone the model if you plan to train it multiple times (e.g., with different feature sets)
                    # from sklearn.base import clone
                    # model_instance = clone(model) # Or ensure initialization provides fresh instances
                    model_instance = model # Assuming fresh instance was provided by initialize_...
                    model_instance.fit(X, y)
                    train_time = time.time() - start_time

                    # Optional: Evaluate on training set (can indicate overfitting)
                    y_pred_train = model_instance.predict(X)
                    mse_train = np.mean((y - y_pred_train) ** 2)
                    mae_train = np.mean(np.abs(y - y_pred_train))

                    trained_model_key = f"{model_name}_{feature_set_name}"
                    results[q][trained_model_key] = {
                        'model': model_instance, # Store the trained model instance itself
                        'features': valid_features,
                        'train_mse': mse_train,
                        'train_mae': mae_train,
                        'train_time_sec': train_time,
                        'feature_importance': dict(zip(valid_features, model_instance.feature_importances_)) if hasattr(model_instance, 'feature_importances_') else {}
                    }
                    logger.info(f"Trained Q{q} {model_name} with '{feature_set_name}' features: MAE={mae_train:.3f} (Train Time: {train_time:.2f}s)")

                    # --- Save the best performing model variation for this quarter ---
                    # Example: Save the model trained with the 'advanced' feature set
                    # You might add logic here to decide WHICH variation to save as the final qX_model.pkl
                    if feature_set_name == 'advanced': # Or based on performance metric
                        save_path = os.path.join(MODEL_SAVE_DIR, f'q{q}_model.pkl')
                        try:
                             joblib.dump(model_instance, save_path)
                             logger.info(f"Saved trained model for Q{q} ('{feature_set_name}' features) to {save_path}")
                        except Exception as save_e:
                             logger.error(f"Error saving trained model for Q{q} to {save_path}: {save_e}")


                except Exception as e:
                    logger.error(f"Error training Q{q} {model_name} with '{feature_set_name}' features: {e}")
                    traceback.print_exc() # Print detailed traceback for debugging

    logger.info("--- Quarter model training finished ---")
    return results


def evaluate_trained_models(test_data: pd.DataFrame, trained_models_results: Dict, target_col_prefix: str = 'home') -> pd.DataFrame:
    """
    Evaluate models that were trained (using the results from train_quarter_models).

    Args:
        test_data: DataFrame containing test features and target variables.
        trained_models_results: The dictionary returned by train_quarter_models.
        target_col_prefix: Prefix for the target quarter score columns.

    Returns:
        DataFrame with evaluation metrics (MSE, RMSE, MAE) for each trained model variation.
    """
    logger.info("Evaluating trained quarter models on test data...")
    eval_results = []
    if not isinstance(test_data, pd.DataFrame) or test_data.empty:
        logger.error("No test data provided for evaluation.")
        return pd.DataFrame()

    for quarter, trained_variations in trained_models_results.items():
        target_col = f"{target_col_prefix}_q{quarter}"
        if target_col not in test_data.columns:
            logger.warning(f"Target column '{target_col}' not found in test data for Q{quarter}. Skipping evaluation.")
            continue

        y_test = test_data[target_col]
        logger.debug(f"Evaluating Q{quarter} models...")

        for model_variation_key, model_info in trained_variations.items():
             model = model_info.get('model')
             features = model_info.get('features')
             # Handle potential key format errors
             try:
                 model_base_name, feature_set_name = model_variation_key.rsplit('_', 1)
             except ValueError:
                 model_base_name = model_variation_key
                 feature_set_name = "unknown"
                 logger.warning(f"Could not parse model key '{model_variation_key}'. Using defaults.")


             if not model or not features:
                 logger.warning(f"Model or features missing for '{model_variation_key}'. Skipping evaluation.")
                 continue

             if all(f in test_data.columns for f in features):
                 X_test = test_data[features]
                 try:
                     y_pred_test = model.predict(X_test)
                     mse = np.mean((y_test - y_pred_test) ** 2)
                     rmse = np.sqrt(mse)
                     mae = np.mean(np.abs(y_test - y_pred_test))

                     eval_results.append({
                         'quarter': f'Q{quarter}',
                         'model': model_base_name,
                         'feature_set': feature_set_name,
                         'mse': mse,
                         'rmse': rmse,
                         'mae': mae,
                         'sample_size': len(X_test)
                     })
                     # logger.debug(f"Evaluated Q{quarter} {model_base_name} ('{feature_set_name}'): RMSE={rmse:.3f}, MAE={mae:.3f}")
                 except Exception as e:
                      logger.error(f"Error evaluating Q{quarter} {model_base_name} ('{feature_set_name}'): {e}")
             else:
                 missing_eval_features = [f for f in features if f not in test_data.columns]
                 logger.warning(f"Missing features in test data for Q{quarter} {model_base_name} ('{feature_set_name}'): {missing_eval_features}. Skipping evaluation.")

    if not eval_results:
        logger.warning("No evaluation results generated.")
        return pd.DataFrame()

    logger.info(f"Evaluation complete. Generated results for {len(eval_results)} model variations.")
    return pd.DataFrame(eval_results)


def compare_evaluation_to_baseline(eval_df: pd.DataFrame, baseline_rmse: Dict[int, float]) -> pd.DataFrame:
    """
    Compare evaluation results (RMSE) against baseline values.

    Args:
        eval_df: DataFrame returned by evaluate_trained_models.
        baseline_rmse: Dictionary mapping quarter number (int) to baseline RMSE value. e.g., {1: 8.0, 2: 7.5}

    Returns:
        DataFrame with added baseline comparison columns (improvement, pct_improvement).
    """
    if not isinstance(eval_df, pd.DataFrame) or eval_df.empty:
        logger.warning("Evaluation DataFrame is empty. Cannot compare to baseline.")
        return pd.DataFrame()
    if not baseline_rmse:
         logger.warning("Baseline RMSE dictionary is empty. Cannot compare.")
         return eval_df

    logger.info("Comparing evaluation results to baseline RMSE...")
    comparison = eval_df.copy()
    try:
         # Extract quarter number safely
         comparison['q_num'] = comparison['quarter'].str.extract('(\d+)').astype(int)
    except Exception as e:
         logger.error(f"Could not extract quarter number from 'quarter' column: {e}")
         return eval_df # Return original df if extraction fails

    comparison['baseline_rmse'] = comparison['q_num'].map(baseline_rmse) # Use map for cleaner NaN handling
    comparison = comparison.dropna(subset=['baseline_rmse', 'rmse']) # Drop rows where baseline or eval RMSE is missing

    if comparison.empty:
         logger.warning("No matching baseline RMSE values found or evaluation RMSE missing for evaluated quarters.")
         return eval_df.assign(baseline_rmse=np.nan, rmse_improvement=np.nan, pct_improvement=np.nan) # Return with empty comparison cols

    comparison['rmse_improvement'] = comparison['baseline_rmse'] - comparison['rmse']
    # Handle potential division by zero if baseline_rmse is 0
    comparison['pct_improvement'] = np.where(
        comparison['baseline_rmse'] != 0,
        (comparison['rmse_improvement'] / comparison['baseline_rmse']) * 100,
        0 # Or np.inf or np.nan depending on desired handling
    )


    logger.info("Baseline comparison complete.")
    return comparison.sort_values(['q_num', 'pct_improvement'], ascending=[True, False]).drop(columns=['q_num'])


# --- Main Training Execution ---
if __name__ == "__main__":
    logger.info("Starting Quarter-Specific Model Training Script...")

    # 1. Load Data (Replace with your actual data loading)
    logger.info("Loading training and testing data...")
    # train_df = pd.read_csv('path/to/your/train_data_with_features.csv', parse_dates=['game_date'])
    # test_df = pd.read_csv('path/to/your/test_data_with_features.csv', parse_dates=['game_date'])
    # Dummy data for demonstration:
    train_df = pd.DataFrame(np.random.rand(200, 15), columns=[f'feature_{i}' for i in range(15)])
    train_df['home_q1'] = np.random.randint(20, 35, 200)
    train_df['home_q2'] = np.random.randint(20, 35, 200)
    train_df['away_q1'] = np.random.randint(20, 35, 200)
    train_df['away_q2'] = np.random.randint(20, 35, 200)
    train_df['rest_days_home'] = np.random.randint(1, 4, 200)
    train_df['rest_days_away'] = np.random.randint(1, 4, 200)
    train_df['rest_advantage'] = train_df['rest_days_home'] - train_df['rest_days_away']
    train_df['rolling_home_score'] = np.random.rand(200) * 30 + 95
    train_df['rolling_away_score'] = np.random.rand(200) * 30 + 95
    train_df['prev_matchup_diff'] = np.random.randn(200) * 10
    train_df['is_back_to_back_home'] = np.random.randint(0, 2, 200)
    train_df['is_back_to_back_away'] = np.random.randint(0, 2, 200)
    train_df['q1_to_q2_momentum'] = np.random.randn(200) * 5


    test_df = pd.DataFrame(np.random.rand(50, 15), columns=[f'feature_{i}' for i in range(15)])
    test_df['home_q1'] = np.random.randint(20, 35, 50)
    test_df['home_q2'] = np.random.randint(20, 35, 50)
    test_df['away_q1'] = np.random.randint(20, 35, 50)
    test_df['away_q2'] = np.random.randint(20, 35, 50)
    test_df['rest_days_home'] = np.random.randint(1, 4, 50)
    test_df['rest_days_away'] = np.random.randint(1, 4, 50)
    test_df['rest_advantage'] = test_df['rest_days_home'] - test_df['rest_days_away']
    test_df['rolling_home_score'] = np.random.rand(50) * 30 + 95
    test_df['rolling_away_score'] = np.random.rand(50) * 30 + 95
    test_df['prev_matchup_diff'] = np.random.randn(50) * 10
    test_df['is_back_to_back_home'] = np.random.randint(0, 2, 50)
    test_df['is_back_to_back_away'] = np.random.randint(0, 2, 50)
    test_df['q1_to_q2_momentum'] = np.random.randn(50) * 5

    # Ensure data is clean (e.g., handle NaNs if necessary before training)
    # train_df = train_df.fillna(train_df.median())
    # test_df = test_df.fillna(test_df.median())


    # 2. Define Feature Sets for Training
    # These are the specific combinations you want to test during training
    # Using subsets from the original class for example
    q1_training_feature_sets = {
        'basic': [
            'rest_days_home', 'rest_days_away', 'rest_advantage',
            'rolling_home_score', 'rolling_away_score'
        ],
        'advanced': [
            'rest_days_home', 'rest_days_away', 'rest_advantage',
            'is_back_to_back_home', 'is_back_to_back_away',
            'rolling_home_score', 'rolling_away_score',
            'prev_matchup_diff'
        ]
    }
    q2_training_feature_sets = {
        'basic': [
            'home_q1', 'away_q1', 'rest_advantage'
        ],
        'advanced': [
            'home_q1', 'away_q1',
            'rest_advantage', 'q1_to_q2_momentum',
            'rolling_home_score', 'rolling_away_score',
            'prev_matchup_diff'
        ]
    }
    all_training_feature_sets = {
        1: q1_training_feature_sets,
        2: q2_training_feature_sets
        # Add Q3/Q4 feature sets here if training them
    }


    # 3. Initialize Models for Training
    # Using XGBoost as an example
    models_to_train = initialize_trainable_models(quarters=QUARTERS_TO_TRAIN, model_type='xgb')

    # 4. Train Models
    if any(models_to_train.values()): # Check if initialization was successful
        training_results = train_quarter_models(
            training_data=train_df,
            trainable_models=models_to_train,
            feature_sets=all_training_feature_sets,
            target_col_prefix=TARGET_COL_PREFIX,
            quarters_to_train=QUARTERS_TO_TRAIN
        )

        # 5. Evaluate Models
        if training_results:
             evaluation_df = evaluate_trained_models(
                 test_data=test_df,
                 trained_models_results=training_results,
                 target_col_prefix=TARGET_COL_PREFIX
             )

             # 6. Compare to Baseline
             if not evaluation_df.empty:
                  comparison_df = compare_evaluation_to_baseline(
                      eval_df=evaluation_df,
                      baseline_rmse=BASELINE_RMSE
                  )

                  logger.info("\n--- Evaluation Results ---")
                  print(evaluation_df.round(3).to_string())

                  logger.info("\n--- Baseline Comparison ---")
                  print(comparison_df[['quarter','model','feature_set','rmse','baseline_rmse','pct_improvement']].round(3).to_string())
             else:
                  logger.warning("Evaluation produced no results.")

        else:
            logger.error("Training failed or produced no results.")
    else:
        logger.error("Model initialization failed. Cannot proceed with training.")

    logger.info("Quarter-Specific Model Training Script Finished.")