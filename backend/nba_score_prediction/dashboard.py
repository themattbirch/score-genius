# backend/score_prediction/dashboard.py

"""
Module for generating dashboards and summary visualizations related to
NBA score predictions. Includes a console dashboard and functions for
plotting key metrics, feature importance, and model agreement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, List, Union

# --- Project Module Imports ---
try:
    # If dashboard needs access to models directly (less common, usually data is passed)
    # from .models import BaseScorePredictor, XGBoostScorePredictor, ...
    # If dashboard needs betting logic (likely)
    from .simulation import generate_betting_recommendations
except ImportError:
    logger = logging.getLogger(__name__) # Need logger defined early
    logger.warning("Could not import generate_betting_recommendations from .simulation. Using dummy function.")
    # Define dummy betting function if import fails
    def generate_betting_recommendations(game_dict):
         return {
             'moneyline': {'recommendation': 'N/A - Func Missing'},
             'spread': {'recommendation': 'N/A - Func Missing'},
             'total': {'recommendation': 'N/A - Func Missing'}
         }

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Plotting Configuration ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('colorblind')
except Exception:
    logger.warning("Seaborn style 'seaborn-v0_8-whitegrid' not found. Using default styles.")
    plt.style.use('default')

SMALL_FIG_SIZE = (10, 6)
DEFAULT_FIG_SIZE = (12, 8) # Renamed from previous LARGE
DASHBOARD_PLOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'dashboard_plots'
DASHBOARD_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Core Metric Calculation (Simplified for Dashboard Use) ---

def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]
                                 ) -> Dict[str, float]:
    """
    Calculates standard regression metrics and mean error (bias).
    Suitable for displaying key metrics on a dashboard.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.

    Returns:
        Dictionary containing MSE, RMSE, MAE, R-squared, and Mean Error (Bias).
        Returns dict with NaNs if input arrays are empty or lengths mismatch.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            logger.error(f"Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mean_error_bias': np.nan}
        if len(y_true) == 0:
            logger.warning("Input arrays are empty.")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mean_error_bias': np.nan}

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(mask) == 0:
             logger.warning("No valid non-NaN pairs found.")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mean_error_bias': np.nan}

        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean) if len(y_true_clean) >= 2 else np.nan
        mean_error = np.mean(y_pred_clean - y_true_clean)

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mean_error_bias': mean_error, 'sample_count': len(y_true_clean)}
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mean_error_bias': np.nan}


# --- Pregame Console Dashboard Function ---

def generate_pregame_dashboard(upcoming_predictions=None, models_metadata=None, backtest_summary=None):
    """
    Generates a text-based summary dashboard for the pregame prediction system (CONSOLE OUTPUT).

    Args:
        upcoming_predictions: DataFrame with upcoming predictions. Must contain columns like
                              'game_date', 'home_team', 'away_team', 'predicted_home_score',
                              'predicted_away_score', 'win_probability'.
        models_metadata: Dictionary containing metadata about trained models, specifically
                         metrics like MAE and potentially feature importance. Expected structure:
                         {'home_score': {'metrics': {'test_mae': ..., 'feature_importance': {...}}}, ...}
                         (Passed *data* about models, not necessarily the model objects themselves).
        backtest_summary: Optional dictionary or DataFrame with key backtest results (e.g., overall MAE).
    """
    # NOTE: Relies on imported `generate_betting_recommendations` function.

    print("\n" + "="*80)
    print(" "*30 + "NBA PREGAME PREDICTION DASHBOARD")
    print("="*80 + "\n")

    # --- Model Performance Summary ---
    print("MODEL PERFORMANCE SUMMARY:")
    if models_metadata and isinstance(models_metadata, dict):
        found_metrics = False
        for target, meta in models_metadata.items():
            if isinstance(meta, dict) and 'metrics' in meta and isinstance(meta['metrics'], dict):
                mae = meta['metrics'].get('test_mae') # Use .get for safety
                rmse = meta['metrics'].get('test_rmse') # Example: add RMSE if available
                if pd.notna(mae):
                    print(f"- {str(target).replace('_', ' ').title()} Test MAE: {mae:.2f}")
                    found_metrics = True
                if pd.notna(rmse):
                    print(f"- {str(target).replace('_', ' ').title()} Test RMSE: {rmse:.2f}")
                    found_metrics = True
        if not found_metrics:
            print("- Key performance metrics (e.g., Test MAE/RMSE) not found in model metadata.")

        # Display top features (if available in metadata for a primary model)
        primary_model_meta = models_metadata.get('home_score') 
        if isinstance(primary_model_meta, dict) and 'metrics' in primary_model_meta and \
           isinstance(primary_model_meta['metrics'], dict) and \
           'feature_importance' in primary_model_meta['metrics'] and \
           isinstance(primary_model_meta['metrics']['feature_importance'], dict):

            importance = primary_model_meta['metrics']['feature_importance']
            valid_importance = {k: v for k, v in importance.items() if pd.notna(v)}
            if valid_importance:
                 top_features = sorted(valid_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                 if top_features:
                      print("\nTOP 5 FEATURES (from Home Score Model):")
                      for feature, imp in top_features:
                         print(f"- {feature}: {imp:.4f}")
    else:
        print("- No valid model metadata provided.")

    # --- Backtest Summary ---
    print("\nBACKTEST SUMMARY:")
    if backtest_summary and isinstance(backtest_summary, dict):
         overall_mae = backtest_summary.get('overall_mae')
         overall_rmse = backtest_summary.get('overall_rmse')
         num_folds = backtest_summary.get('num_folds')
         if pd.notna(overall_mae):
              print(f"- Overall Backtest MAE: {overall_mae:.2f}" + (f" (over {num_folds} folds)" if num_folds else ""))
         if pd.notna(overall_rmse):
             print(f"- Overall Backtest RMSE: {overall_rmse:.2f}" + (f" (over {num_folds} folds)" if num_folds else ""))
         if not pd.notna(overall_mae) and not pd.notna(overall_rmse):
              print("- Key backtest metrics not found in summary.")
    else:
         print("- No backtest summary data provided.")


    # --- Upcoming Games Predictions ---
    print("\nUPCOMING GAMES PREDICTIONS:")
    if upcoming_predictions is not None and isinstance(upcoming_predictions, pd.DataFrame) and not upcoming_predictions.empty:
        required_cols = ['game_date', 'home_team', 'away_team', 'predicted_home_score', 'predicted_away_score', 'win_probability']
        if not all(col in upcoming_predictions.columns for col in required_cols):
             print(f"  Warning: Upcoming predictions DataFrame missing required columns: {required_cols}. Cannot display games.")
        else:
            try:
                upcoming_predictions['game_date_dt'] = pd.to_datetime(upcoming_predictions['game_date'], errors='coerce')
                upcoming_predictions = upcoming_predictions.dropna(subset=['game_date_dt']) # Remove games with invalid dates
                upcoming_predictions['game_date_str'] = upcoming_predictions['game_date_dt'].dt.strftime('%Y-%m-%d')

                today_dt = datetime.now()
                today_str = today_dt.strftime('%Y-%m-%d')
                tomorrow_str = (today_dt + timedelta(days=1)).strftime('%Y-%m-%d')

                # Sort predictions by date/time before selecting today/tomorrow
                upcoming_predictions = upcoming_predictions.sort_values(by='game_date_dt')

                today_games = upcoming_predictions[upcoming_predictions['game_date_str'] == today_str]
                tomorrow_games = upcoming_predictions[upcoming_predictions['game_date_str'] == tomorrow_str]

                if not today_games.empty:
                    print(f"\nTODAY'S GAMES ({today_str}):")
                    for idx, game in today_games.iterrows():
                        home_s = f"{game['predicted_home_score']:.1f}" if pd.notna(game['predicted_home_score']) else "N/A"
                        away_s = f"{game['predicted_away_score']:.1f}" if pd.notna(game['predicted_away_score']) else "N/A"
                        win_p = f"{game['win_probability']:.1%}" if pd.notna(game['win_probability']) else "N/A"
                        print(f"- {game['home_team']} vs {game['away_team']}: {home_s}-{away_s} (Win prob: {win_p})")

                        # Generate betting recommendations
                        if idx == today_games.index[0]:
                            try:
                                recs = generate_betting_recommendations(game.to_dict())
                                print(f"  > ML Rec: {recs.get('moneyline', {}).get('recommendation', 'N/A')}")
                                print(f"  > Spread Rec: {recs.get('spread', {}).get('recommendation', 'N/A')}")
                                print(f"  > Total Rec: {recs.get('total', {}).get('recommendation', 'N/A')}")
                            except Exception as bet_e:
                                logger.error(f"Error generating betting recommendations: {bet_e}", exc_info=True)
                                print(f"  > Betting Recs Error: {bet_e}")
                else:
                    print(f"- No games scheduled or predictions available for today ({today_str}).")

                if not tomorrow_games.empty:
                    print(f"\nTOMORROW'S GAMES ({tomorrow_str}):")
                    for _, game in tomorrow_games.iterrows():
                         home_s = f"{game['predicted_home_score']:.1f}" if pd.notna(game['predicted_home_score']) else "N/A"
                         away_s = f"{game['predicted_away_score']:.1f}" if pd.notna(game['predicted_away_score']) else "N/A"
                         win_p = f"{game['win_probability']:.1%}" if pd.notna(game['win_probability']) else "N/A"
                         print(f"- {game['home_team']} vs {game['away_team']}: {home_s}-{away_s} (Win prob: {win_p})")
                else:
                    print(f"- No games scheduled or predictions available for tomorrow ({tomorrow_str}).")

            except Exception as e:
                logger.error(f"Error processing upcoming predictions for dashboard: {e}", exc_info=True)
                print("- Error displaying upcoming game predictions.")
    else:
        print("- No upcoming game predictions data available or format is invalid.")

    # --- Footer ---
    print("\n" + "="*80)
    try:
        # Attempt to get timezone - may not work reliably on all systems/Python versions
        from time import tzname
        tz = tzname[0]
    except:
        tz = ""
    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + (f" {tz}" if tz else "")
    print(f"Dashboard generated at: {dt_now}")
    print("="*80 + "\n")


# --- Selected Visualization Functions for Dashboard ---

def plot_actual_vs_predicted(y_true: Union[pd.Series, np.ndarray],
                             y_pred: Union[pd.Series, np.ndarray],
                             title: str = "Actual vs. Predicted Scores",
                             metrics: Optional[Dict[str, float]] = None,
                             figsize: tuple = SMALL_FIG_SIZE,
                             save_path: Optional[Union[str, Path]] = None):
    """Generates a scatter plot of actual vs. predicted values (dashboard relevant)."""
    # (Implementation is the same as before, keeping it)
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[mask]; y_pred_clean = y_pred[mask]
        if len(y_true_clean) == 0:
            logger.warning("Invalid or empty input for actual vs predicted plot. Skipping.")
            return
        plt.figure(figsize=figsize)
        plt.scatter(y_true_clean, y_pred_clean, alpha=0.5, label="Predictions", s=30)
        min_val = min(np.min(y_true_clean), np.min(y_pred_clean))-5; max_val = max(np.max(y_true_clean), np.max(y_pred_clean))+5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
        plt.xlabel('Actual Values'); plt.ylabel('Predicted Values'); plt.title(title)
        plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        if metrics and 'r2' in metrics and 'rmse' in metrics and pd.notna(metrics['r2']) and pd.notna(metrics['rmse']):
            plt.text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}\nRMSE = {metrics["rmse"]:.2f}',
                     transform=plt.gca().transAxes, fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except Exception as e: logger.error(f"Error generating actual vs predicted plot: {e}", exc_info=True)


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                            metric_to_plot: str = 'rmse',
                            higher_is_better: bool = False,
                            title: Optional[str] = None,
                            figsize: tuple = SMALL_FIG_SIZE,
                            save_path: Optional[Union[str, Path]] = None):
    """Creates a bar chart comparing a specific metric across different models (dashboard relevant)."""
    # (Implementation is the same as before, keeping it)
    try:
        if not metrics_dict: logger.warning("No metrics data provided for comparison plot."); return
        valid_entries = [(name, metrics_dict[name].get(metric_to_plot, np.nan)) for name in metrics_dict if pd.notna(metrics_dict[name].get(metric_to_plot))]
        if not valid_entries: logger.warning(f"No valid data found for metric '{metric_to_plot}'."); return
        model_names, metric_values = zip(*valid_entries); metric_values = np.array(metric_values)
        if title is None: title = f'{metric_to_plot.upper()} Comparison ({ "Higher" if higher_is_better else "Lower"} is Better)'
        plt.figure(figsize=figsize); bars = plt.bar(model_names, metric_values)
        plt.ylabel(metric_to_plot.upper()); plt.title(title); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7)
        max_val = np.max(metric_values) if len(metric_values) > 0 else 0; min_val = np.min(metric_values) if len(metric_values) > 0 else 0
        val_range = max(abs(max_val), abs(min_val), 1e-6); offset = val_range * 0.02
        for bar in bars:
            yval = bar.get_height(); va = 'bottom' if yval >= 0 else 'top'; text_y = yval + offset if yval >= 0 else yval - offset
            plt.text(bar.get_x()+bar.get_width()/2, text_y, f'{yval:.3f}', ha='center', va=va, fontsize=9)
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except Exception as e: logger.error(f"Error generating metrics comparison plot: {e}", exc_info=True)


def plot_feature_importances(models_metadata: Dict[str, Dict[str, Any]], 
                             primary_model_key: str = 'home_score', 
                             top_n: int = 15,
                             figsize: tuple = (8, 7),
                             save_path: Optional[Union[str, Path]] = None):
    """
    Visualizes top feature importances extracted from model metadata (dashboard relevant).

    Args:
        models_metadata: Dictionary where keys are target names (e.g., 'home_score')
                         and values are dicts containing a 'metrics' dict, which in turn
                         contains a 'feature_importance' dict {feature_name: importance_value}.
        primary_model_key: The key in models_metadata to use for feature importance (e.g., 'home_score').
        top_n: Number of top features to display.
        figsize: Figure size.
        save_path: Optional path to save the figure.
    """
    logger.info(f"\n--- Generating Feature Importance Plot (from {primary_model_key} metadata) ---")
    try:
        model_meta = models_metadata.get(primary_model_key)
        if not isinstance(model_meta, dict) or 'metrics' not in model_meta or \
           not isinstance(model_meta['metrics'], dict) or \
           'feature_importance' not in model_meta['metrics'] or \
           not isinstance(model_meta['metrics']['feature_importance'], dict):
            logger.warning(f"Feature importance data not found or invalid format in metadata for key '{primary_model_key}'. Skipping plot.")
            return

        imp_dict = model_meta['metrics']['feature_importance']
        valid_imp = {k: v for k, v in imp_dict.items() if pd.notna(v)}

        if not valid_imp:
            logger.warning(f"No valid feature importance values found for key '{primary_model_key}'. Skipping plot.")
            return

        imp_df = pd.DataFrame({'Feature': list(valid_imp.keys()), 'Importance': list(valid_imp.values())})
        total_imp = imp_df['Importance'].sum()

        if total_imp > 1e-9:
            imp_df['Normalized Importance'] = imp_df['Importance'] / total_imp
            sort_col = 'Normalized Importance'
        else:
            logger.warning(f"Total importance for {primary_model_key} is near zero. Using raw values.")
            imp_df['Normalized Importance'] = imp_df['Importance'] # Still add column for consistency
            sort_col = 'Importance'

        imp_df = imp_df.sort_values(sort_col, ascending=False).head(top_n)

        if imp_df.empty:
             logger.warning(f"No features left after sorting/filtering for {primary_model_key}. Skipping plot.")
             return

        plt.figure(figsize=figsize)
        sns.barplot(x=sort_col, y='Feature', data=imp_df, palette='viridis')
        plt.title(f'Top {top_n} Features - {primary_model_key.replace("_", " ").title()}')
        plt.xlabel(sort_col.replace("_", " "))
        plt.ylabel("Feature")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Error generating feature importance plot from metadata: {e}", exc_info=True)


def plot_predictions_over_time(dates: Union[List, pd.Series, np.ndarray],
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "Predictions Over Time",
                               target_name: str = "Score",
                               figsize: tuple = (14, 7),
                               save_dir: Optional[Union[str, Path]] = None):
    """Plots actual vs predicted values over time (dashboard relevant)."""
    try:
        dates = np.asarray(dates)
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not np.any(mask): logger.warning("No valid data for time plot. Skipping."); return
        dates_clean = dates[mask]; y_true_clean = y_true[mask]; y_pred_clean = y_pred[mask]
        try: dates_dt = pd.to_datetime(dates_clean); is_datetime_type = True
        except Exception: logger.warning("Could not convert 'dates' to datetime. Plotting against original."); dates_dt = dates_clean; is_datetime_type = False
        df = pd.DataFrame({'Date': dates_dt, 'Actual': y_true_clean, 'Predicted': y_pred_clean})
        try: df = df.sort_values('Date')
        except TypeError: logger.warning("Could not sort by 'Date'. Plotting in original index order."); df = df.reset_index(drop=True); df['Date'] = df.index
        plt.figure(figsize=figsize); x_plot = df['Date']
        plt.plot(x_plot, df['Actual'], 'o-', label='Actual', alpha=0.8, markersize=4, linewidth=1.5)
        plt.plot(x_plot, df['Predicted'], 'o--', label='Predicted', alpha=0.8, markersize=4, linewidth=1.5)
        plt.title(title, fontsize=14); plt.ylabel(target_name, fontsize=12); plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.6)
        if is_datetime_type:
            plt.xlabel('Date', fontsize=12)
            try: plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d')); plt.gcf().autofmt_xdate(rotation=30, ha='right')
            except Exception as fmt_e: logger.warning(f"Date formatting failed: {fmt_e}"); plt.xticks(rotation=30, ha='right')
        else:
            plt.xlabel('Game Sequence / Identifier', fontsize=12)
            if len(df) > 20: step = max(1, len(df)//10); plt.xticks(ticks=x_plot[::step], labels=df['Date'].iloc[::step], rotation=30, ha='right')
            else: plt.xticks(ticks=x_plot, labels=df['Date'], rotation=30, ha='right')
        plt.tight_layout()
        if save_dir:
            save_dir_path = Path(save_dir); save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / "predictions_over_time.png" # Keep consistent filename
            try: plt.savefig(f_path, bbox_inches='tight'); logger.info(f"Plot saved to {f_path}")
            except Exception as e: logger.error(f"Error saving plot {f_path}: {e}", exc_info=True)
        plt.show()
    except Exception as e: logger.error(f"Error generating predictions over time plot: {e}", exc_info=True)


# --- Main Example Block (Simplified for Dashboard) ---
if __name__ == '__main__':
    logger.info("Running dashboard.py example...")
    main_save_dir = DASHBOARD_PLOT_DIR / "main_dashboard_example"

    # --- Simulate data needed for dashboard ---
    np.random.seed(42)
    n_samples = 50 

    # 1. Model Metadata 
    mock_features = [f'feature_{i}' for i in range(10)] + ['rolling_avg_pts', 'pace_last_5', 'elo_diff', 'rest_days', 'home_3p_pct']
    mock_features = mock_features[:12]
    mock_model_metadata = {
        'home_score': {'metrics': {'test_mae': 10.5, 'test_rmse': 13.2, 'feature_importance': {k: np.random.rand() for k in mock_features}}},
        'away_score': {'metrics': {'test_mae': 11.2, 'test_rmse': 14.1}},
        'ensemble': {'metrics': {'test_mae': 9.8, 'test_rmse': 12.5}} # Add ensemble metrics
    }

    # 2. Upcoming Predictions Data
    upcoming_games_df = pd.DataFrame({
        'game_date': [datetime.now().strftime('%Y-%m-%d')] * 2 + [(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')] * 1,
        'home_team': ['Lakers', 'Warriors', 'Nets'],
        'away_team': ['Clippers', 'Suns', 'Knicks'],
        'predicted_home_score': [115.5, 121.0, 110.1],
        'predicted_away_score': [112.1, 118.5, 108.9],
        'win_probability': [0.65, 0.58, 0.53]
    })

    # 3. Backtest Summary Data
    mock_backtest_summary = {'overall_mae': 10.8, 'overall_rmse': 13.9, 'num_folds': 5}


    # --- Generate Console Dashboard ---
    logger.info("\n--- Generating Console Dashboard ---")
    generate_pregame_dashboard(
        upcoming_predictions=upcoming_games_df,
        models_metadata=mock_model_metadata,
        backtest_summary=mock_backtest_summary
    )

    # --- Generate Example Plots Relevant to Dashboard ---
    logger.info("\n--- Generating Example Dashboard Plots ---")

    # a) Feature Importance Plot
    plot_feature_importances(
        models_metadata=mock_model_metadata,
        primary_model_key='home_score', 
        top_n=10,
        save_path=main_save_dir / "top_features.png"
    )

    # b) Metrics Comparison Plot (using metadata)
    metrics_for_comparison = {
        name: meta.get('metrics', {}) for name, meta in mock_model_metadata.items()
    }
    # Filter out entries without the metric we want to plot 
    metrics_for_comparison_filtered = {
        name: metrics for name, metrics in metrics_for_comparison.items() if pd.notna(metrics.get('test_mae'))
    }
    if metrics_for_comparison_filtered:
         plot_metrics_comparison(
             metrics_dict=metrics_for_comparison_filtered,
             metric_to_plot='test_mae',
             higher_is_better=False,
             title="Model Test MAE Comparison",
             save_path=main_save_dir / "mae_comparison.png"
         )

    # c) Example Actual vs Predicted (if recent results data is available)
    y_true_recent = np.random.normal(loc=225, scale=22, size=n_samples).round()
    y_pred_recent_ensemble = y_true_recent + np.random.normal(loc=-2.5, scale=11, size=n_samples)
    metrics_recent = calculate_regression_metrics(y_true_recent, y_pred_recent_ensemble)

    plot_actual_vs_predicted(
        y_true=y_true_recent,
        y_pred=y_pred_recent_ensemble,
        title="Recent Performance: Actual vs Predicted (Ensemble)",
        metrics=metrics_recent,
        save_path=main_save_dir / "recent_actual_vs_pred.png"
    )


    logger.info("\nDashboard script example finished.")