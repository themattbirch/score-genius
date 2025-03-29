# backend/score_prediction/evaluation.py
"""
Module for evaluating model predictions and visualizing results.

Provides functions for calculating standard regression metrics, custom NBA-specific losses,
and generating various plots for residual analysis, bias analysis, feature importance,
and model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, List, Union

# Optional import for LOWESS smoother - plotting will work without it
try:
    import statsmodels.api as sm
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
    print("Note: 'statsmodels' not found. LOWESS smoother will not be available in residual plots.")

# --- Plotting Configuration ---
plt.style.use('fivethirtyeight')
sns.set_palette('colorblind')
DEFAULT_FIG_SIZE = (12, 8)
SMALL_FIG_SIZE = (10, 6)
LARGE_FIG_SIZE = (15, 10)

# --- Core Metric Calculation Functions ---

def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]
                                 ) -> Dict[str, float]:
    """
    Calculates standard regression metrics.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.

    Returns:
        Dictionary containing MSE, RMSE, MAE, and R-squared.
        Returns NaNs if input arrays are empty or lengths mismatch.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            print(f"Error: Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true) == 0:
            print("Warning: Input arrays are empty.")
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        # R2 score requires at least two samples
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}


def calculate_nba_score_loss(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray]
                               ) -> float:
    """
    Calculates a basic mean squared error loss for NBA scores.
    (Note: Based on provided code, this is currently just MSE).

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.

    Returns:
        The mean squared error, or NaN if calculation fails.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            print(f"Error: Input arrays must have the same length for NBA score loss. Got {len(y_true)} and {len(y_pred)}")
            return np.nan
        if len(y_true) == 0:
            print("Warning: Input arrays are empty for NBA score loss.")
            return np.nan

        return mean_squared_error(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating NBA score loss: {e}")
        return np.nan


def calculate_nba_distribution_loss(y_pred: Union[pd.Series, np.ndarray],
                                      target_type: str = 'total'
                                      ) -> float:
    """
    Calculates a loss based on deviation from expected NBA score distributions.
    Penalizes predictions far from typical score ranges.

    Args:
        y_pred: Array or Series of predicted values.
        target_type: Type of score ('home', 'away', 'total', 'diff') to determine
                     expected distribution.

    Returns:
        The mean squared Z-score relative to the expected distribution, or NaN if calculation fails.
    """
    try:
        y_pred = np.asarray(y_pred).flatten()
        if len(y_pred) == 0:
            print("Warning: Input array is empty for NBA distribution loss.")
            return np.nan

        # Expected distribution parameters (refined based on recent seasons, may need tuning)
        if target_type == 'home':
            expected_mean, expected_std = 114, 13.5
        elif target_type == 'away':
            expected_mean, expected_std = 112, 13.5
        elif target_type == 'total':
            expected_mean, expected_std = 226, 23
        elif target_type == 'diff':
            # Home team point differential mean tends to be slightly positive
            expected_mean, expected_std = 2.5, 13.5
        else:
            # Default fallback (e.g., for generic score)
            expected_mean, expected_std = 112, 14

        if expected_std <= 0:
            raise ValueError("Expected standard deviation must be positive.")

        # Calculate squared Z-scores relative to the expected distribution
        z_score_squared = ((y_pred - expected_mean) / expected_std) ** 2
        return np.mean(z_score_squared)

    except Exception as e:
        print(f"Error calculating NBA distribution loss: {e}")
        return np.nan


def evaluate_predictions(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray],
                         target_type: str = 'total',
                         calculate_custom_losses: bool = True
                         ) -> Dict[str, float]:
    """
    Calculates both standard and custom evaluation metrics for predictions.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        target_type: Type of score ('home', 'away', 'total', 'diff') for distribution loss.
        calculate_custom_losses: Flag to include custom NBA loss calculations.

    Returns:
        Dictionary containing calculated metrics. Includes NaNs for failed calculations.
    """
    metrics = calculate_regression_metrics(y_true, y_pred)

    if calculate_custom_losses:
        metrics['nba_score_loss'] = calculate_nba_score_loss(y_true, y_pred)
        metrics['nba_distribution_loss'] = calculate_nba_distribution_loss(y_pred, target_type)

    return metrics

# --- Core Visualization Functions ---

def plot_actual_vs_predicted(y_true: Union[pd.Series, np.ndarray],
                             y_pred: Union[pd.Series, np.ndarray],
                             title: str = "Actual vs. Predicted Scores",
                             metrics: Optional[Dict[str, float]] = None,
                             figsize: tuple = SMALL_FIG_SIZE,
                             save_path: Optional[Union[str, Path]] = None):
    """
    Generates a scatter plot of actual vs. predicted values.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        title: Title for the plot.
        metrics: Optional dict with 'r2' and 'rmse' to display on plot.
        figsize: Figure size.
        save_path: Optional path (string or Path object) to save the figure.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            print("Warning: Invalid input arrays for actual vs predicted plot. Skipping.")
            return

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
        min_val = min(np.min(y_true), np.min(y_pred)) - 5
        max_val = max(np.max(y_true), np.max(y_pred)) + 5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Add R² and RMSE text if provided and valid
        if metrics and 'r2' in metrics and 'rmse' in metrics:
            r2 = metrics['r2']
            rmse = metrics['rmse']
            if not (np.isnan(r2) or np.isnan(rmse)):
                plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}',
                         transform=plt.gca().transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error generating actual vs predicted plot: {e}")


def plot_residuals_distribution(y_true: Union[pd.Series, np.ndarray],
                                y_pred: Union[pd.Series, np.ndarray],
                                title: str = "Residuals Analysis",
                                figsize: tuple = (15, 6),
                                save_path_prefix: Optional[Union[str, Path]] = None):
    """
    Generates plots for residual analysis: histogram and residuals vs. predicted.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        title: Title prefix for the plots.
        figsize: Figure size for the combined plot.
        save_path_prefix: Optional path prefix (string or Path object) to save the figure.
                          Appends '_distribution.png'.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            print("Warning: Invalid input arrays for residuals distribution plot. Skipping.")
            return

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        # 1. Residuals Histogram
        sns.histplot(residuals, kde=True, ax=axes[0])
        axes[0].axvline(x=0, color='r', linestyle='--')
        axes[0].set_xlabel('Residual (Actual - Predicted)')
        axes[0].set_title(f'Residuals Distribution (Mean: {np.mean(residuals):.2f})')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 2. Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs. Predicted Values')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        if save_path_prefix:
            save_path = Path(f"{save_path_prefix}_distribution.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error generating residuals distribution plot: {e}")


def plot_error_by_prediction_range(y_true: Union[pd.Series, np.ndarray],
                                   y_pred: Union[pd.Series, np.ndarray],
                                   num_bins: int = 10,
                                   title: str = "Mean Prediction Error by Score Range",
                                   figsize: tuple = SMALL_FIG_SIZE,
                                   save_path: Optional[Union[str, Path]] = None):
    """
    Visualizes the mean residual (error) across different bins of predicted values.
    Uses Standard Error of the Mean (SEM) for error bars.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        num_bins: Number of bins to divide the prediction range into.
        title: Title for the plot.
        figsize: Figure size.
        save_path: Optional path (string or Path object) to save the figure.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            print("Warning: Invalid input arrays for error by range plot. Skipping.")
            return

        residuals = y_true - y_pred

        # Create DataFrame for easier binning and aggregation
        df = pd.DataFrame({'prediction': y_pred, 'residual': residuals})

        # Handle edge case with very few data points where binning might fail
        if len(df) < num_bins * 2:
            print(f"Warning: Not enough data points ({len(df)}) for {num_bins} bins. Skipping error range plot.")
            return

        # Use quantiles for potentially more equal bin sizes
        df['bin_q'] = pd.qcut(df['prediction'], q=num_bins, labels=False, duplicates='drop')
        n_actual_bins = df['bin_q'].nunique()
        if n_actual_bins < 2:
             print(f"Warning: Could only create {n_actual_bins} bin(s) using quantiles. Skipping error range plot.")
             return

        # Calculate mean residual, standard error of the mean (SEM), and bin center
        binned_stats = df.groupby('bin_q').agg(
            mean_residual=('residual', 'mean'),
            sem_residual=('residual', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
            bin_center=('prediction', 'mean'),
            count=('residual', 'size')
        ).reset_index()

        # Filter out bins with unreliable SEM
        binned_stats = binned_stats[binned_stats['count'] > 1]

        if binned_stats.empty:
            print("Warning: No bins with sufficient data found for error range plot.")
            return

        # Sort by bin center for plotting
        binned_stats = binned_stats.sort_values('bin_center')

        plt.figure(figsize=figsize)
        plt.errorbar(binned_stats['bin_center'], binned_stats['mean_residual'],
                     yerr=binned_stats['sem_residual'], fmt='o-', capsize=5, label='Mean Residual ± SEM')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Average Predicted Score in Bin')
        plt.ylabel('Mean Residual (Actual - Predicted)')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    except ValueError as e:
        print(f"Warning: Could not create bins for error range plot. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during error range plot generation: {e}")


def plot_score_distribution_density(y_true: Union[pd.Series, np.ndarray],
                                     y_pred: Union[pd.Series, np.ndarray],
                                     title: str = "Score Distribution Density",
                                     figsize: tuple = SMALL_FIG_SIZE,
                                     save_path: Optional[Union[str, Path]] = None):
    """
    Visualizes the density distribution of actual vs. predicted scores using KDE plots.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        title: Title for the plot.
        figsize: Figure size.
        save_path: Optional path (string or Path object) to save the figure.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            print("Warning: Invalid input arrays for score density plot. Skipping.")
            return

        plt.figure(figsize=figsize)
        sns.kdeplot(y_true, label='Actual Distribution', color='blue', fill=True, alpha=0.3, bw_adjust=0.5)
        sns.kdeplot(y_pred, label='Predicted Distribution', color='red', fill=True, alpha=0.3, bw_adjust=0.5)
        plt.title(title)
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error generating score density plot: {e}")


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                            metric_to_plot: str = 'rmse',
                            higher_is_better: bool = False,
                            title: Optional[str] = None,
                            figsize: tuple = SMALL_FIG_SIZE,
                            save_path: Optional[Union[str, Path]] = None):
    """
    Creates a bar chart comparing a specific metric across different models/evaluations.

    Args:
        metrics_dict: Dictionary where keys are model/evaluation names and
                      values are dictionaries of metrics (e.g., {'model_a': {'rmse': 10, 'r2': 0.8}}).
        metric_to_plot: The key of the metric to visualize (e.g., 'rmse', 'r2', 'test_mae').
        higher_is_better: Set to True if a higher value of the metric is better (e.g., for R²).
        title: Optional title for the plot. If None, a default title is generated.
        figsize: Figure size.
        save_path: Optional path (string or Path object) to save the figure.
    """
    try:
        if not metrics_dict:
            print("Warning: No metrics data provided for comparison plot.")
            return

        model_names = list(metrics_dict.keys())
        metric_values = [metrics_dict[name].get(metric_to_plot, np.nan) for name in model_names]

        # Filter out entries where the metric is missing or NaN
        valid_entries = [(name, val) for name, val in zip(model_names, metric_values) if pd.notna(val)]
        if not valid_entries:
            print(f"Warning: No valid data found for metric '{metric_to_plot}' in the provided dictionary.")
            return

        model_names, metric_values = zip(*valid_entries)
        metric_values = np.array(metric_values) # For easier calculations

        if title is None:
            title = f'{metric_to_plot.upper()} Comparison ({ "Higher" if higher_is_better else "Lower"} is Better)'

        plt.figure(figsize=figsize)
        bars = plt.bar(model_names, metric_values)
        plt.ylabel(metric_to_plot.upper())
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add values on top of bars
        max_val = np.max(metric_values) if len(metric_values) > 0 else 0
        min_val = np.min(metric_values) if len(metric_values) > 0 else 0
        val_range = max(abs(max_val), abs(min_val))
        offset = val_range * 0.02 if val_range > 0 else 0.1 # Default offset if range is 0

        for bar in bars:
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            text_y = yval + offset if yval >= 0 else yval - offset
            plt.text(bar.get_x() + bar.get_width()/2, text_y,
                     f'{yval:.3f}', ha='center', va=va, fontsize=9)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Error generating metrics comparison plot: {e}")


def plot_ensemble_component_comparison(component_metrics: Dict[str, Dict[str, float]],
                                       ensemble_metrics: Optional[Dict[str, float]] = None,
                                       target_name: str = "Score",
                                       figsize: tuple = (14, 10),
                                       save_path_prefix: Optional[Union[str, Path]] = None):
    """
    Visualizes RMSE and R² comparison between ensemble components and the final ensemble.

    Args:
        component_metrics: Dict where keys are component names and values are metrics dicts
                           (must include 'rmse', 'r2', and optionally 'weight').
        ensemble_metrics: Optional metrics dict for the final ensemble (must include 'rmse', 'r2').
        target_name: Name of the target variable for titles (e.g., "Total Score").
        figsize: Figure size for the combined plot.
        save_path_prefix: Optional path prefix (string or Path object) to save the figures.
                          Appends '_component_comparison.png'.
    """
    try:
        metrics_to_plot = {}
        metrics_to_plot.update(component_metrics)
        if ensemble_metrics:
            metrics_to_plot['Ensemble'] = ensemble_metrics

        if not metrics_to_plot:
            print("Warning: No component or ensemble metrics provided for comparison.")
            return

        # --- Prepare data for plotting ---
        df_data = []
        for name, metrics in metrics_to_plot.items():
            # Ensure required metrics exist and are valid numbers
            if ('rmse' in metrics and pd.notna(metrics['rmse']) and
                'r2' in metrics and pd.notna(metrics['r2'])):
                 df_data.append({
                    'Model': name.replace('_', ' ').title(), # Prettify name
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2'],
                    'Weight': metrics.get('weight', np.nan) # Get weight if available
                })
            else:
                print(f"Warning: Skipping model '{name}' due to missing or invalid 'rmse' or 'r2' metric.")

        if not df_data:
            print("Warning: No valid data entries after checking for required metrics.")
            return

        comp_df = pd.DataFrame(df_data)
        comp_df = comp_df.sort_values(by='RMSE') # Sort by RMSE for clearer comparison

        # --- Create Visualization ---
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True) # Share x-axis
        fig.suptitle(f'Ensemble vs. Component Performance for {target_name}', fontsize=16)

        # Plot colors - highlight Ensemble if present
        colors = ['#d62728' if model == 'Ensemble' else '#1f77b4' for model in comp_df['Model']]

        # 1. RMSE comparison
        sns.barplot(x='Model', y='RMSE', data=comp_df, ax=axes[0], palette=colors, dodge=False)
        axes[0].set_title('RMSE Comparison (Lower is Better)')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
        # axes[0].tick_params(axis='x', rotation=15) # Rotation handled by sharex

        # Add RMSE values and weights (if available)
        rmse_max = comp_df['RMSE'].max()
        for i, (idx, row) in enumerate(comp_df.iterrows()):
            rmse_val = row['RMSE']
            weight_val = row['Weight']
            axes[0].text(i, rmse_val + rmse_max * 0.01, f'{rmse_val:.2f}', ha='center', va='bottom', fontsize=9)
            if pd.notna(weight_val) and row['Model'] != 'Ensemble':
                axes[0].text(i, rmse_val + rmse_max * 0.05, f'W: {weight_val:.2f}', ha='center', va='bottom', fontsize=8, color='gray')


        # 2. R² comparison
        sns.barplot(x='Model', y='R2', data=comp_df, ax=axes[1], palette=colors, dodge=False)
        axes[1].set_title('R² Comparison (Higher is Better)')
        axes[1].set_ylabel('R² Score')
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=30, ha='right') # Rotate labels on bottom plot
        # Adjust y-lims for better visibility, ensuring 0 is included if R2 can be negative
        r2_min = comp_df['R2'].min()
        r2_max = comp_df['R2'].max()
        axes[1].set_ylim(bottom=min(0, r2_min - 0.05), top=min(1.05, r2_max + 0.05))

        # Add R² values
        for i, (idx, row) in enumerate(comp_df.iterrows()):
            r2_val = row['R2']
            axes[1].text(i, r2_val + 0.005, f'{r2_val:.4f}', ha='center', va='bottom', fontsize=9)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        if save_path_prefix:
            save_path = Path(f"{save_path_prefix}_component_comparison.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

        print("\nPerformance Summary:")
        print(comp_df[['Model', 'RMSE', 'R2', 'Weight']].to_string(index=False, float_format='%.4f'))

    except Exception as e:
        print(f"Error generating ensemble component comparison plot: {e}")


# --- Feature Importance Visualization ---

def _get_feature_importance(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """Helper to extract feature importance from various model types."""
    # Check for feature_importances_ (common in tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(feature_names) == len(importances):
            return dict(zip(feature_names, importances))
        else:
            print(f"Warning: Feature name count ({len(feature_names)}) doesn't match model importance count ({len(importances)}) for {type(model).__name__}. Skipping.")
            return None
    # Check for coef_ (common in linear models)
    elif hasattr(model, 'coef_'):
        # Coef_ can be multi-dimensional for multi-output models, handle simple case
        if model.coef_.ndim == 1 or model.coef_.shape[0] == 1:
            coefs = model.coef_.flatten()
            if len(feature_names) == len(coefs):
                # Use absolute coefficient value as importance measure
                return dict(zip(feature_names, np.abs(coefs)))
            else:
                print(f"Warning: Feature name count ({len(feature_names)}) doesn't match model coefficient count ({len(coefs)}) for {type(model).__name__}. Skipping.")
                return None
        else:
            print(f"Warning: Model {type(model).__name__} has multi-output coefficients. Importance extraction not implemented for this case.")
            return None
    # Check if it's a pipeline and try the final step
    elif hasattr(model, 'named_steps') and hasattr(model, 'steps') and model.steps:
        final_step_name = model.steps[-1][0]
        final_step_model = model.named_steps[final_step_name]
        print(f"Attempting to extract importance from final pipeline step: '{final_step_name}' ({type(final_step_model).__name__})")
        return _get_feature_importance(final_step_model, feature_names)

    print(f"Warning: Feature importance not directly available or extraction not implemented for model type {type(model).__name__}.")
    return None


def plot_feature_importances(models_dict: Dict[str, Any],
                             feature_names: List[str],
                             ensemble_weights: Optional[Dict[str, float]] = None,
                             top_n: int = 20,
                             plot_groups: bool = True,
                             feature_group_config: Optional[Dict[str, List[str]]] = None,
                             figsize_individual: tuple = (10, 8),
                             figsize_group: tuple = (10, 10),
                             save_dir: Optional[Union[str, Path]] = None):
    """
    Visualizes feature importance for individual models and a weighted ensemble average.

    Args:
        models_dict: Dictionary where keys are model names and values are fitted model objects.
                     Models should have 'feature_importances_' or 'coef_' attributes, or be pipelines
                     where the final step has one of these.
        feature_names: List of feature names corresponding to the model inputs, in the correct order.
        ensemble_weights: Optional dictionary mapping model names (must match keys in models_dict)
                          to their weights in an ensemble. Required for weighted average plot.
        top_n: Number of top features to display for each model/ensemble.
        plot_groups: Whether to plot feature importance grouped by category (pie chart for ensemble).
        feature_group_config: Optional dictionary defining feature groups. Keys are group
                              names, values are lists of keywords found in feature names.
                              If None, uses default NBA-related groups.
        figsize_individual: Figure size for individual importance plots (ignored, size adjusts).
        figsize_group: Figure size for group importance plot.
        save_dir: Optional directory (string or Path object) to save plots.
    """
    print("\n--- Generating Feature Importance Report ---")
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {save_dir_path}")

    importances = {}
    valid_model_names_for_ensemble = []

    # --- Get Individual Importances ---
    print("Extracting feature importance from models...")
    for name, model in models_dict.items():
        imp = _get_feature_importance(model, feature_names)
        if imp:
            importances[name] = imp
            # Only consider for ensemble average if weights are provided and positive
            if ensemble_weights and name in ensemble_weights and ensemble_weights[name] > 0:
                 valid_model_names_for_ensemble.append(name)
        else:
            print(f"-> Could not extract importance for model: {name} ({type(model).__name__})")

    if not importances:
        print("Error: No feature importance data could be extracted from any model.")
        return

    # --- Calculate Weighted Ensemble Importance ---
    if ensemble_weights and valid_model_names_for_ensemble:
        print(f"Calculating weighted ensemble importance using models: {', '.join(valid_model_names_for_ensemble)}")
        weighted_imp = {}

        relevant_weights = {name: ensemble_weights[name] for name in valid_model_names_for_ensemble}
        norm_factor = sum(relevant_weights.values())

        if norm_factor > 0:
            normalized_weights = {name: weight / norm_factor for name, weight in relevant_weights.items()}
            print(f"  Normalized weights used: { {k: f'{v:.2f}' for k, v in normalized_weights.items()} }")

            all_relevant_features = set(f for name in valid_model_names_for_ensemble for f in importances[name])

            for feature in all_relevant_features:
                feature_score = 0.0
                for name in valid_model_names_for_ensemble:
                    imp_value = importances[name].get(feature, 0.0)
                    feature_score += normalized_weights[name] * imp_value

                if feature_score > 1e-9:
                    weighted_imp[feature] = feature_score

            if weighted_imp:
                importances["Ensemble (Weighted)"] = weighted_imp
                print("  Successfully calculated weighted importance.")
            else:
                 print("  Warning: Weighted importance calculation resulted in zero values.")
        else:
             print("  Warning: Sum of weights for models with importance is zero.")
    elif ensemble_weights:
         print("Warning: Ensemble weights provided, but no importance could be extracted from weighted models.")


    # --- Plot Individual Model Importances ---
    num_plots = len(importances)
    if num_plots == 0:
         print("No importance data to plot.")
         return

    n_cols = min(3, num_plots)
    n_rows = (num_plots + n_cols - 1) // n_cols
    # Adjust figsize dynamically based on number of plots and top_n features
    fig_height = max(5, n_rows * (top_n * 0.3 + 1)) # Estimate height needed
    fig_width = n_cols * 6.5
    fig_ind, axes_ind = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_ind = axes_ind.flatten()

    plot_idx = 0
    print("Generating individual importance plots...")
    for name, imp_dict in importances.items():
        if not imp_dict: continue

        imp_df = pd.DataFrame({'Feature': list(imp_dict.keys()), 'Importance': list(imp_dict.values())})

        total_imp = imp_df['Importance'].sum()
        if total_imp > 1e-9:
            imp_df['Normalized Importance'] = imp_df['Importance'] / total_imp
        else:
            print(f"  Warning: Total importance for {name} is near zero. Using raw values.")
            imp_df['Normalized Importance'] = imp_df['Importance']

        imp_df = imp_df.sort_values('Normalized Importance', ascending=False).head(top_n)

        if not imp_df.empty:
            ax = axes_ind[plot_idx]
            sns.barplot(x='Normalized Importance', y='Feature', data=imp_df, ax=ax, palette='viridis')
            ax.set_title(f'Top {top_n} Features - {name}')
            ax.set_xlabel("Normalized Importance")
            ax.tick_params(axis='y', labelsize=9)
            plot_idx += 1
        else:
             print(f"  No importance values to plot for {name}.")

    for i in range(plot_idx, len(axes_ind)):
        fig_ind.delaxes(axes_ind[i])

    if plot_idx > 0:
        fig_ind.suptitle('Feature Importance by Model', fontsize=16, y=1.01) # Adjust y slightly
        fig_ind.tight_layout(rect=[0, 0.03, 1, 0.98])
        if save_dir_path:
            f_path = save_dir_path / "feature_importance_individual.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                print(f"Plot saved to {f_path}")
            except Exception as e:
                print(f"Error saving plot {f_path}: {e}")
        plt.show()
    else:
        plt.close(fig_ind)


    # --- Plot Feature Group Importance (Pie Chart) for Ensemble ---
    if plot_groups and "Ensemble (Weighted)" in importances:
        ensemble_imp = importances["Ensemble (Weighted)"]
        if not ensemble_imp:
            print("Skipping group importance plot as weighted ensemble importance is empty.")
        else:
            print("Generating feature group importance plot (Ensemble Weighted)...")
            if feature_group_config is None:
                feature_group_config = {
                    'Recent Form': ['form', 'rolling', 'streak', 'last_'],
                    'Pace & Efficiency': ['pace', 'poss', 'eff', 'rating', 'ortg', 'drtg', 'netrtg'],
                    'Shooting': ['fgm', 'fga', 'pct', '3p', 'ftm', 'fta', 'efg', 'ts_'],
                    'Rebounding': ['oreb', 'dreb', 'reb', '_reb_rate', '_reb_pct'],
                    'Playmaking & Defense': ['ast', 'stl', 'blk', 'tov', 'pf', 'assist_ratio', 'turnover_ratio', 'defensive'],
                    'Team Performance': ['win', 'loss', 'elo', '_rank', 'seed', 'record', 'ppg', 'papg'],
                    'Matchup': ['matchup', 'h2h', 'vs_', '_adv', '_diff', '_comp'],
                    'Rest & Schedule': ['rest', 'b2b', 'back_to_back', 'days_since', 'travel'],
                    'Game State': ['time_rem', 'quarter', 'score_lead', 'in_game', 'clutch'],
                }
                print("  Using default NBA feature group definitions.")

            group_totals = {group: 0.0 for group in feature_group_config}
            group_totals['Other'] = 0.0
            assigned_features = set()

            for feature, importance in ensemble_imp.items():
                feature_lower = feature.lower()
                assigned = False
                for group, keywords in feature_group_config.items():
                    if any(keyword in feature_lower for keyword in keywords):
                        group_totals[group] += importance
                        assigned_features.add(feature)
                        assigned = True
                        break
                if not assigned:
                    group_totals['Other'] += importance
                    assigned_features.add(feature)

            group_totals_filtered = {k: v for k, v in group_totals.items() if v > 1e-6}

            if group_totals_filtered:
                labels = list(group_totals_filtered.keys())
                sizes = list(group_totals_filtered.values())
                explode = [0.1 if label == 'Other' and sizes[i]/sum(sizes) > 0.01 else 0.01 for i, label in enumerate(labels)]

                plt.figure(figsize=figsize_group)
                wedges, texts, autotexts = plt.pie(
                    sizes, labels=labels, autopct='%1.1f%%', startangle=110,
                    pctdistance=0.80, explode=explode, shadow=False,
                    textprops={'fontsize': 9}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')

                plt.title('Feature Importance by Group (Ensemble Weighted Average)', fontsize=14)
                plt.axis('equal')
                plt.tight_layout()
                if save_dir_path:
                    f_path = save_dir_path / "feature_importance_groups.png"
                    try:
                        plt.savefig(f_path, bbox_inches='tight')
                        print(f"Plot saved to {f_path}")
                    except Exception as e:
                        print(f"Error saving plot {f_path}: {e}")
                plt.show()
            else:
                print("Could not plot feature group importance (total importance is zero or near-zero).")

    elif plot_groups:
         print("Skipping group importance plot as weighted ensemble importance was not calculated.")

    print("--- Feature Importance Report Complete ---")


# --- Prediction Range / Model Agreement Visualization ---

def plot_model_agreement(predictions_dict: Dict[str, np.ndarray],
                         game_identifiers: Union[List, pd.Series, np.ndarray],
                         y_true: Optional[Union[pd.Series, np.ndarray]] = None,
                         ensemble_pred: Optional[np.ndarray] = None,
                         ensemble_weights: Optional[Dict[str, float]] = None,
                         target_name: str = "Score",
                         num_games_to_plot: int = 20,
                         figsize: tuple = (15, 8),
                         save_dir: Optional[Union[str, Path]] = None):
    """
    Visualizes predictions from multiple models for recent games, showing model agreement
    based on the standard deviation of component predictions.

    Args:
        predictions_dict: Dictionary where keys are model names and values are prediction arrays
                          (all arrays must be for the same games in the same order).
        game_identifiers: List or Series or array identifying the games (e.g., "TeamA vs TeamB Date").
                          Must be same length as prediction arrays.
        y_true: Optional array/Series of actual true values for the games.
        ensemble_pred: Optional pre-calculated ensemble prediction array.
        ensemble_weights: Optional weights dict, used to calculate ensemble if not provided.
                          Weights should correspond to keys in predictions_dict.
        target_name: Name of the target variable for the y-axis label.
        num_games_to_plot: Number of most recent games (samples) to plot. If <=0 or None, plots all.
        figsize: Figure size.
        save_dir: Optional directory (string or Path object) to save plots.
    """
    print("\n--- Generating Model Agreement Plot ---")
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    if not predictions_dict:
        print("Error: No component predictions provided in predictions_dict.")
        return

    # Validate input lengths
    n_preds = -1
    model_names = list(predictions_dict.keys())
    if len(model_names) < 2:
        print("Warning: Model agreement plot requires predictions from at least 2 models.")

    for name, preds in predictions_dict.items():
        preds_arr = np.asarray(preds)
        predictions_dict[name] = preds_arr # Ensure numpy arrays
        if n_preds == -1:
            n_preds = len(preds_arr)
        elif len(preds_arr) != n_preds:
            print(f"Error: Prediction array length mismatch for '{name}'. Expected {n_preds}, got {len(preds_arr)}.")
            return

    game_identifiers = np.asarray(game_identifiers)
    if len(game_identifiers) != n_preds:
         print(f"Error: Game identifiers length ({len(game_identifiers)}) mismatch with predictions ({n_preds}).")
         return

    if y_true is not None:
        y_true = np.asarray(y_true)
        if len(y_true) != n_preds:
             print(f"Error: True values length ({len(y_true)}) mismatch ({n_preds}).")
             return

    if ensemble_pred is not None:
        ensemble_pred = np.asarray(ensemble_pred)
        if len(ensemble_pred) != n_preds:
             print(f"Error: Provided ensemble prediction length ({len(ensemble_pred)}) mismatch ({n_preds}).")
             return


    # --- Data Slicing for Plotting ---
    if num_games_to_plot is None or num_games_to_plot <= 0 or num_games_to_plot >= n_preds:
         plot_slice = slice(None) # Plot all games
         num_games_plotted = n_preds
    else:
        plot_slice = slice(-num_games_to_plot, None)
        num_games_plotted = num_games_to_plot

    game_identifiers_plot = game_identifiers[plot_slice]
    y_true_plot = y_true[plot_slice] if y_true is not None else None
    predictions_plot = {name: preds[plot_slice] for name, preds in predictions_dict.items()}
    ensemble_pred_plot = ensemble_pred[plot_slice] if ensemble_pred is not None else None

    if len(game_identifiers_plot) == 0:
        print("Error: No games left to plot after slicing.")
        return

    x_indices = np.arange(len(game_identifiers_plot))

    # --- Calculate Ensemble and Standard Deviation ---
    component_preds_array = np.array(list(predictions_plot.values())) # Use list() for consistency
    model_agreement_std = None
    if component_preds_array.shape[0] > 1:
         model_agreement_std = np.std(component_preds_array, axis=0, ddof=1)

    if ensemble_pred_plot is None and ensemble_weights and len(model_names) > 0:
        print("Calculating ensemble prediction from components and weights...")
        weighted_preds_sum = np.zeros(len(game_identifiers_plot))
        total_weight = 0
        valid_weights = 0
        for name in model_names:
            weight = ensemble_weights.get(name, 0)
            if weight > 0 and name in predictions_plot: # Check name exists
                weighted_preds_sum += predictions_plot[name] * weight
                total_weight += weight
                valid_weights += 1
        if total_weight > 0:
            ensemble_pred_plot = weighted_preds_sum / total_weight
            print(f"  Ensemble calculated using {valid_weights} models with total weight {total_weight:.2f}.")
        else:
            print("  Warning: Sum of valid weights is zero. Cannot calculate weighted ensemble.")
            if component_preds_array.shape[0] > 0:
                 ensemble_pred_plot = np.mean(component_preds_array, axis=0)
                 print("  Using simple average of components as fallback ensemble.")


    # --- Plotting ---
    plt.figure(figsize=figsize)

    colors = plt.cm.get_cmap('tab10', len(model_names))
    for i, name in enumerate(model_names):
        if name in predictions_plot: # Check again before plotting
            plt.scatter(x_indices, predictions_plot[name], label=f"{name} Pred.",
                        alpha=0.6, s=35, color=colors(i))

    if ensemble_pred_plot is not None:
        plt.plot(x_indices, ensemble_pred_plot, 'k-o', linewidth=2, markersize=5, label='Ensemble Pred.', zorder=len(model_names) + 1) # Ensure it's on top
        if model_agreement_std is not None:
            ci_low = ensemble_pred_plot - 1.96 * model_agreement_std
            ci_high = ensemble_pred_plot + 1.96 * model_agreement_std
            plt.fill_between(x_indices, ci_low, ci_high, color='gray', alpha=0.25,
                             label='Model Agreement (±1.96 * StDev)')

    if y_true_plot is not None:
        plt.plot(x_indices, y_true_plot, 'r-X', linewidth=2, markersize=7, label='Actual Value', zorder=len(model_names) + 2)

    plt.xticks(x_indices, game_identifiers_plot, rotation=90, fontsize=9)
    plt.ylabel(target_name)
    plt.title(f'Model Predictions & Agreement ({num_games_plotted} Games)')
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.05, 0.88, 0.97])

    if save_dir_path:
        f_path = save_dir_path / "model_agreement_range.png"
        try:
            plt.savefig(f_path, bbox_inches='tight')
            print(f"Plot saved to {f_path}")
        except Exception as e:
            print(f"Error saving plot {f_path}: {e}")
    plt.show()

    # --- Plot Error vs Model Agreement (if possible) ---
    if y_true_plot is not None and ensemble_pred_plot is not None and model_agreement_std is not None:
        print("Generating error vs. model agreement plot...")
        errors = np.abs(y_true_plot - ensemble_pred_plot)
        valid_std_mask = np.isfinite(model_agreement_std) & (model_agreement_std >= 0)

        if np.sum(valid_std_mask) < 5:
             print("  Skipping error vs agreement plot: Insufficient valid standard deviation values.")
             print("--- Model Agreement Plot Complete ---")
             return

        agreement_df = pd.DataFrame({
            'Absolute Error': errors[valid_std_mask],
            'Model Agreement (StDev)': model_agreement_std[valid_std_mask]
        })

        try:
            num_quantiles = min(3, max(1, len(agreement_df) // 5))
            if num_quantiles < 2:
                 print("  Skipping error vs agreement plot: Not enough data for multiple bins.")
                 print("--- Model Agreement Plot Complete ---")
                 return

            agreement_df['Agreement Level'] = pd.qcut(agreement_df['Model Agreement (StDev)'], q=num_quantiles,
                                                     labels=[f'Level {i+1}' for i in range(num_quantiles)],
                                                     duplicates='drop')

            # Generate descriptive labels dynamically based on actual bins created
            unique_bins = sorted(agreement_df['Agreement Level'].unique())
            level_map = {bin_label: f'High-{i+1}' for i, bin_label in enumerate(unique_bins)}
            desc_labels = [level_map[bin_label] for bin_label in unique_bins]
            agreement_df['Agreement Level Desc'] = agreement_df['Agreement Level'].map(level_map)


            plt.figure(figsize=(max(6, len(desc_labels)*2.5), 5))
            sns.boxplot(x='Agreement Level Desc', y='Absolute Error', data=agreement_df,
                        order=desc_labels, palette='coolwarm')
            plt.title('Prediction Error vs. Model Agreement Level')
            plt.xlabel('Model Agreement (Lower StDev = Higher Agreement)')
            plt.ylabel('Absolute Prediction Error')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            if save_dir_path:
                f_path = save_dir_path / "error_vs_agreement.png"
                try:
                    plt.savefig(f_path, bbox_inches='tight')
                    print(f"Plot saved to {f_path}")
                except Exception as e:
                    print(f"Error saving plot {f_path}: {e}")
            plt.show()

        except ValueError as e:
             print(f"  Warning: Could not create bins for error vs agreement plot: {e}")
        except Exception as e:
             print(f"  An unexpected error occurred during error vs agreement plot: {e}")

    # Print reason for skipping if applicable
    elif y_true_plot is None: print("Skipping error vs agreement plot: Actual values (y_true) not provided.")
    elif ensemble_pred_plot is None: print("Skipping error vs agreement plot: Ensemble predictions not available.")
    elif model_agreement_std is None: print("Skipping error vs agreement plot: Model agreement (StDev) could not be calculated.")

    print("--- Model Agreement Plot Complete ---")


# --- Prediction Trend Plot ---
def plot_predictions_over_time(dates: Union[List, pd.Series, np.ndarray],
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "Predictions Over Time",
                               target_name: str = "Score",
                               figsize: tuple = (14, 7),
                               save_dir: Optional[Union[str, Path]] = None):
    """
    Plots actual vs predicted values over time. Requires dates to be sortable.

    Args:
        dates: List, Series or array of dates/timestamps or other sortable identifiers
               corresponding to the predictions.
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        title: Title for the plot.
        target_name: Name of the target variable for the y-axis label.
        figsize: Figure size.
        save_dir: Optional directory (string or Path object) to save plots.
    """
    try:
        dates = np.asarray(dates)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not (len(dates) == len(y_true) == len(y_pred)):
            print(f"Error: Length mismatch between dates ({len(dates)}), y_true ({len(y_true)}), and y_pred ({len(y_pred)}).")
            return

        try:
            dates_dt = pd.to_datetime(dates)
            is_datetime_type = True
        except Exception:
            print("Warning: Could not convert 'dates' to datetime objects. Plotting against original values.")
            dates_dt = dates
            is_datetime_type = False

        df = pd.DataFrame({'Date': dates_dt, 'Actual': y_true, 'Predicted': y_pred})

        try:
            df = df.sort_values('Date')
        except TypeError:
            print("Warning: Could not sort by 'Date'. Plotting in original order.")

        plt.figure(figsize=figsize)
        # Use index for x-axis if dates are not reliably numeric/datetime for plotting
        x_plot = df.index if not is_datetime_type else df['Date'].values
        plt.plot(x_plot, df['Actual'], 'o-', label='Actual', alpha=0.8, markersize=4, linewidth=1.5)
        plt.plot(x_plot, df['Predicted'], 'o--', label='Predicted', alpha=0.8, markersize=4, linewidth=1.5)

        plt.title(title, fontsize=14)
        if is_datetime_type:
            plt.xlabel('Date', fontsize=12)
            try:
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate(rotation=30, ha='right')
            except Exception as fmt_e:
                print(f"Could not apply date formatting: {fmt_e}")
                plt.xticks(rotation=30, ha='right')
        else:
            plt.xlabel('Game Sequence / Identifier', fontsize=12)
            # Only show limited ticks if identifiers are non-numeric and numerous
            if len(df) > 20:
                 step = len(df) // 10
                 plt.xticks(ticks=df.index[::step], labels=df['Date'].iloc[::step], rotation=30, ha='right')
            else:
                 plt.xticks(ticks=df.index, labels=df['Date'], rotation=30, ha='right')


        plt.ylabel(target_name, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / "predictions_over_time.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                print(f"Plot saved to {f_path}")
            except Exception as e:
                print(f"Error saving plot {f_path}: {e}")
        plt.show()

    except Exception as e:
        print(f"Error generating predictions over time plot: {e}")

# --- Advanced Residual/Bias Analysis Visualizations ---

def plot_residuals_analysis_detailed(y_true: Union[pd.Series, np.ndarray],
                                     y_pred: Union[pd.Series, np.ndarray],
                                     title_prefix: str = "",
                                     figsize: tuple = (12, 10),
                                     save_dir: Optional[Union[str, Path]] = None):
    """
    Generates a detailed set of residual analysis plots.

    Includes: Distribution, Q-Q plot, Residuals vs. Predicted (with optional LOWESS),
              and Residuals vs. Actual.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted values.
        title_prefix: Prefix for the plot titles (e.g., "Training Data").
        figsize: Figure size for the combined plot.
        save_dir: Optional directory (string or Path object) to save the figure.
    """
    print(f"\n--- Generating Detailed Residual Analysis: {title_prefix} ---")
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            print("Warning: Invalid input arrays for detailed residual analysis. Skipping.")
            return

        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{title_prefix} Residual Analysis'.strip(), fontsize=16)
        axes = axes.flatten()

        # 1. Residuals Histogram
        sns.histplot(residuals, kde=True, bins=30, ax=axes[0])
        axes[0].axvline(0, color='r', linestyle='--', label='Zero Residual')
        axes[0].axvline(mean_residual, color='g', linestyle='-', label=f'Mean: {mean_residual:.2f}')
        axes[0].set_xlabel('Residual (Actual - Predicted)')
        axes[0].set_title('Distribution of Residuals')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        if _has_statsmodels: # Check if import succeeded
            try:
                lowess = sm.nonparametric.lowess(residuals, y_pred, frac=0.3)
                axes[1].plot(lowess[:, 0], lowess[:, 1], color='orange', lw=2, label='LOWESS Smoother')
                axes[1].legend()
            except Exception as e:
                print(f"Note: Could not add LOWESS smoother: {e}")
        axes[1].set_xlabel('Predicted Value')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals vs. Predicted Values')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 3. Q-Q plot
        try:
            stats.probplot(residuals, dist="norm", plot=axes[2])
            # Style the Q-Q plot points and line for consistency
            if len(axes[2].get_lines()) == 2: # Ensure plot generated lines
                 axes[2].get_lines()[0].set_markerfacecolor('C0')
                 axes[2].get_lines()[0].set_markeredgecolor('C0')
                 axes[2].get_lines()[0].set_markersize(4.0)
                 axes[2].get_lines()[1].set_color('r')
            axes[2].set_title('Q-Q Plot (Normality Check)')
            axes[2].grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            print(f"Could not generate Q-Q plot: {e}")
            axes[2].set_title('Q-Q Plot (Error)')


        # 4. Residuals vs. True Values
        axes[3].scatter(y_true, residuals, alpha=0.5)
        axes[3].axhline(y=0, color='r', linestyle='--')
        axes[3].set_xlabel('Actual Value')
        axes[3].set_ylabel('Residual')
        axes[3].set_title('Residuals vs. Actual Values')
        axes[3].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_name = f"{title_prefix.replace(' ', '_').lower()}_residuals_detailed.png".lstrip("_")
            f_path = save_dir_path / f_name
            try:
                plt.savefig(f_path, bbox_inches='tight')
                print(f"Plot saved to {f_path}")
            except Exception as e:
                print(f"Error saving plot {f_path}: {e}")
        plt.show()

    except Exception as e:
        print(f"Error generating detailed residual analysis plot: {e}")


def plot_conditional_bias(y_pred: Union[pd.Series, np.ndarray],
                          residuals: Union[pd.Series, np.ndarray],
                          n_bins: int = 10,
                          title: str = "Conditional Bias by Predicted Range",
                          figsize: tuple = (12, 6),
                          save_dir: Optional[Union[str, Path]] = None):
    """
    Analyzes and plots how the mean residual (bias) varies across predicted value ranges.
    Uses quantile-based bins for robustness.

    Args:
        y_pred: Array or Series of predicted values.
        residuals: Array or Series of corresponding residuals (actual - predicted).
        n_bins: Target number of bins to divide the prediction range into using quantiles.
        title: Title for the plot.
        figsize: Figure size.
        save_dir: Optional directory (string or Path object) to save the figure.
    """
    print("\n--- Generating Conditional Bias Plot ---")
    try:
        y_pred = np.asarray(y_pred)
        residuals = np.asarray(residuals)

        if len(y_pred) != len(residuals) or len(y_pred) == 0:
            print("Warning: Invalid input arrays for conditional bias plot. Skipping.")
            return

        df = pd.DataFrame({'prediction': y_pred, 'residual': residuals})

        if len(df) < n_bins * 2: # Need sufficient points for binning
            print(f"Warning: Not enough data points ({len(df)}) for {n_bins} bins. Skipping conditional bias plot.")
            return

        # Use quantiles for binning
        df['bin_q'] = pd.qcut(df['prediction'], q=n_bins, labels=False, duplicates='drop')
        n_actual_bins = df['bin_q'].nunique()
        if n_actual_bins < 2:
             print(f"Warning: Could only create {n_actual_bins} valid bin(s) using quantiles. Skipping plot.")
             return

        analysis = df.groupby('bin_q').agg(
            bin_center=('prediction', 'mean'),
            mean_residual=('residual', 'mean'),
            std_residual=('residual', 'std'),
            count=('residual', 'size')
        ).reset_index()

        # Calculate 95% CI, requires count > 1
        analysis = analysis[analysis['count'] > 1].copy() # Use copy to avoid SettingWithCopyWarning
        if analysis.empty:
             print("Warning: No bins with sufficient data (>1) found for conditional bias plot.")
             return

        analysis['sem'] = analysis['std_residual'] / np.sqrt(analysis['count'])
        analysis['ci_lower'] = analysis['mean_residual'] - 1.96 * analysis['sem']
        analysis['ci_upper'] = analysis['mean_residual'] + 1.96 * analysis['sem']

        analysis = analysis.sort_values('bin_center')

        # Visualize conditional bias
        plt.figure(figsize=figsize)
        plt.plot(analysis['bin_center'], analysis['mean_residual'], 'o-', label='Mean Residual (Bias)')
        plt.fill_between(
            analysis['bin_center'], analysis['ci_lower'], analysis['ci_upper'],
            alpha=0.3, label='95% CI for Mean Residual'
        )
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel('Average Predicted Value in Bin')
        plt.ylabel('Mean Residual (Bias)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / "conditional_bias.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                print(f"Plot saved to {f_path}")
            except Exception as e:
                print(f"Error saving plot {f_path}: {e}")
        plt.show()

    except ValueError as e: # Catch qcut errors etc.
        print(f"Warning: Could not create bins for conditional bias plot. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during conditional bias plot generation: {e}")

    print("--- Conditional Bias Plot Complete ---")


def plot_temporal_bias(dates: Union[List, pd.Series, np.ndarray],
                       residuals: Union[pd.Series, np.ndarray],
                       freq: str = 'M',
                       title: str = "Temporal Bias Trend",
                       figsize: tuple = (12, 6),
                       save_dir: Optional[Union[str, Path]] = None):
    """
    Analyzes and plots the trend of mean residual (bias) over time.

    Args:
        dates: List, Series or array of dates/timestamps corresponding to the residuals.
               Must be convertible to datetime objects.
        residuals: Array or Series of corresponding residuals (actual - predicted).
        freq: Pandas frequency string for grouping (e.g., 'M', 'W', 'D', 'Q').
        title: Title for the plot.
        figsize: Figure size.
        save_dir: Optional directory (string or Path object) to save the figure.
    """
    print("\n--- Generating Temporal Bias Plot ---")
    try:
        residuals = np.asarray(residuals)
        try:
            # Attempt conversion, raise error immediately if it fails fundamentally
            dates_dt = pd.to_datetime(dates, errors='raise')
        except Exception as e:
            print(f"Error: Could not convert 'dates' to datetime objects: {e}")
            return

        if len(dates_dt) != len(residuals) or len(dates_dt) == 0:
            print("Error: Length mismatch or empty arrays for temporal bias plot.")
            return

        df = pd.DataFrame({'date': dates_dt, 'residual': residuals})

        # Group by the specified frequency
        df['period'] = df['date'].dt.to_period(freq)
        period_bias = df.groupby('period')['residual'].agg(['mean', 'std', 'count']).reset_index()
        period_bias['timestamp'] = period_bias['period'].dt.to_timestamp() # For plotting

        # Filter periods with enough data points
        period_bias = period_bias[period_bias['count'] > 2].copy()
        if period_bias.empty:
            print(f"Warning: No periods with sufficient data (>2) found for frequency '{freq}'. Skipping temporal bias plot.")
            return

        period_bias['sem'] = period_bias['std'] / np.sqrt(period_bias['count'])
        period_bias['ci_lower'] = period_bias['mean'] - 1.96 * period_bias['sem']
        period_bias['ci_upper'] = period_bias['mean'] + 1.96 * period_bias['sem']

        period_bias = period_bias.sort_values('timestamp')

        # Visualize temporal bias
        plt.figure(figsize=figsize)
        plt.plot(period_bias['timestamp'], period_bias['mean'], 'o-', label=f'Mean Bias ({freq})')
        plt.fill_between(
            period_bias['timestamp'], period_bias['ci_lower'], period_bias['ci_upper'],
            alpha=0.3, label='95% CI for Mean Bias'
        )
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Bias')
        plt.title(title + f" ({freq})")
        plt.xlabel('Time Period')
        plt.ylabel('Mean Residual (Bias)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        try:
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gcf().autofmt_xdate(rotation=30, ha='right')
        except Exception as fmt_e:
            print(f"Could not apply date formatting: {fmt_e}")
            plt.xticks(rotation=30, ha='right')

        plt.tight_layout()

        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / f"temporal_bias_{freq}.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                print(f"Plot saved to {f_path}")
            except Exception as e:
                print(f"Error saving plot {f_path}: {e}")
        plt.show()

    except Exception as e:
         print(f"An unexpected error occurred during temporal bias plot generation: {e}")

    print("--- Temporal Bias Plot Complete ---")


# --- Higher-Level Report Generation ---
def generate_evaluation_report(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               model_name: str,
                               target_type: str = 'total',
                               dates: Optional[Union[List, pd.Series, np.ndarray]] = None,
                               calculate_custom_losses: bool = True,
                               include_bias_analysis: bool = True,
                               save_dir: Optional[Union[str, Path]] = None):
    """
    Generates a comprehensive evaluation report with metrics and various plots.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        model_name: Name of the model being evaluated (for titles/filenames).
        target_type: Type of score ('home', 'away', 'total', 'diff').
        dates: Optional dates corresponding to predictions for temporal analysis.
        calculate_custom_losses: Whether to include custom loss calculations.
        include_bias_analysis: Whether to include detailed residual and bias plots.
        save_dir: Optional directory (string or Path object) to save plots.

    Returns:
        Dictionary of calculated metrics.
    """
    print(f"\n{'='*20} Evaluation Report: {model_name} ({target_type}) {'='*20}")
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {save_dir_path}")

    # --- 1. Calculate Metrics ---
    print("\n--- Calculating Metrics ---")
    metrics = evaluate_predictions(y_true, y_pred, target_type, calculate_custom_losses)
    print("Evaluation Metrics:")
    if all(pd.isna(v) for v in metrics.values()):
         print("  Error: All metrics are NaN. Cannot proceed with report.")
         return metrics

    for key, value in metrics.items():
        # Check if value is not NaN before formatting
        if pd.notna(value):
            print(f"  {key.replace('_', ' ').title():<25}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title():<25}: NaN")


    # --- 2. Generate Core Visualizations ---
    print("\n--- Generating Core Visualizations ---")
    base_filename = f"{model_name}_{target_type}".replace(" ", "_").lower()

    plot_actual_vs_predicted(
        y_true, y_pred,
        title=f"{model_name} - Actual vs. Predicted ({target_type})",
        metrics=metrics,
        save_path=save_dir_path / f"{base_filename}_actual_vs_pred.png" if save_dir_path else None
    )

    plot_score_distribution_density(
        y_true, y_pred,
        title=f"{model_name} - Score Distribution Density ({target_type})",
        save_path=save_dir_path / f"{base_filename}_density.png" if save_dir_path else None
    )

    # --- 3. Generate Residual and Bias Analysis Plots (Optional) ---
    if include_bias_analysis:
        print("\n--- Generating Residual & Bias Analysis ---")
        residuals = np.asarray(y_true) - np.asarray(y_pred)

        plot_residuals_analysis_detailed(
            y_true, y_pred,
            title_prefix=f"{model_name} ({target_type})",
            save_dir=save_dir_path
        )

        plot_conditional_bias(
            y_pred, residuals,
            title=f"{model_name} - Conditional Bias ({target_type})",
            save_dir=save_dir_path
        )

        plot_error_by_prediction_range(
            y_true, y_pred,
            title=f"{model_name} - Mean Error by Range ({target_type})",
            save_path=save_dir_path / f"{base_filename}_error_range.png" if save_dir_path else None
        )

        if dates is not None:
            plot_temporal_bias(
                dates, residuals, freq='M', # Default to Monthly
                title=f"{model_name} - Temporal Bias ({target_type})",
                save_dir=save_dir_path
            )
        else:
            print("Skipping temporal bias plot: 'dates' not provided.")

    # --- 4. Generate Time Series Plot (Optional) ---
    if dates is not None:
         print("\n--- Generating Time Series Plot ---")
         plot_predictions_over_time(
             dates=dates,
             y_true=y_true,
             y_pred=y_pred,
             title=f"{model_name} - Predictions Over Time ({target_type})",
             target_name=target_type.replace('_', ' ').title(),
             save_dir=save_dir_path
         )
    else:
         print("Skipping predictions over time plot: 'dates' not provided.")


    print(f"\n{'='*20} Report Generation Complete: {model_name} ({target_type}) {'='*20}")
    return metrics


# --- Main Example Block ---
if __name__ == '__main__':
    print("Running evaluation.py example...")
    print(f"Statsmodels available: {_has_statsmodels}")

    # --- Simulate some data ---
    np.random.seed(42)
    n_samples = 250
    y_true_total = np.random.normal(loc=225, scale=22, size=n_samples).round()
    # Simulate ensemble predictions (with some bias and noise)
    y_pred_ensemble = y_true_total + np.random.normal(loc=-2.5, scale=11, size=n_samples)
    # Simulate component predictions
    y_pred_xgb = y_true_total + np.random.normal(loc=-3, scale=13, size=n_samples)
    y_pred_rf = y_true_total + np.random.normal(loc=-1, scale=15, size=n_samples)
    y_pred_ridge = y_true_total + np.random.normal(loc=-4, scale=19, size=n_samples)
    # Simulate dates
    dates_sim = pd.to_datetime('2025-01-01') + pd.to_timedelta(np.arange(n_samples) * (12/n_samples) , unit='M') # Spread over a year

    # --- Generate Full Report for Ensemble ---
    ensemble_metrics = generate_evaluation_report(
        y_true=y_true_total,
        y_pred=y_pred_ensemble,
        model_name="Example Ensemble",
        target_type='total',
        dates=dates_sim, # Include dates for temporal plots
        calculate_custom_losses=True,
        include_bias_analysis=True,
        save_dir="./evaluation_plots/ensemble_report" # Save all plots here
    )

    # --- Evaluate Component Models (for comparison plots) ---
    print("\n--- Evaluating Components for Comparison ---")
    comp_metrics = {}
    comp_preds = {'XGBoost': y_pred_xgb, 'RandomForest': y_pred_rf, 'Ridge': y_pred_ridge}
    weights = {'XGBoost': 0.5, 'RandomForest': 0.3, 'Ridge': 0.2}

    for name, pred in comp_preds.items():
        metrics = evaluate_predictions(y_true_total, pred, target_type='total')
        metrics['weight'] = weights.get(name, 0)
        comp_metrics[name] = metrics
        if pd.notna(metrics.get('rmse')) and pd.notna(metrics.get('r2')):
             print(f"  {name} Metrics: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
        else:
             print(f"  {name} Metrics: Calculation failed.")


    # --- Generate Comparison Plots ---
    print("\n--- Generating Comparison Plots ---")
    # Combine metrics for plotting functions
    all_metrics_for_plots = {}
    all_metrics_for_plots.update(comp_metrics)
    if ensemble_metrics and not all(pd.isna(v) for v in ensemble_metrics.values()): # Check if ensemble metrics are valid
         all_metrics_for_plots['Ensemble'] = ensemble_metrics

    plot_ensemble_component_comparison(
        component_metrics=comp_metrics,
        ensemble_metrics=ensemble_metrics,
        target_name="Total Score",
        save_path_prefix="./evaluation_plots/comparison" # Save plot here
    )

    plot_metrics_comparison(
        all_metrics_for_plots,
        metric_to_plot='rmse',
        higher_is_better=False,
        save_path="./evaluation_plots/comparison/rmse_comparison.png"
    )

    plot_metrics_comparison(
        all_metrics_for_plots,
        metric_to_plot='r2',
        higher_is_better=True,
        save_path="./evaluation_plots/comparison/r2_comparison.png"
    )

    # --- Example: Feature Importance (Requires loading/simulating actual models & features) ---
    print("\n--- Generating Feature Importance Example (Requires Fitted Models) ---")
    # Simulate fitted models (replace with actual loaded models)
    class MockModel:
        def __init__(self, importance):
            self.feature_importances_ = importance
    mock_features = [f'feature_{i}' for i in range(10)]
    models_loaded_example = {
        'XGBoost': MockModel(np.random.rand(10) * 0.5 + 0.1),
        'RandomForest': MockModel(np.random.rand(10) * 0.3 + 0.05),
        'Ridge': MockModel(np.random.rand(10) * 0.1) # Linear model needs coef_ usually, mocking with importances_ here
    }
    # Note: _get_feature_importance would need adjustment for Ridge coef_ if not mocking like this
    plot_feature_importances(
        models_dict=models_loaded_example,
        feature_names=mock_features,
        ensemble_weights=weights,
        save_dir="./evaluation_plots/feature_importance"
    )

    # --- Example: Model Agreement ---
    print("\n--- Generating Model Agreement Example ---")
    all_predictions_dict = {'XGBoost': y_pred_xgb, 'RandomForest': y_pred_rf, 'Ridge': y_pred_ridge}
    # Simulate game identifiers
    game_ids_sim = [f"{d.strftime('%Y-%m-%d')}_Game{i%2+1}" for i, d in enumerate(dates_sim)]

    plot_model_agreement(
        predictions_dict=all_predictions_dict,
        game_identifiers=game_ids_sim, # Use simulated identifiers
        y_true=y_true_total,
        ensemble_pred=y_pred_ensemble, # Pass the simulated ensemble prediction
        target_name="Total Score",
        num_games_to_plot=30, # Plot last 30 "games"
        save_dir="./evaluation_plots/agreement"
    )

    print("\nEvaluation script example finished.")