# backend/mlb_score_prediction/evaluation.py

"""
Module for evaluating MLB model predictions and visualizing results.
Provides functions for calculating standard regression metrics, custom MLB-specific losses,
and generating various plots for residual analysis, bias analysis, feature importance,
and model comparison.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy import stats
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, List, Union
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# ——— SHAP Setup ———
try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False
    logger.warning("SHAP not installed; SHAP plots will be disabled.")

# Optional import for LOWESS smoother
try:
    import statsmodels.api as sm # type: ignore
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
    logger.info("Note: 'statsmodels' not found. LOWESS smoother will not be available in residual plots.")

# --- Plotting Configuration ---
plt.style.use('fivethirtyeight')
sns.set_palette('colorblind')
DEFAULT_FIG_SIZE = (12, 8)
SMALL_FIG_SIZE = (10, 6)
LARGE_FIG_SIZE = (15, 10)
SEED = 42

# --- Core Metric Calculation Functions ---
def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]
                                 ) -> Dict[str, float]:
    """Calculates standard regression metrics."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr):
             logger.error(f"Error: Input arrays must have the same length. Got {len(y_true_arr)} and {len(y_pred_arr)}")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true_arr) == 0:
             logger.warning("Warning: Input arrays are empty.")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        # R2 score requires at least 2 samples
        r2 = r2_score(y_true_arr, y_pred_arr) if len(y_true_arr) >= 2 else np.nan
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    except Exception as e:
         logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
         return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

def calculate_mlb_runs_loss(y_true: Union[pd.Series, np.ndarray],
                            y_pred: Union[pd.Series, np.ndarray]
                            ) -> float:
    """Calculates a basic mean squared error loss for MLB runs."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr):
            logger.error("Length mismatch for MLB runs loss calculation.")
            return np.nan
        if len(y_true_arr) == 0:
            logger.warning("Empty input for MLB runs loss calculation.")
            return np.nan
        return mean_squared_error(y_true_arr, y_pred_arr)
    except Exception as e:
        logger.error(f"Error in calculate_mlb_runs_loss: {e}", exc_info=True)
        return np.nan

def calculate_mlb_distribution_loss(y_pred: Union[pd.Series, np.ndarray],
                                    target_type: str = 'total'
                                    ) -> float:
    """
    Calculates a loss based on deviation from expected MLB run distributions.
    NOTE: expected_mean and expected_std are placeholders and should be
    empirically derived from historical MLB data for better accuracy.
    """
    try:
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_pred_arr) == 0:
            logger.warning("Empty input for MLB distribution loss.")
            return np.nan

        # Placeholder MLB values - REPLACE WITH EMPIRICALLY DERIVED VALUES
        if target_type == 'home': expected_mean, expected_std = 4.6, 3.1  # Example home runs
        elif target_type == 'away': expected_mean, expected_std = 4.4, 3.0  # Example away runs
        elif target_type == 'total': expected_mean, expected_std = 9.0, 5.5 # Example total runs
        elif target_type == 'diff': expected_mean, expected_std = 0.2, 4.5  # Example run differential (home - away)
        else: # Default to similar to away/home if target_type is unknown
            logger.warning(f"Unknown target_type '{target_type}' for MLB distribution loss. Using default stats.")
            expected_mean, expected_std = 4.5, 3.0
        
        if expected_std <= 0:
            raise ValueError("Expected standard deviation must be positive for distribution loss.")
            
        z_score_squared = ((y_pred_arr - expected_mean) / expected_std) ** 2
        return np.mean(z_score_squared)
    except Exception as e:
        logger.error(f"Error calculating MLB distribution loss: {e}", exc_info=True)
        return np.nan

def evaluate_predictions(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray],
                         target_type: str = 'total', # e.g., 'home_runs', 'away_runs', 'total_runs'
                         calculate_custom_losses: bool = True
                         ) -> Dict[str, float]:
    """Calculates both standard and custom evaluation metrics for MLB predictions."""
    metrics = calculate_regression_metrics(y_true, y_pred)
    if calculate_custom_losses:
        metrics['mlb_runs_loss'] = calculate_mlb_runs_loss(y_true, y_pred)
        metrics['mlb_distribution_loss'] = calculate_mlb_distribution_loss(y_pred, target_type)
    return metrics

# --- Core Visualization Functions ---
def plot_actual_vs_predicted(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "Actual vs. Predicted Runs",
                               metrics_dict: Optional[Dict[str, float]] = None,
                               figsize: tuple = SMALL_FIG_SIZE,
                               save_path: Optional[Union[str, Path]] = None,
                               show_plot: bool = True):
    """Generates a scatter plot of actual vs. predicted MLB run values."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
            logger.warning("Invalid input for actual vs predicted plot.")
            if show_plot: 
                plt.close() # Close any potentially open plot
            return

        plt.figure(figsize=figsize)
        plt.scatter(y_true_arr, y_pred_arr, alpha=0.5, label="Predictions")
        # Adjust plot limits for typical MLB run values (e.g., 0 to 20)
        min_val = max(0, min(np.min(y_true_arr), np.min(y_pred_arr)) - 2)
        max_val = max(np.max(y_true_arr), np.max(y_pred_arr)) + 2
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
        plt.xlabel('Actual Runs'); plt.ylabel('Predicted Runs'); plt.title(title)
        plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

        if metrics_dict and 'r2' in metrics_dict and 'rmse' in metrics_dict:
             r2 = metrics_dict['r2']; rmse = metrics_dict['rmse']
             if not (np.isnan(r2) or np.isnan(rmse)):
                 plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}', # Adjusted R2 precision
                          transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Error generating actual vs predicted plot: {e}", exc_info=True)
        if show_plot: plt.close() # Ensure plot is closed on error if not shown

def plot_residuals_distribution(y_true: Union[pd.Series, np.ndarray],
                                y_pred: Union[pd.Series, np.ndarray],
                                title: str = "MLB Residuals Analysis",
                                figsize: tuple = (15, 6),
                                save_path_prefix: Optional[Union[str, Path]] = None,
                                show_plot: bool = True):
    """Generates plots for MLB residual analysis: histogram and residuals vs. predicted."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
            logger.warning("Invalid input for residuals distribution plot.")
            if show_plot: plt.close()
            return
        
        residuals = y_true_arr - y_pred_arr
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        sns.histplot(residuals, kde=True, ax=axes[0])
        axes[0].axvline(x=0, color='r', linestyle='--')
        axes[0].set_xlabel('Residual (Actual Runs - Predicted Runs)')
        axes[0].set_title(f'Residuals Distribution (Mean: {np.mean(residuals):.2f})')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        axes[1].scatter(y_pred_arr, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Runs')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs. Predicted Runs')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path_prefix:
            save_path = Path(f"{save_path_prefix}_distribution.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Error generating residuals distribution plot: {e}", exc_info=True)
        if show_plot: plt.close()


def plot_error_by_prediction_range(y_true: Union[pd.Series, np.ndarray],
                                   y_pred: Union[pd.Series, np.ndarray],
                                   num_bins: int = 8, # Adjusted for potentially smaller MLB run ranges
                                   title: str = "Mean Prediction Error by Runs Range (MLB)",
                                   figsize: tuple = SMALL_FIG_SIZE,
                                   save_path: Optional[Union[str, Path]] = None,
                                   show_plot: bool = True):
    """Visualizes the mean residual (error) across different bins of predicted MLB runs."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
            logger.warning("Invalid input for error range plot.")
            if show_plot: plt.close()
            return

        residuals = y_true_arr - y_pred_arr
        df = pd.DataFrame({'prediction': y_pred_arr, 'residual': residuals})
        
        if len(df) < num_bins * 2: # Ensure enough data for binning
            logger.warning("Not enough data for requested bins in error range plot. Attempting fewer bins or aborting.")
            num_bins = max(2, len(df) // 5) # Dynamically reduce bins if data is scarce
            if num_bins < 2:
                if show_plot: plt.close()
                return

        df['bin_q'] = pd.qcut(df['prediction'], q=num_bins, labels=False, duplicates='drop')
        n_actual_bins = df['bin_q'].nunique()
        
        if n_actual_bins < 2:
            logger.warning("Could not create enough distinct bins for error range plot.")
            if show_plot: plt.close()
            return
            
        binned_stats = df.groupby('bin_q').agg(
            mean_residual=('residual', 'mean'),
            sem_residual=('residual', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
            bin_center=('prediction', 'mean'),
            count=('residual', 'size')
        ).reset_index()
        
        binned_stats = binned_stats[binned_stats['count'] > 1] # Ensure bins have more than one observation for SEM
        if binned_stats.empty:
            logger.warning("No bins with sufficient data after grouping for error range plot.")
            if show_plot: plt.close()
            return

        binned_stats = binned_stats.sort_values('bin_center')
        
        plt.figure(figsize=figsize)
        plt.errorbar(binned_stats['bin_center'], binned_stats['mean_residual'], 
                     yerr=binned_stats['sem_residual'], fmt='o-', capsize=5, label='Mean Residual ± SEM')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Average Predicted Runs in Bin'); plt.ylabel('Mean Residual (Actual - Predicted)')
        plt.title(title); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    except ValueError as e: # Specifically for qcut errors
        logger.warning(f"Could not create bins for error range plot: {e}")
        if show_plot: plt.close()
    except Exception as e:
        logger.error(f"Unexpected error during error range plot: {e}", exc_info=True)
        if show_plot: plt.close()


def plot_run_distribution_density(y_true: Union[pd.Series, np.ndarray], # Renamed from score to run
                                  y_pred: Union[pd.Series, np.ndarray],
                                  title: str = "Run Distribution Density (MLB)",
                                  figsize: tuple = SMALL_FIG_SIZE,
                                  save_path: Optional[Union[str, Path]] = None,
                                  show_plot: bool = True):
    """Visualizes the density distribution of actual vs. predicted MLB runs using KDE plots."""
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
            logger.warning("Invalid input for run density plot.")
            if show_plot: plt.close()
            return

        plt.figure(figsize=figsize)
        sns.kdeplot(y_true_arr, label='Actual Distribution', color='blue', fill=True, alpha=0.3, bw_adjust=0.5)
        sns.kdeplot(y_pred_arr, label='Predicted Distribution', color='red', fill=True, alpha=0.3, bw_adjust=0.5)
        plt.title(title); plt.xlabel('Runs'); plt.ylabel('Density')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Error generating run density plot: {e}", exc_info=True)
        if show_plot: plt.close()

# plot_metrics_comparison, _get_feature_importance, plot_shap_summary, plot_feature_importances
# are largely sport-agnostic and can be reused with minor title/label changes if needed,
# which would typically be handled by the calling function.

# (Re-pasting the NBA versions for these as they are mostly fine, will adjust titles in generate_evaluation_report)

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], metric_to_plot: str = 'rmse', higher_is_better: bool = False, title: Optional[str] = None, figsize: tuple = SMALL_FIG_SIZE, save_path: Optional[Union[str, Path]] = None, show_plot: bool = True):
    """ Creates a bar chart comparing a specific metric across different models/evaluations. """
    try:
        if not metrics_dict: logger.warning("Metrics dictionary is empty."); return
        model_names = list(metrics_dict.keys()); metric_values = [metrics_dict[name].get(metric_to_plot, np.nan) for name in model_names]
        valid_entries = [(name, val) for name, val in zip(model_names, metric_values) if pd.notna(val)]
        if not valid_entries: logger.warning(f"No valid values found for metric '{metric_to_plot}'."); return
        model_names_valid, metric_values_valid = zip(*valid_entries)
        metric_values_np = np.array(metric_values_valid)

        if title is None: title = f'{metric_to_plot.upper()} Comparison ({ "Higher" if higher_is_better else "Lower"} is Better)'
        
        plt.figure(figsize=figsize); bars = plt.bar(model_names_valid, metric_values_np)
        plt.ylabel(metric_to_plot.upper()); plt.title(title); plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        max_val = np.max(metric_values_np) if len(metric_values_np) > 0 else 0
        min_val = np.min(metric_values_np) if len(metric_values_np) > 0 else 0
        val_range = max(abs(max_val), abs(min_val))
        offset = val_range * 0.02 if val_range > 0 else 0.1
        
        for bar in bars:
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            text_y = yval + offset if yval >= 0 else yval - offset
            plt.text(bar.get_x() + bar.get_width()/2, text_y, f'{yval:.3f}', ha='center', va=va, fontsize=9)
            
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Error generating metrics comparison plot: {e}", exc_info=True)
        if show_plot: plt.close()


def plot_residuals_analysis_detailed(y_true: Union[pd.Series, np.ndarray],
                                     y_pred: Union[pd.Series, np.ndarray],
                                     title_prefix: str = "", # e.g., "MLB Home Runs"
                                     figsize: tuple = (12, 10),
                                     save_dir: Optional[Union[str, Path]] = None,
                                     show_plot: bool = True):
    """Generates a detailed set of residual analysis plots for MLB predictions."""
    logger.info(f"\n--- Generating Detailed Residual Analysis: {title_prefix} ---")
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
            logger.warning("Mismatch in lengths of y_true and y_pred or empty input for detailed residuals.")
            if show_plot: plt.close()
            return
            
        residuals = y_true_arr - y_pred_arr
        mean_residual = np.mean(residuals)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{title_prefix} Residual Analysis'.strip(), fontsize=16)
        axes = axes.flatten()

        # Plot 1: Histogram
        sns.histplot(residuals, kde=True, bins=30, ax=axes[0])
        axes[0].axvline(0, color='r', linestyle='--', label='Zero Residual')
        axes[0].axvline(mean_residual, color='g', linestyle='-', label=f'Mean: {mean_residual:.2f}')
        axes[0].set_xlabel('Residual (Actual - Predicted)')
        axes[0].set_title('Distribution of Residuals')
        axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Residuals vs Predicted
        axes[1].scatter(y_pred_arr, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        if _has_statsmodels:
            try:
                lowess = sm.nonparametric.lowess(residuals, y_pred_arr, frac=0.3)
                axes[1].plot(lowess[:, 0], lowess[:, 1], color='orange', lw=2, label='LOWESS Smoother')
                axes[1].legend()
            except Exception as e:
                logger.info(f"LOWESS smoothing failed: {e}")
        axes[1].set_xlabel('Predicted Value'); axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals vs. Predicted Values'); axes[1].grid(True, linestyle='--', alpha=0.6)

        # Plot 3: Q-Q Plot
        try:
            stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot (Normality Check)')
            axes[2].grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            logger.warning(f"Q-Q plot failed: {e}")
            axes[2].set_title('Q-Q Plot (Error)')

        # Plot 4: Residuals vs Actual
        axes[3].scatter(y_true_arr, residuals, alpha=0.5)
        axes[3].axhline(y=0, color='r', linestyle='--')
        axes[3].set_xlabel('Actual Value'); axes[3].set_ylabel('Residual')
        axes[3].set_title('Residuals vs. Actual Values'); axes[3].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_name = f"{title_prefix.replace(' ', '_').lower()}_residuals_detailed.png".lstrip("_")
            f_path = save_dir_path / f_name
            try:
                plt.savefig(f_path, bbox_inches='tight')
                logger.info(f"Plot saved to {f_path}")
            except Exception as e:
                logger.error(f"Error saving detailed residual plot: {e}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Error generating detailed residual plot: {e}", exc_info=True)
        if show_plot: plt.close()

def plot_conditional_bias(y_pred: Union[pd.Series, np.ndarray],
                          residuals: Union[pd.Series, np.ndarray],
                          n_bins: int = 8, # Adjusted for MLB
                          title: str = "Conditional Bias by Predicted Runs Range (MLB)",
                          figsize: tuple = (12, 6),
                          save_dir: Optional[Union[str, Path]] = None,
                          show_plot: bool = True):
    """Analyzes and plots how the mean residual (bias) varies across predicted MLB run ranges."""
    logger.info("\n--- Generating Conditional Bias Plot ---")
    try:
        y_pred_arr = np.asarray(y_pred)
        residuals_arr = np.asarray(residuals)
        if len(y_pred_arr) != len(residuals_arr) or len(y_pred_arr) == 0:
            logger.warning("Length mismatch or empty array for conditional bias plot.")
            if show_plot:
                plt.close()
            return

        df = pd.DataFrame({'prediction': y_pred_arr, 'residual': residuals_arr})
        if len(df) < n_bins * 2:
            logger.warning("Insufficient data for specified bins in conditional bias plot.")
            n_bins = max(2, len(df) // 5)
        if n_bins < 2:
            logger.warning("Too few bins to plot conditional bias.")
            if show_plot:
                plt.close()
            return

        df['bin_q'] = pd.qcut(df['prediction'], q=n_bins, labels=False, duplicates='drop')
        n_actual_bins = df['bin_q'].nunique()
        if n_actual_bins < 2:
            logger.warning("Not enough distinct bins created.")
            if show_plot:
                plt.close()
            return

        analysis = df.groupby('bin_q').agg(
            bin_center=('prediction', 'mean'),
            mean_residual=('residual', 'mean'),
            std_residual=('residual', 'std'),
            count=('residual', 'size')
        ).reset_index()
        analysis = analysis[analysis['count'] > 1].copy()
        if analysis.empty:
            logger.warning("No bins with sufficient data for analysis.")
            if show_plot:
                plt.close()
            return

        analysis['sem'] = analysis['std_residual'] / np.sqrt(analysis['count'])
        analysis['ci_lower'] = analysis['mean_residual'] - 1.96 * analysis['sem']
        analysis['ci_upper'] = analysis['mean_residual'] + 1.96 * analysis['sem']
        analysis = analysis.sort_values('bin_center')

        plt.figure(figsize=figsize)
        plt.plot(
            analysis['bin_center'],
            analysis['mean_residual'],
            'o-',
            label='Mean Residual (Bias)'
        )
        plt.fill_between(
            analysis['bin_center'],
            analysis['ci_lower'],
            analysis['ci_upper'],
            alpha=0.3,
            label='95% CI for Mean Residual'
        )
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel('Average Predicted Runs in Bin')
        plt.ylabel('Mean Residual (Bias)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / "mlb_conditional_bias.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                logger.info(f"Plot saved to {f_path}")
            except Exception as e:
                logger.error(f"Error saving conditional bias plot: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    except ValueError as e:
        logger.warning(f"Could not create bins for conditional bias plot: {e}")
        if show_plot:
            plt.close()

    except Exception as e:
        logger.error(f"Unexpected error during conditional bias plot: {e}", exc_info=True)
        if show_plot:
            plt.close()

    logger.info("--- Conditional Bias Plot Complete ---")



def plot_temporal_bias(dates: Union[List, pd.Series, np.ndarray],
                       residuals: Union[pd.Series, np.ndarray],
                       freq: str = 'M', # Monthly frequency
                       title: str = "Temporal Bias Trend (MLB)",
                       figsize: tuple = (12, 6),
                       save_dir: Optional[Union[str, Path]] = None,
                       show_plot: bool = True):
    """Analyzes and plots the trend of mean residual (bias) over time for MLB predictions."""
    logger.info("\n--- Generating Temporal Bias Plot ---")
    try:
        residuals_arr = np.asarray(residuals)
        try:
            # Ensure dates are pandas Datetime objects for period operations
            if not isinstance(dates, pd.Series) or not is_datetime(dates):
                dates_dt = pd.to_datetime(pd.Series(dates), errors='raise')
            else:
                dates_dt = dates # Already a pandas datetime Series
        except Exception as e:
            logger.error(f"Error converting dates to datetime: {e}. Dates must be convertible."); 
            if show_plot: 
                plt.close(); return
            
        if len(dates_dt) != len(residuals_arr) or len(dates_dt) == 0:
            logger.error("Mismatch in length or empty data for temporal bias plot."); 
            if show_plot: 
                plt.close(); return
            
        df = pd.DataFrame({'date': dates_dt, 'residual': residuals_arr})
        df['period'] = df['date'].dt.to_period(freq)
        
        period_bias = df.groupby('period')['residual'].agg(['mean', 'std', 'count']).reset_index()
        period_bias['timestamp'] = period_bias['period'].dt.to_timestamp() # For plotting
        period_bias = period_bias[period_bias['count'] > 2].copy() # Need enough samples for meaningful SEM/CI
        
        if period_bias.empty:
            logger.warning("Not enough data in any period_bias group after filtering for count > 2."); 
            if show_plot: 
                plt.close(); return
            
        period_bias['sem'] = period_bias['std'] / np.sqrt(period_bias['count'])
        period_bias['ci_lower'] = period_bias['mean'] - 1.96 * period_bias['sem']
        period_bias['ci_upper'] = period_bias['mean'] + 1.96 * period_bias['sem']
        period_bias = period_bias.sort_values('timestamp')
        
        plt.figure(figsize=figsize)
        plt.plot(period_bias['timestamp'], period_bias['mean'], 'o-', label=f'Mean Bias ({freq})')
        plt.fill_between(period_bias['timestamp'], period_bias['ci_lower'], period_bias['ci_upper'], alpha=0.3, label='95% CI for Mean Bias')
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Bias')
        plt.title(f"{title} ({freq})"); plt.xlabel('Time Period'); plt.ylabel('Mean Residual (Bias)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        
        try: # Date formatting for x-axis
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gcf().autofmt_xdate(rotation=30, ha='right')
        except Exception as fmt_e:
            logger.warning(f"Error formatting date axis: {fmt_e}")
            plt.xticks(rotation=30, ha='right') # Basic rotation as fallback
            
        plt.tight_layout()
        if save_dir:
            save_dir_path = Path(save_dir); save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / f"mlb_temporal_bias_{freq}.png"
            try: plt.savefig(f_path, bbox_inches='tight'); logger.info(f"Plot saved to {f_path}")
            except Exception as e: logger.error(f"Error saving temporal bias plot: {e}")
        if show_plot: plt.show()
        else: plt.close()
    except Exception as e:
        logger.error(f"Unexpected error during temporal bias plot: {e}", exc_info=True); 
        if show_plot: 
            plt.close();
    logger.info("--- Temporal Bias Plot Complete ---")


def _get_feature_importance(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    final_estimator = model
    if isinstance(model, Pipeline) and hasattr(model, 'steps') and model.steps:
        final_estimator = model.steps[-1][1] # Get the actual model from pipeline

    # For tree-based models (like XGBoost, RandomForest, etc.)
    if hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
        if len(importances) == len(feature_names):
            return dict(zip(feature_names, importances))
        else:
            logger.warning(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features for tree model.")
            return None
    # For linear models (like Ridge, Lasso, LogisticRegression, etc.)
    elif hasattr(final_estimator, 'coef_'):
        coefficients = final_estimator.coef_
        # Handle cases where coef_ might be 2D (e.g., multi-class LogisticRegression)
        if coefficients.ndim > 1:
            # Sum absolute coefficients across classes/outputs for a single importance score per feature
            coefficients = np.abs(coefficients).sum(axis=0) 
        else:
            coefficients = np.abs(coefficients.flatten()) # Ensure 1D and take absolute values
        
        if len(coefficients) == len(feature_names):
            return dict(zip(feature_names, coefficients))
        else:
            logger.warning(f"Length mismatch: {len(coefficients)} coeffs vs {len(feature_names)} features for linear model.")
            return None
    else:
        logger.debug(f"Model type {type(final_estimator).__name__} does not have feature_importances_ or coef_ attribute.")
        return None

def plot_shap_summary(
    model: Any,
    X: pd.DataFrame, # Test/validation data for SHAP values
    max_display: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
):
    """Compute and plot a SHAP summary bar chart (mean absolute SHAP values)."""
    if not _has_shap:
        logger.warning("SHAP library not available. Skipping SHAP summary plot.")
        return

    try:
        # Ensure X has feature names for SHAP plots
        if not isinstance(X, pd.DataFrame) or X.empty:
            logger.error("X must be a non-empty pandas DataFrame for SHAP plots.")
            return

        final_estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        
        # Use a small background sample for KernelExplainer if needed (for non-tree models)
        # For TreeExplainer, background data is not strictly necessary for shap_values, but good for consistency
        background_data = shap.sample(X, min(100, X.shape[0]), random_state=SEED)

        # Create explainer instance
        # Prefer TreeExplainer for tree models for speed and exactness
        if hasattr(final_estimator, 'feature_importances_') and not isinstance(final_estimator, (Pipeline)): # Simple check for tree-like
             explainer = shap.TreeExplainer(final_estimator, data=background_data) # Pass background for consistency if desired
        else: # Fallback to general Explainer, might use KernelExplainer
            # KernelExplainer requires a prediction function and background data
            # Ensure the prediction function outputs a single array for standard SHAP
            def pred_func(data_array):
                # SHAP passes numpy array, model expects DataFrame with feature names
                df_for_pred = pd.DataFrame(data_array, columns=X.columns)
                preds = final_estimator.predict(df_for_pred)
                return preds.flatten() # Ensure 1D output
            
            explainer = shap.KernelExplainer(pred_func, background_data)
        
        # Calculate SHAP values on the (potentially larger) X set
        shap_values = explainer.shap_values(X) # For KernelExplainer, this can be slow on large X

        # Create the SHAP summary plot (bar plot of mean absolute SHAP values)
        # For multi-output models, shap_values might be a list. We'd typically pick one or average.
        # Assuming single output regression for this plot.
        if isinstance(shap_values, list) and len(shap_values) > 0 :
            # Heuristic: if regression, shap_values might be a list of one item for multi-target Sklearn models.
            # Or for some explainers directly the array.
            # If it's for classification with multiple classes, one might pick a class.
            # For regression with TreeExplainer on XGBoost, shap_values is usually just the array.
             if len(shap_values) == 2 and isinstance(shap_values[0], np.ndarray): # Common for binary classification, less so regression
                 logger.info("SHAP values appear to be for binary classification; using positive class for summary.")
                 shap_values_to_plot = shap_values[1]
             else: # Take the first element if it's a list of arrays (e.g. multi-output regression)
                 shap_values_to_plot = shap_values[0] if isinstance(shap_values[0], np.ndarray) else shap_values
        else:
            shap_values_to_plot = shap_values
        
        # Ensure shap_values_to_plot is compatible with shap.plots.bar
        # It expects SHAP values directly, not an Explanation object for this specific plot type sometimes
        if hasattr(shap_values_to_plot, 'values') and hasattr(shap_values_to_plot, 'feature_names'): # If it's an Explanation object
            pass # Use as is
        elif isinstance(shap_values_to_plot, np.ndarray) and isinstance(X, pd.DataFrame):
             # Create a basic Explanation object if we only have raw SHAP values array
             shap_values_to_plot = shap.Explanation(values=shap_values_to_plot, data=X.values, feature_names=X.columns)
        else:
            logger.error(f"SHAP values are not in an expected format for plotting. Type: {type(shap_values_to_plot)}")
            return

        plt.figure() # Ensure a new figure context for SHAP plot
        shap.summary_plot(shap_values_to_plot, X, plot_type="bar", max_display=max_display, show=False)
        plt.title(f"SHAP Feature Importance (Top {max_display})")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating SHAP summary plot: {e}", exc_info=True)
        if show_plot and plt.gcf().get_axes(): plt.close() # Close plot if error occurred mid-way

def plot_feature_importances(
    models_dict: Dict[str, Any], # Dict of model_name: model_object
    feature_names: List[str],    # From training data X.columns
    X_test: Optional[pd.DataFrame] = None, # For SHAP plots if available
    top_n: int = 20,
    save_dir: Optional[Union[str, Path]] = None,
    show_plot: bool = True
):
    """
    Generates and saves feature importance plots for each model.
    Uses native importance if available, otherwise attempts SHAP.
    """
    logger.info("\n--- Generating Feature Importance Plots ---")
    out_dir_path = Path(save_dir) if save_dir else None
    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    all_models_importances: Dict[str, pd.Series] = {}

    for model_name, model_instance in models_dict.items():
        logger.info(f"Processing feature importance for model: {model_name}")
        
        # 1. Try native feature importance
        native_importances = _get_feature_importance(model_instance, feature_names)
        
        if native_importances:
            importance_series = pd.Series(native_importances).sort_values(ascending=False)
            all_models_importances[model_name] = importance_series
            logger.info(f"Native feature importances extracted for {model_name}.")

            # Save full list to CSV and TXT
            if out_dir_path:
                df_full = importance_series.reset_index()
                df_full.columns = ['feature', 'importance']
                csv_path = out_dir_path / f"native_feature_importance_{model_name}_full.csv"
                txt_path = out_dir_path / f"native_feature_importance_{model_name}_full.txt"
                try:
                    df_full.to_csv(csv_path, index=False)
                    with open(txt_path, 'w') as f: f.write(df_full.to_string(index=False))
                    logger.info(f"Saved full native feature importances for '{model_name}' to {out_dir_path}")
                except Exception as e:
                    logger.error(f"Error saving native importances for {model_name}: {e}")
        else:
            logger.info(f"No native feature importances found for {model_name}. SHAP will be primary if available.")

        # 2. Generate SHAP summary plot if SHAP is available and X_test is provided
        if _has_shap and X_test is not None and isinstance(X_test, pd.DataFrame) and not X_test.empty:
            # Ensure X_test has the feature_names as columns, in the correct order
            X_test_aligned = X_test[feature_names] if all(f in X_test.columns for f in feature_names) else X_test
            
            shap_save_path = out_dir_path / f"shap_summary_{model_name}.png" if out_dir_path else None
            plot_shap_summary(
                model=model_instance,
                X=X_test_aligned, # Use aligned test data
                max_display=top_n,
                save_path=shap_save_path,
                show_plot=False # Controlled by outer show_plot or batch saving
            )
            # Note: If SHAP values were to be used for the bar chart below, they'd need to be extracted here.
            # The current _get_feature_importance doesn't use SHAP for the combined bar chart.
        elif _has_shap and (X_test is None or X_test.empty):
             logger.warning(f"X_test not provided or empty; cannot generate SHAP plot for {model_name}.")


    # Plotting combined bar chart of NATIVE importances (if any were found)
    if not all_models_importances:
        logger.warning("No native feature importance data extracted for any model. Skipping combined bar plot.")
        if show_plot: plt.close() # Close any open figures from SHAP if they weren't closed
        return

    num_models_with_importance = len(all_models_importances)
    if num_models_with_importance == 0:
        if show_plot: plt.close(); 
        return

    # Determine overall top_n features across all models for consistent y-axis, if desired,
    # or plot top_n for each model individually. Current code plots top_n for each.
    
    # Max features to display in each subplot (can be different from SHAP's max_display)
    num_features_to_plot = min(top_n, len(feature_names)) 

    fig, axes = plt.subplots(1, num_models_with_importance, 
                             figsize=(num_models_with_importance * 6, num_features_to_plot * 0.3 + 2), 
                             squeeze=False) # Ensure axes is always 2D
    
    for ax_idx, (model_name, imp_series) in enumerate(all_models_importances.items()):
        ax = axes[0, ax_idx]
        top_features = imp_series.head(num_features_to_plot).sort_values(ascending=True)
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax, palette="viridis")
        ax.set_title(f"Native Importance: {model_name}")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")

    plt.tight_layout()
    if out_dir_path:
        try:
            plt.savefig(out_dir_path / "native_feature_importance_summary.png", bbox_inches='tight')
            logger.info(f"Combined native feature importance plot saved to {out_dir_path}")
        except Exception as e:
            logger.error(f"Error saving combined feature importance plot: {e}")
            
    if show_plot:
        plt.show()
    else:
        plt.close() # Close all figures if not showing interactively

    logger.info("--- Feature Importance Plotting Complete ---")


# ---  plot_predictions_over_time ---
def plot_predictions_over_time(dates: Union[List, pd.Series, np.ndarray],
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "MLB Predictions Over Time",
                               target_name: str = "Runs", # Changed from Score
                               figsize: tuple = (14, 7),
                               save_dir: Optional[Union[str, Path]] = None,
                               show_plot: bool = True):
    """Plots actual vs predicted MLB run values over time. Requires dates to be sortable."""
    logger.info("\n--- Generating Predictions Over Time Plot ---")
    try:
        y_true_arr = np.asarray(y_true).flatten()
        y_pred_arr = np.asarray(y_pred).flatten()
        
        try:
            if not isinstance(dates, pd.Series) or not is_datetime(dates):
                dates_dt = pd.to_datetime(pd.Series(dates), errors='raise')
            else:
                dates_dt = dates
        except Exception as e:
            logger.error(f"Error converting dates to datetime for time plot: {e}");
            if show_plot: 
                plt.close(); return

        if not (len(dates_dt) == len(y_true_arr) == len(y_pred_arr)) or len(dates_dt) == 0:
            logger.warning("Length mismatch or empty data for predictions over time plot."); 
            if show_plot: 
                plt.close(); return

        df = pd.DataFrame({'date': dates_dt, 'Actual': y_true_arr, 'Predicted': y_pred_arr}).sort_values(by='date')

        plt.figure(figsize=figsize)
        plt.plot(df['date'], df['Actual'], label=f'Actual {target_name}', marker='o', linestyle='-', alpha=0.7, markersize=4)
        plt.plot(df['date'], df['Predicted'], label=f'Predicted {target_name}', marker='x', linestyle='--', alpha=0.7, markersize=4)
        
        plt.title(title); plt.xlabel("Date"); plt.ylabel(target_name)
        plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        if save_dir:
            save_dir_path = Path(save_dir); save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / "mlb_predictions_over_time.png"
            try: plt.savefig(f_path, bbox_inches='tight'); logger.info(f"Plot saved to {f_path}")
            except Exception as e: logger.error(f"Error saving predictions over time plot: {e}")
        
        if show_plot: plt.show()
        else: plt.close()
    except Exception as e:
        logger.error(f"Error in plot_predictions_over_time: {e}", exc_info=True); 
        if show_plot: 
            plt.close();
    logger.info("--- Predictions Over Time Plot Complete ---")

# --- Higher-Level Report Generation ---
def generate_evaluation_report(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               model_name: str,
                               target_type: str = 'total_runs', # e.g. home_runs, total_runs
                               dates: Optional[Union[List, pd.Series, np.ndarray]] = None,
                               calculate_custom_losses: bool = True,
                               include_bias_analysis: bool = True,
                               save_dir: Optional[Union[str, Path]] = None,
                               show_plots_flag: bool = True): # Added flag to control showing plots
    """Generates a comprehensive MLB evaluation report with metrics and various plots."""
    logger.info(f"\n{'='*20} MLB Evaluation Report: {model_name} ({target_type}) {'='*20}")
    
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to: {save_dir_path}")

    logger.info("\n--- Calculating Metrics ---")
    metrics = evaluate_predictions(y_true, y_pred, target_type, calculate_custom_losses)
    print("Evaluation Metrics:")
    if all(pd.isna(v) for v in metrics.values()):
        logger.error("Error: All metrics are NaN. Cannot proceed with report.")
        return metrics # Return NaN metrics

    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title():<25}: {value:.4f}" if pd.notna(value) else f"  {key.replace('_', ' ').title():<25}: NaN")

    logger.info("\n--- Generating Core Visualizations ---")
    base_filename = f"mlb_{model_name}_{target_type}".replace(" ", "_").lower()
    
    actual_vs_pred_sp = save_dir_path / f"{base_filename}_actual_vs_pred.png" if save_dir_path else None
    plot_actual_vs_predicted(y_true, y_pred, title=f"{model_name} - Actual vs. Predicted ({target_type.replace('_',' ').title()})",
                               metrics_dict=metrics, save_path=actual_vs_pred_sp, show_plot=show_plots_flag)

    density_sp = save_dir_path / f"{base_filename}_density.png" if save_dir_path else None
    plot_run_distribution_density(y_true, y_pred, title=f"{model_name} - Run Distribution Density ({target_type.replace('_',' ').title()})",
                                   save_path=density_sp, show_plot=show_plots_flag)

    if include_bias_analysis:
        logger.info("\n--- Generating Residual & Bias Analysis ---")
        try:
            residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
            plot_residuals_analysis_detailed(y_true, y_pred, title_prefix=f"{model_name} ({target_type.replace('_',' ').title()})",
                                             save_dir=save_dir_path, show_plot=show_plots_flag)
            plot_conditional_bias(y_pred, residuals, title=f"{model_name} - Conditional Bias ({target_type.replace('_',' ').title()})",
                                  save_dir=save_dir_path, show_plot=show_plots_flag)
            
            error_range_sp = save_dir_path / f"{base_filename}_error_range.png" if save_dir_path else None
            plot_error_by_prediction_range(y_true, y_pred, title=f"{model_name} - Mean Error by Range ({target_type.replace('_',' ').title()})",
                                           save_path=error_range_sp, show_plot=show_plots_flag)
            
            if dates is not None and len(dates) == len(residuals):
                plot_temporal_bias(dates, residuals, freq='M', title=f"{model_name} - Temporal Bias ({target_type.replace('_',' ').title()})",
                                   save_dir=save_dir_path, show_plot=show_plots_flag)
            elif dates is None:
                logger.info("Skipping temporal bias plot: 'dates' not provided.")
            elif len(dates) != len(residuals):
                logger.warning(f"Skipping temporal bias plot: 'dates' length ({len(dates)}) != residuals length ({len(residuals)}).")

        except Exception as bias_e:
            logger.error(f"Error during bias analysis: {bias_e}", exc_info=True)


    if dates is not None and len(dates) == len(y_true): # Ensure dates match y_true length
        logger.info("\n--- Generating Time Series Plot ---")
        plot_predictions_over_time(dates=dates, y_true=y_true, y_pred=y_pred,
                                   title=f"{model_name} - Predictions Over Time ({target_type.replace('_',' ').title()})",
                                   target_name=target_type.replace('_', ' ').title(),
                                   save_dir=save_dir_path, show_plot=show_plots_flag)
    elif dates is None:
        logger.info("Skipping predictions over time plot: 'dates' not provided.")
    elif len(dates) != len(y_true):
         logger.warning(f"Skipping predictions over time plot: 'dates' length ({len(dates)}) != y_true length ({len(y_true)}).")


    logger.info(f"\n{'='*20} MLB Report Generation Complete: {model_name} ({target_type}) {'='*20}")
    return metrics

# --- Main Example Block ---
if __name__ == '__main__':
    logger.info("Running mlb_evaluation.py example...")
    # Example:
    # Create dummy data for MLB context
    num_samples = 200
    y_true_home = np.random.poisson(4.5, num_samples) # Avg home runs
    y_pred_home = np.clip(y_true_home + np.random.normal(0, 1.5, num_samples), 0, 20).round() # Predictions with some noise
    
    y_true_total = np.random.poisson(9.0, num_samples) # Avg total runs
    y_pred_total = np.clip(y_true_total + np.random.normal(0, 2.5, num_samples), 0, 30).round()

    example_dates = pd.date_range(end=datetime.now(), periods=num_samples, freq='D')

    # Test the main report function for home runs
    logger.info("\n\n--- Example for Home Runs ---")
    metrics_home = generate_evaluation_report(
        y_true=y_true_home,
        y_pred=y_pred_home,
        model_name="ExampleMLBModel",
        target_type="home_runs",
        dates=example_dates,
        calculate_custom_losses=True,
        include_bias_analysis=True,
        save_dir="temp_mlb_eval_reports/home_runs", # Example save directory
        show_plots_flag=False # Set to False for automated runs, True for interactive
    )
    print("\nFinal Metrics for Home Runs:", metrics_home)

    # Test the main report function for total runs
    logger.info("\n\n--- Example for Total Runs ---")
    metrics_total = generate_evaluation_report(
        y_true=y_true_total,
        y_pred=y_pred_total,
        model_name="ExampleMLBModel",
        target_type="total_runs",
        dates=example_dates,
        save_dir="temp_mlb_eval_reports/total_runs",
        show_plots_flag=False
    )
    print("\nFinal Metrics for Total Runs:", metrics_total)

    # Example for comparing metrics
    all_model_metrics = {
        "ExampleMLBModel_Home": metrics_home,
        "AnotherModel_Home": {k: v * 0.9 for k, v in metrics_home.items() if pd.notna(v)}, # Dummy better model
        "YetAnother_Home": {k: v * 1.1 for k, v in metrics_home.items() if pd.notna(v)} # Dummy worse model
    }
    if metrics_home and 'rmse' in metrics_home : # Check if rmse is available
        plot_metrics_comparison(
            all_model_metrics,
            metric_to_plot='rmse',
            higher_is_better=False,
            title="Home Runs Model RMSE Comparison",
            save_path="temp_mlb_eval_reports/metrics_comparison_rmse_home.png",
            show_plot=False
        )

    logger.info("MLB evaluation.py example run complete.")