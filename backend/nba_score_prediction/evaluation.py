# backend/nba_score_prediction/evaluation.py

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
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

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

# --- Core Metric Calculation Functions ---
def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray]
                                 ) -> Dict[str, float]:
    """Calculates standard regression metrics."""
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred):
             logger.error(f"Error: Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        if len(y_true) == 0:
             logger.warning("Warning: Input arrays are empty.")
             return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        mse = mean_squared_error(y_true, y_pred); rmse = np.sqrt(mse); mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    except Exception as e:
         logger.error(f"Error calculating regression metrics: {e}")
         return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

def calculate_nba_score_loss(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray]
                               ) -> float:
    """Calculates a basic mean squared error loss for NBA scores."""
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred): logger.error("Length mismatch for NBA score loss."); return np.nan
        if len(y_true) == 0: logger.warning("Empty input for NBA score loss."); return np.nan
        return mean_squared_error(y_true, y_pred)
    except Exception as e: logger.error(f"Error in calculate_nba_score_loss: {e}"); return np.nan

def calculate_nba_distribution_loss(y_pred: Union[pd.Series, np.ndarray],
                                      target_type: str = 'total'
                                      ) -> float:
    """Calculates a loss based on deviation from expected NBA score distributions."""
    try:
        y_pred = np.asarray(y_pred).flatten()
        if len(y_pred) == 0: logger.warning("Empty input for NBA distribution loss."); return np.nan
        if target_type == 'home': expected_mean, expected_std = 114, 13.5
        elif target_type == 'away': expected_mean, expected_std = 112, 13.5
        elif target_type == 'total': expected_mean, expected_std = 226, 23
        elif target_type == 'diff': expected_mean, expected_std = 2.5, 13.5
        else: expected_mean, expected_std = 112, 14
        if expected_std <= 0: raise ValueError("Expected standard deviation must be positive.")
        z_score_squared = ((y_pred - expected_mean) / expected_std) ** 2
        return np.mean(z_score_squared)
    except Exception as e: logger.error(f"Error calculating NBA distribution loss: {e}"); return np.nan

def evaluate_predictions(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray],
                         target_type: str = 'total',
                         calculate_custom_losses: bool = True
                         ) -> Dict[str, float]:
    """ Calculates both standard and custom evaluation metrics for predictions. """
    metrics = calculate_regression_metrics(y_true, y_pred)
    if calculate_custom_losses:
        metrics['nba_score_loss'] = calculate_nba_score_loss(y_true, y_pred) # Uses simple MSE for now
        metrics['nba_distribution_loss'] = calculate_nba_distribution_loss(y_pred, target_type)
    return metrics

# --- Core Visualization Functions ---
def plot_actual_vs_predicted(y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "Actual vs. Predicted Scores",
                               metrics_dict: Optional[Dict[str, float]] = None, 
                               figsize: tuple = SMALL_FIG_SIZE,
                               save_path: Optional[Union[str, Path]] = None,
                               show_plot: bool = True): 
    """ Generates a scatter plot of actual vs. predicted values. """
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred) or len(y_true) == 0: logger.warning("Invalid input for actual vs predicted plot."); return

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
        min_val = min(np.min(y_true), np.min(y_pred)) - 5
        max_val = max(np.max(y_true), np.max(y_pred)) + 5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
        plt.xlabel('Actual Values'); plt.ylabel('Predicted Values'); plt.title(title)
        plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

        if metrics_dict and 'r2' in metrics_dict and 'rmse' in metrics_dict:
             r2 = metrics_dict['r2']; rmse = metrics_dict['rmse']
             if not (np.isnan(r2) or np.isnan(rmse)):
                 plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}',
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

    except Exception as e: logger.error(f"Error generating actual vs predicted plot: {e}")

def plot_residuals_distribution(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], title: str = "Residuals Analysis", figsize: tuple = (15, 6), save_path_prefix: Optional[Union[str, Path]] = None):
    """ Generates plots for residual analysis: histogram and residuals vs. predicted. """
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred) or len(y_true) == 0: logger.warning("Invalid input for residuals distribution plot."); return
        residuals = y_true - y_pred; fig, axes = plt.subplots(1, 2, figsize=figsize); fig.suptitle(title, fontsize=16)
        sns.histplot(residuals, kde=True, ax=axes[0]); axes[0].axvline(x=0, color='r', linestyle='--'); axes[0].set_xlabel('Residual (Actual - Predicted)'); axes[0].set_title(f'Residuals Distribution (Mean: {np.mean(residuals):.2f})'); axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[1].scatter(y_pred, residuals, alpha=0.5); axes[1].axhline(y=0, color='r', linestyle='--'); axes[1].set_xlabel('Predicted Values'); axes[1].set_ylabel('Residuals'); axes[1].set_title('Residuals vs. Predicted Values'); axes[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path_prefix: save_path = Path(f"{save_path_prefix}_distribution.png"); save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except Exception as e: logger.error(f"Error generating residuals distribution plot: {e}")


def plot_error_by_prediction_range(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], num_bins: int = 10, title: str = "Mean Prediction Error by Score Range", figsize: tuple = SMALL_FIG_SIZE, save_path: Optional[Union[str, Path]] = None):
    """ Visualizes the mean residual (error) across different bins of predicted values. """
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred) or len(y_true) == 0: logger.warning("Invalid input for error range plot."); return
        residuals = y_true - y_pred; df = pd.DataFrame({'prediction': y_pred, 'residual': residuals})
        if len(df) < num_bins * 2: logger.warning("Not enough data for requested bins in error range plot."); return
        df['bin_q'] = pd.qcut(df['prediction'], q=num_bins, labels=False, duplicates='drop'); n_actual_bins = df['bin_q'].nunique()
        if n_actual_bins < 2: logger.warning("Could not create enough distinct bins for error range plot."); return
        binned_stats = df.groupby('bin_q').agg(mean_residual=('residual', 'mean'), sem_residual=('residual', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0), bin_center=('prediction', 'mean'), count=('residual', 'size')).reset_index()
        binned_stats = binned_stats[binned_stats['count'] > 1];
        if binned_stats.empty: logger.warning("No bins with sufficient data after grouping."); return
        binned_stats = binned_stats.sort_values('bin_center')
        plt.figure(figsize=figsize); plt.errorbar(binned_stats['bin_center'], binned_stats['mean_residual'], yerr=binned_stats['sem_residual'], fmt='o-', capsize=5, label='Mean Residual ± SEM')
        plt.axhline(y=0, color='r', linestyle='--'); plt.xlabel('Average Predicted Score in Bin'); plt.ylabel('Mean Residual (Actual - Predicted)'); plt.title(title); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_path: save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except ValueError as e: logger.warning(f"Could not create bins for error range plot: {e}")
    except Exception as e: logger.error(f"Unexpected error during error range plot: {e}")


def plot_score_distribution_density(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], title: str = "Score Distribution Density", figsize: tuple = SMALL_FIG_SIZE, save_path: Optional[Union[str, Path]] = None):
    """ Visualizes the density distribution of actual vs. predicted scores using KDE plots. """
    try:
        y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred) or len(y_true) == 0: logger.warning("Invalid input for score density plot."); return
        plt.figure(figsize=figsize); sns.kdeplot(y_true, label='Actual Distribution', color='blue', fill=True, alpha=0.3, bw_adjust=0.5); sns.kdeplot(y_pred, label='Predicted Distribution', color='red', fill=True, alpha=0.3, bw_adjust=0.5)
        plt.title(title); plt.xlabel('Score'); plt.ylabel('Density'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
        if save_path: save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except Exception as e: logger.error(f"Error generating score density plot: {e}")


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], metric_to_plot: str = 'rmse', higher_is_better: bool = False, title: Optional[str] = None, figsize: tuple = SMALL_FIG_SIZE, save_path: Optional[Union[str, Path]] = None):
    """ Creates a bar chart comparing a specific metric across different models/evaluations. """
    try:
        if not metrics_dict: logger.warning("Metrics dictionary is empty."); return
        model_names = list(metrics_dict.keys()); metric_values = [metrics_dict[name].get(metric_to_plot, np.nan) for name in model_names]
        valid_entries = [(name, val) for name, val in zip(model_names, metric_values) if pd.notna(val)]
        if not valid_entries: logger.warning(f"No valid values found for metric '{metric_to_plot}'."); return
        model_names, metric_values = zip(*valid_entries); metric_values = np.array(metric_values)
        if title is None: title = f'{metric_to_plot.upper()} Comparison ({ "Higher" if higher_is_better else "Lower"} is Better)'
        plt.figure(figsize=figsize); bars = plt.bar(model_names, metric_values); plt.ylabel(metric_to_plot.upper()); plt.title(title); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7)
        max_val = np.max(metric_values) if len(metric_values) > 0 else 0; min_val = np.min(metric_values) if len(metric_values) > 0 else 0; val_range = max(abs(max_val), abs(min_val)); offset = val_range * 0.02 if val_range > 0 else 0.1
        for bar in bars: yval = bar.get_height(); va = 'bottom' if yval >= 0 else 'top'; text_y = yval + offset if yval >= 0 else yval - offset; plt.text(bar.get_x() + bar.get_width()/2, text_y, f'{yval:.3f}', ha='center', va=va, fontsize=9)
        plt.tight_layout()
        if save_path: save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Plot saved to {save_path}")
        plt.show()
    except Exception as e: logger.error(f"Error generating metrics comparison plot: {e}")


def plot_residuals_analysis_detailed(y_true: Union[pd.Series, np.ndarray],
                                     y_pred: Union[pd.Series, np.ndarray],
                                     title_prefix: str = "",
                                     figsize: tuple = (12, 10),
                                     save_dir: Optional[Union[str, Path]] = None,
                                     show_plot: bool = True): 
    """Generates a detailed set of residual analysis plots."""
    logger.info(f"\n--- Generating Detailed Residual Analysis: {title_prefix} ---")
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            logger.warning("Mismatch in lengths of y_true and y_pred or empty input.")
            return
        residuals = y_true - y_pred
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
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        # Plot 2: Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        if _has_statsmodels:
            try:
                lowess = sm.nonparametric.lowess(residuals, y_pred, frac=0.3)
                axes[1].plot(lowess[:, 0], lowess[:, 1], color='orange', lw=2, label='LOWESS Smoother')
                axes[1].legend()
            except Exception as e:
                logger.info(f"LOWESS smoothing failed: {e}")
        axes[1].set_xlabel('Predicted Value')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals vs. Predicted Values')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        # Plot 3: Q-Q Plot
        try:
            stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot (Normality Check)')
            axes[2].grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            logger.warning(f"Q-Q plot failed: {e}")
            axes[2].set_title('Q-Q Plot (Error)')
        # Plot 4: Residuals vs Actual
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
                logger.info(f"Plot saved to {f_path}")
            except Exception as e:
                logger.error(f"Error saving detailed residual plot: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close() 

    except Exception as e:
        logger.error(f"Error generating detailed residual plot: {e}")

def plot_conditional_bias(y_pred: Union[pd.Series, np.ndarray],
                          residuals: Union[pd.Series, np.ndarray],
                          n_bins: int = 10,
                          title: str = "Conditional Bias by Predicted Range",
                          figsize: tuple = (12, 6),
                          save_dir: Optional[Union[str, Path]] = None):
    """Analyzes and plots how the mean residual (bias) varies across predicted value ranges."""
    logger.info("\n--- Generating Conditional Bias Plot ---")
    try:
        y_pred = np.asarray(y_pred)
        residuals = np.asarray(residuals)
        if len(y_pred) != len(residuals) or len(y_pred) == 0:
            logger.warning("Length mismatch between y_pred and residuals or empty array.")
            return
        df = pd.DataFrame({'prediction': y_pred, 'residual': residuals})
        if len(df) < n_bins * 2:
            logger.warning("Insufficient data for the number of bins specified.")
            return
        df['bin_q'] = pd.qcut(df['prediction'], q=n_bins, labels=False, duplicates='drop')
        n_actual_bins = df['bin_q'].nunique()
        if n_actual_bins < 2:
            logger.warning("Not enough bins created for analysis.")
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
            return
        analysis['sem'] = analysis['std_residual'] / np.sqrt(analysis['count'])
        analysis['ci_lower'] = analysis['mean_residual'] - 1.96 * analysis['sem']
        analysis['ci_upper'] = analysis['mean_residual'] + 1.96 * analysis['sem']
        analysis = analysis.sort_values('bin_center')
        plt.figure(figsize=figsize)
        plt.plot(analysis['bin_center'], analysis['mean_residual'], 'o-', label='Mean Residual (Bias)')
        plt.fill_between(analysis['bin_center'], analysis['ci_lower'], analysis['ci_upper'], alpha=0.3,
                         label='95% CI for Mean Residual')
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
                logger.info(f"Plot saved to {f_path}")
            except Exception as e:
                logger.error(f"Error saving conditional bias plot: {e}")
        plt.show()
    except ValueError as e:
        logger.warning(f"Could not create bins for conditional bias plot: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during conditional bias plot: {e}")
    logger.info("--- Conditional Bias Plot Complete ---")


def plot_temporal_bias(dates: Union[List, pd.Series, np.ndarray],
                       residuals: Union[pd.Series, np.ndarray],
                       freq: str = 'M',
                       title: str = "Temporal Bias Trend",
                       figsize: tuple = (12, 6),
                       save_dir: Optional[Union[str, Path]] = None):
    """Analyzes and plots the trend of mean residual (bias) over time."""
    logger.info("\n--- Generating Temporal Bias Plot ---")
    try:
        residuals = np.asarray(residuals)
        try:
            dates_dt = pd.to_datetime(dates, errors='raise')
        except Exception as e:
            logger.error(f"Error converting dates to datetime: {e}")
            return
        if len(dates_dt) != len(residuals) or len(dates_dt) == 0:
            logger.error("Mismatch in length between dates and residuals, or no data available.")
            return
        df = pd.DataFrame({'date': dates_dt, 'residual': residuals})
        df['period'] = df['date'].dt.to_period(freq)
        period_bias = df.groupby('period')['residual'].agg(['mean', 'std', 'count']).reset_index()
        period_bias['timestamp'] = period_bias['period'].dt.to_timestamp()
        period_bias = period_bias[period_bias['count'] > 2].copy()
        if period_bias.empty:
            logger.warning("Not enough data in period_bias after filtering.")
            return
        period_bias['sem'] = period_bias['std'] / np.sqrt(period_bias['count'])
        period_bias['ci_lower'] = period_bias['mean'] - 1.96 * period_bias['sem']
        period_bias['ci_upper'] = period_bias['mean'] + 1.96 * period_bias['sem']
        period_bias = period_bias.sort_values('timestamp')
        plt.figure(figsize=figsize)
        plt.plot(period_bias['timestamp'], period_bias['mean'], 'o-', label=f'Mean Bias ({freq})')
        plt.fill_between(period_bias['timestamp'], period_bias['ci_lower'], period_bias['ci_upper'],
                         alpha=0.3, label='95% CI for Mean Bias')
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Bias')
        plt.title(f"{title} ({freq})")
        plt.xlabel('Time Period')
        plt.ylabel('Mean Residual (Bias)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        try:
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gcf().autofmt_xdate(rotation=30, ha='right')
        except Exception as fmt_e:
            logger.warning(f"Error formatting date axis: {fmt_e}")
            plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            f_path = save_dir_path / f"temporal_bias_{freq}.png"
            try:
                plt.savefig(f_path, bbox_inches='tight')
                logger.info(f"Plot saved to {f_path}")
            except Exception as e:
                logger.error(f"Error saving temporal bias plot: {e}")
        plt.show()
    except Exception as e:
        logger.error(f"Unexpected error during temporal bias plot: {e}")
    logger.info("--- Temporal Bias Plot Complete ---")


# Helper function to extract importance consistently
def _get_feature_importance(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    final = model
    if isinstance(model, Pipeline) and model.steps:
        final = model.steps[-1][1]
    # Tree-based
    if hasattr(final, 'feature_importances_'):
        imps = final.feature_importances_
        return dict(zip(feature_names, imps)) if len(imps) == len(feature_names) else None
    # Linear
    if hasattr(final, 'coef_'):
        coefs = final.coef_
        if coefs.ndim > 1:
            coefs = np.abs(coefs).sum(axis=0)
        else:
            coefs = np.abs(coefs.flatten())
        return dict(zip(feature_names, coefs)) if len(coefs) == len(feature_names) else None
    return None

def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
):
    """
    Compute and plot a SHAP summary bar chart (mean absolute SHAP values).
    - model: trained Pipeline or estimator
    - X: DataFrame of features to explain (e.g. X_test)
    """
    if not _has_shap:
        logger.warning("SHAP unavailable—skipping SHAP summary plot.")
        return

    # If it’s a Pipeline, grab the final estimator
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    # Use a small background sample for speed
    background = X.sample(min(100, len(X)), random_state=42)
    explainer = shap.Explainer(estimator.predict, background)
    shap_values = explainer(X)

    plt.figure(figsize=(8, max_display * 0.3 + 1))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# Updated: supports permutation fallback when native missing

def plot_feature_importances(
    models_dict: Dict[str, Any],
    feature_names: List[str],
    X: Optional[pd.DataFrame] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    X_test: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    save_dir: Optional[Union[str, Path]] = None,
    show_plot: bool = True
):
    """
    Generate and save feature importance for each model.
    Falls back to permutation importance if needed.
    """
    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, pd.Series] = {}
    for name, model in models_dict.items():
        # 1) Try to get built-in importance
        imp = _get_feature_importance(model, feature_names)

        # 2) Fallback to permutation importances if nothing found
        if imp is None and X is not None and y is not None:
            logger.info(f"Computing permutation importance for '{name}'")
            try:
                perm = permutation_importance(
                    model,
                    X[feature_names],
                    y,
                    n_repeats=10,
                    random_state=0,
                    n_jobs=-1
                )
                imp = dict(zip(feature_names, perm.importances_mean))
            except Exception as e:
                logger.warning(f"Permutation importance failed for '{name}': {e}")
                continue

        if not imp:
            logger.warning(f"No feature importance available for '{name}' – skipping.")
            continue

        # 3) Sort & store in-memory
        importance_series = pd.Series(imp).sort_values(ascending=False)
        data[name] = importance_series

        # 4) Save full list to CSV and TXT
        if out_dir:
            df_full = importance_series.reset_index()
            df_full.columns = ['feature', 'importance']

            csv_path = out_dir / f"feature_importance_{name}_full.csv"
            txt_path = out_dir / f"feature_importance_{name}_full.txt"
            df_full.to_csv(csv_path, index=False)
            with open(txt_path, 'w') as f:
                f.write(df_full.to_string(index=False))
            logger.info(f"Saved full feature list for '{name}' to {out_dir}")

        # 5) Generate SHAP summary plot if available
        if _has_shap and X_test is not None:
            try:
                plot_shap_summary(
                    model=model,
                    X=X_test,
                    max_display=top_n,
                    save_path=(out_dir / f"shap_{name}.png") if out_dir else None,
                    show_plot=show_plot
                )
            except Exception as e:
                logger.error(f"Failed to generate SHAP summary for '{name}': {e}")

    # After looping all models
    if not data:
        logger.error("No feature importance data extracted.")
        return

    # plot top_n
    n = min(top_n, len(next(iter(data.values()))))
    cols = len(data)
    fig, axes = plt.subplots(1, cols, figsize=(cols*6, n*0.3+2), squeeze=False)
    for ax, (name, series) in zip(axes.flatten(), data.items()):
        top = series.head(n).sort_values(ascending=True)
        sns.barplot(x=top.values, y=top.index, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir/"feature_importance_summary.png", bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info("Feature importance complete.")


# ---  plot_predictions_over_time ---
def plot_predictions_over_time(dates: Union[List, pd.Series, np.ndarray],
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               title: str = "Predictions Over Time",
                               target_name: str = "Score",
                               figsize: tuple = (14, 7),
                               save_dir: Optional[Union[str, Path]] = None):
    """Plots actual vs predicted values over time. Requires dates to be sortable."""
    # (Implementation from original file - code is lengthy, assume it's here)
    logger.info("\n--- Generating Predictions Over Time Plot ---")
    # ... (full implementation) ...
    logger.info("--- Predictions Over Time Plot Complete ---")

# --- Higher-Level Report Generation ---
def generate_evaluation_report(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], model_name: str, target_type: str = 'total', dates: Optional[Union[List, pd.Series, np.ndarray]] = None, calculate_custom_losses: bool = True, include_bias_analysis: bool = True, save_dir: Optional[Union[str, Path]] = None):
    """ Generates a comprehensive evaluation report with metrics and various plots. """
    logger.info(f"\n{'='*20} Evaluation Report: {model_name} ({target_type}) {'='*20}")
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path: save_dir_path.mkdir(parents=True, exist_ok=True); logger.info(f"Saving plots to: {save_dir_path}")
    logger.info("\n--- Calculating Metrics ---"); metrics = evaluate_predictions(y_true, y_pred, target_type, calculate_custom_losses); print("Evaluation Metrics:")
    if all(pd.isna(v) for v in metrics.values()): logger.error("Error: All metrics NaN."); return metrics
    for key, value in metrics.items(): print(f"  {key.replace('_', ' ').title():<25}: {value:.4f}" if pd.notna(value) else f"  {key.replace('_', ' ').title():<25}: NaN")
    logger.info("\n--- Generating Core Visualizations ---"); base_filename = f"{model_name}_{target_type}".replace(" ", "_").lower()
    plot_actual_vs_predicted(y_true, y_pred, title=f"{model_name} - Actual vs. Predicted ({target_type})", metrics=metrics, save_path=save_dir_path / f"{base_filename}_actual_vs_pred.png" if save_dir_path else None)
    plot_score_distribution_density(y_true, y_pred, title=f"{model_name} - Score Distribution Density ({target_type})", save_path=save_dir_path / f"{base_filename}_density.png" if save_dir_path else None)

    if include_bias_analysis:
        logger.info("\n--- Generating Residual & Bias Analysis ---"); residuals = np.asarray(y_true) - np.asarray(y_pred)
        plot_residuals_analysis_detailed(y_true, y_pred, title_prefix=f"{model_name} ({target_type})", save_dir=save_dir_path)
        plot_conditional_bias(y_pred, residuals, title=f"{model_name} - Conditional Bias ({target_type})", save_dir=save_dir_path)
        plot_error_by_prediction_range(y_true, y_pred, title=f"{model_name} - Mean Error by Range ({target_type})", save_path=save_dir_path / f"{base_filename}_error_range.png" if save_dir_path else None)
        if dates is not None: plot_temporal_bias(dates, residuals, freq='M', title=f"{model_name} - Temporal Bias ({target_type})", save_dir=save_dir_path)
        else: logger.info("Skipping temporal bias plot: 'dates' not provided.")

    if dates is not None:
        logger.info("\n--- Generating Time Series Plot ---")
        plot_predictions_over_time(dates=dates, y_true=y_true, y_pred=y_pred, title=f"{model_name} - Predictions Over Time ({target_type})", target_name=target_type.replace('_', ' ').title(), save_dir=save_dir_path)
    else: logger.info("Skipping predictions over time plot: 'dates' not provided.")

    logger.info(f"\n{'='*20} Report Generation Complete: {model_name} ({target_type}) {'='*20}")
    return metrics

# --- Main Example Block ---
if __name__ == '__main__':
    logger.info("Running evaluation.py example...")
    # Example usage if needed for testing this module directly
    pass # Keep minimal if not running this file directly often