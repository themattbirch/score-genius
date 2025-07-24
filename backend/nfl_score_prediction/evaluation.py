# backend/nfl_score_prediction/evaluation.py
"""
evaluation.py - NFL Model Evaluation Utilities

Provides:
  • Metric calculation (MAE, RMSE, R², etc.)
  • Plots: actual vs predicted, residuals, feature importances
  • Text report generator for final ensemble results
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------- #
# Logging / Matplotlib
# ----------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

plt.style.use("fivethirtyeight")


# ----------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------- #

def _clean_arrays(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaNs in predictions; align shapes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def calculate_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Return MAE, MSE, RMSE, R². NaN-safe for predictions.
    """
    try:
        yt, yp = _clean_arrays(y_true, y_pred)
        if yt.size == 0:
            return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan}

        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        r2 = r2_score(yt, yp) if yt.size >= 2 else np.nan
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    except Exception as e:
        logger.error("Metric calc error: %s", e)
        return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan}


# ----------------------------------------------------------------------------- #
# Plotting helpers
# ----------------------------------------------------------------------------- #

def plot_actual_vs_predicted(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    title: str = "Actual vs Predicted",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 8),
):
    """
    Scatter of actual vs predicted with y=x reference.
    """
    yt, yp = _clean_arrays(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(yt, yp, alpha=0.55)
    lim_low = min(yt.min(), yp.min()) - 2
    lim_high = max(yt.max(), yp.max()) + 2
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "r--", lw=1, label="Ideal (y=x)")
    ax.set(
        xlim=(lim_low, lim_high),
        ylim=(lim_low, lim_high),
        xlabel="Actual",
        ylabel="Predicted",
        title=title,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_or_show(fig, save_path)


def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    title: str = "Residuals (Predicted - Actual)",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 5),
    bins: int = 30,
):
    """
    Histogram of residuals.
    """
    yt, yp = _clean_arrays(y_true, y_pred)
    residuals = yp - yt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(residuals, bins=bins, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel="Residual", ylabel="Count")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def _extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[pd.Series]:
    """
    Handles sklearn pipelines & raw estimators.
    Returns None if no importances available.
    """
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(getattr(est, "coef_"))
    else:
        return None

    return pd.Series(vals, index=feature_names).sort_values(ascending=False)


def plot_feature_importances(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Bar plot of top-N importances. Skips if model doesn't expose importances/coef.
    """
    importances = _extract_feature_importances(model, feature_names)
    if importances is None:
        logger.warning("No feature importances for '%s'.", model_name)
        return

    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index[::-1], top.values[::-1])  # reversed for top at top
    ax.set(title=f"Top {top_n} Features – {model_name}", xlabel="Importance", ylabel="Feature")
    fig.tight_layout()

    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[Path]):
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved → %s", save_path)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Report
# ----------------------------------------------------------------------------- #

def generate_evaluation_report(
    results_df: pd.DataFrame,
    overall_mae: float,
    report_path: Path,
    *,
    save_metrics_csv: Optional[Path] = None,
):
    """
    Write a text report summarizing ensemble performance.

    results_df columns required:
        actual_home_score, predicted_home_score,
        actual_away_score, predicted_away_score
    """
    logger.info("Writing evaluation report → %s", report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Derived arrays
    ah, ph = results_df["actual_home_score"], results_df["predicted_home_score"]
    aa, pa = results_df["actual_away_score"], results_df["predicted_away_score"]

    actual_margin = ah - aa
    predicted_margin = ph - pa
    actual_total = ah + aa
    predicted_total = ph + pa

    # Betting-ish metric: winner correctness
    winner_correct = float(np.mean(np.sign(actual_margin) == np.sign(predicted_margin))) * 100.0

    # Metrics dict
    metrics = {
        "Home Score": calculate_regression_metrics(ah, ph),
        "Away Score": calculate_regression_metrics(aa, pa),
        "Point Margin": calculate_regression_metrics(actual_margin, predicted_margin),
        "Total Points": calculate_regression_metrics(actual_total, predicted_total),
    }

    with open(report_path, "w") as f:
        f.write("=" * 66 + "\n")
        f.write("   NFL ENSEMBLE EVALUATION REPORT\n")
        f.write(f"   Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 66 + "\n\n")
        f.write(f"Overall MAE (primary KPI): {overall_mae:.4f}\n")
        f.write(f"Predicted Winner Accuracy: {winner_correct:.2f}%\n\n")
        f.write("-" * 66 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("-" * 66 + "\n")
        for name, m in metrics.items():
            f.write(f"\n{name}:\n")
            f.write(f"  MAE : {m['mae']:.4f}\n")
            f.write(f"  RMSE: {m['rmse']:.4f}\n")
            f.write(f"  R^2 : {m['r2']:.4f}\n")

    logger.info("Report saved.")

    if save_metrics_csv:
        rows = []
        for name, m in metrics.items():
            row = {"target": name, **m}
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_metrics_csv, index=False)
        logger.info("Metrics CSV saved → %s", save_metrics_csv)


# ----------------------------------------------------------------------------- #
# Quick self-test
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running quick evaluation self-test...")
    n = 200
    rng = np.random.default_rng(42)
    ah = rng.integers(0, 50, n)
    aa = rng.integers(0, 50, n)
    ph = np.clip(ah + rng.normal(0, 7, n), 0, 60)
    pa = np.clip(aa + rng.normal(0, 7, n), 0, 60)

    df = pd.DataFrame(
        {
            "actual_home_score": ah,
            "predicted_home_score": ph,
            "actual_away_score": aa,
            "predicted_away_score": pa,
        }
    )

    overall = 0.5 * (
        mean_absolute_error(ah, ph) + mean_absolute_error(aa, pa)
    )

    out_dir = Path("temp_eval")
    generate_evaluation_report(df, overall, out_dir / "eval_report.txt", save_metrics_csv=out_dir / "metrics.csv")
    plot_actual_vs_predicted(ah, ph, "Home: Actual vs Predicted", out_dir / "home_avp.png")
    plot_residuals(ah, ph, "Home Residuals", out_dir / "home_resid.png")
    logger.info("Self-test complete.")
