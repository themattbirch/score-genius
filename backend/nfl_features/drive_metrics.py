# backend/nfl_features/drive_metrics.py

"""Drive‑level efficiency metrics for NFL team game logs.

This transformer enriches a *team‑level* box‑score DataFrame (one row per
team‑game) with per‑drive efficiency statistics. By normalizing totals by
``total_drives``, we capture pace and effectiveness in a way that is
comparable across teams and weeks.

The resulting columns are intended for *rolling aggregation* in the database
(view ``mv_nfl_recent_form``) but can also be used stand‑alone for single‑game
analysis.
"""
from typing import List
import logging

import pandas as pd

from .utils import compute_rate, DEFAULTS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__: List[str] = ["compute_drive_metrics"]

# Required and optional columns for validation
_REQUIRED = {
    "team_id",
    "game_id",
    "total_drives",
    "yards_total",
    "plays_total",
    "turnovers_total",
}
_OPTIONAL = {"team_score", "points_against_total"}


def compute_drive_metrics(
    box_scores_df: pd.DataFrame,
    *,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Return a copy of the input DataFrame with drive‑based metrics appended."""
    if box_scores_df.empty:
        logger.warning("drive_metrics: received empty DataFrame; returning unchanged.")
        return box_scores_df.copy()

    missing = _REQUIRED - set(box_scores_df.columns)
    if missing:
        logger.error("drive_metrics: missing required columns: %s", missing)
        return box_scores_df.copy()

    df = box_scores_df.copy()

    # Core per‑drive and per-play metrics
    metrics = {
        "drive_yards_per_drive": ("yards_total", "total_drives"),
        "drive_plays_per_drive": ("plays_total", "total_drives"),
        "drive_turnovers_per_drive": ("turnovers_total", "total_drives"),
        "drive_yards_per_play": ("yards_total", "plays_total"),
    }
    if "team_score" in df.columns:
        metrics["drive_points_per_drive"] = ("team_score", "total_drives")

    for new_col, (numer, denom) in metrics.items():
        df[new_col] = compute_rate(df[numer], df[denom], fill=DEFAULTS.get(new_col, 0.0))
        if flag_imputations:
            imputed_flag = (
                df[numer].isna() | df[denom].isna() | (df[denom] == 0)
            ).astype(int)
            df[f"{new_col}_imputed"] = imputed_flag

    # Defensive metrics: points allowed per drive
    if {"points_against_total", "total_drives"}.issubset(df.columns):
        df["points_allowed_per_drive"] = compute_rate(
            df["points_against_total"], df["total_drives"],
            fill=DEFAULTS.get("points_allowed_per_drive", 0.0)
        )
        if flag_imputations:
            imputed_flag = (
                df["points_against_total"].isna() | df["total_drives"].isna() | (df["total_drives"] == 0)
            ).astype(int)
            df["points_allowed_per_drive_imputed"] = imputed_flag

    logger.debug(
        "drive_metrics: added %d new columns", len(df.columns) - len(box_scores_df.columns)
    )
    return df
