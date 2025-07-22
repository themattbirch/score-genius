# backend/nfl_features/advanced.py

"""Win‑loss *form* and current streak features for NFL teams.

The goal is to quantify recent momentum without leaking future information.
We work from a **historical* games DataFrame (all past results) and attach
rolling win‑percentage and streak metrics to an *upcoming* games DataFrame.

Key steps:
1. Convert the historical schedule to **long format** so each row represents a
   single team in a single game.
2. Compute outcome (win=1, loss=‑1, tie=0) and build a streak counter.
3. For each team, roll over the previous ``lookback_window`` games *excluding*
   the current match (via ``shift(1)``) to produce leakage‑free win%.
4. Merge the resulting features onto upcoming games for both home and away
   sides, then derive differentials.

Output columns (for ``lookback_window=5``):
- ``home_form_win_pct_5`` / ``away_…``
- ``home_current_streak`` / ``away_…`` (positive = winning streak, negative = losing)
- ``form_win_pct_5_diff``
- ``current_streak_diff``

Missing values (early season or expansion teams) are filled with ``utils.DEFAULTS``
and flagged if desired.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import utils
from .utils import (
    DEFAULTS,
    normalize_team_name,
    prefix_columns,
    safe_divide,  # we will use the updated alias with default_val
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "transform",  # existing form/streak transformer
    "compute_advanced_metrics",  # NEW – required by unit tests
]

# --------------------------------------------------------------------------- #
#  Existing form/streak helpers (unchanged)
# --------------------------------------------------------------------------- #

def _prepare_long_format(historical: pd.DataFrame) -> pd.DataFrame:  # unchanged
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    hist["home_team_norm"] = hist["home_team_norm"].apply(normalize_team_name)
    hist["away_team_norm"] = hist["away_team_norm"].apply(normalize_team_name)

    home_games = hist.rename(
        columns={
            "home_team_norm": "team",
            "away_team_norm": "opponent",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    )
    away_games = hist.rename(
        columns={
            "away_team_norm": "team",
            "home_team_norm": "opponent",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    )
    long_df = pd.concat([home_games, away_games], ignore_index=True)
    return long_df.sort_values(["team", "game_date"])


def _compute_outcomes(long_df: pd.DataFrame) -> pd.DataFrame:  # unchanged
    long_df = long_df.copy()
    long_df["outcome"] = np.select(
        [long_df["team_score"] > long_df["opp_score"], long_df["team_score"] < long_df["opp_score"]],
        [1, -1],
        default=0,
    )

    long_df["_streak_len"] = (
        long_df.groupby("team")["outcome"].apply(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    )
    long_df["current_streak"] = long_df["_streak_len"] * long_df["outcome"]
    long_df.drop(columns="_streak_len", inplace=True)
    return long_df


# --------------------------------------------------------------------------- #
#  PUBLIC API – form/streak transformer (unchanged)
# --------------------------------------------------------------------------- #

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    lookback_window: int = 5,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Attach leakage‑free *form* features to ``games``. (Existing docstring)"""
    # <existing implementation unchanged>
    if historical_df is None or historical_df.empty:
        logger.warning("form: No historical data – defaulting all form features.")
        base = games[["game_id"]].copy()
        for side in ("home", "away"):
            base[f"{side}_form_win_pct_{lookback_window}"] = DEFAULTS["form_win_pct"]
            base[f"{side}_current_streak"] = DEFAULTS["current_streak"]
            if flag_imputations:
                base[f"{side}_form_imputed"] = 1
                base[f"{side}_streak_imputed"] = 1
        base[f"form_win_pct_{lookback_window}_diff"] = 0.0
        base["current_streak_diff"] = 0
        return base

    long_df = _compute_outcomes(_prepare_long_format(historical_df))

    long_df[f"form_win_pct_{lookback_window}"] = (
        long_df.groupby("team")["outcome"].shift(1).rolling(window=lookback_window, min_periods=1).apply(lambda x: (x > 0).mean(), raw=True)
    )

    feat_cols = [
        "team",
        "game_date",
        f"form_win_pct_{lookback_window}",
        "current_streak",
    ]
    features_long = long_df[feat_cols]

    upcoming = games.copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"])
    upcoming.sort_values("game_date", inplace=True)

    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        tmp = pd.merge_asof(
            upcoming,
            features_long.rename(columns={"team": team_col}),
            on="game_date",
            by=team_col,
            direction="backward",
        )
        merged[side] = prefix_columns(
            tmp[["game_id", f"form_win_pct_{lookback_window}", "current_streak"]],
            f"{side}",
            exclude=["game_id"],
        )

    result = merged["home"].merge(merged["away"], on="game_id")

    fills = {
        f"home_form_win_pct_{lookback_window}": DEFAULTS["form_win_pct"],
        f"away_form_win_pct_{lookback_window}": DEFAULTS["form_win_pct"],
        "home_current_streak": DEFAULTS["current_streak"],
        "away_current_streak": DEFAULTS["current_streak"],
    }
    for col, dval in fills.items():
        if flag_imputations:
            result[f"{col}_imputed"] = result[col].isna().astype(int)
        result[col] = result[col].fillna(dval)

    result[f"form_win_pct_{lookback_window}_diff"] = (
        result[f"home_form_win_pct_{lookback_window}"] - result[f"away_form_win_pct_{lookback_window}"]
    )
    result["current_streak_diff"] = result["home_current_streak"] - result["away_current_streak"]

    return result


# --------------------------------------------------------------------------- #
#  NEW: Team‑game advanced box‑score metrics
# --------------------------------------------------------------------------- #

def _exists(df: pd.DataFrame, cols: set[str]) -> bool:
    """Quick helper to check all columns present."""
    return cols.issubset(df.columns)


def compute_advanced_metrics(
    box_scores: pd.DataFrame,
    *,
    flag_imputations: bool = True,
) -> pd.DataFrame:
    """Compute per‑team advanced box‑score metrics.

    Returns a DataFrame with prefixed ``adv_`` rate / efficiency columns that
    match the unit‑test expectations, plus a raw ``possession_seconds`` helper
    column used in some downstream logic.
    """

    if box_scores.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    out = box_scores[["game_id", "team_id"]].copy()

    def _add_rate(col_out: str, numer, denom, default_key: str | None = None, *, multiply: float | None = None):
        """Add rate column with imputation option.

        Parameters
        ----------
        col_out : str
            Output column name.
        numer / denom : pd.Series
            Numerator and denominator.
        default_key : str | None
            Key in ``DEFAULTS`` for the fallback value. If ``None`` use 0.0.
        multiply : float | None
            If provided, multiply the computed rate by this factor (e.g. 100).
        """
        default_val = DEFAULTS.get(default_key, 0.0) if default_key else 0.0
        rate = safe_divide(numer, denom, default_val=default_val)
        if multiply is not None:
            rate = rate * multiply
        out[col_out] = rate
        if flag_imputations:
            out[f"{col_out}_imputed"] = (denom == 0) | denom.isna() | numer.isna()

    # ------------------------------------------------------------------
    # simple rate metrics (prefixed with adv_)
    # ------------------------------------------------------------------
    if _exists(box_scores, {"third_down_made", "third_down_attempts"}):
        _add_rate("adv_third_down_pct", box_scores["third_down_made"], box_scores["third_down_attempts"], None)

    if _exists(box_scores, {"fourth_down_made", "fourth_down_attempts"}):
        _add_rate("adv_fourth_down_pct", box_scores["fourth_down_made"], box_scores["fourth_down_attempts"], None)

    if _exists(box_scores, {"red_zone_made", "red_zone_att"}):
        _add_rate("adv_red_zone_pct", box_scores["red_zone_made"], box_scores["red_zone_att"], None)

    # ------------------------------------------------------------------
    # yards / play & rushing yards / rush (keep previous names but also add adv prefix aliases if required)
    # ------------------------------------------------------------------
    if _exists(box_scores, {"yards_total", "plays_total"}):
        _add_rate("yards_per_play", box_scores["yards_total"], box_scores["plays_total"], "yards_per_play_avg")
        out.rename(columns={"yards_per_play": "adv_yards_per_play"}, inplace=True)
    if _exists(box_scores, {"rushings_total", "rushings_attempts"}):
        _add_rate("rushings_yards_per_rush", box_scores["rushings_total"], box_scores["rushings_attempts"], "rushings_yards_per_rush_avg")
        out.rename(columns={"rushings_yards_per_rush": "adv_rushings_yards_per_rush"}, inplace=True)

    # ------------------------------------------------------------------
    # turnovers per game (copy raw) & turnover rate per play
    # ------------------------------------------------------------------
    if "turnovers_total" in box_scores.columns:
        out["turnovers_total"] = box_scores["turnovers_total"].fillna(DEFAULTS["turnovers_per_game_avg"])
        if flag_imputations:
            out["turnovers_total_imputed"] = box_scores["turnovers_total"].isna()
    if _exists(box_scores, {"turnovers_total", "plays_total"}):
        _add_rate("adv_turnover_rate_per_play", box_scores["turnovers_total"], box_scores["plays_total"], None)

    # ------------------------------------------------------------------
    # derived from total_drives & penalties
    # ------------------------------------------------------------------
    if _exists(box_scores, {"yards_total", "total_drives"}):
        _add_rate("adv_yards_per_drive", box_scores["yards_total"], box_scores["total_drives"], None)

    if _exists(box_scores, {"penalty_yards", "penalties"}):
        _add_rate("adv_yards_per_penalty", box_scores["penalty_yards"], box_scores["penalties"], None)

    # ------------------------------------------------------------------
    # possession‑time based metrics
    # ------------------------------------------------------------------
    if "possession_time" in box_scores.columns:
        def _to_seconds(val):
            if pd.isna(val):
                return np.nan
            try:
                mins, secs = str(val).split(":")
                return int(mins) * 60 + int(secs)
            except Exception:
                return np.nan
        possession_seconds = box_scores["possession_time"].apply(_to_seconds)
        out["possession_seconds"] = possession_seconds
        if _exists(box_scores, {"plays_total"}):
            # plays per minute = plays / (possession_seconds / 60)
            denom = possession_seconds / 60.0
            _add_rate("adv_plays_per_minute", box_scores["plays_total"], denom, None)

    # ------------------------------------------------------------------
    # points per 100 yards (optional)
    # ------------------------------------------------------------------
    if _exists(box_scores, {"team_score", "yards_total"}):
        _add_rate("adv_points_per_100_yards", box_scores["team_score"], box_scores["yards_total"], "points_for_avg", multiply=100)

    return out
