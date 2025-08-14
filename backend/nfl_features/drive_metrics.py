# backend/nfl_features/drive_metrics.py
"""
Drive-level efficiency metrics for NFL team game logs.

This transformer enriches a *team-level* box-score DataFrame (one row per
team-game) with *drive_* prefixed, leakage-safe efficiency features for both
offense and defense. By normalizing totals by ``total_drives`` and (when
available) opponent totals, we capture pace and effectiveness in a way that is
comparable across teams and weeks.

The returned frame preserves the input shape (team-game rows) and simply
appends columns. A separate game-level transform can aggregate these to
home_/away_/total_/diff per game_id for model merges.

Idempotence contract:
- No temporary columns leak to the output.
- Function can be applied repeatedly to the same input without changing results.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import logging

import pandas as pd
import numpy as np

from .utils import compute_rate, DEFAULTS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__: List[str] = ["compute_drive_metrics"]

# ---------------------------------------------------------------------
# Required / optional schema
# ---------------------------------------------------------------------
_REQUIRED = {
    "team_id",
    "game_id",
    "total_drives",
    "yards_total",
    "plays_total",
    "turnovers_total",
}
# We detect many optional aliases below. These are the most common:
_OPTIONAL_HINTS: Dict[str, Tuple[str, ...]] = {
    # scoring & points
    "team_score": ("team_score", "points_for_total", "points_scored_total"),
    "points_against_total": ("points_against_total", "points_allowed_total"),
    # red zone
    "red_zone_trips": ("red_zone_trips", "rz_trips", "rza"),
    "red_zone_tds": ("red_zone_tds", "rz_tds"),
    # third down
    "third_down_att": ("third_down_att", "third_down_attempts", "third_att"),
    "third_down_conv": ("third_down_conv", "third_down_conversions", "third_conv"),
    # fourth down
    "fourth_down_att": ("fourth_down_att", "fourth_down_attempts", "fourth_att"),
    "fourth_down_conv": ("fourth_down_conv", "fourth_down_conversions", "fourth_conv"),
    # penalties & sacks (offense perspective)
    "penalties": ("penalties", "penalty_count", "penalties_total"),
    "penalty_yards": ("penalty_yards", "penalties_yards", "penalty_yards_total"),
    "sacks_allowed": ("sacks_allowed", "sacks_taken"),
    # explosiveness
    "explosive_plays_20_plus": (
        "explosive_plays_20_plus",
        "plays_20_plus_total",
        "pass_20_plus_total",
        "rush_20_plus_total",
    ),
    # pace / time
    "possession_time_seconds": ("possession_time_seconds", "time_of_possession_sec"),
    # defense (if present on same row; otherwise derived via opponent merge)
    "yards_against_total": ("yards_against_total", "yards_allowed_total"),
    "plays_against_total": ("plays_against_total", "plays_allowed_total"),
    "turnovers_forced_total": ("turnovers_forced_total", "takeaways_total"),
    "sacks_made": ("sacks_made", "sacks_for"),
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _find_first(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    if s.empty or not np.issubdtype(s.dtype, np.number):
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    if pd.isna(lo) or pd.isna(hi):
        return s
    return s.clip(lo, hi)

def _add_rate(
    df: pd.DataFrame,
    numer_col: str | None,
    denom_col: str | None,
    out_col: str,
    *,
    flag_imputations: bool,
) -> None:
    """Compute numer/denom → out_col using compute_rate, set *_imputed flag if needed."""
    if numer_col is None or denom_col is None or numer_col not in df.columns or denom_col not in df.columns:
        # Create column with default fill (0.0) for stability if missing
        df[out_col] = DEFAULTS.get(out_col, 0.0)
        if flag_imputations:
            df[f"{out_col}_imputed"] = 1
        return

    df[out_col] = compute_rate(df[numer_col], df[denom_col], fill=DEFAULTS.get(out_col, 0.0))
    if flag_imputations:
        df[f"{out_col}_imputed"] = (
            df[numer_col].isna() | df[denom_col].isna() | (df[denom_col] == 0)
        ).astype(int)

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

def _dedupe_to_two_rows_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically enforce ≤ 2 rows per game_id for opponent pairing.

    Strategy:
      1) Sort by a stable order:
         - Prefer 'updated_at' if present, else by 'team_id' then index.
      2) Drop duplicate (game_id, team_id) keeping the last (most recent).
      3) Within each game_id, keep the last two rows (or all if <2).
    """
    order_cols = ["game_id"]
    if "updated_at" in df.columns:
        order_cols.append("updated_at")
    order_cols.append("team_id")
    df_sorted = df.sort_values(order_cols, kind="stable")

    # Deduplicate duplicate (game_id, team_id), keep last occurrence
    if {"game_id", "team_id"}.issubset(df_sorted.columns):
        df_sorted = df_sorted.drop_duplicates(["game_id", "team_id"], keep="last")

    # Enforce ≤ 2 rows per game_id deterministically
    limited = df_sorted.groupby("game_id", group_keys=False).apply(lambda g: g.tail(2))
    limited = limited.reset_index(drop=True)
    return limited

def _derive_opponent_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an opponent slice for quick per-game joins.
    Returns a frame keyed by (game_id, team_id) with opponent metrics renamed 'opp_*'.
    Assumes at most two teams per game_id; if more, deterministically trims to two.
    """
    # Choose a minimal set we might need from the opponent row
    cols_of_interest = [
        "game_id", "team_id",
        "total_drives", "yards_total", "plays_total", "turnovers_total",
    ]
    for key, cand in _OPTIONAL_HINTS.items():
        c = _find_first(df, cand)
        if c is not None:
            cols_of_interest.append(c)

    opp = df.loc[:, [c for c in cols_of_interest if c in df.columns]].copy()

    # Enforce ≤ 2 rows per game deterministically (handles noisy inputs)
    opp = _dedupe_to_two_rows_per_game(opp)

    # Stable order for pairing
    opp = opp.sort_values(["game_id", "team_id"], kind="stable").copy()

    # Pair rows 0 <-> 1 within each game using a local row index
    opp["__row_in_game__"] = opp.groupby("game_id").cumcount()
    opp["__opp_row_in_game__"] = 1 - opp["__row_in_game__"]

    opp_idx = opp.set_index(["game_id", "__row_in_game__"])
    opp_partner = opp.set_index(["game_id", "__opp_row_in_game__"]).add_prefix("opp_")
    joined = opp_idx.join(opp_partner, how="left").reset_index()

    # Keep only columns from the opponent perspective plus keys
    keep = ["game_id", "team_id"] + [c for c in joined.columns if c.startswith("opp_")]
    out = joined[keep].copy()

    # Unique (game_id, team_id)
    out = out.drop_duplicates(["game_id", "team_id"], keep="last")
    return out

# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def compute_drive_metrics(
    box_scores_df: pd.DataFrame,
    *,
    flag_imputations: bool = True,
    winsorize: bool = True,
    winsor_limits: Tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    """
    Return a copy of the input team-game DataFrame with drive-based metrics appended.

    - Emits ONLY `drive_*` prefixed features (and associated `*_imputed` flags),
      plus `drive_offense/defense_samples` and low-sample flags.
    - Computes both offensive and defensive per-drive/per-play where possible.
    - Optionally winsorizes numeric drive rates to stabilize outliers.
    - Idempotent: safe to call repeatedly on the same input.
    """
    if box_scores_df.empty:
        logger.warning("drive_metrics: received empty DataFrame; returning unchanged.")
        return box_scores_df.copy()

    missing = _REQUIRED - set(box_scores_df.columns)
    if missing:
        logger.info("drive_metrics: missing required columns %s; returning unchanged.", missing)
        return box_scores_df.copy()

    # Work on a copy; track original columns for cleanup
    df = box_scores_df.copy()
    original_cols = set(df.columns)

    # Normalize numeric types for denominators to avoid string "0" surprises
    _coerce_numeric(df, ["total_drives", "plays_total"])

    # Resolve common optional columns via alias table
    opt: Dict[str, str | None] = {k: _find_first(df, v) for k, v in _OPTIONAL_HINTS.items()}

    # ---- Possession time: minutes → seconds alias (temp col), if needed ----
    # Many upstream feeds provide `possession_time` in MINUTES. Map it to seconds.
    temp_poss_col = None
    if opt["possession_time_seconds"] is None:
        # Try minutes column
        if "possession_time" in df.columns:
            temp_poss_col = "__possession_time_seconds__"
            df[temp_poss_col] = pd.to_numeric(df["possession_time"], errors="coerce") * 60.0
            opt["possession_time_seconds"] = temp_poss_col  # use this for pace metrics

    # -----------------------------------------------------------------
    # Offensive per-drive / per-play
    # -----------------------------------------------------------------
    _add_rate(df, "yards_total", "total_drives", "drive_yards_per_drive", flag_imputations=flag_imputations)
    _add_rate(df, "plays_total", "total_drives", "drive_plays_per_drive", flag_imputations=flag_imputations)
    _add_rate(df, "turnovers_total", "total_drives", "drive_turnovers_per_drive", flag_imputations=flag_imputations)
    _add_rate(df, "yards_total", "plays_total", "drive_yards_per_play", flag_imputations=flag_imputations)

    # Points per drive (strong for totals)
    _add_rate(df, opt["team_score"], "total_drives", "drive_points_per_drive", flag_imputations=flag_imputations)

    # Scoring/TD/FG rates if available
    td_candidates = _find_first(df, ("touchdowns_total", "offensive_touchdowns", "td_total", "td"))
    fg_candidates = _find_first(df, ("field_goals_made", "fg_made", "fg"))
    _add_rate(df, td_candidates, "total_drives", "drive_td_rate", flag_imputations=flag_imputations)
    _add_rate(df, fg_candidates, "total_drives", "drive_fg_rate", flag_imputations=flag_imputations)

    # Red zone efficiency
    _add_rate(df, opt["red_zone_tds"], opt["red_zone_trips"], "drive_red_zone_td_pct", flag_imputations=flag_imputations)
    _add_rate(df, opt["red_zone_trips"], "total_drives", "drive_red_zone_trips_per_drive", flag_imputations=flag_imputations)

    # Third/Fourth down (conversion pct and attempts per drive)
    _add_rate(df, opt["third_down_conv"], opt["third_down_att"], "drive_third_down_conv_pct", flag_imputations=flag_imputations)
    _add_rate(df, opt["third_down_att"], "total_drives", "drive_third_down_att_per_drive", flag_imputations=flag_imputations)

    _add_rate(df, opt["fourth_down_conv"], opt["fourth_down_att"], "drive_fourth_down_conv_pct", flag_imputations=flag_imputations)
    _add_rate(df, opt["fourth_down_att"], "total_drives", "drive_fourth_down_att_per_drive", flag_imputations=flag_imputations)

    # Penalties, sacks, explosiveness
    _add_rate(df, opt["penalties"], "total_drives", "drive_penalties_per_drive", flag_imputations=flag_imputations)
    _add_rate(df, opt["penalty_yards"], "total_drives", "drive_penalty_yards_per_drive", flag_imputations=flag_imputations)
    _add_rate(df, opt["sacks_allowed"], "total_drives", "drive_sacks_allowed_per_drive", flag_imputations=flag_imputations)

    # Explosive plays (≥20yd) rate per play if any candidate exists
    exp_col = opt["explosive_plays_20_plus"]
    _add_rate(df, exp_col, "plays_total", "drive_explosive_play_rate", flag_imputations=flag_imputations)

    # Pace proxies (seconds per play/drive)
    _add_rate(df, opt["possession_time_seconds"], "plays_total", "drive_seconds_per_play", flag_imputations=flag_imputations)
    _add_rate(df, opt["possession_time_seconds"], "total_drives", "drive_seconds_per_drive", flag_imputations=flag_imputations)

    # Reliability exposure
    df["drive_offense_samples"] = df["total_drives"].fillna(0).astype(float)
    df["drive_low_sample_off"] = (df["drive_offense_samples"] < 8).astype(int)

    # -----------------------------------------------------------------
    # Defensive mirrors (prefer explicit *_against_* columns; else derive via opponent row)
    # -----------------------------------------------------------------
    # Build an opponent slice once (cheap) for any metrics not present as *_against_*
    opp_slice = _derive_opponent_slice(df)

    def _ensure_col(name: str, against_name: str | None, fallback_from_opp: str | None) -> pd.Series | None:
        """
        Return a series for defensive counterpart:
          1) if explicit 'against' col exists → use it
          2) elif opponent slice has fallback_from_opp → use that
          3) else → None
        """
        if against_name and against_name in df.columns:
            return pd.to_numeric(df[against_name], errors="coerce")
        if fallback_from_opp and f"opp_{fallback_from_opp}" in opp_slice.columns:
            joined = df[["game_id", "team_id"]].merge(
                opp_slice[["game_id", "team_id", f"opp_{fallback_from_opp}"]],
                on=["game_id", "team_id"], how="left"
            )
            return pd.to_numeric(joined[f"opp_{fallback_from_opp}"], errors="coerce")
        return None

    # Defensive denominators (opponent drives/plays)
    opp_drives = _ensure_col("opp_total_drives", _find_first(df, ("opp_total_drives", "def_total_drives_allowed")), "total_drives")
    opp_plays  = _ensure_col("opp_plays_total", _find_first(df, ("plays_against_total", "plays_allowed_total")), "plays_total")

    # Defensive numerators
    yards_against = _ensure_col("yards_against_total", _OPTIONAL_HINTS["yards_against_total"] and _find_first(df, _OPTIONAL_HINTS["yards_against_total"]), "yards_total")
    points_against = _ensure_col("points_against_total", _find_first(df, ("points_against_total", "points_allowed_total")), "team_score")
    turnovers_forced = _ensure_col("turnovers_forced_total", _OPTIONAL_HINTS["turnovers_forced_total"] and _find_first(df, _OPTIONAL_HINTS["turnovers_forced_total"]), "turnovers_total")
    sacks_made = _ensure_col("sacks_made", _OPTIONAL_HINTS["sacks_made"] and _find_first(df, _OPTIONAL_HINTS["sacks_made"]), "sacks_allowed")

    # Now compute defensive per-drive/per-play where possible
    # If opponent denominators are missing, fall back to team denominators (approximate)
    denom_drives_def = opp_drives if opp_drives is not None else pd.to_numeric(df["total_drives"], errors="coerce")
    denom_plays_def  = opp_plays  if opp_plays  is not None else pd.to_numeric(df["plays_total"], errors="coerce")

    def _add_def_rate(numer: pd.Series | None, denom: pd.Series, out_col: str):
        if numer is None or denom is None:
            df[out_col] = DEFAULTS.get(out_col, 0.0)
            if flag_imputations:
                df[f"{out_col}_imputed"] = 1
            return
        df[out_col] = compute_rate(numer, denom, fill=DEFAULTS.get(out_col, 0.0))
        if flag_imputations:
            df[f"{out_col}_imputed"] = (
                numer.isna() | denom.isna() | (denom == 0)
            ).astype(int)

    _add_def_rate(points_against, denom_drives_def, "drive_points_allowed_per_drive")
    _add_def_rate(yards_against,  denom_drives_def, "drive_yards_allowed_per_drive")
    _add_def_rate(turnovers_forced, denom_drives_def, "drive_turnovers_forced_per_drive")
    _add_def_rate(sacks_made, denom_drives_def, "drive_sacks_made_per_drive")
    _add_def_rate(yards_against,  denom_plays_def,  "drive_yards_per_play_allowed")

    # Defensive reliability
    df["drive_defense_samples"] = (pd.to_numeric(denom_drives_def, errors="coerce").fillna(0)).astype(float)
    df["drive_low_sample_def"] = (df["drive_defense_samples"] < 8).astype(int)

    # -----------------------------------------------------------------
    # Winsorize selected continuous drive rates to stabilize tails
    # -----------------------------------------------------------------
    if winsorize and 0.0 <= winsor_limits[0] < winsor_limits[1] <= 1.0:
        to_wins = [
            "drive_yards_per_drive", "drive_plays_per_drive", "drive_turnovers_per_drive",
            "drive_yards_per_play", "drive_points_per_drive",
            "drive_td_rate", "drive_fg_rate",
            "drive_red_zone_td_pct", "drive_red_zone_trips_per_drive",
            "drive_third_down_conv_pct", "drive_third_down_att_per_drive",
            "drive_fourth_down_conv_pct", "drive_fourth_down_att_per_drive",
            "drive_penalties_per_drive", "drive_penalty_yards_per_drive",
            "drive_sacks_allowed_per_drive", "drive_explosive_play_rate",
            "drive_seconds_per_play", "drive_seconds_per_drive",
            "drive_points_allowed_per_drive", "drive_yards_allowed_per_drive",
            "drive_turnovers_forced_per_drive", "drive_sacks_made_per_drive",
            "drive_yards_per_play_allowed",
        ]
        lo, hi = winsor_limits
        for c in to_wins:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = _winsorize_series(df[c], lo, hi)

    # -----------------------------------------------------------------
    # Cleanup: drop any local temp cols; enforce idempotence
    # -----------------------------------------------------------------
    temps_to_drop = []
    if temp_poss_col and temp_poss_col in df.columns:
        temps_to_drop.append(temp_poss_col)

    # Drop any internal __*__ temps that are not part of the original input
    for c in list(df.columns):
        if c.startswith("__") and c not in original_cols:
            temps_to_drop.append(c)

    if temps_to_drop:
        df.drop(columns=list(set(temps_to_drop)), inplace=True, errors="ignore")

    # Final logging
    added = [c for c in df.columns if c not in box_scores_df.columns]
    logger.debug("drive_metrics: added %d new columns", len(added))

    return df
