# backend/mlb_features/season.py

"""
Attaches previous-season statistical context for MLB home and away teams.
Falls back to defaults if stats are unavailable or missing.
Optionally creates boolean flags '<feature>_imputed' when values are default-filled.
"""

from __future__ import annotations
import logging
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name, determine_season
    logger.info("Imported MLB_DEFAULTS, normalize_team_name, and determine_season from .utils")
except ImportError:
    logger.warning("Could not import utils; using local fallbacks in season.py")
    MLB_DEFAULTS: Dict[str, Any] = {}

    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip() if pd.notna(team_id) else "unknown"

    def determine_season(game_date: pd.Timestamp, sport: str = "mlb") -> int:
        if pd.isna(game_date):
            logger.warning("Cannot determine season for NaT; defaulting to 0")
            return 0
        return game_date.year


def _previous_season(year: int) -> int:
    """Return the prior season year, or fallback if invalid."""
    if isinstance(year, int) and year > 0:
        return year - 1
    logger.warning(f"Invalid year '{year}' in _previous_season. Returning as-is.")
    return year


def _fill_defaults_season(df: pd.DataFrame, placeholder_map: Dict[str, float], flags: bool) -> pd.DataFrame:
    """
    Assign default values (and imputed flags if requested) for both home/away.
    """
    for side in ("home", "away"):
        for feat, default in placeholder_map.items():
            col = f"{side}_{feat}"
            df[col] = default
            if flags:
                df[f"{col}_imputed"] = True
    return df


__all__ = ["transform"]


def transform(
    df: pd.DataFrame,
    *,
    team_stats_df: Optional[pd.DataFrame] = None,
    flag_imputations: bool = True,
    debug: bool = False,
    # Current‐game column names
    game_date_col: str = "game_date_et",
    home_team_col: str = "home_team_id",
    away_team_col: str = "away_team_id",
    # Team‐stats table columns
    ts_team_id_col: str = "team_id",
    ts_season_col: str = "season",
    ts_win_pct_col: str = "wins_all_percentage",
    ts_runs_for_col: str = "runs_for_avg_all",
    ts_runs_against_col: str = "runs_against_avg_all",
) -> pd.DataFrame:
    """
    Attach previous-season MLB statistical context to each game row.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG: Starting season.transform")

    logger.info("Adding previous-season context features for MLB")
    result = df.copy()

    # Ensure critical current‐game columns exist
    required = {"game_id", game_date_col, home_team_col, away_team_col}
    if not required.issubset(result.columns):
        missing = required - set(result.columns)
        logger.error(f"Missing columns in df: {missing}. Skipping season features.")
        return df

    # Parse and drop invalid dates
    result["game_date_parsed"] = pd.to_datetime(result[game_date_col], errors="coerce")
    result = result.dropna(subset=["game_date_parsed"])
    if result.empty:
        logger.warning("No valid game dates → skipping season features.")
        return df

    # Build the placeholder defaults mapping
    placeholder_map = {
        "prev_season_win_pct":        float(MLB_DEFAULTS.get("mlb_prev_season_win_pct",
                                                           MLB_DEFAULTS.get("mlb_win_pct", 0.5))),
        "prev_season_avg_runs_for":   float(MLB_DEFAULTS.get("mlb_prev_season_avg_runs_for",
                                                           MLB_DEFAULTS.get("mlb_avg_runs_for", 4.0))),
        "prev_season_avg_runs_against": float(MLB_DEFAULTS.get("mlb_prev_season_avg_runs_against",
                                                            MLB_DEFAULTS.get("mlb_avg_runs_against", 4.0))),
    }
    all_new_cols = []
    for side in ("home", "away"):
        for feat in placeholder_map:
            all_new_cols.append(f"{side}_{feat}")

    # If no team_stats_df, assign defaults & flags then compute diffs
    if team_stats_df is None or team_stats_df.empty:
        logger.warning("No team_stats_df provided → filling defaults for season features")
        result = _fill_defaults_season(result, placeholder_map, flag_imputations)

    else:
        ts = team_stats_df.copy()
        # Validate team_stats_df columns
        required_ts = {ts_team_id_col, ts_season_col, ts_win_pct_col, ts_runs_for_col, ts_runs_against_col}
        if not required_ts.issubset(ts.columns):
            missing_ts = required_ts - set(ts.columns)
            logger.error(f"Missing columns in team_stats_df: {missing_ts} → using defaults")
            result = _fill_defaults_season(result, placeholder_map, flag_imputations)
        else:
            # Determine seasons for lookup
            result["season_year"] = result["game_date_parsed"].apply(lambda d: determine_season(d, "mlb"))
            result["prev_season_year"] = result["season_year"].apply(_previous_season)
            # Normalize team IDs
            result["home_norm"] = result[home_team_col].apply(normalize_team_name)
            result["away_norm"] = result[away_team_col].apply(normalize_team_name)

            # Prepare ts lookup
            ts["team_norm"] = ts[ts_team_id_col].apply(normalize_team_name)
            ts[ts_season_col] = ts[ts_season_col].astype(int)
            ts["lookup"] = ts["team_norm"] + "_" + ts[ts_season_col].astype(str)
            ts_sel = ts.set_index("lookup")[[ts_win_pct_col, ts_runs_for_col, ts_runs_against_col]]
            ts_sel = ts_sel.rename(columns={
                ts_win_pct_col: "prev_season_win_pct",
                ts_runs_for_col: "prev_season_avg_runs_for",
                ts_runs_against_col: "prev_season_avg_runs_against",
            })

            # Merge per side
            for side in ("home", "away"):
                key_col = f"{side}_lookup"
                result[key_col] = result[f"{side}_norm"] + "_" + result["prev_season_year"].astype(str)
                side_ts = ts_sel.rename(columns={
                    feat: f"{side}_{feat}" for feat in placeholder_map
                })
                result = result.merge(
                    side_ts,
                    how="left",
                    left_on=key_col,
                    right_index=True
                )
                # Flag & fill
                for feat, default in placeholder_map.items():
                    col = f"{side}_{feat}"
                    if col not in result:
                        result[col] = default
                        if flag_imputations:
                            result[f"{col}_imputed"] = True
                    else:
                        if flag_imputations:
                            result[f"{col}_imputed"] = result[col].isna()
                        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(default)

            # Drop helper cols
            result.drop(columns=["season_year", "prev_season_year", "home_norm", "away_norm"] +
                              [f"{s}_lookup" for s in ("home", "away")], inplace=True)

    # Compute diffs and net ratings
    for col in all_new_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(placeholder_map[col.split("_",1)[1]])

    result["prev_season_win_pct_diff"]       = result["home_prev_season_win_pct"] - result["away_prev_season_win_pct"]
    result["prev_season_runs_for_diff"]      = result["home_prev_season_avg_runs_for"] - result["away_prev_season_avg_runs_for"]
    result["prev_season_runs_against_diff"]  = result["home_prev_season_avg_runs_against"] - result["away_prev_season_avg_runs_against"]
    result["home_prev_season_net_rating"]    = result["home_prev_season_avg_runs_for"] - result["home_prev_season_avg_runs_against"]
    result["away_prev_season_net_rating"]    = result["away_prev_season_avg_runs_for"] - result["away_prev_season_avg_runs_against"]
    result["prev_season_net_rating_diff"]    = result["home_prev_season_net_rating"] - result["away_prev_season_net_rating"]

    # Drop the parsed date column
    result.drop(columns=["game_date_parsed"], inplace=True, errors="ignore")

    # Optional: reorder columns (orig → stats → flags → diffs)
    orig_cols = list(df.columns)
    stat_cols = [c for c in result.columns if any(c.startswith(s+"_prev_season") for s in ("home","away"))]
    flag_cols = [c for c in result.columns if c.endswith("_imputed")]
    diff_cols = [c for c in result.columns if c.endswith("_diff") or c.endswith("net_rating")]
    ordered = orig_cols + stat_cols + flag_cols + diff_cols
    # Dedupe while preserving order
    seen = set()
    final_order = [c for c in ordered if c in result.columns and not (c in seen or seen.add(c))]
    final_order += [c for c in result.columns if c not in seen]
    result = result[final_order]

    logger.info(f"Completed season.transform; output shape={result.shape}")
    if debug:
        logger.setLevel(orig_level)
    return result
