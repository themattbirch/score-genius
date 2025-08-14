# backend/nfl_features/drive.py
"""
Game-level DRIVE features (leakage-safe).

This stage adapts team-game drive metrics into *pre-game* features for each
game_id in the current season chunk. It:

1) Enriches the historical team-game box scores with per-drive/per-play rates
   via `compute_drive_metrics` (from drive_metrics.py).
2) Builds *pre-game*, leakage-safe team features by taking a rolling window
   (default: last 5 games) of those rates, with a 1-game shift to exclude the
   current game.
3) Aggregates to game-level for the season chunk:
     - drive_home_* : home team pre-game rolling means
     - drive_away_* : away team pre-game rolling means
     - drive_*_avg_diff : (home − away)
     - drive_total_*_avg: (home + away) where meaningful
4) Returns exactly one row per `game_id` with all columns prefixed `drive_`
   so the engine buckets them under "drive".

Inputs expected:
- `games_df` (season spine) with at least: game_id, game_date/game_time or kickoff_ts,
  and ideally home_team_id and away_team_id. If IDs are missing, we will try to
  match via normalized team names (best-effort).
- `historical_team_stats_df`: team-game box scores containing the raw counts
  that `compute_drive_metrics` needs (team_id, game_id, total_drives, plays_total, etc.).

All features are pre-game: we only use games strictly prior to each game’s kickoff
for the rolling calculations (implemented via groupby().apply(shift→rolling mean)).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .drive_metrics import compute_drive_metrics
from .utils import normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _build_ts_utc(df: pd.DataFrame) -> pd.Series:
    """
    Build a UTC kickoff timestamp for each row in df:
      - Prefer 'kickoff_ts' if present (assumed UTC or tz-aware).
      - Else combine 'game_date' + 'game_time' (assume America/New_York) → UTC.
    """
    # Prefer kickoff_ts if present
    if "kickoff_ts" in df.columns:
        ts_utc = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
    else:
        ts_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # Fallback from date+time in ET
    date_str = df.get("game_date", pd.Series("", index=df.index)).astype(str)
    time_str = df.get("game_time", pd.Series("00:00:00", index=df.index)).astype(str)
    dt_combo = (date_str + " " + time_str).str.strip()

    try:
        fallback_et = (
            pd.to_datetime(dt_combo, errors="coerce")
            .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        )
        fallback_utc = fallback_et.dt.tz_convert("UTC")
    except Exception:
        fallback_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    ts_utc = ts_utc.fillna(fallback_utc)
    return ts_utc


def _ensure_ids_on_spine(spine: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure home_team_id / away_team_id exist on the season spine.
    If missing, try to backfill from normalized team names (best-effort).
    """
    out = spine.copy()
    # Normalize name columns (string, lower)
    for c in ("home_team_norm", "away_team_norm"):
        if c in out.columns:
            out[c] = out[c].astype(str).str.lower()
        elif c.replace("_norm", "") in out.columns:
            out[c] = out[c.replace("_norm", "")].astype(str).apply(normalize_team_name).str.lower()
    return out


def _coerce_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _rolling_pre_game(
    df_team: pd.DataFrame,
    group_key: str,
    metrics: List[str],
    window: int,
) -> pd.DataFrame:
    """
    Compute *pre-game* rolling means for the given metrics per team,
    using a 1-row shift to exclude the current game.

    Returns the original frame with new columns named f"{m}_avg".
    """
    if df_team.empty:
        return df_team

    df_team = df_team.sort_values(["team_id", "ts_utc", "game_id"]).copy()
    gb = df_team.groupby(group_key, sort=False)

    for m in metrics:
        if m not in df_team.columns:
            df_team[f"{m}_avg"] = np.nan
            continue
        # Rolling mean over last `window` games, excluding current row (shift first)
        df_team[f"{m}_avg"] = (
            gb[m]
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # Reliability diagnostics
    df_team["drive_prior_games"] = gb.cumcount()  # number of prior games before current
    df_team["drive_prior_games_capped"] = df_team["drive_prior_games"].clip(upper=window).astype(int)
    df_team["drive_low_sample_any"] = (df_team["drive_prior_games"] < max(3, window // 2)).astype(int)

    return df_team


def _aggregate_to_game_level(
    df_team: pd.DataFrame,
    spine: pd.DataFrame,
    metrics_avg: List[str],
    make_totals: bool = True,
    make_diffs: bool = True,
) -> pd.DataFrame:
    """
    Pivot team-level pre-game averages to one row per game_id with:
      - drive_home_<metric_base>_avg
      - drive_away_<metric_base>_avg
      - drive_<metric_base>_avg_diff (home - away)    [optional]
      - drive_total_<metric_base>_avg                 [optional]
    Also includes sample counts/flags.
    """
    # Bring home/away membership by joining spine on game_id
    need = ["game_id"]
    if "home_team_id" in spine.columns:
        need.append("home_team_id")
    if "away_team_id" in spine.columns:
        need.append("away_team_id")
    side_map = spine[need].copy()

    # Coerce IDs to string for stable comparison
    for c in ("home_team_id", "away_team_id"):
        if c in side_map.columns:
            side_map[c] = _coerce_str(side_map[c])

    df = df_team.merge(side_map, on="game_id", how="inner")
    if "home_team_id" in df.columns and "away_team_id" in df.columns:
        df["team_id_str"] = _coerce_str(df["team_id"])
        df["is_home"] = (df["team_id_str"] == df["home_team_id"]).astype(int)
        df["is_away"] = (df["team_id_str"] == df["away_team_id"]).astype(int)
    else:
        # Fallback to name-based matching if IDs missing
        for c in ("home_team_norm", "away_team_norm"):
            if c not in spine.columns and c.replace("_norm", "") in spine.columns:
                spine[c] = spine[c.replace("_norm", "")].apply(normalize_team_name).str.lower()
        df = df.merge(
            spine[["game_id", "home_team_norm", "away_team_norm"]],
            on="game_id", how="left", suffixes=("", "_spine")
        )
        # Attempt to pick a team name column if available
        this_team = None
        for cand in ("team_name_norm", "team", "team_abbr"):
            if cand in df.columns:
                this_team = df[cand].astype(str).apply(normalize_team_name).str.lower()
                break
        if this_team is not None:
            df["is_home"] = (this_team == _coerce_str(df["home_team_norm"])).astype(int)
            df["is_away"] = (this_team == _coerce_str(df["away_team_norm"])).astype(int)
        else:
            logger.warning("drive.transform: unable to determine home/away for some rows; dropping them.")
            return pd.DataFrame(columns=["game_id"])

    # Build per-row, per-side columns
    keep_cols = ["game_id"]
    side_cols_map: Dict[str, Dict[str, str]] = {}  # base -> {'home': col, 'away': col}

    # Reliability exposures per side
    for side in ("home", "away"):
        mask_col = "is_home" if side == "home" else "is_away"
        prior_col = f"drive_{side}_prior_games"
        low_col = f"drive_{side}_low_sample"
        df[prior_col] = np.where(df[mask_col] == 1, df["drive_prior_games_capped"], np.nan)
        df[low_col] = np.where(df[mask_col] == 1, df["drive_low_sample_any"], np.nan)
        keep_cols.extend([prior_col, low_col])

    # Metric averages per side
    for m_avg in metrics_avg:
        base = m_avg.replace("_avg", "")  # e.g., drive_points_per_drive
        suffix = base.split("drive_")[-1]  # e.g., points_per_drive

        side_cols_map[suffix] = {}
        for side in ("home", "away"):
            mask_col = "is_home" if side == "home" else "is_away"
            col = f"drive_{side}_{suffix}_avg"
            df[col] = np.where(df[mask_col] == 1, df[m_avg], np.nan)
            side_cols_map[suffix][side] = col
            keep_cols.append(col)

    # Reduce to one row per game_id (max picks the filled value among home/away rows)
    out = (
        df[keep_cols]
        .groupby("game_id", as_index=False)
        .max(numeric_only=True)
    )

    # Totals/diffs computed after wide form exists
    if make_totals or make_diffs:
        for suffix, sides in side_cols_map.items():
            hcol = sides.get("home")
            acol = sides.get("away")
            if hcol not in out.columns or acol not in out.columns:
                continue
            if make_totals:
                out[f"drive_total_{suffix}_avg"] = out[hcol] + out[acol]
            if make_diffs:
                out[f"drive_{suffix}_avg_diff"] = out[hcol] - out[acol]

    # Cast reliability flags to int
    for c in out.columns:
        if c.endswith("_low_sample"):
            out[c] = out[c].fillna(0).astype(int)
        if c.endswith("_prior_games"):
            out[c] = out[c].fillna(0).astype(int)

    return out


def _list_available_metrics(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


# ------------------------------------------------------------
# Public API: engine stage
# ------------------------------------------------------------

def transform(
    games_df: pd.DataFrame,
    *,
    historical_team_stats_df: pd.DataFrame,
    debug: bool = False,
    flag_imputations: bool = True,
    window: int = 5,
) -> pd.DataFrame:
    """
    DRIVE stage entrypoint...
    """
    if games_df is None or games_df.empty:
        logger.warning("ENGINE:drive received empty games_df; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    if historical_team_stats_df is None or historical_team_stats_df.empty:
        logger.warning("ENGINE:drive missing historical_team_stats_df; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    spine = _ensure_ids_on_spine(games_df.copy())
    spine["ts_utc"] = _build_ts_utc(spine)
    team = historical_team_stats_df.copy()

    # <<< START OF NEW DIAGNOSTIC CODE >>>
    logger.info("DRIVE_DEBUG: Preparing to call compute_drive_metrics.")
    pre_existing_col = "__opp_row_in_game__"
    if pre_existing_col in team.columns:
        logger.warning(
            "DRIVE_DEBUG: Column '%s' FOUND in input dataframe. Attempting to drop.",
            pre_existing_col
        )
        team = team.drop(columns=[pre_existing_col], errors='ignore')
        if pre_existing_col not in team.columns:
            logger.info("DRIVE_DEBUG: Column '%s' successfully dropped.", pre_existing_col)
        else:
            logger.error("DRIVE_DEBUG: FAILED to drop column '%s'.", pre_existing_col)
    else:
        logger.info("DRIVE_DEBUG: Column '%s' not found in input dataframe. Proceeding.", pre_existing_col)
    # <<< END OF NEW DIAGNOSTIC CODE >>>

    # Enrich team-game rows with team-drive metrics (utility layer)
    team = compute_drive_metrics(team, flag_imputations=flag_imputations)


    # Timestamps for team rows (ordering for rolling)
    team["ts_utc"] = _build_ts_utc(team)

    # Basic schema checks
    must_have = {"game_id", "team_id", "ts_utc"}
    missing = must_have - set(team.columns)
    if missing:
        logger.warning("ENGINE:drive team DF missing required columns %s; returning empty.", missing)
        return pd.DataFrame(columns=["game_id"])

    # Candidate metric list (team-level instantaneous rates) to roll over
    metric_candidates = [
        # offense
        "drive_points_per_drive",
        "drive_yards_per_play",
        "drive_plays_per_drive",
        "drive_turnovers_per_drive",
        "drive_red_zone_td_pct",
        "drive_red_zone_trips_per_drive",
        "drive_third_down_conv_pct",
        "drive_fourth_down_conv_pct",
        "drive_explosive_play_rate",
        "drive_seconds_per_play",
        "drive_seconds_per_drive",
        # defense
        "drive_points_allowed_per_drive",
        "drive_yards_per_play_allowed",
        "drive_turnovers_forced_per_drive",
        "drive_sacks_made_per_drive",
    ]
    metrics_available = _list_available_metrics(team, metric_candidates)

    if not metrics_available:
        logger.warning("ENGINE:drive no drive_* metrics available after compute_drive_metrics; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    # Compute *pre-game* rolling means per team (shifted)
    team = _rolling_pre_game(team, group_key="team_id", metrics=metrics_available, window=window)

    # Keep only rows for games present in the current season chunk
    season_game_ids = set(spine["game_id"].unique().tolist())
    team_chunk = team.loc[team["game_id"].isin(season_game_ids)].copy()

    if team_chunk.empty:
        logger.warning("ENGINE:drive no matching team rows for season chunk; returning empty.")
        return pd.DataFrame(columns=["game_id"])

    # Aggregate team pre-game features to game-level
    metrics_avg = [f"{m}_avg" for m in metrics_available]
    game_level = _aggregate_to_game_level(
        df_team=team_chunk,
        spine=spine,
        metrics_avg=metrics_avg,
        make_totals=True,
        make_diffs=True,
    )

    # Final sanity: ensure one row per game_id
    if game_level.duplicated("game_id").any():
        dup_ct = int(game_level.duplicated("game_id").sum())
        logger.warning("ENGINE:drive produced %d duplicate game_id rows; keeping last.", dup_ct)
        game_level = game_level.sort_values("game_id").drop_duplicates("game_id", keep="last")

    if debug:
        new_cols = [c for c in game_level.columns if c != "game_id"]
        nulls = game_level[new_cols].isna().sum().sort_values(ascending=False).head(12).to_dict()
        zeros = (
            game_level[new_cols].select_dtypes(include=np.number)
            .eq(0).sum().sort_values(ascending=False).head(12).to_dict()
        )
        logger.debug("ENGINE:drive emitted %d cols | null_top=%s | zero_top=%s", len(new_cols), nulls, zeros)
        logger.debug("ENGINE:drive sample cols=%s", new_cols[:18])

    return game_level
