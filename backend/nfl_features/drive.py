# backend/nfl_features/drive.py
"""
Game-level DRIVE features (leakage-safe).

This stage adapts team-game drive metrics into *pre-game* features for each
game_id in the current season chunk. It:

1) Enriches team-game logs with per-drive/per-play rates via `compute_drive_metrics`
   (from drive_metrics.py). The helper is idempotent and trims noisy games to ≤2 rows.
2) Builds *pre-game*, leakage-safe team features by taking a rolling window
   (default: last 5 games) of those rates, with a 1-game shift to exclude the
   current game. Rolling can optionally reset at season boundaries.
3) Aggregates to game-level for the season chunk:
     - drive_home_* : home team pre-game rolling means
     - drive_away_* : away team pre-game rolling means
     - drive_*_avg_diff : (home − away)
     - drive_total_*_avg: (home + away) where meaningful
     - drive_home/away_prior_games / drive_home/away_low_sample flags
4) Returns exactly one row per `game_id` with all columns prefixed `drive_`.

Parameters:
- window (int): rolling window size (games), default 5.
- reset_by_season (bool): if True, rolling resets at season boundaries.
- min_prior_games (int): if >0, mask *_avg to NaN when prior_games < threshold.
- soft_fail (bool): if True, return {game_id, drive_features_unavailable=1} on error.
- debug (bool): rich logging and optional idempotence spot-check.

Inputs expected:
- `games_df` (season spine) with: game_id, game_date/game_time or kickoff_ts,
  season, week, and ideally home_team_id / away_team_id. If IDs missing, fallback
  to normalized team names (best-effort).
- `historical_team_stats_df`: team-game box scores containing the raw counts
  required by `compute_drive_metrics` (team_id, game_id, total_drives, plays_total, etc.).

All features are pre-game: only games strictly prior to each kickoff are used
(implemented via groupby shift → rolling mean).
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


def _list_available_metrics(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _audit_rows_per_game(df_team: pd.DataFrame) -> None:
    """Log any games with >2 team rows (post-helper this should be zero)."""
    if df_team.empty or "game_id" not in df_team.columns:
        return
    counts = df_team["game_id"].value_counts(dropna=False)
    bad = counts[counts > 2]
    if not bad.empty:
        sample = bad.head(5).to_dict()
        logger.warning("DRIVE_AUDIT: games with >2 team rows=%d | sample=%s", int(bad.size), sample)
    else:
        logger.debug("DRIVE_AUDIT: games with >2 team rows=0")


def _rolling_pre_game(
    df_team: pd.DataFrame,
    group_keys: List[str],
    metrics: List[str],
    window: int,
    min_prior_games: int = 0,
) -> pd.DataFrame:
    """
    Compute *pre-game* rolling means for the given metrics per group (e.g., team or team+season),
    using a 1-row shift to exclude the current game. Optionally mask low-sample rows.

    Returns the original frame with new columns named f"{m}_avg", plus:
      - drive_prior_games: count of prior games before current (per group)
      - drive_prior_games_capped: clipped at window
      - drive_low_sample_any: 1 if prior_games < max(3, window//2), else 0
    """
    if df_team.empty:
        return df_team

    # Stable ordering by kickoff ts then game_id to break ties
    sort_cols = [c for c in ("ts_utc", "game_id") if c in df_team.columns]
    df_team = df_team.sort_values(sort_cols).copy()

    gb = df_team.groupby(group_keys, sort=False)

    # Prior games (per group) before current
    df_team["drive_prior_games"] = gb.cumcount()
    df_team["drive_prior_games_capped"] = df_team["drive_prior_games"].clip(upper=window).astype(int)
    df_team["drive_low_sample_any"] = (df_team["drive_prior_games"] < max(3, window // 2)).astype(int)

    # Shifted rolling mean for each metric
    for m in metrics:
        if m not in df_team.columns:
            df_team[f"{m}_avg"] = np.nan
            continue
        shifted = gb[m].shift(1)
        df_team[f"{m}_avg"] = shifted.rolling(window=window, min_periods=1).mean()

        # Optional masking for very low samples
        if min_prior_games and min_prior_games > 0:
            mask = df_team["drive_prior_games"] < int(min_prior_games)
            df_team.loc[mask, f"{m}_avg"] = np.nan

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
    Also includes per-side sample counts/flags.
    """
    # Bring home/away membership by joining spine on game_id; also carry season (already present on spine)
    need = ["game_id"]
    if "home_team_id" in spine.columns: need.append("home_team_id")
    if "away_team_id" in spine.columns: need.append("away_team_id")
    side_map = spine[need].copy()

    # Normalize IDs to a common comparable dtype (string) for side membership
    for c in ("home_team_id", "away_team_id"):
        if c in side_map.columns:
            side_map[c] = _coerce_str(side_map[c])

    df = df_team.merge(side_map, on="game_id", how="inner")

    # Compute side flags (prefer IDs; fallback to normalized names if IDs absent)
    have_ids = ("home_team_id" in df.columns) and ("away_team_id" in df.columns)
    if have_ids:
        df["team_id_str"] = _coerce_str(df["team_id"])
        df["is_home"] = (df["team_id_str"] == df["home_team_id"]).astype(int)
        df["is_away"] = (df["team_id_str"] == df["away_team_id"]).astype(int)
    else:
        # Fallback: names on spine and some team name column in df
        spine_fallback = spine.copy()
        for c in ("home_team_norm", "away_team_norm"):
            if c not in spine_fallback.columns and c.replace("_norm", "") in spine_fallback.columns:
                spine_fallback[c] = spine_fallback[c.replace("_norm", "")].apply(normalize_team_name).str.lower()
        df = df.merge(
            spine_fallback[["game_id", "home_team_norm", "away_team_norm"]],
            on="game_id", how="left", suffixes=("", "_spine")
        )
        name_col = None
        for cand in ("team_name_norm", "team", "team_abbr"):
            if cand in df.columns:
                name_col = df[cand].astype(str).apply(normalize_team_name).str.lower()
                break
        if name_col is None:
            logger.warning("drive.transform: unable to determine home/away for some rows; dropping them.")
            return pd.DataFrame(columns=["game_id"])
        df["is_home"] = (name_col == _coerce_str(df["home_team_norm"])).astype(int)
        df["is_away"] = (name_col == _coerce_str(df["away_team_norm"])).astype(int)

    # Reliability exposures per side
    keep_cols = ["game_id"]
    for side in ("home", "away"):
        mask_col = "is_home" if side == "home" else "is_away"
        prior_col = f"drive_{side}_prior_games"
        low_col = f"drive_{side}_low_sample"
        df[prior_col] = np.where(df[mask_col] == 1, df["drive_prior_games_capped"], np.nan)
        df[low_col] = np.where(df[mask_col] == 1, df["drive_low_sample_any"], np.nan)
        keep_cols.extend([prior_col, low_col])

    # Metric averages per side
    side_cols_map: Dict[str, Dict[str, str]] = {}  # base -> {'home': col, 'away': col}
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

    # Reduce to one row per game_id using max as a selector (since only one side populates each column)
    out = (
        df[keep_cols]
        .groupby("game_id", as_index=False)
        .max(numeric_only=True)
    )

    # Totals/diffs
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
    reset_by_season: bool = True,
    min_prior_games: int = 0,
    soft_fail: bool = True,
) -> pd.DataFrame:
    """
    DRIVE stage entrypoint. Returns one row per game_id with drive_* features.

    Notes:
      - Uses kickoff timestamps from the season spine to order games for rolling.
      - Rolling resets by season when reset_by_season=True.
      - If min_prior_games>0, *_avg values are masked to NaN when prior games are below threshold.
      - On error and soft_fail=True, returns a sentinel frame with `drive_features_unavailable=1`.
    """
    try:
        if games_df is None or games_df.empty:
            logger.warning("ENGINE:drive received empty games_df; returning empty.")
            return pd.DataFrame(columns=["game_id"])

        if historical_team_stats_df is None or historical_team_stats_df.empty:
            logger.warning("ENGINE:drive missing historical_team_stats_df; returning empty.")
            return pd.DataFrame(columns=["game_id"])

        # Season spine with canonical IDs and kickoff timestamps
        spine = _ensure_ids_on_spine(games_df.copy())
        spine["ts_utc"] = _build_ts_utc(spine)
        # Keep season on spine for optional rolling reset
        if "season" not in spine.columns and "game_date" in spine.columns:
            # Conservative fallback: season from year (engine usually sets this)
            spine["season"] = pd.to_datetime(spine["game_date"], errors="coerce").dt.year

        # Start from team-level stats; drop any leaked temp if present
        team = historical_team_stats_df.copy()
        leak_col = "__opp_row_in_game__"
        if leak_col in team.columns:
            team.drop(columns=[leak_col], inplace=True, errors="ignore")

        # Enrich team-game rows with drive metrics
        team = compute_drive_metrics(team, flag_imputations=flag_imputations)

        # Merge kickoff timestamp & season from spine for correct rolling order and group
        team = team.merge(spine[["game_id", "ts_utc", "season"]].drop_duplicates("game_id"), on="game_id", how="left")

        # Basic schema checks
        must_have = {"game_id", "team_id", "ts_utc"}
        missing = must_have - set(team.columns)
        if missing:
            logger.warning("ENGINE:drive team DF missing required columns %s; returning empty.", missing)
            return pd.DataFrame(columns=["game_id"])

        # Quick audit (post-helper this should be zero)
        _audit_rows_per_game(team)

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
        logger.info("ENGINE:drive metrics_available=%d | %s", len(metrics_available), metrics_available)

        if not metrics_available:
            logger.warning("ENGINE:drive no drive_* metrics available after compute_drive_metrics; returning empty.")
            return pd.DataFrame(columns=["game_id"])

        # Compute *pre-game* rolling means per group (team or team+season)
        group_keys = ["team_id", "season"] if reset_by_season and "season" in team.columns else ["team_id"]
        team = _rolling_pre_game(
            team,
            group_keys=group_keys,
            metrics=metrics_available,
            window=window,
            min_prior_games=min_prior_games,
        )

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

        # Strict schema: all non-key columns must start with 'drive_'
        bad_cols = [c for c in game_level.columns if c != "game_id" and not c.startswith("drive_")]
        if bad_cols:
            logger.warning("ENGINE:drive found non-prefixed columns; dropping: %s", bad_cols)
            game_level = game_level.drop(columns=bad_cols, errors="ignore")

        # Quick visibility on prior-games distributions
        pg_cols = [c for c in game_level.columns if c.endswith("_prior_games")]
        if pg_cols:
            stats = {c: float(game_level[c].median()) for c in pg_cols}
            logger.debug("ENGINE:drive prior_games median per side → %s", stats)

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

    except Exception as e:
        logger.error("ENGINE:drive fatal error: %s", e, exc_info=True)
        if soft_fail:
            # Return sentinel so downstream stays healthy
            uniq = pd.DataFrame({"game_id": games_df.get("game_id", pd.Series(dtype="Int64")).drop_duplicates()})
            if uniq.empty:
                return pd.DataFrame(columns=["game_id"])
            uniq["drive_features_unavailable"] = 1
            return uniq
        raise
