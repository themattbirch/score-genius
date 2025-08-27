# backend/nfl_features/drive.py
"""
Game-level DRIVE features (leakage-safe, idempotent).

Pipeline (per season chunk):
  1) Enrich team-game logs with per-drive/per-play rates via `compute_drive_metrics`.
  2) Attach kickoff timestamps (UTC) and season labels from the season spine.
  3) Build *pre-game* rolling means (shift(1) → exclude current), optionally resetting by season.
  4) Aggregate to game-level:
       - drive_home_<metric>_avg, drive_away_<metric>_avg
       - drive_<metric>_avg_diff  (home - away)
       - drive_total_<metric>_avg (home + away)
       - drive_home/away_prior_games, drive_home/away_low_sample
  5) Return exactly one row per `game_id` with all columns prefixed `drive_` (plus the `game_id` key).

Idempotence contract:
  - No helper columns leak (any __* internal temps are dropped before return).
  - Re-running on the same inputs yields identical outputs.
  - Defensive to noisy/duplicated team-game rows (≥2 per game trimmed upstream in drive_metrics).

Failure behavior:
  - On exceptions and soft_fail=True, returns a sentinel frame with
    {game_id, drive_features_unavailable=1} so the engine can proceed safely.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .drive_metrics import compute_drive_metrics
from .utils import normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]

# -----------------------------------------------------------------------------
# Metric universe (team-level instantaneous rates to roll over)
# -----------------------------------------------------------------------------
METRIC_CANDIDATES: List[str] = [
    # Offense
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
    # Defense
    "drive_points_allowed_per_drive",
    "drive_yards_per_play_allowed",
    "drive_turnovers_forced_per_drive",
    "drive_sacks_made_per_drive",
]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _build_ts_utc(df: pd.DataFrame) -> pd.Series:
    """
    Build a UTC kickoff timestamp for each row in df:
      - Prefer 'kickoff_ts' if present (assumed tz-aware or UTC).
      - Else combine 'game_date' + 'game_time' (assume America/New_York) → UTC.
    """
    # Prefer kickoff_ts if present
    if "kickoff_ts" in df.columns:
        ts_utc = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
    else:
        ts_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # Also accept a single ISO timestamp field that is already UTC
    for alt in ("game_timestamp", "scheduled_time_utc"):
        if ts_utc.isna().all() and alt in df.columns:
            alt_ts = pd.to_datetime(df[alt], errors="coerce", utc=True)
            ts_utc = ts_utc.fillna(alt_ts)


    # Fallback from date+time in ET
    date_str = df.get("game_date", pd.Series("", index=df.index)).astype(str)
    time_str = df.get("game_time", pd.Series("00:00:00", index=df.index)).astype(str)
    combo = (date_str + " " + time_str).str.strip()

    try:
        et = (
            pd.to_datetime(combo, errors="coerce")
            .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        )
        ts_fallback = et.dt.tz_convert("UTC")
    except Exception:
        ts_fallback = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    return ts_utc.fillna(ts_fallback)


def _ensure_ids_on_spine(spine: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure normalized team names are present (best-effort) for fallback side-mapping.
    """
    out = spine.copy()
    for c in ("home_team_norm", "away_team_norm"):
        base = c.replace("_norm", "")
        if c in out.columns:
            out[c] = out[c].astype(str).str.lower()
        elif base in out.columns:
            out[c] = out[base].astype(str).apply(normalize_team_name).str.lower()
    return out


def _coerce_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _list_available_metrics(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _audit_rows_per_game(df_team: pd.DataFrame) -> None:
    """Warn if any game_id has >2 team rows after upstream trimming."""
    if df_team.empty or "game_id" not in df_team.columns:
        return
    counts = df_team["game_id"].value_counts(dropna=False)
    bad = counts[counts > 2]
    if not bad.empty:
        logger.warning("DRIVE_AUDIT: games with >2 team rows=%d | sample=%s", int(bad.size), bad.head(5).to_dict())


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
      - drive_prior_games_capped: clipped at window (nullable Int64)
      - drive_low_sample_any: 1 if prior_games < max(3, window//2), else 0
    """
    if df_team.empty:
        return df_team

    # Stable ordering by kickoff ts then game_id to break ties
    sort_cols = [c for c in ("ts_utc", "game_id") if c in df_team.columns]
    df_team = df_team.sort_values(sort_cols).copy()

    # Include NA keys as their own group to avoid introducing NaN via groupby
    gb = df_team.groupby(group_keys, sort=False, dropna=False)

    # Prior games (per group) before current — cumcount is already 0-based
    prior = gb.cumcount()

    # Ensure numeric, then build capped and low-sample flags safely
    prior = pd.to_numeric(prior, errors="coerce")
    df_team["drive_prior_games"] = prior

    # Treat NaN prior counts as 0 for downstream logic
    prior_filled = prior.fillna(0)

    # Use nullable Int64 to allow safe integer representation
    df_team["drive_prior_games_capped"] = (
        prior_filled.clip(upper=window).astype("Int64")
    )
    df_team["drive_low_sample_any"] = (
        (prior_filled < max(3, window // 2)).astype("int8")
    )

    # Shifted rolling mean for each metric (exclude current game via shift(1)),
    # strictly within each group (team_id [+ season])
    for m in metrics:
        out_col = f"{m}_avg"
        if m not in df_team.columns:
            df_team[out_col] = np.nan
            continue

        df_team[out_col] = gb[m].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Optional masking for very low samples
        if min_prior_games and min_prior_games > 0:
            mask = prior_filled < int(min_prior_games)
            df_team.loc[mask, out_col] = np.nan



        # Optional masking for very low samples
        if min_prior_games and min_prior_games > 0:
            mask = prior_filled < int(min_prior_games)
            df_team.loc[mask, out_col] = np.nan


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
    Also includes per-side sample counts/flags:
      - drive_home_prior_games, drive_home_low_sample
      - drive_away_prior_games, drive_away_low_sample
    """
    need = ["game_id"]
    if "home_team_id" in spine.columns: need.append("home_team_id")
    if "away_team_id" in spine.columns: need.append("away_team_id")
    side_map = spine[need].copy()

    # Normalize IDs for robust side-mapping
    for c in ("home_team_id", "away_team_id"):
        if c in side_map.columns:
            side_map[c] = _coerce_str(side_map[c])

    df = df_team.merge(side_map, on="game_id", how="inner")

    # Determine membership by IDs when available, else fallback to normalized names
    have_ids = ("home_team_id" in df.columns) and ("away_team_id" in df.columns) and ("team_id" in df.columns)
    if have_ids:
        df["team_id_str"] = _coerce_str(df["team_id"])
        df["is_home"] = (df["team_id_str"] == df["home_team_id"]).astype(int)
        df["is_away"] = (df["team_id_str"] == df["away_team_id"]).astype(int)
    else:
        # Fallback to normalized names
        spine_fb = spine.copy()
        for c in ("home_team_norm", "away_team_norm"):
            base = c.replace("_norm", "")
            if c not in spine_fb.columns and base in spine_fb.columns:
                spine_fb[c] = spine_fb[base].astype(str).apply(normalize_team_name).str.lower()

        df = df.merge(
            spine_fb[["game_id", "home_team_norm", "away_team_norm"]],
            on="game_id", how="left", suffixes=("", "_spine")
        )

        # Try to find a team name column on df
        name_col: Optional[pd.Series] = None
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
        df[low_col]   = np.where(df[mask_col] == 1, df["drive_low_sample_any"],   np.nan)
        keep_cols.extend([prior_col, low_col])

    # Metric averages per side
    side_cols_map: Dict[str, Dict[str, str]] = {}  # base -> {'home': col, 'away': col}
    for m_avg in metrics_avg:
        base = m_avg.replace("_avg", "")  # e.g., drive_points_per_drive
        suffix = base.split("drive_", 1)[-1]  # e.g., points_per_drive
        side_cols_map[suffix] = {}
        for side in ("home", "away"):
            mask_col = "is_home" if side == "home" else "is_away"
            col = f"drive_{side}_{suffix}_avg"
            df[col] = np.where(df[mask_col] == 1, df[m_avg], np.nan)
            side_cols_map[suffix][side] = col
            keep_cols.append(col)

    # Reduce to one row per game_id
    out = df[keep_cols].groupby("game_id", as_index=False).max(numeric_only=True)

    # Totals/diffs
    if make_totals or make_diffs:
        for suffix, sides in side_cols_map.items():
            hcol = sides.get("home")
            acol = sides.get("away")
            if hcol not in out.columns or acol not in out.columns:
                continue
            # inside the make_totals/make_diffs loop
            if make_totals:
                out[f"drive_total_{suffix}_avg"] = out[hcol].fillna(0) + out[acol].fillna(0)
            if make_diffs:
                out[f"drive_{suffix}_avg_diff"] = out[hcol].fillna(0) - out[acol].fillna(0)



    # Cast reliability flags to int
    for c in out.columns:
        if c.endswith("_low_sample"):
            out[c] = out[c].fillna(0).astype(int)
        if c.endswith("_prior_games"):
            out[c] = out[c].fillna(0).astype(int)

    return out


# -----------------------------------------------------------------------------
# Public API: engine stage
# -----------------------------------------------------------------------------

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
        if "season" not in spine.columns and "game_date" in spine.columns:
            spine["season"] = pd.to_datetime(spine["game_date"], errors="coerce").dt.year

        # Start from team-level stats; proactively drop any leaked temp helpers
        team = historical_team_stats_df.copy()
        for tmp in ("__opp_row_in_game__", "__row_in_game__"):
            if tmp in team.columns:
                team.drop(columns=[tmp], inplace=True, errors="ignore")

        # Enrich team-game rows with drive metrics (idempotent)
        team = compute_drive_metrics(team, flag_imputations=flag_imputations)

        # Merge kickoff timestamp & season from spine for correct rolling order and grouping
        team = team.merge(
            spine[["game_id", "ts_utc", "season"]].drop_duplicates("game_id"),
            on="game_id", how="left"
        )

        # Minimal schema checks (ts_utc is helpful but not hard-required here)
        must_have = {"game_id", "team_id"}
        missing = must_have - set(team.columns)
        if missing:
            logger.warning("ENGINE:drive team DF missing required columns %s; continuing with soft policy.", missing)
            if soft_fail:
                uniq = pd.DataFrame({"game_id": games_df.get("game_id", pd.Series(dtype="Int64")).drop_duplicates()})
                if uniq.empty:
                    return pd.DataFrame(columns=["game_id"])
                uniq["drive_features_unavailable"] = 1
                return uniq
            return pd.DataFrame(columns=["game_id"])



        # Quick audit
        _audit_rows_per_game(team)

        # Candidate metrics present
        metrics_available = _list_available_metrics(team, METRIC_CANDIDATES)
        logger.info("ENGINE:drive metrics_available=%d | %s", len(metrics_available), metrics_available)

        if not metrics_available:
            logger.warning("ENGINE:drive no drive_* metrics available after compute_drive_metrics; returning empty.")
            return pd.DataFrame(columns=["game_id"])

        # Pre-game rolling per group (team or team+season)
        group_keys = ["team_id", "season"] if reset_by_season else ["team_id"]
        team = _rolling_pre_game(
            df_team=team,
            group_keys=group_keys,
            metrics=metrics_available,
            window=window,
            min_prior_games=min_prior_games,
        )

        # After _rolling_pre_game(...) has produced rolling columns like <metric>_avg
        metrics_avg = [f"{m}_avg" for m in metrics_available]

        # Base frame (one row per game)
        base = spine[["game_id", "ts_utc", "home_team_id", "away_team_id"]].drop_duplicates("game_id").copy()

        # Normalize IDs to strings for robust joins
        base["home_team_id"] = base["home_team_id"].astype(str)
        base["away_team_id"] = base["away_team_id"].astype(str)

        team_asof = team.copy()
        team_asof["team_id"] = team_asof["team_id"].astype(str)
        team_asof = team_asof.sort_values(["team_id", "ts_utc"])

        # Columns we’ll pull from the team-time series
        pull_cols = ["team_id", "ts_utc", "drive_prior_games_capped", "drive_low_sample_any"] + metrics_avg
        team_asof = team_asof[pull_cols]

        def _asof_side(side: str) -> pd.DataFrame:
            # Build left keys: game ts + the side’s team_id
            left = base[["game_id", "ts_utc", f"{side}_team_id"]].rename(columns={f"{side}_team_id": "team_id"}).copy()
            left = left.sort_values(["team_id", "ts_utc"])

            # As-of join: last known rolling averages at/just before kickoff
            merged = pd.merge_asof(
                left,
                team_asof.sort_values(["team_id", "ts_utc"]),
                by="team_id",
                left_on="ts_utc",
                right_on="ts_utc",
                direction="backward",           # <= kickoff
                allow_exact_matches=True,
            )

            # Suffix columns with side
            out = merged.drop(columns=["ts_utc"])
            rename_map = {}
            # reliability
            rename_map["drive_prior_games_capped"] = f"drive_{side}_prior_games"
            rename_map["drive_low_sample_any"]     = f"drive_{side}_low_sample"

            # metrics
            for m in metrics_avg:
                base_name = m.replace("_avg", "")       # e.g. drive_points_per_drive
                suffix    = base_name.split("drive_", 1)[-1]
                rename_map[m] = f"drive_{side}_{suffix}_avg"

            out = out.rename(columns=rename_map)
            return out.drop(columns=["team_id"], errors="ignore")

        home_asof = _asof_side("home")
        away_asof = _asof_side("away")

        # Merge sides back to game-level
        game_level = base[["game_id"]].merge(home_asof, on="game_id", how="left").merge(away_asof, on="game_id", how="left")

        # Totals & diffs
        for m in metrics_available:
            suffix = m.split("drive_", 1)[-1]
            h = f"drive_home_{suffix}_avg"
            a = f"drive_away_{suffix}_avg"
            if h in game_level.columns and a in game_level.columns:
                game_level[f"drive_total_{suffix}_avg"] = game_level[h].fillna(0) + game_level[a].fillna(0)
                game_level[f"drive_{suffix}_avg_diff"]  = game_level[h].fillna(0) - game_level[a].fillna(0)

        # Cast reliability flags to ints
        for c in game_level.columns:
            if c.endswith("_low_sample"):
                game_level[c] = game_level[c].fillna(0).astype(int)
            if c.endswith("_prior_games"):
                game_level[c] = game_level[c].fillna(0).astype(int)

        # Enforce one row per game_id
        if game_level.duplicated("game_id").any():
            dup_ct = int(game_level.duplicated("game_id").sum())
            logger.warning("ENGINE:drive produced %d duplicate game_id rows; keeping last.", dup_ct)
            game_level = game_level.sort_values("game_id").drop_duplicates("game_id", keep="last")

        # Strict schema: all non-key columns start with 'drive_'
        bad_cols = [c for c in game_level.columns if c != "game_id" and not c.startswith("drive_")]
        if bad_cols:
            logger.warning("ENGINE:drive found non-prefixed columns; dropping: %s", bad_cols)
            game_level.drop(columns=bad_cols, inplace=True, errors="ignore")

        # Drop any internal temps (idempotence)
        for c in list(game_level.columns):
            if c.startswith("__"):
                game_level.drop(columns=[c], inplace=True, errors="ignore")

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
            uniq = pd.DataFrame({"game_id": games_df.get("game_id", pd.Series(dtype="Int64")).drop_duplicates()})
            if uniq.empty:
                return pd.DataFrame(columns=["game_id"])
            uniq["drive_features_unavailable"] = 1
            return uniq
        raise
