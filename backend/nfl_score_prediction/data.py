# backend/nfl_score_prediction/data.py
"""
data.py - NFL Data Fetching Module

Fetches historical games, team game stats, and upcoming schedules from Supabase
for the NFL score prediction pipeline.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class NFLDataFetcher:
    """Handles data fetching from Supabase for NFL prediction models."""

    def __init__(self, supabase_client: Any):
        if supabase_client is None:
            raise ValueError("A valid Supabase client must be provided.")
        self.supabase = supabase_client
        logger.info("NFLDataFetcher initialized.")

    # ------------------------------------------------------------------ #
    # Internal utils
    # ------------------------------------------------------------------ #

    def _execute_paginated_query(
    self,
    table: str,
    columns: Sequence[str],
    date_column: str,
    start_date: str,
    *,
    page_size: int = 1000,
    ) -> pd.DataFrame:
        if date_column not in columns:
            logger.warning(
                "'%s' not in selected columns for %s; filtering may be ineffective.",
                date_column,
                table,
            )

        select_str = ", ".join(columns)
        all_rows: List[dict] = []
        start = 0

        try:
            while True:
                qb = (
                    self.supabase.table(table)
                    .select(select_str)
                )

                # Use server-side date filter only if the column is likely valid
                if date_column in columns:
                    qb = qb.gte(date_column, start_date).order(date_column, desc=False)
                else:
                    # ← fallback: don’t push an invalid filter; just order by the first column
                    qb = qb.order(columns[0], desc=False)

                resp = qb.range(start, start + page_size - 1).execute()

                if not hasattr(resp, "data"):
                    raise ValueError(f"Supabase response from '{table}' missing 'data'.")

                batch = resp.data or []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                start += page_size

            df = pd.DataFrame(all_rows)
            # Client-side filter if we couldn’t apply server-side
            if not df.empty and date_column in df.columns:
                # robust to strings/timestamps
                df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
                df = df[df[date_column] >= pd.to_datetime(start_date, errors="coerce")]
            return df if not df.empty else pd.DataFrame()

        except Exception:
            logger.error(
                "Paginated query failed for '%s'.\n%s", table, traceback.format_exc()
            )
            return pd.DataFrame()


    @staticmethod
    def _cast_ids(df: pd.DataFrame, cols: Sequence[str]) -> None:
        """In-place cast of ID columns to pandas' Int64 (nullable int)."""
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    @staticmethod
    def _parse_possession_time(val: Optional[str]) -> float:
        """
        Convert 'MM:SS' or 'H:MM:SS' strings to minutes (float).
        """
        if not val or not isinstance(val, str):
            return 0.0
        parts = val.split(":")
        try:
            if len(parts) == 2:
                m, s = map(int, parts)
                return m + s / 60.0
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return 60 * h + m + s / 60.0
        except (ValueError, TypeError):
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    # Public fetch methods
    # ------------------------------------------------------------------ #

    def fetch_historical_games(self, days_lookback: int = 1825) -> pd.DataFrame:
        """
        Historical outcomes from 'nfl_historical_game_stats'.
        """
        now = datetime.now()
        start_date = (now - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
        logger.info("Fetching historical games since %s...", start_date)

        req_cols = [
            "game_id",
            "season",
            "week",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
        ]
        num_cols = ["season", "week", "home_score", "away_score"]

        # ✅ USE game_date as the filter & sort key
        df = self._execute_paginated_query(
            "nfl_historical_game_stats", req_cols, "game_date", start_date
        )
        if df.empty:
            logger.warning("No historical game data found.")
            return df

        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"])

        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        self._cast_ids(df, ["game_id", "home_team_id", "away_team_id"])
        df = df.sort_values("game_date").reset_index(drop=True)
        logger.info("Fetched %d historical game rows.", len(df))
        return df


    def fetch_historical_team_game_stats(self, days_lookback: int = 1825) -> pd.DataFrame:
        """
        Per-team, per-game stats for the feature engine.

        Primary source:  public.nfl_team_game_core_by_game  (materialized view)
        - columns expected:
            game_id, team_id, game_date, team_score, points_against_total,
            yards_total, plays_total, turnovers_total, yards_per_play_calc,
            passing_yards_per_pass, passing_interceptions, third_down_pct,
            fourth_down_pct, red_zone_pct, sacks_total, penalty_yards,
            rushings_yards_per_rush, possession_time, total_drives (nullable)

        Fallback (if MV empty/unavailable): join legacy tables in Python
        - public.nfl_historical_game_stats
        - public.nfl_historical_game_team_stats
        """
        now = datetime.now()
        start_date = (now - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
        logger.info("Fetching team-game stats since %s...", start_date)

        MV_TABLE = "nfl_team_game_core_by_game"

        # ---- preferred: MV with per-game fields already composed ----
        mv_req_cols = [
            "game_id",
            "team_id",
            "game_date",
            "team_score",
            "points_against_total",
            "yards_total",
            "plays_total",
            "turnovers_total",
            "yards_per_play_calc",
            "passing_yards_per_pass",
            "passing_interceptions",
            "third_down_pct",
            "fourth_down_pct",
            "red_zone_pct",
            "sacks_total",
            "penalty_yards",
            "rushings_yards_per_rush",
            "possession_time",
            "total_drives",
        ]
        num_cols_mv = [
            "team_score",
            "points_against_total",
            "yards_total",
            "plays_total",
            "turnovers_total",
            "yards_per_play_calc",
            "passing_yards_per_pass",
            "passing_interceptions",
            "third_down_pct",
            "fourth_down_pct",
            "red_zone_pct",
            "sacks_total",
            "penalty_yards",
            "rushings_yards_per_rush",
            "total_drives",
        ]

        df = self._execute_paginated_query(
            MV_TABLE, mv_req_cols, "game_date", start_date
        )

        if df.empty:
            logger.warning(
                "MV %s returned no rows; falling back to legacy tables join.",
                MV_TABLE,
            )
            # ---------- Fallback path: compose per-game dataset in Python ----------
            # 1) filter games by date
            games = self._execute_paginated_query(
                "nfl_historical_game_stats",
                ["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"],
                "game_date",
                start_date,
            )
            if games.empty:
                logger.error("Fallback failed: no rows in nfl_historical_game_stats.")
                return pd.DataFrame()

            games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
            games = games.dropna(subset=["game_date"])
            self._cast_ids(games, ["game_id", "home_team_id", "away_team_id"])

            # 2) team-game stats (use updated_at as recency filter)
            team = self._execute_paginated_query(
                "nfl_historical_game_team_stats",
                [
                    "game_id",
                    "team_id",
                    "yards_total",
                    "plays_total",
                    "turnovers_total",
                    "possession_time",
                    "passing_yards_per_pass",
                    "passing_interceptions",
                    "third_down_pct",
                    "fourth_down_pct",
                    "red_zone_pct",
                    "sacks_total",
                    "penalty_yards",
                    "rushings_yards_per_rush",
                    "points_against_total",
                    "total_drives",
                    "updated_at",
                ],
                "updated_at",
                start_date,
            )
            if team.empty:
                logger.error("Fallback failed: no rows in nfl_historical_game_team_stats.")
                return pd.DataFrame()

            # 3) join games -> team, compute team_score from home/away
            self._cast_ids(team, ["game_id", "team_id"])

            # Convert possession_time to minutes
            team["possession_time"] = team.get("possession_time", 0).apply(self._parse_possession_time)

            # Coerce numerics (tolerant)
            for c in [
                "yards_total", "plays_total", "turnovers_total",
                "passing_yards_per_pass", "passing_interceptions",
                "third_down_pct", "fourth_down_pct", "red_zone_pct",
                "sacks_total", "penalty_yards", "rushings_yards_per_rush",
                "points_against_total", "total_drives",
            ]:
                if c in team.columns:
                    team[c] = pd.to_numeric(team[c], errors="coerce")

            merged = team.merge(
                games[["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"]],
                on="game_id",
                how="left",
            )

            # team_score from side; if team_id matches neither side, set NaN
            merged["team_score"] = merged.apply(
                lambda r: r["home_score"]
                if pd.notna(r.get("team_id")) and r["team_id"] == r["home_team_id"]
                else (r["away_score"] if pd.notna(r.get("team_id")) and r["team_id"] == r["away_team_id"] else float("nan")),
                axis=1,
            )

            # yards_per_play_calc (avoid vendor quirks)
            merged["yards_per_play_calc"] = merged.apply(
                lambda r: (float(r["yards_total"]) / float(r["plays_total"]))
                if pd.notna(r["yards_total"]) and pd.notna(r["plays_total"]) and float(r["plays_total"]) > 0
                else float("nan"),
                axis=1,
            )

            # finalize columns to MV parity
            out_cols = mv_req_cols
            for col in out_cols:
                if col not in merged.columns:
                    merged[col] = float("nan") if col in num_cols_mv else None

            df = merged[out_cols + ["home_team_id", "away_team_id"]].copy()
            # points_against_total fallback from opponent score if missing
            mask_pa_missing = df["points_against_total"].isna()
            if mask_pa_missing.any():
                # need home/away team ids; they exist in 'merged'
                df.loc[mask_pa_missing, "points_against_total"] = merged.loc[mask_pa_missing].apply(
                    lambda r: r["away_score"] if r["team_id"] == r["home_team_id"]
                    else (r["home_score"] if r["team_id"] == r["away_team_id"] else float("nan")),
                    axis=1,
                )
            # drop helper ids
            df = df.drop(columns=["home_team_id", "away_team_id"], errors="ignore")

        # ---- Normalize the final frame (both MV and fallback arrive here) ----
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"])

        # Possession time string -> minutes (MV already leaves it as text)
        if "possession_time" in df.columns:
            df["possession_time"] = df["possession_time"].apply(self._parse_possession_time)
        else:
            df["possession_time"] = 0.0

        # Coerce numerics & fill safe zeros (do NOT auto-fill plays/drives)
        for c in num_cols_mv:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        fill_zero = [c for c in num_cols_mv if c not in ("plays_total", "total_drives")]
        df[fill_zero] = df[fill_zero].fillna(0.0)

        # If vendor 'total_drives' looks non-per-game (too low/high), treat as missing.
        if "total_drives" in df.columns:
            bad = (df["total_drives"] < 7) | (df["total_drives"] > 20)
            if bad.any():
                logger.warning(
                    "Sanitizing total_drives: %d rows outside [7,20]; setting to NaN for safe imputation.",
                    int(bad.sum())
                )
            df.loc[bad, "total_drives"] = float("nan")


        # IDs & ordering
        self._cast_ids(df, ["game_id", "team_id"])

        # Minimal sanity checks
        try:
            dupe = df.duplicated(subset=["game_id", "team_id"]).sum()
            if dupe:
                logger.warning("team-game stats: %d duplicate (game_id,team_id) rows; keeping last.", dupe)
                df = df.sort_values(["game_id", "team_id", "game_date"]).drop_duplicates(["game_id", "team_id"], keep="last")

            plays_med = float(df["plays_total"].median(skipna=True)) if "plays_total" in df.columns else 0.0
            drv_med   = float(df["total_drives"].median(skipna=True)) if "total_drives" in df.columns else 0.0
            if plays_med and (plays_med < 40 or plays_med > 90):
                logger.warning("team-game stats: plays_total median=%.1f looks unusual (expected ~55–75).", plays_med)
            if drv_med and (drv_med < 6 or drv_med > 16):
                logger.warning("team-game stats: total_drives median=%.1f looks unusual (expected ~9–13).", drv_med)
        except Exception:
            pass

        df = df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)
        logger.info("Fetched %d team-game stat rows (per-game core).", len(df))
        return df

    def _fetch_team_directory(self) -> pd.DataFrame:
        """
        Try a couple of possible team tables and return id -> team_name map.
        """
        for table in ("nfl_teams_dim", "nfl_teams"):
            try:
                resp = self.supabase.table(table).select("id, team_name").execute()
                if hasattr(resp, "data") and resp.data:
                    df = pd.DataFrame(resp.data)
                    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
                    return df[["id", "team_name"]]
            except Exception:
                continue
        logger.warning("No team directory table found (tried nfl_teams_dim, nfl_teams). Team names will be empty.")
        return pd.DataFrame(columns=["id", "team_name"])

    def fetch_upcoming_games(self, days_ahead: Optional[int] = None) -> pd.DataFrame:
        """
        Schedule from 'nfl_game_schedule'. If days_ahead is None, fetch ALL rows (source of truth).
        Otherwise, fetch bounded window [today, today+days_ahead].
        """
        try:
            qb = self.supabase.table("nfl_game_schedule").select(
                "game_id, game_date, home_team_id, away_team_id"
            )
            if days_ahead is not None:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                future = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                logger.info("Fetching upcoming games [%s → %s]...", today, future)
                qb = qb.gte("game_date", today).lte("game_date", future)
            else:
                logger.info("Fetching ALL games from nfl_game_schedule (no date filter)…")

            resp = qb.order("game_date", desc=False).execute()
            if not hasattr(resp, "data"):
                raise ValueError("Response from 'nfl_game_schedule' missing 'data'.")

            df = pd.DataFrame(resp.data or [])
            if df.empty:
                logger.info("No rows found in nfl_game_schedule.")
                return df

            # Types & sort
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            self._cast_ids(df, ["game_id", "home_team_id", "away_team_id"])
            df = df.sort_values("game_date").reset_index(drop=True)

            # Optional: enrich with team names via separate lookup
            team_dir = self._fetch_team_directory()
            if not team_dir.empty:
                df = df.merge(
                    team_dir.rename(columns={"id": "home_team_id", "team_name": "home_team_name"}),
                    on="home_team_id",
                    how="left",
                )
                df = df.merge(
                    team_dir.rename(columns={"id": "away_team_id", "team_name": "away_team_name"}),
                    on="away_team_id",
                    how="left",
                )
            else:
                df["home_team_name"] = None
                df["away_team_name"] = None

            cols = [
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
            ]
            return df[cols]

        except Exception:
            logger.error("Failed to fetch upcoming/all games.\n%s", traceback.format_exc())
            return pd.DataFrame()

