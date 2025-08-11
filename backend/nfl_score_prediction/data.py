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
        """
        Generic paginated select with >= start_date filter on `date_column`.
        """
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
                resp = (
                    self.supabase.table(table)
                    .select(select_str)
                    .gte(date_column, start_date)
                    .order(date_column, desc=False)
                    .range(start, start + page_size - 1)
                    .execute()
                )

                if not hasattr(resp, "data"):
                    raise ValueError(f"Supabase response from '{table}' missing 'data'.")

                batch = resp.data or []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                start += page_size

            df = pd.DataFrame(all_rows)
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
        Team-level boxscore stats per game from 'nfl_historical_game_team_stats'.
        Uses 'updated_at' as proxy for recency unless a dedicated date exists.
        """
        now = datetime.now()
        start_date = (now - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
        logger.info("Fetching team-game stats since %s...", start_date)

        req_cols = [
            "game_id",
            "team_id",
            "turnovers_total",
            "yards_total",
            "possession_time",
            "passing_yards_per_pass",
            "passing_interceptions",
            "third_down_pct",
            "fourth_down_pct",
            "red_zone_pct",
            "sacks_total",
            "penalty_yards",
            "rushings_yards_per_rush",
            "updated_at",
        ]
        num_cols = [c for c in req_cols if c not in ("game_id", "team_id", "possession_time", "updated_at")]

        df = self._execute_paginated_query(
            "nfl_historical_game_team_stats", req_cols, "updated_at", start_date
        )
        if df.empty:
            logger.warning("No team-game stat data found.")
            return df

        # Possession time to minutes
        df["possession_time"] = df["possession_time"].apply(self._parse_possession_time)

        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        self._cast_ids(df, ["game_id", "team_id"])
        df = df.drop(columns=["updated_at"]).reset_index(drop=True)
        logger.info("Fetched %d team-game stat rows.", len(df))
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

