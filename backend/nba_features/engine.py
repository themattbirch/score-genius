"""
Orchestrates the execution of the modular feature-engineering pipeline for NBA games.
Each module exports a `transform(df, **kwargs)` function; we call them in order.
The engine can run fully offline (pass your own DataFrames) or auto-fetch the
needed helpers (season stats, historical games, rolling windows) from Supabase
when you hand in a `db_conn`.
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────── .env loading ──────────────────────────────
# Look for .env in backend/ or project root – first hit wins.
for _env in (
    Path(__file__).resolve().parents[1] / ".env",  # …/backend/.env
    Path(__file__).resolve().parents[2] / ".env",  # project-root /.env
):
    if _env.is_file():
        load_dotenv(_env, override=True)
        break

# ────────────────────────────────── Imports ─────────────────────────────────
import logging
import time
from typing import Any, Optional, List

import pandas as pd
import numpy as np  # noqa: F401 – downstream modules may expect np present

# Feature modules
from .advanced import transform as advanced_transform
from .rolling import transform as rolling_transform
from .rest import transform as rest_transform
from .h2h import transform as h2h_transform
from .season import transform as season_transform
from .form import transform as form_transform

# Rolling helper
from .base_windows import fetch_rolling

# Optional Supabase client singleton
from caching.supabase_client import supabase as supabase_client

# ─────────────────────────────── Logger setup ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s:%(funcName)s:%(lineno)d] – %(message)s",
)
logger = logging.getLogger(__name__)

__all__: list[str] = ["run_feature_pipeline"]

# ─────────────────────────────── Constants ─────────────────────────────────
DEFAULT_EXECUTION_ORDER: list[str] = [
    "advanced",
    "rest",
    "h2h",
    "form",
    "season",
    "rolling",
]

TRANSFORM_MAP: dict[str, callable] = {
    "advanced": advanced_transform,
    "rest": rest_transform,
    "h2h": h2h_transform,
    "form": form_transform,
    "season": season_transform,
    "rolling": rolling_transform,
}

# ───────────────────────────── Helper fetchers ─────────────────────────────

def _fetch_team_stats(db):
    """Pull season-to-date team stats from Supabase."""
    rows = (
        db.table("nba_historical_team_stats")
        .select("*")
        .execute()
        .data
    )
    return pd.DataFrame(rows or [])


def _fetch_historical_games(db):
    """Pull full historical game log for H2H calculations."""
    rows = (
        db.table("nba_historical_game_stats")
        .select("*")
        .execute()
        .data
    )
    return pd.DataFrame(rows or [])

# ───────────────────────────── Main entrypoint ─────────────────────────────

def run_feature_pipeline(
    df: pd.DataFrame,
    *,
    db_conn: Optional[Any] = None,
    historical_games_df: pd.DataFrame | None = None,
    team_stats_df: pd.DataFrame | None = None,
    rolling_windows: List[int] | None = None,
    h2h_window: int = 7,
    execution_order: List[str] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Run all feature-engineering modules and return the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``game_id``, ``game_date``, ``home_team``, ``away_team``.
    db_conn : Any, optional
        Supabase client. If provided and helper DataFrames are missing, we fetch them.
    historical_games_df : pd.DataFrame, optional
        Pre-loaded game log. Required when the ``h2h`` module runs.
    team_stats_df : pd.DataFrame, optional
        Season-to-date team stats. Required when the ``season`` or ``form`` modules run.
    rolling_windows : list[int]
        Window sizes for the rolling module. Empty list disables the module.
    h2h_window : int
        How many prior games per matchup to consider in the H2H transform.
    execution_order : list[str]
        Override the default module order.
    debug : bool
        Toggle extra logging & tracebacks.
    """

    rolling_windows = rolling_windows or [5, 10, 20]
    execution_order = execution_order or DEFAULT_EXECUTION_ORDER.copy()

    # Disable modules if configured off
    if not rolling_windows and "rolling" in execution_order:
        execution_order.remove("rolling")

    # Logging verbosity
    original_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.info("Starting feature pipeline → %s", execution_order)

    # Sanity-check the input DataFrame
    required = ["game_id", "game_date", "home_team", "away_team"]
    if df is None or df.empty:
        logger.error("Input DataFrame is empty – nothing to process.")
        return pd.DataFrame()
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        return pd.DataFrame()

    processed_df = df.copy()

    # ─── Auto-fetch helper tables ────────────────────────────────────────────
    if db_conn is not None:
        if team_stats_df is None and "season" in execution_order:
            logger.info("Auto-fetching team stats from Supabase …")
            team_stats_df = _fetch_team_stats(db_conn)
        if historical_games_df is None and "h2h" in execution_order:
            logger.info("Auto-fetching historical games from Supabase …")
            historical_games_df = _fetch_historical_games(db_conn)

        # Pre-fetch rolling stats and merge before rolling transform
        if "rolling" in execution_order:
            logger.info("Pre-fetching rolling stats from Supabase …")
            try:
                game_ids = processed_df['game_id'].unique().tolist()
                rolling_df = fetch_rolling(db_conn, game_ids)
                if {'game_id','team_id'}.issubset(rolling_df.columns):
                    processed_df = processed_df.merge(
                        rolling_df,
                        on=['game_id','team_id'],
                        how='left'
                    )
                else:
                    logger.warning(
                        "Skipping rolling pre-merge: missing 'game_id' or 'team_id' in rolling_df columns: %s",
                        list(rolling_df.columns)
                    )
            except Exception as e:
                logger.error("Error fetching or merging rolling stats: %s", e)

    # ─── Execute modules ────────────────────────────────────────────────────
    start_total = time.time()
    for module in execution_order:
        func = TRANSFORM_MAP.get(module)
        if func is None:
            logger.warning("Unknown module '%s' – skipping", module)
            continue

        logger.info("Running '%s'…", module)
        t0 = time.time()

        kwargs: dict[str, Any] = {"debug": debug}
        if module == "rolling":
            kwargs["window_sizes"] = rolling_windows
        elif module == "h2h":
            kwargs.update({
                "historical_df": historical_games_df,
                "max_games": h2h_window,
            })
        elif module in ("season", "form"):
            kwargs["team_stats_df"] = team_stats_df

        try:
            processed_df = func(processed_df, **kwargs)
        except Exception as e:
            logger.error("'%s' error: %s", module, e, exc_info=debug)
            break

        logger.debug(
            "%s done in %.2fs – rows=%s cols=%s",
            module,
            time.time() - t0,
            *processed_df.shape,
        )
        if processed_df.empty:
            logger.error("DataFrame empty after '%s' – stopping.", module)
            break

    logger.info(
        "Pipeline complete in %.2fs – final shape=%s",
        time.time() - start_total,
        processed_df.shape,
    )
    if debug:
        logger.setLevel(original_level)
    return processed_df

# ─────────────────────────────── CLI smoke test ─────────────────────────────
if __name__ == "__main__":
    logger.info("Running smoke test with dummy data…")

    dummy_games = pd.DataFrame({
        "game_id": [101, 102],
        # Use standard ASCII hyphens
        "game_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "home_team": ["BOS", "LAL"],
        "away_team": ["NYK", "CHI"],
        # Dummy team IDs for rolling merge
        "team_id": [1, 2],
    })

    features = run_feature_pipeline(
        dummy_games,
        db_conn=supabase_client,
        debug=True,
    )

    print("Generated columns:", len(features.columns))
