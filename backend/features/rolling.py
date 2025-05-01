# backend/features/rolling.py

from __future__ import annotations
from typing import Any
import pandas as pd
import logging

from .utils import DEFAULTS
from .base_windows import fetch_rolling  # your SQL fetch helper
from .legacy.feature_engineering import FeatureEngine as LegacyFeatureEngine

logger = logging.getLogger(__name__)

# Spin up one legacy engine for fallback reuse
try:
    _legacy_engine = LegacyFeatureEngine()
    logger.info("Rolling: Legacy FeatureEngine instantiated for fallback.")
except Exception:
    logger.exception("Rolling: failed to init legacy FeatureEngine—SQL path only." )
    _legacy_engine = None

__all__ = ["transform"]


def _legacy_fallback(df: pd.DataFrame, window_sizes: list[int], reason: str) -> pd.DataFrame:
    logger.warning(f"Rolling: falling back to legacy Python (reason: {reason})")
    if not _legacy_engine:
        logger.error("Rolling: no legacy engine available, returning original DataFrame.")
        return df
    try:
        return _legacy_engine.add_rolling_features(df, window_sizes=window_sizes)
    except Exception:
        logger.exception("Rolling: error in legacy fallback—returning input df.")
        return df


def transform(
    df: pd.DataFrame,
    *,
    conn: Any | None = None,
    window_sizes: list[int] = [5, 10, 20],
    debug: bool = False
) -> pd.DataFrame:
    """
    Primary: fetch precomputed rolling stats via `fetch_rolling(conn, game_ids)`.
    Fallback: call legacy Python implementation.
    """
    if df is None or df.empty:
        if debug:
            logger.debug("Rolling.transform: empty input, returning as-is.")
        return df

    # Make sure we have game_id & normalized team columns
    missing = [col for col in ("game_id", "home_team_norm", "away_team_norm") if col not in df.columns]
    if missing:
        return _legacy_fallback(df, window_sizes, f"missing columns {missing}")

    # If no DB connection, skip SQL
    if conn is None:
        return _legacy_fallback(df, window_sizes, "no DB connection")

    # Attempt SQL fetch
    try:
        ids = df["game_id"].astype(str).unique().tolist()
        rolled = fetch_rolling(conn, ids)

        if rolled is None or rolled.empty:
            return _legacy_fallback(df, window_sizes, "SQL returned no rows")

        # Prepare merges
        rolled["game_id"] = rolled["game_id"].astype(str)
        rolled["team_norm"] = rolled["team_norm"].astype(str)

        # Build rename maps
        stats_cols = [c for c in rolled.columns if c not in ("game_id", "team_norm", "game_date")]
        home_map = {c: f"home_{c}" for c in stats_cols}
        away_map = {c: f"away_{c}" for c in stats_cols}

        # Merge home
        out = df.copy()
        out["game_id"] = out["game_id"].astype(str)
        out = pd.merge(
            out,
            rolled.rename(columns=home_map).drop(columns=["team_norm","game_date"], errors="ignore"),
            on=["game_id","home_team_norm"],
            how="left",
        )
        # Merge away
        out = pd.merge(
            out,
            rolled.rename(columns=away_map).drop(columns=["team_norm","game_date"], errors="ignore"),
            on=["game_id","away_team_norm"],
            how="left",
        )

        # Fill any NaNs in newly added cols with defaults
        new_cols = list(home_map.values()) + list(away_map.values())
        for col in new_cols:
            # derive default key: strip prefix/home_/away_
            key = col.split("_",2)[-1]
            if key.endswith("_std"):
                default = DEFAULTS.get(key, 0.0)
            else:
                default = DEFAULTS.get(key, 0.0)
            out[col] = out[col].fillna(default)
            if key.endswith("_std"):
                # no negatives on std
                out[col] = out[col].clip(lower=0)

        if debug:
            logger.debug(f"Rolling.transform: merged {len(rolled)} rows, added {len(new_cols)} cols.")

        return out

    except Exception as e:
        logger.exception(f"Rolling: SQL path error ({e}), falling back.")
        return _legacy_fallback(df, window_sizes, f"sql error {e}")
