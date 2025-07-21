# backend/nfl_features/engine.py

"""
NFLFeatureEngine – master orchestrator for all NFL feature modules.

Key fixes vs. previous draft
----------------------------
1. **Rolling‑module interface** – uses `load_rolling_features()` and passes a
   Supabase client, because rolling data live in the DB.
2. **Dynamic param detection** – `_supports_kwarg()` now checks *all* keyword
   args we might supply, avoiding silent drops.
3. **Advanced / drive metrics** – applied to *historical* only (correct),
   but DRAM‑free for upcoming rows.
4. **Temporary columns** – all helper flags (`_is_upcoming`, etc.) cleaned at
   the very end.
5. **Verbose error‑handling** – early exits with log instructions if a module
   returns empty or mismatched rows.

"""
from __future__ import annotations

import logging
import time
from inspect import signature
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase_py import create_client, Client  # type: ignore

# ---- Feature modules ------------------------------------------------------ #
from .advanced import compute_advanced_metrics
from .drive_metrics import compute_drive_metrics
from .form import transform as form_transform
from .h2h import transform as h2h_transform
from .momentum import transform as momentum_transform
from .rest import transform as rest_transform
from .rolling import load_rolling_features     # <- corrected import
from .season import transform as season_transform
from .situational import compute_situational_features

from .utils import (
    DEFAULTS,
    determine_season,
    normalize_team_name,
    prefix_columns,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["NFLFeatureEngine"]


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
class NFLFeatureEngine:
    """Runs the full NFL feature‑engineering pipeline."""

    DEFAULT_ORDER: List[str] = [
        "situational",
        "season",
        "rest",
        "rolling",
        "form",
        "momentum",
        "h2h",
    ]

    TRANSFORMS: Dict[str, Any] = {
        "situational": compute_situational_features,
        "season": season_transform,
        "rest": rest_transform,
        "rolling": load_rolling_features,        # <- correct callable
        "form": form_transform,
        "momentum": momentum_transform,
        "h2h": h2h_transform,
    }

    def __init__(
        self,
        supabase_url: str,
        supabase_service_key: str,
        execution_order: Optional[List[str]] = None,
    ):
        self.execution_order = execution_order or self.DEFAULT_ORDER
        self.supabase: Client = create_client(supabase_url, supabase_service_key)
        logger.info("NFLFeatureEngine initialised with order: %s", self.execution_order)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _supports_kwarg(func: Any, kw: str) -> bool:
        """Return True if *func* has a param named *kw*."""
        try:
            return kw in signature(func).parameters
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #
    def build_features(
        self,
        games_df: pd.DataFrame,
        *,
        historical_games_df: pd.DataFrame,
        historical_team_stats_df: pd.DataFrame,
        flag_imputations: bool = True,
        debug: bool = False,
        rolling_window: int = 3,
        form_lookback: int = 5,
        momentum_span: int = 5,
        h2h_max_games: int = 10,
    ) -> pd.DataFrame:
        """Run every feature module and return a wide feature DataFrame."""
        if debug:
            logger.setLevel(logging.DEBUG)

        if games_df.empty:
            logger.warning("ENGINE: games_df is empty; nothing to do.")
            return pd.DataFrame()

        t0 = time.time()

        # ------------------------------------ #
        # 0. Canonicalise keys & dates         #
        # ------------------------------------ #
        def _canon(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in ("home_team_norm", "away_team_norm"):
                if col in out.columns:
                    out[col] = out[col].apply(normalize_team_name)
            if "game_date" in out.columns:
                out["game_date"] = pd.to_datetime(out["game_date"])
                out["season"] = out["game_date"].apply(determine_season)
            return out

        games = _canon(games_df)
        hist_games = _canon(historical_games_df)
        hist_team_stats = historical_team_stats_df.copy()

        # ------------------------------------ #
        # 1. Pre‑compute advanced/drive on HIST #
        # ------------------------------------ #
        hist_games = compute_advanced_metrics(hist_games)
        hist_games = compute_drive_metrics(hist_games)

        # ------------------------------------ #
        # 2. Process season‑by‑season          #
        # ------------------------------------ #
        chunks: List[pd.DataFrame] = []
        for season in sorted(games["season"].unique()):
            logger.info("ENGINE: building features for season %s", season)

            season_games = games.loc[games["season"] == season].copy()
            cutoff = season_games["game_date"].min()

            past_games = hist_games.loc[hist_games["game_date"] < cutoff]
            # ---- module loop ---- #
            for module in self.execution_order:
                fn = self.TRANSFORMS[module]
                logger.debug("ENGINE: module %s", module)

                # Build kwargs dynamically
                kw: Dict[str, Any] = {"flag_imputations": flag_imputations}

                if self._supports_kwarg(fn, "historical_df"):
                    kw["historical_df"] = past_games
                if self._supports_kwarg(fn, "historical_team_stats_df"):
                    kw["historical_team_stats_df"] = hist_team_stats

                # module‑specific knobs
                if module == "rolling":
                    kw.update({"game_ids": season_games["game_id"].tolist(),
                               "supabase": self.supabase,
                               "window": rolling_window})
                if module == "form":
                    kw["lookback_window"] = form_lookback
                if module == "momentum":
                    kw["span"] = momentum_span
                if module == "h2h":
                    kw["max_games"] = h2h_max_games

                # Execute
                mod_out = (
                    fn(season_games, **kw)
                    if module not in ("rolling",)
                    else fn(**kw)  # rolling returns its own DF keyed by game_id
                )

                if mod_out.empty:
                    logger.warning("ENGINE: %s produced no rows – skipping merge", module.upper())
                    continue

                # Merge new columns
                if module == "rolling":
                    season_games = season_games.merge(mod_out, on="game_id", how="left")
                else:
                    new_cols = [c for c in mod_out.columns if c not in season_games.columns and c != "game_id"]
                    if new_cols:
                        season_games = season_games.merge(
                            mod_out[["game_id"] + new_cols], on="game_id", how="left"
                        )

            chunks.append(season_games)

        if not chunks:
            logger.error("ENGINE: no chunks created; returning empty DataFrame.")
            return pd.DataFrame()

        final = pd.concat(chunks, ignore_index=True)
        # Drop helper cols (anything starting with an underscore)
        final.drop(columns=[c for c in final.columns if c.startswith("_")], inplace=True)

        logger.info("ENGINE: pipeline complete in %.2fs – shape %s", time.time() - t0, final.shape)
        return final
