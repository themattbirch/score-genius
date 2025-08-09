# backend/nfl_features/rest.py

"""
Rest‑day & schedule‑spacing features for NFL games.

For each upcoming matchup we calculate:
  • rest_days             – full days between a team’s previous game and kickoff
  • is_on_short_week      – rest_days ≤ 3   (Thu after Sun / Sat after Mon, etc.)
  • is_off_bye            – rest_days ≥ 13  (the week after a bye)
  • rest_advantage        – home_rest_days – away_rest_days
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .utils import DEFAULTS, normalize_team_name, prefix_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _default_rest(games: pd.DataFrame, *, flag_imputations: bool = True) -> pd.DataFrame:
    base = games[["game_id"]].copy()

    # pass through schedule context so next modules have what they need
    passthru = [c for c in (
        "game_date", "game_time",
        "season",
        "home_team_norm", "away_team_norm",
        "home_score", "away_score",
        "kickoff_ts"
    ) if c in games.columns]
    if passthru:
        base = base.merge(games[["game_id"] + passthru], on="game_id", how="left")

    default_rd = DEFAULTS["days_since_last_game"]
    for side in ("home", "away"):
        base[f"{side}_rest_days"] = default_rd
        base[f"{side}_is_on_short_week"] = 0
        base[f"{side}_is_off_bye"] = 0
        if flag_imputations:
            base[f"{side}_rest_days_imputed"] = 1

    base["rest_advantage"] = 0.0

    # compact dtypes
    for side in ("home", "away"):
        base[f"{side}_rest_days"] = base[f"{side}_rest_days"].astype("float32")
        base[f"{side}_is_on_short_week"] = base[f"{side}_is_on_short_week"].astype("int8")
        base[f"{side}_is_off_bye"] = base[f"{side}_is_off_bye"].astype("int8")
        if flag_imputations:
            base[f"{side}_rest_days_imputed"] = base[f"{side}_rest_days_imputed"].astype("int8")
    base["rest_advantage"] = base["rest_advantage"].astype("float32")
    return base


def _timeline(historical: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Return long DataFrame: one row = team-game, historical+upcoming."""
    hist = historical.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce")
    hist["_is_upcoming"] = False

    upc = upcoming.copy()
    upc["game_date"] = pd.to_datetime(upc["game_date"], errors="coerce")
    upc["_is_upcoming"] = True

    # Canonical team keys for grouping (id/abbr/name → canonical)
    for col in ("home_team_norm", "away_team_norm"):
        if col in hist.columns:
            hist[col] = hist[col].apply(normalize_team_name).astype(str).str.lower()
        if col in upc.columns:
            upc[col] = upc[col].apply(normalize_team_name).astype(str).str.lower()

    combined = pd.concat([hist, upc], ignore_index=True, sort=False)

    home = combined.rename(columns={"home_team_norm": "team"})
    away = combined.rename(columns={"away_team_norm": "team"})

    long_df = pd.concat([home, away], ignore_index=True, sort=False)
    long_df = long_df[["game_id", "game_date", "team", "_is_upcoming"]]
    return long_df.sort_values(["team", "game_date"], kind="mergesort").reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    flag_imputations: bool = True,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    # Early default if history missing
    if historical_df is None or historical_df.empty:
        logger.warning("rest: no historical data – defaulting to 7-day rest.")
        return _default_rest(games, flag_imputations=flag_imputations)

    # ── 1) Normalize upcoming teams & make sure we have game_date ────────────
    games_norm = games.copy()

    # Ensure game_date; derive from kickoff_ts if needed
    if "game_date" not in games_norm.columns or games_norm["game_date"].isna().all():
        if "kickoff_ts" in games_norm.columns:
            ts = pd.to_datetime(games_norm["kickoff_ts"], errors="coerce", utc=True)
            if ts.notna().any():
                games_norm["game_date"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
            else:
                logger.warning("rest: kickoff_ts present but unparsable; using default rest days.")
                return _default_rest(games, flag_imputations=flag_imputations)
        else:
            logger.warning("rest: missing game_date and kickoff_ts; using default rest days.")
            return _default_rest(games, flag_imputations=flag_imputations)
    else:
        games_norm["game_date"] = pd.to_datetime(games_norm["game_date"], errors="coerce")

    # Normalize team keys (id/abbr/name → canonical)
    for col in ("home_team_norm", "away_team_norm"):
        if col in games_norm.columns:
            games_norm[col] = games_norm[col].apply(normalize_team_name).astype(str).str.lower()

    # ── 2) Build timeline from history and upcoming games ────────────────────
    long_df = _timeline(historical_df, games_norm)

    # Compute previous game per team
    grp = long_df.groupby("team", sort=False)
    long_df["prev_game_date"] = grp["game_date"].shift(1)

    # Rest days (full days between previous game and kickoff, excluding both days → −1)
    long_df["rest_days"] = (long_df["game_date"] - long_df["prev_game_date"]).dt.days - 1
    long_df["is_on_short_week"] = (long_df["rest_days"] <= 3).astype("int8")
    long_df["is_off_bye"] = (long_df["rest_days"] >= 13).astype("int8")

    feats = long_df.loc[long_df["_is_upcoming"], ["game_id", "team", "rest_days", "is_on_short_week", "is_off_bye"]]

    # ── 3) Merge per side ────────────────────────────────────────────────────
    merged = {}
    for side, team_col in (("home", "home_team_norm"), ("away", "away_team_norm")):
        df_side = games_norm.merge(
            feats.rename(columns={"team": team_col}),
            on=["game_id", team_col],
            how="left",
        )
        merged[side] = prefix_columns(
            df_side[["game_id", "rest_days", "is_on_short_week", "is_off_bye"]],
            side,
            exclude=["game_id"],
        )

    out = merged["home"].merge(merged["away"], on="game_id")

    # ── 4) Fill defaults & optional imputation flags ─────────────────────────
    default_rd = DEFAULTS["days_since_last_game"]
    for side in ("home", "away"):
        rd_col = f"{side}_rest_days"
        if flag_imputations:
            out[f"{rd_col}_imputed"] = out[rd_col].isna().astype("int8")
        out[rd_col] = out[rd_col].fillna(default_rd).astype("float32")
        for flag_col in (f"{side}_is_on_short_week", f"{side}_is_off_bye"):
            out[flag_col] = out[flag_col].fillna(0).astype("int8")

    out["rest_advantage"] = (out["home_rest_days"] - out["away_rest_days"]).astype("float32")

    # Pass through the same schedule columns
    passthru = [c for c in (
        "game_date", "game_time",
        "season",
        "home_team_norm", "away_team_norm",
        "home_score", "away_score",
        "kickoff_ts"
    ) if c in games_norm.columns]

    if passthru:
        out = out.merge(games_norm[["game_id"] + passthru], on="game_id", how="left")

    new_cols = [c for c in out.columns if c not in ("game_id",)]
    return out[["game_id"] + new_cols]
