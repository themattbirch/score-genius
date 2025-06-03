# backend/mlb_features/rest.py

"""
Calculates rest-based features for MLB games.

Focus
-----
1.  Strict “no-look-ahead” guarantee when multiple games share a date.
2.  Correct rest-day math, B2B flags, 7-/14-day density, and advantages.
3.  Idempotency + NaN-free outputs.

MLB Specifics:
-   Uses 'game_date_et' for game dates.
-   Uses 'home_team_id' and 'away_team_id' for team identification.
-   Rest and schedule density metrics are generally applicable, though MLB game
    frequency is high. B2B (rest_days == 1) is a common scenario.
"""

from __future__ import annotations
import logging
from typing import List, Dict as TypingDict, Optional

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS_IMPORT
    MLB_DEFAULTS: TypingDict[str, float] = MLB_DEFAULTS_IMPORT
except ImportError:
    logger.warning("Could not import DEFAULTS from .utils. Using local fallbacks.")
    MLB_DEFAULTS: TypingDict[str, float] = {}

# If DEFAULTS does not have "mlb_rest_days", fall back to 3.0
DEF_REST_MLB = float(MLB_DEFAULTS.get("mlb_rest_days", 3.0))

PLACEHOLDER_COLS: List[str] = [
    "rest_days_home",
    "rest_days_away",
    "games_last_7_days_home",
    "games_last_14_days_home",
    "games_last_7_days_away",
    "games_last_14_days_away",
    "is_back_to_back_home",
    "is_back_to_back_away",
    "rest_advantage",
    "schedule_advantage",
]


def _fill_defaults_mlb(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure placeholder columns exist and contain no NaNs with correct dtypes."""
    df_filled = df.copy()
    for col in PLACEHOLDER_COLS:
        if col.startswith("is_back_to_back"):
            default_val = 0
            dtype = int
        elif col.startswith("games_last_"):
            default_val = 0
            dtype = int
        elif col.startswith("rest_days"):
            default_val = DEF_REST_MLB
            dtype = float
        else:  # advantages
            default_val = 0.0
            dtype = float

        if col not in df_filled.columns:
            df_filled[col] = default_val
        else:
            df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce").fillna(default_val)

        df_filled[col] = df_filled[col].astype(dtype)

    return df_filled


def transform(
    df: pd.DataFrame, *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Calculates rest and schedule-based features for MLB games.

    Args:
        df: DataFrame with columns:
            - 'game_id'
            - 'game_date_et'
            - 'home_team_id', 'away_team_id'
        debug: enable DEBUG logging for this run.

    Returns:
        df with all PLACEHOLDER_COLS added and original key columns preserved.
    """
    orig_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Starting mlb_features.rest.transform in DEBUG mode")

    if df is None:
        logger.warning(
            "Null input DataFrame to rest.transform; returning empty DataFrame with placeholders."
        )
        return _fill_defaults_mlb(pd.DataFrame(columns=PLACEHOLDER_COLS))

    # --- Map original → internal and internal → original column names ---
    col_map_to_internal = {
        "game_id": "game_id",
        "game_date_et": "game_date",
        "home_team_id": "home_team",
        "away_team_id": "away_team",
    }
    col_map_to_original = {v: k for k, v in col_map_to_internal.items()}

    # Check for required original columns
    missing_original_cols = [
        orig_col for orig_col in col_map_to_internal if orig_col not in df.columns
    ]
    if missing_original_cols:
        logger.error(
            f"Missing required columns for rest.py: {missing_original_cols}. "
            "Input DataFrame returned with default placeholders."
        )
        df_to_fill = df.copy()
        return _fill_defaults_mlb(df_to_fill)

    # Drop any prior runs of placeholders or intermediate columns (idempotency)
    intermediate_cols_to_drop = ["prev_home_game_date", "prev_away_game_date"]
    cols_to_drop_for_idempotency = PLACEHOLDER_COLS + intermediate_cols_to_drop
    current_df = df.copy()
    existing_to_drop = [c for c in cols_to_drop_for_idempotency if c in current_df.columns]
    if existing_to_drop:
        current_df = current_df.drop(columns=existing_to_drop, errors="ignore")

    # If both "game_date" and "game_date_et" exist, drop old "game_date" first
    if "game_date" in current_df.columns and "game_date_et" in current_df.columns:
        current_df = current_df.drop(columns=["game_date"])

    # Rename to internal names (this maps "game_date_et" → "game_date")
    current_df = current_df.rename(columns=col_map_to_internal)

    # Convert "game_date" to datetime and drop invalid rows
    current_df["game_date"] = pd.to_datetime(
        current_df["game_date"], errors="coerce"
    ).dt.tz_localize(None)

    if current_df["game_date"].isna().any():
        n_bad = int(current_df["game_date"].isna().sum())
        logger.warning(f"Dropping {n_bad} rows with invalid game_date")
        current_df = current_df.dropna(subset=["game_date"])
        if current_df.empty:
            logger.warning("DataFrame empty after dropping NaNs in game_date.")
            empty_with_placeholders = _fill_defaults_mlb(
                pd.DataFrame(
                    columns=list(col_map_to_original.values()) + PLACEHOLDER_COLS
                )
            )
            return empty_with_placeholders

    if current_df.empty:
        logger.warning(
            "Empty input to rest.transform; returning placeholders only."
        )
        original_keys_present = [k for k in df.columns if k in col_map_to_internal]
        return _fill_defaults_mlb(pd.DataFrame(columns=original_keys_present + PLACEHOLDER_COLS))

    # Ensure team columns are strings
    current_df["home_team"] = current_df["home_team"].astype(str)
    current_df["away_team"] = current_df["away_team"].astype(str)
    current_df = current_df.reset_index(drop=True)

    # ─── Build a “long” team–game log using a temporary column "team_temp" ───
    home_log = current_df[["game_id", "game_date", "home_team"]].rename(
        columns={"home_team": "team_temp"}
    )
    away_log = current_df[["game_id", "game_date", "away_team"]].rename(
        columns={"away_team": "team_temp"}
    )
    full_log = pd.concat([home_log, away_log], ignore_index=True)

    # Compute each team’s previous game_date (drop duplicates to handle same-day multiple games)
    prev_log = (
        full_log.sort_values(["team_temp", "game_date", "game_id"], ignore_index=True)
        .drop_duplicates(["team_temp", "game_date"], keep="first")
    )
    prev_log["prev_game_date"] = prev_log.groupby("team_temp")["game_date"].shift(1)
    prev_map = prev_log.set_index(["team_temp", "game_date"])["prev_game_date"]

    # Merge prev_game_date for home and away
    for side in ("home", "away"):
        current_df = (
            current_df.merge(
                prev_map.rename("prev_game_date_temp"),
                left_on=[f"{side}_team", "game_date"],
                right_index=True,
                how="left",
            )
            .rename(columns={"prev_game_date_temp": f"prev_{side}_game_date"})
        )

    # Calculate raw rest days
    current_df["rest_days_home"] = (
        current_df["game_date"] - current_df["prev_home_game_date"]
    ).dt.days
    current_df["rest_days_away"] = (
        current_df["game_date"] - current_df["prev_away_game_date"]
    ).dt.days

    # Identify each team’s first game_id, then force DEF_REST_MLB on that game
    first_game_for_team_map = (
        full_log.sort_values(["team_temp", "game_date", "game_id"])
        .drop_duplicates("team_temp", keep="first")
        .set_index("team_temp")["game_id"]
    )
    home_first_ids = current_df["home_team"].map(first_game_for_team_map)
    away_first_ids = current_df["away_team"].map(first_game_for_team_map)

    mask_home_first = current_df["game_id"] == home_first_ids
    mask_away_first = current_df["game_id"] == away_first_ids

    current_df.loc[mask_home_first, "rest_days_home"] = DEF_REST_MLB
    current_df.loc[mask_away_first, "rest_days_away"] = DEF_REST_MLB

    # Fill any remaining NaNs in rest_days with DEF_REST_MLB
    current_df["rest_days_home"] = current_df["rest_days_home"].fillna(DEF_REST_MLB)
    current_df["rest_days_away"] = current_df["rest_days_away"].fillna(DEF_REST_MLB)

    # ─── Schedule density: 7/14-day window (Exclude current game_date) ───
    sched_log = (
        full_log.drop_duplicates(["team_temp", "game_date"], keep="first")
        .set_index("game_date")
        .sort_index()
    )
    if not sched_log.empty:
        cnt7 = (
            sched_log.groupby("team_temp")["game_id"]
            .rolling("7D", closed="left")
            .count()
            .rename("cnt7")
        )
        cnt14 = (
            sched_log.groupby("team_temp")["game_id"]
            .rolling("14D", closed="left")
            .count()
            .rename("cnt14")
        )
        sched_counts = pd.concat([cnt7, cnt14], axis=1).reset_index()
        sched_counts = sched_counts.set_index(["team_temp", "game_date"])

        for side in ("home", "away"):
            # Prepare temp DataFrame indexed by (team_temp, game_date)
            temp = sched_counts[["cnt7", "cnt14"]].rename(
                columns={
                    "cnt7": f"games_last_7_days_{side}",
                    "cnt14": f"games_last_14_days_{side}",
                }
            )
            # Step 1: set index on current_df to (home_team/away_team, game_date)
            idx_df = current_df.set_index([f"{side}_team", "game_date"])
            # Step 2: rename the index level so it matches 'team_temp'
            idx_df.index.set_names(["team_temp", "game_date"], inplace=True)
            # Step 3: join the rolling counts
            joined = idx_df.join(temp, how="left").reset_index()
            # Step 4: rename 'team_temp' column back to f"{side}_team"
            joined = joined.rename(columns={"team_temp": f"{side}_team"})
            current_df = joined

            # Fill any NaNs with 0 and ensure int dtype
            current_df[f"games_last_7_days_{side}"] = current_df[
                f"games_last_7_days_{side}"
            ].fillna(0).astype(int)
            current_df[f"games_last_14_days_{side}"] = current_df[
                f"games_last_14_days_{side}"
            ].fillna(0).astype(int)
    else:
        for side in ("home", "away"):
            current_df[f"games_last_7_days_{side}"] = 0
            current_df[f"games_last_14_days_{side}"] = 0

    # B2B flags and advantage metrics
    current_df["is_back_to_back_home"] = (current_df["rest_days_home"] == 1.0).astype(int)
    current_df["is_back_to_back_away"] = (current_df["rest_days_away"] == 1.0).astype(int)
    current_df["rest_advantage"] = (
        current_df["rest_days_home"] - current_df["rest_days_away"]
    )
    current_df["schedule_advantage"] = (
        current_df.get("games_last_7_days_away", 0)
        - current_df.get("games_last_7_days_home", 0)
    )

    # Drop intermediate columns
    current_df.drop(
        columns=["prev_home_game_date", "prev_away_game_date"], errors="ignore", inplace=True
    )

    # Final defaults + type enforcement
    out_df = _fill_defaults_mlb(current_df)

    # ─── Rename internal columns back to original names ───
    out_df = out_df.rename(columns=col_map_to_original, errors="ignore")

    logger.debug("Finished rest.transform; output shape %s", out_df.shape)
    if debug:
        logger.setLevel(orig_level)
    return out_df
