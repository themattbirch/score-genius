# backend/mlb_features/h2h.py

"""
Calculates historical head-to-head (H2H) features for MLB games.

Focus
-----
1.  Accurate H2H statistics based on a defined lookback window of past games.
2.  Metrics calculated from the perspective of the current game's home team.
3.  No look-ahead bias (only uses games prior to the current game date).
4.  Handles missing historical data gracefully using defaults.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

# ────── LOGGER SETUP ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ────── DEFAULTS & HELPERS ──────
try:
    from .utils import DEFAULTS as MLB_DEFAULTS, normalize_team_name
    logger.info("Imported MLB_DEFAULTS and normalize_team_name")
except ImportError:
    logger.warning("Could not import MLB_DEFAULTS or normalize_team_name; using fallbacks")
    MLB_DEFAULTS: Dict[str, Any] = {}

    def normalize_team_name(team_id: Any) -> str:
        return str(team_id).strip() if pd.notna(team_id) else "unknown"


H2H_PLACEHOLDER_COLS: List[str] = [
    "matchup_num_games",
    "matchup_avg_run_diff",
    "matchup_home_win_pct",
    "matchup_avg_total_runs",
    "matchup_avg_home_team_runs",
    "matchup_avg_away_team_runs",
    "matchup_last_game_date",
    "matchup_home_team_streak",
]


def _default_val(col: str) -> Any:
    """Get default for each placeholder column."""
    if col == "matchup_last_game_date":
        return pd.NaT
    return float(MLB_DEFAULTS.get(f"mlb_{col}", MLB_DEFAULTS.get(col.replace("matchup_", ""), 0.0)))


def _get_matchup_history_single(
    *,
    home_norm: str,
    away_norm: str,
    hist_subset: pd.DataFrame,
    max_games: int = 7,
    debug: bool = False,
    idx: Optional[Any] = None, # Changed from Optional[int] to Optional[Any] to match DataFrame index type
) -> Dict[str, Any]:
    """
    Compute H2H stats for one upcoming game, from home team's viewpoint.
    """
    defaults = {c: _default_val(c) for c in H2H_PLACEHOLDER_COLS}
    if hist_subset is None or hist_subset.empty or max_games < 1:
        if debug:
            logger.debug(f"H2H[{idx}]: No history (hist_subset empty or max_games < 1) → defaults")
        return defaults

    # Prepare up to max_games most recent matchups
    # Ensure required columns exist before dropna to avoid errors on empty but structured DFs
    required_hist_cols = ["game_date", "home_score", "away_score", "home_team_norm", "away_team_norm"]
    if not all(col in hist_subset.columns for col in required_hist_cols):
        if debug:
            logger.debug(f"H2H[{idx}]: hist_subset missing one or more required columns {required_hist_cols} → defaults")
        return defaults

    recent = (
        hist_subset
        .sort_values("game_date", ascending=False)
        .head(max_games)
        .dropna(subset=required_hist_cols)
    )

    if recent.empty:
        if debug:
            logger.debug(f"H2H[{idx}]: After cleaning (dropna, head) → empty")
        return defaults

    # Collect stats
    diffs, totals, homes, aways = [], [], [], []
    home_wins = 0
    streak, last_win = 0, None

    # Process in chronological order for streak logic
    for _, g in recent.sort_values("game_date").iterrows():
        # Map actual scores into home/away perspective
        if g.home_team_norm == home_norm and g.away_team_norm == away_norm:
            h_runs, a_runs = g.home_score, g.away_score
        elif g.away_team_norm == home_norm and g.home_team_norm == away_norm:
            h_runs, a_runs = g.away_score, g.home_score
        else:
            # This case should ideally not be reached if hist_subset is correctly filtered by matchup_key
            if debug:
                game_id_info = g.get("game_id", "N/A") # Assuming 'game_id' might exist
                logger.warning(f"H2H[{idx}]: Team mismatch in historical game {game_id_info}, skipping. Home: {g.home_team_norm} vs Away: {g.away_team_norm}. Expected: {home_norm} vs {away_norm}")
            continue

        # Ensure scores are numeric before calculation
        if not (pd.api.types.is_number(h_runs) and pd.api.types.is_number(a_runs)):
            if debug:
                game_id_info = g.get("game_id", "N/A")
                logger.warning(f"H2H[{idx}]: Non-numeric scores in game {game_id_info} ({h_runs}, {a_runs}), skipping.")
            continue


        diff = h_runs - a_runs
        diffs.append(diff)
        totals.append(h_runs + a_runs)
        homes.append(h_runs)
        aways.append(a_runs)
        win = diff > 0
        home_wins += int(win)

        # Streak calc
        if last_win is None or win != last_win:
            streak = 1 if win else -1
        else:
            streak += 1 if win else -1
        last_win = win

    if not diffs:
        if debug:
            logger.debug(f"H2H[{idx}]: No valid games after processing loop → defaults")
        return defaults

    last_date = recent["game_date"].max()
    stats = {
        "matchup_num_games": len(diffs),
        "matchup_avg_run_diff": float(np.mean(diffs)),
        "matchup_home_win_pct": float(home_wins / len(diffs)),
        "matchup_avg_total_runs": float(np.mean(totals)),
        "matchup_avg_home_team_runs": float(np.mean(homes)),
        "matchup_avg_away_team_runs": float(np.mean(aways)),
        "matchup_last_game_date": pd.to_datetime(last_date),
        "matchup_home_team_streak": int(streak),
    }
    # Fill any missing keys with defaults (should not happen if all H2H_PLACEHOLDER_COLS are calculated)
    for c in H2H_PLACEHOLDER_COLS:
        stats.setdefault(c, defaults[c])
    return stats


def transform(
    df: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame] = None,
    max_games: int = 7,
    debug: bool = False,
    # Current-game columns
    game_date_col: str = "game_date_et",
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    # Historical-game columns
    hist_date_col: str = "game_date_time_utc",
    hist_home_col: str = "home_team_id",
    hist_away_col: str = "away_team_id",
    hist_home_score: str = "home_score",
    hist_away_score: str = "away_score"
) -> pd.DataFrame:
    """
    Add H2H features to df, given prior historical_df.
    """
    lvl = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Starting mlb_features.h2h.transform")

    out = df.copy()
    # Initialize placeholders
    for c in H2H_PLACEHOLDER_COLS:
        out[c] = _default_val(c)

    if out.empty:
        logger.warning("h2h.transform: no upcoming games (input df is empty) → placeholders only")
        if debug: logger.setLevel(lvl)
        return out

    # Validate essential columns in the upcoming games DataFrame
    req_upcoming_cols = [game_date_col, home_col, away_col]
    missing_upcoming_cols = [c for c in req_upcoming_cols if c not in out.columns]
    if missing_upcoming_cols:
        logger.error(f"h2h.transform: upcoming DataFrame 'df' is missing essential columns: {missing_upcoming_cols}")
        if debug: logger.setLevel(lvl)
        return out # Return with placeholders

    # Prepare historical lookup
    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: no historical data provided or it's empty → placeholders only")
        if debug: logger.setLevel(lvl)
        return out

    hist = historical_df.copy()

    # Validate essential columns in the historical DataFrame
    req_hist_cols = [hist_date_col, hist_home_col, hist_away_col, hist_home_score, hist_away_score]
    missing_hist_cols = [c for c in req_hist_cols if c not in hist.columns]
    if missing_hist_cols:
        logger.error(f"h2h.transform: historical_df is missing essential columns: {missing_hist_cols}")
        if debug: logger.setLevel(lvl)
        return out # Return with placeholders

    # Normalize and parse dates in historical data
    hist["game_date"] = pd.to_datetime(hist[hist_date_col], errors="coerce").dt.tz_localize(None)
    # Drop rows where essential data for H2H calculation is missing after conversion
    hist = hist.dropna(subset=["game_date", hist_home_score, hist_away_score, hist_home_col, hist_away_col])
    if hist.empty:
        logger.warning("h2h.transform: historical_df empty after cleaning (date conversion, dropna)")
        if debug: logger.setLevel(lvl)
        return out

    # Normalize team IDs in historical data
    hist["home_team_norm"] = hist[hist_home_col].apply(normalize_team_name)
    hist["away_team_norm"] = hist[hist_away_col].apply(normalize_team_name)
    hist["matchup_key"] = hist.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm, r.away_team_norm])), axis=1
    )

    # Build lookup dict of DataFrames per matchup_key
    lookup: Dict[str, pd.DataFrame] = {
        k: g.sort_values("game_date", ascending=False)
        for k, g in hist.groupby("matchup_key", observed=True) # observed=True is good practice
    }

    # Prepare upcoming games: parse dates and normalize team IDs
    out["game_date_parsed_"] = pd.to_datetime(out[game_date_col], errors="coerce").dt.tz_localize(None)
    # Drop rows from 'out' if critical info for processing is missing
    out.dropna(subset=["game_date_parsed_", home_col, away_col], inplace=True)
    if out.empty:
        logger.warning("h2h.transform: upcoming games DataFrame (out) is empty after dropna on essential columns.")
        # df.copy() was returned with placeholders earlier if original df was empty.
        # If it became empty here, we need to return a similarly structured DataFrame.
        # Create a new empty DataFrame with expected columns if `df` wasn't initially empty.
        result_df_columns = df.columns.tolist() + H2H_PLACEHOLDER_COLS
        result_df_columns = [col for col in result_df_columns if col not in ["game_date_parsed_", "home_team_norm_", "away_team_norm_", "matchup_key_"]] # Clean temp cols
        result_df_columns = list(dict.fromkeys(result_df_columns)) # Remove duplicates
        if debug: logger.setLevel(lvl)
        return pd.DataFrame(columns=result_df_columns)


    out["home_team_norm_"] = out[home_col].apply(normalize_team_name)
    out["away_team_norm_"] = out[away_col].apply(normalize_team_name)
    out["matchup_key_"] = out.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm_, r.away_team_norm_])), axis=1
    )

    # Compute H2H for each row
    stats_list: List[Dict[str, Any]] = []
    for idx, row in out.iterrows():
        current_matchup_key = row.matchup_key_
        # Retrieve historical games for this specific matchup.
        # If no history, hist_games_for_matchup will be an empty DataFrame.
        hist_games_for_matchup = lookup.get(current_matchup_key, pd.DataFrame())
        
        current_game_date = row.game_date_parsed_ # This is a datetime object (already checked for NaT by dropna)

        # *** THE FIX IS HERE ***
        if hist_games_for_matchup.empty:
            # If no historical games for this matchup, subset is empty.
            # _get_matchup_history_single handles empty hist_subset.
            subset = pd.DataFrame() 
        else:
            # Filter historical games to those strictly before the current game's date.
            # 'game_date' column in hist_games_for_matchup was created from hist_date_col.
            # Ensure 'game_date' column exists in hist_games_for_matchup (it should if not empty and processed correctly)
            if "game_date" not in hist_games_for_matchup.columns:
                 if debug: logger.warning(f"H2H[{idx}]: 'game_date' column unexpectedly missing from non-empty hist_games_for_matchup for key {current_matchup_key}. Treating as no history.")
                 subset = pd.DataFrame()
            else:
                subset = hist_games_for_matchup[hist_games_for_matchup["game_date"] < current_game_date]
        
        stats = _get_matchup_history_single(
            home_norm=row.home_team_norm_,
            away_norm=row.away_team_norm_,
            hist_subset=subset,
            max_games=max_games,
            debug=debug,
            idx=idx, # Pass index for logging
        )
        stats_list.append(stats)

    # Merge results back
    if not stats_list: # If out was not empty, but no stats were generated (e.g. all lookups empty)
        # This case should be covered by _get_matchup_history_single returning defaults,
        # so stats_list should have one dict of defaults per row in 'out'.
        # If 'out' was non-empty, stats_list should also be non-empty.
        # If 'out' became empty after preprocessing, this loop isn't run.
        pass # Handled by H2H_PLACEHOLDER_COLS initialization if out DataFrame was empty.


    if not out.empty and stats_list: # Ensure there are rows in 'out' to merge onto
        h2h_df = pd.DataFrame(stats_list, index=out.index)
        for c in H2H_PLACEHOLDER_COLS:
            # Ensure the column exists in h2h_df before trying to access it
            if c in h2h_df:
                out[c] = h2h_df[c].fillna(_default_val(c))
            else: # Should not happen if _get_matchup_history_single returns all keys
                out[c] = _default_val(c)
    # If 'out' is empty or 'stats_list' is empty (though it shouldn't be if 'out' isn't),
    # 'out' already has default placeholder values.

    # ─── Final type & null enforcement ───
    for c in H2H_PLACEHOLDER_COLS:
        if c not in out.columns: # If 'out' was empty and columns were not added
            out[c] = _default_val(c) # Should already be set, but as safeguard
            
        # If it’s supposed to be numeric and somehow is still all NaNs, fill default
        if out[c].isnull().all() and c != "matchup_last_game_date":
            out[c] = _default_val(c)
        
        # Enforce correct dtype
        if c == "matchup_last_game_date":
            out[c] = pd.to_datetime(out[c], errors="coerce")
        elif c in ("matchup_num_games", "matchup_home_team_streak"):
            # Ensure conversion to numeric first, then fillna, then int
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(_default_val(c)).astype(int)
        else:  # averages, percentages
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(_default_val(c)).astype(float)

    # Final cleanup: drop temporary helper columns
    # Added underscores to temporary column names to reduce chance of collision
    drop_cols = ["game_date_parsed_", "home_team_norm_", "away_team_norm_", "matchup_key_"]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True, errors="ignore")

    logger.info(f"Completed mlb_features.h2h.transform. Output shape: {out.shape}")
    if debug:
        logger.setLevel(lvl)
    return out