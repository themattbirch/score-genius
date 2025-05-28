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
    idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute H2H stats for one upcoming game, from home team's viewpoint.
    """
    defaults = {c: _default_val(c) for c in H2H_PLACEHOLDER_COLS}
    if hist_subset is None or hist_subset.empty or max_games < 1:
        if debug:
            logger.debug(f"H2H[{idx}]: No history → defaults")
        return defaults

    # Prepare up to max_games most recent matchups
    recent = (
        hist_subset
        .sort_values("game_date", ascending=False)
        .head(max_games)
        .dropna(subset=["game_date", "home_score", "away_score",
                        "home_team_norm", "away_team_norm"])
    )
    if recent.empty:
        if debug:
            logger.debug(f"H2H[{idx}]: After cleaning → empty")
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
            if debug:
                logger.warning(f"H2H[{idx}]: Team mismatch, skipping {g.game_id}")
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
            logger.debug(f"H2H[{idx}]: No valid games → defaults")
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
    # Fill any missing keys with defaults
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
        logger.warning("h2h.transform: no upcoming games → placeholders only")
        logger.setLevel(lvl)
        return out

    # Validate essential columns
    req = [game_date_col, home_col, away_col]
    if any(c not in out.columns for c in req):
        missing = [c for c in req if c not in out.columns]
        logger.error(f"h2h.transform: missing cols {missing}")
        logger.setLevel(lvl)
        return out

    # Prepare historical lookup
    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: no historical data → placeholders only")
        logger.setLevel(lvl)
        return out

    hist = historical_df.copy()
    # Normalize and parse dates
    hist["game_date"] = (
        pd.to_datetime(hist[hist_date_col], errors="coerce")
        .dt.tz_localize(None)
    )
    hist = hist.dropna(subset=["game_date", hist_home_score, hist_away_score])
    if hist.empty:
        logger.warning("h2h.transform: historical_df empty after clean")
        logger.setLevel(lvl)
        return out

    # Normalize team IDs
    hist["home_team_norm"] = hist[hist_home_col].apply(normalize_team_name)
    hist["away_team_norm"] = hist[hist_away_col].apply(normalize_team_name)
    hist["matchup_key"] = hist.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm, r.away_team_norm])), axis=1
    )

    # Build lookup dict of DataFrames per matchup_key
    lookup: Dict[str, pd.DataFrame] = {
        k: g.sort_values("game_date", ascending=False)
        for k, g in hist.groupby("matchup_key", observed=True)
    }

    # Prepare upcoming games
    out["game_date"] = (
        pd.to_datetime(out[game_date_col], errors="coerce")
        .dt.tz_localize(None)
    )
    out = out.dropna(subset=["game_date"])
    out["home_team_norm"] = out[home_col].apply(normalize_team_name)
    out["away_team_norm"] = out[away_col].apply(normalize_team_name)
    out["matchup_key"] = out.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm, r.away_team_norm])), axis=1
    )

    # Compute H2H for each row
    stats_list: List[Dict[str, Any]] = []
    for idx, row in out.iterrows():
        key = row.matchup_key
        hist_games = lookup.get(key, pd.DataFrame())
        subset = hist_games[hist_games.game_date < row.game_date]
        stats = _get_matchup_history_single(
            home_norm=row.home_team_norm,
            away_norm=row.away_team_norm,
            hist_subset=subset,
            max_games=max_games,
            debug=debug,
            idx=idx,
        )
        stats_list.append(stats)

    # Merge results back
    h2h_df = pd.DataFrame(stats_list, index=out.index)
    for c in H2H_PLACEHOLDER_COLS:
        out[c] = h2h_df[c].fillna(_default_val(c))

    # ─── Final type & null enforcement ───
    for c in H2H_PLACEHOLDER_COLS:
        # If it’s supposed to be numeric and somehow is still all NaNs, fill default
        if out[c].isnull().all() and c != "matchup_last_game_date":
            out[c] = _default_val(c)
        # Enforce correct dtype
        if c == "matchup_last_game_date":
            out[c] = pd.to_datetime(out[c], errors="coerce")
        elif c in ("matchup_num_games", "matchup_home_team_streak"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(_default_val(c)).astype(int)
        else:  # averages, percentages
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(_default_val(c)).astype(float)


    # Final cleanup: drop helper columns
    drop_cols = ["game_date", "home_team_norm", "away_team_norm", "matchup_key"]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True, errors="ignore")

    logger.info("Completed mlb_features.h2h.transform")
    if debug:
        logger.setLevel(lvl)
    return out
