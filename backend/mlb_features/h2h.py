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

_COLUMN_SYNONYMS = {
    "game_id":   ["game_id", "uid", "gid"],
    "game_date": ["game_date", "game_date_et", "game_date_time_utc"],
    "home_team": ["home_team", "home_team_id"],
    "away_team": ["away_team", "away_team_id"],
}

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any recognised alias to its canonical column name."""
    ren = {}
    for canon, alts in _COLUMN_SYNONYMS.items():
        if canon in df.columns:
            continue
        for alt in alts:
            if alt in df.columns:
                ren[alt] = canon
                break
    return df.rename(columns=ren, errors="ignore") if ren else df

# ── Robust wrapper around utils.normalize_team_name ──────────
def _safe_norm(val: Any) -> str:
    norm = normalize_team_name(val)
    if not isinstance(norm, str):
        norm = str(norm)
    norm = norm.strip().lower()
    if norm in {"unknown", "unknown_team", ""}:
        norm = str(val).strip().lower()
    return norm.replace(" ", "")


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
    idx: Optional[Any] = None,  # Changed from Optional[int] to Optional[Any]
) -> Dict[str, Any]:
    """
    Compute H2H stats for one upcoming game, from home team's viewpoint.
    """
    defaults = {c: _default_val(c) for c in H2H_PLACEHOLDER_COLS}
    if hist_subset is None or hist_subset.empty or max_games < 1:
        if debug:
            logger.debug(f"H2H[{idx}]: No history (hist_subset empty or max_games < 1) → defaults")
        return defaults

    # Ensure required columns exist before we dropna
    required_hist_cols = ["game_date", "home_score", "away_score", "home_team_norm", "away_team_norm"]
    if not all(col in hist_subset.columns for col in required_hist_cols):
        if debug:
            logger.debug(f"H2H[{idx}]: hist_subset missing one or more required columns {required_hist_cols} → defaults")
        return defaults

    # ─── Drop any NaNs first, then take the top max_games ───
    clean = hist_subset.dropna(subset=required_hist_cols)
    recent = clean.sort_values("game_date", ascending=False).head(max_games)

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
            # Should not happen if hist_subset was pre‐filtered by matchup_key
            if debug:
                game_id_info = g.get("game_id", "N/A")
                logger.warning(
                    f"H2H[{idx}]: Team mismatch in historical game {game_id_info}, "
                    f"skipping. Home: {g.home_team_norm} vs Away: {g.away_team_norm}. "
                    f"Expected: {home_norm} vs {away_norm}"
                )
            continue

        # Now that we've dropped NaNs, both h_runs and a_runs should be numbers.
        # But just in case, skip if anything is still NaN:
        if pd.isna(h_runs) or pd.isna(a_runs):
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
    # Current-game columns (not used directly, since we canonicalize below)
    game_date_col: str = "game_date",
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    # Historical‐game columns (not used directly, since we canonicalize below)
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

    # 1) Copy & remember which columns existed in the input
    out = df.copy()
    original_cols = set(out.columns)

    # 2) Drop any pre‐existing H2H columns, standardize names, then rename
    out = out.drop(columns=[c for c in H2H_PLACEHOLDER_COLS if c in out], errors="ignore")
    out = _standardize_cols(out)

    # 3) Use canonical names for the MLB tests
    game_date_col = "game_date"
    home_col      = "home_team"
    away_col      = "away_team"

    # 4) Initialize all placeholders with their default values
    for c in H2H_PLACEHOLDER_COLS:
        out[c] = _default_val(c)

    # 5) Restore "uid" (if present in the original) so idempotency test can still find it
    if ("uid" in original_cols) and ("uid" not in out.columns) and ("game_id" in out.columns):
        out["uid"] = out["game_id"]

    # 6) If the upcoming‐games DF is empty, just return placeholders
    if out.empty:
        logger.warning("h2h.transform: no upcoming games (input df is empty) → placeholders only")
        if debug:
            logger.setLevel(lvl)
        return out

    # 7) Ensure essential upcoming columns exist
    req_upcoming_cols = [game_date_col, home_col, away_col]
    missing_upcoming_cols = [c for c in req_upcoming_cols if c not in out.columns]
    if missing_upcoming_cols:
        logger.error(
            f"h2h.transform: upcoming DataFrame 'df' is missing essential columns: {missing_upcoming_cols}"
        )
        if debug:
            logger.setLevel(lvl)
        return out  # return with placeholders

    # 8) If no historical data, return placeholder‐enriched upcoming DF
    if historical_df is None or historical_df.empty:
        logger.warning("h2h.transform: no historical data provided or it's empty → placeholders only")
        if debug:
            logger.setLevel(lvl)
        return out

    # 9) Canonicalize the historical columns as well
    hist = _standardize_cols(historical_df.copy())
    hist_date_col = "game_date"
    hist_home_col = "home_team"
    hist_away_col = "away_team"

    # 10) Convert date & drop anything missing required historical fields
    hist["game_date"] = pd.to_datetime(hist[hist_date_col], errors="coerce").dt.tz_localize(None)
    hist = hist.dropna(subset=[hist_date_col, hist_home_col, hist_away_col, hist_home_score, hist_away_score])
    if hist.empty:
        logger.warning("h2h.transform: historical_df empty after cleaning → placeholders only")
        if debug:
            logger.setLevel(lvl)
        return out

    # 11) Normalize team IDs in historical data
    hist["home_team_norm"] = hist[hist_home_col].apply(_safe_norm)
    hist["away_team_norm"] = hist[hist_away_col].apply(_safe_norm)
    hist["matchup_key"] = hist.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm, r.away_team_norm])), axis=1
    )

    # 12) Build a lookup of DataFrames by matchup_key
    lookup: Dict[str, pd.DataFrame] = {
        k: g.sort_values("game_date", ascending=False)
        for k, g in hist.groupby("matchup_key", observed=True)
    }

    # 13) Prepare the upcoming‐games DataFrame:
    out["game_date_parsed_"] = pd.to_datetime(out[game_date_col], errors="coerce").dt.tz_localize(None)
    out.dropna(subset=["game_date_parsed_", home_col, away_col], inplace=True)
    if out.empty:
        logger.warning("h2h.transform: upcoming games DF is empty after dropna → placeholders")
        # Rebuild an empty DataFrame with the same columns:
        result_df_columns = list(dict.fromkeys(df.columns.tolist() + H2H_PLACEHOLDER_COLS))
        if debug:
            logger.setLevel(lvl)
        return pd.DataFrame(columns=result_df_columns)

    out["home_team_norm_"] = out[home_col].apply(_safe_norm)
    out["away_team_norm_"] = out[away_col].apply(_safe_norm)
    out["matchup_key_"] = out.apply(
        lambda r: "_vs_".join(sorted([r.home_team_norm_, r.away_team_norm_])), axis=1
    )

    # 14) Compute H2H for each row
    stats_list: List[Dict[str, Any]] = []
    for idx, row in out.iterrows():
        key = row.matchup_key_
        hist_games_for_matchup = lookup.get(key, pd.DataFrame())
        current_game_date = row.game_date_parsed_

        if hist_games_for_matchup.empty:
            subset = pd.DataFrame()
        else:
            if "game_date" not in hist_games_for_matchup.columns:
                if debug:
                    logger.warning(
                        f"H2H[{idx}]: 'game_date' missing in hist_games_for_matchup for key {key} → no history"
                    )
                subset = pd.DataFrame()
            else:
                subset = hist_games_for_matchup[hist_games_for_matchup["game_date"] < current_game_date]

        stats = _get_matchup_history_single(
            home_norm=row.home_team_norm_,
            away_norm=row.away_team_norm_,
            hist_subset=subset,
            max_games=max_games,
            debug=debug,
            idx=idx,
        )
        stats_list.append(stats)

    if not out.empty and stats_list:
        h2h_df = pd.DataFrame(stats_list, index=out.index)
        for c in H2H_PLACEHOLDER_COLS:
            if c in h2h_df:  # normally true
                out[c] = h2h_df[c].fillna(_default_val(c))
            else:
                out[c] = _default_val(c)

    # 15) Final type coercion & null‐fill
    for c in H2H_PLACEHOLDER_COLS:
        if c not in out.columns:
            out[c] = _default_val(c)
        if out[c].isnull().all() and c != "matchup_last_game_date":
            out[c] = _default_val(c)

        if c == "matchup_last_game_date":
            out[c] = pd.to_datetime(out[c], errors="coerce")
        elif c in ("matchup_num_games", "matchup_home_team_streak"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(_default_val(c)).astype(int)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(_default_val(c)).astype(float)

    # 16) Drop temporary helper columns
    drop_cols = ["game_date_parsed_", "home_team_norm_", "away_team_norm_", "matchup_key_"]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True, errors="ignore")

    logger.info(f"Completed mlb_features.h2h.transform. Output shape: {out.shape}")
    if debug:
        logger.setLevel(lvl)
    return out
