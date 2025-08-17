# backend/nfl_features/rolling.py
"""Rolling-window (recent form) feature loader.

This module takes a pre-fetched DataFrame of recent-form stats, reshapes it
into home/away columns, applies sensible defaults, and derives differentials.

Enhancements:
- Column hygiene + dtype trimming (float32 numerics).
- Soft anti-leak guard: per-team as-of filter using earliest upcoming game date.
- Multi-window support:
    • via `window` column (e.g., 3/5/10) → pivots to _w3/_w5/_w10
    • via column suffixes (e.g., _3g / _w3) → normalized to _wN
- Blended recency features (e.g., 0.5*w3 + 0.3*w5 + 0.2*w10).
- Stronger differentials including attack-vs-defense cross-diffs.
- Light clipping to keep outliers from swamping models.
- Lean return: only game_id + emitted rolling features (typed).
"""
from __future__ import annotations

import logging
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Configuration (DB contract / logical keys)
# ---------------------------------------------------------------------------
_BASE_FEATURE_COLS: Mapping[str, str] = {
    # DB/source column → logical key for DEFAULTS lookup
    "rolling_points_for_avg": "points_for_avg",
    "rolling_points_against_avg": "points_against_avg",
    "rolling_yards_per_play_avg": "yards_per_play_avg",
    "rolling_turnover_differential_avg": "turnover_differential_avg",
}
# Defense counterparts (if present) for cross-diffs
_DEFENSE_ALIASES: Mapping[str, str] = {
    # offense_col -> expected defense-col
    "rolling_yards_per_play_avg": "rolling_yards_per_play_allowed_avg",
    # points is handled by points_against already
}

# Derived locals we compute (per suffix/window)
_DERIVED_LOCAL: Mapping[str, tuple[str, str]] = {
    # new_col = lhs - rhs
    "rolling_point_differential_avg": (
        "rolling_points_for_avg",
        "rolling_points_against_avg",
    ),
}

# Preferred blend weights (renormalized to present windows)
_BLEND_WINDOWS: Sequence[int] = (3, 5, 10)
_BLEND_WEIGHTS: Mapping[int, float] = {3: 0.5, 5: 0.3, 10: 0.2}

# Light caps (keep signal strong but bounded)
_CLIP_RANGES_RAW = {
    # raw rates/levels
    r"^home_rolling_points_(?:for|against)_avg(?:_w\d+)?$": (8.0, 45.0),
    r"^away_rolling_points_(?:for|against)_avg(?:_w\d+)?$": (8.0, 45.0),
    r"^(home|away)_rolling_yards_per_play_(?:avg|allowed_avg)(?:_w\d+)?$": (3.5, 8.5),
    r"^(home|away)_rolling_turnover_differential_avg(?:_w\d+)?$": (-2.5, 2.5),
    # blends
    r"^(home|away)_rolling_points_(?:for|against)_avg_blend$": (8.0, 45.0),
    r"^(home|away)_rolling_yards_per_play_avg_blend$": (3.5, 8.5),
}
_CLIP_RANGES_DIFF = {
    # symmetric diffs
    r"_points_.*_diff(?:_w\d+)?$": (-20.0, 20.0),
    r"_yards_per_play.*_diff(?:_w\d+)?$": (-3.0, 3.0),
    r"_turnover_differential_avg.*_diff(?:_w\d+)?$": (-2.0, 2.0),
    # cross-diffs
    r"attack_vs_defense_diff(?:_w\d+)?$": (-20.0, 20.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dt_utc(s: pd.Series) -> pd.Series:
    """Parse to UTC tz-aware; tolerate None/NaT."""
    return pd.to_datetime(s, errors="coerce", utc=True)

def _find_stats_time_col(df: pd.DataFrame) -> Optional[str]:
    """Probe for a usable timestamp column in stats_df."""
    candidates = [
        "asof_ts", "asof_date", "stat_ts", "stat_date", "snapshot_ts",
        "updated_at", "computed_at", "game_date", "kickoff_ts",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ensure_team_norm_games(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    # make sure we have game_date for as-of logic; fallback to kickoff_ts if present
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    elif "kickoff_ts" in out.columns:
        ts = _to_dt_utc(out["kickoff_ts"])
        if ts.notna().any():
            out["game_date"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            out["game_date"] = pd.NaT
    else:
        out["game_date"] = pd.NaT

    for c in ("home_team_norm", "away_team_norm"):
        if c in out.columns:
            out[c] = out[c].apply(normalize_team_name).astype(str).str.lower()
    # allow deriving from team_id if norm missing
    if "home_team_norm" not in out.columns and "home_team_id" in out.columns:
        out["home_team_norm"] = out["home_team_id"].apply(normalize_team_name).astype(str).str.lower()
    if "away_team_norm" not in out.columns and "away_team_id" in out.columns:
        out["away_team_norm"] = out["away_team_id"].apply(normalize_team_name).astype(str).str.lower()
    return out

def _ensure_team_norm_stats(s: pd.DataFrame) -> pd.DataFrame:
    out = s.copy()
    if "team_norm" in out.columns:
        out["team_norm"] = out["team_norm"].apply(normalize_team_name).astype(str).str.lower()
    elif "team_id" in out.columns:
        out["team_norm"] = out["team_id"].apply(normalize_team_name).astype(str).str.lower()
    elif "team" in out.columns:
        out["team_norm"] = out["team"].apply(normalize_team_name).astype(str).str.lower()
    else:
        # best effort: look for any column that smells like team
        cand = next((c for c in out.columns if "team" in c), None)
        if cand:
            out["team_norm"] = out[cand].apply(normalize_team_name).astype(str).str.lower()
        else:
            out["team_norm"] = None
    return out

def _normalize_window_suffix(name: str) -> tuple[str, Optional[int]]:
    """
    Detect window from column suffixes:
      rolling_points_for_avg_3g  → (rolling_points_for_avg, 3)
      rolling_points_for_avg_w5  → (rolling_points_for_avg, 5)
    Returns (base_name, window) where window is None if not detected.
    """
    m = re.match(r"^(.*)_(?:w)?(\d+)(?:g)?$", name)
    if m:
        base, w = m.group(1), int(m.group(2))
        return base, w
    return name, None

def _gather_base_metric_cols(df: pd.DataFrame) -> set[str]:
    """Collect base metric column names present in df (without window suffixes)."""
    present = set()
    for col in df.columns:
        base, w = _normalize_window_suffix(col)
        if base in _BASE_FEATURE_COLS or base in _DERIVED_LOCAL or base in _DEFENSE_ALIASES.values():
            present.add(base)
    return present

def _asof_trim_stats(stats: pd.DataFrame, games: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    """
    Soft anti-leak: per-team, trim stats rows to those strictly before that team's
    earliest upcoming game_date; keep the last prior row.

    If there's no usable datetime time_col, we cannot safely compare to game_date,
    so we just return the last row per team.
    """
    s = stats.copy()

    # 1) Identify a usable datetime timestamp column
    real_time_col = None
    if time_col and np.issubdtype(s[time_col].dtype, np.datetime64):
        real_time_col = time_col
    elif time_col:
        # Try to coerce to UTC datetimes
        coerced = pd.to_datetime(s[time_col], errors="coerce", utc=True)
        if coerced.notna().any():
            s[time_col] = coerced
            real_time_col = time_col

    if real_time_col is None:
        # No real datetime column available → fallback to last-per-team
        s = s.sort_values(["team_norm"], kind="mergesort")
        return s.groupby("team_norm", sort=False).tail(1)

    # 2) Build earliest upcoming game_date per team (coerce to UTC for safe compare)
    dates = []
    for c in ("home_team_norm", "away_team_norm"):
        if c in games.columns:
            sub = games[["game_date", c]].rename(columns={c: "team_norm"}).copy()
            sub["game_date"] = pd.to_datetime(sub["game_date"], errors="coerce", utc=True)
            dates.append(sub)

    if not dates:
        # No join key → fallback to last-per-team
        s = s.sort_values(["team_norm", real_time_col], kind="mergesort")
        return s.groupby("team_norm", sort=False).tail(1)

    team_min = pd.concat(dates, ignore_index=True)
    team_min = team_min.dropna(subset=["team_norm"])
    if team_min.empty:
        s = s.sort_values(["team_norm", real_time_col], kind="mergesort")
        return s.groupby("team_norm", sort=False).tail(1)

    team_min = team_min.groupby("team_norm", sort=False, as_index=False)["game_date"].min()

    # 3) Join & strictly-before filter
    merged = s.merge(team_min, on="team_norm", how="left", suffixes=("", "__game"))
    # Coerce game_date to UTC (safe if already UTC)
    merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce", utc=True)

    # If we still can't compare, fallback to last-per-team
    if merged["game_date"].isna().all():
        s = s.sort_values(["team_norm", real_time_col], kind="mergesort")
        return s.groupby("team_norm", sort=False).tail(1)

    # Strictly before earliest upcoming game_date
    keep = merged[real_time_col] < merged["game_date"]
    kept = merged[keep.fillna(False)]
    if kept.empty:
        s = s.sort_values(["team_norm", real_time_col], kind="mergesort")
        return s.groupby("team_norm", sort=False).tail(1)

    kept = kept.sort_values(["team_norm", real_time_col], kind="mergesort")
    return kept.groupby("team_norm", sort=False).tail(1)

def _pivot_windows_if_needed(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Supports:
      1) 'window' column → pivot to _wN columns
      2) column suffixes (…_3g / …_w3) → normalized to _wN columns
      3) single-window → passthrough
    Always returns one row per team_norm.
    """
    s = stats.copy()

    # Case 1: explicit 'window' column
    if "window" in s.columns:
        # keep only known metric columns + defense aliases if present
        metric_bases = _gather_base_metric_cols(s)
        keep_cols = ["team_norm", "window"] + [c for c in s.columns if c in metric_bases]
        s = s[keep_cols].copy()

        # ensure numeric dtypes
        for c in keep_cols:
            if c not in ("team_norm", "window"):
                s[c] = pd.to_numeric(s[c], errors="coerce").astype("float32")

        # pivot per team → wide _wN
        wide_parts = []
        for base in metric_bases:
            sub = s.pivot_table(index="team_norm", columns="window", values=base, aggfunc="last")
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = [c[-1] for c in sub.columns]
            sub = sub.rename(columns={w: f"{base}_w{int(w)}" for w in sub.columns})
            wide_parts.append(sub)
        if not wide_parts:
            return s.groupby("team_norm", sort=False).tail(1).drop(columns=["window"], errors="ignore")
        wide = pd.concat(wide_parts, axis=1).reset_index()
        return wide

    # Case 2: detect suffix-coded windows
    suffix_cols = [c for c in s.columns if _normalize_window_suffix(c)[1] is not None]
    if suffix_cols:
        # Normalize names to base + _wN
        renames = {}
        for c in suffix_cols:
            base, w = _normalize_window_suffix(c)
            if w is not None:
                renames[c] = f"{base}_w{w}"
        s = s.rename(columns=renames)

        # If multiple rows per team, collapse by last
        s = s.sort_values("team_norm", kind="mergesort").groupby("team_norm", sort=False).tail(1)
        return s

    # Case 3: single-window snapshot (one row per team)
    return s.sort_values("team_norm", kind="mergesort").groupby("team_norm", sort=False).tail(1)

def _build_blends(stats: pd.DataFrame) -> pd.DataFrame:
    """Create blended recency features if multiple windows exist."""
    s = stats.copy()

    def _blend_cols(base: str) -> None:
        # collect available windows for this base
        present = []
        for w in _BLEND_WINDOWS:
            col = f"{base}_w{w}"
            if col in s.columns:
                present.append((w, col))
        if not present:
            return
        # renormalize weights to present windows
        total = sum(_BLEND_WEIGHTS[w] for w, _ in present)
        if total <= 0:
            return
        parts = [(s[col].astype("float32") * (_BLEND_WEIGHTS[w] / total)) for w, col in present]
        s[f"{base}_blend"] = np.sum(parts, axis=0).astype("float32")

    # apply to useful bases
    for base in ("rolling_points_for_avg", "rolling_points_against_avg", "rolling_yards_per_play_avg"):
        _blend_cols(base)

    return s

def _ensure_derived(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute local derived columns (including for each window and blend)."""
    s = stats.copy()

    # build list of suffixes to consider: "", _w3/_w5/_w10, _blend (if present)
    suffixes = {""}
    for w in _BLEND_WINDOWS:
        suffixes.add(f"_w{w}")
    if any(c.endswith("_blend") for c in s.columns):
        suffixes.add("_blend")

    for new_base, (lhs_base, rhs_base) in _DERIVED_LOCAL.items():
        for suf in sorted(suffixes):
            lhs = f"{lhs_base}{suf}"
            rhs = f"{rhs_base}{suf}"
            new = f"{new_base}{suf}"
            if lhs in s.columns and rhs in s.columns and new not in s.columns:
                s[new] = (s[lhs] - s[rhs]).astype("float32")

    return s

def _prefix_merge_side(
    games_df: pd.DataFrame,
    stats_team_df: pd.DataFrame,
    side: str,
    join_col: str,
    keep_metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Merge a minimal set of columns for one side and prefix them."""
    subset = stats_team_df[["team_norm"] + list(keep_metric_cols)].copy()
    subset = subset.rename(columns={"team_norm": join_col})
    prefixed = subset.add_prefix(f"{side}_")
    return games_df.merge(prefixed, on=f"{side}_{join_col}", how="left")

def _emit_diffs(result: pd.DataFrame, bases: Sequence[str], suffixes: Sequence[str]) -> pd.DataFrame:
    """Emit symmetric and cross diffs for the provided bases/suffixes."""
    out = result.copy()

    # symmetric diffs: (home - away) per base/suffix
    for base in bases:
        for suf in suffixes:
            h = f"home_{base}{suf}"
            a = f"away_{base}{suf}"
            d = f"{base}{suf}_diff"
            if h in out.columns and a in out.columns and d not in out.columns:
                # Prefer NaN when both missing; otherwise compute normal diff
                both_na = out[h].isna() & out[a].isna()
                out[d] = out[h] - out[a]
                out.loc[both_na, d] = np.nan
                out[d] = out[d].astype("float32")

    # cross diffs: points attack (for) vs defense (against)
    for suf in suffixes:
        h_for = f"home_rolling_points_for_avg{suf}"
        a_against = f"away_rolling_points_against_avg{suf}"
        if h_for in out.columns and a_against in out.columns:
            d = f"rolling_points_attack_vs_defense_diff{suf}"
            both_na = out[h_for].isna() & out[a_against].isna()
            out[d] = (out[h_for] - out[a_against]).astype("float32")
            out.loc[both_na, d] = np.nan

        # reverse perspective (away attack vs home defense)
        a_for = f"away_rolling_points_for_avg{suf}"
        h_against = f"home_rolling_points_against_avg{suf}"
        if a_for in out.columns and h_against in out.columns:
            d = f"rolling_points_attack_vs_defense_reverse_diff{suf}"
            both_na = out[a_for].isna() & out[h_against].isna()
            out[d] = (out[a_for] - out[h_against]).astype("float32")
            out.loc[both_na, d] = np.nan

    # optional ypp cross diffs (only if defense counterpart exists)
    for suf in suffixes:
        off = f"rolling_yards_per_play_avg{suf}"
        deff = f"rolling_yards_per_play_allowed_avg{suf}"
        h_off = f"home_{off}"
        a_def = f"away_{deff}"
        if h_off in out.columns and a_def in out.columns:
            d = f"rolling_yards_per_play_attack_vs_defense_diff{suf}"
            both_na = out[h_off].isna() & out[a_def].isna()
            out[d] = (out[h_off] - out[a_def]).astype("float32")
            out.loc[both_na, d] = np.nan

        a_off = f"away_{off}"
        h_def = f"home_{deff}"
        if a_off in out.columns and h_def in out.columns:
            d = f"rolling_yards_per_play_attack_vs_defense_reverse_diff{suf}"
            both_na = out[a_off].isna() & out[h_def].isna()
            out[d] = (out[a_off] - out[h_def]).astype("float32")
            out.loc[both_na, d] = np.nan

    return out

def _clip_by_patterns(df: pd.DataFrame, patterns_to_ranges: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    """Apply clip ranges to matching columns (regex patterns)."""
    out = df.copy()
    for pat, (lo, hi) in patterns_to_ranges.items():
        cols = [c for c in out.columns if re.search(pat, c)]
        if not cols:
            continue
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32").clip(lo, hi)
    return out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rolling_features(
    games: pd.DataFrame,
    *,
    recent_form_df: Optional[pd.DataFrame] = None,
    **kwargs,  # absorb unused engine kwargs
) -> pd.DataFrame:
    """
    Attaches recent-form metrics to a games DataFrame using a pre-fetched
    DataFrame of stats. Returns only ['game_id'] + rolling feature columns.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    if recent_form_df is None or recent_form_df.empty:
        logger.warning("rolling: No recent_form_df provided. Returning only game_id.")
        return games[["game_id"]].copy()

    # --- A) Column hygiene & dtypes ---
    games_df = _ensure_team_norm_games(games)
    stats_df = _ensure_team_norm_stats(recent_form_df)

    # identify a usable time column (for anti-leak)
    time_col = _find_stats_time_col(stats_df)

    # keep only columns we might need (team_norm, time_col, metrics)
    metric_bases_present = _gather_base_metric_cols(stats_df)
    keep = ["team_norm"]
    if time_col:
        keep.append(time_col)
    # include all columns that belong to bases (windowed or not)
    for c in stats_df.columns:
        base, _ = _normalize_window_suffix(c)
        if base in metric_bases_present:
            keep.append(c)
    stats_df = stats_df[sorted(set(keep))].copy()

    # cast numerics
    for c in stats_df.columns:
        if c not in ("team_norm", time_col):
            stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").astype("float32")

    # --- B) Soft anti-leak per team ---
    stats_df = _asof_trim_stats(stats_df, games_df, time_col=time_col)

    # --- C) Multi-window support (pivot/normalize) ---
    stats_df = _pivot_windows_if_needed(stats_df)

    # --- D) Blended recency features ---
    stats_df = _build_blends(stats_df)

    # --- Derived locals for each suffix (E: emit later after merges) ---
    stats_df = _ensure_derived(stats_df)

    # All metric columns we intend to carry forward
    metric_cols = [c for c in stats_df.columns if c != "team_norm"]

    # --- E) Merge only intended columns per side ---
    # Ensure join keys on games
    for req in ("game_id", "home_team_norm", "away_team_norm"):
        if req not in games_df.columns:
            logger.error("rolling: games_df missing required column %s; returning only game_id.", req)
            return games[["game_id"]].copy()

    # Home
    merged = games_df.copy()
    merged = _prefix_merge_side(
        merged, stats_df, side="home", join_col="team_norm", keep_metric_cols=metric_cols
    )
    # Away
    merged = _prefix_merge_side(
        merged, stats_df, side="away", join_col="team_norm", keep_metric_cols=metric_cols
    )

    # --- H) Fill raw rolling columns with sensible defaults (diffs handled later) ---
    # Build map of base->default for non-window & window/blend columns
    def _default_for(base: str) -> float:
        key = _BASE_FEATURE_COLS.get(base, None)
        if key is None:
            # for locally-derived base we can compute from components or fallback to 0.0
            if base == "rolling_point_differential_avg":
                # computed later; default ~ (pf - pa) using defaults if needed
                pf = DEFAULTS.get("points_for_avg", 0.0)
                pa = DEFAULTS.get("points_against_avg", 0.0)
                return float(pf - pa)
            return 0.0
        return float(DEFAULTS.get(key, 0.0))

    # Determine bases actually present (without prefixes/suffixes)
    all_metric_bases = set()
    for c in metric_cols:
        base, suf_w = _normalize_window_suffix(c)
        all_metric_bases.add(base)

    # Fill per side for raw columns; diffs will be NaN if both missing (policy)
    for side in ("home", "away"):
        for base in all_metric_bases:
            # raw base (no suffix)
            col = f"{side}_{base}"
            if col in merged.columns:
                merged[col] = merged[col].fillna(_default_for(base)).astype("float32")
            # windowed/blend variants
            for w in _BLEND_WINDOWS:
                c_w = f"{side}_{base}_w{w}"
                if c_w in merged.columns:
                    merged[c_w] = merged[c_w].fillna(_default_for(base)).astype("float32")
            c_b = f"{side}_{base}_blend"
            if c_b in merged.columns:
                merged[c_b] = merged[c_b].fillna(_default_for(base)).astype("float32")

    # --- F) Stronger diffs (symmetric + cross) ---
    # Build suffix list present
    suffixes = [""]
    for w in _BLEND_WINDOWS:
        if any(col.endswith(f"_w{w}") for col in merged.columns):
            suffixes.append(f"_w{w}")
    if any(col.endswith("_blend") for col in merged.columns):
        suffixes.append("_blend")

    merged = _emit_diffs(
        merged,
        bases=sorted(all_metric_bases),
        suffixes=suffixes,
    )

    # --- G) Light clipping ---
    merged = _clip_by_patterns(merged, _CLIP_RANGES_RAW)
    merged = _clip_by_patterns(merged, _CLIP_RANGES_DIFF)

    # --- I) Diagnostics ---
    if logger.isEnabledFor(logging.DEBUG):
        feat_cols = [c for c in merged.columns if c != "game_id" and (
            c.startswith("home_") or c.startswith("away_") or c.endswith("_diff")
        )]
        null_rate = float(pd.isna(merged[feat_cols]).mean().mean() * 100) if feat_cols else 0.0
        windows_detected = [suf for suf in suffixes if suf]
        logger.debug(
            "[ROLLING] rows=%d | bases=%d | windows=%s | emitted=%d | null_rate≈%.2f%%",
            len(merged), len(all_metric_bases), windows_detected, len(feat_cols), null_rate
        )

    # --- J) Return lean, typed frame ---
    feature_cols = [
        c for c in merged.columns
        if c != "game_id" and (
            c.startswith("home_rolling_") or
            c.startswith("away_rolling_") or
            c.endswith("_diff")
        )
    ]
    # ensure float32 numerics for compactness
    for c in feature_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")

    return merged[["game_id"] + feature_cols]
