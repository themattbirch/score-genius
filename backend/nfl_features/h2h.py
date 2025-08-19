# backend/nfl_features/h2h.py
"""
Head-to-head (H2H) matchup features for NFL games.

Totals are leakage-safe and tempered:
  • Baseline = blend(prev-season league mean, season-to-date league mean up to t_now)
  • Asymmetric caps so upward adjustments aren't choked off
  • Division de-emphasis (configurable) for totals
  • Metric-specific masks for totals to avoid discarding usable evidence

Unit-test-aligned semantics:
  • h2h_games_played       = count of ALL prior meetings before the game (not windowed)
  • h2h_home_win_pct       = plain mean of prior games' (home_team won) over last `max_games`
  • h2h_avg_point_diff     = plain mean of (home_score - away_score) over last `max_games`
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import logging
import numpy as np
import pandas as pd

from .utils import DEFAULTS, normalize_team_name

# Optional division map: canonical team -> "AFC East" etc.
try:
    from .situational import TEAM_DIVISIONS  # Dict[str, str]
except Exception:
    TEAM_DIVISIONS = {}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _normalize_team(s: pd.Series) -> pd.Series:
    return s.apply(normalize_team_name).astype(str).str.lower()


def _mk_ts(df: pd.DataFrame) -> pd.Series:
    """UTC kickoff timestamp (prefer kickoff_ts; else game_date+game_time; else noon)."""
    gd = pd.to_datetime(df.get("game_date", pd.NaT), errors="coerce", utc=True)

    if "game_time" in df.columns:
        tt = df["game_time"].astype(str).fillna("00:00:00")
        ts = pd.to_datetime(gd.dt.strftime("%Y-%m-%d") + " " + tt, errors="coerce", utc=True)
        ts = ts.fillna(gd + pd.Timedelta(hours=12))
    else:
        ts = gd + pd.Timedelta(hours=12)

    if "kickoff_ts" in df.columns:
        ko = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
        ts = ko.where(ko.notna(), ts)

    return ts


def _pair_key_df(df: pd.DataFrame) -> pd.Series:
    a = df[["home_team_norm", "away_team_norm"]].min(axis=1)
    b = df[["home_team_norm", "away_team_norm"]].max(axis=1)
    return (a + "__" + b).astype(str)


def _div_conf(team_norm: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(team_norm, str) or not TEAM_DIVISIONS:
        return None, None
    div = TEAM_DIVISIONS.get(team_norm)
    if not div or not isinstance(div, str):
        return None, None
    conf = div.split()[0].upper() if " " in div else None
    if conf not in ("AFC", "NFC"):
        conf = None
    return div, conf


def _nfl_season_from_ts(ts_utc: pd.Series) -> pd.Series:
    """
    NFL season labeling from UTC ts: months 9–12 → that year; 1–8 → previous year.
    """
    if ts_utc.dt.tz is None:
        z = ts_utc.dt.tz_localize("UTC")
    else:
        z = ts_utc
    y = z.dt.year.astype("int32")
    m = z.dt.month.astype("int32")
    return (y.where(m >= 9, y - 1)).astype("Int64")


def _league_total_mean_by_season(history: pd.DataFrame) -> Dict[int, float]:
    """Leak-safe league total points mean per season from historical results."""
    if history is None or history.empty:
        return {}
    df = history.copy()
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce", utc=True)
    df = df.dropna(subset=["game_date"])
    pts = pd.to_numeric(df.get("home_score"), errors="coerce") + pd.to_numeric(df.get("away_score"), errors="coerce")
    df["total_points"] = pts
    df["season"] = _nfl_season_from_ts(df["game_date"])
    df = df.dropna(subset=["season", "total_points"])
    grp = df.groupby("season", dropna=False)["total_points"].mean().reset_index()
    out: Dict[int, float] = {}
    for _, row in grp.iterrows():
        s = int(row["season"])
        v = float(row["total_points"])
        if np.isfinite(v):
            out[s] = v
    return out


def _prep_league_total_cum_by_season(history: pd.DataFrame) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Precompute, per season, a sorted timeline of (ts_sec, cumulative average of total points).
    Enables O(log n) leak-safe season-to-date lookup at arbitrary t_now.
    Returns: {season: {"ts": np.ndarray[float64], "cum_avg": np.ndarray[float64]}}
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    if history is None or history.empty:
        return out

    df = history.copy()
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce", utc=True)
    df = df.dropna(subset=["game_date"])
    df["season"] = _nfl_season_from_ts(df["game_date"])
    df = df.dropna(subset=["season"])

    df["total_points"] = (
        pd.to_numeric(df.get("home_score"), errors="coerce") +
        pd.to_numeric(df.get("away_score"), errors="coerce")
    ).astype("float64")
    df = df.dropna(subset=["total_points"])

    df = df.sort_values(["season", "game_date"], kind="mergesort")
    for season, g in df.groupby("season", sort=False):
        ts = g["game_date"].view("int64").to_numpy(dtype="float64") / 1e9  # seconds
        csum = np.cumsum(g["total_points"].to_numpy(dtype="float64"))
        ccnt = np.arange(1, len(g) + 1, dtype="float64")
        cav = csum / ccnt
        out[int(season)] = {"ts": ts, "cum_avg": cav}
    return out


def _lookup_s2d_mean(cum_map: Dict[int, Dict[str, np.ndarray]], season: Optional[int], t_now_sec: float) -> Optional[float]:
    """Return season-to-date league mean total points for `season` up to `t_now_sec` (if any)."""
    if season is None or season not in cum_map:
        return None
    arr = cum_map[season]
    ts = arr["ts"]
    idx = np.searchsorted(ts, t_now_sec, side="right") - 1
    if idx >= 0:
        v = arr["cum_avg"][idx]
        return float(v) if np.isfinite(v) else None
    return None


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    max_games: int = 10,
    flag_imputations: bool = True,
    # Decay/shrink controls for totals:
    tau_days: float = 450.0,     # time-decay scale (~1.2y)
    k_div: float = 5.0,          # shrink constant for divisional matchups
    k_conf: float = 8.0,         # same-conference, non-divisional
    k_xconf: float = 12.0,       # cross-conference
    max_years_lookback_nondiv: int = 6,  # ignore non-divisional meetings older than this (for totals evidence only)
    # Stability/tempering:
    point_diff_cap: float = 14.0,      # clip per-game diff before aggregation (when we do compute oriented diffs)
    # Totals-specific guardrails to avoid downward drift:
    k_total_scale: float = 1.5,        # ↓ from 1.75; still conservative but less choking
    # Asymmetric caps (↑ upward headroom; ↓ tighter on downward)
    total_shrink_cap_up: float = 0.75,
    total_shrink_cap_down: float = 0.35,
    total_shrink_cap_lowN_up: float = 0.50,
    total_shrink_cap_lowN_down: float = 0.25,
    n_eff_min_total: float = 2.5,      # below this effective N, apply lowN caps
    # Division handling for totals:
    division_totals_mode: str = "deemphasize",  # {"deemphasize", "delta_only", "normal"}
    k_total_div_multiplier: float = 2.0,        # if "deemphasize", enlarge k for totals in divisional games
    # Baseline blending:
    baseline_prev_weight: float = 0.6,  # baseline = w_prev * prev_season + (1 - w_prev) * season_to_date
    debug: bool = False,
) -> pd.DataFrame:
    """
    Attach leakage-safe, tempered H2H features to `games`.

    NOTE (tests):
      • h2h_games_played = ALL prior meetings (uncut)
      • h2h_home_win_pct & h2h_avg_point_diff = PLAIN MEANS over the last `max_games` meetings (no decay/shrink)
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    # 0) Early exit
    if games is None or games.empty:
        return pd.DataFrame()

    # 1) If no history, emit defaults (plus light matchup context)
    if historical_df is None or historical_df.empty:
        base = games[["game_id", "home_team_norm", "away_team_norm"]].copy()
        base["home_team_norm"] = _normalize_team(base["home_team_norm"])
        base["away_team_norm"] = _normalize_team(base["away_team_norm"])
        # division flag (best-effort)
        is_div: List[int] = []
        for h, a in zip(base["home_team_norm"], base["away_team_norm"]):
            dh, _ = _div_conf(h)
            da, _ = _div_conf(a)
            is_div.append(int(dh is not None and da is not None and dh == da))

        out = games[["game_id"]].copy()
        out["h2h_games_played"] = np.int16(0)
        out["h2h_home_win_pct"] = np.float32(DEFAULTS.get("matchup_home_win_pct", 0.5))
        out["h2h_avg_point_diff"] = np.float32(DEFAULTS.get("matchup_avg_point_diff", 0.0))
        out["h2h_avg_total_points"] = np.float32(DEFAULTS.get("matchup_avg_total_points", 44.0))
        out["h2h_total_delta"] = np.float32(0.0)
        out["h2h_days_since_last_meeting"] = np.float32(3650.0)
        out["h2h_effective_n"] = np.float32(0.0)
        out["is_division_matchup"] = pd.Series(is_div, index=base.index, dtype="int8")
        if flag_imputations:
            for c in ("h2h_games_played", "h2h_home_win_pct", "h2h_avg_point_diff",
                      "h2h_avg_total_points", "h2h_days_since_last_meeting"):
                out[f"{c}_imputed"] = np.int8(1)
        return out

    # 2) Normalize & prepare combined timeline (history + upcoming)
    hist = historical_df.copy()
    hist["home_team_norm"] = _normalize_team(hist["home_team_norm"])
    hist["away_team_norm"] = _normalize_team(hist["away_team_norm"])
    hist["game_date"] = pd.to_datetime(hist.get("game_date"), errors="coerce", utc=True)
    hist["_is_upcoming"] = False

    upc = games.copy()
    upc["home_team_norm"] = _normalize_team(upc["home_team_norm"])
    upc["away_team_norm"] = _normalize_team(upc["away_team_norm"])
    upc["game_date"] = pd.to_datetime(upc.get("game_date"), errors="coerce", utc=True)
    upc["_is_upcoming"] = True

    combined = pd.concat([hist, upc], ignore_index=True, sort=False)
    combined["__ts__"] = _mk_ts(combined)
    combined = combined.dropna(subset=["__ts__"])
    combined["pair_key"] = _pair_key_df(combined)

    # Observability metadata
    div_flag, conf_same = [], []
    for h, a in zip(combined["home_team_norm"], combined["away_team_norm"]):
        dh, ch = _div_conf(h)
        da, ca = _div_conf(a)
        div_flag.append(int(dh is not None and da is not None and dh == da))
        conf_same.append(int(ch is not None and ca is not None and ch == ca))
    combined["__is_div__"] = pd.Series(div_flag, index=combined.index, dtype="int8")
    combined["__same_conf__"] = pd.Series(conf_same, index=combined.index, dtype="int8")

    # Scores (history only)
    is_hist_mask = combined["_is_upcoming"] == False
    for c in ("home_score", "away_score"):
        if c in combined.columns:
            combined.loc[is_hist_mask, c] = pd.to_numeric(combined.loc[is_hist_mask, c], errors="coerce")

    # Stable sort & day dedup
    combined = combined.sort_values(["pair_key", "__ts__", "game_id"], kind="mergesort")
    combined["__day__"] = combined["__ts__"].dt.date
    combined = combined.drop_duplicates(["pair_key", "__day__"], keep="last").reset_index(drop=True)

    # Per-game metrics (history rows only)
    combined["__home_win__"] = np.nan
    combined.loc[is_hist_mask, "__home_win__"] = (
        combined.loc[is_hist_mask, "home_score"] > combined.loc[is_hist_mask, "away_score"]
    ).astype("float32")

    combined["__point_diff__"] = np.nan
    if "home_score" in combined.columns and "away_score" in combined.columns:
        combined.loc[is_hist_mask, "__point_diff__"] = (
            pd.to_numeric(combined.loc[is_hist_mask, "home_score"], errors="coerce")
            - pd.to_numeric(combined.loc[is_hist_mask, "away_score"], errors="coerce")
        ).clip(-float(point_diff_cap), float(point_diff_cap)).astype("float32")

    combined["__total_points__"] = np.nan
    if "home_score" in combined.columns and "away_score" in combined.columns:
        combined.loc[is_hist_mask, "__total_points__"] = (
            pd.to_numeric(combined.loc[is_hist_mask, "home_score"], errors="coerce")
            + pd.to_numeric(combined.loc[is_hist_mask, "away_score"], errors="coerce")
        ).astype("float32")

    # Leak-safe league totals baseline components
    league_total_prev = _league_total_mean_by_season(hist)
    league_total_cum = _prep_league_total_cum_by_season(hist)

    # 3) Group-wise pass to compute features
    out_rows: List[tuple] = []

    yr_cut_secs = float(max_years_lookback_nondiv) * 365.0 * 86400.0
    tau = float(tau_days)

    def _blend(a: float, b: float, w_a: float) -> float:
        """blend = w_a * a + (1 - w_a) * b"""
        return float(w_a * a + (1.0 - w_a) * b)

    for _, gdf in combined.groupby("pair_key", sort=False):
        idx = gdf.index.to_numpy()
        ts = gdf["__ts__"].view("int64").to_numpy()  # ns
        ts_sec = ts.astype("float64") / 1e9
        season_cur = _nfl_season_from_ts(gdf["__ts__"]).to_numpy()

        home = gdf["home_team_norm"].to_numpy()
        # For totals we don't need orientation; totals are symmetric

        home_win = pd.to_numeric(gdf["__home_win__"], errors="coerce").to_numpy()
        pt_diff = pd.to_numeric(gdf["__point_diff__"], errors="coerce").to_numpy()
        tot_pts = pd.to_numeric(gdf["__total_points__"], errors="coerce").to_numpy()
        is_upc = gdf["_is_upcoming"].to_numpy().astype(bool)
        is_div = int(gdf["__is_div__"].mode(dropna=False).iloc[0]) if "__is_div__" in gdf else 0
        same_conf = int(gdf["__same_conf__"].mode(dropna=False).iloc[0]) if "__same_conf__" in gdf else 0

        k_type = float(k_div if is_div else (k_conf if same_conf else k_xconf))
        k_type_totals = max(1e-6, k_type * float(k_total_scale))
        if division_totals_mode == "deemphasize" and is_div:
            k_type_totals *= float(k_total_div_multiplier)

        n = len(gdf)

        for i in range(n):
            if not is_upc[i]:
                continue  # only emit for upcoming rows

            t_now = ts_sec[i]
            s_cur = int(season_cur[i]) if pd.notna(season_cur[i]) else None

            # --- Prior indices ---
            prior_idx_all_uncut = np.arange(0, i, dtype=int)          # ALL prior meetings (for count & last-meeting)
            games_played_all = int(prior_idx_all_uncut.size)

            # Window for *plain means* over the last `max_games` meetings (unit-test semantics)
            window_idx = prior_idx_all_uncut[-max_games:]

            # Defaults for plain stats
            if window_idx.size > 0:
                hs = pd.to_numeric(gdf.loc[idx[window_idx], "home_score"], errors="coerce")
                as_ = pd.to_numeric(gdf.loc[idx[window_idx], "away_score"], errors="coerce")

                # Home team's win flag per historical game (no re-orientation)
                home_win_flags = (hs > as_).astype(float)
                home_win_pct_plain = float(home_win_flags.mean()) if home_win_flags.notna().any() else 0.5

                # Plain mean of (home_score - away_score)
                if (hs.notna() & as_.notna()).any():
                    point_diff_plain = float((hs - as_).mean())
                else:
                    point_diff_plain = 0.0
            else:
                home_win_pct_plain = 0.5
                point_diff_plain = 0.0

            # Days since last meeting (based on ALL priors)
            if games_played_all > 0:
                last_hist_idx = prior_idx_all_uncut[-1]
                days_since = (t_now - ts_sec[last_hist_idx]) / 86400.0
            else:
                days_since = 3650.0

            # --- Totals evidence indices (bounded window + optional non-div age cutoff) ---
            start_j = max(0, i - max_games)
            prior_idx_evidence = np.arange(start_j, i, dtype=int)

            if not is_div and (prior_idx_evidence.size > 0) and (max_years_lookback_nondiv > 0):
                age_ok = (t_now - ts_sec[prior_idx_evidence]) <= (yr_cut_secs)
                prior_idx_evidence = prior_idx_evidence[age_ok]

            # Metric-specific masks for totals
            idx_tot = prior_idx_evidence[np.isfinite(tot_pts[prior_idx_evidence])] if prior_idx_evidence.size else np.array([], dtype=int)
            totals = tot_pts[idx_tot] if idx_tot.size else np.array([], dtype="float64")

            # --- League baseline for totals (prev-season + season-to-date up to t_now) ---
            if s_cur is not None and (s_cur - 1) in league_total_prev:
                prev_mean = league_total_prev[s_cur - 1]
            else:
                past = [s for s in league_total_prev.keys() if (s_cur is None) or (s <= (s_cur - 1))]
                prev_mean = league_total_prev[max(past)] if past else float(DEFAULTS.get("matchup_avg_total_points", 44.0))
            s2d_mean = _lookup_s2d_mean(league_total_cum, s_cur, t_now)
            if s2d_mean is None:
                tot_base = float(prev_mean)
            else:
                tot_base = float(baseline_prev_weight * float(prev_mean) + (1.0 - baseline_prev_weight) * float(s2d_mean))

            # If no totals evidence, anchor absolute to baseline and delta=0
            if idx_tot.size == 0:
                out_rows.append((
                    int(gdf.loc[idx[i], "game_id"]),
                    games_played_all,                    # ALL prior meetings
                    float(home_win_pct_plain),           # plain mean over last `max_games`
                    float(point_diff_plain),             # plain mean over last `max_games`
                    float(tot_base),                     # anchored
                    float(0.0),                          # delta
                    float(days_since),
                    int(is_div),
                    float(0.0),                          # n_eff for totals
                ))
                continue

            # Build totals weights (EW decay)
            days_t = (t_now - ts_sec[idx_tot]) / 86400.0
            w_tot = np.exp(-days_t / tau).astype("float64")
            n_eff_t = float(np.sum(w_tot)) if w_tot.size else 0.0

            total_mean = float(np.sum(w_tot * totals) / np.sum(w_tot)) if n_eff_t > 0 else np.nan

            # Shrink for totals
            shrink_t = float(np.clip(n_eff_t / (n_eff_t + k_type_totals), 0.0, 1.0)) if n_eff_t > 0 else 0.0

            # Asymmetric caps by direction & lowN
            delta_sign = np.sign(total_mean - tot_base) if np.isfinite(total_mean) else 0.0
            lowN = (n_eff_t < float(n_eff_min_total))
            cap_up = total_shrink_cap_lowN_up if lowN else total_shrink_cap_up
            cap_dn = total_shrink_cap_lowN_down if lowN else total_shrink_cap_down
            cap = cap_up if (delta_sign > 0) else (cap_dn if (delta_sign < 0) else max(cap_up, cap_dn))
            shrink_t = float(min(shrink_t, cap))

            # Division handling for totals
            if division_totals_mode == "delta_only" and is_div:
                tot_t_raw = float(tot_base)
                total_delta = float((total_mean - tot_base) * shrink_t) if np.isfinite(total_mean) else 0.0
            else:
                if np.isfinite(total_mean):
                    tot_t_raw = float(shrink_t * total_mean + (1.0 - shrink_t) * tot_base)
                    total_delta = float(tot_t_raw - tot_base)
                else:
                    tot_t_raw = float(tot_base)
                    total_delta = 0.0

            out_rows.append((
                int(gdf.loc[idx[i], "game_id"]),
                games_played_all,                    # ALL prior meetings
                float(home_win_pct_plain),           # plain mean over last `max_games`
                float(point_diff_plain),             # plain mean over last `max_games`
                float(tot_t_raw),
                float(total_delta),
                float(days_since),
                int(is_div),
                float(n_eff_t),
            ))

    # 4) Assemble feature frame for upcoming rows only
    if not out_rows:
        base = games[["game_id"]].copy()
        base["h2h_games_played"] = np.int16(0)
        base["h2h_home_win_pct"] = np.float32(DEFAULTS.get("matchup_home_win_pct", 0.5))
        base["h2h_avg_point_diff"] = np.float32(DEFAULTS.get("matchup_avg_point_diff", 0.0))
        base["h2h_avg_total_points"] = np.float32(DEFAULTS.get("matchup_avg_total_points", 44.0))
        base["h2h_total_delta"] = np.float32(0.0)
        base["h2h_days_since_last_meeting"] = np.float32(3650.0)
        base["h2h_effective_n"] = np.float32(0.0)
        base["is_division_matchup"] = np.int8(0)
        if flag_imputations:
            for c in ("h2h_games_played", "h2h_home_win_pct", "h2h_avg_point_diff",
                      "h2h_avg_total_points", "h2h_days_since_last_meeting"):
                base[f"{c}_imputed"] = np.int8(1)
        return base

    feats = pd.DataFrame.from_records(
        out_rows,
        columns=[
            "game_id",
            "h2h_games_played",
            "h2h_home_win_pct",
            "h2h_avg_point_diff",
            "h2h_avg_total_points",
            "h2h_total_delta",
            "h2h_days_since_last_meeting",
            "is_division_matchup",
            "h2h_effective_n",
        ],
    )

    # Keep only upcoming game_ids present in `games`
    feats = feats.merge(games[["game_id"]], on="game_id", how="right")

    # Fill defaults where missing
    fills = {
        "h2h_games_played": 0,
        "h2h_home_win_pct": float(DEFAULTS.get("matchup_home_win_pct", 0.5)),
        "h2h_avg_point_diff": float(DEFAULTS.get("matchup_avg_point_diff", 0.0)),
        "h2h_avg_total_points": float(DEFAULTS.get("matchup_avg_total_points", 44.0)),
        "h2h_total_delta": 0.0,
        "h2h_days_since_last_meeting": 3650.0,
        "h2h_effective_n": 0.0,
        "is_division_matchup": 0,
    }

    for col, dval in fills.items():
        if col not in feats.columns:
            feats[col] = np.nan
        if col == "is_division_matchup":
            feats[col] = feats[col].fillna(dval).astype("int8")
        elif col in ("h2h_games_played",):
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(dval).astype("int16")
        elif col in ("h2h_days_since_last_meeting", "h2h_effective_n", "h2h_total_delta"):
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(dval).astype("float32")
        else:
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(dval).astype("float32")

    if flag_imputations:
        for c in ("h2h_games_played", "h2h_home_win_pct", "h2h_avg_point_diff",
                  "h2h_avg_total_points", "h2h_days_since_last_meeting"):
            imp = feats[c].isna().astype("int8") if c in feats.columns else np.int8(1)
            feats[f"{c}_imputed"] = imp
            feats[c] = feats[c].fillna(fills[c]).astype(feats[c].dtype)

    if feats["game_id"].duplicated().any():
        dups = int(feats["game_id"].duplicated().sum())
        logger.warning("h2h: %d duplicate game_id rows; keeping last", dups)
        feats = feats.sort_values("game_id").drop_duplicates("game_id", keep="last")

    keep = [
        "game_id",
        "h2h_games_played",
        "h2h_home_win_pct",
        "h2h_avg_point_diff",
        "h2h_avg_total_points",
        "h2h_total_delta",
        "h2h_days_since_last_meeting",
        "h2h_effective_n",
        "is_division_matchup",
    ] + [c for c in feats.columns if c.endswith("_imputed")]

    return feats[keep].copy()
