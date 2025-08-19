# backend/nfl_features/momentum.py
"""
Inter-game *momentum* features for NFL teams.

Momentum is computed as an exponentially weighted moving average (EWMA) of
*prior* point differentials, per team. We use robust, per-team UTC timestamps
to ensure strictly pre-game information (no leakage), de-duplicate any same-day
entries, and apply light regularization (blowout caps + sparse-history shrink).

Outputs (for each span S):
    home_momentum_ewma_S
    away_momentum_ewma_S
    momentum_ewma_S_diff
    (+ optional *_imputed flags when `flag_imputations=True`)

Optional:
- `spans` can be a list of ints (default [3, 7]); `span` kept for
  backward-compat if you prefer a single value.
- `emit_home_away_context` to include home-only and away-only EWMAs.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import normalize_team_name, DEFAULTS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["transform"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _mk_ts(df: pd.DataFrame) -> pd.Series:
    """Build a robust UTC kickoff timestamp per row."""
    gd = pd.to_datetime(
        df.get("game_date", pd.Series(index=df.index, dtype="object")),
        errors="coerce",
        utc=True,
    )

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


def _norm_team(s: pd.Series) -> pd.Series:
    return s.apply(normalize_team_name).astype(str).str.lower()


def _prep_long_history(historical: pd.DataFrame) -> pd.DataFrame:
    """Build long-form per-team history with robust UTC timestamp and point_diff."""
    hist = historical.copy()

    for c in ("home_team_norm", "away_team_norm"):
        if c in hist.columns:
            hist[c] = _norm_team(hist[c])

    for c in ("home_score", "away_score"):
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    home = hist.rename(
        columns={
            "home_team_norm": "team",
            "away_team_norm": "opponent",
            "home_score": "team_score",
            "away_score": "opp_score",
        }
    )
    away = hist.rename(
        columns={
            "away_team_norm": "team",
            "home_team_norm": "opponent",
            "away_score": "team_score",
            "home_score": "opp_score",
        }
    )

    long_df = pd.concat([home, away], ignore_index=True, sort=False)

    long_df["__ts__"] = _mk_ts(long_df)
    long_df["__day__"] = long_df["__ts__"].dt.date

    # Sort & de-duplicate same-day duplicates (keep last kickoff per team/day)
    sort_keys = ["team", "__ts__"]
    if "game_id" in long_df.columns:
        sort_keys.append("game_id")
    long_df = long_df.sort_values(sort_keys, kind="mergesort")
    long_df = long_df.drop_duplicates(["team", "__day__"], keep="last")

    # Point differential (cap blowouts)
    long_df["point_diff"] = (
        pd.to_numeric(long_df["team_score"], errors="coerce")
        - pd.to_numeric(long_df["opp_score"], errors="coerce")
    ).clip(lower=-28, upper=28)

    keep = ["team", "opponent", "__ts__", "__day__", "point_diff"]
    if "game_id" in long_df.columns:
        keep.append("game_id")

    return long_df[keep]


def _prep_upcoming_sides(games: pd.DataFrame) -> pd.DataFrame:
    """Make a per-side upcoming table with ['game_id','team','__ts__'] for asof joins."""
    g = games.copy()

    for c in ("home_team_norm", "away_team_norm"):
        if c in g.columns:
            g[c] = _norm_team(g[c])

    g["__ts__"] = _mk_ts(g)

    left = g[["game_id", "__ts__", "home_team_norm", "away_team_norm"]].copy()
    home = left.rename(columns={"home_team_norm": "team"})[["game_id", "__ts__", "team"]]
    away = left.rename(columns={"away_team_norm": "team"})[["game_id", "__ts__", "team"]]

    home["side"] = "home"
    away["side"] = "away"

    up_long = pd.concat([home, away], ignore_index=True, sort=False)
    up_long = up_long.sort_values(["team", "__ts__", "game_id"], kind="mergesort")

    return up_long


def _ensure_spans(span: int, spans: Optional[Sequence[int]]) -> list[int]:
    if spans is None:
        return [int(span)]
    uniq = sorted({int(s) for s in spans if int(s) > 0})
    return uniq or [int(span)]


def _asof_join_per_team(
    left: pd.DataFrame,   # ['game_id','team','__ts__']
    right: pd.DataFrame,  # ['team','__ts__', <value cols...>]
    value_cols: List[str],
    allow_exact_matches: bool = False,
) -> pd.DataFrame:
    """As-of join within each team (no `by=`). Guarantees stable ordering and avoids cross-team constraints."""
    out_parts: List[pd.DataFrame] = []

    for team, lsub in left.groupby("team", sort=False):
        rsub = right[right["team"] == team]
        if rsub.empty:
            tmp = lsub[["game_id"]].copy()
            for c in value_cols:
                tmp[c] = pd.NA
            out_parts.append(tmp)
            continue

        lsub = lsub.sort_values(["__ts__", "game_id"], kind="mergesort")
        rsub = rsub.sort_values(["__ts__"], kind="mergesort")

        joined = pd.merge_asof(
            lsub,
            rsub,
            left_on="__ts__",
            right_on="__ts__",
            direction="backward",
            allow_exact_matches=allow_exact_matches,
        )
        out_parts.append(joined[["game_id"] + value_cols].copy())

    return (
        pd.concat(out_parts, ignore_index=True)
        if out_parts
        else pd.DataFrame(columns=["game_id"] + value_cols)
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def transform(
    games: pd.DataFrame,
    *,
    historical_df: Optional[pd.DataFrame],
    span: int = 5,
    spans: Optional[Sequence[int]] = (3, 7),
    emit_home_away_context: bool = False,
    flag_imputations: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Attach leakage-free momentum features to `games`.

    Parameters
    ----------
    games : DataFrame
        Upcoming or target games. Requires: game_id, game_date (and optionally game_time/kickoff_ts),
        home_team_norm, away_team_norm.
    historical_df : DataFrame
        Historical games with scores; used to compute prior point differentials.
    span : int
        Default single EWMA span if `spans` is None.
    spans : sequence[int]
        One or more EWM spans to compute (default (3, 7)).
    emit_home_away_context : bool
        Also emit home-only and away-only (context) momentum EWMAs.
    flag_imputations : bool
        Emit *_imputed flags when defaults are used.
    debug : bool
        Elevate logger to DEBUG.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    # Early exits
    if games is None or games.empty:
        return pd.DataFrame()

    if historical_df is None or historical_df.empty:
        # Neutral defaults
        base = games[["game_id"]].copy()
        use_spans = _ensure_spans(span, spans)
        default_val = float(DEFAULTS.get("momentum_direction", 0.0))
        for s in use_spans:
            base[f"home_momentum_ewma_{s}"] = default_val
            base[f"away_momentum_ewma_{s}"] = default_val
            base[f"momentum_ewma_{s}_diff"] = 0.0
            if flag_imputations:
                base[f"home_momentum_ewma_{s}_imputed"] = np.int8(1)
                base[f"away_momentum_ewma_{s}_imputed"] = np.int8(1)
        return base

    # 1) Build long history (per team, robust UTC timestamp)
    long_hist = _prep_long_history(historical_df).dropna(subset=["__ts__"])
    if long_hist.empty:
        logger.info("momentum: long history has no timestamps; defaulting.")
        return transform(
            games,
            historical_df=None,
            spans=spans,
            span=span,
            emit_home_away_context=emit_home_away_context,
            flag_imputations=flag_imputations,
            debug=debug,
        )

    # 2) Prepare upcoming sides (per-team rows for asof join)
    up_long = _prep_upcoming_sides(games).dropna(subset=["__ts__", "team"])
    if up_long.empty:
        logger.info("momentum: upcoming schedule lacks timestamps or teams; returning empty.")
        return pd.DataFrame()

    # 3) Per-team shifted inputs and counters (pre-game shift ensures no leakage)
    long_hist = long_hist.sort_values(["team", "__ts__"], kind="mergesort")
    grp_team = long_hist.groupby("team", sort=False, group_keys=False)
    long_hist["point_diff_shift"] = grp_team["point_diff"].shift(1).fillna(0.0)
    long_hist["games_played_before"] = grp_team.cumcount().astype("int32")

    # 4) Compute EWMAs for requested spans (with shrink toward 0 for sparse history)
    use_spans = _ensure_spans(span, spans)
    default_val = float(DEFAULTS.get("momentum_direction", 0.0))

    feat_tables: Dict[int, pd.DataFrame] = {}
    for s in use_spans:
        ewma = grp_team["point_diff_shift"].transform(
            lambda x: x.ewm(span=int(s), adjust=False, min_periods=1).mean()
        )
        ewma = ewma.clip(lower=-21, upper=21).astype("float32")
        shrink = np.minimum(1.0, long_hist["games_played_before"] / float(s)).astype("float32")
        ewma = (ewma * shrink).astype("float32")

        ft = long_hist[["team", "__ts__", "games_played_before"]].copy()
        ft[f"momentum_ewma_{s}"] = ewma.values
        feat_tables[s] = ft  # sorted already by ["team","__ts__"]

    # 5) Optional home/away-context EWMAs
    context_tables: Dict[Tuple[int, str], pd.DataFrame] = {}
    if emit_home_away_context:
        hist = historical_df.copy()
        for c in ("home_team_norm", "away_team_norm"):
            if c in hist.columns:
                hist[c] = _norm_team(hist[c])
        hist["__ts__"] = _mk_ts(hist)
        hist = hist.dropna(subset=["__ts__"])

        home = hist[["__ts__", "home_team_norm"]].rename(columns={"home_team_norm": "team"})
        home["at_home"] = 1
        away = hist[["__ts__", "away_team_norm"]].rename(columns={"away_team_norm": "team"})
        away["at_home"] = 0

        ctx_long = pd.concat([home, away], ignore_index=True, sort=False).dropna(subset=["team"])
        ctx_long = ctx_long.sort_values(["team", "__ts__"], kind="mergesort")

        ctx = ctx_long.merge(
            long_hist[["team", "__ts__", "point_diff_shift", "games_played_before"]],
            on=["team", "__ts__"],
            how="left",
        ).fillna({"point_diff_shift": 0.0, "games_played_before": 0})
        ctx = ctx.sort_values(["team", "__ts__"], kind="mergesort")

        grp_ctx = ctx.groupby(["team", "at_home"], sort=False, group_keys=False)
        for s in use_spans:
            ew = grp_ctx["point_diff_shift"].transform(
                lambda x: x.ewm(span=int(s), adjust=False, min_periods=1).mean()
            )
            ew = ew.clip(-21, 21).astype("float32")
            shrink = np.minimum(1.0, ctx["games_played_before"] / float(s)).astype("float32")
            ew = (ew * shrink).astype("float32")

            t = ctx[["team", "__ts__", "at_home"]].copy()
            t[f"momentum_home_ewma_{s}"] = ew.values

            t_home = t[t["at_home"] == 1][["team", "__ts__", f"momentum_home_ewma_{s}"]]
            t_away = t[t["at_home"] == 0][["team", "__ts__", f"momentum_home_ewma_{s}"]]
            context_tables[(s, "home")] = t_home.sort_values(["team", "__ts__"], kind="mergesort")
            context_tables[(s, "away")] = t_away.sort_values(["team", "__ts__"], kind="mergesort")

    # 6) Per-side joins (no by=; join within team)
    out_parts: List[pd.DataFrame] = []
    for side in ("home", "away"):
        side_left = up_long[up_long["side"] == side][["game_id", "team", "__ts__"]].copy()
        side_left = side_left.sort_values(["team", "__ts__", "game_id"], kind="mergesort")

        # Accumulator starts with game_id
        acc = side_left[["game_id", "team", "__ts__"]].copy()

        for s in use_spans:
            ft = feat_tables[s]  # ['team','__ts__', f'momentum_ewma_{s}', 'games_played_before']
            val_col = f"momentum_ewma_{s}"
            joined_vals = _asof_join_per_team(side_left, ft[["team", "__ts__", val_col]], [val_col])

            col = f"{side}_momentum_ewma_{s}"
            acc[col] = joined_vals[val_col].values
            if flag_imputations:
                acc[f"{col}_imputed"] = acc[col].isna().astype("int8")
            acc[col] = acc[col].fillna(default_val).astype("float32")

        if emit_home_away_context:
            ctx_key = "home" if side == "home" else "away"
            for s in use_spans:
                tctx = context_tables.get((s, ctx_key))
                if tctx is not None and not tctx.empty:
                    cval = f"momentum_home_ewma_{s}"
                    joined_ctx = _asof_join_per_team(side_left, tctx[["team", "__ts__", cval]], [cval])
                    out_col = f"{side}_momentum_{ctx_key}_ewma_{s}"
                    acc[out_col] = joined_ctx[cval].values
                    if flag_imputations:
                        acc[f"{out_col}_imputed"] = acc[out_col].isna().astype("int8")
                    acc[out_col] = acc[out_col].fillna(default_val).astype("float32")

        out_parts.append(acc.drop(columns=["team", "__ts__"]))

    # 7) Assemble wide + diffs
    home_wide, away_wide = out_parts
    out = home_wide.merge(away_wide, on="game_id", how="inner")

    for s in use_spans:
        out[f"momentum_ewma_{s}_diff"] = (
            out[f"home_momentum_ewma_{s}"] - out[f"away_momentum_ewma_{s}"]
        ).astype("float32")

        if emit_home_away_context:
            hcol = f"home_momentum_home_ewma_{s}"
            acol = f"away_momentum_away_ewma_{s}"
            if hcol in out.columns and acol in out.columns:
                out[f"momentum_home_context_ewma_{s}_diff"] = (out[hcol] - out[acol]).astype("float32")

    # Ensure imputed flags are int8 (if any)
    for c in [c for c in out.columns if c.endswith("_imputed")]:
        out[c] = out[c].astype("int8")

    # Deduplicate by game_id if needed
    if out["game_id"].duplicated().any():
        dup_ct = int(out["game_id"].duplicated().sum())
        logger.info("momentum: %d duplicate game_id rows; keeping last", dup_ct)
        out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

    # Final column order: game_id first
    keep = ["game_id"] + [c for c in out.columns if c != "game_id"]
    return out[keep]
