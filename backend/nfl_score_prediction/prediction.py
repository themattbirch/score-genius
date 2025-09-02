# backend/nfl_score_prediction/prediction.py
"""
prediction.py – NFL Score Prediction Generation Script (accuracy-tuned, rich DEBUG)

Pipeline:
1) Fetch schedule (date-bounded) + historical context (lookback).
2) Build features via NFLFeatureEngine().build_features().
3) Load artifacts (selected features + ensemble weights + train stats); deterministic overlay.
4) Optional pre-model shaping:
   • H2H shrink toward priors (week-aware k)
   • Week-adaptive attenuation of volatile diffs (or explicit alphas)
   • Data-driven clipping via train quantiles if available (fallback constants)
   • Train-mean imputation for advanced features using feature_means (+ flags)
5) Vectorized ensemble predictions for margin & total.
6) Soft mean calibration (week-aware priors; tunable lambdas) [+ optional market blend].
7) Contextual feasibility guard (total ≥ |margin| + buffer that respects week, dome/cold).
8) Derive home/away, clamp non-negatives, tiny floor bump when feasibility just activates.
9) Optional residual bias corrector from recent history (tiny ridge; guarded & capped).
10) Upsert and/or print.

CLI adds:
  --market-w                        : blend weight toward market lines (default 0.0 off)
  --residual-corrector / --no-residual-corrector
  --atten-total-alpha / --atten-margin-alpha can be omitted to enable week-adaptive defaults
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from supabase import Client, create_client

# Optional: tiny residual corrector
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SK_OK = True
except Exception:
    _SK_OK = False

# Project imports
try:
    from backend import config
    from backend.nfl_features.engine import NFLFeatureEngine
    from backend.nfl_score_prediction.data import NFLDataFetcher
    from backend.nfl_score_prediction.ensemble import NFLEnsemble
    from backend.nfl_score_prediction.models import MODEL_DIR, derive_scores_from_predictions
except ImportError as e:
    print(f"Import error: {e}. Ensure PYTHONPATH includes project root.")
    sys.exit(1)

# ----------------------------------------------------------------------------- #
# Logging / paths
# ----------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ----------------------------------------------------------------------------- #
# Small diagnostics helpers
# ----------------------------------------------------------------------------- #
def _df_profile(df: pd.DataFrame, name: str, top_n: int = 10) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"name": name, "rows": 0, "cols": 0, "null_top": {}, "zero_top": {}}
    nulls = df.isna().sum().sort_values(ascending=False)
    num_cols = df.select_dtypes(include="number").columns
    zeros = (df[num_cols] == 0).sum().sort_values(ascending=False) if len(num_cols) else pd.Series(dtype=int)
    return {
        "name": name,
        "rows": len(df),
        "cols": df.shape[1],
        "null_top": {k: int(v) for k, v in nulls.head(top_n).to_dict().items() if v > 0},
        "zero_top": {k: int(v) for k, v in zeros.head(top_n).to_dict().items() if v > 0},
    }

def _log_df(df: pd.DataFrame, name: str, debug: bool):
    if not debug:
        return
    prof = _df_profile(df, name)
    logger.debug(
        "[DF] %-22s rows=%-5d cols=%-4d | null_top=%s | zero_top=%s",
        prof["name"], prof["rows"], prof["cols"], prof.get("null_top"), prof.get("zero_top")
    )

def _order_crc(cols: Sequence[str]) -> str:
    m = hashlib.md5()
    m.update(",".join(list(cols)).encode("utf-8"))
    return m.hexdigest()[:8]

def _safe_int(v, default=None):
    try:
        if v is None:
            return default
        try:
            import pandas as pd  # local import to avoid hard dep here
            if pd.isna(v):
                return default
        except Exception:
            pass
        return int(v)
    except Exception:
        return default

def _check_features(X: pd.DataFrame, needed: Sequence[str], label: str, debug: bool):
    if not debug:
        return
    missing = [c for c in needed if c not in X.columns]
    extra   = [c for c in X.columns if c not in needed]
    if missing:
        logger.debug("[%s][WARN] missing=%d: %s", label, len(missing), missing[:20])
    if extra:
        logger.debug("[%s] extra=%d (ignored in predict matrix)", label, len(extra))
    non_numeric = [c for c in needed if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        logger.debug("[%s][WARN] non-numeric columns: %s", label, non_numeric)

# ----------------------------------------------------------------------------- #
# Focus selection
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Focus:
    date_str: Optional[str]                 # 'YYYY-MM-DD'
    pairs: Set[Tuple[int, int]]             # {(away_id, home_id), ...}
    ids: Set[int]                           # {game_id, ...}

def _parse_focus_games(arg: Optional[str]) -> Set[Tuple[int, int]]:
    if not arg:
        return set()
    out: Set[Tuple[int, int]] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok or "@" not in tok:
            continue
        a, h = tok.split("@", 1)
        try:
            out.add((int(a), int(h)))
        except ValueError:
            continue
    return out

def _parse_focus_ids(arg: Optional[str]) -> Set[int]:
    if not arg:
        return set()
    out: Set[int] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out

def _normalize_date_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.strip())
        return d.date().isoformat()
    except Exception:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date().isoformat()
        except Exception:
            return None
        
def _soft_mean_calibration(
    series: pd.Series,
    target_mean: float,
    lam: float,
    min_trigger: float,
    label: str,
    debug: bool,
) -> pd.Series:
    """
    Shift a prediction series toward target_mean by lam * (target - observed)
    if the absolute mean gap is at least min_trigger.

    No NaN filling is performed; existing NaNs are preserved.
    """
    if series is None or len(series) == 0 or lam <= 0.0:
        return series

    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return series

    obs_mean = float(s.mean())
    delta = float(target_mean) - obs_mean
    if abs(delta) < float(min_trigger):
        if debug:
            logger.debug(
                "PRED[CAL] %s mean-calibration skipped | obs=%.2f target=%.2f (|Δ|=%.2f < %.2f)",
                label, obs_mean, target_mean, abs(delta), min_trigger
            )
        return series

    adjusted = s.astype(float) + float(lam) * delta
    if debug:
        logger.debug(
            "PRED[CAL] %s mean-calibration | obs=%.2f → target=%.2f (λ=%.2f, Δ=%.2f)",
            label, obs_mean, target_mean, lam, delta
        )
    # keep original index & dtype behavior (don’t coerce NaNs)
    return adjusted.reindex(series.index)


def _ensure_season_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure season is populated (derived from game_date) and core IDs are numeric.
    NFL season rule: months 9–12 => that year, months 1–8 => previous year.
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce", utc=False)

        def _season_from_date(d: pd.Timestamp) -> Optional[int]:
            if pd.isna(d):
                return None
            y = d.year
            return y if d.month >= 9 else (y - 1)

        derived = out["game_date"].apply(_season_from_date)
        if "season" not in out.columns:
            out["season"] = derived
        else:
            out["season"] = out["season"].where(out["season"].notna(), derived)

    for col in ("game_id", "home_team_id", "away_team_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_supabase_client() -> Optional[Client]:
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error("Supabase client init failed: %s", e)
        return None

def _load_core_artifacts(model_dir: Path, debug: bool = False) -> Dict[str, Any]:
    """
    Deterministic artifact loader:
      - Always require {margin,total}_{selected_features,ensemble_weights}
      - Overlay extras from first existing file among:
            nfl_artifacts.json, nfl_train_artifacts.json, artifacts.json
      - If feature_means absent, try {nfl_feature_means.json, feature_means.json}
    """
    out: Dict[str, Any] = {}

    # Required pairs
    for target in ("margin", "total"):
        feats_path = model_dir / f"nfl_{target}_selected_features.json"
        wts_path   = model_dir / f"nfl_{target}_ensemble_weights.json"
        try:
            out[f"{target}_features"] = json.loads(feats_path.read_text())
            out[f"{target}_weights"]  = json.loads(wts_path.read_text())
            if debug:
                logger.debug("PRED[ART] %s selected(n=%d) | weights(n=%d)",
                             target.upper(), len(out[f"{target}_features"]), len(out[f"{target}_weights"]))
        except FileNotFoundError as e:
            logger.critical("Missing artifact for %s: %s", target, e)
            return {}

    # Optional overlay, first file wins per key (left-to-right precedence)
    overlay_sources = [
        model_dir / "nfl_artifacts.json",
        model_dir / "nfl_train_artifacts.json",
        model_dir / "artifacts.json",
    ]
    overlay: Dict[str, Any] = {}
    for p in overlay_sources:
        try:
            if p.exists():
                d = json.loads(p.read_text())
                # only set keys that aren't set yet to respect first-win
                for k, v in d.items():
                    if k not in overlay:
                        overlay[k] = v
                if debug:
                    logger.debug("PRED[ART] overlay source used: %s (keys=%d)", p.name, len(d))
        except Exception as e:
            if debug:
                logger.debug("PRED[ART] Skipping overlay %s: %s", p, e)

    out.update(overlay)

    # feature_means fallback file(s)
    if not out.get("feature_means"):
        for fname in ("nfl_feature_means.json", "feature_means.json"):
            p = model_dir / fname
            try:
                if p.exists():
                    out["feature_means"] = json.loads(p.read_text())
                    if debug:
                        logger.debug("PRED[ART] feature_means loaded from %s (keys=%d)", p.name, len(out["feature_means"]))
                    break
            except Exception as e:
                if debug:
                    logger.debug("PRED[ART] Could not read %s: %s", p, e)

    # Final means with defaults
    out["total_train_mean"]  = float(out.get("total_train_mean", 43.5))
    out["margin_train_mean"] = float(out.get("margin_train_mean", 0.0))
    out["feature_means"] = out.get("feature_means", {}) or {}
    if debug:
        logger.debug(
            "PRED[ART] means: total_train_mean=%.2f margin_train_mean=%.2f | feature_means=%d keys",
            out["total_train_mean"], out["margin_train_mean"], len(out["feature_means"])
        )
    return out

def chunked(lst: List[Dict[str, Any]], size: int = 500):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def upsert_score_predictions(preds: List[Dict[str, Any]], sb: Client, debug: bool = False) -> int:
    if not preds:
        return 0
    updated = 0
    for p in preds:
        gid = int(p["game_id"])
        payload = {
            "predicted_home_score": float(p["predicted_home_score"]),
            "predicted_away_score": float(p["predicted_away_score"]),
        }
        try:
            resp = (
                sb.table("nfl_game_schedule")
                  .update(payload)
                  .eq("game_id", gid)
                  .execute()
            )
            n = len(resp.data or [])
            updated += n
            if debug:
                logger.debug("Update game_id=%s -> modified=%d", gid, n)
        except Exception as e:
            logger.error("Update error for game_id=%s: %s", gid, e)
    logger.info("Updated predictions for %d games", updated)
    return updated

def display_prediction_summary(preds: List[Dict[str, Any]]) -> None:
    if not preds:
        return
    df = pd.DataFrame(preds).sort_values("game_date")
    print("-" * 80)
    print(f"{'DATE':<12}{'MATCHUP':<42}{'PREDICTED SCORE':<20}")
    print("-" * 80)
    for _, r in df.iterrows():
        d = datetime.fromisoformat(str(r["game_date"])).strftime("%Y-%m-%d")
        hn = r.get("home_team_name") or f"home_id {r.get('home_team_id')}"
        an = r.get("away_team_name") or f"away_id {r.get('away_team_id')}"
        matchup = f"{an} @ {hn}"
        score = f"{r['predicted_away_score']:.1f} - {r['predicted_home_score']:.1f}"
        print(f"{d:<12}{matchup:<42}{score:<20}")
    print("-" * 80)

# ----------------------------------------------------------------------------- #
# Feature shaping utilities
# ----------------------------------------------------------------------------- #
def _clip_feature_columns(
    df: pd.DataFrame,
    bounds: Dict[str, tuple],
    debug: bool = False,
    focus_ids: Optional[set] = None,
) -> pd.DataFrame:
    out = df.copy()
    focus_ids = set(focus_ids or [])
    clipped_stats = {}

    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        pre = s.copy()
        out[col] = s.clip(lower=float(lo), upper=float(hi))
        changed = (out[col] != pre) & pre.notna()
        n_changed = int(changed.sum())
        if n_changed > 0:
            clipped_stats[col] = n_changed
            if debug and focus_ids and "game_id" in out.columns:
                for gid in focus_ids:
                    mask = (out["game_id"] == gid)
                    if mask.any():
                        pre_v = pre.loc[mask].iloc[0]
                        post_v = out.loc[mask, col].iloc[0]
                        if pd.notna(pre_v) and pd.notna(post_v) and float(pre_v) != float(post_v):
                            logger.debug("PRED[CLIP] game_id=%s %s: %.3f → %.3f (lo=%.2f hi=%.2f)",
                                         gid, col, float(pre_v), float(post_v), lo, hi)

    if debug:
        logger.debug("PRED[CLIP] counts=%s", clipped_stats or {})
    return out

def _attenuate_total_diffs(
    X_total: pd.DataFrame,
    alpha: float = 0.85,
    debug: bool = False,
) -> pd.DataFrame:
    if X_total is None or X_total.empty or not (alpha < 1.0):
        return X_total
    diffs = [
        "rolling_points_for_avg_diff",
        "rolling_points_against_avg_diff",
        "rolling_yards_per_play_avg_diff",
    ]
    cols = [c for c in diffs if c in X_total.columns]
    if not cols:
        return X_total
    X = X_total.copy()
    X.loc[:, cols] = X.loc[:, cols] * float(alpha)
    if debug:
        logger.debug("PRED[ATTN] TOTAL diffs scaled by alpha=%.2f, cols=%s", alpha, cols)
    return X

def _attenuate_margin_diffs(
    X: pd.DataFrame,
    alpha: float = 0.6,
    debug: bool = False,
) -> pd.DataFrame:
    if X is None or X.empty:
        return X
    cols = [
        "rolling_points_for_avg_diff",
        "rolling_points_against_avg_diff",
        "rolling_point_differential_avg_diff",
        "rolling_yards_per_play_avg_diff",
        "rolling_turnover_differential_avg_diff",
        "momentum_ewma_5_diff",
    ]
    existing = [c for c in cols if c in X.columns]
    if not existing:
        return X.copy()
    X2 = X.copy()
    X2.loc[:, existing] = X2.loc[:, existing] * float(alpha)
    if "h2h_avg_point_diff" in X2.columns:
        X2.loc[:, "h2h_avg_point_diff"] = X2["h2h_avg_point_diff"].clip(-7.0, 7.0)
    if debug:
        logger.debug("PRED[ATTN][MARGIN] diffs scaled by alpha=%.2f, cols=%s", alpha, existing)
    return X2

def _shrink_h2h_features(
    df: pd.DataFrame,
    mu_total: float = 44.0,
    mu_margin: float = 0.0,
    k: float = 6.0,
    debug: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["h2h_games_played", "h2h_avg_total_points", "h2h_avg_point_diff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    w_raw = out["h2h_games_played"].astype(float) if "h2h_games_played" in out.columns else pd.Series(0.0, index=out.index)
    w = (w_raw / float(k)).clip(lower=0.0, upper=1.0).fillna(0.0) if k > 0 else pd.Series(0.0, index=out.index)

    def _shrink(series: pd.Series, mu: float) -> Tuple[pd.Series, pd.Series]:
        base = pd.to_numeric(series, errors="coerce").fillna(mu)
        shrunk = mu + w * (base - mu)
        delta = shrunk - base
        return shrunk, delta

    if "h2h_avg_total_points" in out.columns:
        out["h2h_avg_total_points"], _ = _shrink(out["h2h_avg_total_points"], mu_total)
    if "h2h_avg_point_diff" in out.columns:
        out["h2h_avg_point_diff"], _ = _shrink(out["h2h_avg_point_diff"], mu_margin)

    if debug:
        touched = int((w > 0).sum())
        logger.debug("PRED[H2H][SUMMARY] rows_touched=%d (k=%.1f, muT=%.1f, muM=%.1f)", touched, k, mu_total, mu_margin)
    return out

# ----------------------------------------------------------------------------- #
# Bucketing (for debug snapshots)
# ----------------------------------------------------------------------------- #
_BUCKET_RULES: List[Tuple[str, str]] = [
    (r"^rolling_", "rolling"),
    (r"(?:^season_|_season_)", "season"),
    (r"^(?:advanced_|adv_)", "advanced"),
    (r"^(?:form_|momentum_)", "form"),
    (r"^rest_", "rest"),
    (r"^h2h_", "h2h"),
    (r"^drive_", "drive"),
]

def _bucket_of(feature: str) -> str:
    for pat, bucket in _BUCKET_RULES:
        if re.search(pat, feature):
            return bucket
    if feature.endswith("_diff"):
        return "diffs"
    return "other"

def _row_snapshot(
    gid: int,
    row_meta: Mapping[str, Any],
    X_row: pd.Series,
    selected_cols: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    impute_flags: Dict[str, Any] = {}
    for col in selected_cols:
        val = X_row[col] if col in X_row.index else 0.0
        b = _bucket_of(col)
        grouped.setdefault(b, {})
        grouped[b][col] = float(val) if pd.notna(val) else None
        if col.endswith("_imputed") or "imputed" in col:
            impute_flags[col] = grouped[b][col]
    counts = {b: len(cols) for b, cols in grouped.items()}
    meta = {
        "game_id": gid,
        "date": (row_meta.get("game_date").date().isoformat()
                 if isinstance(row_meta.get("game_date"), datetime)
                 else (row_meta.get("game_date").isoformat() if isinstance(row_meta.get("game_date"), date)
                       else str(row_meta.get("game_date"))[:10])),
        "away_id": _safe_int(row_meta.get("away_team_id")),
        "home_id": _safe_int(row_meta.get("home_team_id")),
        "key": f"{row_meta.get('away_team_id')}@{row_meta.get('home_team_id')}",
    }
    return {"meta": meta, "grouped": grouped, "impute_flags": impute_flags, "bucket_counts": counts}

def _log_snapshot(tag: str, snap: Dict[str, Dict[str, Any]]):
    m = snap["meta"]
    logger.debug(
        "PRED[ROW] %s game_id=%s, date=%s, away_id=%s, home_id=%s, key=%s",
        tag, m["game_id"], m["date"], m["away_id"], m["home_id"], m["key"]
    )
    for bucket in ("season", "rolling", "advanced", "form", "rest", "h2h", "drive", "diffs", "other"):
        if bucket in snap["grouped"] and snap["grouped"][bucket]:
            logger.debug("  %s=%s", bucket, {k: snap["grouped"][bucket][k] for k in list(snap["grouped"][bucket].keys())[:12]})
    if snap["impute_flags"]:
        logger.debug("  impute_flags=%s", snap["impute_flags"])
    logger.debug("  bucket_counts=%s", snap["bucket_counts"])

# ----------------------------------------------------------------------------- #
# Smarter composites + matrix builders + targeted imputation
# ----------------------------------------------------------------------------- #
def add_total_composites_smart(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _sum_if_present(h: str, a: str, t: str):
        if h in out.columns and a in out.columns and t not in out.columns:
            out[t] = pd.to_numeric(out[h], errors="coerce") + pd.to_numeric(out[a], errors="coerce")

    for base in [
        "rolling_points_for_avg",
        "rolling_points_against_avg",
        "rolling_yards_per_play_avg",
        "rolling_turnover_differential_avg",
        "form_win_pct_5",
    ]:
        _sum_if_present(f"home_{base}", f"away_{base}", f"total_{base}")

    for base in [
        "adv_red_zone_pct",
        "adv_third_down_pct",
        "adv_time_of_possession_seconds",
        "adv_turnovers_per_game",
        "adv_yards_per_drive",
        "adv_pythagorean_win_pct",
    ]:
        _sum_if_present(f"home_{base}", f"away_{base}", f"total_{base}")

    drive_pairs = [
        "points_per_drive_avg",
        "yards_per_play_avg",
        "plays_per_drive_avg",
        "turnovers_per_drive_avg",
        "red_zone_td_pct_avg",
        "seconds_per_play_avg",
        "seconds_per_drive_avg",
        "points_allowed_per_drive_avg",
        "yards_per_play_allowed_avg",
        "turnovers_forced_per_drive_avg",
        "sacks_made_per_drive_avg",
    ]
    for base in drive_pairs:
        _sum_if_present(f"drive_home_{base}", f"drive_away_{base}", f"drive_total_{base}")

    return out

def _approx_season_week(dates: pd.Series) -> int:
    if dates is None or len(dates) == 0:
        return 18
    ds = pd.to_datetime(dates, errors="coerce")
    seasons = ds.dt.year.where(ds.dt.month >= 9, ds.dt.year - 1)
    season_start = pd.to_datetime(seasons.astype(str) + "-09-01")
    offset = ((3 - season_start.dt.dayofweek) % 7).astype(int)  # next Thu
    kickoff = season_start + pd.to_timedelta(offset, unit="D")
    weeks = ((ds - kickoff).dt.days // 7) + 1
    weeks = weeks.clip(lower=1, upper=22).fillna(18)
    return int(weeks.median())

def _dynamic_h2h_k(upcoming_df: pd.DataFrame, base_k: float) -> float:
    wk = _approx_season_week(upcoming_df["game_date"]) if "game_date" in upcoming_df.columns else 18
    return max(float(base_k), 10.0) if wk <= 10 else float(base_k)

def _feature_fill_value(feat: str, feature_means: Mapping[str, float]) -> float:
    if feat in feature_means:
        return float(feature_means[feat])
    if feat.endswith("_diff"):
        return 0.0
    return 0.0

def _build_matrix(
    features_df: pd.DataFrame,
    selected: Sequence[str],
    feature_means: Mapping[str, float],
    label: str,
    debug: bool = False,
) -> pd.DataFrame:
    X = features_df.reindex(columns=list(selected), fill_value=np.nan).copy()

    missing_cols = [c for c in selected if c not in features_df.columns]
    created_with_mean = {}
    for c in missing_cols:
        fv = _feature_fill_value(c, feature_means)
        X[c] = fv
        created_with_mean[c] = fv

    before_na = X.isna().sum().sum()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    na_counts = X.isna().sum()
    filled_counts = {}
    for c, n_na in na_counts.items():
        if n_na > 0:
            fv = _feature_fill_value(c, feature_means)
            X[c] = X[c].fillna(fv)
            filled_counts[c] = int(n_na)

    if debug:
        ord_crc = _order_crc(X.columns)
        if created_with_mean:
            logger.debug("PRED[%s] created %d cols using train mean/fallback (sample=%s)",
                         label, len(created_with_mean), dict(list(created_with_mean.items())[:5]))
        if filled_counts:
            logger.debug("PRED[%s] filled NaNs in %d cols (sample=%s) | order_crc=%s",
                         label, len(filled_counts), dict(list(filled_counts.items())[:5]), ord_crc)
        else:
            logger.debug("PRED[%s] no NaNs to fill | order_crc=%s", label, ord_crc)

    return X

def _impute_advanced_features(
    df: pd.DataFrame,
    feature_means: Mapping[str, float],
    debug: bool = False
) -> pd.DataFrame:
    out = df.copy()
    adv_cols = [c for c in out.columns if c.startswith(("adv_", "advanced_"))]
    if not adv_cols:
        return out

    touched = {}
    for c in adv_cols:
        col = pd.to_numeric(out[c], errors="coerce")
        flag_col = f"{c}_imputed"
        if flag_col in out.columns:
            flags = pd.to_numeric(out[flag_col], errors="coerce").fillna(0).astype(int)
            need = col.isna() | ((flags == 1) & (col.fillna(0.0) == 0.0))
        else:
            need = col.isna()

        if need.any():
            if c in feature_means:
                fill_val = float(feature_means[c])
                out.loc[need, c] = fill_val
                if flag_col not in out.columns:
                    out[flag_col] = 0
                out.loc[need, flag_col] = 1
                touched[c] = int(need.sum())
            else:
                if debug:
                    logger.debug("PRED[IMPUTE][SKIP] %s has NaNs but no train mean; left untouched.", c)

    if debug and touched:
        logger.debug("PRED[IMPUTE] advanced imputed with train means: %s", touched)
    return out

def _normalize_team_key(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("washington football team", "washington commanders")
    t = t.replace("oakland raiders", "las vegas raiders")
    t = t.replace("san diego chargers", "los angeles chargers")
    t = t.replace("st. louis rams", "los angeles rams")
    return t

def _fetch_team_mapping(sb: Client) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        resp = (
            sb.table("nfl_historical_game_team_stats")
              .select("team_id,team_name")
              .limit(100000)
              .execute()
        )
        rows = resp.data or []
        for r in rows:
            tid = r.get("team_id")
            tname = r.get("team_name")
            if tid is None or tname is None:
                continue
            key = _normalize_team_key(tname)
            if key and key not in mapping:
                mapping[key] = int(tid)
    except Exception as e:
        logger.debug("TEAM MAP (game_team_stats) skipped: %s", e)

    if not mapping:
        try:
            resp = (
                sb.table("nfl_historical_team_stats")
                  .select("team_id,team_name")
                  .limit(100000)
                  .execute()
            )
            rows = resp.data or []
            for r in rows:
                tid = r.get("team_id")
                tname = r.get("team_name")
                if tid is None or tname is None:
                    continue
                key = _normalize_team_key(tname)
                if key and key not in mapping:
                    mapping[key] = int(tid)
        except Exception as e:
            logger.debug("TEAM MAP (team_stats) skipped: %s", e)

    if not mapping:
        logger.warning("TEAM MAP: no mappings found; schedule hydration will be a no-op.")
    else:
        logger.debug("TEAM MAP: loaded %d name→id entries", len(mapping))
    return mapping

def _hydrate_schedule_team_ids(df: pd.DataFrame, mapping: Dict[str, int]) -> pd.DataFrame:
    if df is None or df.empty or not mapping:
        return df

    out = df.copy()
    if "home_team_name" not in out.columns and "home_team" in out.columns:
        out["home_team_name"] = out["home_team"]
    if "away_team_name" not in out.columns and "away_team" in out.columns:
        out["away_team_name"] = out["away_team"]

    for col in ("home_team_name", "away_team_name"):
        if col in out.columns:
            out[f"__norm__{col}"] = out[col].apply(_normalize_team_key)
        else:
            out[f"__norm__{col}"] = None

    for side in ("home", "away"):
        id_col = f"{side}_team_id"
        name_norm_col = f"__norm__{side}_team_name"
        if id_col not in out.columns:
            out[id_col] = pd.Series([pd.NA] * len(out), dtype="Int64")

        def _lookup(row):
            if pd.notna(row.get(id_col)):
                return row.get(id_col)
            key = row.get(name_norm_col)
            if key and key in mapping:
                return int(mapping[key])
            return pd.NA

        out[id_col] = out.apply(_lookup, axis=1).astype("Int64")

    drop_cols = [c for c in out.columns if c.startswith("__norm__")]
    out.drop(columns=drop_cols, inplace=True, errors="ignore")
    out = _ensure_season_and_ids(out)
    return out

# ----------------------------------------------------------------------------- #
# Core
# ----------------------------------------------------------------------------- #
def generate_predictions(
    *,
    # Windowing
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    days_window: Optional[int] = None,
    all_games: bool = False,

    # History
    historical_lookback: int = 1825,

    # Debug
    debug_mode: bool = False,
    focus: Optional[Focus] = None,
    dump_dir: Optional[Path] = None,

    # Attenuation knobs (None → week-adaptive)
    no_atten: bool = False,
    atten_total_alpha: Optional[float] = None,
    atten_margin_alpha: Optional[float] = None,

    # H2H shrink
    no_h2h_shrink: bool = False,
    h2h_k: float = 6.0,

    # Mean calibration (soft)
    mean_cal_lambda_total: float = 0.45,
    mean_cal_lambda_margin: float = 0.50,
    mean_cal_min_trigger_total: float = 0.25,
    mean_cal_min_trigger_margin: float = 0.15,

    # Feasibility + caps
    feas_buffer_max: float = 6.0,
    margin_cap: float = 24.0,

    # Optional TOTAL clip band (post-calibration)
    total_clip_min: Optional[float] = None,
    total_clip_max: Optional[float] = None,

    # Optional market blend
    market_w: float = 0.0,

    # Residual corrector
    residual_corrector: bool = True,
) -> List[Dict[str, Any]]:
    """
    End-to-end prediction pipeline with gentle, tunable shaping and strong debug.
    """
    logger.info("--- NFL Prediction Pipeline ---")
    t0 = time.time()

    FOCUS_IDS: Set[int] = set(focus.ids) if (focus and focus.ids) else set()

    # ------------------------------------------------------------------ #
    # Supabase + artifacts
    # ------------------------------------------------------------------ #
    sb = get_supabase_client()
    if not sb:
        logger.critical("No Supabase client. Abort.")
        return []

    artifacts = _load_core_artifacts(MODEL_DIR, debug=debug_mode)
    if not artifacts:
        logger.critical("Missing artifacts. Abort.")
        return []

    total_train_mean  = float(artifacts["total_train_mean"])
    margin_train_mean = float(artifacts["margin_train_mean"])
    feature_means: Dict[str, float] = artifacts["feature_means"]

    # ------------------------------------------------------------------ #
    # Fetch schedule (bounded by date window)
    # ------------------------------------------------------------------ #
    try:
        sched_resp = sb.table("nfl_game_schedule").select("*").execute()
        upcoming_df = pd.DataFrame(sched_resp.data or [])
    except Exception as e:
        logger.critical("Failed to fetch nfl_game_schedule: %s", e)
        return []

    upcoming_df = _ensure_season_and_ids(upcoming_df)

    # Hydrate missing team_ids
    try:
        team_map = _fetch_team_mapping(sb)
        if team_map:
            before_missing = int(
                ((~upcoming_df["home_team_id"].notna()) | (~upcoming_df["away_team_id"].notna())).sum()
            ) if ("home_team_id" in upcoming_df.columns and "away_team_id" in upcoming_df.columns) else -1

            upcoming_df = _hydrate_schedule_team_ids(upcoming_df, team_map)

            after_missing = int(
                ((~upcoming_df["home_team_id"].notna()) | (~upcoming_df["away_team_id"].notna())).sum()
            ) if ("home_team_id" in upcoming_df.columns and "away_team_id" in upcoming_df.columns) else -1

            logger.info("TEAM MAP hydration: missing IDs before=%s after=%s",
                        before_missing if before_missing >= 0 else "n/a",
                        after_missing if after_missing >= 0 else "n/a")
    except Exception as e:
        logger.debug("TEAM MAP hydration skipped: %s", e)

    if upcoming_df.empty:
        logger.info("No games found in nfl_game_schedule.")
        return []

    if debug_mode:
        n_total = len(upcoming_df)
        date_min = pd.to_datetime(upcoming_df.get("game_date"), errors="coerce").min()
        date_max = pd.to_datetime(upcoming_df.get("game_date"), errors="coerce").max()
        logger.debug("SCHEDULE pre-window: rows=%d | date_range=[%s, %s]", n_total, str(date_min)[:10], str(date_max)[:10])

    apply_window = (not all_games) and (bool(date_start) or bool(date_end) or (days_window is not None))

    if apply_window and "game_date" in upcoming_df.columns:
        dfw = upcoming_df.copy()
        dfw["game_date"] = pd.to_datetime(dfw["game_date"], errors="coerce")

        lower = None
        upper = None

        if date_start:
            try:
                lower = pd.to_datetime(date_start)
            except Exception:
                lower = None

        if date_end:
            try:
                upper = pd.to_datetime(date_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            except Exception:
                upper = None

        if (lower is None and upper is None) and (days_window is not None):
            today = pd.Timestamp(datetime.now(timezone.utc).date())
            lower = today
            upper = today + timedelta(days=int(days_window))

        if lower is not None:
            dfw = dfw.loc[dfw["game_date"] >= lower]
        if upper is not None:
            dfw = dfw.loc[dfw["game_date"] <= upper]

        if debug_mode:
            logger.debug(
                "SCHEDULE post-window: rows=%d | lower=%s upper=%s",
                len(dfw),
                lower.isoformat() if lower is not None else None,
                upper.isoformat() if upper is not None else None,
            )

        upcoming_df = dfw

    if upcoming_df.empty:
        logger.info("No games within the requested window (or after filtering). Use --all to include everything.")
        return []

    fetcher = NFLDataFetcher(sb)
    games_hist = fetcher.fetch_historical_games(historical_lookback)
    stats_hist = fetcher.fetch_historical_team_game_stats(historical_lookback)

    _log_df(upcoming_df, "upcoming_df", debug_mode)
    _log_df(games_hist,  "games_hist",  debug_mode)
    _log_df(stats_hist,  "stats_hist",  debug_mode)

    # ------------------------------------------------------------------ #
    # Feature pipeline
    # ------------------------------------------------------------------ #
    logger.info("Building features for %d games…", len(upcoming_df))
    feat_t = time.time()
    nfl_engine = NFLFeatureEngine(
        supabase_url=config.SUPABASE_URL,
        supabase_service_key=config.SUPABASE_SERVICE_KEY
    )

    # Pass through drive tuning if present in artifacts
    drive_kwargs = {}
    for _k in ("drive_window", "drive_reset_by_season", "drive_min_prior_games", "drive_soft_fail"):
        if _k in artifacts:
            drive_kwargs[_k] = artifacts[_k]

    features_df = nfl_engine.build_features(
        games_df=upcoming_df,
        historical_games_df=games_hist,
        historical_team_stats_df=stats_hist,
        debug=debug_mode,
        **drive_kwargs,
    )

    logger.info("Feature pipeline done in %.2fs", time.time() - feat_t)
    _log_df(features_df, "features_df", debug_mode)

    if features_df.empty:
        logger.error("Feature DF empty. Abort.")
        return []

    # Add train-time composites when available (prevents 0.0 fallbacks)
    features_df = add_total_composites_smart(features_df)
    if debug_mode:
        built_totals = [c for c in features_df.columns if c.startswith(("total_", "drive_total_"))]
        logger.debug("PRED[COMPOSITE] built totals (sample)=%s", built_totals[:24])

    # ------------------------------------------------------------------ #
    # H2H shrink (optional, week-aware k)
    # ------------------------------------------------------------------ #
    if not no_h2h_shrink:
        h2h_k_eff = _dynamic_h2h_k(upcoming_df, h2h_k)
        if debug_mode:
            try:
                wk_guess = _approx_season_week(upcoming_df["game_date"])
            except Exception:
                wk_guess = -1
            logger.debug("PRED[H2H] using k=%.1f (base=%.1f, week≈%s)", h2h_k_eff, h2h_k, wk_guess)
        features_df = _shrink_h2h_features(
            features_df,
            mu_total=total_train_mean,
            mu_margin=margin_train_mean,
            k=float(h2h_k_eff),
            debug=debug_mode,
        )
    else:
        if debug_mode:
            logger.debug("PRED[H2H] shrink bypassed (--no-h2h-shrink).")

    # ------------------------------------------------------------------ #
    # Advanced feature mean imputation (+ flags) using train means
    # ------------------------------------------------------------------ #
    if feature_means:
        features_df = _impute_advanced_features(features_df, feature_means, debug=debug_mode)
    else:
        if debug_mode:
            logger.debug("PRED[IMPUTE] No train feature_means provided; skipping advanced imputation.")

    # ------------------------------------------------------------------ #
    # Volatile diff clipping (quantile-driven if available)
    # ------------------------------------------------------------------ #
    default_bounds = {
        "rolling_points_for_avg_diff": (-12.0, 12.0),
        "rolling_points_against_avg_diff": (-12.0, 12.0),
        "rolling_point_differential_avg_diff": (-20.0, 20.0),
        "rolling_yards_per_play_avg_diff": (-1.5, 1.5),
        "rolling_turnover_differential_avg_diff": (-2.0, 2.0),
        "momentum_ewma_5_diff": (-25.0, 25.0),
    }
    # Prefer train quantiles if emitted at training time
    q_bounds = artifacts.get("feature_bounds", {}) or {}
    clip_bounds = {}
    for k, v in default_bounds.items():
        if k in q_bounds and isinstance(q_bounds[k], (list, tuple)) and len(q_bounds[k]) == 2:
            lo, hi = q_bounds[k]
            # be slightly permissive beyond the quantiles to avoid over-clipping
            pad_lo = float(lo) - 0.10 * abs(lo)
            pad_hi = float(hi) + 0.10 * abs(hi)
            clip_bounds[k] = (pad_lo, pad_hi)
        else:
            clip_bounds[k] = v

    features_df = _clip_feature_columns(features_df, clip_bounds, debug=debug_mode, focus_ids=FOCUS_IDS)

    # ------------------------------------------------------------------ #
    # Model setup + matrices
    # ------------------------------------------------------------------ #
    for target in ("margin", "total"):
        selected = artifacts.get(f"{target}_features", [])
        missing = [c for c in selected if c not in features_df.columns]
        logger.info("PRED[SETUP] %s selected=%d | missing-in-frame=%d", target.upper(), len(selected), len(missing))
        if debug_mode and missing:
            logger.debug("PRED[SETUP] %s missing detail: %s", target.upper(), missing[:40])

    margin_ensemble = NFLEnsemble(artifacts["margin_weights"], MODEL_DIR)
    total_ensemble  = NFLEnsemble(artifacts["total_weights"],  MODEL_DIR)
    margin_ensemble.load_models()
    total_ensemble.load_models()

    if "game_id" in features_df.columns:
        features_df = features_df.set_index("game_id", drop=False)

    X_margin = _build_matrix(features_df, artifacts["margin_features"], feature_means, label="MARGIN", debug=debug_mode)
    X_total  = _build_matrix(features_df, artifacts["total_features"],  feature_means, label="TOTAL",  debug=debug_mode)

    if debug_mode:
        for label, X in (("MARGIN", X_margin), ("TOTAL", X_total)):
            prof = _df_profile(X, f"X_{label}")
            ord_crc = _order_crc(X.columns)
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            # per-bucket counts
            counts = {}
            for c in X.columns:
                b = _bucket_of(c)
                counts[b] = counts.get(b, 0) + 1
            logger.debug("PRED[FRAME] %s rows=%d | null_top=%s | order_crc=%s | bucket_counts=%s",
                         label, prof["rows"], prof["null_top"], ord_crc, counts)
            if non_numeric:
                logger.debug("PRED[FRAME][WARN] %s non-numeric cols: %s", label, non_numeric)

    _check_features(X_margin, artifacts["margin_features"], "X_margin", debug_mode)
    _check_features(X_total,  artifacts["total_features"],  "X_total",  debug_mode)

    # ------------------------------------------------------------------ #
    # Week-adaptive attenuation (unless explicit alphas provided or --no-atten)
    # ------------------------------------------------------------------ #
    if not no_atten:
        wk = _approx_season_week(upcoming_df["game_date"]) if "game_date" in upcoming_df.columns else 18
        if atten_total_alpha is None or atten_margin_alpha is None:
            # defaults by week
            if wk <= 4:
                aT, aM = 0.80, 0.60
            elif wk <= 10:
                aT, aM = 0.90, 0.75
            else:
                aT, aM = 1.00, 0.85
            # override if user provided one side explicitly
            if atten_total_alpha is not None:
                aT = float(atten_total_alpha)
            if atten_margin_alpha is not None:
                aM = float(atten_margin_alpha)
        else:
            aT, aM = float(atten_total_alpha), float(atten_margin_alpha)

        # Apply on the subset columns (keep matrix shape)
        X_total  = _attenuate_total_diffs(X_total,  alpha=aT, debug=debug_mode)
        X_margin = _attenuate_margin_diffs(X_margin, alpha=aM, debug=debug_mode)
    else:
        if debug_mode:
            logger.debug("PRED[ATTN] pre-model attenuation disabled (--no-atten).")

    # ------------------------------------------------------------------ #
    # Vectorized predict (ENSEMBLE)
    # ------------------------------------------------------------------ #
    pred_t = time.time()
    margin_preds = margin_ensemble.predict(X_margin)
    total_preds  = total_ensemble.predict(X_total)

    if isinstance(margin_preds, np.ndarray):
        margin_preds = pd.Series(margin_preds, index=features_df.index, name="pred_margin")
    if isinstance(total_preds, np.ndarray):
        total_preds = pd.Series(total_preds, index=features_df.index, name="pred_total")

    if debug_mode:
        logger.debug("PRED[ENS] MARGIN | weights=%s", artifacts.get("margin_weights", {}))
        logger.debug("PRED[ENS] TOTAL  | weights=%s", artifacts.get("total_weights", {}))
        logger.debug("Predict time: %.3fs", time.time() - pred_t)

    # ------------------------------------------------------------------ #
    # Week-aware priors for mean calibration
    # ------------------------------------------------------------------ #
    # Recent environment (current season preferred)
    recent_total_mean = None
    try:
        gh = games_hist.copy()
        gh["game_date"] = pd.to_datetime(gh["game_date"], errors="coerce")
        # scope to last ~120d or current season
        if "season" in gh.columns and "season" in upcoming_df.columns and upcoming_df["season"].notna().any():
            cur_season = int(pd.Series(upcoming_df["season"].dropna().astype(int)).mode().iloc[0])
            gh = gh.loc[gh["season"] == cur_season]
        else:
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=120)
            gh = gh.loc[gh["game_date"] >= cutoff]
        gh = gh.dropna(subset=["home_score", "away_score"])
        if not gh.empty:
            recent_total_mean = float((pd.to_numeric(gh["home_score"], errors="coerce") +
                                       pd.to_numeric(gh["away_score"], errors="coerce")).mean())
    except Exception:
        recent_total_mean = None

    # Blend rules
    wk_guess = _approx_season_week(upcoming_df["game_date"]) if "game_date" in upcoming_df.columns else 18
    if recent_total_mean is not None:
        if wk_guess <= 4:
            target_total_mean = 0.30 * recent_total_mean + 0.70 * total_train_mean
        else:
            target_total_mean = 0.60 * recent_total_mean + 0.40 * total_train_mean
    else:
        target_total_mean = total_train_mean

    # MARGIN target (0). Optional small HFA early if model didn't learn it (artifact flag)
    hfa_learned = bool(artifacts.get("hfa_learned", False))
    target_margin_mean = 0.0
    if (wk_guess <= 4) and (not hfa_learned):
        # tiny nudge toward historical home-field if provided
        target_margin_mean = float(artifacts.get("hfa_baseline", 0.4))

    total_preds  = _soft_mean_calibration(
        total_preds,  target_mean=target_total_mean,  lam=float(mean_cal_lambda_total),
        min_trigger=float(mean_cal_min_trigger_total), label="TOTAL",  debug=debug_mode
    )
    margin_preds = _soft_mean_calibration(
        margin_preds, target_mean=target_margin_mean, lam=float(mean_cal_lambda_margin),
        min_trigger=float(mean_cal_min_trigger_margin), label="MARGIN", debug=debug_mode
    )

    # Optional market blend (tiny anchor)
    def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    if market_w and market_w > 0.0:
        mt_col = _find_col(upcoming_df, ["market_total", "vegas_total", "consensus_total"])
        ms_col = _find_col(upcoming_df, ["market_spread", "vegas_spread", "consensus_spread"])
        if mt_col or ms_col:
            if mt_col:
                mt = pd.to_numeric(upcoming_df.set_index("game_id")[mt_col], errors="coerce")
                total_preds = (1.0 - market_w) * total_preds + market_w * mt.reindex(total_preds.index).fillna(total_preds)
            if ms_col:
                # convention: spread is home favorite negative; margin = home - away
                ms = pd.to_numeric(upcoming_df.set_index("game_id")[ms_col], errors="coerce")
                implied_margin = -ms  # home - away
                margin_preds = (1.0 - market_w) * margin_preds + market_w * implied_margin.reindex(margin_preds.index).fillna(margin_preds)
            logger.info("Market blend applied w=%.2f to %d rows (mt_col=%s, ms_col=%s)", market_w, len(total_preds), mt_col, ms_col)
        elif debug_mode:
            logger.debug("Market blend requested but no market columns found; skipped.")

    # Optional TOTAL clip (after calibration)
    if (total_clip_min is not None) or (total_clip_max is not None):
        lo = float(total_clip_min) if total_clip_min is not None else float("-inf")
        hi = float(total_clip_max) if total_clip_max is not None else float("+inf")
        total_preds = total_preds.clip(lower=lo, upper=hi)
        if debug_mode:
            logger.debug("PRED[CAL] TOTAL clip applied: [%.1f, %.1f]", lo, hi)

    # ------------------------------------------------------------------ #
    # Contextual Feasibility: total ≥ |margin| + buffer
    # ------------------------------------------------------------------ #
    pair = pd.DataFrame({"total": total_preds.astype(float), "margin": margin_preds.astype(float)}, index=total_preds.index)

    # base buffer
    base = 0.35 * pair["margin"].abs() + 3.0

    # context bumps
    wk_bump = 0.5 if wk_guess <= 4 else 0.0

    # dome/indoor flag (search a few plausible columns)
    def _bool_from_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        for c in cols:
            if c in df.columns:
                try:
                    v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    out = out.where(out != 0.0, v)  # first found
                except Exception:
                    pass
        return out

    # evaluate on *features_df* (aligned index)
    dome_series = _bool_from_cols(features_df, [
        "situational_is_indoor", "map_is_dome", "stadium_is_dome", "stadium_is_indoor"
    ])
    cold_series = pd.to_numeric(features_df.get("map_fore_temp_f", pd.Series(np.nan, index=features_df.index)), errors="coerce")
    is_very_cold = (cold_series <= 25).fillna(0).astype(float)

    context_bump = wk_bump + ( -0.5 * (dome_series >= 1).astype(float) ) + ( 0.5 * is_very_cold )

    dyn_buffer = (base + context_bump).clip(lower=2.5, upper=float(feas_buffer_max))
    need_raise = pair["total"] < (pair["margin"].abs() + dyn_buffer)
    if need_raise.any():
        min_allowed = pair["margin"].abs() + dyn_buffer
        total_preds = pair["total"].where(~need_raise, min_allowed).astype(float)
        if debug_mode:
            logger.debug("PRED[FEAS] total raised on %d rows to satisfy total ≥ |margin| + buffer", int(need_raise.sum()))
    logger.info("Feasibility raises applied to %d of %d rows", int(need_raise.sum()), len(pair))

    # Hard cap on margin (±)
    if float(margin_cap) > 0:
        over_cap = margin_preds.abs() > float(margin_cap)
        if over_cap.any():
            margin_preds = margin_preds.clip(lower=-float(margin_cap), upper=float(margin_cap))
            if debug_mode:
                logger.debug("PRED[CAP] MARGIN hard-cap applied at ±%.1f on %d rows", float(margin_cap), int(over_cap.sum()))

    # ------------------------------------------------------------------ #
    # Derive scores and tiny floor bump after feasibility
    # ------------------------------------------------------------------ #
    scores_df = derive_scores_from_predictions(margin_preds, total_preds)

    if "game_id" in scores_df.columns:
        scores_df = scores_df.set_index("game_id", drop=False)
    for c in ("predicted_home_score", "predicted_away_score"):
        if c in scores_df.columns:
            scores_df[c] = scores_df[c].astype(float).clip(lower=0.0)

    # If feasibility just activated (total very close to |margin|+buffer), nudge +0.2 both sides
    with np.errstate(all="ignore"):
        tot = scores_df["predicted_home_score"] + scores_df["predicted_away_score"]
        mar = scores_df["predicted_home_score"] - scores_df["predicted_away_score"]
    near_edge = (tot - mar.abs()) <= 0.05
    if near_edge.any():
        scores_df.loc[near_edge, "predicted_home_score"] += 0.2
        scores_df.loc[near_edge, "predicted_away_score"] += 0.2

    _log_df(scores_df, "scores_df", debug_mode)

    final_df = upcoming_df.set_index("game_id", drop=False)
    if "game_id" in scores_df.columns:
        scores_df = scores_df.set_index("game_id", drop=False)

    final_df.index  = final_df.index.astype(str)
    scores_df.index = scores_df.index.astype(str)

    final_df.loc[scores_df.index, "predicted_home_score"] = scores_df["predicted_home_score"]
    final_df.loc[scores_df.index, "predicted_away_score"] = scores_df["predicted_away_score"]

    final_df["pred_total"]  = (final_df["predicted_home_score"] + final_df["predicted_away_score"]).astype(float)
    final_df["pred_margin"] = (final_df["predicted_home_score"] - final_df["predicted_away_score"]).astype(float)

    # Pretty-print (sample) – tolerate missing IDs gracefully
    def _fmt(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "NA"

    for _, row in final_df.loc[scores_df.index].iterrows():
        date_s = str(row.get("game_date"))[:10]
        home_id = _safe_int(row.get("home_team_id"))
        away_id = _safe_int(row.get("away_team_id"))
        home = _fmt(row.get("predicted_home_score"))
        away = _fmt(row.get("predicted_away_score"))
        tot  = _fmt(row.get("pred_total"))
        mar  = _fmt(row.get("pred_margin"))
        if home_id is None or away_id is None:
            if debug_mode:
                logger.debug("PPRINT skip: missing team_id(s) for row on %s (home_id=%s, away_id=%s)", date_s, home_id, away_id)
            continue
        print(f"{date_s}  away_id {away_id} @ home_id {home_id}   {away} - {home}   (T={tot}, M={mar})")

    if debug_mode:
        n_assigned = final_df["predicted_home_score"].notna().sum()
        logger.debug("PRED[JOIN] assigned predictions for %d games", n_assigned)

    # ------------------------------------------------------------------ #
    # Residual bias corrector (recent games only; guarded; capped)
    # ------------------------------------------------------------------ #
    if residual_corrector and _SK_OK:
        try:
            # Build recent historical features (~last 8 weeks / 60 days)
            hist_cut = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=60)
            gh = games_hist.copy()
            gh["game_date"] = pd.to_datetime(gh["game_date"], errors="coerce")
            gh = gh.loc[gh["game_date"] < upcoming_df["game_date"].min()]  # strictly past
            gh_recent = gh.loc[gh["game_date"] >= hist_cut]
            if not gh_recent.empty:
                feats_recent = nfl_engine.build_features(
                    games_df=gh_recent,
                    historical_games_df=games_hist,
                    historical_team_stats_df=stats_hist,
                    debug=False,
                    **drive_kwargs,
                )
                feats_recent = add_total_composites_smart(feats_recent)
                feats_recent = feats_recent.dropna(subset=["home_score", "away_score"])
                if not feats_recent.empty:
                    feats_recent = feats_recent.set_index("game_id", drop=False)
                    Xm = _build_matrix(feats_recent, artifacts["margin_features"], feature_means, label="HIST_MARGIN", debug=False)
                    Xt = _build_matrix(feats_recent, artifacts["total_features"],  feature_means, label="HIST_TOTAL",  debug=False)
                    # Use same attenuation logic as current week (but no debug)
                    if not no_atten:
                        if atten_total_alpha is None or atten_margin_alpha is None:
                            if wk_guess <= 4:
                                aT, aM = 0.80, 0.60
                            elif wk_guess <= 10:
                                aT, aM = 0.90, 0.75
                            else:
                                aT, aM = 1.00, 0.85
                            if atten_total_alpha is not None:
                                aT = float(atten_total_alpha)
                            if atten_margin_alpha is not None:
                                aM = float(atten_margin_alpha)
                        else:
                            aT, aM = float(atten_total_alpha), float(atten_margin_alpha)
                        Xt = _attenuate_total_diffs(Xt, aT, False)
                        Xm = _attenuate_margin_diffs(Xm, aM, False)

                    hist_margin_pred = margin_ensemble.predict(Xm)
                    hist_total_pred  = total_ensemble.predict(Xt)

                    if isinstance(hist_margin_pred, np.ndarray):
                        hist_margin_pred = pd.Series(hist_margin_pred, index=feats_recent.index)
                    if isinstance(hist_total_pred, np.ndarray):
                        hist_total_pred = pd.Series(hist_total_pred, index=feats_recent.index)

                    actual_total  = pd.to_numeric(feats_recent["home_score"], errors="coerce") + pd.to_numeric(feats_recent["away_score"], errors="coerce")
                    actual_margin = pd.to_numeric(feats_recent["home_score"], errors="coerce") - pd.to_numeric(feats_recent["away_score"], errors="coerce")

                    res_total  = (actual_total  - hist_total_pred).astype(float)
                    res_margin = (actual_margin - hist_margin_pred).astype(float)

                    # tiny, regularized model on simple covariates
                    covar_cols = [c for c in [
                        "map_fore_temp_f",
                        "situational_is_indoor", "stadium_is_indoor", "map_is_dome", "stadium_is_dome",
                        "home_is_on_short_week", "away_is_on_short_week",
                        "home_rest_days", "away_rest_days",
                        "week",
                    ] if c in feats_recent.columns]

                    if covar_cols:
                        Z = feats_recent[covar_cols].copy()
                        for c in Z.columns:
                            Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(0.0)

                        def _fit_and_adjust(y_res: pd.Series, pred_series: pd.Series, cap: float) -> pd.Series:
                            # time-series friendly CV alphas
                            model = Pipeline([
                                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                                ("ridge", RidgeCV(alphas=(1.5, 3.0, 5.0, 7.5, 10.0)))
                            ])
                            model.fit(Z.values, y_res.values)
                            adj = model.predict(Z.values[:0])  # no need; just confirm fit
                            # Now predict adjustments for UPCOMING rows (build Z2 from features_df if possible)
                            Z2 = features_df.reindex(columns=covar_cols).copy()
                            for c in Z2.columns:
                                Z2[c] = pd.to_numeric(Z2[c], errors="coerce").fillna(0.0)
                            pred_adj = model.predict(Z2.values)
                            pred_adj = np.clip(pred_adj, -cap, cap)
                            return pred_series + pd.Series(pred_adj, index=pred_series.index)

                        # apply (caps keep it tiny)
                        total_preds  = _fit_and_adjust(res_total,  total_preds,  cap=1.5)
                        margin_preds = _fit_and_adjust(res_margin, margin_preds, cap=0.7)
                        logger.info("Residual corrector applied (recent=%d games, covariates=%d)", len(Z), len(covar_cols))
        except Exception as e:
            if debug_mode:
                logger.debug("Residual corrector skipped due to error: %s", e)
    elif residual_corrector and not _SK_OK:
        logger.info("Residual corrector requested but sklearn is unavailable; skipping.")

    # ------------------------------------------------------------------ #
    # Optional CSV dumps for focused games (post-ensemble & cal)
    # ------------------------------------------------------------------ #
    if debug_mode and FOCUS_IDS:
        base = dump_dir or (PROJECT_ROOT / "debug" / "pred_inspect")
        for gid in [g for g in FOCUS_IDS if str(g) in final_df.index]:
            row = final_df.loc[str(gid)]
            gd = row.get("game_date")
            date_str = gd.date().isoformat() if isinstance(gd, (pd.Timestamp, datetime)) else str(gd)[:10]
            subdir = Path(base) / date_str
            subdir.mkdir(parents=True, exist_ok=True)

            csv_row: Dict[str, Any] = {
                "game_id": int(gid),
                "date": date_str,
                "away_team_id": _safe_int(row["away_team_id"]),
                "home_team_id": _safe_int(row["home_team_id"]),
                "away_team_name": row.get("away_team_name"),
                "home_team_name": row.get("home_team_name"),
                "predicted_away_score": float(row.get("predicted_away_score")) if "predicted_away_score" in row else None,
                "predicted_home_score": float(row.get("predicted_home_score")) if "predicted_home_score" in row else None,
                "weights_total_json": json.dumps(artifacts.get("total_weights", {})),
                "weights_margin_json": json.dumps(artifacts.get("margin_weights", {})),
            }

            # Record order CRCs used at predict time
            csv_row["order_crc_total"]  = _order_crc(artifacts["total_features"])
            csv_row["order_crc_margin"] = _order_crc(artifacts["margin_features"])

            # Selected feature vectors (union) — leave missing as blank (None)
            union_cols = list(dict.fromkeys(artifacts["total_features"] + artifacts["margin_features"]))
            for c in union_cols:
                if c in features_df.columns and pd.notna(features_df.at[int(gid), c]):
                    try:
                        csv_row[c] = float(features_df.at[int(gid), c])
                    except Exception:
                        csv_row[c] = None
                else:
                    csv_row[c] = None

            impute_cols = [c for c in features_df.columns if "imput" in c.lower()]
            for c in impute_cols:
                key = f"flag__{c}"
                try:
                    v = features_df.at[int(gid), c]
                    csv_row[key] = float(v) if pd.notna(v) else None
                except Exception:
                    csv_row[key] = None

            out_path = subdir / f"{date_str}_{row['away_team_id']}@{row['home_team_id']}_{gid}.csv"
            pd.DataFrame([csv_row]).to_csv(out_path, index=False)
            logger.debug("PRED[DUMP] wrote %s", out_path)

    # ------------------------------------------------------------------ #
    # Build payload
    # ------------------------------------------------------------------ #
    payload: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for gid, row in final_df.iterrows():
        if pd.isna(row.get("predicted_home_score")):
            if debug_mode:
                logger.debug("Skipping game_id=%s due to NaN prediction", gid)
            continue

        home_id = _safe_int(row.get("home_team_id"))
        away_id = _safe_int(row.get("away_team_id"))
        if home_id is None or away_id is None:
            if debug_mode:
                logger.debug("Skipping payload for game_id=%s due to missing team_id(s): home_id=%s away_id=%s", gid, home_id, away_id)
            continue

        payload.append(
            {
                "game_id":              int(gid),
                "game_date":            row["game_date"].isoformat() if hasattr(row["game_date"], "isoformat") else str(row["game_date"]),
                "home_team_id":         home_id,
                "away_team_id":         away_id,
                "home_team_name":       row.get("home_team_name"),
                "away_team_name":       row.get("away_team_name"),
                "predicted_home_score": round(float(row["predicted_home_score"]), 2),
                "predicted_away_score": round(float(row["predicted_away_score"]), 2),
                "prediction_utc":       now_iso,
            }
        )

    logger.info("Generated %d predictions in %.2fs", len(payload), time.time() - t0)
    return payload

# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate and Upsert NFL Score Predictions")

    # Windowing
    parser.add_argument("--date-start", type=str, default=None, help="Start date inclusive (YYYY-MM-DD).")
    parser.add_argument("--date-end",   type=str, default=None, help="End date inclusive (YYYY-MM-DD).")
    parser.add_argument("--days",       type=int, default=None, help="Relative window [today, today+days] if no date range is given.")
    parser.add_argument("--all", dest="all_games", action="store_true",
                    help="Ignore date windowing and predict for ALL rows in nfl_game_schedule.")

    # History
    parser.add_argument("--lookback", type=int, default=1825, help="Historical days for features.")

    # Persistence
    parser.add_argument("--no-upsert", action="store_true", help="Skip DB upsert.")

    # Debug / focus
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--focus-date", type=str, default=None, help="Focus on a specific date (YYYY-MM-DD).")
    parser.add_argument("--focus-games", type=str, default=None, help="Comma-separated away@home pairs, e.g., 25@26,21@28.")
    parser.add_argument("--focus-ids", type=str, default=None, help="Comma-separated game_id list.")
    parser.add_argument("--dump-dir", type=str, default=None, help="Directory to write per-focused-game CSVs.")

    # Pre-model attenuation (None → week-adaptive defaults)
    parser.add_argument("--no-atten", action="store_true", help="Disable pre-model diff attenuation.")
    parser.add_argument("--atten-total-alpha", type=float, default=None, help="Total diff attenuation factor (0..1]. Leave empty for week-adaptive.")
    parser.add_argument("--atten-margin-alpha", type=float, default=None, help="Margin diff attenuation factor (0..1]. Leave empty for week-adaptive.")

    # H2H shrink
    parser.add_argument("--no-h2h-shrink", action="store_true", help="Disable H2H shrink toward priors.")
    parser.add_argument("--h2h-k", type=float, default=6.0, help="Games to “trust” H2H fully (larger=weaker shrink).")

    # Mean calibration (soft, week-aware targets)
    parser.add_argument("--mean-calibration-lambda-total", type=float, default=0.45, help="Lambda for TOTAL soft mean calibration.")
    parser.add_argument("--mean-calibration-lambda-margin", type=float, default=0.50, help="Lambda for MARGIN soft mean calibration.")
    parser.add_argument("--mean-cal-min-trigger-total", type=float, default=0.25, help="Min |Δ| to trigger TOTAL calibration.")
    parser.add_argument("--mean-cal-min-trigger-margin", type=float, default=0.15, help="Min |Δ| to trigger MARGIN calibration.")

    # Feasibility + caps
    parser.add_argument("--feas-buffer-max", type=float, default=6.0, help="Max dynamic feasibility buffer.")
    parser.add_argument("--margin-cap", type=float, default=24.0, help="Hard cap for absolute margin.")

    # Optional TOTAL clip band
    parser.add_argument("--total-clip-min", type=float, default=None, help="Clip TOTAL predictions to lower bound (post-calibration).")
    parser.add_argument("--total-clip-max", type=float, default=None, help="Clip TOTAL predictions to upper bound (post-calibration).")

    # Optional market blend
    parser.add_argument("--market-w", type=float, default=0.0, help="Blend weight toward market totals/spreads if columns exist (0..0.25 recommended).")

    # Residual corrector
    rc_group = parser.add_mutually_exclusive_group()
    rc_group.add_argument("--residual-corrector", dest="residual_corrector", action="store_true", help="Enable residual bias corrector (default).")
    rc_group.add_argument("--no-residual-corrector", dest="residual_corrector", action="store_false", help="Disable residual bias corrector.")
    parser.set_defaults(residual_corrector=True)

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    focus = Focus(
        date_str=_normalize_date_str(args.focus_date),
        pairs=_parse_focus_games(args.focus_games),
        ids=_parse_focus_ids(args.focus_ids),
    )

    preds = generate_predictions(
        date_start=args.date_start,
        date_end=args.date_end,
        days_window=args.days,
        all_games=args.all_games,

        historical_lookback=args.lookback,
        debug_mode=args.debug,
        focus=focus,
        dump_dir=Path(args.dump_dir) if args.dump_dir else None,

        no_atten=args.no_atten,
        atten_total_alpha=args.atten_total_alpha,
        atten_margin_alpha=args.atten_margin_alpha,

        no_h2h_shrink=args.no_h2h_shrink,
        h2h_k=args.h2h_k,

        mean_cal_lambda_total=args.mean_calibration_lambda_total,
        mean_cal_lambda_margin=args.mean_calibration_lambda_margin,
        mean_cal_min_trigger_total=args.mean_cal_min_trigger_total,
        mean_cal_min_trigger_margin=args.mean_cal_min_trigger_margin,

        feas_buffer_max=args.feas_buffer_max,
        margin_cap=args.margin_cap,

        total_clip_min=args.total_clip_min,
        total_clip_max=args.total_clip_max,

        market_w=args.market_w,
        residual_corrector=args.residual_corrector,
    )

    if not preds:
        logger.info("No predictions produced.")
        sys.exit(0)

    display_prediction_summary(preds)

    if not args.no_upsert:
        sb = get_supabase_client()
        if sb:
            upsert_score_predictions(preds, sb, debug=args.debug)
        else:
            logger.error("Supabase unavailable; cannot upsert.")
    else:
        logger.info("--no-upsert specified. Skipping upsert.")

if __name__ == "__main__":
    main()
