# backend/nba_score_prediction/prediction.py

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import pytz
import pandas as pd
import numpy as np
import math
import json
import re
from typing import List, Dict, Optional, Any, Tuple

import logging

# -----------------------------------------------------------------------------
# Logging: kill WARNING/INFO in CI (or when LOG_LEVEL_OVERRIDE=ERROR)
if os.getenv("CI") or os.getenv("LOG_LEVEL_OVERRIDE", "").upper() == "ERROR":
    logging.disable(logging.ERROR - 1)      # == disable < ERROR
# -----------------------------------------------------------------------------

# Nothing else to configure – ERROR and CRITICAL will still show
logger = logging.getLogger(__name__)               # ← overwrite handlers added earlier

# ensure project root is on PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# also add backend/ as a top-level package
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Third‐party clients & config ---
import config
from supabase import create_client
from caching.supabase_client import supabase as supabase_client_instance
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY


# --- Constants & paths ---
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180
DEFAULT_UPCOMING_DAYS_WINDOW = 3

ENSEMBLE_WEIGHTS_FILENAME = "ensemble_weights.json"
FALLBACK_ENSEMBLE_WEIGHTS: Dict[str, float] = {"ridge": 0.5, "svr": 0.5}

REQUIRED_HISTORICAL_COLS = [
    "game_id", "game_date", "home_team", "away_team", "home_score", "away_score",
    "home_q1", "home_q2", "home_q3", "home_q4", "home_ot",
    "away_q1", "away_q2", "away_q3", "away_q4", "away_ot",
    "home_fg_made", "home_fg_attempted", "away_fg_made", "away_fg_attempted",
    "home_3pm", "home_3pa", "away_3pm", "away_3pa",
    "home_ft_made", "home_ft_attempted", "away_ft_made", "away_ft_attempted",
    "home_off_reb", "home_def_reb", "home_total_reb",
    "away_off_reb", "away_def_reb", "away_total_reb",
    "home_turnovers", "away_turnovers",
    "home_assists", "home_steals", "home_blocks", "home_fouls",
    "away_assists", "away_steals", "away_blocks", "away_fouls"
]
REQUIRED_TEAM_STATS_COLS = [
    "team_name", "season", "wins_all_percentage", "points_for_avg_all",
    "points_against_avg_all", "current_form"
]
UPCOMING_GAMES_COLS = ["game_id", "scheduled_time", "home_team", "away_team"]

# --- Project module imports (left as-is) ---
try:
    from nba_features.engine import run_feature_pipeline
    from nba_score_prediction.models import RidgeScorePredictor, SVRScorePredictor
    from nba_score_prediction.simulation import PredictionUncertaintyEstimator
    import nba_score_prediction.utils as utils
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logger.error(f"Critical imports failed: {e}", exc_info=True)
    PROJECT_MODULES_IMPORTED = False

# --- Supabase helper ---
def get_supabase_client() -> Optional[Any]:
    """Prefer the cached service-role client, else fall back on anon."""
    if supabase_client_instance:
        return supabase_client_instance
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None

# --- Data Loading Functions ---
def load_recent_historical_data(supabase_client: Any, days_lookback: int) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
    logger.info(f"Loading historical game data from {start_date} onwards...")
    
    select_cols_str = ", ".join(REQUIRED_HISTORICAL_COLS)
    all_historical_data = []
    page_size = 1000
    start_index = 0
    has_more = True

    try:
        # --- Pagination loop ---
        while has_more:
            resp = (
                supabase_client
                .table("nba_historical_game_stats")
                .select(select_cols_str)
                .gte("game_date", start_date)
                .order("game_date", desc=False)
                .range(start_index, start_index + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            all_historical_data.extend(batch)
            if len(batch) < page_size:
                has_more = False
            else:
                start_index += page_size

        # --- No data? return empty DF ---
        if not all_historical_data:
            logger.warning(f"No historical game data found since {start_date}.")
            return pd.DataFrame()

        # --- Build & clean DataFrame ---
        df = pd.DataFrame(all_historical_data)
        df["game_date"] = (
            pd.to_datetime(df["game_date"], errors="coerce")
              .dt.tz_localize(None)
        )
        df = df.dropna(subset=["game_date"])

        numeric_cols = [
            c for c in REQUIRED_HISTORICAL_COLS 
            if c not in ("game_id", "game_date", "home_team", "away_team")
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        for col in ("game_id", "home_team", "away_team"):
            df[col] = df[col].astype(str).fillna("")

        # --- Sort & reset index before returning ---
        return df.sort_values("game_date").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error loading historical data: {e}", exc_info=True)
        return pd.DataFrame()

def load_team_stats_data(supabase_client: Any) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    logger.info("Loading team stats data from 'nba_historical_team_stats'...")
    select_cols_str = ", ".join(REQUIRED_TEAM_STATS_COLS)
    all_stats = []
    page_size = 1000
    start_idx = 0
    try:
        while True:
            resp = (
                supabase_client
                .table("nba_historical_team_stats")
                .select(select_cols_str)
                .order("season", desc=False)
                .range(start_idx, start_idx + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            if not batch:
                break
            all_stats.extend(batch)
            if len(batch) < page_size:
                break
            start_idx += page_size

        if not all_stats:
            logger.warning("No team stats data found.")
            return pd.DataFrame()

        df = pd.DataFrame(all_stats)
        numeric_cols = [c for c in REQUIRED_TEAM_STATS_COLS if c not in ("team_name","season","current_form")]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
        for col in ("team_name","season","current_form"):
            df[col] = df[col].astype(str).fillna("")
        return df
    except Exception as e:
        logger.error(f"Error loading team stats: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client: Any, days_window: int) -> pd.DataFrame:
    # …
    # Compute UTC start/end at midnight UTC
    now_utc   = datetime.now(pytz.utc)
    start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end_utc   = start_utc + timedelta(days=days_window)

    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end_utc.strftime(  "%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Fetching upcoming games between {start_str} and {end_str} (UTC)...")

    try:
        resp = (
            supabase_client
            .table("nba_game_schedule")
            .select(", ".join(UPCOMING_GAMES_COLS))
            .gte("scheduled_time", start_str)
            .lt( "scheduled_time", end_str)
            .order("scheduled_time", desc=False)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            logger.warning("No upcoming games found.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["scheduled_time_utc"] = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["scheduled_time_utc"])
        df["game_time_pt"] = df["scheduled_time_utc"].dt.tz_convert(PACIFIC_TZ)
        df["game_date"] = pd.to_datetime(df["game_time_pt"].dt.date)
        for c in ("game_id","home_team","away_team"):
            df[c] = df[c].astype(str)
        return df.sort_values("game_time_pt").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}", exc_info=True)
        return pd.DataFrame()

# --- Model Loading ---
# --- Model Loading -----------------------------------------------------------
def load_trained_models(
    model_dir: Path = MODELS_DIR,
    load_feature_list: bool = False,   # ⬅ default OFF for now
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """
    Load any saved `*_score_predictor.joblib` models from `model_dir`.
    If `load_feature_list` is True and a clean selected_features.json is present,
    return it; otherwise return None.
    """
    models: Dict[str, Any] = {}
    for name, Cls in [("svr", SVRScorePredictor), ("ridge", RidgeScorePredictor)]:
        try:
            pred = Cls(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            pred.load_model()
            models[name] = pred
            logger.info("Loaded %s predictor.", name)
        except Exception as e:
            logger.error("Could not load '%s' model: %s", name, e, exc_info=True)

    feature_list: Optional[List[str]] = None
    if load_feature_list:
        fp = model_dir / "selected_features.json"
        if fp.is_file():
            try:
                with fp.open() as f:
                    feature_list = json.load(f)
            except Exception as e:
                logger.warning("Ignoring unreadable selected_features.json: %s", e)

    if not models:
        raise RuntimeError(f"No models found in {model_dir}")

    return models, feature_list

# --- Betting Odds Parsing Functions ---
def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    if not line_str:
        return None, None
    match = re.search(
        r"([\+\-]?\d+(?:\.\d+)?)\s*(?:\(?\s*(?:[ou])?\s*([\+\-]\d+)\s*\)?)?",
        str(line_str).strip()
    )
    if match:
        try:
            line = float(match.group(1))
        except (ValueError, TypeError):
            line = None
        try:
            odds = int(match.group(2)) if match.group(2) else -110
        except (ValueError, TypeError):
            odds = -110
        return line, odds
    return None, None

def parse_moneyline_str(ml_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[int], Optional[int]]:
    home_ml, away_ml = None, None
    if not ml_str or not isinstance(ml_str, str):
        return home_ml, away_ml
    try:
        mls = re.findall(r"([\+\-]\d{3,})", ml_str)
        if len(mls) == 2:
            ml1, ml2 = int(mls[0]), int(mls[1])
            if ml1 < 0 <= ml2:
                home_ml, away_ml = ml1, ml2
            elif ml2 < 0 <= ml1:
                home_ml, away_ml = ml2, ml1
            else:
                home_ml, away_ml = ml1, ml2
        elif len(mls) == 1:
            logger.warning(f"Only one moneyline found in '{ml_str}'.")
        else:
            logger.warning(f"Could not find two moneylines in '{ml_str}'.")
    except Exception as e:
        logger.warning(f"Could not parse moneyline '{ml_str}': {e}")
    return home_ml, away_ml

def parse_spread_str(spread_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    home_line, away_line = None, None
    if not spread_str or not isinstance(spread_str, str):
        return home_line, away_line
    try:
        nums = re.findall(r"([\+\-]?\d+(?:\.\d+)?)", spread_str)
        if len(nums) == 2:
            v1, v2 = float(nums[0]), float(nums[1])
            if abs(v1 + v2) < 0.1:
                home_line = v1 if v1 < 0 else v2
                away_line = -home_line
            else:
                home_line = v1
                away_line = -v1
                logger.warning(f"Spread '{spread_str}' doesn’t sum to zero; assuming {v1} is home.")
        elif len(nums) == 1:
            home_line = float(nums[0])
            away_line = -home_line
        else:
            logger.debug(f"Could not parse spread from '{spread_str}'.")
    except Exception as e:
        logger.warning(f"Could not parse spread '{spread_str}': {e}")
    return home_line, away_line

def parse_total_str(total_str: Optional[str]) -> Optional[float]:
    if not total_str or not isinstance(total_str, str):
        return None
    try:
        nums = re.findall(r"(\d{3}(?:\.\d+)?)", total_str)
        if nums:
            return float(nums[0])
        nums = re.findall(r"(\d+(?:\.\d+)?)", total_str)
        if nums:
            return float(nums[0])
    except Exception as e:
        logger.warning(f"Could not parse total '{total_str}': {e}")
    return None

def fetch_and_parse_betting_odds(
    supabase_client: Any,
    game_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    if not supabase_client or not game_ids:
        logger.warning("Skipping odds fetch due to missing client or game IDs.")
        return {}

    def make_moneyline_str(r):
        ml = r.get("moneyline") or {}
        h = ml.get(r["home_team"])
        a = ml.get(r["away_team"])
        def price(x):
            if isinstance(x, dict):
                return x.get("price")
            elif isinstance(x, (int, float)):
                return int(x)
        if h is None or a is None:
            return None
        return f"{r['home_team']} {price(h):+d} / {r['away_team']} {price(a):+d}"

    def make_spread_str(r):
        sp = r.get("spread") or {}
        h = sp.get(r["home_team"])
        a = sp.get(r["away_team"])
        def point(x):
            if isinstance(x, dict):
                return x.get("point")
            elif isinstance(x, (int, float)):
                return float(x)
        if h is None or a is None:
            return None
        return f"{r['home_team']} {point(h):+.1f} / {r['away_team']} {point(a):+.1f}"

    def make_total_str(r):
        tot = r.get("total") or {}
        over = tot.get("Over", {}).get("point")
        if over is None:
            return None
        return f"Over {over:.1f} / Under {over:.1f}"

    odds_dict: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50
    for i in range(0, len(game_ids), chunk_size):
        chunk = game_ids[i : i + chunk_size]
        try:
            resp = (
                supabase_client
                .table("nba_game_schedule")
                .select("game_id, home_team, away_team, moneyline, spread, total")
                .in_("game_id", chunk)
                .execute()
            )
            rows = resp.data or []
            df = pd.DataFrame(rows)
            for col in ("moneyline","spread","total"):
                df[col] = df[col].apply(lambda x: x or {})

            for _, r in df.iterrows():
                gid = str(r["game_id"])
                ml_str  = make_moneyline_str(r)
                sp_str  = make_spread_str(r)
                tot_str = make_total_str(r)

                # write back cleaned strings
                supabase_client.table("nba_game_schedule")\
                    .update({
                        "moneyline_clean": ml_str,
                        "spread_clean":   sp_str,
                        "total_clean":    tot_str
                    })\
                    .eq("game_id", int(gid))\
                    .execute()

                home_ml, away_ml = parse_moneyline_str(ml_str, r["home_team"], r["away_team"])
                home_sp, _       = parse_spread_str(sp_str, r["home_team"], r["away_team"])
                total_line       = parse_total_str(tot_str)
                default_odds     = -110

                entry = {
                    "moneyline": {"home": home_ml, "away": away_ml},
                    "spread": {
                        "home_line": home_sp,
                        "away_line": -home_sp if home_sp is not None else None,
                        "home_odds": default_odds,
                        "away_odds": default_odds
                    },
                    "total": {"line": total_line, "over_odds": default_odds, "under_odds": default_odds},
                    "bookmaker": "Parsed from Supabase",
                    "last_update": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")                }
                if any([
                    entry["moneyline"]["home"] is not None,
                    entry["moneyline"]["away"] is not None,
                    entry["spread"]["home_line"] is not None,
                    entry["total"]["line"] is not None
                ]):
                    odds_dict[gid] = entry
        except Exception as e:
            logger.error(f"Error fetching odds chunk starting at {i}: {e}", exc_info=True)

    logger.info(f"Finished fetching odds for {len(odds_dict)} games.")
    return odds_dict


# --- Calibration (no more confidence_pct) ---
#def calibrate_prediction_with_odds(
    #prediction: Dict,
   # odds_info: Optional[Dict],
  #  blend_factor: float = 0.3
#) -> Dict:
   # calibrated = prediction.copy()
   # calibrated['betting_odds'] = odds_info
   # calibrated['calibration_blend_factor'] = blend_factor
   # calibrated['is_calibrated'] = False

    #if not odds_info:
        #logger.debug(f"No odds info for game {prediction.get('game_id')}; skipping calibration.")
        #return calibrated

   # try:
        # extract market_spread & total...
       # spread = odds_info.get('spread', {})
      #  if spread.get('home_line') is not None:
         #   market_spread = spread['home_line']
       # elif spread.get('away_line') is not None:
          #  market_spread = -spread['away_line']
       # else:
          #  logger.debug("Missing market spread.")
           # return calibrated

       # total = odds_info.get('total', {}).get('line')
        #if total is None:
         #   logger.debug("Missing market total.")
          #  return calibrated

       # raw_diff  = float(prediction['predicted_point_diff'])
       # raw_total = float(prediction['predicted_total_score'])

       # cal_total = blend_factor * total + (1 - blend_factor) * raw_total
       # cal_diff  = blend_factor * market_spread + (1 - blend_factor) * raw_diff
       # cal_home  = (cal_total + cal_diff) / 2.0
       # cal_away  = (cal_total - cal_diff) / 2.0
       # cal_winp  = 1 / (1 + math.exp(-0.15 * cal_diff))

        #calibrated.update({
          #  'predicted_home_score':  round(cal_home, 1),
          #  'predicted_away_score':  round(cal_away, 1),
          #  'predicted_point_diff':  round(cal_diff, 1),
          #  'predicted_total_score': round(cal_total, 1),
           # 'win_probability':       round(cal_winp, 3),
          #  'is_calibrated':         True
       # })
        #logger.info(
            #f"Calibrated game {prediction.get('game_id')}: "
           #f"H {cal_home:.1f}, A {cal_away:.1f}, Diff {cal_diff:+.1f}, Total {cal_total:.1f}"
       # )
    #except Exception as e:
        #logger.error(f"Error during calibration for game {prediction.get('game_id')}: {e}", exc_info=True)

    #return calibrated

# --- Display Summary (dropped CONF % column) ---
def display_prediction_summary(preds: List[Dict]) -> None:
    header_line = "=" * 145
    header_title = " " * 55 + "NBA PREGAME PREDICTION SUMMARY"
    header_cols = (
        f"{'DATE':<11} {'MATCHUP':<30} {'FINAL SCORE':<12} {'FINAL DIFF':<10} "
        f"{'L/U BOUND':<12} {'RAW SCORE':<12} {'RAW DIFF':<10} "
        f"{'WIN PROB':<10} {'CALIB?':<6}"
    )
    header_sep = "-" * 145

    print(f"\n{header_line}\n{header_title}\n{header_line}")
    print(header_cols)
    print(header_sep)

    try:
        df = pd.DataFrame(preds)
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors="coerce")
            df = df.dropna(subset=['game_date'])
            df = df.sort_values(['game_date', 'predicted_point_diff'],
                                key=lambda x: abs(x) if x.name == 'predicted_point_diff' else x)
    except Exception as e:
        logger.error(f"Error sorting for display: {e}")
        df = pd.DataFrame(preds)

    for _, g in df.iterrows():
        try:
            date_str = g['game_date'].strftime("%Y-%m-%d") if pd.notna(g.get('game_date')) else "N/A"
            matchup = f"{g.get('home_team','?')[:14]} vs {g.get('away_team','?')[:14]}"
            raw_home = g.get('raw_predicted_home_score', np.nan)
            raw_away = g.get('raw_predicted_away_score', np.nan)
            raw_diff = g.get('raw_predicted_point_diff', np.nan)
            final_home = g.get('predicted_home_score', np.nan)
            final_away = g.get('predicted_away_score', np.nan)
            final_diff = g.get('predicted_point_diff', np.nan)
            winp = g.get('win_probability', np.nan)
            lb = g.get('lower_bound', np.nan)
            ub = g.get('upper_bound', np.nan)
            calib_str = "Yes" if g.get('is_calibrated') else "No"

            final_score = f"{final_home:.1f}-{final_away:.1f}" if pd.notna(final_home) and pd.notna(final_away) else "N/A"
            final_diff_str = f"{final_diff:+.1f}" if pd.notna(final_diff) else "N/A"
            bound_str = f"{lb:.1f}/{ub:.1f}" if pd.notna(lb) and pd.notna(ub) else "N/A"
            raw_score = f"{raw_home:.1f}-{raw_away:.1f}" if pd.notna(raw_home) and pd.notna(raw_away) else "N/A"
            raw_diff_str = f"{raw_diff:+.1f}" if pd.notna(raw_diff) else "N/A"
            win_prob_str = f"{winp*100:.1f}%" if pd.notna(winp) else "N/A"

            print(
                f"{date_str:<11} {matchup:<30} {final_score:<12} {final_diff_str:<10} "
                f"{bound_str:<12} {raw_score:<12} {raw_diff_str:<10} "
                f"{win_prob_str:<10} {calib_str:<6}"
            )
        except Exception as e:
            logger.error(f"Error displaying game {g.get('game_id')}: {e}")

    print(header_line)

# --- Core Pipeline -----------------------------------------------------------
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW,
    model_dir: Path = MODELS_DIR,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
    debug_mode: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    if not PROJECT_MODULES_IMPORTED:
        logger.critical("Project modules not imported correctly.")
        return [], []

    logger.info("--- Starting NBA Prediction Pipeline ---")
    start_ts = time.time()

    # optional DEBUG verbosity toggle
    root_logger = logging.getLogger()
    original_levels = [h.level for h in root_logger.handlers]
    orig_level = logger.level
    def _restore():
        logger.setLevel(orig_level)
        for lvl, h in zip(original_levels, root_logger.handlers):
            h.setLevel(lvl)
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        for h in root_logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled")

    # ---------- DATA ---------------------------------------------------------
    supabase = get_supabase_client()
    if not supabase:
        logger.critical("Cannot proceed without Supabase client")
        _restore()
        return [], []

    hist_df       = load_recent_historical_data(supabase, historical_lookback)
    team_stats_df = load_team_stats_data(supabase)
    upcoming_df   = fetch_upcoming_games_data(supabase, days_window)

    if upcoming_df.empty:
        logger.warning("No upcoming games; exiting.")
        _restore()
        return [], []
    if hist_df.empty:
        hist_df = None
    if team_stats_df.empty:
        team_stats_df = None

    cov_file = REPORTS_DIR / "historical_coverage_stats.csv"
    cov_df   = pd.read_csv(cov_file) if cov_file.is_file() else None
    uncertainty = PredictionUncertaintyEstimator(historical_coverage_stats=cov_df, debug=debug_mode)

    # ---------- MODELS -------------------------------------------------------
    models, feature_list = load_trained_models(model_dir, load_feature_list=False)
    if not models:
        _restore()
        return [], []

    # ---------- ENSEMBLE WEIGHTS -------------------------------------------
    weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()
    wfile = model_dir / ENSEMBLE_WEIGHTS_FILENAME
    if wfile.is_file():
        try:
            with wfile.open() as f:
                loaded = json.load(f)

            expected = set(weights.keys())  # {"ridge", "svr"}
            if abs(sum(loaded.values()) - 1.0) < 1e-5 and expected.issubset(loaded):
                subset = {k: float(loaded[k]) for k in expected}
                tot = sum(subset.values())
                if abs(tot - 1.0) > 1e-5 and tot > 0:
                    subset = {k: v / tot for k, v in subset.items()}
                weights = subset
                logger.info("Using ensemble weights: %s", weights)
            else:
                logger.warning("Invalid ensemble_weights.json – falling back to 50/50.")
        except Exception as e:
            logger.error("Error reading ensemble weights: %s", e)

    # ---------- FEATURES -----------------------------------------------------
    try:
        features_df = run_feature_pipeline(
            df=upcoming_df.copy(),
            historical_games_df=hist_df,
            team_stats_df=team_stats_df,
            rolling_windows=[5, 10, 20],
            h2h_window=7,
            debug=debug_mode
        )
    except Exception as e:
        logger.error(f"Feature pipeline error: {e}", exc_info=debug_mode)
        _restore()
        return [], []

    if features_df.empty:
        logger.error("No features returned; exiting.")
        _restore()
        return [], []

    # ---------- BUILD THE FEATURE MATRIX ------------------------------------
    # If selected_features.json is missing, fall back to *all* columns emitted by
    # the feature pipeline.  This keeps inference alive while you fix leakage.
    if feature_list is None:
        feature_list = list(features_df.columns)
        logger.warning(
            "selected_features.json not found — using all %d feature columns "
            "produced by the pipeline.",
            len(feature_list)
        )

    missing_cols = [c for c in feature_list if c not in features_df.columns]
    if missing_cols:
        logger.info("Adding %d missing columns as zeros: %s", len(missing_cols), missing_cols)

    X = (
        features_df
        .reindex(columns=feature_list, fill_value=0)      # add any missing cols
        .fillna(0)                                        # NaNs → 0
        .replace([np.inf, -np.inf], 0)                    # Infs → 0
    )

    # per-model preds
    preds_by_model: Dict[str, pd.DataFrame] = {}
    for name, mdl in models.items():
        if not hasattr(mdl, "predict"):
            logger.warning(f"Model '{name}' has no predict(); skipping.")
            continue
        try:
            arr = mdl.predict(X)
            preds_by_model[name] = pd.DataFrame(
                arr,
                columns=["predicted_home_score","predicted_away_score"],
                index=X.index
            )
        except Exception as e:
            logger.error(f"Error in {name}.predict(): {e}", exc_info=debug_mode)

    if not preds_by_model:
        logger.error("No predictions produced by any model; exiting.")
        _restore()
        return [], []

    info_df = features_df.set_index(X.index)[["game_id","game_date","home_team","away_team"]]
    raw_preds: List[Dict] = []

    for idx in X.index:
        row = info_df.loc[idx]
        comps = {
            name: {
                "home": float(df.at[idx,"predicted_home_score"]),
                "away": float(df.at[idx,"predicted_away_score"])
            }
            for name, df in preds_by_model.items()
            if idx in df.index
        }
        if not comps:
            continue

        # ensemble
        w_sum = sum(weights.get(n,0) for n in comps)
        if w_sum < 1e-6:
            h_ens = np.mean([c["home"] for c in comps.values()])
            a_ens = np.mean([c["away"] for c in comps.values()])
        else:
            h_ens = sum(comps[n]["home"] * weights.get(n,0) for n in comps) / w_sum
            a_ens = sum(comps[n]["away"] * weights.get(n,0) for n in comps) / w_sum

        diff = h_ens - a_ens
        tot  = h_ens + a_ens
        winp = 1 / (1 + math.exp(-0.1 * diff))

        try:
            tmp = pd.DataFrame({
                "predicted_home_score":[h_ens],
                "predicted_away_score":[a_ens]
            }, index=[idx])
            ints = uncertainty.add_prediction_intervals(tmp)
            lb = float(ints.at[idx,"total_score_lower"])
            ub = float(ints.at[idx,"total_score_upper"])
        except Exception:
            lb = ub = float("nan")

        raw_preds.append({
            "game_id":                row["game_id"],
            "game_date":              row["game_date"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team":              row["home_team"],
            "away_team":              row["away_team"],
            "predicted_home_score":   round(h_ens,1),
            "predicted_away_score":   round(a_ens,1),
            "predicted_point_diff":   round(diff,1),
            "predicted_total_score":  round(tot,1),
            "win_probability":        round(winp,3),
            "lower_bound":            (round(lb,1) if not math.isnan(lb) else None),
            "upper_bound":            (round(ub,1) if not math.isnan(ub) else None),
            "raw_predicted_home_score":  round(h_ens,1),
            "raw_predicted_away_score":  round(a_ens,1),
            "raw_predicted_point_diff":  round(diff,1),
            "raw_predicted_total_score": round(tot,1),
            "raw_win_probability":       round(winp,3),
            "raw_lower_bound":           (round(lb,1) if not math.isnan(lb) else None),
            "raw_upper_bound":           (round(ub,1) if not math.isnan(ub) else None),
            "component_predictions":     comps
        })

    # optional calibration
    #final_preds = []
   # if calibrate_with_odds:
      #  odds = fetch_and_parse_betting_odds(supabase, [r["game_id"] for r in raw_preds])
      #  if odds:
          #  for r in raw_preds:
            #    final_preds.append(calibrate_prediction_with_odds(r, odds.get(r["game_id"]), blend_factor))
       #else:
           # final_preds = raw_preds
    #else:
       # final_preds = raw_preds
     # skip any calibration—use raw model outputs directly
    final_preds = raw_preds

    display_prediction_summary(final_preds)
    logger.info(f"Pipeline completed in {(time.time() - start_ts):.1f}s")
    _restore()
    return final_preds, raw_preds

# --- Upsert Function ---
def upsert_score_predictions(predictions: List[Dict[str, Any]]) -> None:
    from supabase import create_client, Client
    from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    updated = 0

    for pred in predictions:
        gid = pred.get("game_id")
        if gid is None:
            print("Skipping prediction with missing game_id:", pred)
            continue

        update_payload = {
            "predicted_home_score": pred["predicted_home_score"],
            "predicted_away_score": pred["predicted_away_score"],
        }

        try:
            resp = (
                supabase
                .table("nba_game_schedule")
                .update(update_payload, returning="representation")
                .eq("game_id", int(gid))
                .execute()
            )
            if resp.data:
                print(f"Updated predicted scores for game_id {gid}.")
                updated += 1
            else:
                print(f"No row found to update for game_id {gid}.")
        except Exception as e:
            print(f"Error updating game_id {gid}: {e}")

    print(f"Finished updating predicted scores for {updated} games.")

# --- Main Execution ---
def main():
    import argparse
    import sys
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser(description="Generate NBA Score Predictions")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_UPCOMING_DAYS_WINDOW,
        help=f"Upcoming days to predict (default: {DEFAULT_UPCOMING_DAYS_WINDOW})"
    )
    parser.add_argument(
        "--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES,
        help=f"Historical lookback days (default: {DEFAULT_LOOKBACK_DAYS_FOR_FEATURES})"
    )
    parser.add_argument(
        "--model_dir", type=Path, default=MODELS_DIR,
        help=f"Directory of saved models (default: {MODELS_DIR})"
    )
    parser.add_argument(
        "--no_calibrate", action="store_true",
        help="Skip odds calibration step"
    )
    #parser.add_argument(
        #"--blend", type=float, default=0.3,
       # help="Blend factor for calibration (0=model, 1=odds)"
   # )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if not PROJECT_MODULES_IMPORTED:
        sys.exit("Exiting: Required project modules failed to import.")

    final_preds, raw_preds = generate_predictions(
        days_window=args.days,
        model_dir=args.model_dir,
        historical_lookback=args.lookback,
        debug_mode=args.debug
    )

    if final_preds:
        logger.info(f"Upserting {len(final_preds)} final predictions...")
        try:
            upsert_score_predictions(final_preds)
            logger.info("Upsert successful.")
        except NameError:
            logger.error("upsert_score_predictions not defined.")
        except Exception as e:
            logger.error(f"Error during upsert: {e}", exc_info=args.debug)
        logger.info("Sample Final Prediction:")
        print(json.dumps(final_preds[0], indent=2))
    else:
        logger.info("No final predictions to upsert.")

if __name__ == "__main__":
    main()