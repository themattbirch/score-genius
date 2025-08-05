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
# Send all DEBUG+ messages to console
logging.basicConfig(format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

# suppress all HTTPX / HTTPCore / HPACK / bit debug spam
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
logging.getLogger("bit").setLevel(logging.WARNING)

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
from config import MAIN_MODELS_DIR as MODELS_DIR, REPORTS_DIR

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180
DEFAULT_UPCOMING_DAYS_WINDOW = 3

ENSEMBLE_WEIGHTS_FILENAME = "ensemble_weights.json"
FALLBACK_ENSEMBLE_WEIGHTS: Dict[str, float] = {"svr": 0.75, "ridge": 0.25}

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
    from nba_features.engine import run_feature_pipeline, DEFAULT_ORDER
    from nba_score_prediction.models import RidgeScorePredictor, SVRScorePredictor, XGBoostScorePredictor
    from nba_score_prediction.simulation import PredictionUncertaintyEstimator
    from backend.nba_features import utils
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

def load_nba_seasonal_splits_data(supabase_client: Any, season: int) -> pd.DataFrame:
    """Fetches all team seasonal splits for a given season using a bulk RPC."""
    if not supabase_client:
        return pd.DataFrame()
    logger.info(f"Loading all seasonal splits for season {season} via RPC...")
    try:
        resp = supabase_client.rpc(
            'rpc_get_nba_all_seasonal_splits',
            {'p_season': season}
        ).execute()
        if not resp.data:
            logger.warning(f"No seasonal splits data found for season {season}.")
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as e:
        logger.error(f"Error loading seasonal splits via RPC: {e}", exc_info=True)
        return pd.DataFrame()

def load_nba_rolling_features_data(supabase_client: Any, upcoming_games_df: pd.DataFrame) -> pd.DataFrame:
    """Fetches all rolling 20-game features for upcoming games using a bulk RPC."""
    if not supabase_client or upcoming_games_df.empty:
        return pd.DataFrame()
    
    logger.info("Preparing keys to fetch rolling features for upcoming games...")
    
    # Create the keys needed by the RPC: (game_id, team_id, game_date)
    home_keys = upcoming_games_df[['game_id', 'home_team', 'game_date']].rename(columns={'home_team': 'team_id'})
    away_keys = upcoming_games_df[['game_id', 'away_team', 'game_date']].rename(columns={'away_team': 'team_id'})
    
    keys_df = pd.concat([home_keys, away_keys], ignore_index=True)
    keys_df['game_date'] = keys_df['game_date'].dt.strftime('%Y-%m-%d')
    
    # Format for the RPC: array of composite type strings
    # Looks like: {"(game_id_val,team_id_val,game_date_val)","(...)"}
    rpc_keys = [f"({row['game_id']},{row['team_id']},{row['game_date']})" for _, row in keys_df.iterrows()]
    
    logger.info(f"Loading rolling features for {len(upcoming_games_df)} games via RPC...")
    try:
        resp = supabase_client.rpc(
            'rpc_get_nba_rolling_features_for_games',
            {'p_keys': rpc_keys}
        ).execute()
        if not resp.data:
            logger.warning("No rolling features data returned from RPC.")
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as e:
        logger.error(f"Error loading rolling features via RPC: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client: Any, days_window: int) -> pd.DataFrame:
    # Expand UTC cutoff to align with ET-midnight → UTC
    from zoneinfo import ZoneInfo

    ET  = ZoneInfo("America/New_York")
    UTC = ZoneInfo("UTC")

    # midnight at ET today, then + days_window
    today_et = datetime.now(ET).date()
    start_et = datetime(today_et.year, today_et.month, today_et.day, tzinfo=ET)
    end_et   = start_et + timedelta(days=days_window)

    # convert to UTC for querying
    start_utc = start_et.astimezone(UTC)
    end_utc   = end_et.astimezone(UTC)

    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Fetching upcoming games between {start_str} and {end_str} (UTC ← ET window)...")

    try:
        resp = (
            supabase_client
            .table("nba_game_schedule")
            .select(", ".join(UPCOMING_GAMES_COLS))
            .gte("scheduled_time", start_str)
            .lt("scheduled_time", end_str)
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

# --- Model Loading -----------------------------------------------------------
def load_trained_models(
    model_dir: Path = MODELS_DIR,
    load_feature_list: bool = False,   # if True, try to read selected_features.json
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """
    Load saved `*_score_predictor.joblib` models from `model_dir`.
    Also attempts to load `selected_features.json` from the same directory.

    Returns:
      - models: dict mapping model keys (e.g. 'svr', 'ridge') to predictor instances
      - feature_list: list of selected features if JSON is present, else None
    """
    # --- Load selected feature list if available
    feature_list: Optional[List[str]] = None
    features_fp = model_dir / "selected_features.json"
    if features_fp.is_file():
        try:
            with open(features_fp, 'r') as f:
                feature_list = json.load(f)
            logger.info(f"Loaded selected_features.json with {len(feature_list)} features")
        except Exception as e:
            logger.warning(f"Could not read selected_features.json: {e}. Continuing without it.")
    else:
        logger.warning("selected_features.json not found — will use all pipeline features.")

    # --- Load models
    models: Dict[str, Any] = {}
    for name, Cls in [("svr", SVRScorePredictor), ("ridge", RidgeScorePredictor)]:
        try:
            pred = Cls(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            pred.load_model()
            models[name] = pred
            logger.info(f"Loaded '{name}' predictor from {model_dir}")
        except Exception as e:
            logger.error(f"Could not load '{name}' model: {e}", exc_info=True)

    if not models:
        raise RuntimeError(f"No models could be loaded from {model_dir}")

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
                    .eq("game_id", gid)\
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

    # ————— Setup logging/debug toggle —————
    root_logger = logging.getLogger()
    orig_levels = [h.level for h in root_logger.handlers]
    orig_level = logger.level
    def _restore():
        logger.setLevel(orig_level)
        for lvl, h in zip(orig_levels, root_logger.handlers):
            h.setLevel(lvl)

    if debug_mode:
        logger.setLevel(logging.DEBUG)
        for h in root_logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled")

    start_ts = time.time()
    logger.info("--- Starting NBA Prediction Pipeline ---")

    # ————— LOAD DATA —————
    supabase      = get_supabase_client()
    if not supabase:
        logger.critical("Cannot proceed without Supabase client")
        _restore()
        return [], []

    # -- Determine current NBA season (e.g., 2024 for the 2024-25 season) --
    now = datetime.now()
    current_season = now.year if now.month >= 8 else now.year - 1
    logger.info(f"Determined current NBA season to be: {current_season}")

    # -- Fetch data from base tables (no change here) --
    hist_df       = load_recent_historical_data(supabase, historical_lookback)
    upcoming_df   = fetch_upcoming_games_data(supabase, days_window)

    # -- FETCH DATA USING NEW RPCS --
    seasonal_splits_df = load_nba_seasonal_splits_data(supabase, current_season)
    rolling_features_df = load_nba_rolling_features_data(supabase, upcoming_df)
    
    # Note: load_team_stats_data is no longer called as its data should
    # be replaced by the more detailed seasonal_splits_df

    if upcoming_df.empty:
        logger.warning("No upcoming games; exiting.")
        _restore()
        return [], []

    # If no history or other data, pass None downstream
    if hist_df.empty: hist_df = None
    if seasonal_splits_df.empty: seasonal_splits_df = None
    if rolling_features_df.empty: rolling_features_df = None

    # Uncertainty estimator (optional, no change)
    cov_file = REPORTS_DIR / "historical_coverage_stats.csv"
    cov_df   = pd.read_csv(cov_file) if cov_file.is_file() else None
    uncertainty = PredictionUncertaintyEstimator(
        historical_coverage_stats=cov_df,
        debug=debug_mode
    )

    # ————— LOAD MODELS & FEATURE LIST (no change) —————
    models, feature_list = load_trained_models(model_dir, load_feature_list=True)
    if not models:
        logger.error("No models loaded; aborting.")
        _restore()
        return [], []

    # ————— ENSEMBLE WEIGHTS (no change) —————
    weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()
    wfile   = model_dir / ENSEMBLE_WEIGHTS_FILENAME
    if wfile.is_file():
        try:
            # (logic remains the same)
            raw = json.loads(wfile.read_text())
            if not isinstance(raw, dict) or not raw: raise ValueError("must be a non-empty dict")
            base_weights: Dict[str, float] = {}
            for fullname, val in raw.items():
                base = fullname.replace("_score_predictor", "")
                base_weights[base] = float(val)
            total = sum(base_weights.values())
            expected = set(weights.keys())
            loaded   = set(base_weights.keys())
            if abs(total - 1.0) > 1e-6 or not loaded.issubset(expected): raise ValueError(f"sum={total:.4f}, keys={loaded}")
            for m in expected - loaded: base_weights[m] = 0.0
            subtotal = sum(base_weights.values())
            if abs(subtotal - 1.0) > 1e-6 and subtotal > 0: base_weights = {k: v / subtotal for k, v in base_weights.items()}
            weights = base_weights
            logger.info("Loaded ensemble weights from %s: %s", wfile.name, weights)
        except Exception as e:
            logger.warning("%s invalid (%s); falling back to defaults %s", wfile.name, e, FALLBACK_ENSEMBLE_WEIGHTS)
            weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()
    else:
        logger.warning("%s not found in %s; using defaults %s", wfile.name, model_dir, FALLBACK_ENSEMBLE_WEIGHTS)
    logger.debug("Final ensemble weights = %s", weights)


    # ————— FEATURE ENGINEERING —————

    # Merge historical + upcoming so advanced→rolling sees box scores
    if hist_df is not None:
        combined = pd.concat([hist_df, upcoming_df], ignore_index=True)
    else:
        combined = upcoming_df.copy()

    try:
        # UPDATED: Pass the pre-fetched DataFrames directly into the pipeline
        # The db_conn argument is removed as it's no longer needed for these features
        features_all = run_feature_pipeline(
            df=combined,
            seasonal_splits_data=seasonal_splits_df,  # New argument
            rolling_features_data=rolling_features_df, # New argument
            rolling_windows=[5, 10, 20],
            h2h_lookback=7,
            execution_order=DEFAULT_ORDER,
            adv_splits_lookup_offset=-1,
            flag_imputations_all=True,
            debug=debug_mode
        )
    except Exception as e:
        logger.error("Feature pipeline error: %s", e, exc_info=debug_mode)
        _restore()
        return [], []

    # ————— (The rest of the function for prediction, assembly, and display remains the same) —————

    # Slice back out just the upcoming games
    features_df = features_all.loc[
        features_all["game_id"].isin(upcoming_df["game_id"])
    ].reset_index(drop=True)

    if features_df.empty:
        logger.error("No features returned; exiting.")
        _restore()
        return [], []

    # ————— BUILD FEATURE MATRIX —————
    if feature_list:
        logger.info("Applying %d selected features from selected_features.json", len(feature_list))
        missing = [c for c in feature_list if c not in features_df.columns]
        if missing:
            logger.warning("Missing %d features; filling with zeros: %s", len(missing), missing)
        X = features_df.reindex(columns=feature_list, fill_value=0)
    else:
        logger.warning("selected_features.json not found — using all %d pipeline-generated features", features_df.shape[1])
        X = features_df.copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # ————— PER-MODEL PREDICTIONS —————
    preds_by_model: Dict[str, pd.DataFrame] = {}
    for name, mdl in models.items():
        if not hasattr(mdl, "predict"):
            logger.warning("Model '%s' has no predict(); skipping.", name)
            continue
        try:
            arr = mdl.predict(X)
            preds_by_model[name] = pd.DataFrame(
                arr,
                columns=["predicted_home_score","predicted_away_score"],
                index=X.index
            )
        except Exception as e:
            logger.error("Error in %s.predict(): %s", name, e, exc_info=debug_mode)
    if not preds_by_model:
        logger.error("No predictions produced by any model; exiting.")
        _restore()
        return [], []

    # ————— ASSEMBLE RAW PREDICTIONS —————
    info_df = features_df.set_index(X.index)[["game_id","game_date","home_team","away_team"]]
    raw_preds: List[Dict] = []
    for idx in X.index:
        row = info_df.loc[idx]
        comps = {m: {"home": float(df.at[idx,"predicted_home_score"]), "away": float(df.at[idx,"predicted_away_score"])} for m, df in preds_by_model.items() if idx in df.index}
        if not comps: continue
        logger.debug("Blending models %s at idx %s with weights %s", list(comps.keys()), idx, weights)
        w_sum = sum(weights.get(m,0) for m in comps)
        if w_sum < 1e-6:
            h_ens = np.mean([c["home"] for c in comps.values()])
            a_ens = np.mean([c["away"] for c in comps.values()])
        else:
            h_ens = sum(comps[m]["home"] * weights[m] for m in comps) / w_sum
            a_ens = sum(comps[m]["away"] * weights[m] for m in comps) / w_sum
        diff, tot, winp = h_ens - a_ens, h_ens + a_ens, 1 / (1 + math.exp(-0.1 * (h_ens - a_ens)))
        try:
            tmp = pd.DataFrame({"predicted_home_score":[h_ens], "predicted_away_score":[a_ens]}, index=[idx])
            ints = uncertainty.add_prediction_intervals(tmp)
            lb, ub = float(ints.at[idx,"total_score_lower"]), float(ints.at[idx,"total_score_upper"])
        except Exception: lb, ub = float("nan"), float("nan")
        raw_preds.append({
            "game_id": row["game_id"], "game_date": row["game_date"].strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": row["home_team"], "away_team": row["away_team"],
            "predicted_home_score": round(h_ens,1), "predicted_away_score": round(a_ens,1), "predicted_point_diff": round(diff,1), "predicted_total_score": round(tot,1),
            "win_probability": round(winp,3), "lower_bound": (round(lb,1) if not math.isnan(lb) else None), "upper_bound": (round(ub,1) if not math.isnan(ub) else None),
            "raw_predicted_home_score": round(h_ens,1), "raw_predicted_away_score": round(a_ens,1), "raw_predicted_point_diff": round(diff,1),
            "raw_predicted_total_score": round(tot,1), "raw_win_probability": round(winp,3),
            "raw_lower_bound": (round(lb,1) if not math.isnan(lb) else None), "raw_upper_bound": (round(ub,1) if not math.isnan(ub) else None),
            "component_predictions": comps
        })

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