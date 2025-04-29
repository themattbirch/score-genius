# backend/nba_score_prediction/prediction.py

import os
import sys
import logging
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

# ensure project root is on PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Third‐party clients & config ---
from supabase import create_client
from caching.supabase_client import supabase as supabase_client_instance
from config import SUPABASE_URL, SUPABASE_ANON_KEY

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants & paths ---
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved'
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PACIFIC_TZ = pytz.timezone("America/Los_Angeles")
DEFAULT_LOOKBACK_DAYS_FOR_FEATURES = 180
DEFAULT_UPCOMING_DAYS_WINDOW = 3

ENSEMBLE_WEIGHTS_FILENAME = "ensemble_weights.json"
FALLBACK_ENSEMBLE_WEIGHTS: Dict[str, float] = {"ridge": 0.5, "svr": 0.5}

# --- Data Column Requirements ---
REQUIRED_HISTORICAL_COLS = [
    'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
    'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_ot',
    'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_ot',
    'home_fg_made', 'home_fg_attempted', 'away_fg_made', 'away_fg_attempted',
    'home_3pm', 'home_3pa', 'away_3pm', 'away_3pa',
    'home_ft_made', 'home_ft_attempted', 'away_ft_made', 'away_ft_attempted',
    'home_off_reb', 'home_def_reb', 'home_total_reb',
    'away_off_reb', 'away_def_reb', 'away_total_reb',
    'home_turnovers', 'away_turnovers',
    'home_assists', 'home_steals', 'home_blocks', 'home_fouls',
    'away_assists', 'away_steals', 'away_blocks', 'away_fouls'
]
REQUIRED_TEAM_STATS_COLS = [
    'team_name', 'season', 'wins_all_percentage', 'points_for_avg_all',
    'points_against_avg_all', 'current_form'
]
UPCOMING_GAMES_COLS = ["game_id", "scheduled_time", "home_team", "away_team"]
PROJECT_MODULES_IMPORTED = True

try:
    from backend.features.legacy.feature_engineering import FeatureEngine    
    from nba_score_prediction.models import RidgeScorePredictor, SVRScorePredictor
    from nba_score_prediction.simulation import PredictionUncertaintyEstimator
    import nba_score_prediction.utils as utils
    PROJECT_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Critical imports failed: {e}", exc_info=True)
    PROJECT_MODULES_IMPORTED = False

# --- Supabase helper ---
def get_supabase_client() -> Optional[Any]:
    """Prefer the cached service-role client, else fall back on anon."""
    if supabase_client_instance:
        return supabase_client_instance
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None

# --- Data Loading Functions ---
def load_recent_historical_data(supabase_client: Any, days_lookback: int) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical game data from {start_date} onwards...")
    select_cols_str = ", ".join(map(str, REQUIRED_HISTORICAL_COLS)) 
    all_historical_data = []
    page_size = 1000
    start_index = 0
    has_more = True
    try:
        while has_more:
            logger.debug(f"Fetching historical data batch starting at index {start_index}...")
            response = (
                supabase_client
                .table("nba_historical_game_stats")
                .select(select_cols_str)
                .gte("game_date", start_date)
                .order('game_date', desc=False) 
                .range(start_index, start_index + page_size - 1)
                .execute()
            )
            batch = response.data
            b_size = len(batch)
            all_historical_data.extend(batch)
            logger.debug(f"Retrieved {b_size} historical records in this batch.")
            if b_size < page_size:
                has_more = False
            else:
                start_index += page_size
        if not all_historical_data:
            logger.warning(f"No historical game data found since {start_date}.")
            return pd.DataFrame()
        historical_df = pd.DataFrame(all_historical_data)
        logger.info(f"Loaded {len(historical_df)} total historical games from Supabase.")
        if 'game_date' not in historical_df.columns:
            logger.error("Critical column 'game_date' missing.")
            return pd.DataFrame()
        historical_df['game_date'] = (
            pd.to_datetime(historical_df['game_date'], errors='coerce')
              .dt.tz_localize(None)
        )
        historical_df = historical_df.dropna(subset=['game_date'])
        numeric_cols = [
            col for col in REQUIRED_HISTORICAL_COLS
            if col not in ['game_id', 'game_date', 'home_team', 'away_team']
        ]
        for col in numeric_cols:
            if col in historical_df.columns:
                historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"Required historical column '{col}' missing. Filling with 0.")
                historical_df[col] = 0
        for col in ['game_id', 'home_team', 'away_team']:
            if col in historical_df.columns:
                historical_df[col] = historical_df[col].astype(str)
            else:
                logger.warning(f"Required historical column '{col}' missing. Filling with empty string.")
                historical_df[col] = "" 
        return historical_df.sort_values(by='game_date').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading historical data from Supabase: {e}", exc_info=True)
        return pd.DataFrame()

def load_team_stats_data(supabase_client: Any) -> pd.DataFrame:
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()

    logger.info("Loading team stats data from Supabase table 'nba_historical_team_stats'...")
    select_cols_str = ", ".join(REQUIRED_TEAM_STATS_COLS)
    all_stats = []
    page_size = 1000
    start_idx = 0

    try:
        # Page through in case you have >1k rows, ordering by season so results are deterministic
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
            logger.warning("No team stats data found in nba_historical_team_stats.")
            return pd.DataFrame()

        team_stats_df = pd.DataFrame(all_stats)
        logger.info(f"Loaded {len(team_stats_df)} team stat records.")

        # Coerce numeric columns
        numeric_cols = [
            col for col in REQUIRED_TEAM_STATS_COLS
            if col not in ['team_name', 'season', 'current_form']
        ]
        for col in numeric_cols:
            if col in team_stats_df.columns:
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0.0)
            else:
                logger.warning(f"Required team stat column '{col}' missing. Filling with 0.0.")
                team_stats_df[col] = 0.0

        # Ensure the string columns exist
        for col in ['team_name', 'season', 'current_form']:
            if col in team_stats_df.columns:
                team_stats_df[col] = team_stats_df[col].astype(str).fillna('')
            else:
                logger.warning(f"Required team stat column '{col}' missing. Filling with empty string.")
                team_stats_df[col] = ''

        return team_stats_df

    except Exception as e:
        logger.error(f"Error loading team stats from Supabase: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_upcoming_games_data(supabase_client: Any, days_window: int) -> pd.DataFrame:
    """
    Fetches upcoming game schedules from Supabase for a given days window.
    """
    if not supabase_client:
        logger.error("Supabase client unavailable.")
        return pd.DataFrame()
    now_pt = datetime.now(PACIFIC_TZ)
    start_utc = now_pt.astimezone(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_utc   = (start_utc + timedelta(days=days_window)).replace(hour=0, minute=0, second=0, microsecond=0)    
    start_utc_str = start_utc.strftime('%Y-%m-%dT%H:%M:%S%z')
    end_utc_str   = end_utc.strftime('%Y-%m-%dT%H:%M:%S%z') 
    logger.info(f"Fetching upcoming games between {start_utc_str} and {end_utc_str} (UTC)...")
    select_cols_str = ", ".join(UPCOMING_GAMES_COLS)
    try:
        response = supabase_client.table("nba_game_schedule") \
            .select(select_cols_str) \
            .gte("scheduled_time", start_utc_str) \
            .lt("scheduled_time", end_utc_str) \
            .order("scheduled_time", desc=False) \
            .execute()
        data = response.data
        if not data:
            logger.warning("No upcoming games found.")
            return pd.DataFrame()
        games_df = pd.DataFrame(data)
        logger.info(f"Fetched {len(games_df)} upcoming games.")
        if "scheduled_time" in games_df.columns:
            games_df["scheduled_time_utc"] = pd.to_datetime(games_df["scheduled_time"], errors='coerce', utc=True)
            games_df = games_df.dropna(subset=['scheduled_time_utc'])
            games_df["game_time_pt"] = games_df["scheduled_time_utc"].dt.tz_convert(PACIFIC_TZ)
            games_df["game_date"] = pd.to_datetime(games_df["game_time_pt"].dt.date)
        else:
            logger.error("Column 'scheduled_time' missing from schedule data.")
            return pd.DataFrame()
        for col in ["home_team", "away_team", "game_id"]:
            games_df[col] = games_df[col].astype(str)
        for col in set(REQUIRED_HISTORICAL_COLS) - {"game_id", "game_date", "home_team", "away_team"}:
            if col not in games_df.columns:
                games_df[col] = np.nan
        return games_df.sort_values(by='game_time_pt').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}", exc_info=True)
        return pd.DataFrame()

# --- Betting Odds Parsing Functions ---
def parse_odds_line(line_str: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    if not line_str:
        return None, None
    match = re.search(r"([\+\-]?\d+(?:\.\d+)?)\s*(?:\(?\s*(?:[ou])?\s*([\+\-]\d+)\s*\)?)?", str(line_str).strip())
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
            if ml1 < 0 and ml2 > 0: 
                home_ml = ml1; away_ml = ml2
            elif ml2 < 0 and ml1 > 0: 
                home_ml = ml2; away_ml = ml1
            elif ml1 < 0 and ml2 < 0:
                # Both moneylines negative: keep the assignments as they are
                home_ml = ml1; away_ml = ml2
                logger.debug(f"Both moneylines negative in '{ml_str}'. Using {ml1} for home and {ml2} for away.")
            elif ml1 > 0 and ml2 > 0:
                home_ml = ml1; away_ml = ml2
                logger.warning(f"Both moneylines positive in '{ml_str}'. Assigning {ml1} to home.")
            else:
                home_ml = ml1; away_ml = ml2
        elif len(mls) == 1:
            logger.warning(f"Only one moneyline found in '{ml_str}'. Cannot assign reliably.")
        else:
            logger.warning(f"Could not find two moneylines in '{ml_str}'.")
    except Exception as e:
        logger.warning(f"Could not parse moneyline string '{ml_str}': {e}")
    return home_ml, away_ml

def parse_spread_str(spread_str: Optional[str], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    home_line, away_line = None, None
    if not spread_str or not isinstance(spread_str, str):
        return home_line, away_line
    try:
        numbers = re.findall(r"([\+\-]?\d+(?:\.\d+)?)", spread_str)
        if len(numbers) == 2:
            val1, val2 = float(numbers[0]), float(numbers[1])
        
            if abs(val1 + val2) < 0.1:
                 home_line = val1 if val1 < 0 else val2
                 away_line = -home_line
            else: 
                home_line = val1
                away_line = -val1
                logger.warning(f"Spread values in '{spread_str}' don't sum to zero. Assuming {val1} is home line.")
        elif len(numbers) == 1:
            home_line = float(numbers[0])
            away_line = -home_line
        else:
            logger.debug(f"Could not parse spread from '{spread_str}'.")
            return None, None
    except Exception as e:
        logger.warning(f"Could not parse spread string '{spread_str}': {e}")
    return home_line, away_line

def parse_total_str(total_str: Optional[str]) -> Optional[float]:
    total_line = None
    if not total_str or not isinstance(total_str, str):
        return total_line
    try:
        numbers = re.findall(r"(\d{3}(?:\.\d+)?)", total_str)
        if numbers:
            total_line = float(numbers[0])
        else: 
             numbers = re.findall(r"(\d+(?:\.\d+)?)", total_str)
             if numbers:
                  total_line = float(numbers[0])
    except Exception as e:
        logger.warning(f"Could not parse total string '{total_str}': {e}")
    return total_line

def fetch_and_parse_betting_odds(
    supabase_client: Any,
    game_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    if not supabase_client or not game_ids:
        logger.warning("Skipping odds fetch due to missing client or game IDs.")
        return {}

    # Helper: build "Team +130 / Other -110"
    def make_moneyline_str(r):
        ml = r.get("moneyline") or {}
        h_raw = ml.get(r["home_team"])
        a_raw = ml.get(r["away_team"])
        # extract price whether nested or flat
        def price(x):
            if isinstance(x, dict):
                return x.get("price")
            elif isinstance(x, (int, float)):
                return int(x)
            else:
                return None
        h = price(h_raw)
        a = price(a_raw)
        if h is None or a is None:
            return None
        return f"{r['home_team']} {h:+d} / {r['away_team']} {a:+d}"

    # Helper: build "Team -2.5 / Other +2.5"
    def make_spread_str(r):
        sp = r.get("spread") or {}
        h_raw = sp.get(r["home_team"])
        a_raw = sp.get(r["away_team"])
        # extract point whether nested or flat
        def point(x):
            if isinstance(x, dict):
                return x.get("point")
            elif isinstance(x, (int, float)):
                return float(x)
            else:
                return None
        h = point(h_raw)
        a = point(a_raw)
        if h is None or a is None:
            return None
        return f"{r['home_team']} {h:+.1f} / {r['away_team']} {a:+.1f}"

    # Helper: still nested under Over/Under
    def make_total_str(r):
        tot = r.get("total") or {}
        over = tot.get("Over", {}).get("point")
        if over is None:
            return None
        return f"Over {over:.1f} / Under {over:.1f}"

    odds_dict: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50
    chunks = [game_ids[i : i + chunk_size] for i in range(0, len(game_ids), chunk_size)]
    select_cols = "game_id, home_team, away_team, moneyline, spread, total"

    logger.info(f"Fetching betting odds for {len(game_ids)} games in {len(chunks)} chunk(s)...")
    for idx, chunk in enumerate(chunks, start=1):
        logger.debug(f"Fetching odds chunk {idx}/{len(chunks)}…")
        try:
            resp = (
                supabase_client
                .table("nba_game_schedule")
                .select(select_cols)
                .in_("game_id", chunk)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                logger.debug(f"Chunk {idx} empty.")
                continue

            df = pd.DataFrame(rows)
            # ensure JSONB fields are dicts
            for c in ("moneyline","spread","total"):
                df[c] = df[c].apply(lambda x: x or {})

            for _, r in df.iterrows():
                gid = int(r["game_id"])
                ml_str  = make_moneyline_str(r)
                sp_str  = make_spread_str(r)
                tot_str = make_total_str(r)

                # persist the cleans back to your table
                supabase_client.table("nba_game_schedule")\
                    .update({
                        "moneyline_clean": ml_str,
                        "spread_clean":   sp_str,
                        "total_clean":    tot_str
                    })\
                    .eq("game_id", gid)\
                    .execute()

                # now parse for calibration
                home_ml,  away_ml  = parse_moneyline_str(ml_str,  r["home_team"], r["away_team"])
                home_sp,  _        = parse_spread_str(  sp_str,  r["home_team"], r["away_team"])
                total_line         = parse_total_str(  tot_str)
                default_odds = -110

                entry = {
                    "moneyline": {"home": home_ml,   "away": away_ml},
                    "spread":    {
                        "home_line": home_sp,
                        "away_line": -home_sp if home_sp is not None else None,
                        "home_odds": default_odds,
                        "away_odds": default_odds
                    },
                    "total":     {"line": total_line, "over_odds": default_odds, "under_odds": default_odds},
                    "bookmaker": "Parsed from Supabase",
                    "last_update": pd.Timestamp.now(tz="UTC")
                }
                if any([
                    entry["moneyline"]["home"] is not None,
                    entry["moneyline"]["away"] is not None,
                    entry["spread"]["home_line"] is not None,
                    entry["total"]["line"] is not None
                ]):
                    odds_dict[str(gid)] = entry
                else:
                    logger.debug(f"No usable odds for game {gid}.")

            logger.debug(f"Processed {len(df)} rows in chunk {idx}.")
        except Exception as e:
            logger.error(f"Error in odds chunk {idx}: {e}", exc_info=True)

    logger.info(f"Finished fetching odds. Parsed data for {len(odds_dict)} games.")
    return odds_dict

# --- Model Loading Function ---
def load_trained_models(model_dir: Path = MODELS_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Loads the feature-list JSON and then each saved model without
    any of the over-strict consistency checks.
    """
    # --- Step 1: load the feature list ---
    feature_list_path = model_dir / "selected_features.json"
    if not feature_list_path.is_file():
        logger.error(f"Feature list not found at {feature_list_path}")
        return None, None
    try:
        with open(feature_list_path, 'r') as f:
            feature_list = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read feature list: {e}", exc_info=True)
        return None, None

    # --- Step 2: load each model ---
    models: Dict[str, Any] = {}
    for name, Cls in [("svr", SVRScorePredictor), ("ridge", RidgeScorePredictor)]:
        try:
            pred = Cls(model_dir=str(model_dir), model_name=f"{name}_score_predictor")
            pred.load_model()
            models[name] = pred
            logger.info(f"Loaded {name} predictor.")
        except Exception as e:
            logger.error(f"Could not load '{name}' model: {e}", exc_info=True)

    if not models:
        logger.error("No base models loaded.")
        return None, None

    return models, feature_list

# --- Calibration Function (Restored Implementation) ---
def calibrate_prediction_with_odds(
    prediction: Dict, 
    odds_info: Optional[Dict],
    blend_factor: float = 0.3
) -> Dict:
    """ Calibrates raw ensemble predictions with market odds if available. """
    calibrated = prediction.copy() 

    # Store raw values before potential overwrite
    calibrated['raw_predicted_home_score'] = prediction.get('predicted_home_score')
    calibrated['raw_predicted_away_score'] = prediction.get('predicted_away_score')
    calibrated['raw_predicted_point_diff'] = prediction.get('predicted_point_diff')
    calibrated['raw_predicted_total_score'] = prediction.get('predicted_total_score')
    calibrated['raw_win_probability'] = prediction.get('win_probability')
    # Keep uncertainty estimates based on raw prediction for now
    calibrated['raw_lower_bound'] = prediction.get('lower_bound')
    calibrated['raw_upper_bound'] = prediction.get('upper_bound')
    calibrated['raw_confidence_pct'] = prediction.get('confidence_pct')

    calibrated['betting_odds'] = odds_info 
    calibrated['calibration_blend_factor'] = blend_factor
    calibrated['is_calibrated'] = False 

    game_id = prediction.get('game_id', 'N/A')

    if not odds_info:
        logger.debug(f"No odds info for game {game_id}. Calibration skipped.")
        return calibrated 

    try:
        market_spread, market_total = None, None

        spread_data = odds_info.get('spread', {})
        if spread_data.get('home_line') is not None:
            market_spread = spread_data['home_line'] 
        elif spread_data.get('away_line') is not None:
            market_spread = -spread_data['away_line']
        else:
             logger.debug(f"No spread line found in odds for game {game_id}.")

        # Extract market total
        total_data = odds_info.get('total', {})
        market_total = total_data.get('line')
        if market_total is None:
             logger.debug(f"No total line found in odds for game {game_id}.")

        # Get raw predictions needed for calibration
        pred_diff_raw = calibrated.get('raw_predicted_point_diff')
        pred_total_raw = calibrated.get('raw_predicted_total_score')

        # Check if we have the necessary components
        can_calibrate = True
        if market_spread is None: logger.debug(f"Cannot calibrate spread for game {game_id}: Missing market spread."); can_calibrate = False
        if market_total is None: logger.debug(f"Cannot calibrate total for game {game_id}: Missing market total."); can_calibrate = False
        if pred_diff_raw is None: logger.debug(f"Cannot calibrate spread for game {game_id}: Missing raw point diff."); can_calibrate = False
        if pred_total_raw is None: logger.debug(f"Cannot calibrate total for game {game_id}: Missing raw total score."); can_calibrate = False

        if not can_calibrate:
            logger.warning(f"Calibration skipped for game {game_id} due to missing values.")
            return calibrated 

        logger.info(f"Calibrating game {game_id} with blend factor {blend_factor} (Market Spread: {market_spread:+.1f}, Total: {market_total:.1f})")

        cal_total = blend_factor * market_total + (1 - blend_factor) * pred_total_raw
        cal_diff = blend_factor * market_spread + (1 - blend_factor) * pred_diff_raw 

        # Recalculate scores based on calibrated total and difference
        cal_home = (cal_total + cal_diff) / 2.0
        cal_away = (cal_total - cal_diff) / 2.0

        # Recalculate win probability based on calibrated difference
        cal_win_prob = 1 / (1 + math.exp(-0.15 * cal_diff))

        # Update the prediction dictionary with calibrated values
        calibrated['predicted_home_score'] = round(cal_home, 1)
        calibrated['predicted_away_score'] = round(cal_away, 1)
        calibrated['predicted_point_diff'] = round(cal_diff, 1)
        calibrated['predicted_total_score'] = round(cal_total, 1)
        calibrated['win_probability'] = round(cal_win_prob, 3)
        # Keep uncertainty based on raw prediction for now
        calibrated['lower_bound'] = calibrated['raw_lower_bound']
        calibrated['upper_bound'] = calibrated['raw_upper_bound']
        calibrated['confidence_pct'] = calibrated['raw_confidence_pct']
        calibrated['is_calibrated'] = True 

        logger.info(f"Calibrated game {game_id}: H {cal_home:.1f}, A {cal_away:.1f}, Diff {cal_diff:+.1f}, Total {cal_total:.1f}, WP {cal_win_prob:.1%}")

    except Exception as e:
        logger.error(f"Error during calibration for game {game_id}: {e}", exc_info=True)
        calibrated['is_calibrated'] = False 

    return calibrated


# --- Reporting Utility  ---
def display_prediction_summary(predictions: List[Dict]) -> None:
    """ Displays a formatted summary of final predictions. """
    header_line = "=" * 145
    header_title = " " * 55 + "NBA PREGAME PREDICTION SUMMARY"
    header_cols = (f"{'DATE':<11} {'MATCHUP':<30} {'FINAL SCORE':<12} {'FINAL DIFF':<10} "
                   f"{'L/U BOUND':<12} {'CONF %':<7} {'RAW SCORE':<12} {'RAW DIFF':<10} {'WIN PROB':<10} {'CALIB?':<6}")
    header_sep = "-" * 145

    print(f"\n{header_line}\n{header_title}\n{header_line}")
    print(header_cols)
    print(header_sep)
    try:
        # Create DataFrame for sorting
        predictions_df = pd.DataFrame(predictions)
        if 'game_date' not in predictions_df.columns:
             logger.error("Missing 'game_date' column for display sorting.")
        else:
             predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'], errors='coerce')
             predictions_df = predictions_df.dropna(subset=['game_date'])
             predictions_df = predictions_df.sort_values(
                 ['game_date', 'predicted_point_diff'],
                 key=lambda x: abs(x) if x.name == 'predicted_point_diff' else x,
                 na_position='last' 
            )
    except Exception as e:
        logger.error(f"Error sorting predictions for display: {e}")
        predictions_df = pd.DataFrame(predictions) 

    for _, game in predictions_df.iterrows():
        try:
            date_str = game['game_date'].strftime('%Y-%m-%d') if pd.notna(game.get('game_date')) else "N/A       "
            matchup = f"{game.get('home_team','?')[:14]} vs {game.get('away_team','?')[:14]}"
            is_calib = game.get('is_calibrated', False)
            calib_str = "Yes" if is_calib else "No "

            # Use .get with default=np.nan for numeric fields for safe formatting
            final_home = game.get('predicted_home_score', np.nan)
            final_away = game.get('predicted_away_score', np.nan)
            final_diff = game.get('predicted_point_diff', np.nan)
            final_win_prob = game.get('win_probability', np.nan)
            lower_b = game.get('lower_bound', np.nan)
            upper_b = game.get('upper_bound', np.nan)
            conf_pct = game.get('confidence_pct', np.nan)
            raw_home = game.get('raw_predicted_home_score', np.nan)
            raw_away = game.get('raw_predicted_away_score', np.nan)
            raw_diff = game.get('raw_predicted_point_diff', np.nan)

            # Format safely
            final_score_str = f"{final_home:.1f}-{final_away:.1f}" if pd.notna(final_home) and pd.notna(final_away) else "N/A"
            final_diff_str = f"{final_diff:<+10.1f}" if pd.notna(final_diff) else "   N/A    "
            bound_str = f"{lower_b:.1f}/{upper_b:.1f}" if pd.notna(lower_b) and pd.notna(upper_b) else "  N/A/N/A   "
            conf_str = f"{conf_pct:.1f}" if pd.notna(conf_pct) else "N/A"
            raw_score_str = f"{raw_home:.1f}-{raw_away:.1f}" if pd.notna(raw_home) and pd.notna(raw_away) else "N/A"
            raw_diff_str = f"{raw_diff:<+10.1f}" if pd.notna(raw_diff) else "   N/A    "
            win_prob_str = f"{final_win_prob*100:.1f}%" if pd.notna(final_win_prob) else "   N/A   "

            print(f"{date_str:<11} {matchup:<30} {final_score_str:<12} {final_diff_str:<10} "
                  f"{bound_str:<12} {conf_str:<7} {raw_score_str:<12} {raw_diff_str:<10} {win_prob_str:<10} {calib_str:<6}")
        except Exception as display_e:
            logger.error(f"Error displaying summary row for game {game.get('game_id', 'N/A')}: {display_e}")

    print("=" * 145)


# --- Main Prediction Orchestration ---
def generate_predictions(
    days_window: int = DEFAULT_UPCOMING_DAYS_WINDOW,
    model_dir: Path = MODELS_DIR,
    calibrate_with_odds: bool = True,
    blend_factor: float = 0.3,
    historical_lookback: int = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES
) -> Tuple[List[Dict], List[Dict]]:

    logger.info("--- Starting NBA Prediction Pipeline (Weighted Ensemble) ---")
    start_time = time.time()

    # 1) Supabase client & engines
    supabase = get_supabase_client()
    if not supabase:
        logger.critical("Cannot proceed without Supabase client")
        return [], []

    engine = FeatureEngine(supabase_client=supabase, debug=False)
    logger.info("FeatureEngine initialized")

    # 2) Load data
    hist_df = load_recent_historical_data(supabase, days_lookback=historical_lookback)
    team_stats_df = load_team_stats_data(supabase)
    upcoming_games_df = fetch_upcoming_games_data(supabase, days_window)
    if upcoming_games_df.empty:
        logger.warning("No upcoming games found. Exiting.")
        return [], []
    if hist_df.empty:
        logger.warning("Historical data empty; feature quality reduced.")
    if team_stats_df.empty:
        logger.warning("Team stats data empty; context features reduced.")

    # 3) Uncertainty estimator
    coverage_path = REPORTS_DIR / "historical_coverage_stats.csv"
    cov_df = pd.read_csv(coverage_path) if coverage_path.is_file() else None
    uncertainty_estimator = PredictionUncertaintyEstimator(
        historical_coverage_stats=cov_df, debug=False
    )

    # 4) Load models & ensemble weights
    models, feature_list = load_trained_models(model_dir)
    if not models or not feature_list:
        logger.error("Failed to load base models or feature list.")
        return [], []

    weights_path = model_dir / ENSEMBLE_WEIGHTS_FILENAME
    if weights_path.is_file():
        try:
            w = json.load(open(weights_path))
            if abs(sum(w.values()) - 1.0) > 1e-5:
                logger.warning("Ensemble weights do not sum to 1. Falling back.")
                ensemble_weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()
            else:
                ensemble_weights = {k: float(v) for k, v in w.items()}
        except Exception as e:
            logger.error("Error loading ensemble weights, using fallback.", exc_info=e)
            ensemble_weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()
    else:
        logger.warning("No ensemble weights file found; using fallback.")
        ensemble_weights = FALLBACK_ENSEMBLE_WEIGHTS.copy()

    # 5) Feature generation
    logger.info("Generating features...")
    features_df = engine.generate_all_features(
        df=upcoming_games_df,
        historical_games_df=(hist_df if not hist_df.empty else None),
        team_stats_df=(team_stats_df if not team_stats_df.empty else None),
        rolling_windows=[5, 10, 20]
    )
    if features_df is None or features_df.empty:
        logger.error("Feature generation failed.")
        return [], []

    # 6) Validate & slice features
    missing = set(feature_list) - set(features_df.columns)
    if missing:
        logger.error(f"Missing required features: {missing}")
        return [], []

    features_df = features_df[features_df['game_id'].isin(upcoming_games_df['game_id'])]
    X = utils.slice_dataframe_by_features(
        df=features_df, feature_list=feature_list, fill_value=0.0
    ).fillna(0.0)

    # 7) Base model predictions
    preds_by_model: Dict[str, pd.DataFrame] = {}
    for name, predictor in models.items():
        try:
            preds_df = predictor.predict(X)
            preds_df.index = X.index
            preds_by_model[name] = preds_df[['predicted_home_score', 'predicted_away_score']]
            logger.info(f"Base model {name} prediction successful. Shape: {preds_df.shape}")
        except Exception as e:
            logger.error(f"Error predicting with base model {name}: {e}")
            preds_by_model[name] = None

    # 8) Ensemble + uncertainty + raw collection
    raw_list: List[Dict] = []
    # Make sure this is the DataFrame you generated features on:
    feature_rows = features_df  # or whatever variable holds your features with game_id, game_date, home_team, away_team

    for idx in X.index:
        row = feature_rows.loc[idx]
        comp = {
            name: {
                'home': preds_by_model[name].at[idx, 'predicted_home_score'],
                'away': preds_by_model[name].at[idx, 'predicted_away_score']
            }
            for name in preds_by_model if preds_by_model[name] is not None
        }

        # weighted average
        h_ens = a_ens = wsum = 0.0
        for m, p in comp.items():
            w = ensemble_weights.get(m, 0.0)
            if w > 0:
                h_ens += p['home'] * w
                a_ens += p['away'] * w
                wsum += w
        if wsum > 0:
            h_ens /= wsum
            a_ens /= wsum
            method = 'weighted_avg'
        else:
            h_ens = np.mean([p['home'] for p in comp.values()])
            a_ens = np.mean([p['away'] for p in comp.values()])
            method = 'simple_avg_fallback'

        diff = h_ens - a_ens
        tot = h_ens + a_ens
        prob = 1 / (1 + math.exp(-0.15 * diff))

        lb, ub, cp = uncertainty_estimator.dynamically_adjust_interval(
            prediction=tot, current_quarter=0, historic_accuracy=None
        )

        raw_list.append({
            'game_id': row['game_id'],
            'game_date': row['game_date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'predicted_home_score': round(h_ens, 1),
            'predicted_away_score': round(a_ens, 1),
            'predicted_point_diff': round(diff, 1),
            'predicted_total_score': round(tot, 1),
            'win_probability': round(prob, 3),
            'lower_bound': round(lb, 1),
            'upper_bound': round(ub, 1),
            'confidence_pct': round(cp, 1),
            'component_predictions': comp,
            'ensemble_method': method,
            'is_calibrated': False
        })


    # 9) Calibrate against odds
    final_list: List[Dict] = []
    if calibrate_with_odds:
        gids = [r['game_id'] for r in raw_list]
        odds = fetch_and_parse_betting_odds(supabase, gids)
        for r in raw_list:
            final_list.append(calibrate_prediction_with_odds(r, odds.get(r['game_id']), blend_factor))
    else:
        final_list = [calibrate_prediction_with_odds(r, None, blend_factor) for r in raw_list]

    # 10) Summary & return
    display_prediction_summary(final_list)
    logger.info(f"Pipeline completed in {time.time() - start_time:.1f}s")
    return final_list, raw_list


# --- Script Execution ---
if __name__ == "__main__":
    logger.info("--- Running Prediction Script Directly ---")
    config_days_window = DEFAULT_UPCOMING_DAYS_WINDOW
    config_calibrate = True
    config_blend_factor = 0.3 
    config_hist_lookback = DEFAULT_LOOKBACK_DAYS_FOR_FEATURES

    logger.info(f"Config: Days={config_days_window}, Calibrate={config_calibrate}, Blend={config_blend_factor}, Lookback={config_hist_lookback}")

    final_preds, raw_preds = generate_predictions(
        days_window=config_days_window,
        model_dir=MODELS_DIR,
        calibrate_with_odds=config_calibrate,
        blend_factor=config_blend_factor,
        historical_lookback=config_hist_lookback
    )

    if final_preds:
        logger.info(f"Successfully generated {len(final_preds)} final predictions.")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"predictions_output_{ts}.csv"
            df_to_save = pd.DataFrame(final_preds)
            cols_to_drop_for_csv = ['component_predictions', 'betting_odds']
            df_to_save = df_to_save.drop(columns=cols_to_drop_for_csv, errors='ignore')
            numeric_cols_out = [ 
                 'predicted_home_score', 'predicted_away_score', 'predicted_point_diff',
                 'predicted_total_score', 'win_probability', 'lower_bound', 'upper_bound',
                 'confidence_pct', 'raw_predicted_home_score', 'raw_predicted_away_score',
                 'raw_predicted_point_diff', 'raw_predicted_total_score', 'raw_win_probability',
                 'raw_lower_bound', 'raw_upper_bound', 'raw_confidence_pct', 'calibration_blend_factor'
            ]
            for nc in numeric_cols_out:
                 if nc in df_to_save.columns:
                     df_to_save[nc] = pd.to_numeric(df_to_save[nc], errors='coerce')
                 else:
                      logger.debug(f"Numeric column '{nc}' not found in final DataFrame for CSV conversion.")

            df_to_save.to_csv(output_path, index=False, float_format="%.3f")
            logger.info(f"Final predictions saved to {output_path}")
        except Exception as e:
             logger.error(f"Failed to save predictions to CSV: {e}", exc_info=True)
    else:
        logger.warning("Prediction pipeline finished but produced no final results.")
    logger.info("--- Prediction Script Finished ---")

def upsert_score_predictions(predictions: List[Dict[str, Any]]) -> None:
    from supabase import create_client, Client
    from config import SUPABASE_URL, SUPABASE_SERVICE_KEY  # ← use service_role for writes!

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
            response = (
                supabase
                .table("nba_game_schedule")
                .update(update_payload, returning="representation")   # ← ask for the updated row
                .eq("game_id", int(gid))                              # ← make sure it's an int
                .execute()
            )
            if response.data:
                print(f"Updated predicted scores for game_id {gid}.")
                updated += 1
            else:
                print(f"No row found to update for game_id {gid}.")
        except Exception as e:
            print(f"Error updating game_id {gid}: {e}")

    print(f"Finished updating predicted scores for {updated} games.")



# Example usage:
if __name__ == "__main__":
    # ... Your existing prediction logic (e.g., generate_predictions) to obtain final_preds ...
    final_preds, raw_preds = generate_predictions(
        days_window=DEFAULT_UPCOMING_DAYS_WINDOW,
        model_dir=MODELS_DIR,
        calibrate_with_odds=True,
        blend_factor=0.3,
        historical_lookback=DEFAULT_LOOKBACK_DAYS_FOR_FEATURES
    )
    
    if final_preds:
        upsert_score_predictions(final_preds)
    else:
        print("No final predictions generated to upsert.")