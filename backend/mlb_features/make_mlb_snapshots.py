# backend/mlb_features/make_mlb_snapshots.py
"""
Generate and upsert per-game MLB feature snapshots for frontend display.
Orchestrates data fetching, runs the MLB Python feature pipeline for core features,
computes display-specific features (e.g., handedness), integrates data from specific RPCs,
and assembles payloads for headlines, bar, radar, and pie charts.
"""
import os
import sys
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import pandas as pd
from supabase import create_client, Client
from dateutil import parser as dateutil_parser # For robust date parsing
import numpy as np # For pd.NA handling

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Path Setup ---
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- Config and Utility Imports ---
try:
    from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL/key missing in make_mlb_snapshots.py")

# --- Import MLB Feature Engine and specific modules ---
try:
    from backend.mlb_features.engine import run_mlb_feature_pipeline
    from backend.mlb_features.utils import normalize_team_name, determine_season, DEFAULTS as MLB_DEFAULTS
    from backend.mlb_features.handedness_for_display import transform as handedness_transform # New module
    logger.info("Successfully imported MLB engine and feature modules.")
except ImportError as e:
    logger.error(f"Could not import MLB feature dependencies: {e}")
    logger.error("Please ensure backend/mlb_features/engine.py, utils.py, handedness_for_display.py exist and are correctly defined.")
    sys.exit(1)

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Supabase Client ---
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Fetch Helpers ---
def fetch_table(table_name: str, match_criteria: Dict[str, Any], select_cols: str = "*") -> pd.DataFrame:
    logger.debug(f"Fetching from table '{table_name}' with criteria: {match_criteria}")
    try:
        response = sb_client.table(table_name).select(select_cols).match(match_criteria).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Error fetching from table '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

def fetch_mlb_full_history() -> pd.DataFrame:
    """Entire historical game stats for H2H and rolling features for MLB."""
    logger.debug("Fetching full MLB historical game stats...")
    response = sb_client.table("mlb_historical_game_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    # Ensure date column is parsed for form strings and filtering
    if 'game_date_time_utc' in df.columns:
        df['parsed_game_date'] = pd.to_datetime(df['game_date_time_utc'], errors='coerce').dt.date
    elif 'game_date_et' in df.columns:
        df['parsed_game_date'] = pd.to_datetime(df['game_date_et'], errors='coerce').dt.date
    logger.debug(f"Fetched {len(df)} rows from mlb_historical_game_stats.")
    return df

def fetch_mlb_team_season_stats() -> pd.DataFrame:
    """All teams' season stats from mlb_historical_team_stats for MLB."""
    logger.debug("Fetching MLB historical team season stats...")
    response = sb_client.table("mlb_historical_team_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    logger.debug(f"Fetched {len(df)} rows from mlb_historical_team_stats.")
    return df

def fetch_mlb_pitcher_splits_data(season_year: int) -> pd.DataFrame:
    """Fetches pitcher splits data for a given season from mlb_historical_team_stats."""
    # Assuming mlb_historical_team_stats contains season_avg_runs_vs_lhp/rhp
    # This table has the required columns: team_id, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp
    logger.debug(f"Fetching MLB pitcher splits data for season: {season_year}")
    response = sb_client.table("mlb_historical_team_stats").select(
        "team_id, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp"
    ).eq("season", season_year).execute()
    df = pd.DataFrame(response.data or [])
    if df.empty:
        logger.warning(f"No MLB pitcher splits found for season {season_year} in mlb_historical_team_stats.")
    return df

# --- Helper Function _get_mlb_team_form_string ---
def _get_mlb_team_form_string(
    team_id_to_check: Union[str, int],
    current_game_date_et: pd.Timestamp | Any, # Can be date object or string for parsing
    all_historical_games: pd.DataFrame,
    num_form_games: int = 5
) -> str:
    team_id_str = str(team_id_to_check)
    
    current_game_date_obj = pd.to_datetime(current_game_date_et, errors='coerce').date()

    if pd.isna(current_game_date_obj) or all_historical_games.empty:
        logger.debug(f"Cannot get form for {team_id_str}: invalid date or empty historical games.")
        return "N/A"

    # Ensure parsed_date_for_form is created and used consistently
    if 'parsed_date_for_form' not in all_historical_games.columns:
        if 'game_date_et' in all_historical_games.columns:
            all_historical_games['parsed_date_for_form'] = pd.to_datetime(all_historical_games['game_date_et'], errors='coerce').dt.date
        elif 'game_date_time_utc' in all_historical_games.columns:
            all_historical_games['parsed_date_for_form'] = pd.to_datetime(all_historical_games['game_date_time_utc'], errors='coerce').dt.date # Keep as date only for comparison
        else:
            logger.warning("No suitable date column found in all_historical_games for form string generation.")
            return "N/A"
    
    # Ensure team_id columns are string for consistent comparison
    if 'home_team_id' in all_historical_games.columns:
        all_historical_games['home_team_id_str'] = all_historical_games['home_team_id'].astype(str)
    if 'away_team_id' in all_historical_games.columns:
        all_historical_games['away_team_id_str'] = all_historical_games['away_team_id'].astype(str)

    team_games = all_historical_games[
        ((all_historical_games['home_team_id_str'] == team_id_str) | \
         (all_historical_games['away_team_id_str'] == team_id_str)) & \
        (all_historical_games['parsed_date_for_form'] < current_game_date_obj) & \
        (all_historical_games['status_short'].isin(['FT', 'F']) | all_historical_games['status_long'] == 'Finished') & \
        (pd.notna(all_historical_games['home_score']) & pd.notna(all_historical_games['away_score'])) # Ensure scores are present for result
    ].copy()

    if team_games.empty:
        logger.debug(f"No completed games found for {team_id_str} before {current_game_date_obj} for form string.")
        return "N/A"

    recent_games = team_games.sort_values(by='parsed_date_for_form', ascending=False).head(num_form_games)

    if recent_games.empty:
        logger.debug(f"Not enough recent completed games found for {team_id_str} for form string.")
        return "N/A"

    form_results = []
    for _, game_row in recent_games.sort_values(by='parsed_date_for_form', ascending=True).iterrows():
        is_home = str(game_row['home_team_id_str']) == team_id_str
        team_score = pd.to_numeric(game_row['home_score'] if is_home else game_row['away_score'], errors='coerce')
        opponent_score = pd.to_numeric(game_row['away_score'] if is_home else game_row['home_score'], errors='coerce')
        
        if pd.isna(team_score) or pd.isna(opponent_score):
            form_results.append("?") # Indicate unknown result if scores are missing
        elif team_score > opponent_score:
            form_results.append("W")
        elif team_score < opponent_score:
            form_results.append("L")
        else:
            form_results.append("T") # MLB can have ties in rare cases before extra innings, or if game called due to weather/etc.

    return "".join(form_results) if form_results else "N/A"


# --- Snapshot Generator ---
def make_mlb_snapshot(
    game_id: Union[str, int],
    # Default column names used in this function that access df_game (from schedule/historical)
    input_game_date_col: str = "game_date_et", # Primary date col in mlb_game_schedule
    input_game_date_utc_col: str = "game_date_time_utc", # Fallback date col, typically in mlb_historical_game_stats
    input_home_team_id_col: str = "home_team_id",
    input_away_team_id_col: str = "away_team_id",
    input_home_pitcher_hand_col: str = "home_starter_pitcher_handedness", # Updated from home_probable_pitcher_handedness
    input_away_pitcher_hand_col: str = "away_starter_pitcher_handedness", # Updated from away_probable_pitcher_handedness
    input_home_score_col: str = "home_score", # From mlb_historical_game_stats
    input_away_score_col: str = "away_score",
    ):
    game_id_str = str(game_id)
    logger.info(f"--- Generating MLB Snapshot for game_id: {game_id_str} ---")

    # 1) Load raw game data: Try historical stats (completed games) first
    # Ensure all columns required by features (like inning scores, pitcher handedness) are selected
    game_data_cols_select = "*, h_inn_1, h_inn_2, h_inn_3, h_inn_4, h_inn_5, h_inn_6, h_inn_7, h_inn_8, h_inn_9, h_inn_extra, " \
                            "a_inn_1, a_inn_2, a_inn_3, a_inn_4, a_inn_5, a_inn_6, a_inn_7, a_inn_8, a_inn_9, a_inn_extra, " \
                            "home_starter_pitcher_handedness, away_starter_pitcher_handedness"
    df_game = fetch_table("mlb_historical_game_stats", {"game_id": game_id_str}, select_cols=game_data_cols_select)
    
    is_historical_game = not df_game.empty 
    
    if not is_historical_game:
        df_game = fetch_table("mlb_game_schedule", {"game_id": game_id_str})
        if not df_game.empty:
            logger.debug(f"Found MLB game {game_id_str} in mlb_game_schedule.")
            # Add placeholder columns for pre-game, to avoid key errors in feature pipeline
            # These columns are expected by the feature engine and later by the display-only handedness module
            for col in [
                input_home_score_col, input_away_score_col, 
                input_home_pitcher_hand_col, input_away_pitcher_hand_col,
                'h_inn_1', 'h_inn_2', 'h_inn_3', 'h_inn_4', 'h_inn_5', 'h_inn_6', 'h_inn_7', 'h_inn_8', 'h_inn_9', 'h_inn_extra',
                'a_inn_1', 'a_inn_2', 'a_inn_3', 'a_inn_4', 'a_inn_5', 'a_inn_6', 'a_inn_7', 'a_inn_8', 'a_inn_9', 'a_inn_extra',
            ]:
                if col not in df_game.columns:
                    df_game[col] = pd.NA # Use pandas NA for nullable integer/float columns
        else:
            logger.error(f"No primary data found for game_id={game_id_str}. Cannot generate snapshot.")
            return

    df_game['is_historical_game'] = is_historical_game

    # Determine game date and season
    current_game_date_et = None
    if input_game_date_col in df_game.columns and pd.notna(df_game[input_game_date_col].iloc[0]):
        current_game_date_et = pd.to_datetime(df_game[input_game_date_col].iloc[0], errors='coerce').date()
    elif input_game_date_utc_col in df_game.columns and pd.notna(df_game[input_game_date_utc_col].iloc[0]):
        try:
            dt_obj_utc = pd.to_datetime(df_game[input_game_date_utc_col].iloc[0], errors='coerce')
            if pd.isna(dt_obj_utc): # Handle cases where to_datetime yields NaT
                current_game_date_et = None
            else:
                # If the datetime object is naive (no timezone info), localize it to UTC.
                # If it's already tz-aware, tz_localize will raise an error, so handle it.
                if dt_obj_utc.tz is None: 
                    dt_obj_utc = dt_obj_utc.tz_localize('UTC')
                # Now convert to the target timezone (ET) and get the date part
                current_game_date_et = dt_obj_utc.tz_convert('America/New_York').date()
        except Exception as e:
            logger.error(f"Error parsing {input_game_date_utc_col} for {game_id_str}: {e}", exc_info=True) # Added exc_info=True for full traceback

    if current_game_date_et is None:
        logger.error(f"No valid game date found for game_id={game_id_str}. Cannot proceed.")
        return
    
    # 1) Determine season for this game
    season_year = determine_season(pd.Timestamp(current_game_date_et))

    # 2) Radar metric → feature-column map
    radar_metrics_map: dict[str, dict[str, str | int]] = {
        "Venue Win %": {
            "home_col": "home_venue_win_pct_home",
            "away_col": "away_venue_win_pct_away",
            "round": 3
        },
        "Season Runs Scored": {
            "home_col": "home_season_runs_for_avg",
            "away_col": "away_season_runs_for_avg",
            "round": 2
        },
        "Season Runs Against": {
            "home_col": "home_season_runs_against_avg",
            "away_col": "away_season_runs_against_avg",
            "round": 2
        },
        "Home/Away Win Advantage": {
            "home_col": "home_venue_win_advantage",
            "away_col": "away_venue_win_advantage",
            "round": 3
        },
    }

    # 3) Fetch historical context
    df_full_history = fetch_mlb_full_history()      # For H2H & form
    df_team_stats  = fetch_mlb_team_season_stats()  # For league‐wide stats

    # 4) Generate form strings
    home_id = df_game[input_home_team_id_col].iloc[0]
    away_id = df_game[input_away_team_id_col].iloc[0]
    df_game["home_current_form"] = _get_mlb_team_form_string(home_id, current_game_date_et, df_full_history)
    df_game["away_current_form"] = _get_mlb_team_form_string(away_id, current_game_date_et, df_full_history)
    logger.debug(f"Form strings: Home={df_game['home_current_form'].iloc[0]}, Away={df_game['away_current_form'].iloc[0]}")

    # 5) Run core MLB feature pipeline
    logger.info(f"Running core MLB feature pipeline for game {game_id_str}…")
    df_features = run_mlb_feature_pipeline(
        df=df_game.copy(),
        mlb_historical_games_df=df_full_history,
        mlb_historical_team_stats_df=df_team_stats,
        debug=True,
        keep_display_only_features=True
    )
    if df_features.empty:
        logger.error(f"MLB Feature pipeline returned empty for game_id={game_id_str}")
        return
    if len(df_features) > 1:
        logger.warning(f"Expected 1 feature row, got {len(df_features)}; using first.")
    row = df_features.iloc[0]

    # 6) Add display-only handedness features
    df_pitcher_splits = fetch_mlb_pitcher_splits_data(season_year)
    home_hand_col, away_hand_col = (
        ("home_starter_pitcher_handedness", "away_starter_pitcher_handedness")
        if is_historical_game
        else ("home_probable_pitcher_handedness", "away_probable_pitcher_handedness")
    )
    row[home_hand_col] = df_game[home_hand_col].iloc[0]
    row[away_hand_col] = df_game[away_hand_col].iloc[0]
    df_hand = handedness_transform(
        df=pd.DataFrame([row]),
        mlb_pitcher_splits_df=df_pitcher_splits,
        home_team_col_param="home_team_norm",
        away_team_col_param="away_team_norm",
        home_pitcher_hand_col=home_hand_col,
        away_pitcher_hand_col=away_hand_col,
        debug=True
    )
    if not df_hand.empty:
        row = df_hand.iloc[0]

    # 7) Fetch advanced team stats RPC for bar chart defaults
    rpc_adv = sb_client.rpc("get_mlb_advanced_team_stats_splits", {"p_season": season_year}).execute().data or []
    df_rpc_adv = pd.DataFrame(rpc_adv)
    if "team_id" in df_rpc_adv.columns:
        df_rpc_adv["team_norm"] = df_rpc_adv["team_id"].apply(normalize_team_name)
    home_norm = normalize_team_name(row.get(input_home_team_id_col))
    away_norm = normalize_team_name(row.get(input_away_team_id_col))
    home_adv_rpc_data = (
        df_rpc_adv.loc[df_rpc_adv["team_norm"] == home_norm].iloc[0]
        if not df_rpc_adv[df_rpc_adv["team_norm"] == home_norm].empty
        else pd.Series()
    )
    away_adv_rpc_data = (
        df_rpc_adv.loc[df_rpc_adv["team_norm"] == away_norm].iloc[0]
        if not df_rpc_adv[df_rpc_adv["team_norm"] == away_norm].empty
        else pd.Series()
    )

    # 8) Build headline stats
    headlines = [
        {"label": "Rest Advantage (Home)", "value": float(row.get("rest_advantage", MLB_DEFAULTS.get("mlb_rest_days", 0.0)))},
        {"label": "Form Win% Diff", "value": round(float(row.get("form_win_pct_diff", 0.0)), 3)},
        {"label": "Prev Season Win% Diff", "value": round(float(row.get("prev_season_win_pct_diff", 0.0)), 3)},
        {"label": f"H2H Home Win% (L{int(row.get('matchup_num_games',0))})", "value": round(float(row.get("matchup_home_win_pct", 0.0)), 3)},
    ]
    for h in headlines:
        if h["label"] == "Rest Advantage (Home)":
            h["value"] = int(h["value"])
        else:
            h["value"] = round(h["value"], 2)

    # 9) Build bar chart data
    if is_historical_game and (bar_rpc := sb_client.rpc("get_mlb_game_bar_data", {"p_game_id": game_id_str}).execute().data):
        bar_chart_data = bar_rpc[0].get("bar_chart_data", [])
    else:
        hf = home_adv_rpc_data.get("runs_for_avg_overall", MLB_DEFAULTS.get("mlb_avg_runs_for", 0.0))
        af = away_adv_rpc_data.get("runs_for_avg_overall", MLB_DEFAULTS.get("mlb_avg_runs_for", 0.0))
        ha = home_adv_rpc_data.get("runs_against_avg_overall", MLB_DEFAULTS.get("mlb_avg_runs_against", 0.0))
        aa = away_adv_rpc_data.get("runs_against_avg_overall", MLB_DEFAULTS.get("mlb_avg_runs_against", 0.0))
        bar_chart_data = [
            {"name": "Avg Runs For", "Home": round(float(hf), 2), "Away": round(float(af), 2)},
            {"name": "Avg Runs Against", "Home": round(float(ha), 2), "Away": round(float(aa), 2)},
        ]

    # 10) Fetch league ranges via RPC for radar
    ranges_rows = sb_client.rpc("get_mlb_metric_ranges", {"p_season": season_year}).execute().data or []
    league_ranges = {
        r["metric"]: {
            "min":    float(r["min_value"]),
            "max":    float(r["max_value"]),
            "invert": (r["metric"] == "Season Runs Against")
        }
        for r in ranges_rows
    }
    if not league_ranges:
        logger.warning("get_mlb_metric_ranges RPC empty; falling back to df_team_stats")
        league_ranges = {
            "Venue Win %": {
                "min": df_team_stats["wins_home_percentage"].min(),
                "max": df_team_stats["wins_home_percentage"].max(),
                "invert": False
            },
            "Season Runs Scored": {
                "min": df_team_stats["runs_for_avg_all"].min(),
                "max": df_team_stats["runs_for_avg_all"].max(),
                "invert": False
            },
            "Season Runs Against": {
                "min": df_team_stats["runs_against_avg_all"].min(),
                "max": df_team_stats["runs_against_avg_all"].max(),
                "invert": True
            },
            "Home/Away Win Advantage": {
                "min": (df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).min(),
                "max": (df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).max(),
                "invert": False
            },
        }

    # 11) Build radar payload
    radar_payload: List[Dict[str, Any]] = []
    for metric, cfg in radar_metrics_map.items():
        h = float(row.get(cfg["home_col"], 0.0))
        a = float(row.get(cfg["away_col"], 0.0))
        rng = league_ranges[metric]
        def scale(v: float) -> float:
            if rng["max"] == rng["min"]:
                pct = 50.0
            else:
                pct = 100.0 * (v - rng["min"]) / (rng["max"] - rng["min"])
            return 100.0 - pct if rng["invert"] else pct

        radar_payload.append({
            "metric":   metric,
            "home_raw": round(h, cfg["round"]),
            "away_raw": round(a, cfg["round"]),
            "home_idx": round(scale(h), 1),
            "away_idx": round(scale(a), 1),
        })
    logger.debug(f"radar_payload: {radar_payload}")

    # New Handedness Matchup Visualization Data
    pie_payload = []
    logger.debug("Building handedness matchup visualization payload...")

    home_hand_val = float(row.get("h_team_off_avg_runs_vs_opp_hand", 0.0))
    away_hand_val = float(row.get("a_team_off_avg_runs_vs_opp_hand", 0.0))

    # --- OPTION 1: Your Pie Chart Idea ---
    # This shows the PROPORTION of the advantage between the two teams.
    pie_payload = [
        {
            "category": f"Home Offense vs Opp. Hand ({round(home_hand_val, 2)} Runs)", 
            "value": home_hand_val, 
            "color": "#60a5fa" # Blue for Home
        },
        {
            "category": f"Away Offense vs Opp. Hand ({round(away_hand_val, 2)} Runs)",
            "value": away_hand_val, 
            "color": "#4ade80" # Green for Away
        },
    ]
    # The chart title in the frontend should be "Handedness Scoring Advantage"

    # --- OPTION 2: Recommended Bar Chart ---
    # For a clearer, direct comparison of the raw values.
    # To use this, you would change your frontend component from a Pie Chart to a Bar Chart.
    # The payload name `pie_chart_data` is kept for simplicity, but it would feed a bar chart.
    # UNCOMMENT THE BLOCK BELOW to use this version.
    """
    pie_payload = [
        {"name": "Home", "Runs vs Opp. Hand": round(home_hand_val, 2)},
        {"name": "Away", "Runs vs Opp. Hand": round(away_hand_val, 2)},
    ]
    # The chart title in the frontend should be "Offensive Runs vs. Opposing Pitcher Hand"
    """
    logger.debug(f"Handedness visualization payload: {pie_payload}")

    # 10) Upsert into Supabase
    snapshot_payload_final = {
        "game_id": game_id_str,
        "game_date": current_game_date_et.isoformat(),
        "season": str(season_year), # Use the integer season year
        "is_historical": is_historical_game,
        "headline_stats": headlines,
        "bar_chart_data": bar_chart_data,
        "radar_chart_data": radar_payload,
        "pie_chart_data": pie_payload,
        "last_updated": pd.Timestamp.utcnow().isoformat()
    }
    
    logger.info(f"Upserting MLB snapshot for game_id: {game_id_str}")
    upsert_response = sb_client.table("mlb_snapshots").upsert(snapshot_payload_final, on_conflict="game_id").execute()
    
    if hasattr(upsert_response, 'error') and upsert_response.error:
        logger.error(f"MLB Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
        logger.error(f"Supabase response: {upsert_response}")
    elif hasattr(upsert_response, 'data') and not upsert_response.data and not (hasattr(upsert_response, 'count') and upsert_response.count is not None and upsert_response.count > 0) :
        logger.warning(f"MLB Snapshot upsert for game_id={game_id_str} may have had an issue (no data/count returned). Response: {upsert_response}")
    else:
        logger.info(f"✅ MLB Snapshot upserted for game_id={game_id_str}")

# --- CLI entry point ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python backend/mlb_features/make_mlb_snapshots.py <game_id1> [<game_id2> ...]")
        logger.info("Example: python backend/mlb_features/make_mlb_snapshots.py YOUR_MLB_GAME_ID_HERE")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_mlb_snapshot(game_id_arg)
            except Exception as e_main:
                logger.error(f"CRITICAL ERROR processing game_id {game_id_arg} in snapshot generation: {e_main}", exc_info=True)