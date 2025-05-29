# backend/mlb_features/make_mlb_snapshots.py
"""
Generate and upsert per-game MLB feature snapshots for frontend display.
Fetches raw game data, historical context, computes features via engine,
and assembles payloads for headlines, bar, radar, and pie charts.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import pandas as pd
from supabase import create_client, Client

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
    raise RuntimeError("Supabase URL/key missing. Ensure backend.config is correct or env vars are set.")

from backend.mlb_features.engine import run_mlb_feature_pipeline
from backend.mlb_features.utils import normalize_team_name, DEFAULTS as MLB_DEFAULTS

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Supabase Client ---
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Fetch Helpers ---
def fetch_table(table_name: str, match_criteria: Dict[str, Any]) -> pd.DataFrame:
    logger.debug(f"Fetching from table '{table_name}' with criteria: {match_criteria}")
    try:
        response = sb_client.table(table_name).select("*").match(match_criteria).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Error fetching from table '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()

# --- Helper Function _get_mlb_team_form_string ---
def _get_mlb_team_form_string(
    team_id_to_check: Union[str, int],
    current_game_date_et: pd.Timestamp, # Expecting a date object or parsable string
    all_historical_games: pd.DataFrame,
    num_form_games: int = 5
) -> str:
    team_id_str = str(team_id_to_check)
    
    current_game_date_obj = pd.to_datetime(current_game_date_et, errors='coerce').date()

    if pd.isna(current_game_date_obj) or all_historical_games.empty:
        return "N/A"

    date_col_in_hist = None
    all_historical_games_copy = all_historical_games.copy() # Work on a copy

    if 'game_date_et' in all_historical_games_copy.columns:
        all_historical_games_copy['parsed_date_for_form'] = pd.to_datetime(all_historical_games_copy['game_date_et'], errors='coerce').dt.date
        date_col_in_hist = 'parsed_date_for_form'
    elif 'game_date_time_utc' in all_historical_games_copy.columns:
        all_historical_games_copy['parsed_date_for_form'] = pd.to_datetime(all_historical_games_copy['game_date_time_utc'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.date
        date_col_in_hist = 'parsed_date_for_form'
    else:
        logger.warning("No suitable date column found in all_historical_games for form string generation.")
        return "N/A"
    
    if date_col_in_hist not in all_historical_games_copy.columns:
        logger.error(f"Internal error: date_col_in_hist '{date_col_in_hist}' not found after parsing.")
        return "N/A"

    team_games = all_historical_games_copy[
        ((all_historical_games_copy['home_team_id'].astype(str) == team_id_str) | \
         (all_historical_games_copy['away_team_id'].astype(str) == team_id_str)) & \
        (all_historical_games_copy[date_col_in_hist] < current_game_date_obj) & \
        (all_historical_games_copy['status_short'].isin(['FT', 'F']) | all_historical_games_copy['status_long'] == 'Finished')
    ]

    if team_games.empty:
        return "N/A"

    recent_games = team_games.sort_values(by=date_col_in_hist, ascending=False).head(num_form_games)

    if recent_games.empty:
        return "N/A"

    form_results = []
    for _, game_row in recent_games.sort_values(by=date_col_in_hist, ascending=True).iterrows():
        is_home = str(game_row['home_team_id']) == team_id_str
        team_score = pd.to_numeric(game_row['home_score'] if is_home else game_row['away_score'], errors='coerce')
        opponent_score = pd.to_numeric(game_row['away_score'] if is_home else game_row['home_score'], errors='coerce')
        
        if pd.isna(team_score) or pd.isna(opponent_score):
            form_results.append("?")
        elif team_score > opponent_score:
            form_results.append("W")
        elif team_score < opponent_score:
            form_results.append("L")
        else:
            form_results.append("T") 

    return "".join(form_results) if form_results else "N/A"

# --- Snapshot Generator ---
def make_mlb_snapshot(
    game_id: Union[str, int],
    # Default column names used in this function that access df_game (from schedule/historical)
    # These can be overridden if df_game has different names.
    input_game_date_col: str = "game_date_et", # Primary date col in mlb_game_schedule
    input_game_date_utc_col: str = "game_date_time_utc", # Fallback date col, typically in mlb_historical_game_stats
    input_home_team_col: str = "home_team_id",
    input_away_team_col: str = "away_team_id",
    input_home_pitcher_hand_col: str = "home_probable_pitcher_handedness",
    input_away_pitcher_hand_col: str = "away_probable_pitcher_handedness",
    input_home_score_col: str = "home_score", # From mlb_historical_game_stats
    # ... add other input column name parameters if needed
    ):
    game_id_str = str(game_id)
    logger.info(f"--- Generating MLB Snapshot for game_id: {game_id_str} ---")

    df_game = fetch_table("mlb_historical_game_stats", {"game_id": game_id_str})
    is_post_game = not df_game.empty
    
    if not is_post_game:
        df_game = fetch_table("mlb_game_schedule", {"game_id": game_id_str})
    
    if df_game.empty:
        logger.error(f"No primary data found for game_id={game_id_str}. Cannot generate snapshot.")
        return

    current_game_date_et = None
    if input_game_date_col in df_game.columns and pd.notna(df_game[input_game_date_col].iloc[0]):
        current_game_date_et = pd.to_datetime(df_game[input_game_date_col].iloc[0], errors='coerce').date()
    elif input_game_date_utc_col in df_game.columns and pd.notna(df_game[input_game_date_utc_col].iloc[0]):
        try:
            current_game_date_et = pd.to_datetime(df_game[input_game_date_utc_col].iloc[0], errors='coerce').tz_localize('UTC').tz_convert('America/New_York').date()
        except Exception as e:
            logger.error(f"Error parsing {input_game_date_utc_col} for {game_id_str}: {e}")
    
    if current_game_date_et is None:
        logger.error(f"No valid game date found for game_id={game_id_str}. Cannot proceed.")
        return
    
    # Add a consistent date column name to df_game for the engine (engine uses 'game_date_et' by default)
    df_game['game_date_et'] = current_game_date_et # This is now a date object

    hist_resp = sb_client.table("mlb_historical_game_stats").select(
        "game_id, home_team_id, away_team_id, home_score, away_score, game_date_et, game_date_time_utc, status_short, status_long"
    ).execute()
    df_full_history = pd.DataFrame(hist_resp.data or [])

    team_stats_resp = sb_client.table("mlb_historical_team_stats").select("*").execute()
    df_team_stats = pd.DataFrame(team_stats_resp.data or [])

    season_year = current_game_date_et.year

    home_team_id_val = df_game[input_home_team_col].iloc[0] # Use parameterized col name
    away_team_id_val = df_game[input_away_team_col].iloc[0] # Use parameterized col name

    df_game["home_current_form"] = _get_mlb_team_form_string(
        home_team_id_val, current_game_date_et, df_full_history
    )
    df_game["away_current_form"] = _get_mlb_team_form_string(
        away_team_id_val, current_game_date_et, df_full_history
    )
    
    logger.info(f"Running feature pipeline for game {game_id_str}...")
    df_features = run_mlb_feature_pipeline(
        df_game.copy(), 
        mlb_historical_games_df=df_full_history,
        mlb_historical_team_stats_df=df_team_stats,
        season_to_lookup=season_year,
        form_home_col="home_current_form",
        form_away_col="away_current_form",
        # Pass through column name parameters to the engine if its modules need them
        # or ensure engine's defaults match what df_game now provides.
        # Engine call uses its own defaults if not specified, e.g., for game_date_col.
        debug=False
    )

    if df_features.empty:
        logger.error(f"Feature pipeline returned empty DataFrame for game_id={game_id_str}")
        return
    
    if len(df_features) != 1:
        logger.error(f"Feature pipeline did not return exactly one row for game_id={game_id_str}. Shape: {df_features.shape}")
        return
    row = df_features.iloc[0]

    logger.debug(f"Fetching season advanced stats from RPC for season {season_year} for game {game_id_str}...")
    rpc_call_name = "get_mlb_advanced_team_stats"
    rpc_data = sb_client.rpc(rpc_call_name, {"p_season": season_year}).execute().data or []
    df_rpc = pd.DataFrame(rpc_data)
    
    if not df_rpc.empty and 'team_id' in df_rpc.columns:
        df_rpc["team_norm"] = df_rpc["team_id"].apply(normalize_team_name)
    else:
        logger.warning(f"RPC '{rpc_call_name}' returned no data or no 'team_id' column for season {season_year}.")
        df_rpc = pd.DataFrame(columns=['team_id', 'team_norm']) # Ensure team_norm exists

    # Use the input parameter names for home/away team IDs from the original df_game/row
    home_norm = normalize_team_name(row[input_home_team_col])
    away_norm = normalize_team_name(row[input_away_team_col])
    
    home_adv_season_stats = df_rpc[df_rpc["team_norm"] == home_norm] if "team_norm" in df_rpc else pd.DataFrame()
    away_adv_season_stats = df_rpc[df_rpc["team_norm"] == away_norm] if "team_norm" in df_rpc else pd.DataFrame()

    logger.info(f"Building snapshot components for game {game_id_str}...")
    headlines = [
        {"label": "Rest Adv (Home)", "value": int(row.get("home_rest_days", 0) - row.get("away_rest_days", 0))},
        {"label": "Form Win% Diff", "value": round(float(row.get("form_win_pct_diff", 0.0)), 3)},
        {"label": "Prev Season Win% Diff", "value": round(float(row.get("prev_season_win_pct_diff", 0.0)), 3)},
        {"label": f"H2H Home Win% (L{int(row.get('h_matchup_num_games',0))})", "value": round(float(row.get("h_matchup_home_win_pct", 0.0)),3)},
        {"label": "Home Off. Runs vs Opp Hand", "value": round(float(row.get("h_team_off_avg_runs_vs_opp_hand", 0.0)), 2)},
        {"label": "Away Off. Runs vs Opp Hand", "value": round(float(row.get("a_team_off_avg_runs_vs_opp_hand", 0.0)), 2)},
    ]
    
    bar_data = []
    if pd.notna(row.get(input_home_score_col)): # Use parameterized input_home_score_col
        for i in range(1, 10):
            bar_data.append({
                "name": f"Inn {i}",
                "Home": int(row.get(f"h_inn_{i}", 0) or 0),
                "Away": int(row.get(f"a_inn_{i}", 0) or 0)
            })
    else: 
        home_r_for_key = "runs_for_avg_all" 
        home_r_allowed_key = "runs_against_avg_all"
        default_avg_runs = MLB_DEFAULTS.get('mlb_avg_runs_for', 0.0)

        home_r_for = float(home_adv_season_stats[home_r_for_key].iloc[0]) if not home_adv_season_stats.empty and home_r_for_key in home_adv_season_stats.columns and not home_adv_season_stats[home_r_for_key].empty and pd.notna(home_adv_season_stats[home_r_for_key].iloc[0]) else row.get("h_team_hist_HA_runs_for_avg", default_avg_runs)
        away_r_for = float(away_adv_season_stats[home_r_for_key].iloc[0]) if not away_adv_season_stats.empty and home_r_for_key in away_adv_season_stats.columns and not away_adv_season_stats[home_r_for_key].empty and pd.notna(away_adv_season_stats[home_r_for_key].iloc[0]) else row.get("a_team_hist_HA_runs_for_avg", default_avg_runs)
        
        home_r_against = float(home_adv_season_stats[home_r_allowed_key].iloc[0]) if not home_adv_season_stats.empty and home_r_allowed_key in home_adv_season_stats.columns and not home_adv_season_stats[home_r_allowed_key].empty and pd.notna(home_adv_season_stats[home_r_allowed_key].iloc[0]) else row.get("h_team_hist_HA_runs_against_avg", default_avg_runs)
        away_r_against = float(away_adv_season_stats[home_r_allowed_key].iloc[0]) if not away_adv_season_stats.empty and home_r_allowed_key in away_adv_season_stats.columns and not away_adv_season_stats[home_r_allowed_key].empty and pd.notna(away_adv_season_stats[home_r_allowed_key].iloc[0]) else row.get("a_team_hist_HA_runs_against_avg", default_avg_runs)

        bar_data = [
            {"name": "Avg Runs For", "Home": round(home_r_for,2), "Away": round(away_r_for,2)},
            {"name": "Avg Runs Against", "Home": round(home_r_against,2), "Away": round(away_r_against,2)},
        ]

    radar_payload = []
    radar_metrics_map = {
        "AVG": "team_batting_avg", "OBP": "team_on_base_pct", "SLG": "team_slugging_pct",
        "ERA": "team_era", "WHIP": "team_whip"
    }
    default_radar_values = {"AVG": .250, "OBP": .320, "SLG": .410, "ERA": 4.00, "WHIP": 1.30}

    for display_metric, rpc_col_name in radar_metrics_map.items():
        home_val = default_radar_values[display_metric]
        if not home_adv_season_stats.empty and rpc_col_name in home_adv_season_stats.columns and pd.notna(home_adv_season_stats[rpc_col_name].iloc[0]):
            home_val = float(home_adv_season_stats[rpc_col_name].iloc[0])
        
        away_val = default_radar_values[display_metric]
        if not away_adv_season_stats.empty and rpc_col_name in away_adv_season_stats.columns and pd.notna(away_adv_season_stats[rpc_col_name].iloc[0]):
            away_val = float(away_adv_season_stats[rpc_col_name].iloc[0])
            
        radar_payload.append({'metric': display_metric, 'home_value': round(home_val,3), 'away_value': round(away_val,3)})
        
    pie_payload = [
        {"category": f"Home Off. vs Opp Pitch Hand ({row.get(input_away_pitcher_hand_col,'?')})", # Use parameterized col name
         "value": round(float(row.get("h_team_off_avg_runs_vs_opp_hand", MLB_DEFAULTS.get('mlb_avg_runs_vs_hand',0.0))), 2)},
        {"category": "Home Off. Overall Avg Runs", 
         "value": round(home_r_for,2) 
        }
    ]

    snapshot_payload_final = {
        "game_id": game_id_str, "headline_stats": headlines, "bar_chart_data": bar_data,
        "radar_chart_data": radar_payload, "pie_chart_data": pie_payload,
        "last_updated": pd.Timestamp.utcnow().isoformat()
    }
    
    logger.info(f"Upserting MLB snapshot for game_id: {game_id_str} with payload: {snapshot_payload_final}")
    upsert_response = sb_client.table("mlb_snapshots").upsert(snapshot_payload_final, on_conflict="game_id").execute()
    
    if hasattr(upsert_response, 'error') and upsert_response.error:
        logger.error(f"MLB Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
    elif hasattr(upsert_response, 'data') and not upsert_response.data and not (hasattr(upsert_response, 'count') and upsert_response.count is not None and upsert_response.count > 0) :
        logger.warning(f"MLB Snapshot upsert for game_id={game_id_str} may have had an issue (no data/count returned). Response: {upsert_response}")
    else:
        logger.info(f"✅ MLB Snapshot upserted for game_id={game_id_str}")

# --- CLI entry point ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python backend/mlb_features/make_mlb_snapshots.py <game_id1> [<game_id2> ...]")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_mlb_snapshot(game_id_arg) # Call without specific col names, will use defaults in signature
            except Exception as e_main:
                logger.error(f"CRITICAL ERROR processing game_id {game_id_arg} in snapshot generation: {e_main}", exc_info=True)