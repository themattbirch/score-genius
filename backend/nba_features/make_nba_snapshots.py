# backend/nba_features/make_nba_snapshots.py
"""
Generate and upsert per-game NBA feature snapshots for frontend display.
Fetches raw game data, historical context, computes features via an NBA engine,
and assembles payloads for headlines, bar, radar, and pie charts.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import pandas as pd
from supabase import create_client, Client
from dateutil import parser as dateutil_parser # For robust date parsing

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
    raise RuntimeError("Supabase URL/key missing in make_nba_snapshots.py")

# Assume an nba_features.engine exists or will be created
from backend.nba_features.engine import run_nba_feature_pipeline
from backend.nba_features.utils import normalize_team_name as normalize_nba_team_name
from backend.nba_features.utils import DEFAULTS as NBA_DEFAULTS
from backend.nba_features.utils import determine_season as determine_nba_season # For NBA season string

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Supabase Client ---
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Fetch Helpers (from your existing NBA script) ---
# Using more specific names to avoid confusion if mlb_features.utils also has these.
def fetch_nba_raw_game_data(game_id: Union[str, int]) -> pd.DataFrame:
    """
    Return one-row DataFrame: nba_historical_game_stats if available, else nba_game_schedule.
    Ensures a 'game_date' column is present and parsed.
    """
    game_id_str = str(game_id)
    logger.debug(f"Fetching raw game data for NBA game_id: {game_id_str}")
    
    # Try historical stats (completed games)
    # Selecting all columns needed by advanced.py and for display
    # This select list should encompass all fields from nba_historical_game_stats used directly
    # or by the feature modules if df_game is the source for those raw stats.
    hist_cols_select = "*" # For simplicity, or list explicitly
    response = sb_client.table("nba_historical_game_stats").select(hist_cols_select).eq("game_id", game_id_str).execute()
    df = pd.DataFrame(response.data or [])

    if not df.empty:
        logger.debug(f"Found NBA game {game_id_str} in nba_historical_game_stats.")
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
        return df

    # Fallback to game schedule for pre-game info
    logger.debug(f"NBA Game {game_id_str} not in historical, trying nba_game_schedule.")
    # Select columns needed for pre-game snapshot and feature engine input
    schedule_cols_select = "game_id, game_date, scheduled_time_utc, home_team, away_team, home_team_id, away_team_id" # Add team_ids if available
    response2 = (
        sb_client.table("nba_game_schedule")
        .select(schedule_cols_select)
        .eq("game_id", game_id_str)
        .execute()
    )
    df2 = pd.DataFrame(response2.data or [])
    if not df2.empty:
        logger.debug(f"Found NBA game {game_id_str} in nba_game_schedule.")
        if 'game_date' in df2.columns:
            df2['game_date'] = pd.to_datetime(df2['game_date'], errors='coerce').dt.date
        # Ensure 'home_team_id' and 'away_team_id' are present if possible,
        # mapping from 'home_team' (name) if necessary.
        # For this example, we assume nba_game_schedule might have IDs or names that can be normalized.
    return df2

def fetch_nba_full_history() -> pd.DataFrame:
    """Entire historical game stats for H2H and rolling features for NBA."""
    logger.debug("Fetching full NBA historical game stats...")
    response = sb_client.table("nba_historical_game_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    logger.debug(f"Fetched {len(df)} rows from nba_historical_game_stats.")
    return df

def fetch_nba_team_season_stats() -> pd.DataFrame:
    """All teams' season stats from nba_historical_team_stats for NBA."""
    logger.debug("Fetching NBA historical team season stats...")
    response = sb_client.table("nba_historical_team_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    logger.debug(f"Fetched {len(df)} rows from nba_historical_team_stats.")
    return df

def fetch_nba_season_advanced_stats_rpc(season_identifier: str | int) -> pd.DataFrame: # season could be "2023-24" or 2023
    """Season-to-date advanced metrics for each NBA team via RPC."""
    rpc_name = "get_nba_advanced_team_stats" # This RPC needs to be defined/confirmed
    logger.debug(f"Fetching NBA season advanced stats for season '{season_identifier}' via RPC: {rpc_name}")
    try:
        # Adjust param name if your NBA RPC expects something different (e.g., p_season_str or p_start_year)
        response = sb_client.rpc(rpc_name, {"p_season_identifier": season_identifier}).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from RPC {rpc_name}.")
        # Ensure 'team_id' or a normalizable 'team_name' for matching
        return df
    except Exception as e:
        logger.error(f"Error calling RPC '{rpc_name}': {e}", exc_info=True)
        return pd.DataFrame()


# --- Form String Helper (NBA specific, using nba_historical_game_stats) ---
def _get_nba_team_form_string(
    team_identifier: str | int, # Can be team name or ID, normalized inside
    current_game_date: pd.Timestamp | str,
    all_historical_games: pd.DataFrame, # Full nba_historical_game_stats
    num_form_games: int = 5
) -> str:
    normalized_team = normalize_nba_team_name(str(team_identifier)) # Normalize the input team identifier
    
    current_game_dt = pd.to_datetime(current_game_date, errors='coerce').date()
    if pd.isna(current_game_dt) or all_historical_games.empty:
        return "N/A"

    # Create normalized team name columns in historical_games if they don't exist
    if 'home_team_norm' not in all_historical_games.columns:
        all_historical_games['home_team_norm'] = all_historical_games['home_team'].apply(normalize_nba_team_name)
    if 'away_team_norm' not in all_historical_games.columns:
        all_historical_games['away_team_norm'] = all_historical_games['away_team'].apply(normalize_nba_team_name)
    
    # Ensure date column is parsed
    if 'parsed_game_date' not in all_historical_games.columns:
        all_historical_games['parsed_game_date'] = pd.to_datetime(all_historical_games['game_date'], errors='coerce').dt.date


    team_games = all_historical_games[
        ((all_historical_games['home_team_norm'] == normalized_team) | \
         (all_historical_games['away_team_norm'] == normalized_team)) & \
        (all_historical_games['parsed_game_date'] < current_game_dt) & \
        (pd.notna(all_historical_games['home_score']) & pd.notna(all_historical_games['away_score'])) # Basic check for completed
    ].copy()

    if team_games.empty:
        return "N/A"

    recent_games = team_games.sort_values(by='parsed_game_date', ascending=False).head(num_form_games)
    if recent_games.empty: return "N/A"

    form_results = []
    for _, game_row in recent_games.sort_values(by='parsed_game_date', ascending=True).iterrows():
        is_home = game_row['home_team_norm'] == normalized_team
        team_score = int(game_row['home_score'] if is_home else game_row['away_score'])
        opponent_score = int(game_row['away_score'] if is_home else game_row['home_score'])
        
        if team_score > opponent_score: form_results.append("W")
        elif team_score < opponent_score: form_results.append("L")
        else: form_results.append("T")
    return "".join(form_results) if form_results else "N/A"


# --- NBA Snapshot Generator ---
def make_nba_snapshot(game_id: Union[str, int]):
    game_id_str = str(game_id)
    logger.info(f"--- Generating NBA Snapshot for game_id: {game_id_str} ---")

    # 1) Load raw game data
    df_game = fetch_nba_raw_game_data(game_id_str)
    if df_game.empty:
        logger.error(f"No raw data found for NBA game_id {game_id_str}.")
        return

    current_game_dt_obj = df_game['game_date'].iloc[0] # Already a date object from fetcher
    if pd.isna(current_game_dt_obj):
        logger.error(f"Invalid game_date for NBA game_id {game_id_str}.")
        return

    # Determine NBA season (e.g., "2023-24" or start year 2023)
    # Pass the full Timestamp if determine_nba_season expects it, or just the date part
    nba_season_identifier = determine_nba_season(pd.to_datetime(current_game_dt_obj)) # `determine_nba_season` from utils

    # 2) Fetch historical context
    df_full_history = fetch_nba_full_history()
    df_historical_team_stats = fetch_nba_team_season_stats()

    # 3) Generate Form Strings and add to df_game
    # Ensure home_team/away_team are the correct keys for team names/IDs in df_game
    # The fetch_nba_raw_game_data provides 'home_team', 'away_team' (likely names)
    df_game["home_current_form"] = _get_nba_team_form_string(
        df_game["home_team"].iloc[0], current_game_dt_obj, df_full_history
    )
    df_game["away_current_form"] = _get_nba_team_form_string(
        df_game["away_team"].iloc[0], current_game_dt_obj, df_full_history
    )

    # 4) Compute features via NBA engine
    logger.info(f"Running NBA feature pipeline for game {game_id_str}...")
    # The engine needs to be designed for NBA and use NBA column names
    # For now, let's assume its signature is similar to MLB, but it calls NBA modules.
    df_features = run_nba_feature_pipeline( # This function would need to be defined in nba_features.engine
        df_game.copy(), # Pass a copy
        nba_historical_games_df=df_full_history,
        nba_historical_team_stats_df=df_historical_team_stats,
        season_to_lookup=nba_season_identifier, # Or just the start year if season.py expects int
        form_home_col="home_current_form",
        form_away_col="away_current_form",
        # Add other relevant params for NBA engine: rolling_windows, h2h_max_games, etc.
        debug=False
    )
    if df_features.empty:
        logger.error(f"NBA Feature pipeline returned empty for game_id={game_id_str}")
        return
    if len(df_features) != 1:
        logger.error(f"NBA Feature pipeline did not return 1 row for game_id={game_id_str}")
        return
    row = df_features.iloc[0] # Assuming single game processing

    # 5) Fetch season-to-date advanced stats from NBA RPC
    logger.debug(f"Fetching NBA season advanced stats from RPC for season {nba_season_identifier}...")
    df_adv_rpc_data = fetch_nba_season_advanced_stats_rpc(nba_season_identifier)
    if not df_adv_rpc_data.empty and ('team_id' in df_adv_rpc_data.columns or 'team_name' in df_adv_rpc_data.columns):
        # Assuming RPC returns 'team_name', normalize it
        rpc_team_key = 'team_name' if 'team_name' in df_adv_rpc_data.columns else 'team_id'
        df_adv_rpc_data["team_norm"] = df_adv_rpc_data[rpc_team_key].apply(normalize_nba_team_name)
    else:
        logger.warning(f"RPC 'get_nba_advanced_team_stats' returned no data or no team identifier for season {nba_season_identifier}.")
        df_adv_rpc_data = pd.DataFrame(columns=[rpc_team_key, 'team_norm'] if 'rpc_team_key' in locals() else ['team_norm'])


    # Normalize team names/IDs from the current game row
    # Ensure 'home_team' and 'away_team' are the correct column names in df_features/row
    current_game_home_team_norm = normalize_nba_team_name(row.get("home_team"))
    current_game_away_team_norm = normalize_nba_team_name(row.get("away_team"))

    home_adv_season_stats = df_adv_rpc_data[df_adv_rpc_data["team_norm"] == current_game_home_team_norm] if "team_norm" in df_adv_rpc_data else pd.DataFrame()
    away_adv_season_stats = df_adv_rpc_data[df_adv_rpc_data["team_norm"] == current_game_away_team_norm] if "team_norm" in df_adv_rpc_data else pd.DataFrame()

    # 6) Build snapshot components (using NBA metrics from `row` and `*_adv_season_stats`)
    logger.info(f"Building NBA snapshot components for game {game_id_str}...")
    
    # Headline: Similar structure to MLB, but use NBA feature names from `row`
    headline = [
        {'label': 'Rest Advantage (Home)', 'value': int(row.get("rest_advantage", 0))}, # from rest.py
        {'label': 'Form Momentum Diff',    'value': round(float(row.get("momentum_diff", 0.0)), 2)}, # from form.py
        {'label': 'Form Win % Diff',       'value': round(float(row.get("form_win_pct_diff", 0.0)), 3)}, # from form.py
        {'label': 'Season Win % Diff',     'value': round(float(row.get("season_win_pct_diff", 0.0)), 3)}, # from season.py
        # Efficiency differential (Net Rating Diff) will come from advanced stats
        # If advanced.py (new version) adds hist_net_rtg_home/away:
        # {'label': 'Hist. Net Rating Diff', 'value': round(float(row.get("h_team_hist_net_rtg_home",0) - row.get("a_team_hist_net_rtg_away",0)),1)},
        # Or, if using the RPC data for current season net rating:
        {'label': 'Season NetRtg Diff (H-A)', 
         'value': round(float(home_adv_season_stats["net_rating"].iloc[0] if not home_adv_season_stats.empty and "net_rating" in home_adv_season_stats else 0) - \
                        float(away_adv_season_stats["net_rating"].iloc[0] if not away_adv_season_stats.empty and "net_rating" in away_adv_season_stats else 0), 1)
        },
    ]

    # Bar: Home quarter points (from `row`, which originates from `df_game` if post-game)
    bar = []
    if pd.notna(row.get("home_score")): # Post-game
        for i in (1, 2, 3, 4):
            bar.append({'name': f'Q{i}', 'value': int(row.get(f'home_q{i}', 0) or 0)})
        if pd.notna(row.get('home_ot')) and int(row.get('home_ot', 0) > 0): # Check if home_ot exists and is > 0
             bar.append({'name': 'OT', 'value': int(row.get('home_ot',0))})
    else: # Pre-game, use avg points for from RPC or historical
        home_pts_key = "avg_pts_for" # Assumed key from your get_nba_advanced_team_stats RPC or historical_team_stats
        home_val = float(home_adv_season_stats[home_pts_key].iloc[0]) if not home_adv_season_stats.empty and home_pts_key in home_adv_season_stats else row.get("home_prev_season_avg_pts_for", NBA_DEFAULTS.get('avg_pts_for',0))
        bar = [{'name': 'Avg Pts For', 'value': round(home_val,1)}]


    # Radar: Advanced five-metric spider (from RPC data: home_adv_season_stats)
    # Ensure your get_nba_advanced_team_stats RPC returns these metrics.
    radar_map = {
        "Pace": "pace", "OffRtg": "off_rtg", "DefRtg": "def_rtg", 
        "eFG%": "efg_pct", "TOV%": "tov_pct" # tov_pct might be tov_rate
    }
    default_radar_nba = {"Pace": 100.0, "OffRtg": 115.0, "DefRtg": 115.0, "eFG%": 0.53, "TOV%": 13.5}
    radar_home = []
    radar_away = []

    for display_name, rpc_key in radar_map.items():
        home_val = float(home_adv_season_stats[rpc_key].iloc[0]) if not home_adv_season_stats.empty and rpc_key in home_adv_season_stats else default_radar_nba[display_name]
        away_val = float(away_adv_season_stats[rpc_key].iloc[0]) if not away_adv_season_stats.empty and rpc_key in away_adv_season_stats else default_radar_nba[display_name]
        radar_home.append({'metric': display_name, 'value': round(home_val, 2 if display_name not in ["Pace", "OffRtg", "DefRtg"] else 1)})
        radar_away.append({'metric': display_name, 'value': round(away_val, 2 if display_name not in ["Pace", "OffRtg", "DefRtg"] else 1)})
    
    # Consolidate radar data if your frontend expects one array with A and B values per metric
    radar_payload = []
    for i, (display_name, _) in enumerate(radar_map.items()):
        radar_payload.append({
            'metric': display_name,
            'home_value': radar_home[i]['value'],
            'away_value': radar_away[i]['value']
        })

    # Pie: Shot distribution (2P, 3P, FT) for Home team
    # These come from df_features (row), which should get them if df_game was post-game
    # and nba_advanced.py (original version) calculated them, or if new advanced.py attaches them.
    # If using the *original* advanced.py that calculates from single game:
    home_fgm = float(row.get('home_fg_made', 0))
    home_3pm = float(row.get('home_3pm', 0))
    home_ftm = float(row.get('home_ft_made', 0))

    # If using *new* advanced.py that fetches historical/seasonal shooting %:
    # This would require your RPC to provide something like 'pct_2p_shots', 'pct_3p_shots', 'pct_ft_points'
    # For now, let's assume it's from single game stats if available (post-game)
    pie = []
    if pd.notna(row.get("home_score")): # Post-game
        twop_made = home_fgm - home_3pm
        pie = [
            {'category': '2P FG Made', 'value': int(twop_made),  'color': '#4ade80'},
            {'category': '3P FG Made', 'value': int(home_3pm), 'color': '#60a5fa'},
            {'category': 'FT Made',    'value': int(home_ftm),    'color': '#fbbf24'},
        ]
    else: # Pre-game, maybe show team's typical shooting distribution from RPC? (e.g. % of points from 2s, 3s, FTs)
        # This requires RPC to provide such data. For now, empty or different pie for pre-game.
        pie = [{"category": "Pre-Game Distribution N/A", "value": 1}]


    # 7) Upsert into Supabase
    snapshot_payload = {
        'game_id':        game_id_str,
        'headline_stats': headline,
        'bar_chart_data': bar, # Renamed from bar_data
        'radar_chart_data': radar_payload, # Renamed from radar
        'pie_chart_data': pie, # Renamed from pie
        'last_updated': pd.Timestamp.utcnow().isoformat()
    }

    logger.info(f"Upserting NBA snapshot for game_id: {game_id_str} with payload: {snapshot_payload}")
    upsert_response = sb_client.table('nba_snapshots').upsert(snapshot_payload, on_conflict='game_id').execute()

    if hasattr(upsert_response, 'error') and upsert_response.error:
        logger.error(f"NBA Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
    elif hasattr(upsert_response, 'data') and not upsert_response.data and not (hasattr(upsert_response, 'count') and upsert_response.count is not None and upsert_response.count > 0) :
        logger.warning(f"NBA Snapshot upsert for game_id={game_id_str} may have had an issue (no data/count returned). Response: {upsert_response}")
    else:
        logger.info(f"✅ NBA Snapshot upserted for game_id={game_id_str}")


# --- CLI entry point ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python backend/nba_features/make_nba_snapshots.py <game_id1> [<game_id2> ...]")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_nba_snapshot(game_id_arg)
            except Exception as e_main:
                logger.error(f"CRITICAL ERROR processing NBA game_id {game_id_arg} in snapshot generation: {e_main}", exc_info=True)