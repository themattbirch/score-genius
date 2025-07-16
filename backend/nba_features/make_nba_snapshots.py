# backend/nba_features/make_nba_snapshots.py

"""
Generate and upsert per-game NBA feature snapshots for frontend display.
Fetches raw game data, historical context, computes features via an NBA engine,
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
    raise RuntimeError("Supabase URL/key missing in make_nba_snapshots.py")

# Assume an nba_features.engine exists or will be created
try:
    # Renamed to run_nba_feature_pipeline to avoid potential conflict with MLB's run_feature_pipeline
    from backend.nba_features.engine import run_feature_pipeline as run_nba_feature_pipeline
    logger.info("Successfully imported run_nba_feature_pipeline from engine.")
except ImportError:
    logger.error("Could not import run_feature_pipeline from backend.nba_features.engine.")
    logger.error("Please ensure backend/nba_features/engine.py exists and defines run_feature_pipeline.")
    sys.exit(1) # Exit if the core engine isn't available

from backend.nba_features.utils import normalize_team_name as normalize_nba_team_name
from backend.nba_features.utils import DEFAULTS as NBA_DEFAULTS
from backend.nba_features.utils import determine_season as determine_nba_season # For NBA season string

# --- Supabase Client ---
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Fetch Helpers ---
def fetch_nba_raw_game_data(game_id: Union[str, int]) -> pd.DataFrame:
    """
    Return one-row DataFrame: nba_historical_game_stats if available, else nba_game_schedule.
    Ensures a 'game_date' column is present and parsed.
    """
    # Normalize ID to int when possible (your column is int8)
    game_id_str = str(game_id)
    try:
        game_id_val = int(game_id_str)
    except ValueError:
        game_id_val = game_id_str

    logger.info(f"[RAW DATA] Fetching nba_historical_game_stats where game_id == {game_id_val!r}")
    hist_response = (
        sb_client
        .table("nba_historical_game_stats")
        .select("*")
        .filter("game_id", "eq", game_id_val)
        .execute()
    )
    logger.info(f"[RAW DATA] → historical rows returned: {len(hist_response.data or [])}")

    df = pd.DataFrame(hist_response.data or [])
    if not df.empty:
        logger.info(f"[RAW DATA] Historical columns: {list(df.columns)[:10]}…")
        # Parse game_date and flag
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
        df['is_historical_game'] = True
        return df

    # Fallback to game schedule for pre-game info
    logger.info(f"[RAW DATA] No historical row for {game_id_val!r}; falling back to schedule")
    schedule_response = (
        sb_client
        .table("nba_game_schedule")
        .select("game_id, game_date, scheduled_time, home_team, away_team")
        .eq("game_id", game_id_str)
        .execute()
    )
    df2 = pd.DataFrame(schedule_response.data or [])
    if not df2.empty:
        logger.info(f"[RAW DATA] Schedule row found; marking as pre-game")
        if 'game_date' in df2.columns:
            df2['game_date'] = pd.to_datetime(df2['game_date'], errors='coerce').dt.date
        df2['is_historical_game'] = False
        # Add any missing pre-game columns
        for col in [
            'home_score', 'away_score',
            'home_q1','home_q2','home_q3','home_q4','home_ot',
            'home_fg_made','home_3pm','home_ft_made',
            'home_fg_attempted','home_3pa','home_ft_attempted',
            'home_off_reb','home_def_reb','home_total_reb','home_turnovers',
            'away_fg_made','away_3pm','away_ft_made',
            'away_fg_attempted','away_3pa','away_ft_attempted',
            'away_off_reb','away_def_reb','away_total_reb','away_turnovers',
        ]:
            if col not in df2.columns:
                df2[col] = pd.NA
    return df2

def fetch_nba_full_history() -> pd.DataFrame:
    """Entire historical game stats for H2H and rolling features for NBA."""
    logger.debug("Fetching full NBA historical game stats...")
    response = sb_client.table("nba_historical_game_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    # Ensure game_date is parsed for form strings and filtering
    if 'game_date' in df.columns:
        df['parsed_game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
    logger.debug(f"Fetched {len(df)} rows from nba_historical_game_stats.")
    return df

def fetch_nba_team_season_stats() -> pd.DataFrame:
    """All teams' season stats from nba_historical_team_stats for NBA."""
    logger.debug("Fetching NBA historical team season stats...")
    response = sb_client.table("nba_historical_team_stats").select("*").execute()
    df = pd.DataFrame(response.data or [])
    logger.debug(f"Fetched {len(df)} rows from nba_historical_team_stats.")
    return df

def fetch_nba_season_advanced_stats_rpc(season_identifier: Union[str, int]) -> pd.DataFrame:
    """Season-to-date advanced metrics for each NBA team via RPC."""
    rpc_name = "get_nba_advanced_team_stats"
    logger.debug(f"Fetching NBA season advanced stats for season '{season_identifier}' via RPC: {rpc_name}")
    try:
        response = sb_client.rpc(rpc_name, {"p_season_year": season_identifier}).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from RPC {rpc_name}.")
        if not df.empty and ('team_name' in df.columns):
            df["team_norm"] = df["team_name"].apply(normalize_nba_team_name)
        elif not df.empty and 'team_id' in df.columns: # Fallback if RPC returns team_id instead of name
             logger.warning(f"RPC {rpc_name} returned 'team_id' instead of 'team_name'. This might require a team_id to name mapping.")
             # For now, if no team_name, try to map team_id if a robust mapping is available globally,
             # or handle 'team_id' directly if advanced.py expects it.
             # For this script, we'll assume `team_name` is preferred/used for normalization.
        return df
    except Exception as e:
        logger.error(f"Error calling RPC '{rpc_name}': {e}", exc_info=True)
        return pd.DataFrame()


# --- Form String Helper (NBA specific, using nba_historical_game_stats) ---
def _get_nba_team_form_string(
    team_identifier: Union[str, int],
    current_game_date: pd.Timestamp | str,
    all_historical_games: pd.DataFrame,
    num_form_games: int = 5
) -> str:
    normalized_team = normalize_nba_team_name(str(team_identifier))

    current_game_dt = pd.to_datetime(current_game_date, errors='coerce').date()
    if pd.isna(current_game_dt) or all_historical_games.empty:
        logger.debug(f"Cannot get form for {team_identifier}: invalid date or no historical games.")
        return "N/A"

    if 'home_team_norm' not in all_historical_games.columns:
        all_historical_games['home_team_norm'] = all_historical_games['home_team'].apply(normalize_nba_team_name)
    if 'away_team_norm' not in all_historical_games.columns:
        all_historical_games['away_team_norm'] = all_historical_games['away_team'].apply(normalize_nba_team_name)
    if 'parsed_game_date' not in all_historical_games.columns:
        all_historical_games['parsed_game_date'] = pd.to_datetime(all_historical_games['game_date'], errors='coerce').dt.date

    team_games = all_historical_games[
        ((all_historical_games['home_team_norm'] == normalized_team) | \
         (all_historical_games['away_team_norm'] == normalized_team)) & \
        (all_historical_games['parsed_game_date'] < current_game_dt) & \
        (pd.notna(all_historical_games['home_score']) & pd.notna(all_historical_games['away_score']))
    ].copy()

    if team_games.empty:
        logger.debug(f"No completed games found for {team_identifier} before {current_game_dt}.")
        return "N/A"

    recent_games = team_games.sort_values(by='parsed_game_date', ascending=False).head(num_form_games)
    if recent_games.empty:
        logger.debug(f"Not enough recent completed games for {team_identifier} before {current_game_dt}.")
        return "N/A"

    form_results = []
    for _, game_row in recent_games.sort_values(by='parsed_game_date', ascending=True).iterrows():
        is_home = game_row['home_team_norm'] == normalized_team
        team_score = int(game_row['home_score'] if is_home else game_row['away_score'])
        opponent_score = int(game_row['away_score'] if is_home else game_row['home_score'])

        if team_score > opponent_score: form_results.append("W")
        elif team_score < opponent_score: form_results.append("L")
        else: form_results.append("T")
    return "".join(form_results) if form_results else "N/A"

def _get_season_form_string(
    team_identifier: Union[str, int],
    current_game_date: pd.Timestamp | str,
    all_historical_games: pd.DataFrame,
    season_year: int,
    num_form_games: int = 5,
) -> str:
    normalized = normalize_nba_team_name(str(team_identifier))
    cur_dt = pd.to_datetime(current_game_date, errors='coerce').date()
    if pd.isna(cur_dt) or all_historical_games.empty:
        return "N/A"

    # ensure norm/date columns
    if 'home_team_norm' not in all_historical_games:
        all_historical_games['home_team_norm'] = all_historical_games['home_team'].apply(normalize_nba_team_name)
        all_historical_games['away_team_norm'] = all_historical_games['away_team'].apply(normalize_nba_team_name)
        all_historical_games['parsed_game_date'] = pd.to_datetime(
            all_historical_games['game_date'], errors='coerce'
        ).dt.date

    # filter to this season only
    all_historical_games['season_start'] = pd.to_datetime(
        all_historical_games['season'].str.split('-').str[0], errors='coerce'
    ).dt.year

    mask = (
        (all_historical_games['season_start'] == season_year) &
        (all_historical_games['parsed_game_date'] < cur_dt) &
        (
            (all_historical_games['home_team_norm'] == normalized) |
            (all_historical_games['away_team_norm'] == normalized)
        ) &
        pd.notna(all_historical_games['home_score']) &
        pd.notna(all_historical_games['away_score'])
    )
    team_games = all_historical_games.loc[mask].copy()
    if team_games.empty:
        return "N/A"

    recent = team_games.sort_values('parsed_game_date', ascending=False).head(num_form_games)
    if recent.empty:
        return "N/A"

    results = []
    for _, g in recent.sort_values('parsed_game_date').iterrows():
        is_home = g['home_team_norm'] == normalized
        ts = int(g['home_score'] if is_home else g['away_score'])
        os = int(g['away_score'] if is_home else g['home_score'])
        results.append("W" if ts > os else "L" if ts < os else "T")
    return "".join(results)

# --- NBA Snapshot Generator ---
def make_nba_snapshot(
    game_id: Union[str, int],
    # Default column names used in this function that access df_game (from schedule/historical)
    input_game_date_col: str = "game_date", # Primary date col in nba_game_schedule
    input_home_team_col: str = "home_team",
    input_away_team_col: str = "away_team",
    input_home_score_col: str = "home_score", # From nba_historical_game_stats
    input_away_score_col: str = "away_score",
    ):
    game_id_str = str(game_id)
    logger.info(f"--- Generating NBA Snapshot for game_id: {game_id_str} ---")

    # 1) Load raw game data
    df_game = fetch_nba_raw_game_data(game_id_str)
    if df_game.empty:
        logger.error(f"No raw data found for NBA game_id {game_id_str}. Cannot generate snapshot.")
        return

    is_historical_game = df_game['is_historical_game'].iloc[0]

    current_game_dt_obj = df_game['game_date'].iloc[0]
    if pd.isna(current_game_dt_obj):
        logger.error(f"Invalid game_date for NBA game_id {game_id_str}. Cannot proceed.")
        return

    # Determine NBA season (e.g., "2023-24" or start year 2023)
    # FIX: Pass integer start year to RPC as it expects `p_season_year INT`
    nba_season_full_str = determine_nba_season(pd.to_datetime(current_game_dt_obj))
    nba_season_start_year = int(nba_season_full_str.split('-')[0]) # Extract '2023' from '2023-24'
    logger.debug(f"Determined NBA season string: {nba_season_full_str}, start year for RPC: {nba_season_start_year}")


    # 2) Fetch historical context
    df_full_history = fetch_nba_full_history()
    df_historical_team_stats = fetch_nba_team_season_stats()

    

    # 3) Populate in-df_game form strings
    df_game["home_current_form"] = _get_season_form_string(
        df_game[input_home_team_col].iloc[0],
        current_game_dt_obj,
        df_full_history,
        nba_season_start_year,         # ← pass season here
    )
    df_game["away_current_form"] = _get_season_form_string(
        df_game[input_away_team_col].iloc[0],
        current_game_dt_obj,
        df_full_history,
        nba_season_start_year,
    )

    def _per_game_avg(df: pd.DataFrame, team: str, fg_col: str, three_col: str, ft_col: str) -> tuple[float,float]:
        """
        Season-to-date averages *before the current game date*
        for:   2-P made  (= FG – 3P)   and   FT made.
        df       : full historical games for the season
        team     : canonical team string (normed)
        Returns  : (avg_2pm, avg_ftm)
        """
        # home rows for team
        h = df.loc[df["home_team_norm"] == team, ["home_fg_made",
                                                "home_3pm",
                                                "home_ft_made"]].copy()
        # away rows for team
        a = df.loc[df["away_team_norm"] == team, ["away_fg_made",
                                                "away_3pm",
                                                "away_ft_made"]].copy()

        h.columns = a.columns = ["fg", "tp", "ft"]          # unify names
        allg      = pd.concat([h, a], ignore_index=True)

        if allg.empty:
            return 0.0, 0.0

        allg["2pm"] = allg["fg"] - allg["tp"]
        return float(allg["2pm"].mean()), float(allg["ft"].mean())
    # ------------------------------------------------------------

    # ---------- per-game helper -------------------------------------------------
    def _avg_2pm_ftm(df_season: pd.DataFrame, team_norm: str) -> tuple[float, float]:
        """
        Season-to-date averages (before the snapshot date) of
        2-point FG made and FT made, for the given team.
        """
        h = df_season[df_season["home_team_norm"] == team_norm]
        a = df_season[df_season["away_team_norm"] == team_norm]

        # unify column names
        h = h.rename(columns={"home_fg_made":"fg",
                            "home_3pm":"tp",
                            "home_ft_made":"ft"})
        a = a.rename(columns={"away_fg_made":"fg",
                            "away_3pm":"tp",
                            "away_ft_made":"ft"})

        all_rows = pd.concat([h[["fg","tp","ft"]], a[["fg","tp","ft"]]], ignore_index=True)
        if all_rows.empty:
            return 0.0, 0.0

        all_rows["2pm"] = all_rows["fg"] - all_rows["tp"]
        return float(all_rows["2pm"].mean()), float(all_rows["ft"].mean())
    # ----------------------------------------------------------------------------


    # 4) Manually compute win‐% diff
    home_id = df_game[input_home_team_col].iloc[0]
    away_id = df_game[input_away_team_col].iloc[0]

    home_form = _get_season_form_string(home_id, current_game_dt_obj, df_full_history, nba_season_start_year)
    away_form = _get_season_form_string(away_id, current_game_dt_obj, df_full_history, nba_season_start_year)
    logger.debug(f"RAW FORM STRINGS → Home: {home_form}, Away: {away_form}")

    def win_pct(form_str: str) -> float:
        if form_str in ("", "N/A"):
            return 0.0
        return form_str.count("W") / len(form_str)

    manual_form_diff = round((win_pct(home_form) - win_pct(away_form)) * 100, 2)
    logger.debug(f"MANUAL_FORM_DIFF = {manual_form_diff}%")

    # 4) Compute features via NBA engine
    logger.info(f"Running NBA feature pipeline for game {game_id_str}...")
    # Passing df_game (which includes form strings and potentially quarter scores if historical)
    df_features = run_nba_feature_pipeline(
        df_game.copy(), # Pass a copy to avoid modifying original df_game in engine
        db_conn=sb_client, # Pass Supabase client for internal fetching by engine if needed
        adv_splits_lookup_offset=0, # Look up advanced stats for the current season (2024 in this case)
        debug=True # Set to True for initial debugging
    )
    if df_features.empty:
        logger.error(f"NBA Feature pipeline returned empty for game_id={game_id_str}")
        return
    if len(df_features) != 1:
        logger.error(f"NBA Feature pipeline did not return 1 row for game_id={game_id_str}. Got {len(df_features)} rows.")
        if len(df_features) > 1:
            logger.warning("Proceeding with the first row from feature pipeline output.")
    row = df_features.iloc[0] # Assuming single game processing

    # --- ADD THESE LOGS TO INSPECT FEATURE VALUES ---
    logger.info(f"Raw feature values from feature pipeline for game {game_id_str}:")
    logger.info(f"  rest_advantage: {row.get('rest_advantage')}")
    logger.info(f"  form_win_pct_diff: {row.get('form_win_pct_diff')}")
    logger.info(f"  momentum_diff: {row.get('momentum_diff')}")
    logger.info(f"  season_win_pct_diff: {row.get('season_win_pct_diff')}")
    logger.info(f"  season_net_rating_diff: {row.get('season_net_rating_diff')}")
    # --- END ADD ---


    # 5) Fetch season-to-date advanced stats from NBA RPC (for radar/pre-game bar)
    logger.debug(f"Fetching NBA season advanced stats from RPC for season {nba_season_start_year}...")
    # FIX: Use nba_season_start_year (integer) for the RPC call
    df_adv_rpc_data = fetch_nba_season_advanced_stats_rpc(nba_season_start_year)

    # Ensure RPC data has 'team_norm' column for easy lookup
    if not df_adv_rpc_data.empty and 'team_name' in df_adv_rpc_data.columns:
        df_adv_rpc_data["team_norm"] = df_adv_rpc_data["team_name"].apply(normalize_nba_team_name)
    else:
        logger.warning(f"RPC 'get_nba_advanced_team_stats' returned no data or no 'team_name' column for season {nba_season_start_year}.")
        df_adv_rpc_data = pd.DataFrame(columns=['team_norm']) # Empty df with expected column

    # Normalize team names from the current game row for matching
    home_team_original = df_game[input_home_team_col].iloc[0]
    away_team_original = df_game[input_away_team_col].iloc[0]
    current_game_home_team_norm = normalize_nba_team_name(home_team_original)
    current_game_away_team_norm = normalize_nba_team_name(away_team_original)

    home_adv_season_stats = df_adv_rpc_data[df_adv_rpc_data["team_norm"] == current_game_home_team_norm]
    away_adv_season_stats = df_adv_rpc_data[df_adv_rpc_data["team_norm"] == current_game_away_team_norm]

    # Ensure single row for advanced stats if multiple teams matched (unlikely but safe)
    home_adv_season_stats = home_adv_season_stats.iloc[0] if not home_adv_season_stats.empty else pd.Series()
    away_adv_season_stats = away_adv_season_stats.iloc[0] if not away_adv_season_stats.empty else pd.Series()

    # 6) Build snapshot components
    logger.info(f"Building NBA snapshot components for game {game_id_str}...")

    # Compute rest days for each team (make sure your feature pipeline populates these)
    rest_home = int(row.get("rest_days_home", 0))
    rest_away = int(row.get("rest_days_away", 0))
    logger.debug(f"DEBUG REST: rest_days_home={rest_home}, rest_days_away={rest_away}")

    # Extract and debug Form Win % Diff
    form_win_pct_diff = float(row.get("form_win_pct_diff", 0.0))
    logger.debug(f"DEBUG_FORM: form_win_pct_diff raw = {form_win_pct_diff}")
    if "form_win_pct_diff" not in df_features.columns:
        logger.warning("form_win_pct_diff missing from feature pipeline output")
    else:
        logger.debug(
            f"DEBUG_FORM: df_features form_win_pct_diff = "
            f"{df_features['form_win_pct_diff'].iloc[0]}"
        )

    # Assemble your headlines list
    headlines = [
        {"label": "Rest Days (Home)",  "value": rest_home},
        {"label": "Rest Days (Away)",  "value": rest_away},
        {"label": "Recent Form Margin", "value": float(row.get("momentum_diff", 0.0))},
        {"label": "Form Win % Diff",    "value": manual_form_diff},
        {"label": "Season Winning % Diff",  "value": float(row.get("season_win_pct_diff", 0.0))},
        {"label": "Net Rating Diff (Home/Away)",
        "value": round(
            float(
                row.get(
                    "season_net_rating_diff",
                    home_adv_season_stats.get("net_rating", 0)
                    - away_adv_season_stats.get("net_rating", 0),
                )
            ),
            1,
        ),
        },
    ]
    # Rounding headlines to 1 decimal where appropriate
    for item in headlines:
        if isinstance(item['value'], (float, int)):
            # FIX: The labels here now match the new names shown in your screenshot.
            item['value'] = round(item['value'], 2 if item['label'] in ['Form Win % Diff', 'Season Winning % Diff'] else 1)
            # Adjust rounding for Rest Advantage if it's always an integer
            if item['label'] == 'Rest Advantage (Home)':
                item['value'] = int(item['value'])

    # Bar: Quarter-by-quarter Home vs Away points
    bar_chart_data = []
    if is_historical_game and pd.notna(row.get(input_home_score_col)):
        for q in range(1, 5):
            home_pts = int(row.get(f'home_q{q}', 0) or 0)
            away_pts = int(row.get(f'away_q{q}', 0) or 0)
            bar_chart_data.append({
                'category': f'Q{q}',
                'Home': home_pts,
                'Away': away_pts,
            })
        # Include OT if either team scored
        ot_home = int(row.get('home_ot', 0) or 0)
        ot_away = int(row.get('away_ot', 0) or 0)
        if ot_home > 0 or ot_away > 0:
            bar_chart_data.append({
                'category': 'OT',
                'Home': ot_home,
                'Away': ot_away,
            })
        logger.debug(f"Bar chart (post-game quarters): {bar_chart_data}")
    else:
        # Pre-game: compute per-quarter season averages for Home vs Away
        bar_chart_data = []
        # df_full_history was fetched earlier via fetch_nba_full_history()
        # Filter historical games for each team
        home_hist = df_full_history[
            df_full_history[input_home_team_col] == row[input_home_team_col]
        ]
        away_hist = df_full_history[
            df_full_history[input_away_team_col] == row[input_away_team_col]
        ]
        for q in range(1, 5):
            # Pull the quarter column, drop NA, cast to numeric
            hq = pd.to_numeric(home_hist.get(f'home_q{q}', pd.Series()), errors='coerce').dropna()
            aq = pd.to_numeric(away_hist.get(f'away_q{q}', pd.Series()), errors='coerce').dropna()

            home_avg_q = round(hq.mean(), 1) if len(hq) > 0 else 0
            away_avg_q = round(aq.mean(), 1) if len(aq) > 0 else 0

            bar_chart_data.append({
                'category': f'Q{q}',
                'Home': home_avg_q,
                'Away': away_avg_q,
            })
        logger.info(f"Bar chart (pre-game quarter averages): {bar_chart_data}")


    #
    # ------------------------------------------------------------------
    #  RADAR: league-indexed (0–100) spider for Pace / OffRtg / DefRtg / eFG% / TOV%
    # ------------------------------------------------------------------
    default_radar_nba = {
        "Pace":   100.0,
        "OffRtg": 115.0,
        "DefRtg": 115.0,
        "eFG%":   0.53,
        "TOV%":   0.135,
    }

    metric_map: dict[str, tuple[str, bool]] = {
        "Pace"   : ("pace",     False),
        "OffRtg" : ("off_rtg",  False),
        "DefRtg" : ("def_rtg",  True ),   # invert – lower is better
        "eFG%"   : ("efg_pct",  False),
        "TOV%"   : ("tov_pct",  True ),   # invert
    }

    # ---------------- 1) league ranges ----------------
    league_ranges: dict[str, dict[str, float | bool]] = {}
    for label, (col, invert) in metric_map.items():
        if col in df_adv_rpc_data.columns:
            series = (
                pd.to_numeric(df_adv_rpc_data[col], errors="coerce")
                .dropna()
            )
            if series.empty:
                # column present but all NULL
                mn = default_radar_nba[label] * 0.5
                mx = default_radar_nba[label] * 1.5
            else:
                mn, mx = float(series.min()), float(series.max())
        else:
            # RPC did not send this column – fabricate range
            mn = default_radar_nba[label] * 0.5
            mx = default_radar_nba[label] * 1.5

        league_ranges[label] = {"min": mn, "max": mx, "invert": invert}

    # ---------------- 2) build payload ----------------
    radar_payload: list[dict[str, float | str]] = []

    def to_idx(v: float, rng: dict[str, float | bool]) -> float:
        if np.isnan(v) or rng["max"] == rng["min"]:
            return 50.0
        pct = 100.0 * (v - rng["min"]) / (rng["max"] - rng["min"])
        return 100.0 - pct if rng["invert"] else pct

    for label, (col, _invert) in metric_map.items():
        raw_home = float(home_adv_season_stats.get(col, default_radar_nba[label]))
        raw_away = float(away_adv_season_stats.get(col, default_radar_nba[label]))
        rng      = league_ranges[label]

        dec = 3 if label in ("eFG%", "TOV%") else 1
        radar_payload.append({
            "metric"  : label,
            "home_raw": round(raw_home, dec),
            "away_raw": round(raw_away, dec),
            "home_idx": round(to_idx(raw_home, rng), 1),
            "away_idx": round(to_idx(raw_away, rng), 1),
        })

    logger.debug("radar_payload = %s", radar_payload)
    # ------------------------------------------------------------------

    # ─── Key Metrics Bar Chart (RPG / SPG / 3PG) ────────────────────────────────
    home_team_original = df_game[input_home_team_col].iloc[0]
    away_team_original = df_game[input_away_team_col].iloc[0]
    home_norm = normalize_nba_team_name(home_team_original)
    away_norm = normalize_nba_team_name(away_team_original)

    misc_resp = sb_client.rpc(
        "get_nba_misc_team_stats",
        {"p_season_year": nba_season_start_year}
    ).execute()
    df_misc = pd.DataFrame(misc_resp.data or [])

    # Normalise for lookup
    if not df_misc.empty and "team" in df_misc.columns:
        df_misc["team_norm"] = df_misc["team"].apply(normalize_nba_team_name)
    else:
        df_misc["team_norm"] = pd.Series(dtype=str)

    def _safe_row(df: pd.DataFrame, norm: str) -> dict[str, Any]:
        return (
            df.loc[df["team_norm"] == norm].iloc[0].to_dict()
            if not df.loc[df["team_norm"] == norm].empty
            else {}
        )

    home_misc = _safe_row(df_misc, home_norm)
    away_misc = _safe_row(df_misc, away_norm)

        # --- ADD THE FOLLOWING 3 LINES FOR DEBUGGING ---
    logger.info(f"[DEBUG] Home team norm for lookup: '{home_norm}'")
    logger.info(f"[DEBUG] Away team norm for lookup: '{away_norm}'")
    logger.info(f"[DEBUG] Misc stats data returned from RPC:\n{df_misc.to_string()}")
    # --- END OF ADDED LINES ---



    key_metrics_data = [
        {  # x-axis label expected by BarChartComponent
            "category": "Rebs",
            "Home": round(float(home_misc.get("rpg", 0.0)), 1),
            "Away": round(float(away_misc.get("rpg", 0.0)), 1),
        },
        {
            "category": "Steals",
            "Home": round(float(home_misc.get("spg", 0.0)), 1),
            "Away": round(float(away_misc.get("spg", 0.0)), 1),
        },
        {
            "category": "3PM",
            "Home": round(float(home_misc.get("tp_pct", 0.0) * 100), 1),
            "Away": round(float(away_misc.get("tp_pct", 0.0) * 100), 1),
        },
    ]
    logger.debug("Key Metrics bar chart: %s", key_metrics_data)
    # ────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────── PIE CHART DATA ──────────────────────────────
    pie: list[dict[str, Any]] = []

    if is_historical_game and pd.notna(row.get(input_home_score_col)):
        # ---------- POST-GAME : show HOME scoring mix (2P vs FT) ----------
        home_fgm = float(row.get("home_fg_made", 0) or 0)
        home_3pm = float(row.get("home_3pm",     0) or 0)
        home_ftm = float(row.get("home_ft_made", 0) or 0)

        twop_made = max(0, home_fgm - home_3pm)

        pie = [
            {"category": "2P FG Made", "value": int(twop_made), "color": "#4ade80"},
            {"category": "FT Made",    "value": int(home_ftm),  "color": "#fbbf24"},
        ]
        if all(s["value"] == 0 for s in pie):
            pie = [{"category": "No Scoring Data", "value": 1, "color": "#cccccc"}]

        logger.debug("Pie chart (post-game) = %s", pie)

    else:
        # ---------- PRE-GAME : two separate pies (avg 2P vs avg FT) ----------
        season_rows = df_full_history[
            (df_full_history["season_start"] == nba_season_start_year) &
            (df_full_history["parsed_game_date"] < current_game_dt_obj)
        ]

        home_avg_2pm, home_avg_ftm = _avg_2pm_ftm(season_rows, home_norm)
        away_avg_2pm, away_avg_ftm = _avg_2pm_ftm(season_rows, away_norm)

        pie = [
            {   # pie 1
                "title": "Avg 2P Made",
                "data": [
                    {"category": "Home", "value": round(home_avg_2pm, 1), "color": "#4ade80"},
                    {"category": "Away", "value": round(away_avg_2pm, 1), "color": "#60a5fa"},
                ],
            },
            {   # pie 2
                "title": "Avg FT Made",
                "data": [
                    {"category": "Home", "value": round(home_avg_ftm, 1), "color": "#4ade80"},
                    {"category": "Away", "value": round(away_avg_ftm, 1), "color": "#60a5fa"},
                ],
            },
        ]
        logger.debug("Pie chart (pre-game, two pies) = %s", pie)

    # 10) Upsert into Supabase  —— note: pie_chart_data precedes key_metrics_data
    snapshot_payload_final = {
        "game_id"        : game_id_str,
        "game_date"      : current_game_dt_obj.isoformat(),
        "season"         : str(nba_season_full_str),

        "pie_chart_data" : pie,               # ← ordering matters for FE
        "key_metrics_data": key_metrics_data, #    bar chart follows pie

        "headline_stats" : headlines,
        "bar_chart_data" : bar_chart_data,
        "radar_chart_data": radar_payload,
        "last_updated"   : pd.Timestamp.utcnow().isoformat(),
    }

    logger.info(f"Upserting NBA snapshot for game_id: {game_id_str}")
    upsert_response = sb_client.table('nba_snapshots').upsert(snapshot_payload_final, on_conflict='game_id').execute()

    if hasattr(upsert_response, 'error') and upsert_response.error:
        logger.error(f"NBA Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
        logger.error(f"Supabase response: {upsert_response}")
    elif hasattr(upsert_response, 'data') and not upsert_response.data and not (hasattr(upsert_response, 'count') and upsert_response.count is not None and upsert_response.count > 0) :
        logger.warning(f"NBA Snapshot upsert for game_id={game_id_str} may have had an issue (no data/count returned). Response: {upsert_response}")
    else:
        logger.info(f"✅ NBA Snapshot upserted for game_id={game_id_str}")

# --- CLI entry point ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python backend/nba_features/make_nba_snapshots.py <game_id1> [<game_id2> ...]")
        logger.info("Example: python backend/nba_features/make_nba_snapshots.py 202310240CLE")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_nba_snapshot(game_id_arg)
            except Exception as e_main:
                logger.error(f"CRITICAL ERROR processing NBA game_id {game_id_arg} in snapshot generation: {e_main}", exc_info=True)