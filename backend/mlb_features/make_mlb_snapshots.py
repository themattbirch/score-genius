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
def fetch_table(table_name: str, match_criteria: Dict[str, Any] | None = None, select_cols: str = "*") -> pd.DataFrame:
    logger.debug(f"Fetching from table '{table_name}' with criteria: {match_criteria}")
    try:
        qb = sb_client.table(table_name).select(select_cols)
        if match_criteria:
            qb = qb.match(match_criteria)
        response = qb.execute()
        return pd.DataFrame(response.data or [])
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
    input_game_date_col: str = "game_date_et",
    input_game_date_utc_col: str = "game_date_time_utc",
    input_home_team_id_col: str = "home_team_id",
    input_away_team_id_col: str = "away_team_id",
    input_home_pitcher_hand_col: str = "home_starter_pitcher_handedness",
    input_away_pitcher_hand_col: str = "away_starter_pitcher_handedness",
    input_home_score_col: str = "home_score",
    input_away_score_col: str = "away_score",
):
    game_id_str = str(game_id)
    logger.info(f"--- Generating MLB Snapshot for game_id: {game_id_str} ---")

    # 1) Pull rows
    hist_cols = (
        "game_id,game_date_time_utc,status_short,"
        "home_team_id,away_team_id,"
        "home_score,away_score,"
        "h_inn_1,h_inn_2,h_inn_3,h_inn_4,h_inn_5,h_inn_6,h_inn_7,h_inn_8,h_inn_9,h_inn_extra,"
        "a_inn_1,a_inn_2,a_inn_3,a_inn_4,a_inn_5,a_inn_6,a_inn_7,a_inn_8,a_inn_9,a_inn_extra,"
        "home_starter_pitcher_handedness,away_starter_pitcher_handedness"
    )
    df_hist  = fetch_table("mlb_historical_game_stats", {"game_id": game_id_str}, hist_cols)
    df_sched = fetch_table(    "mlb_game_schedule",
    {"game_id": game_id_str},
    "game_id,home_team_id,away_team_id,game_date_et,home_probable_pitcher_handedness,away_probable_pitcher_handedness"
)


    is_historical_game = not df_hist.empty

    # 2) Build df_game
    if is_historical_game:
        df_game = df_hist.copy()
        if not df_sched.empty and "game_date_et" in df_sched.columns:
            df_game["game_date_et"] = df_sched["game_date_et"].iloc[0]
    else:
        if df_sched.empty:
            logger.error(f"No primary data found for game_id={game_id_str}. Cannot generate snapshot.")
            return
        df_game = df_sched.copy()
        for c in [
            input_home_score_col, input_away_score_col,
            input_home_pitcher_hand_col, input_away_pitcher_hand_col,
            *[f"h_inn_{i}" for i in range(1,10)], "h_inn_extra",
            *[f"a_inn_{i}" for i in range(1,10)], "a_inn_extra",
        ]:
            if c not in df_game.columns:
                df_game[c] = pd.NA

    # 3) Parse date (ET)
    if "game_date_et" in df_game.columns and pd.notna(df_game["game_date_et"].iloc[0]):
        current_game_date_et = pd.to_datetime(df_game["game_date_et"].iloc[0], errors="coerce").date()
    else:
        dt_utc = pd.to_datetime(df_hist[input_game_date_utc_col].iloc[0], errors="coerce", utc=True) if not df_hist.empty else pd.NaT
        current_game_date_et = dt_utc.tz_convert("America/New_York").date() if pd.notna(dt_utc) else None

    if current_game_date_et is None or pd.isna(current_game_date_et):
        logger.error(f"No valid game date for game_id={game_id_str}. Abort.")
        return

    # 4) Pre-game rest (only when NOT historical)
    if not is_historical_game:
        df_full_hist = fetch_mlb_full_history()
        df_full_hist["played_date_et"] = (
            pd.to_datetime(df_full_hist["game_date_time_utc"], errors="coerce", utc=True)
              .dt.tz_convert("America/New_York").dt.date
        )

        home_id = str(df_game[input_home_team_id_col].iloc[0])
        away_id = str(df_game[input_away_team_id_col].iloc[0])

        def _days_rest(tid: str) -> int:
            mask = (
                (df_full_hist["home_team_id"].astype(str) == tid) |
                (df_full_hist["away_team_id"].astype(str) == tid)
            ) & (df_full_hist["played_date_et"] < current_game_date_et)
            if not mask.any():
                return 1
            last = df_full_hist.loc[mask, "played_date_et"].max()
            return max(1, (current_game_date_et - last).days)

        df_game["rest_days_home"] = _days_rest(home_id)
        df_game["rest_days_away"] = _days_rest(away_id)
        df_game["rest_advantage"]  = df_game["rest_days_home"] - df_game["rest_days_away"]
    else:
        df_game["rest_days_home"] = pd.NA
        df_game["rest_days_away"] = pd.NA
        df_game["rest_advantage"]  = pd.NA

    df_game["is_historical_game"] = is_historical_game

    # 5) Season
    season_year = determine_season(pd.Timestamp(current_game_date_et))

    # 6) Context dfs
    df_full_history = fetch_mlb_full_history()
    df_team_stats   = fetch_mlb_team_season_stats()

    # 7) Form strings
    home_id = df_game[input_home_team_id_col].iloc[0]
    away_id = df_game[input_away_team_id_col].iloc[0]
    df_game["home_current_form"] = _get_mlb_team_form_string(home_id, current_game_date_et, df_full_history)
    df_game["away_current_form"] = _get_mlb_team_form_string(away_id, current_game_date_et, df_full_history)

    # 8) Feature pipeline
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
    row_dict = df_features.iloc[0].to_dict()

    # 9) Finalize rest once
    if is_historical_game:
        hr = float(row_dict.get("rest_days_home", 1) or 1)
        ar = float(row_dict.get("rest_days_away", 1) or 1)
        if hr > 14: hr = 1
        if ar > 14: ar = 1
        row_dict["rest_days_home"] = hr
        row_dict["rest_days_away"] = ar
        row_dict["rest_advantage"] = hr - ar
    else:
        row_dict["rest_days_home"] = int(df_game["rest_days_home"].iloc[0])
        row_dict["rest_days_away"] = int(df_game["rest_days_away"].iloc[0])
        row_dict["rest_advantage"]  = row_dict["rest_days_home"] - row_dict["rest_days_away"]

    logger.info("Rest (final): H=%s A=%s Adv=%s",
                row_dict["rest_days_home"], row_dict["rest_days_away"], row_dict["rest_advantage"])

    # 10) Radar metrics
    h_for      = row_dict.get("home_season_runs_for_avg",     4.5)
    h_against  = row_dict.get("home_season_runs_against_avg", 4.5)
    a_for      = row_dict.get("away_season_runs_for_avg",     4.5)
    a_against  = row_dict.get("away_season_runs_against_avg", 4.5)
    row_dict["home_run_differential"] = h_for - h_against
    row_dict["away_run_differential"] = a_for - a_against
    row_dict["home_pythag_win_pct"]   = (h_for**2) / ((h_for**2) + (h_against**2))
    row_dict["away_pythag_win_pct"]   = (a_for**2) / ((a_for**2) + (a_against**2))

    # 11) Handedness display features
    df_pitcher_splits = fetch_mlb_pitcher_splits_data(season_year)
    home_hand_col = "home_starter_pitcher_handedness" if is_historical_game else "home_probable_pitcher_handedness"
    away_hand_col = "away_starter_pitcher_handedness" if is_historical_game else "away_probable_pitcher_handedness"
    row_dict[home_hand_col] = df_game.get(home_hand_col, pd.Series([pd.NA])).iloc[0]
    row_dict[away_hand_col] = df_game.get(away_hand_col, pd.Series([pd.NA])).iloc[0]

    df_hand = handedness_transform(
        df=pd.DataFrame([row_dict]),
        mlb_pitcher_splits_df=df_pitcher_splits,
        home_team_col_param="home_team_norm",
        away_team_col_param="away_team_norm",
        home_pitcher_hand_col=home_hand_col,
        away_pitcher_hand_col=away_hand_col,
        debug=True,
    )
    if not df_hand.empty:
        row_dict.update(df_hand.iloc[0].to_dict())

    # 12) RPC for bar chart
    rpc_adv = sb_client.rpc("get_mlb_advanced_team_stats_splits", {"p_season": season_year}).execute().data or []
    df_rpc_adv = pd.DataFrame(rpc_adv)
    if "team_id" in df_rpc_adv.columns:
        df_rpc_adv["team_norm"] = df_rpc_adv["team_id"].apply(normalize_team_name)

    home_norm = normalize_team_name(row_dict.get(input_home_team_id_col))
    away_norm = normalize_team_name(row_dict.get(input_away_team_id_col))

    def _first_row_as_dict(df: pd.DataFrame) -> dict:
        return df.iloc[0].to_dict() if not df.empty else {}

    home_adv_rpc_data = _first_row_as_dict(df_rpc_adv[df_rpc_adv["team_norm"] == home_norm])
    away_adv_rpc_data = _first_row_as_dict(df_rpc_adv[df_rpc_adv["team_norm"] == away_norm])

    # 13) Headlines
    headlines = [
        {"label": "Rest Advantage (Home)",  "value": int(row_dict["rest_advantage"])},
        {"label": "Form Win% Diff",         "value": round(float(row_dict.get("form_win_pct_diff", 0.0)), 2)},
        {"label": "Prev Season Win% Diff",  "value": round(float(row_dict.get("prev_season_win_pct_diff", 0.0)), 2)},
        {"label": f"H2H Home Win% (L{int(row_dict.get('matchup_num_games', 0))})",
         "value": round(float(row_dict.get("matchup_home_win_pct", 0.0)), 2)},
    ]

    RUNS_FOR_DEFAULT     = float(MLB_DEFAULTS.get("mlb_avg_runs_for", 4.5))
    RUNS_AGAINST_DEFAULT = float(MLB_DEFAULTS.get("mlb_avg_runs_against", 4.5))

    bar_chart_data = [
        {"category": "Avg Runs For",     "Home": round(float(home_adv_rpc_data.get("runs_for_avg_overall",     RUNS_FOR_DEFAULT)), 2),
                                          "Away": round(float(away_adv_rpc_data.get("runs_for_avg_overall",     RUNS_FOR_DEFAULT)), 2)},
        {"category": "Avg Runs Against", "Home": round(float(home_adv_rpc_data.get("runs_against_avg_overall", RUNS_AGAINST_DEFAULT)), 2),
                                          "Away": round(float(away_adv_rpc_data.get("runs_against_avg_overall", RUNS_AGAINST_DEFAULT)), 2)},
    ]

    # 14) Radar ranges
    ranges_rows = sb_client.rpc("get_mlb_metric_ranges", {"p_season": season_year}).execute().data or []
    league_ranges = {
        r["metric"]: {
            "min": float(r["min_value"]),
            "max": float(r["max_value"]),
            "invert": (r["metric"] == "Season Runs Against"),
        } for r in ranges_rows
    }

    if "Pythagorean Win %" not in league_ranges and not df_team_stats.empty:
        pyth = (df_team_stats["runs_for_avg_all"]**2) / ((df_team_stats["runs_for_avg_all"]**2) + (df_team_stats["runs_against_avg_all"]**2))
        league_ranges["Pythagorean Win %"] = {"min": float(pyth.min()), "max": float(pyth.max()), "invert": False}

    if "Run Differential" not in league_ranges and not df_team_stats.empty:
        rd = df_team_stats["runs_for_avg_all"] - df_team_stats["runs_against_avg_all"]
        league_ranges["Run Differential"] = {"min": float(rd.min()), "max": float(rd.max()), "invert": False}

    if not ranges_rows:
        league_ranges.update({
            "Venue Win %": {
                "min": float(df_team_stats["wins_home_percentage"].min()),
                "max": float(df_team_stats["wins_home_percentage"].max()),
                "invert": False,
            },
            "Season Runs Scored": {
                "min": float(df_team_stats["runs_for_avg_all"].min()),
                "max": float(df_team_stats["runs_for_avg_all"].max()),
                "invert": False,
            },
            "Season Runs Against": {
                "min": float(df_team_stats["runs_against_avg_all"].min()),
                "max": float(df_team_stats["runs_against_avg_all"].max()),
                "invert": True,
            },
            "Home/Away Win Advantage": {
                "min": float((df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).min()),
                "max": float((df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).max()),
                "invert": False,
            },
        })

    radar_metrics_map = {
        "Pythagorean W%":    {"db_name": "Pythagorean Win %",        "home_col": "home_pythag_win_pct",      "away_col": "away_pythag_win_pct",      "round": 3},
        "Run Differential":  {"db_name": "Run Differential",          "home_col": "home_run_differential",    "away_col": "away_run_differential",    "round": 2},
        "Venue W%":          {"db_name": "Venue Win %",               "home_col": "home_venue_win_pct_home",  "away_col": "away_venue_win_pct_away",  "round": 3},
        "H/A Advantage":     {"db_name": "Home/Away Win Advantage",   "home_col": "home_venue_win_advantage", "away_col": "away_venue_win_advantage", "round": 3},
    }

    def _scale(v, rng):
        if rng["max"] == rng["min"]:
            pct = 50.0
        else:
            pct = 100.0 * (v - rng["min"]) / (rng["max"] - rng["min"])
        return 100.0 - pct if rng.get("invert", False) else pct

    radar_payload = []
    for disp, cfg in radar_metrics_map.items():
        rng = league_ranges[cfg["db_name"]]
        h_raw = float(row_dict.get(cfg["home_col"], 0.0))
        a_raw = float(row_dict.get(cfg["away_col"], 0.0))
        radar_payload.append({
            "metric": disp,
            "home_raw": round(h_raw, cfg["round"]),
            "away_raw": round(a_raw, cfg["round"]),
            "home_idx": round(_scale(h_raw, rng), 1),
            "away_idx": round(_scale(a_raw, rng), 1),
        })

    home_hand_val = float(row_dict.get("h_team_off_avg_runs_vs_opp_hand", 0.0))
    away_hand_val = float(row_dict.get("a_team_off_avg_runs_vs_opp_hand", 0.0))
    pie_payload = [
        {"category": f"Home Offense vs Starting Pitcher's Hand ({round(home_hand_val, 2)} Runs)", "value": home_hand_val, "color": "#60a5fa"},
        {"category": f"Away Offense vs Starting Pitcher's Hand ({round(away_hand_val, 2)} Runs)", "value": away_hand_val, "color": "#4ade80"},
    ]

    snapshot_payload_final = {
        "game_id": game_id_str,
        "game_date": current_game_date_et.isoformat(),
        "season": str(season_year),
        "is_historical": is_historical_game,
        "headline_stats": headlines,
        "bar_chart_data": bar_chart_data,
        "radar_chart_data": radar_payload,
        "pie_chart_data": pie_payload,
        "last_updated": pd.Timestamp.utcnow().isoformat()
    }

    logger.info(f"Upserting MLB snapshot for game_id: {game_id_str}")
    upsert_response = sb_client.table("mlb_snapshots").upsert(snapshot_payload_final, on_conflict="game_id").execute()

    if getattr(upsert_response, "error", None):
        logger.error(f"MLB Snapshot upsert FAILED for game_id={game_id_str}: {upsert_response.error}")
    elif hasattr(upsert_response, "data") and not upsert_response.data and not getattr(upsert_response, "count", 0):
        logger.warning(f"MLB Snapshot upsert maybe had an issue (no data/count). Resp: {upsert_response}")
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