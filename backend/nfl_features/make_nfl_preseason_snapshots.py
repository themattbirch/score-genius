"""
Generate and upsert per-game NFL PRESEASON feature snapshots.
This version creates a comparative "H2H" stat based on each team's
performance (Win %, Off YPP, Def YPP) in the prior season.
"""
import os
import sys
from pathlib import Path
import pandas as pd
from supabase import create_client, Client
from datetime import date
import logging
from dotenv import load_dotenv

# --- Boilerplate and Supabase Connection ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path: sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))
dotenv_path = BACKEND_DIR / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}.")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL/key missing.")
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
logger.info("Supabase client initialized successfully.")

# --- Helper to calculate win percentage from record ---
def get_win_pct(row):
    try:
        wins, losses, ties = int(row.get('won', 0)), int(row.get('lost', 0)), int(row.get('ties', 0))
        total_games = wins + losses + ties
        if total_games == 0: return 0.0, "0-0"
        win_pct = (wins + 0.5 * ties) / total_games
        record_str = f"{wins}-{losses}" + (f"-{ties}" if ties > 0 else "")
        return win_pct, record_str
    except (ValueError, TypeError):
        return 0.0, "N/A"

# --- Main Snapshot Generator ---
def make_nfl_preseason_snapshot(game_id: str):
    logger.info(f"↪︎ Generating PRESEASON snapshot for game_id={game_id}")

    # 1. Look up the preseason game and SELECT the 'stage' column
    game_q = (
        sb_client.table("nfl_preseason_schedule")
        .select("game_id, game_date, season, stage, home_team, away_team, home_team_id, away_team_id")
        .eq("game_id", game_id)
        .maybe_single()
        .execute()
    )

    if not game_q.data:
        logger.error(f"No data in nfl_preseason_schedule for game_id={game_id}. Aborting.")
        return

    game = game_q.data

    if game.get("season") is None:
        logger.error(f"The 'season' column is NULL for game_id={game_id}. Aborting.")
        return
        
    season_year = int(game["season"])
    home_id, away_id = game.get("home_team_id"), game.get("away_team_id")
    if not home_id or not away_id:
        logger.error(f"Missing home_team_id or away_team_id for game_id={game_id}. Aborting.")
        return
    home_id_str, away_id_str = str(home_id), str(away_id)
    game_date = pd.to_datetime(game["game_date"]).date()
    
    # 2. Fetch PRIOR season data for both teams
    prior_season = season_year - 1
    logger.info(f"Fetching all {prior_season} season stats for teams {home_id_str} and {away_id_str}...")
    
    # Query for won/lost record and points for/against
    team_stats_q = (
        sb_client.table("nfl_historical_team_stats")
        .select("team_id, won, lost, ties, points_for, points_against")
        .eq("season", prior_season)
        .in_("team_id", [home_id_str, away_id_str])
        .execute()
    )
    
    # Query for YPP, rushing/passing stats
    team_games_q = (
        sb_client.table("nfl_historical_game_team_stats")
        .select("game_id, team_id, yards_per_play, turnovers_total, rushings_total, rushings_attempts, passing_total, passing_comp_att")
        .eq("season", prior_season)
        .execute()
    )
    df_team_games_season = pd.DataFrame(team_games_q.data or [])

    # Query for quarter scoring
    all_games_q = (
        sb_client.table("nfl_historical_game_stats")
        .select("home_team_id, away_team_id, season, home_q1, home_q2, home_q3, home_q4, away_q1, away_q2, away_q3, away_q4")
        .eq("season", prior_season)
        .execute()
    )
    df_games_all = pd.DataFrame(all_games_q.data or [])

    # 3. Calculate Headline Stats
    home_stats_row = next((item for item in team_stats_q.data if str(item['team_id']) == home_id_str), None)
    away_stats_row = next((item for item in team_stats_q.data if str(item['team_id']) == away_id_str), None)
    home_win_pct, _ = get_win_pct(home_stats_row) if home_stats_row else (0.0, "N/A")
    away_win_pct, _ = get_win_pct(away_stats_row) if away_stats_row else (0.0, "N/A")

    team_metrics = {}
    for tid_str in (home_id_str, away_id_str):
        tid = int(tid_str)
        off_df = df_team_games_season[df_team_games_season["team_id"] == tid]
        def_df = df_team_games_season[
            (df_team_games_season["game_id"].isin(off_df["game_id"])) &
            (df_team_games_season["team_id"] != tid)
        ]
        gp = len(off_df)
        team_metrics[tid_str] = {
            "off_ypp": off_df["yards_per_play"].mean() if gp > 0 else 0.0,
            "def_ypp": def_df["yards_per_play"].mean() if gp > 0 else 0.0,
        }

    home_off_ypp = team_metrics.get(home_id_str, {}).get("off_ypp", 0.0)
    away_off_ypp = team_metrics.get(away_id_str, {}).get("off_ypp", 0.0)
    home_def_ypp = team_metrics.get(home_id_str, {}).get("def_ypp", 0.0)
    away_def_ypp = team_metrics.get(away_id_str, {}).get("def_ypp", 0.0)

    headlines = [
        {"label": "Season", "value": f"{season_year} Preseason"},
        {"label": f"{prior_season} Win %", "value": f"{home_win_pct:.1%} vs {away_win_pct:.1%}"},
        {"label": "Home Team", "value": game.get("home_team", "N/A")},
        {"label": f"Off. YPP (Home vs Away)", "value": f"{home_off_ypp:.2f} / {away_off_ypp:.2f}"},
        {"label": "Away Team", "value": game.get("away_team", "N/A")},
        {"label": f"Def. YPP (Home vs Away)", "value": f"{home_def_ypp:.2f} / {away_def_ypp:.2f}"},
    ]

    # 4. Calculate Quarter Scoring
    def q_avg(team_id_str: str) -> dict[str, float]:
        tid = int(team_id_str)
        home_cols = ["home_q1","home_q2","home_q3","home_q4"]
        away_cols = ["away_q1","away_q2","away_q3","away_q4"]

        h_games = df_games_all.loc[df_games_all["home_team_id"] == tid, home_cols].rename(columns=lambda c: c.replace("home_",""))
        a_games = df_games_all.loc[df_games_all["away_team_id"] == tid, away_cols].rename(columns=lambda c: c.replace("away_",""))

        return pd.concat([h_games, a_games]).mean(numeric_only=True).fillna(0).to_dict()

    home_quarters = q_avg(home_id_str)
    away_quarters = q_avg(away_id_str)

    bar_chart_data = [
        {"category": q.upper(), "Home": round(home_quarters.get(q, 0), 2), "Away": round(away_quarters.get(q, 0), 2)}
        for q in ("q1", "q2", "q3", "q4")
    ]
    
    logger.info(f"Generated Quarter Scoring data.")

    # 5. Calculate Pie Chart data (Scoring Averages)
    def get_points_avg(row):
        if not row: return 0.0, 0.0
        wins, losses, ties = int(row.get('won', 0)), int(row.get('lost', 0)), int(row.get('ties', 0))
        gp = wins + losses + ties
        pf_avg = (row.get('points_for', 0) / gp) if gp > 0 else 0.0
        pa_avg = (row.get('points_against', 0) / gp) if gp > 0 else 0.0
        return pf_avg, pa_avg

    home_pf_avg, home_pa_avg = get_points_avg(home_stats_row)
    away_pf_avg, away_pa_avg = get_points_avg(away_stats_row)

    pie_chart_data = [
        {"title": "Avg Points For", "data": [
            {"category": f"Home ({home_pf_avg:.1f})", "value": home_pf_avg},
            {"category": f"Away ({away_pf_avg:.1f})", "value": away_pf_avg}
        ]},
        {"title": "Avg Points Against", "data": [
            {"category": f"Home ({home_pa_avg:.1f})", "value": home_pa_avg},
            {"category": f"Away ({away_pa_avg:.1f})", "value": away_pa_avg}
        ]}
    ]
    logger.info("Generated Pie Chart data.")

    # 6. Calculate Key Metrics data (Yards Per Attempt)
    def extract_attempts(series: pd.Series) -> pd.Series:
        return series.astype(str).str.extract(r'\d+\s*[-/]\s*(\d+)')[0].fillna(0).astype(int)

    key_metrics_data = []
    try:
        home_games = df_team_games_season[df_team_games_season["team_id"] == int(home_id_str)]
        away_games = df_team_games_season[df_team_games_season["team_id"] == int(away_id_str)]

        h_rush_ypa = (home_games["rushings_total"].sum() / home_games["rushings_attempts"].sum()) if home_games["rushings_attempts"].sum() else 0.0
        a_rush_ypa = (away_games["rushings_total"].sum() / away_games["rushings_attempts"].sum()) if away_games["rushings_attempts"].sum() else 0.0
        h_pass_ypa = (home_games["passing_total"].sum() / extract_attempts(home_games["passing_comp_att"]).sum()) if extract_attempts(home_games["passing_comp_att"]).sum() else 0.0
        a_pass_ypa = (away_games["passing_total"].sum() / extract_attempts(away_games["passing_comp_att"]).sum()) if extract_attempts(away_games["passing_comp_att"]).sum() else 0.0

        key_metrics_data = [
            {"category": "Passing Yards / Attempt", "Home": round(h_pass_ypa, 2), "Away": round(a_pass_ypa, 2)},
            {"category": "Rushing Yards / Attempt", "Home": round(h_rush_ypa, 2), "Away": round(a_rush_ypa, 2)},
        ]
        logger.info("Generated Key Metrics data.")
    except Exception as e:
        logger.warning(f"Could not generate Key Metrics data due to missing columns or error: {e}")

    # Assemble and upsert the snapshot
    snapshot_row = {
        "game_id": game_id,
        "game_date": game_date.isoformat(),
        "season": str(season_year),
        "is_historical": False,
        "headline_stats": headlines, # Assumes headlines is defined above
        "bar_chart_data": bar_chart_data, # Assumes bar_chart_data is defined
        "radar_chart_data": [],
        "pie_chart_data": pie_chart_data, # Assumes pie_chart_data is defined
        "key_metrics_data": key_metrics_data, # Assumes key_metrics_data is defined
        "last_updated": pd.Timestamp.utcnow().isoformat(),
    }

    # This upsert will now succeed because we are no longer trying to write to a non-existent column
    resp = sb_client.table("nfl_snapshots").upsert(snapshot_row, on_conflict="game_id").execute()
    if getattr(resp, "error", None):
        logger.error(f"Snapshot upsert FAILED for game_id={game_id} – {resp.error}")
    else:
        logger.info(f"✅ Preseason snapshot upserted for game_id={game_id}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python backend/nfl_features/make_nfl_preseason_snapshots.py <game_id1> [<game_id2> ...]")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_nfl_preseason_snapshot(game_id_arg)
            except Exception as e:
                logger.error(f"CRITICAL ERROR processing game_id {game_id_arg}: {e}", exc_info=True)