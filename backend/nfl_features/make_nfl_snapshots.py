# backend/nfl_features/make_nfl_snapshots.py

"""
Generate and upsert per-game NFL feature snapshots for frontend display.
"""
import os
import sys
import re
from pathlib import Path
import pandas as pd
from supabase import create_client, Client
import logging
import math
from dotenv import load_dotenv # <-- Add this import

# --- Boilerplate Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
)
logger = logging.getLogger(__name__)

# --- Path and Environment Variable Setup ---
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent # This is the project root

# Add project and backend directories to Python path for consistent imports
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Explicitly load environment variables from the .env file at the project root
dotenv_path = PROJECT_DIR / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")

# --- Supabase Client Initialization ---
# Now, attempt to get Supabase credentials using the same pattern as your MLB script
try:
    # Best practice: rely on the config module which might have other logic
    from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
    logger.info("Imported Supabase credentials from config module.")
except ImportError:
    # Fallback for standalone script execution
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    logger.info("Attempting to get Supabase credentials from environment variables.")

# Final check to ensure credentials are loaded
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials could not be resolved.")
    raise RuntimeError("Supabase URL/key missing. Ensure .env file is at the project root or variables are set in the environment.")

sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Data Fetching ---

def fetch_data_from_table(table_name: str, select_cols: str = "*") -> pd.DataFrame:
    """Generic helper to fetch all data from a table or view."""
    logger.debug(f"Fetching full data from '{table_name}'...")
    try:
        response = sb_client.table(table_name).select(select_cols).execute()
        df = pd.DataFrame(response.data or [])
        logger.debug(f"Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Error fetching from '{table_name}': {e}", exc_info=True)
        return pd.DataFrame()
  
def sanitize_float(value, default=None):
  """Converts NaN, inf, and -inf to a default value (JSON-compliant)."""
  if value is None or not isinstance(value, (int, float)):
      return value
  if not math.isfinite(value):
      return default
  return value


def _calculate_win_pct_from_record(record_str: str) -> float:
  """Parses a 'W-L-T' string and returns the win percentage."""
  if not record_str or not isinstance(record_str, str):
      return 0.0
  try:
      parts = list(map(int, record_str.split('-')))
      wins, losses, ties = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
      total_games = wins + losses + ties
      if total_games == 0:
          return 0.0
      # Standard NFL win percentage calculation
      return (wins + 0.5 * ties) / total_games
  except (ValueError, IndexError):
      return 0.0


# --- Main Snapshot Generator ---

def make_nfl_snapshot(game_id: str):
    """
    Generate (or refresh) a single NFL snapshot row → public.nfl_snapshots
    ──────────────────────────────────────────────────────────────────────
    The function assumes five supporting tables / views already exist:

      1. nfl_historical_game_stats        • game‑level scores and meta
      2. nfl_historical_game_team_stats   • team‑by‑game box‑score stats
      3. nfl_historical_team_stats        • season‑level team records
      4. v_nfl_team_srs_lite              • SRS‑Lite (strength) view
      5. v_nfl_team_sos                   • SoS % view

    Any column mismatches will be logged and that metric falls back to 0.
    """

    logger.info("↪︎ Generating NFL snapshot for game_id=%s", game_id)

    # ────────────────────────── 1) Core game lookup ──────────────────────────
    game_q = (
        sb_client.table("nfl_historical_game_stats")
        .select("*")
        .eq("game_id", game_id)
        .maybe_single()
        .execute()
    )
    if not game_q.data:
        # Fallback to schedule (future or in‑progress games)
        game_q = (
            sb_client.table("nfl_game_schedule")
            .select("*")
            .eq("game_id", game_id)
            .maybe_single()
            .execute()
        )
        if not game_q.data:
            logger.error("No data in either table for game_id=%s – aborting", game_id)
            return
        is_historical = False
    else:
        is_historical = True

    game = game_q.data
    season_year: int = int(game["season"])
    home_id, away_id = map(str, (game["home_team_id"], game["away_team_id"]))
    game_date = pd.to_datetime(game["game_date"]).date()

    # ────────────────────────── 2) Bulk dataframe pulls ──────────────────────
    df_games_all   = fetch_data_from_table("nfl_historical_game_stats")
    df_team_games  = fetch_data_from_table("nfl_historical_game_team_stats")
    df_season_team = fetch_data_from_table("nfl_historical_team_stats")
    df_srs         = fetch_data_from_table("v_nfl_team_srs_lite")
    df_sos         = fetch_data_from_table("v_nfl_team_sos")

    # ────────────────────────── 3) Season scoping helpers ────────────────────
    season_game_ids = (
        df_games_all.loc[df_games_all["season"] == season_year, "game_id"]
        .astype(int)
        .unique()
    )
    df_team_games_season = df_team_games[
        df_team_games["game_id"].astype(int).isin(season_game_ids)
    ]
    if df_team_games_season.empty:
        logger.warning(
            "No team‑by‑game rows for season %s – using all data (metrics noisy)",
            season_year,
        )
        df_team_games_season = df_team_games.copy()

    # Make sure the YPP + turnover columns exist
    if "yards_per_play" not in df_team_games_season.columns:
        df_team_games_season["yards_per_play"] = 0.0
    if "turnovers_total" not in df_team_games_season.columns:
        df_team_games_season["turnovers_total"] = 0

    # ────────────────────────── 4) Quick utility lambdas ─────────────────────
    pct = lambda num, den: num / den if den else 0.0

    def win_pct(row, col):
        """Extract a percentage safely from season record columns."""
        return sanitize_float(row.get(col, 0.0), default=0.0)

    # ────────────────────────── 5) Derived metrics per team ──────────────────
    team_metrics: dict[str, dict] = {}
    for tid in (home_id, away_id):
        t = int(tid)
        off_df = df_team_games_season[df_team_games_season["team_id"] == t]
        def_df = df_team_games_season[
            (df_team_games_season["game_id"].isin(off_df["game_id"])) &
            (df_team_games_season["team_id"] != t)
        ]

        gp = len(off_df)
        team_metrics[tid] = {
            "off_ypp"           : off_df["yards_per_play"].mean() if gp else 0.0,
            "def_ypp"           : def_df["yards_per_play"].mean() if gp else 0.0,
            "turnover_diff_pg"  : pct(
                                    def_df["turnovers_total"].sum() -
                                    off_df["turnovers_total"].sum(),
                                    gp,
                                  ),
        }

    # ────────────────────────── 6) Rest advantage (days) ─────────────────────
    def _last_played(team: str) -> pd.Timestamp | None:
        mask = (
            (df_games_all["home_team_id"].astype(str) == team) |
            (df_games_all["away_team_id"].astype(str) == team)
        )
        prior_games = df_games_all.loc[mask & (pd.to_datetime(df_games_all["game_date"]).dt.date < game_date)]
        return pd.to_datetime(prior_games["game_date"].max()) if not prior_games.empty else None

    last_home, last_away = map(_last_played, (home_id, away_id))
    rest_advantage = (
        (game_date - last_home.date()).days -
        (game_date - last_away.date()).days
        if last_home is not None and last_away is not None
        else 0
    )

    # ────────────────────────── 7) H2H – last 5 meetings ─────────────────────
    h2h_mask = (
        ((df_games_all["home_team_id"].astype(str) == home_id) &
         (df_games_all["away_team_id"].astype(str) == away_id))
        |
        ((df_games_all["home_team_id"].astype(str) == away_id) &
         (df_games_all["away_team_id"].astype(str) == home_id))
    )
    h2h_recent = df_games_all.loc[h2h_mask].sort_values("game_date", ascending=False).head(5)
    h2h_home_win_pct = pct(
        (
            h2h_recent["home_team_id"].astype(str).eq(home_id) &
            (h2h_recent["home_score"] > h2h_recent["away_score"])
        ).sum(),
        len(h2h_recent),
    )

    # ────────────────────────── 8) Venue win‑% differential ──────────────────
    def _pct_from_record(rec: str | None) -> float:
        """
        Parse '11‑3‑1' / '5‑4' / '0-0-0' → win%.
        Any non‑digit char is ignored; ties count as 0.5 win.
        """
        if not rec or not isinstance(rec, str):
            return 0.0

        nums = list(map(int, re.findall(r"\d+", rec)))
        if len(nums) == 2:
            wins, losses = nums
            ties = 0
        elif len(nums) == 3:
            wins, losses, ties = nums
        else:
            return 0.0

        gp = wins + losses + ties
        return (wins + 0.5 * ties) / gp if gp else 0.0


    def _pct_from_games(team: str, venue: str) -> float:
        """
        Compute win% directly from nfl_historical_game_stats when season row absent.
        `venue` = "home" | "road".
        """
        mask_home = df_games_all["home_team_id"].astype(str) == team
        mask_away = df_games_all["away_team_id"].astype(str) == team
        season_ok = df_games_all["season"] == season_year

        if venue == "home":
            games = df_games_all[mask_home & season_ok]
            wins  = (games["home_score"] > games["away_score"]).sum()
        else:  # road / away
            games = df_games_all[mask_away & season_ok]
            wins  = (games["away_score"] > games["home_score"]).sum()

        ties = (games["home_score"] == games["away_score"]).sum()
        gp   = len(games)

        return (wins + 0.5 * ties) / gp if gp else 0.0


    def _venue_win_pct(row: pd.Series | None, team: str, venue: str) -> float:
        """
        1⃣ record_{venue}  → parse string  
        2⃣ wins_{venue}_percentage (if you later add that col)  
        3⃣ season row missing  → compute from game results
        """
        if row is not None and not row.empty:
            rec_col = f"record_{venue}"
            if rec_col in row and pd.notna(row[rec_col]):
                pct = _pct_from_record(row[rec_col])
                if pct:  # happy path
                    return pct

            pct_col = f"wins_{venue}_percentage"
            if pct_col in row and pd.notna(row[pct_col]):
                return sanitize_float(row[pct_col], default=0.0)

        # Fallback: derive from game log
        return _pct_from_games(team, venue)


    home_id_i, away_id_i = map(int, (home_id, away_id))

    home_row = df_season_team.query(
        "team_id == @home_id_i and season == @season_year"
    ).head(1)
    away_row = df_season_team.query(
        "team_id == @away_id_i and season == @season_year"
    ).head(1)

    home_pct = _venue_win_pct(home_row.iloc[0] if not home_row.empty else None, home_id, "home")
    away_pct = _venue_win_pct(away_row.iloc[0] if not away_row.empty else None, away_id, "road")

    venue_win_pct_diff = home_pct - away_pct
    logger.debug("Venue Win%% — home: %.3f  away: %.3f  diff: %.3f", home_pct, away_pct, venue_win_pct_diff)

    # ────────────────────────── 9) Red‑zone TD % differential ───────────────
    rz_att_col, rz_made_col = "red_zone_att", "red_zone_made"
    if rz_att_col not in df_team_games_season.columns or rz_made_col not in df_team_games_season.columns:
        df_team_games_season[rz_att_col]  = 0
        df_team_games_season[rz_made_col] = 0

    def _rz_pct(team: str, offense: bool = True) -> float:
        t = int(team)
        team_rows = df_team_games_season[df_team_games_season["team_id"] == t]

        if offense:
            att = team_rows[rz_att_col].sum()
            tds = team_rows[rz_made_col].sum()
            return pct(tds, att)
        else:
            # Opponent’s red‑zone success vs this team
            opp_rows = df_team_games_season[
                df_team_games_season["game_id"].isin(team_rows["game_id"]) &
                (df_team_games_season["team_id"] != t)
            ]
            att = opp_rows[rz_att_col].sum()
            tds = opp_rows[rz_made_col].sum()
            return pct(tds, att)

    rz_td_diff = _rz_pct(home_id, offense=True) - _rz_pct(away_id, offense=False)

    # ────────────────────────── 10) Assemble headline payload ────────────────
    headlines = [
        {"label": "Rest Advantage (Home)",        "value": int(rest_advantage)},
        {"label": "Turnover Margin Difference",   "value": sanitize_float(round(
                                                    team_metrics[home_id]["turnover_diff_pg"] -
                                                    team_metrics[away_id]["turnover_diff_pg"], 2))},
        {"label": "Home/Road Win % Differential", "value": sanitize_float(round(venue_win_pct_diff, 2))},
        {"label": "Red Zone TD% Difference",      "value": sanitize_float(round(rz_td_diff, 3))},
    ]

    # ────────────────────────── 11) Quarter‑average bar chart ────────────────
    def q_avg(team: str) -> dict[str, float]:
        t = int(team)
        home_cols = ["home_q1","home_q2","home_q3","home_q4","home_ot"]
        away_cols = ["away_q1","away_q2","away_q3","away_q4","away_ot"]

        h = df_games_all.loc[
            (df_games_all["season"] == season_year) & (df_games_all["home_team_id"] == t),
            home_cols
        ].rename(columns=lambda c: c.replace("home_",""))
        a = df_games_all.loc[
            (df_games_all["season"] == season_year) & (df_games_all["away_team_id"] == t),
            away_cols
        ].rename(columns=lambda c: c.replace("away_",""))

        return pd.concat([h, a]).mean(numeric_only=True).fillna(0).to_dict()

    home_quarters = q_avg(home_id)
    away_quarters = q_avg(away_id)

    bar_chart_data = [
        {"category": q, "Home": round(home_quarters.get(q, 0), 2), "Away": round(away_quarters.get(q, 0), 2)}
        for q in ("q1","q2","q3","q4","ot")
    ]
    if bar_chart_data[-1]["Home"] == bar_chart_data[-1]["Away"] == 0:
        bar_chart_data.pop()  # Drop OT row if irrelevant

    # ────────────────────────── 12) Radar chart (SRS / SoS / YPP) ────────────
    def _value(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        return sanitize_float(df.iloc[0][col], default) if not df.empty else default

    home_srs = _value(df_srs[(df_srs["team_id"] == int(home_id)) & (df_srs["season"] == season_year)], "srs_lite")
    away_srs = _value(df_srs[(df_srs["team_id"] == int(away_id)) & (df_srs["season"] == season_year)], "srs_lite")
    home_sos = _value(df_sos[(df_sos["team_id"] == int(home_id)) & (df_sos["season"] == season_year)], "sos_pct")
    away_sos = _value(df_sos[(df_sos["team_id"] == int(away_id)) & (df_sos["season"] == season_year)], "sos_pct")

    radar_metrics = {
        "SRS"      : (home_srs, home_srs - away_srs),
        "SoS"      : (home_sos, home_sos - away_sos),
        "Off. YPP" : (team_metrics[home_id]["off_ypp"], team_metrics[home_id]["off_ypp"] - team_metrics[away_id]["off_ypp"]),
        "Def. YPP" : (team_metrics[home_id]["def_ypp"], team_metrics[home_id]["def_ypp"] - team_metrics[away_id]["def_ypp"]),
    }

    # Pull league ranges (min/max) – fall back to simple 0‑100 scaling
    ranges_rpc = sb_client.rpc("get_nfl_metric_ranges", {"p_season": season_year}).execute()
    league_ranges = {
        m["metric"]: {"min": float(m["min_value"]), "max": float(m["max_value"]), "invert": (m["metric"] == "Def. YPP")}
        for m in (ranges_rpc.data or [])
    }

    def _scale(val: float, rng: dict[str, float]) -> float:
        if rng["max"] == rng["min"]:
            return 50.0
        pct_val = 100.0 * (val - rng["min"]) / (rng["max"] - rng["min"])
        return 100.0 - pct_val if rng.get("invert", False) else pct_val

    radar_chart_data = []
    for metric, (home_raw, _) in radar_metrics.items():
        rng = league_ranges.get(metric, {"min": 0.0, "max": 1.0, "invert": False})
        away_raw = home_raw - radar_metrics[metric][1]
        radar_chart_data.append({
            "metric"   : metric,
            "home_raw" : round(home_raw, 3),
            "away_raw" : round(away_raw, 3),
            "home_idx" : round(_scale(home_raw, rng), 1),
            "away_idx" : round(_scale(away_raw, rng), 1),
        })

    # --- ADDED: Pie Chart (Scoring Averages) ---
    # This logic was missing from your merged file, I've added it back in.
    home_season_row = df_season_team[(df_season_team['team_id'] == int(home_id)) & (df_season_team['season'] == season_year)].iloc[0]
    away_season_row = df_season_team[(df_season_team['team_id'] == int(away_id)) & (df_season_team['season'] == season_year)].iloc[0]
    gp_home = home_season_row['won'] + home_season_row['lost'] + home_season_row['ties']
    gp_away = away_season_row['won'] + away_season_row['lost'] + away_season_row['ties']
    
    home_pf_avg = (home_season_row['points_for'] / gp_home) if gp_home > 0 else 0.0
    away_pf_avg = (away_season_row['points_for'] / gp_away) if gp_away > 0 else 0.0
    home_pa_avg = (home_season_row['points_against'] / gp_home) if gp_home > 0 else 0.0
    away_pa_avg = (away_season_row['points_against'] / gp_away) if gp_away > 0 else 0.0

    home_pf_label, away_pf_label = f"Home ({sanitize_float(round(home_pf_avg, 1), 0.0)})", f"Away ({sanitize_float(round(away_pf_avg, 1), 0.0)})"
    home_pa_label, away_pa_label = f"Home ({sanitize_float(round(home_pa_avg, 1), 0.0)})", f"Away ({sanitize_float(round(away_pa_avg, 1), 0.0)})"
    
    pie_chart_data = [
        {"title": "Avg Points For", "data": [{"category": home_pf_label, "value": home_pf_avg, "color": "#4ade80"}, {"category": away_pf_label, "value": away_pf_avg, "color": "#60a5fa"}]},
        {"title": "Avg Points Against", "data": [{"category": home_pa_label, "value": home_pa_avg, "color": "#4ade80"}, {"category": away_pa_label, "value": away_pa_avg, "color": "#60a5fa"}]}
    ]

    # --- Key Metrics Bar Chart (Yards Per Attempt) ----------------------------
    logger.info("--- Calculating Key Metrics (Y/A) ---")

    def extract_attempts(series: pd.Series) -> pd.Series:
        """
        Parse the 'C/A' or 'C-A' strings in `passing_comp_att` and return a numeric Series
        containing only the ATT (denominator). Handles both '17/26' and '17-26' formats.
        Non‑matches return 0.
        """
        return (
            series.astype(str)                              # ensure string ops
                  .str.extract(r'\d+\s*[-/]\s*(\d+)')[0]    # capture digits after slash or hyphen
                  .fillna(0)
                  .astype(int)
        )

    # ─── Home calculations ────────────────────────────────────────────────────
    home_team_games = df_team_games_season[df_team_games_season["team_id"] == int(home_id)]

    h_total_rush_yards    = home_team_games["rushings_total"].sum()
    h_total_rush_attempts = home_team_games["rushings_attempts"].sum()
    h_rush_ypa            = (h_total_rush_yards / h_total_rush_attempts) if h_total_rush_attempts else 0.0

    h_total_pass_yards    = home_team_games["passing_total"].sum()
    h_total_pass_attempts = extract_attempts(home_team_games["passing_comp_att"]).sum()
    h_pass_ypa            = (h_total_pass_yards / h_total_pass_attempts) if h_total_pass_attempts else 0.0

    logger.info(
        f"HOME ({home_id}): Pass Yds={h_total_pass_yards}, "
        f"Pass Att={h_total_pass_attempts}, Pass YPA={h_pass_ypa:.2f}"
    )
    logger.info(
        f"HOME ({home_id}): Rush Yds={h_total_rush_yards}, "
        f"Rush Att={h_total_rush_attempts}, Rush YPA={h_rush_ypa:.2f}"
    )

    # ─── Away calculations ────────────────────────────────────────────────────
    away_team_games = df_team_games_season[df_team_games_season["team_id"] == int(away_id)]

    a_total_rush_yards    = away_team_games["rushings_total"].sum()
    a_total_rush_attempts = away_team_games["rushings_attempts"].sum()
    a_rush_ypa            = (a_total_rush_yards / a_total_rush_attempts) if a_total_rush_attempts else 0.0

    a_total_pass_yards    = away_team_games["passing_total"].sum()
    a_total_pass_attempts = extract_attempts(away_team_games["passing_comp_att"]).sum()
    a_pass_ypa            = (a_total_pass_yards / a_total_pass_attempts) if a_total_pass_attempts else 0.0

    logger.info(
        f"AWAY ({away_id}): Pass Yds={a_total_pass_yards}, "
        f"Pass Att={a_total_pass_attempts}, Pass YPA={a_pass_ypa:.2f}"
    )
    logger.info(
        f"AWAY ({away_id}): Rush Yds={a_total_rush_yards}, "
        f"Rush Att={a_total_rush_attempts}, Rush YPA={a_rush_ypa:.2f}"
    )

    # ─── Assemble payload ─────────────────────────────────────────────────────
    key_metrics_data = [
        {
            "category": "Passing Yards / Attempt",
            "Home": sanitize_float(round(h_pass_ypa, 2)),
            "Away": sanitize_float(round(a_pass_ypa, 2)),
        },
        {
            "category": "Rushing Yards / Attempt",
            "Home": sanitize_float(round(h_rush_ypa, 2)),
            "Away": sanitize_float(round(a_rush_ypa, 2)),
        },
    ]
    # ────────────────────────── 13) Assemble + upsert snapshot ───────────────
    snapshot_row = {
        "game_id": game_id,
        "game_date": game_date.isoformat(),
        "season": str(season_year),
        "is_historical": is_historical,
        "headline_stats": headlines,
        "bar_chart_data": bar_chart_data,
        "radar_chart_data": radar_chart_data, # This needs to be defined from your radar section
        "pie_chart_data": pie_chart_data,
        "key_metrics_data": key_metrics_data,
        "last_updated": pd.Timestamp.utcnow().isoformat(),
    }

    resp = sb_client.table("nfl_snapshots").upsert(snapshot_row, on_conflict="game_id").execute()
    if getattr(resp, "error", None):
        logger.error("Snapshot upsert FAILED for game_id=%s – %s", game_id, resp.error)
    else:
        logger.info("✅ Snapshot upserted for game_id=%s", game_id)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python backend/nfl_features/make_nfl_snapshots.py <game_id1> [<game_id2> ...]")
    else:
        for game_id_arg in sys.argv[1:]:
            try:
                make_nfl_snapshot(game_id_arg)
            except Exception as e:
                logger.error(f"CRITICAL ERROR processing game_id {game_id_arg}: {e}", exc_info=True)