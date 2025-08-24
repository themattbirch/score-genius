# backend/nfl_features/make_nfl_snapshots.py

"""
Generate and upsert per-game NFL feature snapshots for frontend display.
"""
import os
import sys
import re
from pathlib import Path
from typing import Union
import pandas as pd
from supabase import create_client, Client
from datetime import date
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
    
def _safe_int(x, default=None):
    if x is None:
        return default
    try:
        # handles numeric strings like "17", also ints
        return int(x)
    except (ValueError, TypeError):
        # handles "None", "", "NA", etc.
        try:
            # attempt to strip and re-int if x is a stringy number with spaces
            s = str(x).strip()
            return int(s) if s.isdigit() or (s.startswith('-') and s[1:].isdigit()) else default
        except Exception:
            return default
        
def _norm(s: str | None) -> str:
    """Lowercase alnum-only for robust name/abbr matching."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _build_local_team_index(df_games_all: pd.DataFrame, df_season_team: pd.DataFrame) -> dict[str, int]:
    """
    Build {normalized_name_or_abbr -> team_id} using historical game logs
    and season team records. Prefers most frequent mapping if conflicts exist.
    """
    counter: dict[tuple[str, int], int] = {}

    # From historical games: home & away name -> id
    if not df_games_all.empty:
        for _, r in df_games_all[["home_team_name","home_team_id"]].dropna().iterrows():
            key = (_norm(r["home_team_name"]), int(r["home_team_id"]))
            counter[key] = counter.get(key, 0) + 1
        for _, r in df_games_all[["away_team_name","away_team_id"]].dropna().iterrows():
            key = (_norm(r["away_team_name"]), int(r["away_team_id"]))
            counter[key] = counter.get(key, 0) + 1

    # From season team table: names and abbrs if present
    name_cols = [c for c in ["team_name","name","full_name","display_name"] if c in df_season_team.columns]
    abbr_cols = [c for c in ["abbr","team_abbr","code","short_name"] if c in df_season_team.columns]
    if not df_season_team.empty:
        for _, r in df_season_team[["team_id"] + name_cols + abbr_cols].fillna("").iterrows():
            tid = _safe_int(r["team_id"], None)
            if tid is None: 
                continue
            for c in name_cols + abbr_cols:
                val = _norm(r[c])
                if val:
                    key = (val, int(tid))
                    counter[key] = counter.get(key, 0) + 1

    # Choose most frequent team_id for each normalized token
    best: dict[str, tuple[int, int]] = {}  # token -> (team_id, count)
    for (token, tid), cnt in counter.items():
        cur = best.get(token)
        if cur is None or cnt > cur[1]:
            best[token] = (tid, cnt)

    return {token: tid for token, (tid, _) in best.items()}

def _resolve_team_id_local(game: dict, side: str, team_index: dict[str, int]) -> int | None:
    """
    Try to recover team_id from schedule row using local index.
    side: "home" or "away".
    """
    # candidate text fields commonly seen in schedule rows
    candidates = [
        game.get(f"{side}_team_name"),
        game.get(f"{side}_name"),
        game.get(f"{side}_team"),
        game.get(f"{side}_full_name"),
        game.get(f"{side}_abbr"),
        game.get(f"{side}_code"),
        game.get(side),  # sometimes just "home": "Dallas Cowboys"
    ]
    for cand in candidates:
        token = _norm(cand)
        if token and token in team_index:
            return team_index[token]
    return None

  
def sanitize_float(value, default=None):
  """Converts NaN, inf, and -inf to a default value (JSON-compliant)."""
  if value is None or not isinstance(value, (int, float)):
      return value
  if not math.isfinite(value):
      return default
  return value

# ───────────────────────── Rest calculation (mirror MLB/NBA) ─────────────
MAX_REASONABLE_REST = 10
MIN_REST = 0

def _compute_rest_days(
    all_hist: pd.DataFrame,
    team_id: Union[str, int],
    game_date: Union[date, str, pd.Timestamp]
) -> int:
    # Normalize game_date to date
    if not isinstance(game_date, date):
        game_date = pd.to_datetime(game_date, errors="coerce").date()

    # Mask for any prior game by that team
    mask = (
        (all_hist["home_team_id"].astype(str) == str(team_id))
        | (all_hist["away_team_id"].astype(str) == str(team_id))
    ) & (all_hist["played_date_et"] < game_date)

    if not mask.any():
        return 1

    last_date = all_hist.loc[mask, "played_date_et"].max()
    if pd.isna(last_date):
        return 1

    diff = (game_date - last_date).days
    diff = max(MIN_REST, diff)
    if diff > MAX_REASONABLE_REST:
        logger.debug("NFL Rest clamp: team=%s, raw=%d days → 1", team_id, diff)
        return 1
    return diff


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

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _compute_league_ranges_local(season_year: int, df_srs: pd.DataFrame, df_sos: pd.DataFrame, df_team_games_season: pd.DataFrame) -> dict[str, dict]:
    ranges: dict[str, dict] = {}

    srs_col = _pick_col(df_srs, ["srs_lite","srs","rating"])
    if srs_col:
        s = df_srs.loc[df_srs["season"] == season_year, srs_col].astype(float)
        if not s.empty:
            ranges["SRS"] = {"min": float(s.min()), "max": float(s.max()), "invert": False}

    sos_col = _pick_col(df_sos, ["sos_pct","sos","sos_rating","sos_index"])
    if sos_col:
        s = df_sos.loc[df_sos["season"] == season_year, sos_col].astype(float)
        if not s.empty:
            ranges["SoS"] = {"min": float(s.min()), "max": float(s.max()), "invert": False}

    if {"team_id","yards_per_play"}.issubset(df_team_games_season.columns):
        off_means = df_team_games_season.groupby("team_id")["yards_per_play"].mean().astype(float)
        if not off_means.empty:
            ranges["Off. YPP"] = {"min": float(off_means.min()), "max": float(off_means.max()), "invert": False}

        g = df_team_games_season[["game_id","team_id","yards_per_play"]].copy()
        opp = g.merge(g, on="game_id", suffixes=("", "_opp"))
        opp = opp[opp["team_id"] != opp["team_id_opp"]]
        if not opp.empty and "yards_per_play_opp" in opp.columns:
            def_means = opp.groupby("team_id")["yards_per_play_opp"].mean().astype(float)
            if not def_means.empty:
                ranges["Def. YPP"] = {"min": float(def_means.min()), "max": float(def_means.max()), "invert": True}

    for metric in ["SRS","SoS","Off. YPP","Def. YPP"]:
        if metric not in ranges:
            ranges[metric] = {"min": 0.0, "max": 1.0, "invert": (metric == "Def. YPP")}
    return ranges

def _season_present(df: pd.DataFrame, season_col: str, season: int) -> bool:
    return (not df.empty) and (season_col in df.columns) and (df[season_col] == season).any()

def choose_stats_season(season_year: int, *frames_with_season_col: tuple[pd.DataFrame, str]) -> tuple[int, bool]:
    """
    Return (stats_season, used_fallback).
    Chooses `season_year` if any frame has rows for it; otherwise chooses `season_year-1` if present.
    If neither present, returns (season_year, False).
    """
    has_current = any(_season_present(df, col, season_year) for df, col in frames_with_season_col)
    if has_current:
        return season_year, False

    prev = season_year - 1
    has_prev = any(_season_present(df, col, prev) for df, col in frames_with_season_col)
    if has_prev:
        return prev, True

    return season_year, False  # nothing available, keep current (will yield zeros)


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
    gid = int(game_id)

    resp_hist = (
        sb_client.table("nfl_historical_game_stats")
        .select("*")
        .eq("game_id", gid)
        .limit(1)
        .execute()
    )
    hist_rows = resp_hist.data or []

    if hist_rows:
        game = hist_rows[0]
        is_historical = True
    else:
        resp_sched = (
            sb_client.table("nfl_game_schedule")
            .select("*")
            .eq("game_id", gid)
            .limit(1)
            .execute()
        )
        sched_rows = resp_sched.data or []
        if not sched_rows:
            logger.error("No data in either table for game_id=%s – aborting", game_id)
            return
        game = sched_rows[0]
        is_historical = False

    # from here on, use `game` directly
    season_year = _safe_int(game.get("season"), default=0)
    if not season_year:
        season_year = pd.to_datetime(game["game_date"]).year
    home_id_raw, away_id_raw = game.get("home_team_id"), game.get("away_team_id")
    home_id_i = _safe_int(home_id_raw, default=None)
    away_id_i = _safe_int(away_id_raw, default=None)

    if home_id_i is None or away_id_i is None:
        # Build local index from existing tables (no extra HTTP calls)
        df_games_all   = fetch_data_from_table("nfl_historical_game_stats")
        df_season_team = fetch_data_from_table("nfl_historical_team_stats")
        team_index = _build_local_team_index(df_games_all, df_season_team)

        if home_id_i is None:
            home_id_i = _resolve_team_id_local(game, "home", team_index)
        if away_id_i is None:
            away_id_i = _resolve_team_id_local(game, "away", team_index)

        if home_id_i is None or away_id_i is None:
            logger.error(
                "Missing team IDs post-resolve (home=%r, away=%r) for game_id=%s — aborting snapshot",
                home_id_raw, away_id_raw, game_id
            )
            return

    # Keep string forms for places you compare as strings
    home_id, away_id = str(home_id_i), str(away_id_i)


    game_date = pd.to_datetime(game["game_date"]).date()


    # ────────────────────────── 2) Bulk dataframe pulls ──────────────────────
    df_games_all   = fetch_data_from_table("nfl_historical_game_stats")
    df_team_games  = fetch_data_from_table("nfl_historical_game_team_stats")   # ← no limit here
    df_season_team = fetch_data_from_table("nfl_historical_team_stats")
    df_srs         = fetch_data_from_table("v_nfl_team_srs_lite")
    df_sos         = fetch_data_from_table("v_nfl_team_sos")

    # Decide which season's stats to use (fallback to previous season if current has no rows)
    stats_season, used_fallback = choose_stats_season(
        season_year,
        (df_team_games,  "season"),
        (df_season_team, "season"),
        (df_srs,         "season"),
        (df_sos,         "season"),
    )

    if used_fallback:
        logger.warning("No rows found for season %s across key tables; falling back to %s for derived stats.",
                    season_year, stats_season)


    # **NEW**: grab only this season’s rows in a single query (no 1 000‑row cap)
    team_games_q = (
        sb_client.table("nfl_historical_game_team_stats")
        .select("*")
        .eq("season", stats_season)   # <— use stats_season here
        .execute()
    )
    df_team_games_season = pd.DataFrame(team_games_q.data or [])
    logger.info(
        "direct Supabase query → %d team-by-game rows for %d",
        len(df_team_games_season),
        stats_season
    )

    if df_team_games_season.empty:
        logger.warning(
            "Direct query returned 0 rows for season %s – falling back to local filter",
            stats_season,
        )
        df_team_games_season = df_team_games[df_team_games["season"] == stats_season]


    # Downstream safety columns
    if "yards_per_play" not in df_team_games_season.columns:
        df_team_games_season["yards_per_play"] = 0.0
    if "turnovers_total" not in df_team_games_season.columns:
        df_team_games_season["turnovers_total"] = 0
    # Boxscore columns used later in YPA section
    for col, default in [
        ("rushings_total", 0), ("rushings_attempts", 0),
        ("passing_total", 0), ("passing_comp_att", "")
    ]:
        if col not in df_team_games_season.columns:
            df_team_games_season[col] = default

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
   # a full-history DF with played_date_et
    df_full_history = fetch_data_from_table("nfl_historical_game_stats", "*")
    df_full_history["played_date_et"] = (
        pd.to_datetime(df_full_history["game_date"], errors="coerce").dt.date
    )

    rest_home = _compute_rest_days(df_full_history, home_id, game_date)
    rest_away = _compute_rest_days(df_full_history, away_id, game_date)
    rest_advantage = rest_home - rest_away
    logger.info(
        "NFL Rest: Home=%d days, Away=%d days → Adv=%d",
        rest_home,
        rest_away,
        rest_advantage
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
        season_ok = df_games_all["season"] == stats_season

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

    home_row = df_season_team.query(
        "team_id == @home_id_i and season == @stats_season"
    ).head(1)
    away_row = df_season_team.query(
        "team_id == @away_id_i and season == @stats_season"
    ).head(1)

    # ── dump the full rows for both teams ─────────────────────────────────────
    if not home_row.empty:
        row = home_row.iloc[0].to_dict()
        logger.info("Home season row: %s", row)
        logger.info("Home record_home → %s", row.get("record_home"))
    else:
        logger.info("Home season row: <NONE>")

    if not away_row.empty:
        row = away_row.iloc[0].to_dict()
        logger.info("Away season row: %s", row)
        logger.info("Away record_road → %s", row.get("record_road"))
    else:
        logger.info("Away season row: <NONE>")


    # ── compute the pct and diff ─────────────────────────────────────────────
    home_pct = _venue_win_pct(home_row.iloc[0] if not home_row.empty else None, home_id, "home")
    away_pct = _venue_win_pct(away_row.iloc[0] if not away_row.empty else None, away_id, "road")
    venue_win_pct_diff = home_pct - away_pct

    logger.info(
        "Venue win%% check → home_pct=%.3f  away_pct=%.3f  diff=%.3f",
        home_pct, away_pct, venue_win_pct_diff
    )

    # If the split diff is 0, fall back to overall record
    if venue_win_pct_diff == 0:
        # Pull overall wins/games from df_season_team
        row_home = home_row.iloc[0] if not home_row.empty else None
        row_away = away_row.iloc[0] if not away_row.empty else None

        def _overall_pct(row):
            if row is None: return 0.0
            wins = row.get("won", 0)
            losses = row.get("lost", 0)
            ties = row.get("ties", 0)
            gp = wins + losses + ties
            return (wins + 0.5 * ties) / gp if gp else 0.0

        overall_home = _overall_pct(row_home)
        overall_away = _overall_pct(row_away)
        venue_win_pct_diff = overall_home - overall_away

        logger.info(
            "Home/Road tied – falling back to overall pct → home=%.3f away=%.3f diff=%.3f",
            overall_home, overall_away, venue_win_pct_diff
        )


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
    delta_pct = (home_pct - away_pct) * 100

    season_label = str(season_year)
    if used_fallback and stats_season != season_year:
        season_label += f" (using {stats_season} stats)"

    headlines = [
        {"label": "Season", "value": season_label},
        {"label": "Rest Days (Home)",      "value": rest_home},
        {"label": "Rest Days (Away)",      "value": rest_away},
        {"label": "Turnover Margin Difference",   "value": sanitize_float(round(
                                                    team_metrics[home_id]["turnover_diff_pg"] -
                                                    team_metrics[away_id]["turnover_diff_pg"], 2))},
    {
        "label": "Home Win % vs Road Win %",
        "value": f"{home_pct:.1%} / {away_pct:.1%}  (Δ {(home_pct - away_pct):+.1%})"
    },
    {
        "label": "Red Zone TD% Difference",
        "value": sanitize_float(round(rz_td_diff, 3))
    },
            ]

    # ────────────────────────── 11) Quarter-average bar chart ────────────────
    def q_avg(team_id: str, season: int) -> dict[str, float]:
        t = int(team_id)

        # Only Q1–Q4 (no OT)
        home_cols = ["home_q1","home_q2","home_q3","home_q4"]
        away_cols = ["away_q1","away_q2","away_q3","away_q4"]

        h = df_games_all.loc[
            (df_games_all["season"] == season) & (df_games_all["home_team_id"] == t),
            home_cols
        ].rename(columns={
            "home_q1":"q1","home_q2":"q2","home_q3":"q3","home_q4":"q4"
        })

        a = df_games_all.loc[
            (df_games_all["season"] == season) & (df_games_all["away_team_id"] == t),
            away_cols
        ].rename(columns={
            "away_q1":"q1","away_q2":"q2","away_q3":"q3","away_q4":"q4"
        })

        df = pd.concat([h, a], axis=0)
        if df.empty:
            return {"q1":0.0, "q2":0.0, "q3":0.0, "q4":0.0}

        # Average points scored per quarter across all games for that season
        m = df.mean(numeric_only=True).fillna(0).to_dict()
        return {k: float(m.get(k, 0.0)) for k in ("q1","q2","q3","q4")}

    # Use the same season you selected for all other derived stats
    home_quarters = q_avg(home_id, stats_season)
    away_quarters = q_avg(away_id, stats_season)

    bar_chart_data = [
        {"category": q, "Home": round(home_quarters.get(q, 0), 2), "Away": round(away_quarters.get(q, 0), 2)}
        for q in ("q1","q2","q3","q4")
    ]



    # ────────────────────────── 12) Radar chart (SRS / SoS / YPP) ────────────
    def _value(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        return sanitize_float(df.iloc[0][col], default) if not df.empty else default

    home_srs = _value(df_srs[(df_srs["team_id"] == home_id_i) & (df_srs["season"] == stats_season)], _pick_col(df_srs, ["srs_lite","srs","rating"]) or "srs_lite")
    away_srs = _value(df_srs[(df_srs["team_id"] == away_id_i) & (df_srs["season"] == stats_season)], _pick_col(df_srs, ["srs_lite","srs","rating"]) or "srs_lite")
    home_sos = _value(df_sos[(df_sos["team_id"] == home_id_i) & (df_sos["season"] == stats_season)], _pick_col(df_sos, ["sos_pct","sos","sos_rating","sos_index"]) or "sos")
    away_sos = _value(df_sos[(df_sos["team_id"] == away_id_i) & (df_sos["season"] == stats_season)], _pick_col(df_sos, ["sos_pct","sos","sos_rating","sos_index"]) or "sos")


    radar_metrics = {
        "SRS"      : (home_srs, home_srs - away_srs),
        "SoS"      : (home_sos, home_sos - away_sos),
        "Off. YPP" : (team_metrics[home_id]["off_ypp"], team_metrics[home_id]["off_ypp"] - team_metrics[away_id]["off_ypp"]),
        "Def. YPP" : (team_metrics[home_id]["def_ypp"], team_metrics[home_id]["def_ypp"] - team_metrics[away_id]["def_ypp"]),
    }

    # Pull league ranges (min/max) – prefer RPC, fallback to local compute
    def _to_float(x, default=None):
        try:
            if x is None:
                return default
            f = float(x)
            if math.isnan(f):
                return default
            return f
        except Exception:
            return default

    league_ranges = {}
    try:
        ranges_rpc = sb_client.rpc("get_nfl_metric_ranges", {"p_season": stats_season}).execute()
        rpc_rows = ranges_rpc.data or []
        wanted = {"SRS", "SoS", "Off. YPP", "Def. YPP"}
        for r in rpc_rows:
            m = r.get("metric")
            if m not in wanted:
                continue
            vmin = _to_float(r.get("min"), default=None)
            vmax = _to_float(r.get("max"), default=None)
            # Some Supabase clients return keys as 'min'/'max', others 'min_value'/'max_value'. Try both.
            if vmin is None and "min_value" in r:
                vmin = _to_float(r.get("min_value"), default=None)
            if vmax is None and "max_value" in r:
                vmax = _to_float(r.get("max_value"), default=None)
            if vmin is None or vmax is None:
                continue  # skip bad rows; we'll fill via local compute
            league_ranges[m] = {"min": vmin, "max": vmax, "invert": (m == "Def. YPP")}
    except Exception as e:
        logger.warning("Ranges RPC failed, will compute locally: %s", e)

    if not league_ranges or any(k not in league_ranges for k in ("SRS","SoS","Off. YPP","Def. YPP")):
        # Fill gaps with local computation
        missing = [k for k in ("SRS","SoS","Off. YPP","Def. YPP") if k not in league_ranges]
        if missing:
            logger.info("Computing local ranges for: %s", ", ".join(missing))
        local_ranges = _compute_league_ranges_local(season_year, df_srs, df_sos, df_team_games_season)
        for k in missing:
            league_ranges[k] = local_ranges.get(k, {"min": 0.0, "max": 1.0, "invert": (k == "Def. YPP")})


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

    # --- Pie Chart (Scoring Averages) [Safe version] ---
    def _first_or_none(df: pd.DataFrame):
        return df.iloc[0] if not df.empty else None

    def _avg_pf_pa(row):
        if row is None:
            return 0.0, 0.0
        gp = (row.get('won', 0) + row.get('lost', 0) + row.get('ties', 0)) or 0
        if gp == 0:
            return 0.0, 0.0
        pf_avg = row.get('points_for', 0) / gp
        pa_avg = row.get('points_against', 0) / gp
        return pf_avg, pa_avg

    home_season_row = _first_or_none(df_season_team[(df_season_team['team_id'] == home_id_i) & (df_season_team['season'] == stats_season)])
    away_season_row = _first_or_none(df_season_team[(df_season_team['team_id'] == away_id_i) & (df_season_team['season'] == stats_season)])


    home_pf_avg, home_pa_avg = _avg_pf_pa(home_season_row)
    away_pf_avg, away_pa_avg = _avg_pf_pa(away_season_row)

    home_pf_label = f"Home ({sanitize_float(round(home_pf_avg, 1), 0.0)})"
    away_pf_label = f"Away ({sanitize_float(round(away_pf_avg, 1), 0.0)})"
    home_pa_label = f"Home ({sanitize_float(round(home_pa_avg, 1), 0.0)})"
    away_pa_label = f"Away ({sanitize_float(round(away_pa_avg, 1), 0.0)})"
        
    pie_chart_data = [
        {"title": "Avg Points For", "data": [{"category": home_pf_label, "value": home_pf_avg, "color": "#4ade80"}, {"category": away_pf_label, "value": away_pf_avg, "color": "#60a5fa"}]},
        {"title": "Avg Points Against", "data": [{"category": home_pa_label, "value": home_pa_avg, "color": "#4ade80"}, {"category": away_pa_label, "value": away_pa_avg, "color": "#60a5fa"}]}
    ]

    # --- Key Metrics Bar Chart (Yards Per Attempt) ----------------------------

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
    home_team_games = df_team_games_season[df_team_games_season["team_id"] == home_id_i]

    h_total_rush_yards    = home_team_games["rushings_total"].sum()
    h_total_rush_attempts = home_team_games["rushings_attempts"].sum()
    h_rush_ypa            = (h_total_rush_yards / h_total_rush_attempts) if h_total_rush_attempts else 0.0

    h_total_pass_yards    = home_team_games["passing_total"].sum()
    h_total_pass_attempts = extract_attempts(home_team_games["passing_comp_att"]).sum()
    h_pass_ypa            = (h_total_pass_yards / h_total_pass_attempts) if h_total_pass_attempts else 0.0

    # ─── Away calculations ────────────────────────────────────────────────────
    away_team_games = df_team_games_season[df_team_games_season["team_id"] == away_id_i]

    a_total_rush_yards    = away_team_games["rushings_total"].sum()
    a_total_rush_attempts = away_team_games["rushings_attempts"].sum()
    a_rush_ypa            = (a_total_rush_yards / a_total_rush_attempts) if a_total_rush_attempts else 0.0

    a_total_pass_yards    = away_team_games["passing_total"].sum()
    a_total_pass_attempts = extract_attempts(away_team_games["passing_comp_att"]).sum()
    a_pass_ypa            = (a_total_pass_yards / a_total_pass_attempts) if a_total_pass_attempts else 0.0

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
        "radar_chart_data": radar_chart_data,
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