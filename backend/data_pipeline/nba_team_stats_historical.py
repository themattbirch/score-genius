# backend/data_pipeline/nba_team_stats_historical.py

import json
from zoneinfo import ZoneInfo
import requests
import sys
import os
import time
import traceback
from datetime import datetime, timedelta

# Add the backend root to Python path for caching & config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# --- Local Config & Variables ---
try:
    from config import (
        API_SPORTS_KEY,
        ODDS_API_KEY,
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )
    print("Successfully imported configuration variables from config.py")
except ImportError:
    print("config.py not found → loading credentials from environment")
    API_SPORTS_KEY       = os.getenv("API_SPORTS_KEY")
    ODDS_API_KEY         = os.getenv("ODDS_API_KEY")
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate
_missing = [
    name
    for name, val in [
        ("API_SPORTS_KEY",       API_SPORTS_KEY),
        ("ODDS_API_KEY",         ODDS_API_KEY),
        ("SUPABASE_URL",         SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
    ]
    if not val
]
if _missing:
    print(f"FATAL ERROR: Missing required config/env vars: {', '.join(_missing)}")
    sys.exit(1)


HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from caching.supabase_client import supabase
from backend.data_pipeline.nba_game_stats_historical import parse_season_for_date


API_KEY = API_SPORTS_KEY
BASE_URL = 'https://v1.basketball.api-sports.io'
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v1.basketball.api-sports.io'
}

##############################################################################
# Helper Functions
##############################################################################

def get_nested_value(data, *keys, default=None):
    """
    Safely navigate nested dictionaries, returning default if path doesn't exist.
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default

def get_current_season():
    """
    Determine the current NBA season based on today's date.
    """
    today = datetime.now()
    if today.month >= 7 and today.month <= 9:
        return f"{today.year}-{today.year + 1}"
    elif today.month >= 10:
        return f"{today.year}-{today.year + 1}"
    else:
        return f"{today.year - 1}-{today.year}"

def is_regular_team(team_name):
    """
    Determine if a team is a regular NBA team or a special event team (All-Star, Rising Stars, etc.).
    Returns True for regular teams, False for special event teams.
    """
    if not team_name:
        return False
        
    # List of keywords that indicate special event teams
    special_event_keywords = [
        "rising stars", 
        "global stars", 
        "young stars", 
        "ogs",
        "all-stars",
        "all stars",
        "rookies",
        "sophomores",
        "world",
        "usa"
    ]
    
    # Check if team name contains any of the special event keywords
    team_name_lower = team_name.lower()
    for keyword in special_event_keywords:
        if keyword in team_name_lower:
            return False
    
    # Check if "team" is in the name (All-Star teams often have this pattern)
    if "team" in team_name_lower:
        return False
        
    # Specific exclusions (add more as needed)
    specific_exclusions = [
        "candace's",
        "chuck's",
        "kenny's",
        "shaq's"
    ]
    
    for exclusion in specific_exclusions:
        if exclusion in team_name_lower:
            return False
    
    # If we've made it here, it's likely a regular team
    return True

##############################################################################
# 1) Fetch from the API
##############################################################################

def get_teams_by_league_season(league: str, season: str) -> list:
    """
    Fetches a list of teams for the given league and season.
    Returns the "response" list from the JSON.
    """
    url = f"{BASE_URL}/teams"
    params = {"league": league, "season": season}
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        print(f"Fetched {len(data.get('response', []))} teams for league {league}, season {season}")
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching teams for league {league}, season {season}: {e}")
        return []

import json  # Make sure to add this import at the top

def get_team_stats(team_id: int, league: str, season: str) -> dict:
    """
    Fetches team statistics from /statistics for a given team, league, and season.
    """
    url = f"{BASE_URL}/statistics"
    params = {
        "team": team_id,
        "league": league,
        "season": season
    }
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
               
        print(f"Fetched stats for team_id={team_id}, league={league}, season={season}")
        # Return directly the response object, not the full dictionary
        # The statistics endpoint returns a single object, not a list
        return data.get("response", {})
    except Exception as e:
        print(f"Error fetching team stats for team_id={team_id}, league={league}, season={season}: {e}")
        return {}

##############################################################################
# 2) Transform Team Stats
##############################################################################

def transform_team_stats(team: dict, stats: dict, league_id: str, season: str) -> dict:
    """
    Transform raw team stats data into the expected record format for our database.
    """
    # Extract basic team info
    team_id = team.get("id")
    team_name = team.get("name")
    
    # Extract games statistics
    games = stats.get("games", {})
    games_played_home = get_nested_value(games, "played", "home", default=0)
    games_played_away = get_nested_value(games, "played", "away", default=0)
    games_played_all = get_nested_value(games, "played", "all", default=0)
    
    # Extract wins statistics
    wins_home_total = get_nested_value(games, "wins", "home", "total", default=0)
    wins_home_percentage = get_nested_value(games, "wins", "home", "percentage", default=0)
    wins_away_total = get_nested_value(games, "wins", "away", "total", default=0)
    wins_away_percentage = get_nested_value(games, "wins", "away", "percentage", default=0)
    wins_all_total = get_nested_value(games, "wins", "all", "total", default=0)
    wins_all_percentage = get_nested_value(games, "wins", "all", "percentage", default=0)
    
    # Extract losses statistics - API might use either "loses" or "losses" as the key
    losses_home_total = get_nested_value(games, "loses", "home", "total", default=0) or get_nested_value(games, "losses", "home", "total", default=0)
    losses_home_percentage = get_nested_value(games, "loses", "home", "percentage", default=0) or get_nested_value(games, "losses", "home", "percentage", default=0)
    losses_away_total = get_nested_value(games, "loses", "away", "total", default=0) or get_nested_value(games, "losses", "away", "total", default=0)
    losses_away_percentage = get_nested_value(games, "loses", "away", "percentage", default=0) or get_nested_value(games, "losses", "away", "percentage", default=0)
    losses_all_total = get_nested_value(games, "loses", "all", "total", default=0) or get_nested_value(games, "losses", "all", "total", default=0)
    losses_all_percentage = get_nested_value(games, "loses", "all", "percentage", default=0) or get_nested_value(games, "losses", "all", "percentage", default=0)
    
    # Extract points for statistics
    points = stats.get("points", {})
    points_for_total_home = get_nested_value(points, "for", "total", "home", default=0)
    points_for_total_away = get_nested_value(points, "for", "total", "away", default=0)
    points_for_total_all = get_nested_value(points, "for", "total", "all", default=0)
    points_for_avg_home = get_nested_value(points, "for", "average", "home", default=0)
    points_for_avg_away = get_nested_value(points, "for", "average", "away", default=0)
    points_for_avg_all = get_nested_value(points, "for", "average", "all", default=0)
    
    # Extract points against statistics
    points_against_total_home = get_nested_value(points, "against", "total", "home", default=0)
    points_against_total_away = get_nested_value(points, "against", "total", "away", default=0)
    points_against_total_all = get_nested_value(points, "against", "total", "all", default=0)
    points_against_avg_home = get_nested_value(points, "against", "average", "home", default=0)
    points_against_avg_away = get_nested_value(points, "against", "average", "away", default=0)
    points_against_avg_all = get_nested_value(points, "against", "average", "all", default=0)
    
    # Extract form/streak data (NEW)
    form = get_nested_value(stats, "form", default="")
    
    # Calculate streak based on form (NEW)
    streak = 0
    if form:
        # Count consecutive wins or losses at the end of the form string
        # Example: "WWLWW" would give a streak of 2 (wins)
        streak_char = form[-1] if form else ""
        for i in range(len(form)-1, -1, -1):
            if form[i] != streak_char:
                break
            streak += 1
        # Make streak negative for losses
        if streak_char.upper() == "L":
            streak = -streak
    
    record = {
        "team_id": team_id,
        "team_name": team_name,
        "season": season,
        "league_id": league_id,
        
        # Games played statistics
        "games_played_home": games_played_home,
        "games_played_away": games_played_away,
        "games_played_all": games_played_all,
        
        # Wins statistics
        "wins_home_total": wins_home_total,
        "wins_home_percentage": wins_home_percentage,
        "wins_away_total": wins_away_total,
        "wins_away_percentage": wins_away_percentage,
        "wins_all_total": wins_all_total,
        "wins_all_percentage": wins_all_percentage,
        
        # Losses statistics
        "losses_home_total": losses_home_total,
        "losses_home_percentage": losses_home_percentage,
        "losses_away_total": losses_away_total,
        "losses_away_percentage": losses_away_percentage,
        "losses_all_total": losses_all_total,
        "losses_all_percentage": losses_all_percentage,
        
        # Points for statistics
        "points_for_total_home": points_for_total_home,
        "points_for_total_away": points_for_total_away,
        "points_for_total_all": points_for_total_all,
        "points_for_avg_home": points_for_avg_home,
        "points_for_avg_away": points_for_avg_away,
        "points_for_avg_all": points_for_avg_all,
        
        # Points against statistics
        "points_against_total_home": points_against_total_home,
        "points_against_total_away": points_against_total_away,
        "points_against_total_all": points_against_total_all,
        "points_against_avg_home": points_against_avg_home,
        "points_against_avg_away": points_against_avg_away,
        "points_against_avg_all": points_against_avg_all,
        
        # NEW FIELDS
        "current_form": form,
        
        # Metadata
        "updated_at": datetime.now().isoformat()
    }
    
    return record

##############################################################################
# 3) Upsert to Supabase
##############################################################################

def upsert_historical_team_stats(record):
    """
    Upsert team statistics to the nba_historical_team_stats table.
    """
    try:
        result = (
            supabase
            .table("nba_historical_team_stats")
            .upsert(
                record,
                on_conflict="team_id,season,updated_at"   # ← changed
            )
            .execute()
        )
        return result
    except Exception as e:
        print(f"Error upserting team stats: {e}")
        return None

##############################################################################
# 4) Process Teams for Season
##############################################################################

def calculate_team_form(team_id, team_name, num_games=5):
    """
    Calculate a team's form based on their most recent games.
    
    Args:
        team_id (int): The ID of the team
        team_name (str): The name of the team (for matching in game records)
        num_games (int): Number of recent games to include in form (default: 5)
        
    Returns:
        str: A string representation of recent form (e.g., "WLWWL")
    """
    try:
        # Query recent games involving this team
        # Use the correct order syntax for your Supabase client
        response = supabase.table("nba_historical_game_stats").select("*").or_(
            f"home_team.eq.{team_name},away_team.eq.{team_name}"
        ).order('game_date.desc').limit(num_games).execute()
        
        games = response.data
        
        # If no games found, return empty form
        if not games:
            print(f"No recent games found for team: {team_name}")
            return ""
        
        form = ""
        for game in games:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            is_home_team = home_team == team_name
            
            # Get relevant scores
            if is_home_team:
                team_score = game.get('home_score', 0)
                opponent_score = game.get('away_score', 0)
            else:
                team_score = game.get('away_score', 0)
                opponent_score = game.get('home_score', 0)
            
            # Determine win or loss
            if team_score > opponent_score:
                form += "W"  # Add win at the end
            else:
                form += "L"  # Add loss at the end
        
        return form
    
    except Exception as e:
        print(f"Error calculating form for team {team_name}: {e}")
        traceback.print_exc()  # Added stacktrace to see more details
        return ""

def process_teams_for_season(league_id: str, season: str) -> int:
    """
    Fetch every regular NBA team in `league_id` / `season`, pull its latest
    season‑to‑date numbers from the API (plus a custom 5‑game “form” string),
    and return how many teams were processed.

    NOTE:  • No database writes happen here.
           • `nba_historical_team_stats` is now populated exclusively by the
             daily snapshot built in `snapshot_yesterday()`.
    """
    print(f"\n=== Processing teams for league {league_id}, season {season} ===")

    # ---------------------------------------------------------------- fetch list
    teams_list = get_teams_by_league_season(league_id, season)
    if not teams_list:
        print("No teams returned by /teams – aborting.")
        return 0

    # ----------------------------------------------------------- filter specials
    regular_teams, specials = [], 0
    for t in teams_list:
        if is_regular_team(t.get("name", "")):
            regular_teams.append(t)
        else:
            specials += 1
    print(f"Filtered out {specials} special‑event teams. "
          f"Processing {len(regular_teams)} regular teams.")

    # ---------------------------------------------------------------- per team
    processed = 0
    for team in regular_teams:
        team_id   = team.get("id")
        team_name = team.get("name")
        print(f"→  {team_name} (ID {team_id})")

        stats = get_team_stats(team_id, league_id, season)
        if not stats:
            print(f"   ⚠︎  No /statistics payload – skipping.")
            continue

        # keep transform + form so any later code using this info still works
        record = transform_team_stats(team, stats, league_id, season)
        record["current_form"] = calculate_team_form(team_id, team_name)

        # (no database write here – snapshot_yesterday() will handle it)

        processed += 1
        time.sleep(1)      # stay well below API‑Sports rate limits

    print(f"Processed {processed} teams for league {league_id}, season {season}")
    return processed

##############################################################################
# 5) Main Runner
##############################################################################

def main():
    # Set your date range here
    start_date = datetime(2025, 3, 10)
    end_date = datetime(2025, 3, 17)
    
    # NBA League ID19
    league_id = "12"
    
    # Get unique seasons between start_date and end_date
    seasons = set()
    current = start_date
    while current <= end_date:
        year = current.year
        month = current.month
        
        # Determine the season based on the date
        if month >= 10:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"
        
        seasons.add(season)
        current += timedelta(days=1)
    
    # Convert to sorted list
    seasons_list = sorted(list(seasons))
    
    print(f"Starting historical team stats import from {start_date} to {end_date}")
    print(f"Seasons to process: {', '.join(seasons_list)}")
    
    total_processed = 0
    for season in seasons_list:
        count = process_teams_for_season(league_id, season)
        total_processed += count
        # Sleep between seasons to avoid rate limiting
        time.sleep(5)
    
    print(f"\nCompleted processing historical TEAM-level stats for {total_processed} teams.")

# ---------------------------------------------------------------------------
# ⬇⬇⬇  SNAPSHOT HELPERS  ⬇⬇⬇
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import pandas as pd

SNAP_FORM_GAMES = 10          # last‑N games to build “current_form”

def _pull_games_upto(season: str, cutoff_dt: datetime) -> pd.DataFrame:
    """
    Grab every game in `season` whose game_date < cutoff_dt (strict).
    """
    resp = (
        supabase.table("nba_historical_game_stats")
        .select("*")
        .eq("season", season)
        .lt("game_date", cutoff_dt.isoformat())
        .execute()
    )
    return pd.DataFrame(resp.data or [])

def _build_form(win_series: pd.Series, n: int = SNAP_FORM_GAMES) -> str:
    """
    Turn the last N booleans (True=win) into a form string like "WLWWL".
    """
    if win_series.empty:
        return ""
    return "".join("W" if w else "L" for w in win_series.tail(n))

def _agg_one(grp: pd.DataFrame) -> dict:
    """
    Aggregate a single team’s rows (either home or away slice).
    Expects columns: team_score, opp_score, is_win
    """
    g_cnt   = len(grp)
    wins    = int(grp["is_win"].sum())
    losses  = g_cnt - wins
    win_pct = wins / g_cnt if g_cnt else 0.5

    return {
        "games_played_all":      g_cnt,
        "wins_all_total":        wins,
        "wins_all_percentage":   round(win_pct, 3),
        "losses_all_total":      losses,
        "losses_all_percentage": round(1 - win_pct, 3),
        "points_for_avg_all":        round(grp["team_score"].mean(),     1),
        "points_against_avg_all":    round(grp["opp_score"].mean(),      1),
        "current_form":          _build_form(grp["is_win"]),
    }

def _aggregate_team(snapshot_dt: datetime, gdf: pd.DataFrame, team_id: int, team_name: str) -> dict:
    """
    Season‑to‑date roll‑up for a single team (home+away combined).
    """
    is_home = gdf["home_team_id"] == team_id
    is_away = gdf["away_team_id"] == team_id
    played  = gdf[is_home | is_away].copy()

    if played.empty:
        return {}

    # build win/loss & score cols vectorised
    played["team_score"] = played["home_score"].where(is_home, played["away_score"])
    played["opp_score"]  = played["away_score"].where(is_home, played["home_score"])
    played["is_win"]     = played["team_score"] > played["opp_score"]

    base = _agg_one(played)

    season_lbl = played["season"].iloc[0]  # homogeneous by design

    return {
        "team_id":     team_id,
        "team_name":   team_name,
        "season":      season_lbl,
        "league_id":   "12",                                   # NBA
        "updated_at":  snapshot_dt.isoformat(timespec="seconds"),
        **base,
    }

# ---------- nightly (00:00 ET) snapshot ------------------------------------
def snapshot_yesterday() -> None:
    et       = ZoneInfo("America/New_York")
    snap_dt  = datetime.now(et).replace(hour=0, minute=0, second=0, microsecond=0)
    season   = parse_season_for_date(snap_dt - timedelta(days=1))
    print(f"\n=== Building season‑to‑date snapshot ({season}) at {snap_dt} ===")

    # 1️⃣ pull the id ↔ name map (built by your per‑team fetcher)
    id_rows = (
        supabase.table("nba_historical_team_stats")
        .select("team_id,team_name")
        .eq("season", season)
        .execute()
        .data or []
    )
    id_map = {r["team_name"]: r["team_id"] for r in id_rows}
    if not id_map:
        print("[WARN] Season rows missing – seed first, then snapshot.")
        return

    # 2️⃣ pull games up to the snapshot moment
    gdf = _pull_games_upto(season, snap_dt)
    if gdf.empty:
        print("No games yet – skipping snapshot.")
        return

    # attach team‑id columns once for fast filtering
    gdf["home_team_id"] = gdf["home_team"].map(id_map).fillna(-1).astype(int)
    gdf["away_team_id"] = gdf["away_team"].map(id_map).fillna(-1).astype(int)

    # 3️⃣ aggregate per team
    rows = []
    for name, tid in id_map.items():
        snap = _aggregate_team(snap_dt, gdf, tid, name)
        if snap:
            rows.append(snap)

    # 4️⃣ upsert on (team_id, season, league_id) – matches unique index
    if rows:
        supabase.table("nba_historical_team_stats") \
            .upsert(rows, on_conflict="team_id,season,league_id") \
            .execute()
        print(f"Snapshot rows upserted: {len(rows)}")
    else:
        print("Nothing to upsert.")

# ---------- one‑off back‑fill helper ---------------------------------------
# ---------------------------------------------------------------------------
# season‑final back‑fill (no leakage, one row / team)  ───────────────────────
# ---------------------------------------------------------------------------
def snapshot_full_season(season_label: str) -> None:
    """
    Build a single snapshot dated *after* the last game of the given season.
    Ideal for historical back‑fill: one row per team with final season averages.

    Uses name→id map from any row already in nba_historical_team_stats
    for that season.  (If you’ve never inserted any rows for the season,
    run the normal team‑stats ETL once first.)
    """
    # 0) id‑map ────────────────────────────────────────────────────────────
    id_rows = (
        supabase.table("nba_historical_team_stats")
        .select("team_id,team_name")
        .eq("season", season_label)
        .execute()
        .data or []
    )
    id_map = {r["team_name"].lower(): r["team_id"] for r in id_rows}
    if not id_map:
        print(f"[WARN] No (team_name→id) rows yet for season {season_label}. "
              "Seed the season first and retry.")
        return

    # 1) pull every game in that season ───────────────────────────────────
    gdf = _pull_games_upto(
        season_label,
        cutoff_dt=datetime.max.replace(tzinfo=ZoneInfo("UTC"))
    )
    if gdf.empty:
        print(f"No games for season {season_label}")
        return

    # 2) add temp id columns so _aggregate_team can work -------------------
    gdf["home_team_id"] = gdf["home_team"].str.lower().map(id_map).fillna(-1).astype(int)
    gdf["away_team_id"] = gdf["away_team"].str.lower().map(id_map).fillna(-1).astype(int)

    last_dt = pd.to_datetime(gdf["game_date"]).max()

    rows = []
    for team_name_lower, team_id in id_map.items():
        # recover pretty name once
        pretty_name = next(r["team_name"] for r in id_rows if r["team_id"] == team_id)

        snap = _aggregate_team(
            snapshot_dt=last_dt + timedelta(seconds=1),   # 1‑sec after final buzzer
            gdf=gdf,
            team_id=team_id,
            team_name=pretty_name
        )
        if snap:
            rows.append(snap)

    if not rows:
        print(f"No rows produced for season {season_label}")
        return

    supabase.table("nba_historical_team_stats") \
            .upsert(rows, on_conflict="team_id,season,league_id") \
            .execute()

    print(f"Back‑filled {len(rows)} team snapshots for season {season_label}")

# ---------------------------------------------------------------------------
# keep daily snapshot after the main per‑game import
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()               # ← existing game importer
    snapshot_yesterday() # nightly leak‑safe snapshot
