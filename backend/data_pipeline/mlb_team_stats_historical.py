import requests
import os
import json
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any

# Import shared Supabase client
try:
    from caching.supabase_client import supabase
    from supabase import Client  # for type hints
except ImportError:
    print("FATAL ERROR: Could not import shared Supabase client.")
    exit(1)

# Load API key
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
if not API_SPORTS_KEY:
    print("FATAL ERROR: API_SPORTS_KEY not found")
    exit(1)

# Configuration
API_BASE_URL = "https://v1.baseball.api-sports.io"
MLB_LEAGUE_ID = 1
TARGET_SEASONS = [2025, 2024, 2023, 2022, 2021]
SUPABASE_TABLE_NAME = "mlb_historical_team_stats"
REQUEST_DELAY_SECONDS = 2
HEADERS = {'x-apisports-key': API_SPORTS_KEY}

def make_api_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        rl_rem = resp.headers.get('x-ratelimit-requests-remaining')
        rl_lim = resp.headers.get('x-ratelimit-requests-limit')
        print(f"API hit for {params}. Rate limit: {rl_rem}/{rl_lim}")
        return resp.json()
    except Exception as e:
        print(f"API request error: {e}")
        return None

def get_teams_for_season(league_id: int, season: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/teams"
    params = {"league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list):
        return data["response"]
    print(f"No teams for season {season}")
    return []

def get_team_stats(team_id: int, league_id: int, season: int, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/teams/statistics"
    params = {"team": team_id, "league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    return data.get("response") if data else None

def safe_float_conversion(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace('%', '')
        return float(value)
    except:
        return None

def transform_stats_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts all the same fields you had before, but
    does *not* yet calculate form—form will be
    injected in the main loop just like the NBA script.
    """
    try:
        games = api_data.get("games", {})
        points = api_data.get("points", {})
        team = api_data.get("team", {})
        league = api_data.get("league", {})

        def get_nested(d, keys, default=None):
            for k in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(k, default)
            return d

        record = {
            'team_id': team.get('id'),
            'team_name': team.get('name'),
            'season': league.get('season'),
            'league_id': league.get('id'),
            'league_name': league.get('name'),

            # games played
            'games_played_home': get_nested(games, ['played','home']),
            'games_played_away': get_nested(games, ['played','away']),
            'games_played_all': get_nested(games, ['played','all']),

            # wins
            'wins_home_total': get_nested(games, ['wins','home','total']),
            'wins_home_percentage': safe_float_conversion(get_nested(games, ['wins','home','percentage'])),
            'wins_away_total': get_nested(games, ['wins','away','total']),
            'wins_away_percentage': safe_float_conversion(get_nested(games, ['wins','away','percentage'])),
            'wins_all_total': get_nested(games, ['wins','all','total']),
            'wins_all_percentage': safe_float_conversion(get_nested(games, ['wins','all','percentage'])),

            # losses
            'losses_home_total': get_nested(games, ['loses','home','total']),
            'losses_home_percentage': safe_float_conversion(get_nested(games, ['loses','home','percentage'])),
            'losses_away_total': get_nested(games, ['loses','away','total']),
            'losses_away_percentage': safe_float_conversion(get_nested(games, ['loses','away','percentage'])),
            'losses_all_total': get_nested(games, ['loses','all','total']),
            'losses_all_percentage': safe_float_conversion(get_nested(games, ['loses','all','percentage'])),

            # runs for
            'runs_for_total_home': get_nested(points, ['for','total','home']),
            'runs_for_total_away': get_nested(points, ['for','total','away']),
            'runs_for_total_all': get_nested(points, ['for','total','all']),
            'runs_for_avg_home': safe_float_conversion(get_nested(points, ['for','average','home'])),
            'runs_for_avg_away': safe_float_conversion(get_nested(points, ['for','average','away'])),
            'runs_for_avg_all': safe_float_conversion(get_nested(points, ['for','average','all'])),

            # runs against
            'runs_against_total_home': get_nested(points, ['against','total','home']),
            'runs_against_total_away': get_nested(points, ['against','total','away']),
            'runs_against_total_all': get_nested(points, ['against','total','all']),
            'runs_against_avg_home': safe_float_conversion(get_nested(points, ['against','average','home'])),
            'runs_against_avg_away': safe_float_conversion(get_nested(points, ['against','average','away'])),
            'runs_against_avg_all': safe_float_conversion(get_nested(points, ['against','average','all'])),

            # keep the raw payload if you like
            'raw_api_response': json.dumps({"games": games, "points": points}),
        }

        # sanity check
        if not (record['team_id'] and record['season'] and record['league_id']):
            return None
        return record

    except Exception as e:
        print(f"Error transforming stats: {e}")
        return None

def upsert_team_stats(supabase_client: Client, stats_data: Dict[str, Any]):
    """Exactly as your NBA upsert does it"""
    try:
        res = supabase_client.table(SUPABASE_TABLE_NAME) \
            .upsert(stats_data, on_conflict="team_id,season,league_id") \
            .execute()
        if hasattr(res, 'error') and res.error:
            print(f"Upsert error: {res.error}")
        else:
            print(f"Upserted team {stats_data['team_id']} season {stats_data['season']}")
    except Exception as e:
        print(f"Supabase upsert exception: {e}")

def calculate_team_form(team_id: int, num_games: int = 5) -> str:
    """
    Calculate a team's recent form (a string of W/L) by pulling
    the last `num_games` from mlb_historical_game_stats.
    """
    try:
        resp = supabase.table("mlb_historical_game_stats") \
            .select("home_team_id,away_team_id,home_score,away_score,game_date_time_utc") \
            .or_(f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}") \
            .order("game_date_time_utc.desc") \
            .limit(num_games) \
            .execute()

        games = resp.data or []
        if not games:
            return ""

        form = ""
        for g in games:
            is_home = (g["home_team_id"] == team_id)
            team_score = g["home_score"] if is_home else g["away_score"]
            opp_score  = g["away_score"] if is_home else g["home_score"]
            form += "W" if team_score > opp_score else "L"

        return form

    except Exception as e:
        print(f"Error calculating form for team_id {team_id}: {e}")
        traceback.print_exc()
        return ""


if __name__ == "__main__":
    print("Starting MLB Historical Team Stats Fetch...")
    total = 0
    seasons_done = 0

    for season in TARGET_SEASONS:
        print(f"\n--- Season {season} ---")
        teams = get_teams_for_season(MLB_LEAGUE_ID, season, HEADERS)
        time.sleep(REQUEST_DELAY_SECONDS)
        if not teams:
            continue
        seasons_done += 1

        for tm in teams:
            tid = tm.get('id')
            tname = tm.get('name')
            if not tid:
                continue

            print(f"Processing {tname} ({tid})")
            stats = get_team_stats(tid, MLB_LEAGUE_ID, season, HEADERS)
            time.sleep(REQUEST_DELAY_SECONDS)
            if not stats:
                continue

            record = transform_stats_data(stats)
            if not record:
                continue

            # ←—— NEW: exactly like NBA script, compute current_form here
            cf = calculate_team_form(tid, tname)
            print(f"Calculated MLB form for {tname}: {cf}")
            record['current_form'] = cf

            upsert_team_stats(supabase, record)
            total += 1

    print(f"\nDone. Seasons processed: {seasons_done}, Team-seasons upserted: {total}")
