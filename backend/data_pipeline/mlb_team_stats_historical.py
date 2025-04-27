# backend/data_pipeline/mlb_team_stats_historical.py

"""
Fetch and upsert historical MLB team stats from API-Sports to Supabase.
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from config import API_SPORTS_KEY
from caching.supabase_client import supabase

# --- Constants ---
API_BASE_URL = "https://v1.baseball.api-sports.io"
HEADERS = {"x-apisports-key": API_SPORTS_KEY}
MLB_LEAGUE_ID = 1
TARGET_SEASONS = [2025]
SUPABASE_TABLE_NAME = "mlb_historical_team_stats"
REQUEST_DELAY_SECONDS = 2


def make_api_request(
    url: str, headers: Dict[str, str], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Perform GET request and return parsed JSON, or None on failure."""
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        rem = resp.headers.get("x-ratelimit-requests-remaining")
        lim = resp.headers.get("x-ratelimit-requests-limit")
        print(f"API hit {params}: rate limit {rem}/{lim}")
        return resp.json()
    except requests.RequestException as exc:
        print(f"API request error for {url} {params}: {exc}")
    return None


def get_teams_for_season(
    league_id: int, season: int, headers: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Return list of teams for a given league & season."""
    url = f"{API_BASE_URL}/teams"
    params = {"league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    if data and isinstance(data.get("response"), list):
        return data["response"]
    print(f"No teams found for season {season}")
    return []


def get_team_stats(
    team_id: int, league_id: int, season: int, headers: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Fetch per-team aggregated stats from the API."""
    url = f"{API_BASE_URL}/teams/statistics"
    params = {"team": team_id, "league": league_id, "season": season}
    data = make_api_request(url, headers, params)
    return data.get("response") if data else None


def safe_float(value: Any) -> Optional[float]:
    """Convert percentage or numeric string to float, or None."""
    if value is None:
        return None
    try:
        text = str(value).rstrip("%")
        return float(text)
    except ValueError:
        return None


def transform_stats_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map raw API response to the Supabase table schema.
    Does not include `current_form`; that’s added later.
    """
    try:
        games = api_data.get("games", {})
        points = api_data.get("points", {})
        team = api_data.get("team", {})
        league = api_data.get("league", {})

        def nested(d: Dict[str, Any], *path, default=None):
            for key in path:
                if not isinstance(d, dict):
                    return default
                d = d.get(key, default)
            return d

        record = {
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "season": league.get("season"),
            "league_id": league.get("id"),
            "league_name": league.get("name"),

            # Games played
            "games_played_home": nested(games, "played", "home"),
            "games_played_away": nested(games, "played", "away"),
            "games_played_all": nested(games, "played", "all"),

            # Wins
            "wins_home_total": nested(games, "wins", "home", "total"),
            "wins_home_pct": safe_float(nested(games, "wins", "home", "percentage")),
            "wins_away_total": nested(games, "wins", "away", "total"),
            "wins_away_pct": safe_float(nested(games, "wins", "away", "percentage")),
            "wins_all_total": nested(games, "wins", "all", "total"),
            "wins_all_pct": safe_float(nested(games, "wins", "all", "percentage")),

            # Losses
            "losses_home_total": nested(games, "loses", "home", "total"),
            "losses_home_pct": safe_float(nested(games, "loses", "home", "percentage")),
            "losses_away_total": nested(games, "loses", "away", "total"),
            "losses_away_pct": safe_float(nested(games, "loses", "away", "percentage")),
            "losses_all_total": nested(games, "loses", "all", "total"),
            "losses_all_pct": safe_float(nested(games, "loses", "all", "percentage")),

            # Runs for
            "runs_for_total_home": nested(points, "for", "total", "home"),
            "runs_for_avg_home": safe_float(nested(points, "for", "average", "home")),
            "runs_for_total_away": nested(points, "for", "total", "away"),
            "runs_for_avg_away": safe_float(nested(points, "for", "average", "away")),
            "runs_for_total_all": nested(points, "for", "total", "all"),
            "runs_for_avg_all": safe_float(nested(points, "for", "average", "all")),

            # Runs against
            "runs_against_total_home": nested(points, "against", "total", "home"),
            "runs_against_avg_home": safe_float(nested(points, "against", "average", "home")),
            "runs_against_total_away": nested(points, "against", "total", "away"),
            "runs_against_avg_away": safe_float(nested(points, "against", "average", "away")),
            "runs_against_total_all": nested(points, "against", "total", "all"),
            "runs_against_avg_all": safe_float(nested(points, "against", "average", "all")),

            "raw_api_response": json.dumps({"games": games, "points": points}),
        }

        # Sanity check required keys
        if not (record["team_id"] and record["season"] and record["league_id"]):
            return None

        return record

    except Exception as exc:
        print(f"Error in transform_stats_data: {exc}")
        return None


def calculate_team_form(team_id: int, season: int, lookback: int = 5) -> str:
    """
    Build a W/L string from the last `lookback` completed games for this team.
    """
    try:
        resp = (
            supabase.table("mlb_historical_game_stats")
            .select(
                "home_team_id,away_team_id,home_score,away_score,game_date_time_utc"
            )
            .eq("season", season)
            .eq("status_short", "FT")
            .or_(f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}")
            .order("game_date_time_utc", desc=True)
            .limit(lookback)
            .execute()
        )
        games = resp.data or []
        form = ""
        for g in games:
            hs = g.get("home_score")
            as_ = g.get("away_score")
            if hs is None or as_ is None:
                continue
            is_home = g["home_team_id"] == team_id
            team_score = hs if is_home else as_
            opp_score = as_ if is_home else hs
            form += "W" if team_score > opp_score else "L"
        return form

    except Exception as exc:
        print(f"Error calculating form for team {team_id}: {exc}")
        return ""


def upsert_team_stats(stats: Dict[str, Any]) -> None:
    """Upsert a single team‐season record into Supabase."""
    try:
        supabase.table(SUPABASE_TABLE_NAME) \
            .upsert(stats, on_conflict="team_id,season,league_id") \
            .execute()
        print(f"Upserted team {stats['team_id']} season {stats['season']}")
    except Exception as exc:
        print(f"Upsert exception: {exc}")


def main() -> None:
    if not API_SPORTS_KEY or supabase is None:
        print("Missing API key or Supabase client; exiting")
        sys.exit(1)

    total_upserted = 0
    seasons_processed = 0

    for season in TARGET_SEASONS:
        print(f"\n=== Season {season} ===")
        teams = get_teams_for_season(MLB_LEAGUE_ID, season, HEADERS)
        time.sleep(REQUEST_DELAY_SECONDS)

        if not teams:
            continue

        seasons_processed += 1

        for team in teams:
            tid = team.get("id")
            name = team.get("name", "Unknown")
            if not tid or name in ("American League", "National League"):
                continue

            print(f"Fetching stats for {name} (ID {tid})")
            stats_data = get_team_stats(tid, MLB_LEAGUE_ID, season, HEADERS)
            time.sleep(REQUEST_DELAY_SECONDS)
            if not stats_data:
                continue

            record = transform_stats_data(stats_data)
            if not record:
                continue

            form = calculate_team_form(tid, season)
            record["current_form"] = form
            upsert_team_stats(record)
            total_upserted += 1

    print(
        f"\nDone. Seasons processed: {seasons_processed}, "
        f"Team records upserted: {total_upserted}"
    )


if __name__ == "__main__":
    main()
