# backend/data_pipeline/mlb_game_stats_historical.py
"""
Fetch and upsert historical MLB game stats from API-Sports to Supabase.
"""

import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

import requests
from dateutil import parser as dateutil_parser

from config import API_SPORTS_KEY
from caching.supabase_client import supabase

# --- Constants ---
API_BASE_URL: str = "https://v1.baseball.api-sports.io"
HEADERS: Dict[str, str] = {"x-apisports-key": API_SPORTS_KEY}
MLB_LEAGUE_ID: int = 1
TARGET_SEASONS: List[int] = [2025]
SUPABASE_TABLE_NAME: str = "mlb_historical_game_stats"
REQUEST_DELAY_SECONDS: int = 2
BATCH_SIZE: int = 50
SEASON_DATE_RANGES: Dict[int, tuple[str, Optional[str]]] = {
    2025: ("2025-04-26", None),
}


def make_api_request(
    url: str, headers: Dict[str, str], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Make an API request and return the JSON response."""
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        rem = resp.headers.get("x-ratelimit-requests-remaining")
        lim = resp.headers.get("x-ratelimit-requests-limit")
        print(f"Fetched {params.get('date', '')}: rate limit {rem}/{lim}")
        return resp.json()
    except requests.RequestException as e:
        print(f"API error for {url} {params}: {e}")
        if e.response:
            print(f"→ Status {e.response.status_code}, Body: {e.response.text[:200]}")
    return None


def get_games_for_date(
    league_id: int, season: int, date_str: str, headers: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Fetch all games for a given league, season, and date."""
    print(f"Getting games for {date_str} (season {season})")
    url = f"{API_BASE_URL}/games"
    params = {"league": league_id, "season": season, "date": date_str}
    data = make_api_request(url, headers, params)
    if not data or data.get("results", 0) == 0:
        print(f"No games on {date_str}")
        return []
    return data.get("response", [])


def transform_game_data(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transform a raw game object into the Supabase row format."""
    try:
        gid = game_data.get("id")
        if not gid:
            print("Warn: missing game id; skipping")
            return None

        dt_utc = None
        if date_str := game_data.get("date"):
            try:
                dt_utc = dateutil_parser.isoparse(date_str)
            except ValueError:
                print(f"Warn: date parse fail '{date_str}' for game {gid}")

        league = game_data.get("league", {})
        status = game_data.get("status", {})
        teams = game_data.get("teams", {})
        scores = game_data.get("scores", {})

        def to_int(x):
            try:
                return int(x) if x is not None else None
            except (ValueError, TypeError):
                return None

        home = teams.get("home", {})
        away = teams.get("away", {})
        home_score = scores.get("home", {})
        away_score = scores.get("away", {})

        def extract_inns(side_score):
            inns = side_score.get("innings", {}) or {}
            return {f"{prefix}_{inn}": to_int(inns.get(inn))
                    for prefix in ("h", "a")
                    for inn in tuple("123456789") + ("extra",)
                    if prefix == side_score is home_score}

        transformed: Dict[str, Any] = {
            "game_id": gid,
            "game_date_time_utc": dt_utc.isoformat() if dt_utc else None,
            "season": to_int(league.get("season")),
            "league_id": to_int(league.get("id")),
            "status_long": status.get("long"),
            "status_short": status.get("short"),
            "home_team_id": to_int(home.get("id")),
            "home_team_name": home.get("name"),
            "away_team_id": to_int(away.get("id")),
            "away_team_name": away.get("name"),
            "home_score": to_int(home_score.get("total")),
            "away_score": to_int(away_score.get("total")),
            "home_hits": to_int(home_score.get("hits")),
            "away_hits": to_int(away_score.get("hits")),
            "home_errors": to_int(home_score.get("errors")),
            "away_errors": to_int(away_score.get("errors")),
            # innings 1–9 and extra
            **{f"h_inn_{i}": to_int(home_score.get("innings", {}).get(str(i))) for i in range(1, 10)},
            "h_inn_extra": to_int(home_score.get("innings", {}).get("extra")),
            **{f"a_inn_{i}": to_int(away_score.get("innings", {}).get(str(i))) for i in range(1, 10)},
            "a_inn_extra": to_int(away_score.get("innings", {}).get("extra")),
            "raw_api_response": json.dumps(game_data),
        }
        return transformed
    except Exception as e:
        print(f"Error transforming game {game_data.get('id')}: {e}")
        return None


def upsert_game_stats_batch(
    supabase_client, game_stats_list: List[Dict[str, Any]]
) -> None:
    """Upsert a batch of game stats into Supabase."""
    if not game_stats_list:
        return
    print(f"Upserting batch of {len(game_stats_list)} records")
    try:
        supabase_client.table(SUPABASE_TABLE_NAME).upsert(
            game_stats_list, on_conflict="game_id"
        ).execute()
        print("Upsert successful")
    except Exception as e:
        print(f"Upsert error: {e}")


def generate_date_range(
    start: str, end: str
) -> Generator[date, None, None]:
    """Yield every date from start to end inclusive."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as ex:
        print(f"Date parse error: {ex}")
        return
    current = s
    while current <= e:
        yield current
        current += timedelta(days=1)


def main() -> None:
    if not API_SPORTS_KEY or supabase is None:
        print("Missing API key or Supabase client; exiting")
        sys.exit(1)

    yesterday = (datetime.utcnow().date() - timedelta(days=1))
    buffer: List[Dict[str, Any]] = []
    total = 0

    for season in TARGET_SEASONS:
        rng = SEASON_DATE_RANGES.get(season)
        if not rng:
            print(f"No date range for season {season}; skipping")
            continue

        start_date, end_cfg = rng
        if season == datetime.utcnow().year:
            end_date = yesterday.strftime("%Y-%m-%d")
        else:
            if not end_cfg:
                print(f"No end date configured for past season {season}; skipping")
                continue
            end_date = end_cfg

        print(f"Processing season {season}: {start_date} → {end_date}")
        for d in generate_date_range(start_date, end_date):
            ds = d.strftime("%Y-%m-%d")
            games = get_games_for_date(MLB_LEAGUE_ID, season, ds, HEADERS)
            for g in games:
                rec = transform_game_data(g)
                if rec:
                    buffer.append(rec)
                    total += 1

            if len(buffer) >= BATCH_SIZE:
                upsert_game_stats_batch(supabase, buffer)
                buffer.clear()

            time.sleep(REQUEST_DELAY_SECONDS)

    if buffer:
        upsert_game_stats_batch(supabase, buffer)

    print(f"Finished. Total games processed: {total}")


if __name__ == "__main__":
    main()
