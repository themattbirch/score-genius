import requests
import time
from datetime import datetime, timedelta
import sys, os

# Add the backend root to the Python path so we can import from caching and config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config import API_SPORTS_KEY
from caching.supabase_client import supabase

API_KEY = API_SPORTS_KEY
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v1.basketball.api-sports.io"
}

def get_games_by_date(league, season, date):
    url = f"{BASE_URL}/games"
    params = {"league": league, "season": season, "date": date}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])

def get_team_box_stats(game_id):
    url = f"{BASE_URL}/games/statistics/teams"
    params = {"id": game_id}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])

def parse_season_from_date(date_obj):
    if date_obj.month >= 10:
        return f"{date_obj.year}-{date_obj.year+1}"
    else:
        return f"{date_obj.year-1}-{date_obj.year}"

def process_day(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    season = parse_season_from_date(date_obj)
    print(f"\n=== {date_str} (Season {season}) ===")

    games_list = get_games_by_date("12", season, date_str)
    if not games_list:
        print("No games found.")
        return

    for g in games_list:
        if g.get("status", {}).get("short") != "FT":
            continue  # Process only final games

        game_id = g["id"]
        home_team = g["teams"]["home"]["name"]
        away_team = g["teams"]["away"]["name"]
        home_score = g["scores"]["home"]["total"]
        away_score = g["scores"]["away"]["total"]

        # Quarter scores
        scores = g.get("scores", {})
        print("Quarter scores =>", scores)
        home_q1 = scores.get("home", {}).get("quarter_1", 0) or 0
        home_q2 = scores.get("home", {}).get("quarter_2", 0) or 0
        home_q3 = scores.get("home", {}).get("quarter_3", 0) or 0
        home_q4 = scores.get("home", {}).get("quarter_4", 0) or 0
        home_ot = scores.get("home", {}).get("over_time", 0) or 0

        away_q1 = scores.get("away", {}).get("quarter_1", 0) or 0
        away_q2 = scores.get("away", {}).get("quarter_2", 0) or 0
        away_q3 = scores.get("away", {}).get("quarter_3", 0) or 0
        away_q4 = scores.get("away", {}).get("quarter_4", 0) or 0
        away_ot = scores.get("away", {}).get("over_time", 0) or 0

        # Team box stats
        stats_teams = get_team_box_stats(game_id)
        print("Home/Away Scores: /games/statistics/teams =>", stats_teams)
        home_box = {}
        away_box = {}

        # Retrieve team IDs from the game object for matching
        home_team_id = g["teams"]["home"]["id"]
        away_team_id = g["teams"]["away"]["id"]

        for t in stats_teams:
            team_stat_id = t.get("team", {}).get("id")
            # Build advanced stats using top-level keys
            adv = {
                "assists": t.get("assists", 0),
                "steals": t.get("steals", 0),
                "blocks": t.get("blocks", 0),
                "turnovers": t.get("turnovers", 0),
                "personal_fouls": t.get("personal_fouls", 0),
                "off_reb": t.get("rebounds", {}).get("offence", 0),
                "def_reb": t.get("rebounds", {}).get("defense", 0),
                "total_reb": t.get("rebounds", {}).get("total", 0),
                "3pm": t.get("threepoint_goals", {}).get("total", 0),
                "3pa": t.get("threepoint_goals", {}).get("attempts", 0)
            }
            if team_stat_id == home_team_id:
                home_box = adv
            elif team_stat_id == away_team_id:
                away_box = adv

        record = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_q1": home_q1,
            "home_q2": home_q2,
            "home_q3": home_q3,
            "home_q4": home_q4,
            "home_ot": home_ot,
            "away_q1": away_q1,
            "away_q2": away_q2,
            "away_q3": away_q3,
            "away_q4": away_q4,
            "away_ot": away_ot,
            "game_date": date_str,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "home_assists": home_box.get("assists", 0),
            "home_steals": home_box.get("steals", 0),
            "home_blocks": home_box.get("blocks", 0),
            "home_turnovers": home_box.get("turnovers", 0),
            "home_fouls": home_box.get("personal_fouls", 0),
            "away_assists": away_box.get("assists", 0),
            "away_steals": away_box.get("steals", 0),
            "away_blocks": away_box.get("blocks", 0),
            "away_turnovers": away_box.get("turnovers", 0),
            "away_fouls": away_box.get("personal_fouls", 0),
            "home_off_reb": home_box.get("off_reb", 0),
            "home_def_reb": home_box.get("def_reb", 0),
            "home_total_reb": home_box.get("total_reb", 0),
            "away_off_reb": away_box.get("off_reb", 0),
            "away_def_reb": away_box.get("def_reb", 0),
            "away_total_reb": away_box.get("total_reb", 0),
            "home_3pm": home_box.get("3pm", 0),
            "home_3pa": home_box.get("3pa", 0),
            "away_3pm": away_box.get("3pm", 0),
            "away_3pa": away_box.get("3pa", 0)
        }

        print("Record =>", record)
        try:
            res = (
                supabase
                .table("nba_historical_game_stats")
                .upsert(record, on_conflict=["game_id"])
                .execute()
            )
            print("Upsert result =>", res)
        except Exception as e:
            print("Error during upsert:", e)
            if hasattr(e, 'response'):
                try:
                    print("Raw error response:", e.response.text)
                except Exception:
                    pass

def main():
    start_date = datetime(2024, 5, 3)
    end_date = datetime(2025, 2, 13)
    current = start_date
    while current <= end_date:
        process_day(current)
        time.sleep(60)  
        current += timedelta(days=1)

if __name__ == "__main__":
    main()
