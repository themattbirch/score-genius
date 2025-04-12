# backend/data_pipeline/process_odds_data.py

import json
from supabase import create_client, Client
from typing import Any, Dict

# Adjust to your actual config references
from config import SUPABASE_URL, SUPABASE_ANON_KEY

def parse_moneyline(moneyline_data: Dict[str, Any]) -> str:
    if not moneyline_data:
        return ""
    parts = []
    for team, price in moneyline_data.items():
        price_str = f"+{price}" if price > 0 else str(price)
        parts.append(f"{team} {price_str}")
    return " / ".join(parts)

def parse_spread(spread_data: Dict[str, Any]) -> str:
    if not spread_data:
        return ""
    parts = []
    for team, info in spread_data.items():
        if not isinstance(info, dict):
            continue
        point = info.get("point", 0)
        point_str = f"+{point}" if point > 0 else str(point)
        parts.append(f"{team} {point_str}")  # Removed the odds in parentheses
    return " / ".join(parts)

def parse_total(total_data: Dict[str, Any]) -> str:
    if not total_data:
        return ""
    parts = []
    for side, info in total_data.items():
        if not isinstance(info, dict):
            continue
        point = info.get("point", 0)
        parts.append(f"{side} {point}")  # Removed the odds in parentheses
    return " / ".join(parts)

def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    # Grab all rows from nba_game_schedule
    response = supabase.table("nba_game_schedule").select("*").execute()
    rows = response.data
    if not rows:
        print("No rows found in nba_game_schedule.")
        return
    
    for row in rows:
        game_id = row.get("game_id")
        if not game_id:
            continue
        
        moneyline_json = row.get("moneyline") or {}
        spread_json = row.get("spread") or {}
        total_json = row.get("total") or {}
        
        # Ensure they're dicts
        if isinstance(moneyline_json, str):
            moneyline_json = json.loads(moneyline_json)
        if isinstance(spread_json, str):
            spread_json = json.loads(spread_json)
        if isinstance(total_json, str):
            total_json = json.loads(total_json)
        
        moneyline_str = parse_moneyline(moneyline_json)
        spread_str = parse_spread(spread_json)
        total_str = parse_total(total_json)
        
        # Update data: set the original columns to None, 
        # and put the cleaned text into the *_clean columns.
        update_data = {
            "moneyline_clean": moneyline_str,
            "spread_clean": spread_str,
            "total_clean": total_str,
            "moneyline": None,  # clear the original column
            "spread": None,     # clear the original column
            "total": None       # clear the original column
        }
        
        supabase.table("nba_game_schedule") \
            .update(update_data) \
            .eq("game_id", game_id) \
            .execute()
        
        print(f"Updated and cleared odds for game_id {game_id}.")
    
    print("Finished processing and clearing odds data.")

if __name__ == "__main__":
    main()
