# find_decimal_stat.py
import json
from pathlib import Path

# point this at your raw_json folder
json_path = Path("backend/data_pipeline/raw_json/raw_nfl_player_stats_3_2024.json")

data = json.loads(json_path.read_text())
for rec in data.get("response", []):
    for team in rec.get("teams", []):
        for grp in team.get("groups", []):
            for stat in grp.get("statistics", []):
                if stat.get("value") == "4.7":
                    print(f"Group: {grp['name']}  →  Stat: {stat['name']}")
