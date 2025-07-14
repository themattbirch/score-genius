import json, glob
import pandas as pd

RAW_PATTERN = "backend/data_pipeline/raw_json/raw_nfl_player_stats_*_2024.json"

decimal_stats = set()
for fp in glob.glob(RAW_PATTERN):
    data = json.load(open(fp))
    for rec in data.get("response", []):
        for team in rec.get("teams", []):
            for grp in team.get("groups", []):
                for stat in grp.get("statistics", []):
                    v = stat.get("value","")
                    if isinstance(v, str) and "." in v:
                        decimal_stats.add((grp["name"], stat["name"]))

df = pd.DataFrame(sorted(decimal_stats), columns=["Group","Stat Name"])
print(df.to_markdown(index=False))
