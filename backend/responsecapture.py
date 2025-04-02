# backend/responsecapture.py


import pandas as pd
from caching.supabase_client import supabase  # Adjust the import if your client is located elsewhere

# Query the nba_historical_game_stats table
response = supabase.table("nba_historical_game_stats").select("*").execute()

# The response might store the results in a 'data' attribute:
data = response.data

# (Alternatively, if your client returns a dict, you might use:
# data = response.get("data")
# )

print("Supabase response data:")
print(data)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)
print("DataFrame preview:")
print(df.head())
