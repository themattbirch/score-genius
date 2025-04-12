import requests
import json
from config import API_SPORTS_KEY

API_KEY = API_SPORTS_KEY
BASE_URL = "https://v1.basketball.api-sports.io"

# Function to make API calls and print results
def fetch_and_print_data(endpoint, params, description):
    print(f"\n\n{'=' * 80}")
    print(f"TESTING ENDPOINT: {endpoint} - {description}")
    print(f"{'=' * 80}")
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "v1.basketball.api-sports.io"
    }
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Print the entire response structure
        print(json.dumps(data, indent=2))
        
        # Check for specific statistics we're interested in
        if "response" in data:
            if endpoint == "statistics":
                if "points" in data["response"]:
                    points_data = data["response"]["points"]
                    print("\nKEY STATS FOUND:")
                    for key in points_data:
                        print(f"- {key}")
            elif endpoint == "games/statistics/teams":
                if isinstance(data["response"], list) and len(data["response"]) > 0:
                    for team_stats in data["response"]:
                        if "statistics" in team_stats:
                            print("\nTEAM STATISTICS AVAILABLE:")
                            for category, stats in team_stats["statistics"].items():
                                print(f"\n{category.upper()}:")
                                for stat_key, stat_value in stats.items():
                                    print(f"- {stat_key}: {stat_value}")
    except Exception as e:
        print(f"Error: {e}")

# Test 1: Team Statistics for a specific team in a season
fetch_and_print_data(
    "statistics", 
    {
        "league": "12",   # NBA league ID
        "season": "2023-2024",  # Current season
        "team": "139"     # Example team ID (Denver Nuggets)
    },
    "Team Season Statistics"
)

# Test 2: Team Statistics for a specific game
fetch_and_print_data(
    "games/statistics/teams", 
    {
        "id": "414694"    # Using the same game ID from user's example
    },
    "Team Game Statistics"
)

# Test 3: Check for field goals data specifically
fetch_and_print_data(
    "games", 
    {
        "id": "414694"    # Check a complete game record
    },
    "Complete Game Data"
)

print("\n\n\nCHECKING FOR SPECIFIC DATA FIELDS OF INTEREST:")
print("=" * 60)
fields_to_check = [
    "field_goals_made", "field_goals_attempted", "field_goal_percentage",
    "free_throws_made", "free_throws_attempted", "free_throw_percentage",
    "pace", "possessions", "injuries", "player_status"
]

print("Fields we're looking for:")
for field in fields_to_check:
    print(f"- {field}")