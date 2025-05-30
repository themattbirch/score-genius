#!/usr/bin/env bash

export API_SPORTS_KEY="d0c358b61e883d071bbc183c8fd72228"
API_HOST="v1.basketball.api-sports.io"
LEAGUE="12"
SEASON="2024-2025"
TIMEZONE="America/New_York"

# Base date (you can change this to today with: date +"%Y-%m-%d")
BASE_DATE="2025-05-30"

for offset in 0 1 2; do
  # macOS date arithmetic:
  DATE=$(date -j -v+"${offset}"d -f "%Y-%m-%d" "$BASE_DATE" "+%Y-%m-%d")
  echo
  echo "=== Fetching for ET date $DATE ==="
  
  # 1) by explicit date
  curl -s -G "https://$API_HOST/games" \
    -H "x-rapidapi-key: $API_SPORTS_KEY" \
    -H "x-rapidapi-host: $API_HOST" \
    --data-urlencode "league=$LEAGUE" \
    --data-urlencode "season=$SEASON" \
    --data-urlencode "date=$DATE" \
    --data-urlencode "timezone=$TIMEZONE" \
    | tee "games_date_${DATE}.json" \
    | jq .

  # 2) fallback via 'next'
  curl -s -G "https://$API_HOST/games" \
    -H "x-rapidapi-key: $API_SPORTS_KEY" \
    -H "x-rapidapi-host: $API_HOST" \
    --data-urlencode "league=$LEAGUE" \
    --data-urlencode "next=10" \
    --data-urlencode "timezone=$TIMEZONE" \
    | tee "games_next_${DATE}.json" \
    | jq '.response[] | select((.date|startswith("'"$DATE"'")))' 
done
