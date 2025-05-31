#!/usr/bin/env bash

# Your RapidAPI key
export RAPIDAPI_KEY="d0c358b61e883d071bbc183c8fd72228"

API_HOST="v1.basketball.api-sports.io"
LEAGUE="12"
# Still use season=2024 for the 2024–2025 season
SEASON="2024"
TIMEZONE="America/New_York"

BASE_DATE="2025-05-30"

for offset in 0 1 2; do
  DATE=$(date -j -v+"${offset}"d -f "%Y-%m-%d" "$BASE_DATE" "+%Y-%m-%d")
  echo
  echo "=== Fetching for ET date $DATE ==="

  # 1) Query by explicit date
  curl -s -G "https://${API_HOST}/games" \
    -H "x-rapidapi-key: ${RAPIDAPI_KEY}" \
    -H "x-rapidapi-host: ${API_HOST}" \
    --data-urlencode "league=${LEAGUE}" \
    --data-urlencode "season=${SEASON}" \
    --data-urlencode "date=${DATE}" \
    --data-urlencode "timezone=${TIMEZONE}" \
    | tee "games_date_${DATE}.json" \
    | jq .

  # 2) Fallback: get next 10 upcoming games, then filter by date
  curl -s -G "https://${API_HOST}/games" \
    -H "x-rapidapi-key: ${RAPIDAPI_KEY}" \
    -H "x-rapidapi-host: ${API_HOST}" \
    --data-urlencode "league=${LEAGUE}" \
    --data-urlencode "season=${SEASON}" \
    --data-urlencode "next=10" \
    --data-urlencode "timezone=${TIMEZONE}" \
    | tee "games_next_${DATE}.json" \
    | jq '.response[] | select((.date|startswith("'"${DATE}"'")))' 
done
