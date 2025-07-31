#!/bin/zsh
# Fetch next 8 days of “Not Started” NFL games via API-Sports

API_SPORTS_KEY=d0c358b61e883d071bbc183c8fd72228
BASE_URL="https://v1.american-football.api-sports.io"
HOST="v1.american-football.api-sports.io"
LEAGUE=1
TZ="America/New_York"
DELAY=5  # seconds

for i in {0..7}; do
  # macOS: advance by i days
  DATE=$(date -v+"$i"d +%F)
  
  echo "→ ${DATE} (ET)"
  curl -s -G "${BASE_URL}/games" \
    -H "x-rapidapi-key: ${API_SPORTS_KEY}" \
    -H "x-rapidapi-host: ${HOST}" \
    -H "Accept: application/json" \
    --data-urlencode "league=${LEAGUE}" \
    --data-urlencode "date=${DATE}" \
    --data-urlencode "timezone=${TZ}" \
  | jq '.response[] | select(.game.status.short=="NS") | {
      game_id: .game.id,
      date: .game.date.date,
      home: .teams.home.name,
      away: .teams.away.name,
      scheduled: (
        if .game.date.timestamp then
          (.game.date.timestamp | tonumber | todateiso8601)
        else
          "\(.game.date.date)T\(.game.date.time):00-04:00"
        end
      ),
      venue: .game.venue.name
    }'

  sleep $DELAY
done
