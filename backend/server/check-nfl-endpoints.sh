#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:10000"

# parallel arrays of names and paths
names=(
  "Health"
  "Swagger UI"
  "Team Full (2024, teams 1&2)"
  "Team Reg‑Only (2024)"
  "Dashboard Cards (2024)"
  "SOS (2024)"
  "SRS lite (2024)"
  "Schedule (today)"
  "Snapshots"
  "Cron health"
  "Validate health"
)
paths=(
  "/health"
  "/api-docs"
  "/api/v1/nfl/teams/2024/full?teamId=1,2"
  "/api/v1/nfl/teams/2024/regonly"
  "/api/v1/nfl/teams/2024/dashboard"
  "/api/v1/nfl/teams/2024/sos"
  "/api/v1/nfl/teams/2024/srs"
  "/api/v1/nfl/schedule?date=$(date +%Y-%m-%d)"
  "/api/v1/nfl/snapshots?gameIds=2024090701,2024090702"
  "/api/v1/nfl/health/cron"
  "/api/v1/nfl/health/validate"
)

echo "→ Testing NFL API endpoints against $BASE"
printf "┌─────────┬─────────────────────────┬──────────┐\n"
printf "│ %-7s │ %-23s │ %-8s │\n" "#" "Endpoint" "Result"
printf "├─────────┼─────────────────────────┼──────────┤\n"

for i in "${!names[@]}"; do
  idx=$(( i + 1 ))
  name="${names[i]}"
  url="$BASE${paths[i]}"
  http_code=$(curl -s -L -o /dev/null -w "%{http_code}" "$url" || echo "000")
  if [[ $http_code =~ ^2 ]]; then
    status="OK  ($http_code)"
  else
    status="FAIL($http_code)"
  fi
  printf "│ %-7s │ %-23s │ %-8s │\n" "$idx" "$name" "$status"
done

printf "└─────────┴─────────────────────────┴──────────┘\n"
