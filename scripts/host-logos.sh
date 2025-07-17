#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Host league team logos in Supabase Storage and update DB logo URLs.
#
# Usage:
#   ./scripts/host-logos.sh [sport] [bucket]
#
#   sport  : nfl | mlb | nba  (default: nfl)
#   bucket : Supabase Storage bucket name (default: team-logos)
#
# Env (auto-filled with your project values; override by exporting before run):
#   SUPABASE_URL
#   SUPABASE_ANON_KEY
#   SUPABASE_SERVICE_KEY
#   SUPABASE_DB_URL
#   PROJECT_REF
#
# Uploads via REST with service key (Supabase CLI not required).
###############################################################################

SPORT="${1:-nfl}"
BUCKET="${2:-team-logos}"

# --- Hard-coded project values (override by exporting before run) -------------
SUPABASE_URL="${SUPABASE_URL:-https://qaytaxyflvafblirxgdr.supabase.co}"
SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFheXRheHlmbHZhZmJsaXJ4Z2RyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzkzNDE3NzksImV4cCI6MjA1NDkxNzc3OX0.OJRZRo5RA4hSBZ_QKyI6fDy4JTmzNQjaNmwbT0GiIJ8}"
SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_KEY:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFheXRheHlmbHZhZmJsaXJ4Z2RyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczOTM0MTc3OSwiZXhwIjoyMDU0OTE3Nzc5fQ.bUkJ0AwTOTR8ctGdYovjY2OLG4gXGaD_F0MUD9jI4Bs}"
SUPABASE_DB_URL="${SUPABASE_DB_URL:-postgresql://postgres:MustW1nBetzz@db.qaytaxyflvafblirxgdr.supabase.co:5432/postgres}"
PROJECT_REF="${PROJECT_REF:-qaytaxyflvafblirxgdr}"

# --- Sanity checks -------------------------------------------------------------
if ! command -v psql >/dev/null 2>&1; then
  echo "psql not found in PATH."
  exit 1
fi
if [ -z "$SUPABASE_SERVICE_KEY" ]; then
  echo "SUPABASE_SERVICE_KEY is required."
  exit 1
fi
if [ -z "$SUPABASE_DB_URL" ]; then
  echo "SUPABASE_DB_URL is required."
  exit 1
fi

# --- Resolve script paths ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGO_DIR="${ROOT_DIR}/logos/${SPORT}"
mkdir -p "$LOGO_DIR"

# --- Map sport -> table + media path ------------------------------------------
case "$SPORT" in
  nfl)
    TEAM_TABLE="nfl_teams_dim"
    HIST_TABLE="nfl_historical_team_stats"
    MEDIA_PATH="american-football/teams"
    ;;
  mlb)
    TEAM_TABLE="mlb_teams_dim"
    HIST_TABLE="mlb_historical_team_stats"
    MEDIA_PATH="baseball/teams"
    ;;
  nba)
    TEAM_TABLE="nba_teams_dim"
    HIST_TABLE="nba_historical_team_stats"
    MEDIA_PATH="basketball/teams"
    ;;
  *)
    echo "Unknown sport '$SPORT' (expected nfl|mlb|nba)."
    exit 1
    ;;
esac

# --- Fetch team IDs (portable; no readarray) -----------------------------------
echo "Fetching team IDs from ${TEAM_TABLE}..."
TEAM_IDS_RAW="$(psql "$SUPABASE_DB_URL" -t -A -c "SELECT team_id FROM ${TEAM_TABLE} ORDER BY team_id;" || true)"

# Strip CRs, compress whitespace, drop empties
TEAM_IDS_TRIM="$(printf '%s\n' "$TEAM_IDS_RAW" | tr -d '\r' | awk 'NF{print $1}')"

if [ -z "$TEAM_IDS_TRIM" ]; then
  echo "No team_ids returned from ${TEAM_TABLE}; aborting."
  exit 1
fi

TEAM_COUNT="$(printf '%s\n' "$TEAM_IDS_TRIM" | wc -l | tr -d '[:space:]')"
echo "Found ${TEAM_COUNT} team IDs:"
printf '  %s\n' $TEAM_IDS_TRIM

# --- Download logos ------------------------------------------------------------
echo "Downloading logos from API-Sports..."
for id in $TEAM_IDS_TRIM; do
  src="https://media.api-sports.io/${MEDIA_PATH}/${id}.png"
  dst="${LOGO_DIR}/${id}.png"
  curl -sSLo "$dst" "$src" || echo "Download failed for team_id=$id"
done

# --- Upload logos to Supabase Storage (REST) -----------------------------------
PUBLIC_BASE="${SUPABASE_URL}/storage/v1/object/public/${BUCKET}"
UPLOAD_BASE="${SUPABASE_URL}/storage/v1/object/${BUCKET}"
echo "Uploading to Supabase Storage: ${UPLOAD_BASE}/${SPORT}/<id>.png ..."
for id in $TEAM_IDS_TRIM; do
  file="${LOGO_DIR}/${id}.png"
  key="${SPORT}/${id}.png"
  if [ ! -f "$file" ]; then
    echo "Missing file for team_id=$id; skipping."
    continue
  fi
  curl -sS -X POST \
    -H "apikey: ${SUPABASE_SERVICE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_KEY}" \
    -H "Content-Type: image/png" \
    --data-binary @"${file}" \
    "${UPLOAD_BASE}/${key}?upsert=true" >/dev/null
  echo "  • Uploaded ${key}"
done

# --- Update DB logo URLs -------------------------------------------------------
echo "Updating ${TEAM_TABLE}.team_logo to hosted URLs..."
psql "$SUPABASE_DB_URL" <<SQL
UPDATE ${TEAM_TABLE}
SET team_logo = '${PUBLIC_BASE}/${SPORT}/' || team_id::text || '.png';

-- Conditionally update historical stats table if it exists
DO \$\$
BEGIN
  IF to_regclass('public.${HIST_TABLE}') IS NOT NULL THEN
    EXECUTE format(
      'UPDATE %I ts
         SET team_logo = d.team_logo,
             updated_at = NOW()
       FROM %I d
      WHERE ts.team_id = d.team_id;',
      '${HIST_TABLE}', '${TEAM_TABLE}'
    );
  END IF;
END
\$\$;
SQL

echo "Done. Logos live at:"
echo "  ${PUBLIC_BASE}/${SPORT}/<team_id>.png"
