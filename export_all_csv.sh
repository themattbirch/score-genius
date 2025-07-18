#!/usr/bin/env bash
set -euo pipefail

# 1) Load backend/.env (exports all vars)
if [ -f backend/.env ]; then
  set -o allexport
  source backend/.env
  set +o allexport
else
  echo "❌  backend/.env not found—make sure you're at the repo root."
  exit 1
fi

# 2) Verify DATABASE_URL is set
: "${DATABASE_URL?❌  DATABASE_URL is not set in backend/.env}"

# 3) Prepare output directory
mkdir -p csv_exports

# 4) Fetch all public‐schema table names
TABLES=$(psql "$DATABASE_URL" -At \
  -c "SELECT table_name FROM information_schema.tables
      WHERE table_schema='public' AND table_type='BASE TABLE';")

# 5) Loop & export
for t in $TABLES; do
  echo "👉 Exporting $t → csv_exports/${t}.csv"
  psql "$DATABASE_URL" \
    -c "\copy public.\"$t\" TO 'csv_exports/${t}.csv' CSV HEADER"
done

echo "✅  Export complete. CSVs are in ./csv_exports/"
