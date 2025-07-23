-- 006_create_nfl_historical_team_stats.sql

BEGIN;

CREATE TABLE IF NOT EXISTS public.nfl_historical_team_stats (
  id                    SERIAL PRIMARY KEY,
  team_id               INTEGER       NOT NULL,
  season                INTEGER       NOT NULL,
  league_id             INTEGER,
  team_name             TEXT,
  team_logo             TEXT,
  position              INTEGER,
  won                   INTEGER       DEFAULT 0,
  lost                  INTEGER       DEFAULT 0,
  ties                  INTEGER       DEFAULT 0,
  points_for            INTEGER       DEFAULT 0,
  points_against        INTEGER       DEFAULT 0,
  points_difference     INTEGER       DEFAULT 0,
  srs_lite              DOUBLE PRECISION,
  record_home           TEXT,
  record_road           TEXT,
  record_division       TEXT,
  record_conference     TEXT,
  conference            TEXT,
  division              TEXT,
  streak                TEXT,
  updated_at            TIMESTAMPTZ   DEFAULT now()
);

-- Speed up lookups by team & season
CREATE INDEX IF NOT EXISTS idx_nfl_historical_team_stats_team_season
  ON public.nfl_historical_team_stats(team_id, season);

COMMIT;
