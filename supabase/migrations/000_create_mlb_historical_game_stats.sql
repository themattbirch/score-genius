-- supabase/migrations/000_create_mlb_historical_game_stats.sql
-- Base table for MLB historical game stats used by rolling views

CREATE TABLE IF NOT EXISTS mlb_historical_game_stats (
  id                     SERIAL PRIMARY KEY,
  game_id                INTEGER    NOT NULL,
  game_date_time_utc     TIMESTAMPTZ,
  status_short           TEXT,
  status_long            TEXT,
  season                 INTEGER,
  home_team_id           INTEGER    NOT NULL,
  home_team_name         TEXT,
  away_team_id           INTEGER    NOT NULL,
  away_team_name         TEXT,
  home_score             INTEGER,
  away_score             INTEGER,
  home_hits              INTEGER,
  away_hits              INTEGER,
  home_errors            INTEGER,
  away_errors            INTEGER,
  -- add any other columns your real table has, for CI completeness:
  created_at             TIMESTAMPTZ DEFAULT NOW(),
  updated_at             TIMESTAMPTZ DEFAULT NOW()
);
