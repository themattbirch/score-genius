-- 008_create_nfl_historical_game_stats.sql
BEGIN;

CREATE TABLE IF NOT EXISTS public.nfl_historical_game_stats (
  id               SERIAL PRIMARY KEY,
  game_id          INTEGER     NOT NULL,
  season           INTEGER,
  week             INTEGER,
  stage            TEXT,
  game_date        DATE,
  game_time        TEXT,           -- or TIME if you prefer
  game_timestamp   TIMESTAMPTZ,    -- if you have full UTC timestamps
  home_team_id     INTEGER NOT NULL,
  away_team_id     INTEGER NOT NULL,
  home_team_name   TEXT,
  away_team_name   TEXT,
  home_score       INTEGER,
  away_score       INTEGER,
  home_q1          INTEGER,
  home_q2          INTEGER,
  home_q3          INTEGER,
  home_q4          INTEGER,
  home_ot          INTEGER,
  away_q1          INTEGER,
  away_q2          INTEGER,
  away_q3          INTEGER,
  away_q4          INTEGER,
  away_ot          INTEGER,
  venue_name       TEXT,
  venue_city       TEXT,
  updated_at       TIMESTAMPTZ DEFAULT now()
);

-- Speed up lookups by game_id
CREATE INDEX IF NOT EXISTS idx_nfl_historical_game_stats_game_id
  ON public.nfl_historical_game_stats(game_id);

COMMIT;
