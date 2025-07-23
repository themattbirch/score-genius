-- 009_create_nfl_historical_game_team_stats.sql
BEGIN;

CREATE TABLE IF NOT EXISTS public.nfl_historical_game_team_stats (
  id                             SERIAL PRIMARY KEY,
  game_id                        INTEGER     NOT NULL,
  team_id                        INTEGER     NOT NULL,
  season                         INTEGER,
  raw_api_response               JSONB,
  home_team_id                   INTEGER,
  away_team_id                   INTEGER,
  home_team_name                 TEXT,
  away_team_name                 TEXT,
  first_downs_total              INTEGER     DEFAULT 0,
  first_downs_passing            INTEGER     DEFAULT 0,
  first_downs_rushing            INTEGER     DEFAULT 0,
  first_downs_penalty            INTEGER     DEFAULT 0,
  plays_total                    INTEGER     DEFAULT 0,
  yards_total                    INTEGER     DEFAULT 0,
  yards_per_play                 DOUBLE PRECISION,
  total_drives                   INTEGER     DEFAULT 0,
  passing_total                  INTEGER     DEFAULT 0,
  passing_comp_att               TEXT,
  passing_yards_per_pass         DOUBLE PRECISION,
  passing_interceptions          INTEGER     DEFAULT 0,
  passing_sacks_yards_lost       INTEGER     DEFAULT 0,
  rushings_total                 INTEGER     DEFAULT 0,
  rushings_attempts              INTEGER     DEFAULT 0,
  rushings_yards_per_rush        DOUBLE PRECISION,
  interceptions_total            INTEGER     DEFAULT 0,
  fumbles_recovered              INTEGER     DEFAULT 0,
  turnovers_total                INTEGER     DEFAULT 0,
  turnovers_lost_fumbles         INTEGER     DEFAULT 0,
  turnovers_interceptions        INTEGER     DEFAULT 0,
  safeties_total                 INTEGER     DEFAULT 0,
  int_touchdowns_total           INTEGER     DEFAULT 0,
  points_against_total           INTEGER     DEFAULT 0,
  third_down_made                INTEGER     DEFAULT 0,
  third_down_attempts            INTEGER     DEFAULT 0,
  third_down_pct                 DOUBLE PRECISION,
  fourth_down_made               INTEGER     DEFAULT 0,
  fourth_down_attempts           INTEGER     DEFAULT 0,
  fourth_down_pct                DOUBLE PRECISION,
  red_zone_made                  INTEGER     DEFAULT 0,
  red_zone_att                   INTEGER     DEFAULT 0,
  red_zone_pct                   DOUBLE PRECISION,
  penalties                      INTEGER     DEFAULT 0,
  penalty_yards                  INTEGER     DEFAULT 0,
  possession_time                TEXT,
  sacks_total                    INTEGER     DEFAULT 0,
  updated_at                     TIMESTAMPTZ DEFAULT now()
);

-- index to speed up lookups by game & team
CREATE INDEX IF NOT EXISTS idx_nfl_hist_game_team_stats_game_team
  ON public.nfl_historical_game_team_stats(game_id, team_id);

COMMIT;
