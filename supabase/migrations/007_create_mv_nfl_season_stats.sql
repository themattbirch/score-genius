-- supabase/migrations/007_create_mv_nfl_season_stats.sql
BEGIN;

-- Remove any old version
DROP MATERIALIZED VIEW IF EXISTS public.mv_nfl_season_stats;

CREATE MATERIALIZED VIEW public.mv_nfl_season_stats AS
SELECT
  team_id,
  season,
  /* existing SRS value */
  srs_lite,

  /* win‑percentage for whole season (ties count as half‑wins) */
  CASE
    WHEN (won + lost + ties) > 0
    THEN (won + (ties * 0.5))::float / (won + lost + ties)
  END AS wins_all_percentage,

  /* average points scored per game */
  CASE
    WHEN (won + lost + ties) > 0
    THEN points_for::float / (won + lost + ties)
  END AS points_for_avg_all,

  /* average points allowed per game */
  CASE
    WHEN (won + lost + ties) > 0
    THEN points_against::float / (won + lost + ties)
  END AS points_against_avg_all

FROM public.nfl_historical_team_stats;

-- Index for fast lookups by team and season
CREATE INDEX idx_mv_nfl_season_stats_team_season
  ON public.mv_nfl_season_stats(team_id, season);

END;
