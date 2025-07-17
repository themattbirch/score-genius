-- supabase/migrations/20250717052035_nfl_team_agg_v1_3.sql
BEGIN;

/* -----------------------------------------------------------
   1)  Aggregation materialized view
----------------------------------------------------------- */
DROP MATERIALIZED VIEW IF EXISTS public.mv_nfl_team_boxscore_agg CASCADE;

CREATE MATERIALIZED VIEW public.mv_nfl_team_boxscore_agg AS
WITH gtm_norm AS (
  SELECT
    gtm_1.id,
    gtm_1.game_id,
    gtm_1.team_id,
    gtm_1.first_downs_total,
    gtm_1.first_downs_passing,
    gtm_1.first_downs_rushing,
    gtm_1.first_downs_penalty,
    gtm_1.plays_total,
    gtm_1.yards_total,
    gtm_1.yards_per_play,
    gtm_1.total_drives,
    gtm_1.passing_yards_per_pass,
    gtm_1.passing_interceptions,
    gtm_1.turnovers_total,
    gtm_1.possession_time,
    gtm_1.sacks_total,
    gtm_1.updated_at,
    gtm_1.raw_api_response,
    gtm_1.season,
    gtm_1.fumbles_recovered,
    gtm_1.penalties,
    gtm_1.interceptions_total,
    gtm_1.passing_total,
    gtm_1.rushings_total,
    gtm_1.rushings_attempts,
    gtm_1.rushings_yards_per_rush,
    gtm_1.turnovers_lost_fumbles,
    gtm_1.turnovers_interceptions,
    gtm_1.safeties_total,
    gtm_1.int_touchdowns_total,
    gtm_1.points_against_total,
    gtm_1.third_down_made,
    gtm_1.third_down_attempts,
    gtm_1.fourth_down_made,
    gtm_1.fourth_down_attempts,
    gtm_1.penalty_yards,
    gtm_1.red_zone_made,
    gtm_1.red_zone_att,
    gtm_1.red_zone_pct,
    gtm_1.third_down_pct,
    gtm_1.fourth_down_pct,
    gtm_1.home_team_id,
    gtm_1.home_team_name,
    gtm_1.away_team_id,
    gtm_1.away_team_name,
    gtm_1.passing_comp_att,
    gtm_1.passing_sacks_yards_lost,
    /* computed fields */
    CASE
      WHEN gtm_1.possession_time ~ '^[0-9]+:[0-9]{2}$'
      THEN split_part(gtm_1.possession_time, ':', 1)::int * 60
         + split_part(gtm_1.possession_time, ':', 2)::int
      ELSE NULL
    END AS possession_seconds,
    CASE
      WHEN regexp_replace(gtm_1.passing_comp_att, '-','/','g') ~ '^[0-9]+/[0-9]+$'
      THEN split_part(regexp_replace(gtm_1.passing_comp_att, '-','/','g'), '/', 2)::int
      ELSE NULL
    END AS pass_attempts
  FROM public.nfl_historical_game_team_stats AS gtm_1
)
SELECT
  gtm.team_id,
  gtm.season,
  COUNT(*) AS games_played,

  /* win / loss / tie counts */
  SUM(CASE
        WHEN (h.home_team_id = gtm.team_id AND h.home_score > h.away_score) OR
             (h.away_team_id = gtm.team_id AND h.away_score > h.home_score)
        THEN 1 ELSE 0
      END) AS wins_total,

  SUM(CASE
        WHEN (h.home_team_id = gtm.team_id AND h.home_score < h.away_score) OR
             (h.away_team_id = gtm.team_id AND h.away_score < h.home_score)
        THEN 1 ELSE 0
      END) AS losses_total,

  SUM(CASE
        WHEN h.home_score = h.away_score
             AND (h.home_team_id = gtm.team_id OR h.away_team_id = gtm.team_id)
        THEN 1 ELSE 0
      END) AS ties_total,

  /* win‑percentage */
  SUM(CASE
        WHEN (h.home_team_id = gtm.team_id AND h.home_score > h.away_score) OR
             (h.away_team_id = gtm.team_id AND h.away_score > h.home_score)
        THEN 1 ELSE 0
      END)::numeric
      / NULLIF(COUNT(*),0)::numeric             AS win_pct,

  /* total / average points */
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE h.away_score END) AS points_for_total,
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.away_score ELSE h.home_score END) AS points_against_total,

  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE h.away_score END)
  - SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.away_score ELSE h.home_score END) AS points_diff_total,

  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE h.away_score END)::numeric
    / NULLIF(COUNT(*),0)::numeric               AS points_for_avg,
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.away_score ELSE h.home_score END)::numeric
    / NULLIF(COUNT(*),0)::numeric               AS points_against_avg,

  /* yardage & play counts */
  SUM(gtm.yards_total)                          AS yards_total,
  AVG(gtm.yards_per_play)                       AS yards_per_play_avg,
  SUM(gtm.plays_total)                          AS plays_total,

  /* first‑downs */
  SUM(gtm.first_downs_total)                    AS first_downs_total,
  SUM(gtm.first_downs_passing)                  AS first_downs_passing,
  SUM(gtm.first_downs_rushing)                  AS first_downs_rushing,
  SUM(gtm.first_downs_penalty)                  AS first_downs_penalty,

  /* passing / rushing yardage */
  SUM(gtm.passing_total)                        AS pass_yards_total,
  AVG(gtm.passing_yards_per_pass)               AS pass_yards_per_pass_avg,
  SUM(gtm.rushings_total)                       AS rush_yards_total,
  AVG(gtm.rushings_yards_per_rush)              AS rush_yards_per_rush_avg,

  /* misc counting stats */
  SUM(gtm.sacks_total)                          AS sacks_total,
  SUM(gtm.turnovers_total)                      AS turnovers_committed_total,
  SUM(gtm.penalties)                            AS penalties_total,
  SUM(gtm.penalty_yards)                        AS penalty_yards_total,

  /* percentage metrics */
  AVG(gtm.third_down_pct)                       AS third_down_pct_avg,
  AVG(gtm.fourth_down_pct)                      AS fourth_down_pct_avg,
  AVG(gtm.red_zone_pct)                         AS red_zone_pct_avg,

  /* drives / possession */
  AVG(gtm.total_drives)                         AS drives_per_game_avg,
  AVG(gtm.possession_seconds)                   AS possession_time_avg_sec,

  /* overtime points */
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_ot ELSE h.away_ot END) AS ot_points_for,

  /* home splits */
  COUNT(*) FILTER (WHERE gtm.team_id = h.home_team_id)                                         AS games_played_home,
  SUM(CASE WHEN gtm.team_id = h.home_team_id AND h.home_score > h.away_score THEN 1 ELSE 0 END) AS wins_home_total,
  SUM(CASE WHEN gtm.team_id = h.home_team_id AND h.home_score < h.away_score THEN 1 ELSE 0 END) AS losses_home_total,
  SUM(CASE WHEN gtm.team_id = h.home_team_id AND h.home_score = h.away_score THEN 1 ELSE 0 END) AS ties_home_total,
  SUM(CASE WHEN gtm.team_id = h.home_team_id AND h.home_score > h.away_score THEN 1 ELSE 0 END)::numeric
      / NULLIF(COUNT(*) FILTER (WHERE gtm.team_id = h.home_team_id),0)::numeric               AS win_pct_home,
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE 0 END)                    AS points_for_home_total,
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.away_score ELSE 0 END)                    AS points_against_home_total,
  SUM(gtm.yards_total) FILTER (WHERE gtm.team_id = h.home_team_id)                            AS yards_total_home,
  SUM(gtm.plays_total) FILTER (WHERE gtm.team_id = h.home_team_id)                            AS plays_total_home,
  SUM(gtm.first_downs_total) FILTER (WHERE gtm.team_id = h.home_team_id)                      AS first_downs_home,
  SUM(gtm.passing_total) FILTER (WHERE gtm.team_id = h.home_team_id)                          AS pass_yards_home,
  SUM(gtm.rushings_total) FILTER (WHERE gtm.team_id = h.home_team_id)                         AS rush_yards_home,
  SUM(gtm.sacks_total)   FILTER (WHERE gtm.team_id = h.home_team_id)                          AS sacks_home,
  SUM(gtm.turnovers_total) FILTER (WHERE gtm.team_id = h.home_team_id)                        AS turnovers_home,
  SUM(gtm.penalties) FILTER (WHERE gtm.team_id = h.home_team_id)                              AS penalties_home,

  /* away splits */
  COUNT(*) FILTER (WHERE gtm.team_id = h.away_team_id)                                         AS games_played_away,
  SUM(CASE WHEN gtm.team_id = h.away_team_id AND h.away_score > h.home_score THEN 1 ELSE 0 END) AS wins_away_total,
  SUM(CASE WHEN gtm.team_id = h.away_team_id AND h.away_score < h.home_score THEN 1 ELSE 0 END) AS losses_away_total,
  SUM(CASE WHEN gtm.team_id = h.away_team_id AND h.away_score = h.home_score THEN 1 ELSE 0 END) AS ties_away_total,
  SUM(CASE WHEN gtm.team_id = h.away_team_id AND h.away_score > h.home_score THEN 1 ELSE 0 END)::numeric
      / NULLIF(COUNT(*) FILTER (WHERE gtm.team_id = h.away_team_id),0)::numeric               AS win_pct_away,
  SUM(CASE WHEN gtm.team_id = h.away_team_id THEN h.away_score ELSE 0 END)                    AS points_for_away_total,
  SUM(CASE WHEN gtm.team_id = h.away_team_id THEN h.home_score ELSE 0 END)                    AS points_against_away_total,
  SUM(gtm.yards_total) FILTER (WHERE gtm.team_id = h.away_team_id)                            AS yards_total_away,
  SUM(gtm.plays_total) FILTER (WHERE gtm.team_id = h.away_team_id)                            AS plays_total_away,
  SUM(gtm.first_downs_total) FILTER (WHERE gtm.team_id = h.away_team_id)                      AS first_downs_away,
  SUM(gtm.passing_total) FILTER (WHERE gtm.team_id = h.away_team_id)                          AS pass_yards_away,
  SUM(gtm.rushings_total) FILTER (WHERE gtm.team_id = h.away_team_id)                         AS rush_yards_away,
  SUM(gtm.sacks_total)   FILTER (WHERE gtm.team_id = h.away_team_id)                          AS sacks_away,
  SUM(gtm.turnovers_total) FILTER (WHERE gtm.team_id = h.away_team_id)                        AS turnovers_away,
  SUM(gtm.penalties) FILTER (WHERE gtm.team_id = h.away_team_id)                              AS penalties_away,

  /* rates */
  SUM(gtm.sacks_total)::numeric    / NULLIF(SUM(gtm.plays_total),0)::numeric                  AS sack_rate_per_play,
  SUM(gtm.sacks_total)::numeric    / NULLIF(SUM(gtm.pass_attempts),0)::numeric                AS sack_rate_per_dropback,
  SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE h.away_score END)::numeric
      / NULLIF(SUM(gtm.plays_total),0)::numeric                                              AS points_per_play,
  SUM(gtm.yards_total)::numeric
      / NULLIF(SUM(CASE WHEN gtm.team_id = h.home_team_id THEN h.home_score ELSE h.away_score END),0)::numeric
                                                                                              AS yards_per_point,
  SUM(gtm.penalties)::numeric      / NULLIF(SUM(gtm.plays_total),0)::numeric                  AS penalties_per_play,

  /* passing volume */
  SUM(gtm.pass_attempts) AS pass_attempts_total
FROM gtm_norm AS gtm
JOIN public.nfl_historical_game_stats AS h
  ON h.game_id = gtm.game_id
GROUP BY
  gtm.team_id,
  gtm.season
;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_nfl_team_boxscore_agg_team_season
  ON public.mv_nfl_team_boxscore_agg(team_id, season);

/* -----------------------------------------------------------
   2)  Convenience view
----------------------------------------------------------- */
CREATE OR REPLACE VIEW public.v_nfl_team_season_full AS
SELECT *
FROM public.mv_nfl_team_boxscore_agg;

/* -----------------------------------------------------------
   3)  Support functions
----------------------------------------------------------- */
/* paste pg_get_functiondef output for each function here,
   ending each with a semicolon */

COMMIT;

/* -----------------------------------------------------------
   4)  Initial populate (outside txn)
----------------------------------------------------------- */
REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_nfl_team_boxscore_agg;

/* -----------------------------------------------------------
   5)  pg_cron jobs
----------------------------------------------------------- */
DO $$BEGIN
  IF EXISTS (SELECT 1 FROM cron.job WHERE jobname='nightly_mv_nfl_team_boxscore_agg')
  THEN PERFORM cron.unschedule('nightly_mv_nfl_team_boxscore_agg');
  END IF;
END$$;

SELECT cron.schedule(
  'nightly_mv_nfl_team_boxscore_agg', '0 6 * * *',
  $$SELECT log_refresh_mv_nfl_team_boxscore_agg();$$
);

/* repeat for nightly_standings_sync, daily_nfl_validation, daily_nfl_cron_health */

/* -----------------------------------------------------------
   6)  Security grants
----------------------------------------------------------- */
DO $$BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='nfl_stats_ro') THEN
    CREATE ROLE nfl_stats_ro NOLOGIN;
  END IF;
END$$;

GRANT CONNECT ON DATABASE postgres TO nfl_stats_ro;
GRANT USAGE  ON SCHEMA  public      TO nfl_stats_ro;
GRANT SELECT ON public.v_nfl_team_season_full TO nfl_stats_ro;
REVOKE SELECT ON public.nfl_historical_game_team_stats FROM nfl_stats_ro;
REVOKE SELECT ON public.nfl_historical_game_stats      FROM nfl_stats_ro;
REVOKE SELECT ON public.nfl_historical_team_stats      FROM nfl_stats_ro;
GRANT nfl_stats_ro TO anon;
GRANT nfl_stats_ro TO authenticated;
