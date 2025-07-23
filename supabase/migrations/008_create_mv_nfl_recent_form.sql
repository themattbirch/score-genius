-- supabase/migrations/008_create_mv_nfl_recent_form.sql

DROP MATERIALIZED VIEW IF EXISTS public.mv_nfl_recent_form;

CREATE MATERIALIZED VIEW public.mv_nfl_recent_form AS
WITH long_format AS (
  /* Explode each game into two team‑centric rows */

  -- Home side
  SELECT
    g.game_date,
    g.home_team_id     AS team_id,
    g.home_score       AS points_for,
    g.away_score       AS points_against,
    hgs.yards_total,
    hgs.plays_total,
    /* takeaways minus giveaways (interceptions + fumbles) */
    (
      COALESCE(hgs.fumbles_recovered, 0)
      + COALESCE(hgs.interceptions_total, 0)
      - (COALESCE(hgs.turnovers_lost_fumbles, 0) + COALESCE(hgs.turnovers_interceptions, 0))
    ) AS turnover_diff
  FROM public.nfl_historical_game_stats AS g
  INNER JOIN public.nfl_historical_game_team_stats AS hgs
    ON g.game_id = hgs.game_id
   AND g.home_team_id = hgs.team_id

  UNION ALL

  -- Away side
  SELECT
    g.game_date,
    g.away_team_id     AS team_id,
    g.away_score       AS points_for,
    g.home_score       AS points_against,
    ags.yards_total,
    ags.plays_total,
    (
      COALESCE(ags.fumbles_recovered, 0)
      + COALESCE(ags.interceptions_total, 0)
      - (COALESCE(ags.turnovers_lost_fumbles, 0) + COALESCE(ags.turnovers_interceptions, 0))
    ) AS turnover_diff
  FROM public.nfl_historical_game_stats AS g
  INNER JOIN public.nfl_historical_game_team_stats AS ags
    ON g.game_id = ags.game_id
   AND g.away_team_id = ags.team_id
),

rolling_stats AS (
  /* Compute leakage‑free 3‑game window (excludes the current game) */
  SELECT
    team_id,
    game_date,
    AVG(points_for)     OVER w  AS rolling_points_for_avg,
    AVG(points_against) OVER w  AS rolling_points_against_avg,
    CASE
      WHEN SUM(plays_total) OVER w > 0
      THEN SUM(yards_total) OVER w::float / SUM(plays_total) OVER w
    END                         AS rolling_yards_per_play_avg,
    AVG(turnover_diff)    OVER w AS rolling_turnover_differential_avg
  FROM long_format
  WINDOW w AS (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
  )
)

-- Keep only the latest snapshot per team
SELECT DISTINCT ON (team_id)
  team_id,
  rolling_points_for_avg,
  rolling_points_against_avg,
  rolling_yards_per_play_avg,
  rolling_turnover_differential_avg
FROM rolling_stats
ORDER BY team_id ASC, game_date DESC;

-- Index to accelerate lookups by team
CREATE INDEX idx_mv_nfl_recent_form_team
  ON public.mv_nfl_recent_form(team_id);

COMMIT;
