-- backend/sql/migration/001_create_team_rolling_20.sql
-- 20-game window materialised view
CREATE MATERIALIZED VIEW IF NOT EXISTS team_rolling_20 AS
SELECT
  game_id,
  game_date,
  team_id,
  AVG(score_for)  OVER w AS rolling_score_for_mean_20,
  STDDEV(score_for) OVER w AS rolling_score_for_std_20,
  AVG(score_against)  OVER w AS rolling_score_against_mean_20,
  STDDEV(score_against) OVER w AS rolling_score_against_std_20,
  AVG(net_rating)  OVER w AS rolling_net_rating_mean_20,
  STDDEV(net_rating) OVER w AS rolling_net_rating_std_20,
  AVG(pace)        OVER w AS rolling_pace_mean_20,
  STDDEV(pace)     OVER w AS rolling_pace_std_20,
  AVG(tov_rate)    OVER w AS rolling_tov_rate_mean_20,
  STDDEV(tov_rate) OVER w AS rolling_tov_rate_std_20
FROM team_box_scores
WINDOW w AS (
  PARTITION BY team_id
  ORDER BY game_date
  ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
);

CREATE INDEX IF NOT EXISTS idx_team_rolling_20
  ON team_rolling_20(team_id, game_date);
