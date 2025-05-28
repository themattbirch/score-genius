-- supabase/migrations/001_create_nba_team_rolling_features.sql
-- NBA 20-game window materialized view, named 'nba_team_rolling_20_features',
-- sourcing from nba_historical_game_stats via CTE.

CREATE MATERIALIZED VIEW IF NOT EXISTS nba_team_rolling_20_features AS -- <<< RENAMED HERE
WITH nba_team_game_box_scores_cte AS (
    -- Unpivot data for home teams
    SELECT
        game_id,
        game_date,
        home_team AS team_id,
        home_score AS score_for,
        away_score AS score_against,
        (home_score - away_score) AS net_score,
        home_fg_attempted AS fga,
        home_ft_attempted AS fta,
        home_off_reb AS oreb,
        home_turnovers AS tov
    FROM
        nba_historical_game_stats
    -- Consider adding a WHERE clause if nba_historical_game_stats contains many incomplete/future games
    -- e.g., WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND home_score > 0

    UNION ALL

    -- Unpivot data for away teams
    SELECT
        game_id,
        game_date,
        away_team AS team_id,
        away_score AS score_for,
        home_score AS score_against,
        (away_score - home_score) AS net_score,
        away_fg_attempted AS fga,
        away_ft_attempted AS fta,
        away_off_reb AS oreb,
        away_turnovers AS tov
    FROM
        nba_historical_game_stats
    -- Consider adding a WHERE clause
),
nba_team_game_metrics_cte AS (
    -- Calculate metrics like possessions and turnover rate
    SELECT
        game_id,
        game_date,
        team_id,
        score_for,
        score_against,
        net_score,
        tov,
        -- Possessions Estimate: FGA + 0.44 * FTA - OREB + TOV
        (COALESCE(fga, 0) + (0.44 * COALESCE(fta, 0)) - COALESCE(oreb, 0) + COALESCE(tov, 0)) AS possessions_est
    FROM
        nba_team_game_box_scores_cte
)
-- Main query for the materialized view, using the CTEs
SELECT
    s.game_id,
    s.game_date,
    s.team_id,
    AVG(s.score_for) OVER w AS rolling_score_for_mean_20,
    STDDEV(s.score_for) OVER w AS rolling_score_for_std_20,
    AVG(s.score_against) OVER w AS rolling_score_against_mean_20,
    STDDEV(s.score_against) OVER w AS rolling_score_against_std_20,
    AVG(s.net_score) OVER w AS rolling_net_rating_mean_20,
    STDDEV(s.net_score) OVER w AS rolling_net_rating_std_20,
    AVG(s.possessions_est) OVER w AS rolling_pace_mean_20,
    STDDEV(s.possessions_est) OVER w AS rolling_pace_std_20,
    AVG(CASE WHEN s.possessions_est > 0 THEN COALESCE(s.tov, 0) / s.possessions_est ELSE 0 END) OVER w AS rolling_tov_rate_mean_20,
    STDDEV(CASE WHEN s.possessions_est > 0 THEN COALESCE(s.tov, 0) / s.possessions_est ELSE 0 END) OVER w AS rolling_tov_rate_std_20
FROM
    nba_team_game_metrics_cte s
WINDOW w AS (
  PARTITION BY s.team_id
  ORDER BY s.game_date
  ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
);

-- Update index names and the ON clause to reflect the new view name
CREATE INDEX IF NOT EXISTS idx_nba_team_rolling_20_features_team_date
  ON nba_team_rolling_20_features(team_id, game_date); 

CREATE INDEX IF NOT EXISTS idx_nba_team_rolling_20_features_game_id -- <<< RENAMED HERE
ON nba_team_rolling_20_features(game_id);