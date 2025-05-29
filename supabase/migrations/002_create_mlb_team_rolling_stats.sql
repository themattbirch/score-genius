-- backend/sql/migration/002_create_mlb_team_rolling_20.sql
-- 10-game window materialized view

CREATE MATERIALIZED VIEW IF NOT EXISTS mlb_team_rolling_10_features AS
WITH mlb_team_game_box_scores AS (
    -- Data for home teams
    SELECT
        game_id,
        DATE(game_date_time_utc)       AS game_date,
        season,
        home_team_id                   AS team_id,
        home_team_name                 AS team_name,
        away_team_id                   AS opponent_team_id,
        home_score                     AS runs_scored,
        away_score                     AS runs_allowed,
        (home_score - away_score)      AS run_differential,
        home_hits                      AS hits_for,
        away_hits                      AS hits_against,
        home_errors                    AS errors_committed,
        away_errors                    AS errors_by_opponent
    FROM
        mlb_historical_game_stats
    WHERE
        status_short = 'FT'
        OR status_long  = 'Finished'

    UNION ALL

    -- Data for away teams
    SELECT
        game_id,
        DATE(game_date_time_utc)       AS game_date,
        season,
        away_team_id                   AS team_id,
        away_team_name                 AS team_name,
        home_team_id                   AS opponent_team_id,
        away_score                     AS runs_scored,
        home_score                     AS runs_allowed,
        (away_score - home_score)      AS run_differential,
        away_hits                      AS hits_for,
        home_hits                      AS hits_against,
        away_errors                    AS errors_committed,
        home_errors                    AS errors_by_opponent
    FROM
        mlb_historical_game_stats
    WHERE
        status_short = 'FT'
        OR status_long  = 'Finished'
)

-- Main SELECT for the materialized view

SELECT
    s.game_id,
    s.game_date,
    s.team_id,
    s.season,
    AVG   (s.runs_scored)      OVER w AS rolling_runs_scored_mean_10,
    STDDEV(s.runs_scored)      OVER w AS rolling_runs_scored_std_10,
    AVG   (s.runs_allowed)     OVER w AS rolling_runs_allowed_mean_10,
    STDDEV(s.runs_allowed)     OVER w AS rolling_runs_allowed_std_10,
    AVG   (s.run_differential) OVER w AS rolling_run_diff_mean_10,
    STDDEV(s.run_differential) OVER w AS rolling_run_diff_std_10,
    AVG   (s.hits_for)         OVER w AS rolling_hits_for_mean_10,
    STDDEV(s.hits_for)         OVER w AS rolling_hits_for_std_10,
    AVG   (s.hits_against)     OVER w AS rolling_hits_against_mean_10,
    STDDEV(s.hits_against)     OVER w AS rolling_hits_against_std_10,
    AVG   (s.errors_committed) OVER w AS rolling_errors_committed_mean_10,
    STDDEV(s.errors_committed) OVER w AS rolling_errors_committed_std_10
FROM
    mlb_team_game_box_scores AS s
WINDOW w AS (
    PARTITION BY s.team_id, s.season
    ORDER BY    s.game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
)
ORDER BY
    s.team_id,
    s.game_date;

-- Supporting indexes

CREATE INDEX IF NOT EXISTS idx_mlb_team_rolling_10_features_team_date
    ON mlb_team_rolling_10_features (team_id, game_date);

-- ✅ UNIQUE so CONCURRENTLY will work
CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_team_roll10_uq
    ON mlb_team_rolling_10_features (game_id, team_id);
