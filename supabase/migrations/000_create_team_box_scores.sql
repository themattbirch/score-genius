-- 000_create_team_box_scores.sql
CREATE TABLE IF NOT EXISTS public.team_box_scores (
  game_id       BIGINT       PRIMARY KEY,
  team_id       INTEGER      NOT NULL,
  game_date     TIMESTAMPTZ  NOT NULL,
  score_for     INT          NOT NULL,
  score_against INT          NOT NULL,
  net_rating    DOUBLE PRECISION,
  pace          DOUBLE PRECISION,
  tov_rate      DOUBLE PRECISION
  -- add any other columns you use…
);