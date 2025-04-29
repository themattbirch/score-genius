-- sqlfluff: disable=all

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE EXTENSION IF NOT EXISTS "pgsodium";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE OR REPLACE FUNCTION "public"."get_mlb_advanced_team_stats"("p_season_year" integer) RETURNS TABLE("team_id" bigint, "team_name" "text", "season" integer, "games_played" bigint, "wins" bigint, "runs_for" bigint, "runs_against" bigint, "run_differential" bigint, "run_differential_avg" numeric, "win_pct" numeric, "pythagorean_win_pct" numeric, "expected_wins" numeric, "luck_factor" numeric, "home_away_win_pct_split" numeric, "home_away_run_diff_avg_split" numeric)
    LANGUAGE "sql" STABLE
    AS $$
SELECT
    t.team_id,
    t.team_name::TEXT,
    t.season::INT,
    t.games_played_all::BIGINT AS games_played,
    t.wins_all_total::BIGINT AS wins,
    t.runs_for_total_all::BIGINT AS runs_for,
    t.runs_against_total_all::BIGINT AS runs_against,

    -- Run Differential (Total)
    (COALESCE(t.runs_for_total_all, 0) - COALESCE(t.runs_against_total_all, 0))::BIGINT AS run_differential,

    -- Run Differential (Average)
    (COALESCE(t.runs_for_avg_all, 0.0) - COALESCE(t.runs_against_avg_all, 0.0))::NUMERIC(5, 2) AS run_differential_avg, -- Format to 2 decimal places

    -- Actual Win Pct (ensure numeric)
    COALESCE(t.wins_all_percentage, 0.0)::NUMERIC(4, 3) AS win_pct, -- Format to 3 decimal places

    -- Pythagorean Win Pct
    CASE
        WHEN (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)) = 0 THEN 0.5 -- Handle 0 RS/RA case
        ELSE (POW(GREATEST(t.runs_for_total_all, 0), 1.83) / NULLIF( (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)), 0 ))
    END::NUMERIC(4, 3) AS pythagorean_win_pct, -- Format to 3 decimal places

    -- Expected Wins (Calculated based on Pythagorean Win Pct)
    CASE
        WHEN (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)) = 0 THEN (0.5 * COALESCE(t.games_played_all, 0)) -- Handle 0 RS/RA case
        ELSE ROUND(
             (POW(GREATEST(t.runs_for_total_all, 0), 1.83) / NULLIF( (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)), 0 ))
             * COALESCE(t.games_played_all, 0)
            )
    END::NUMERIC AS expected_wins,

     -- Luck Factor (Actual Wins - Expected Wins)
    (COALESCE(t.wins_all_total, 0) -
        CASE
          WHEN (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)) = 0 THEN (0.5 * COALESCE(t.games_played_all, 0))
          ELSE ROUND(
               (POW(GREATEST(t.runs_for_total_all, 0), 1.83) / NULLIF( (POW(GREATEST(t.runs_for_total_all, 0), 1.83) + POW(GREATEST(t.runs_against_total_all, 0), 1.83)), 0 ))
               * COALESCE(t.games_played_all, 0)
              )
        END
    )::NUMERIC AS luck_factor,

    -- Home/Away Win Pct Split
    (COALESCE(t.wins_home_percentage, 0.0) - COALESCE(t.wins_away_percentage, 0.0))::NUMERIC(4, 3) AS home_away_win_pct_split, -- Format to 3 decimal places

    -- Home/Away Avg Run Diff Split
    ( (COALESCE(t.runs_for_avg_home, 0.0) - COALESCE(t.runs_against_avg_home, 0.0)) -
      (COALESCE(t.runs_for_avg_away, 0.0) - COALESCE(t.runs_against_avg_away, 0.0))
    )::NUMERIC(5, 2) AS home_away_run_diff_avg_split -- Format to 2 decimal places

FROM
    public.mlb_historical_team_stats t -- Use the correct table name
WHERE
    t.season = p_season_year -- Filter by the input season year
ORDER BY
    t.team_name; -- Order results
$$;


ALTER FUNCTION "public"."get_mlb_advanced_team_stats"("p_season_year" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_nba_advanced_team_stats"("p_season_year" integer) RETURNS TABLE("team_name" "text", "games_played" bigint, "pace" numeric, "off_rtg" numeric, "def_rtg" numeric, "efg_pct" numeric, "tov_pct" numeric, "oreb_pct" numeric)
    LANGUAGE "sql" STABLE
    AS $$
WITH TeamGameStats AS (
    -- Unpivot home/away stats
    SELECT
        game_date, home_team AS team_name, home_fg_attempted AS fga, home_ft_attempted AS fta,
        home_turnovers AS tov, home_off_reb AS oreb, home_score AS pts_for, home_fg_made AS fgm,
        home_3pm AS three_pm, away_fg_attempted AS opp_fga, away_ft_attempted AS opp_fta,
        away_turnovers AS opp_tov, away_off_reb AS opp_oreb, away_def_reb AS opp_dreb,
        away_score AS pts_against
    FROM public.nba_historical_game_stats
    UNION ALL
    SELECT
        game_date, away_team AS team_name, away_fg_attempted AS fga, away_ft_attempted AS fta,
        away_turnovers AS tov, away_off_reb AS oreb, away_score AS pts_for, away_fg_made AS fgm,
        away_3pm AS three_pm, home_fg_attempted AS opp_fga, home_ft_attempted AS opp_fta,
        home_turnovers AS opp_tov, home_off_reb AS opp_oreb, home_def_reb AS opp_dreb,
        home_score AS pts_against
    FROM public.nba_historical_game_stats
),
FilteredGames AS (
    -- Filter by season
    SELECT * FROM TeamGameStats
    WHERE game_date >= (p_season_year || '-07-01')::date
      AND game_date <  ((p_season_year + 1) || '-07-01')::date
),
SeasonTotals AS (
    -- Aggregate components
    SELECT
        team_name, COUNT(*) AS games_played,
        SUM(COALESCE(fga, 0)) AS total_fga, SUM(COALESCE(fta, 0)) AS total_fta,
        SUM(COALESCE(tov, 0)) AS total_tov, SUM(COALESCE(oreb, 0)) AS total_oreb,
        SUM(COALESCE(opp_dreb, 0)) AS total_opp_dreb, SUM(COALESCE(pts_for, 0)) AS total_pts_for,
        SUM(COALESCE(pts_against, 0)) AS total_pts_against, SUM(COALESCE(fgm, 0)) AS total_fgm,
        SUM(COALESCE(three_pm, 0)) AS total_three_pm, SUM(COALESCE(opp_fga, 0)) AS total_opp_fga,
        SUM(COALESCE(opp_fta, 0)) AS total_opp_fta, SUM(COALESCE(opp_tov, 0)) AS total_opp_tov,
        SUM(COALESCE(oreb, 0)) AS total_team_oreb
    FROM FilteredGames
    GROUP BY team_name
)
SELECT
    st.team_name::TEXT,
    st.games_played::BIGINT,
    -- Pace
    COALESCE( (st.total_fga + (0.44 * st.total_fta) - st.total_oreb + st.total_tov) / NULLIF(st.games_played, 0), 0)::NUMERIC AS pace,
    -- OffRtg
    COALESCE( (st.total_pts_for * 100.0) / NULLIF(st.total_fga + (0.44 * st.total_fta) - st.total_oreb + st.total_tov, 0), 0)::NUMERIC AS off_rtg,
    -- DefRtg
    COALESCE( (st.total_pts_against * 100.0) / NULLIF(st.total_opp_fga + (0.44 * st.total_opp_fta) - st.total_team_oreb + st.total_opp_tov, 0), 0)::NUMERIC AS def_rtg,
    -- eFG%
    (COALESCE( (st.total_fgm + 0.5 * st.total_three_pm) * 100.0 / NULLIF(st.total_fga, 0), 0)) / 100.0::NUMERIC AS efg_pct,
    -- TOV% - Divide the result by 100.0
    (COALESCE( (st.total_tov * 100.0) / NULLIF(st.total_fga + (0.44 * st.total_fta) - st.total_oreb + st.total_tov, 0), 0)) / 100.0::NUMERIC AS tov_pct,
    -- OREB% - Divide the result by 100.0
    (COALESCE( (st.total_oreb * 100.0) / NULLIF(st.total_oreb + st.total_opp_dreb, 0), 0)) / 100.0::NUMERIC AS oreb_pct
FROM SeasonTotals st
ORDER BY team_name; -- <<< Simplified ORDER BY clause
$$;


ALTER FUNCTION "public"."get_nba_advanced_team_stats"("p_season_year" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_nba_player_season_stats"("p_season_year" integer, "p_search" "text") RETURNS TABLE("player_id" bigint, "player_name" "text", "team_name" "text", "games_played" bigint, "minutes" numeric, "points" numeric, "rebounds" numeric, "assists" numeric, "steals" numeric, "blocks" numeric, "fg_made" numeric, "fg_attempted" numeric, "three_made" numeric, "three_attempted" numeric, "ft_made" numeric, "ft_attempted" numeric, "three_pct" numeric, "ft_pct" numeric)
    LANGUAGE "sql" STABLE
    AS $$
  -- Your full corrected SELECT statement:
  SELECT
      p.player_id::BIGINT,
      MAX(p.player_name)::TEXT,
      MAX(p.team_name)::TEXT,
      COUNT(*)::BIGINT AS games_played,
      SUM(p.minutes)::NUMERIC AS minutes,
      SUM(p.points)::NUMERIC AS points,
      SUM(p.rebounds)::NUMERIC AS rebounds,
      SUM(p.assists)::NUMERIC AS assists,
      SUM(p.steals)::NUMERIC AS steals,
      SUM(p.blocks)::NUMERIC AS blocks,
      SUM(p.fg_made)::NUMERIC AS fg_made,
      SUM(p.fg_attempted)::NUMERIC AS fg_attempted,
      SUM(p.three_made)::NUMERIC AS three_made,
      SUM(p.three_attempted)::NUMERIC AS three_attempted,
      SUM(p.ft_made)::NUMERIC AS ft_made,
      SUM(p.ft_attempted)::NUMERIC AS ft_attempted,
      (COALESCE( (SUM(p.three_made) * 100.0) / NULLIF(SUM(p.three_attempted), 0), 0) / 100.0)::NUMERIC AS three_pct,
      (COALESCE( (SUM(p.ft_made) * 100.0) / NULLIF(SUM(p.ft_attempted), 0), 0) / 100.0)::NUMERIC AS ft_pct
  FROM public.nba_historical_player_stats p
  WHERE
      p.game_date >= (p_season_year || '-07-01')::date AND
      p.game_date <  ((p_season_year + 1) || '-07-01')::date AND
      (p_search IS NULL OR p.player_name ILIKE ('%' || p_search || '%'))
  GROUP BY p.player_id
  ORDER BY SUM(p.points) DESC;
$$;


ALTER FUNCTION "public"."get_nba_player_season_stats"("p_season_year" integer, "p_search" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."truncate_current_form"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
begin
  if char_length(coalesce(new.current_form, '')) > 5 then
    new.current_form := right(new.current_form, 5);
  end if;
  return new;
end;
$$;


ALTER FUNCTION "public"."truncate_current_form"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."game_cache" (
    "id" integer NOT NULL,
    "game_id" integer NOT NULL,
    "data" "jsonb" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "home_score" integer,
    "away_score" integer
);

ALTER TABLE ONLY "public"."game_cache" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."game_cache" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."game_cache_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."game_cache_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."game_cache_id_seq" OWNED BY "public"."game_cache"."id";



CREATE TABLE IF NOT EXISTS "public"."mlb_game_schedule" (
    "game_id" bigint NOT NULL,
    "scheduled_time_utc" timestamp with time zone,
    "game_date_et" "date",
    "status_detail" "text",
    "status_state" "text",
    "home_team_id" integer,
    "home_team_name" "text",
    "away_team_id" integer,
    "away_team_name" "text",
    "home_probable_pitcher_name" "text",
    "away_probable_pitcher_name" "text",
    "moneyline" "jsonb",
    "spread" "jsonb",
    "total" "jsonb",
    "moneyline_home_clean" "text",
    "moneyline_away_clean" "text",
    "spread_home_line_clean" numeric,
    "spread_home_price_clean" "text",
    "spread_away_price_clean" "text",
    "total_line_clean" numeric,
    "total_over_price_clean" "text",
    "total_under_price_clean" "text",
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "raw_api_response" "jsonb",
    "home_probable_pitcher_handedness" "text",
    "away_probable_pitcher_handedness" "text"
);

ALTER TABLE ONLY "public"."mlb_game_schedule" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_game_schedule" OWNER TO "postgres";


COMMENT ON COLUMN "public"."mlb_game_schedule"."moneyline_home_clean" IS 'Formatted American odds string for home moneyline (e.g., +150, -110)';



COMMENT ON COLUMN "public"."mlb_game_schedule"."spread_home_line_clean" IS 'Home team spread line (numeric, e.g., -1.5, +3.5)';



COMMENT ON COLUMN "public"."mlb_game_schedule"."spread_home_price_clean" IS 'Formatted American odds string for taking home spread';



COMMENT ON COLUMN "public"."mlb_game_schedule"."total_line_clean" IS 'Over/Under line (numeric, e.g., 8.5)';



COMMENT ON COLUMN "public"."mlb_game_schedule"."total_over_price_clean" IS 'Formatted American odds string for taking the Over';



COMMENT ON COLUMN "public"."mlb_game_schedule"."home_probable_pitcher_handedness" IS 'Handedness (R/L/S) of home probable pitcher from FanGraphs.';



COMMENT ON COLUMN "public"."mlb_game_schedule"."away_probable_pitcher_handedness" IS 'Handedness (R/L/S) of away probable pitcher from FanGraphs.';



CREATE TABLE IF NOT EXISTS "public"."mlb_historical_game_stats" (
    "game_id" integer NOT NULL,
    "game_date_time_utc" timestamp with time zone,
    "season" integer,
    "league_id" integer,
    "status_long" "text",
    "status_short" "text",
    "home_team_id" integer,
    "home_team_name" "text",
    "away_team_id" integer,
    "away_team_name" "text",
    "home_score" integer,
    "away_score" integer,
    "home_hits" integer,
    "away_hits" integer,
    "home_errors" integer,
    "away_errors" integer,
    "h_inn_1" integer,
    "h_inn_2" integer,
    "h_inn_3" integer,
    "h_inn_4" integer,
    "h_inn_5" integer,
    "h_inn_6" integer,
    "h_inn_7" integer,
    "h_inn_8" integer,
    "h_inn_9" integer,
    "h_inn_extra" integer,
    "a_inn_1" integer,
    "a_inn_2" integer,
    "a_inn_3" integer,
    "a_inn_4" integer,
    "a_inn_5" integer,
    "a_inn_6" integer,
    "a_inn_7" integer,
    "a_inn_8" integer,
    "a_inn_9" integer,
    "a_inn_extra" integer,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "raw_api_response" "jsonb"
);

ALTER TABLE ONLY "public"."mlb_historical_game_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_historical_game_stats" OWNER TO "postgres";


COMMENT ON TABLE "public"."mlb_historical_game_stats" IS 'Stores historical MLB game results and basic box score data fetched from api-baseball.';



COMMENT ON COLUMN "public"."mlb_historical_game_stats"."game_id" IS 'Unique game identifier from api-baseball.';



CREATE TABLE IF NOT EXISTS "public"."mlb_historical_team_stats" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "team_id" integer NOT NULL,
    "team_name" "text",
    "season" integer NOT NULL,
    "league_id" integer,
    "league_name" "text",
    "games_played_home" integer,
    "games_played_away" integer,
    "games_played_all" integer,
    "wins_home_total" integer,
    "wins_home_percentage" numeric,
    "wins_away_total" integer,
    "wins_away_percentage" numeric,
    "wins_all_total" integer,
    "wins_all_percentage" numeric,
    "losses_home_total" integer,
    "losses_home_percentage" numeric,
    "losses_away_total" integer,
    "losses_away_percentage" numeric,
    "losses_all_total" integer,
    "losses_all_percentage" numeric,
    "runs_for_total_home" integer,
    "runs_for_total_away" integer,
    "runs_for_total_all" integer,
    "runs_for_avg_home" numeric,
    "runs_for_avg_away" numeric,
    "runs_for_avg_all" numeric,
    "runs_against_total_home" integer,
    "runs_against_total_away" integer,
    "runs_against_total_all" integer,
    "runs_against_avg_home" numeric,
    "runs_against_avg_away" numeric,
    "runs_against_avg_all" numeric,
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "raw_api_response" "jsonb",
    "current_form" "text",
    CONSTRAINT "chk_current_form_maxlen" CHECK (("char_length"(COALESCE("current_form", ''::"text")) <= 5))
);

ALTER TABLE ONLY "public"."mlb_historical_team_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_historical_team_stats" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."nba_game_schedule" (
    "id" integer NOT NULL,
    "game_id" integer NOT NULL,
    "game_date" "date" NOT NULL,
    "home_team" "text" NOT NULL,
    "away_team" "text" NOT NULL,
    "scheduled_time" timestamp with time zone,
    "venue" "text",
    "status" "text" DEFAULT 'scheduled'::"text",
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "spread" "jsonb",
    "moneyline" "jsonb",
    "total" "jsonb",
    "moneyline_clean" "text",
    "spread_clean" "text",
    "total_clean" "text",
    "predicted_home_score" double precision,
    "predicted_away_score" double precision
);

ALTER TABLE ONLY "public"."nba_game_schedule" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_game_schedule" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."nba_game_schedule_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."nba_game_schedule_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."nba_game_schedule_id_seq" OWNED BY "public"."nba_game_schedule"."id";



CREATE TABLE IF NOT EXISTS "public"."nba_historical_game_stats" (
    "id" bigint NOT NULL,
    "game_id" bigint NOT NULL,
    "home_team" "text" NOT NULL,
    "away_team" "text" NOT NULL,
    "home_score" integer,
    "away_score" integer,
    "home_q1" integer,
    "home_q2" integer,
    "home_q3" integer,
    "home_q4" integer,
    "home_ot" integer,
    "away_q1" integer,
    "away_q2" integer,
    "away_q3" integer,
    "away_q4" integer,
    "away_ot" integer,
    "game_date" "date",
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "home_assists" integer DEFAULT 0,
    "home_steals" integer DEFAULT 0,
    "home_blocks" integer DEFAULT 0,
    "home_turnovers" integer DEFAULT 0,
    "home_fouls" integer DEFAULT 0,
    "away_assists" integer DEFAULT 0,
    "away_steals" integer DEFAULT 0,
    "away_blocks" integer DEFAULT 0,
    "away_turnovers" integer DEFAULT 0,
    "away_fouls" integer DEFAULT 0,
    "home_off_reb" integer DEFAULT 0,
    "home_def_reb" integer DEFAULT 0,
    "home_total_reb" integer DEFAULT 0,
    "away_off_reb" integer DEFAULT 0,
    "away_def_reb" integer DEFAULT 0,
    "away_total_reb" integer DEFAULT 0,
    "home_3pm" integer DEFAULT 0,
    "home_3pa" integer DEFAULT 0,
    "away_3pm" integer DEFAULT 0,
    "away_3pa" integer DEFAULT 0,
    "home_fg_made" integer,
    "home_fg_attempted" integer,
    "away_fg_made" integer,
    "away_fg_attempted" integer,
    "home_ft_made" integer,
    "home_ft_attempted" integer,
    "away_ft_made" integer,
    "away_ft_attempted" integer
);

ALTER TABLE ONLY "public"."nba_historical_game_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_game_stats" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."nba_historical_player_stats" (
    "id" bigint NOT NULL,
    "game_id" bigint NOT NULL,
    "player_id" bigint NOT NULL,
    "player_name" "text" NOT NULL,
    "team_id" bigint NOT NULL,
    "team_name" "text" NOT NULL,
    "minutes" numeric(4,2),
    "points" integer,
    "rebounds" integer,
    "assists" integer,
    "steals" integer,
    "blocks" integer,
    "turnovers" integer,
    "fouls" integer,
    "fg_made" integer,
    "fg_attempted" integer,
    "three_made" integer,
    "three_attempted" integer,
    "ft_made" integer,
    "ft_attempted" integer,
    "game_date" "date",
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);

ALTER TABLE ONLY "public"."nba_historical_player_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_player_stats" OWNER TO "postgres";


ALTER TABLE "public"."nba_historical_player_stats" ALTER COLUMN "id" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME "public"."nba_historical_game_stats_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



ALTER TABLE "public"."nba_historical_game_stats" ALTER COLUMN "id" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME "public"."nba_historical_game_stats_id_seq1"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."nba_historical_team_stats" (
    "id" integer NOT NULL,
    "team_id" integer NOT NULL,
    "team_name" "text" NOT NULL,
    "season" "text" NOT NULL,
    "league_id" integer NOT NULL,
    "games_played_home" integer DEFAULT 0,
    "games_played_away" integer DEFAULT 0,
    "games_played_all" integer DEFAULT 0,
    "wins_home_total" integer DEFAULT 0,
    "wins_home_percentage" double precision DEFAULT 0,
    "wins_away_total" integer DEFAULT 0,
    "wins_away_percentage" double precision DEFAULT 0,
    "wins_all_total" integer DEFAULT 0,
    "wins_all_percentage" double precision DEFAULT 0,
    "losses_home_total" integer DEFAULT 0,
    "losses_home_percentage" double precision DEFAULT 0,
    "losses_away_total" integer DEFAULT 0,
    "losses_away_percentage" double precision DEFAULT 0,
    "losses_all_total" integer DEFAULT 0,
    "losses_all_percentage" double precision DEFAULT 0,
    "points_for_total_home" integer DEFAULT 0,
    "points_for_total_away" integer DEFAULT 0,
    "points_for_total_all" integer DEFAULT 0,
    "points_for_avg_home" double precision DEFAULT 0,
    "points_for_avg_away" double precision DEFAULT 0,
    "points_for_avg_all" double precision DEFAULT 0,
    "points_against_total_home" integer DEFAULT 0,
    "points_against_total_away" integer DEFAULT 0,
    "points_against_total_all" integer DEFAULT 0,
    "points_against_avg_home" double precision DEFAULT 0,
    "points_against_avg_away" double precision DEFAULT 0,
    "points_against_avg_all" double precision DEFAULT 0,
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "current_form" "text"
);

ALTER TABLE ONLY "public"."nba_historical_team_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_team_stats" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."nba_historical_team_stats_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."nba_historical_team_stats_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."nba_historical_team_stats_id_seq" OWNED BY "public"."nba_historical_team_stats"."id";



CREATE TABLE IF NOT EXISTS "public"."nba_injuries" (
    "injury_id" "text" NOT NULL,
    "player_id" "text",
    "player_display_name" "text",
    "team_id" "text",
    "team_display_name" "text",
    "report_date_utc" timestamp with time zone,
    "injury_status" "text",
    "injury_status_abbr" "text",
    "injury_type" "text",
    "injury_location" "text",
    "injury_detail" "text",
    "injury_side" "text",
    "return_date_est" "date",
    "short_comment" "text",
    "long_comment" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "last_api_update_time" timestamp with time zone,
    "raw_api_response" "jsonb"
);

ALTER TABLE ONLY "public"."nba_injuries" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_injuries" OWNER TO "postgres";


COMMENT ON TABLE "public"."nba_injuries" IS 'Stores NBA player injury reports, likely sourced from RapidAPI Sports / ESPN data.';



COMMENT ON COLUMN "public"."nba_injuries"."injury_id" IS 'Unique ID for the specific injury report event from the API.';



COMMENT ON COLUMN "public"."nba_injuries"."player_id" IS 'Unique Player ID from the API source (e.g., ESPN ID).';



COMMENT ON COLUMN "public"."nba_injuries"."team_id" IS 'Unique Team ID from the API source (e.g., ESPN ID).';



COMMENT ON COLUMN "public"."nba_injuries"."report_date_utc" IS 'Timestamp the injury report was published/updated by the source API.';



COMMENT ON COLUMN "public"."nba_injuries"."last_api_update_time" IS 'Copy of report_date_utc for easier sorting/filtering by latest update.';



CREATE TABLE IF NOT EXISTS "public"."nba_live_game_stats" (
    "id" integer NOT NULL,
    "game_id" integer NOT NULL,
    "home_team" "text",
    "away_team" "text",
    "home_score" integer,
    "away_score" integer,
    "home_q1" integer,
    "home_q2" integer,
    "home_q3" integer,
    "home_q4" integer,
    "home_ot" integer,
    "away_q1" integer,
    "away_q2" integer,
    "away_q3" integer,
    "away_q4" integer,
    "away_ot" integer,
    "game_date" timestamp with time zone,
    "home_assists" integer,
    "home_steals" integer,
    "home_blocks" integer,
    "home_turnovers" integer,
    "home_fouls" integer,
    "away_assists" integer,
    "away_steals" integer,
    "away_blocks" integer,
    "away_turnovers" integer,
    "away_fouls" integer,
    "home_off_reb" integer,
    "home_def_reb" integer,
    "home_total_reb" integer,
    "away_off_reb" integer,
    "away_def_reb" integer,
    "away_total_reb" integer,
    "home_3pm" integer,
    "home_3pa" integer,
    "away_3pm" integer,
    "away_3pa" integer,
    "current_quarter" integer DEFAULT 0,
    "status" "text",
    "home_fg_made" integer,
    "home_fg_attempted" integer,
    "away_fg_made" integer,
    "away_fg_attempted" integer,
    "home_ft_made" integer,
    "home_ft_attempted" integer,
    "away_ft_made" integer,
    "away_ft_attempted" integer
);

ALTER TABLE ONLY "public"."nba_live_game_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_game_stats" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."nba_live_player_stats" (
    "id" bigint NOT NULL,
    "game_id" bigint NOT NULL,
    "player_id" bigint NOT NULL,
    "player_name" "text" NOT NULL,
    "team_id" bigint NOT NULL,
    "team_name" "text" NOT NULL,
    "minutes" "text",
    "points" integer,
    "rebounds" integer,
    "assists" integer,
    "steals" integer,
    "blocks" integer,
    "turnovers" integer,
    "fouls" integer,
    "fg_made" integer,
    "fg_attempted" integer,
    "three_made" integer,
    "three_attempted" integer,
    "ft_made" integer,
    "ft_attempted" integer,
    "game_date" "date",
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "status" "text"
);

ALTER TABLE ONLY "public"."nba_live_player_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_player_stats" OWNER TO "postgres";


ALTER TABLE "public"."nba_live_player_stats" ALTER COLUMN "id" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME "public"."nba_live_game_stats_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE SEQUENCE IF NOT EXISTS "public"."nba_live_game_stats_id_seq1"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."nba_live_game_stats_id_seq1" OWNER TO "postgres";


ALTER SEQUENCE "public"."nba_live_game_stats_id_seq1" OWNED BY "public"."nba_live_game_stats"."id";



CREATE TABLE IF NOT EXISTS "public"."nba_live_team_stats" (
    "id" integer NOT NULL,
    "team_id" integer NOT NULL,
    "team_name" "text" NOT NULL,
    "season" "text" NOT NULL,
    "league_id" integer NOT NULL,
    "games_played_home" integer DEFAULT 0,
    "games_played_away" integer DEFAULT 0,
    "games_played_all" integer DEFAULT 0,
    "wins_home_total" integer DEFAULT 0,
    "wins_home_percentage" double precision DEFAULT 0,
    "wins_away_total" integer DEFAULT 0,
    "wins_away_percentage" double precision DEFAULT 0,
    "wins_all_total" integer DEFAULT 0,
    "wins_all_percentage" double precision DEFAULT 0,
    "losses_home_total" integer DEFAULT 0,
    "losses_home_percentage" double precision DEFAULT 0,
    "losses_away_total" integer DEFAULT 0,
    "losses_away_percentage" double precision DEFAULT 0,
    "losses_all_total" integer DEFAULT 0,
    "losses_all_percentage" double precision DEFAULT 0,
    "points_for_total_home" integer DEFAULT 0,
    "points_for_total_away" integer DEFAULT 0,
    "points_for_total_all" integer DEFAULT 0,
    "points_for_avg_home" double precision DEFAULT 0,
    "points_for_avg_away" double precision DEFAULT 0,
    "points_for_avg_all" double precision DEFAULT 0,
    "points_against_total_home" integer DEFAULT 0,
    "points_against_total_away" integer DEFAULT 0,
    "points_against_total_all" integer DEFAULT 0,
    "points_against_avg_home" double precision DEFAULT 0,
    "points_against_avg_away" double precision DEFAULT 0,
    "points_against_avg_all" double precision DEFAULT 0,
    "last_fetched_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);

ALTER TABLE ONLY "public"."nba_live_team_stats" FORCE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_team_stats" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."nba_live_team_stats_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."nba_live_team_stats_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."nba_live_team_stats_id_seq" OWNED BY "public"."nba_live_team_stats"."id";



CREATE TABLE IF NOT EXISTS "public"."team_box_scores" (
    "game_id" bigint NOT NULL,
    "team_id" integer NOT NULL,
    "game_date" timestamp with time zone NOT NULL,
    "score_for" integer,
    "score_against" integer,
    "net_rating" double precision,
    "pace" double precision,
    "tov_rate" double precision
);


ALTER TABLE "public"."team_box_scores" OWNER TO "postgres";


COMMENT ON TABLE "public"."team_box_scores" IS 'Stores team-level box score statistics for each game.';



COMMENT ON COLUMN "public"."team_box_scores"."game_id" IS 'Unique identifier for the game.';



COMMENT ON COLUMN "public"."team_box_scores"."team_id" IS 'Unique identifier for the team.';



COMMENT ON COLUMN "public"."team_box_scores"."game_date" IS 'Date and time the game was played.';



COMMENT ON COLUMN "public"."team_box_scores"."score_for" IS 'Points scored by the team in the game.';



COMMENT ON COLUMN "public"."team_box_scores"."score_against" IS 'Points scored against the team in the game.';



COMMENT ON COLUMN "public"."team_box_scores"."net_rating" IS 'Offensive rating minus defensive rating for the game.';



COMMENT ON COLUMN "public"."team_box_scores"."pace" IS 'Estimated number of possessions per 48 minutes for the game.';



COMMENT ON COLUMN "public"."team_box_scores"."tov_rate" IS 'Turnover rate (turnovers per 100 possessions) for the game.';



ALTER TABLE ONLY "public"."game_cache" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."game_cache_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."nba_game_schedule" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."nba_game_schedule_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."nba_historical_team_stats" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."nba_historical_team_stats_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."nba_live_game_stats" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."nba_live_game_stats_id_seq1"'::"regclass");



ALTER TABLE ONLY "public"."nba_live_team_stats" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."nba_live_team_stats_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."game_cache"
    ADD CONSTRAINT "game_cache_game_id_key" UNIQUE ("game_id");



ALTER TABLE ONLY "public"."game_cache"
    ADD CONSTRAINT "game_cache_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."mlb_game_schedule"
    ADD CONSTRAINT "mlb_game_schedule_pkey" PRIMARY KEY ("game_id");



ALTER TABLE ONLY "public"."mlb_historical_game_stats"
    ADD CONSTRAINT "mlb_historical_game_stats_pkey" PRIMARY KEY ("game_id");



ALTER TABLE ONLY "public"."mlb_historical_team_stats"
    ADD CONSTRAINT "mlb_historical_team_stats_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_game_schedule"
    ADD CONSTRAINT "nba_game_schedule_game_id_key" UNIQUE ("game_id");



ALTER TABLE ONLY "public"."nba_game_schedule"
    ADD CONSTRAINT "nba_game_schedule_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_historical_player_stats"
    ADD CONSTRAINT "nba_historical_game_stats_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_historical_game_stats"
    ADD CONSTRAINT "nba_historical_game_stats_pkey1" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_historical_player_stats"
    ADD CONSTRAINT "nba_historical_player_stats_game_player_key" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_historical_player_stats"
    ADD CONSTRAINT "nba_historical_player_stats_game_player_uniq" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_historical_team_stats"
    ADD CONSTRAINT "nba_historical_team_stats_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_injuries"
    ADD CONSTRAINT "nba_injuries_pkey" PRIMARY KEY ("injury_id");



ALTER TABLE ONLY "public"."nba_live_player_stats"
    ADD CONSTRAINT "nba_live_game_stats_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_live_game_stats"
    ADD CONSTRAINT "nba_live_game_stats_pkey1" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."nba_live_player_stats"
    ADD CONSTRAINT "nba_live_player_stats_game_player_unique" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_live_team_stats"
    ADD CONSTRAINT "nba_live_team_stats_pkey" PRIMARY KEY ("id");



--ALTER TABLE ONLY "public"."team_box_scores"
  --  ADD CONSTRAINT "team_box_scores_pkey" PRIMARY KEY ("game_id", "team_id");



ALTER TABLE ONLY "public"."nba_historical_game_stats"
    ADD CONSTRAINT "unique_game_historical" UNIQUE ("game_id");



ALTER TABLE ONLY "public"."nba_live_game_stats"
    ADD CONSTRAINT "unique_game_id" UNIQUE ("game_id");



ALTER TABLE ONLY "public"."nba_historical_player_stats"
    ADD CONSTRAINT "unique_game_player" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_historical_player_stats"
    ADD CONSTRAINT "unique_game_player_historical" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_live_player_stats"
    ADD CONSTRAINT "unique_game_player_live" UNIQUE ("game_id", "player_id");



ALTER TABLE ONLY "public"."nba_live_team_stats"
    ADD CONSTRAINT "unique_live_team_season_league" UNIQUE ("team_id", "season", "league_id");



ALTER TABLE ONLY "public"."mlb_historical_team_stats"
    ADD CONSTRAINT "unique_team_season" UNIQUE ("team_id", "season", "league_id");



ALTER TABLE ONLY "public"."nba_historical_team_stats"
    ADD CONSTRAINT "unique_team_season_league" UNIQUE ("team_id", "season", "league_id");



CREATE INDEX "idx_game_cache_game_id" ON "public"."game_cache" USING "btree" ("game_id");



CREATE INDEX "idx_live_team_stats_team_season" ON "public"."nba_live_team_stats" USING "btree" ("team_id", "season");



CREATE INDEX "idx_mlb_hist_game_away_team" ON "public"."mlb_historical_game_stats" USING "btree" ("away_team_id");



CREATE INDEX "idx_mlb_hist_game_date" ON "public"."mlb_historical_game_stats" USING "btree" ("game_date_time_utc");



CREATE INDEX "idx_mlb_hist_game_home_team" ON "public"."mlb_historical_game_stats" USING "btree" ("home_team_id");



CREATE INDEX "idx_mlb_hist_game_season" ON "public"."mlb_historical_game_stats" USING "btree" ("season");



CREATE INDEX "idx_mlb_hist_game_status" ON "public"."mlb_historical_game_stats" USING "btree" ("status_short");



CREATE INDEX "idx_mlb_hist_team_stats_team_season" ON "public"."mlb_historical_team_stats" USING "btree" ("team_id", "season");



CREATE INDEX "idx_mlb_schedule_away_team" ON "public"."mlb_game_schedule" USING "btree" ("away_team_id");



CREATE INDEX "idx_mlb_schedule_game_date_et" ON "public"."mlb_game_schedule" USING "btree" ("game_date_et");



CREATE INDEX "idx_mlb_schedule_home_team" ON "public"."mlb_game_schedule" USING "btree" ("home_team_id");



CREATE INDEX "idx_mlb_schedule_scheduled_time" ON "public"."mlb_game_schedule" USING "btree" ("scheduled_time_utc");



CREATE INDEX "idx_mlb_schedule_status" ON "public"."mlb_game_schedule" USING "btree" ("status_state");



CREATE INDEX "idx_nba_game_schedule_date" ON "public"."nba_game_schedule" USING "btree" ("game_date");



CREATE INDEX "idx_nba_historical_game_stats_game_id" ON "public"."nba_historical_game_stats" USING "btree" ("game_id");



CREATE INDEX "idx_nba_historical_game_stats_player_id" ON "public"."nba_historical_player_stats" USING "btree" ("player_id");



CREATE INDEX "idx_nba_historical_game_stats_team_id" ON "public"."nba_historical_player_stats" USING "btree" ("team_id");



CREATE INDEX "idx_nba_injuries_last_api_update" ON "public"."nba_injuries" USING "btree" ("last_api_update_time");



CREATE INDEX "idx_nba_injuries_player_id" ON "public"."nba_injuries" USING "btree" ("player_id");



CREATE INDEX "idx_nba_injuries_report_date" ON "public"."nba_injuries" USING "btree" ("report_date_utc");



CREATE INDEX "idx_nba_injuries_team_id" ON "public"."nba_injuries" USING "btree" ("team_id");



CREATE INDEX "idx_nba_live_game_stats_game_id" ON "public"."nba_live_player_stats" USING "btree" ("game_id");



CREATE INDEX "idx_nba_live_game_stats_player_id" ON "public"."nba_live_player_stats" USING "btree" ("player_id");



CREATE INDEX "idx_nba_live_game_stats_team_id" ON "public"."nba_live_player_stats" USING "btree" ("team_id");



CREATE INDEX "idx_team_stats_team_season" ON "public"."nba_historical_team_stats" USING "btree" ("team_id", "season");



CREATE OR REPLACE TRIGGER "trg_truncate_current_form" BEFORE INSERT OR UPDATE ON "public"."mlb_historical_team_stats" FOR EACH ROW EXECUTE FUNCTION "public"."truncate_current_form"();



CREATE POLICY "Allow public read access on MLB historical stats" ON "public"."mlb_historical_game_stats" FOR SELECT USING (true);



CREATE POLICY "Allow public read access on NBA historical stats" ON "public"."nba_historical_game_stats" FOR SELECT USING (true);



ALTER TABLE "public"."game_cache" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_game_schedule" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_historical_game_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."mlb_historical_team_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_game_schedule" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_game_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_player_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_historical_team_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_injuries" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_game_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_player_stats" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."nba_live_team_stats" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";




















































































































































































GRANT ALL ON FUNCTION "public"."get_mlb_advanced_team_stats"("p_season_year" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."get_mlb_advanced_team_stats"("p_season_year" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_mlb_advanced_team_stats"("p_season_year" integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."get_nba_advanced_team_stats"("p_season_year" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."get_nba_advanced_team_stats"("p_season_year" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_nba_advanced_team_stats"("p_season_year" integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."get_nba_player_season_stats"("p_season_year" integer, "p_search" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."get_nba_player_season_stats"("p_season_year" integer, "p_search" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_nba_player_season_stats"("p_season_year" integer, "p_search" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."truncate_current_form"() TO "anon";
GRANT ALL ON FUNCTION "public"."truncate_current_form"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."truncate_current_form"() TO "service_role";


















GRANT ALL ON TABLE "public"."game_cache" TO "anon";
GRANT ALL ON TABLE "public"."game_cache" TO "authenticated";
GRANT ALL ON TABLE "public"."game_cache" TO "service_role";



GRANT ALL ON SEQUENCE "public"."game_cache_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."game_cache_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."game_cache_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."mlb_game_schedule" TO "anon";
GRANT ALL ON TABLE "public"."mlb_game_schedule" TO "authenticated";
GRANT ALL ON TABLE "public"."mlb_game_schedule" TO "service_role";



GRANT ALL ON TABLE "public"."mlb_historical_game_stats" TO "anon";
GRANT ALL ON TABLE "public"."mlb_historical_game_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."mlb_historical_game_stats" TO "service_role";



GRANT ALL ON TABLE "public"."mlb_historical_team_stats" TO "anon";
GRANT ALL ON TABLE "public"."mlb_historical_team_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."mlb_historical_team_stats" TO "service_role";



GRANT ALL ON TABLE "public"."nba_game_schedule" TO "anon";
GRANT ALL ON TABLE "public"."nba_game_schedule" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_game_schedule" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_game_schedule_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_game_schedule_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_game_schedule_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."nba_historical_game_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_historical_game_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_historical_game_stats" TO "service_role";



GRANT ALL ON TABLE "public"."nba_historical_player_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_historical_player_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_historical_player_stats" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq1" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq1" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_historical_game_stats_id_seq1" TO "service_role";



GRANT ALL ON TABLE "public"."nba_historical_team_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_historical_team_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_historical_team_stats" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_historical_team_stats_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_historical_team_stats_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_historical_team_stats_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."nba_injuries" TO "anon";
GRANT ALL ON TABLE "public"."nba_injuries" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_injuries" TO "service_role";



GRANT ALL ON TABLE "public"."nba_live_game_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_live_game_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_live_game_stats" TO "service_role";



GRANT ALL ON TABLE "public"."nba_live_player_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_live_player_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_live_player_stats" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq1" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq1" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_live_game_stats_id_seq1" TO "service_role";



GRANT ALL ON TABLE "public"."nba_live_team_stats" TO "anon";
GRANT ALL ON TABLE "public"."nba_live_team_stats" TO "authenticated";
GRANT ALL ON TABLE "public"."nba_live_team_stats" TO "service_role";



GRANT ALL ON SEQUENCE "public"."nba_live_team_stats_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."nba_live_team_stats_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."nba_live_team_stats_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."team_box_scores" TO "anon";
GRANT ALL ON TABLE "public"."team_box_scores" TO "authenticated";
GRANT ALL ON TABLE "public"."team_box_scores" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";






























RESET ALL;
