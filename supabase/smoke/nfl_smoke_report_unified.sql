/* ===================================================================
   NFL DATA PLATFORM: FINAL SMOKE REPORT (Unified JSON, txn-safe)
   =================================================================== */

BEGIN;

-- Keep pg_temp first so unqualified temp names resolve fast (defensive)
SET LOCAL search_path = pg_temp, public, pg_catalog;

-- -------------------------------------------------------------------
-- Temp tables for runtime checks (recreate cleanly)
-- -------------------------------------------------------------------
DROP TABLE IF EXISTS tmp_nfl_smoke_views;
CREATE TEMP TABLE tmp_nfl_smoke_views(
  view_name text,
  ok        boolean,
  msg       text
) ON COMMIT PRESERVE ROWS;

DROP TABLE IF EXISTS tmp_nfl_smoke_functions;
CREATE TEMP TABLE tmp_nfl_smoke_functions(
  fn_name text,
  ok      boolean,
  msg     text
) ON COMMIT PRESERVE ROWS;

TRUNCATE tmp_nfl_smoke_views;
TRUNCATE tmp_nfl_smoke_functions;

-- -------------------------------------------------------------------
-- Populate view runtime results
-- -------------------------------------------------------------------
DO $pop_views$
DECLARE v TEXT;
BEGIN
  FOR v IN
    SELECT unnest(ARRAY[
      'mv_nfl_team_boxscore_agg',
      'v_nfl_team_season_full',
      'mv_nfl_team_boxscore_agg_regonly',
      'v_nfl_team_season_regonly',
      'mv_nfl_team_srs_lite',
      'mv_nfl_team_sos',
      'v_nfl_team_sos',
      'v_nfl_dashboard_cards'
    ])
  LOOP
    BEGIN
      EXECUTE format('SELECT 1 FROM public.%I LIMIT 1;', v);
      INSERT INTO tmp_nfl_smoke_views VALUES (v, TRUE, NULL);
    EXCEPTION WHEN OTHERS THEN
      INSERT INTO tmp_nfl_smoke_views VALUES (v, FALSE, SQLERRM);
    END;
  END LOOP;
END
$pop_views$ LANGUAGE plpgsql;

-- -------------------------------------------------------------------
-- Populate function runtime results
-- -------------------------------------------------------------------
DO $pop_fns$
BEGIN
  -- sync_nfl_standings_regular()
  BEGIN
    PERFORM public.sync_nfl_standings_regular();
    INSERT INTO tmp_nfl_smoke_functions VALUES ('sync_nfl_standings_regular', TRUE, NULL);
  EXCEPTION WHEN OTHERS THEN
    INSERT INTO tmp_nfl_smoke_functions VALUES ('sync_nfl_standings_regular', FALSE, SQLERRM);
  END;

  -- validate_nfl_team_agg()
  BEGIN
    PERFORM public.validate_nfl_team_agg();
    INSERT INTO tmp_nfl_smoke_functions VALUES ('validate_nfl_team_agg', TRUE, NULL);
  EXCEPTION WHEN OTHERS THEN
    INSERT INTO tmp_nfl_smoke_functions VALUES ('validate_nfl_team_agg', FALSE, SQLERRM);
  END;

  -- check_nfl_cron_health()
  BEGIN
    PERFORM public.check_nfl_cron_health();
    INSERT INTO tmp_nfl_smoke_functions VALUES ('check_nfl_cron_health', TRUE, NULL);
  EXCEPTION WHEN OTHERS THEN
    INSERT INTO tmp_nfl_smoke_functions VALUES ('check_nfl_cron_health', FALSE, SQLERRM);
  END;

  -- log_refresh_mv_nfl_team_boxscore_agg()
  BEGIN
    PERFORM public.log_refresh_mv_nfl_team_boxscore_agg();
    INSERT INTO tmp_nfl_smoke_functions VALUES ('log_refresh_mv_nfl_team_boxscore_agg', TRUE, NULL);
  EXCEPTION WHEN OTHERS THEN
    INSERT INTO tmp_nfl_smoke_functions VALUES ('log_refresh_mv_nfl_team_boxscore_agg', FALSE, SQLERRM);
  END;
END
$pop_fns$ LANGUAGE plpgsql;

-- -------------------------------------------------------------------
-- MAIN REPORT: single JSONB row
-- -------------------------------------------------------------------
WITH
schema_validation AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.table_name, x.attnum) AS data
  FROM (
    SELECT
      c.relname AS table_name,
      a.attnum,
      a.attname AS column_name,
      pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
      a.attnotnull AS not_null
    FROM pg_class c
    JOIN pg_attribute a ON a.attrelid = c.oid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname='public'
      AND c.relname IN (
        'nfl_historical_game_stats',
        'nfl_historical_game_team_stats',
        'nfl_historical_team_stats'
      )
      AND a.attnum>0 AND NOT a.attisdropped
  ) x
),
key_constraints AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.table_name, x.constraint_type, x.constraint_name) AS data
  FROM (
    SELECT
      tc.table_name,
      tc.constraint_type,
      tc.constraint_name,
      string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS cols
    FROM information_schema.table_constraints tc
    LEFT JOIN information_schema.key_column_usage kcu
      ON kcu.constraint_name = tc.constraint_name
     AND kcu.table_schema   = tc.table_schema
    WHERE tc.table_schema='public'
      AND tc.table_name IN (
        'nfl_historical_game_stats',
        'nfl_historical_game_team_stats',
        'nfl_historical_team_stats'
      )
    GROUP BY tc.table_name, tc.constraint_type, tc.constraint_name
  ) x
),
row_counts AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.src, x.season) AS data
  FROM (
    SELECT 'game_stats' AS src, season, COUNT(*) AS rows
    FROM public.nfl_historical_game_stats GROUP BY season

    UNION ALL

    SELECT 'game_team_stats' AS src, season, COUNT(*) AS rows
    FROM public.nfl_historical_game_team_stats GROUP BY season

    UNION ALL

    SELECT 'team_stats' AS src, season, COUNT(*) AS rows
    FROM public.nfl_historical_team_stats GROUP BY season
  ) x
),
deps_raw AS (
  WITH RECURSIVE deps AS (
    SELECT c.oid, c.relname, c.relkind
    FROM pg_class c
    JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname='public'
      AND c.relname IN ('nfl_historical_game_stats','nfl_historical_game_team_stats')

    UNION ALL

    SELECT c.oid, c.relname, c.relkind
    FROM pg_depend d
    JOIN deps p         ON p.oid = d.refobjid
    JOIN pg_class c     ON c.oid = d.objid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE d.deptype = 'n'
      AND n.nspname = 'public'
  )
  SELECT DISTINCT relname, relkind FROM deps
),
deps AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.relkind, x.relname) AS data
  FROM deps_raw x
),
view_runtime_check AS (
  SELECT jsonb_agg(to_jsonb(r) ORDER BY r.view_name) AS data
  FROM tmp_nfl_smoke_views r
),
fn_runtime_check AS (
  SELECT jsonb_agg(to_jsonb(r) ORDER BY r.fn_name) AS data
  FROM tmp_nfl_smoke_functions r
),
triggers AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.table_name, x.trigger_name) AS data
  FROM (
    SELECT
      event_object_table AS table_name,
      trigger_name,
      action_timing,
      event_manipulation
    FROM information_schema.triggers
    WHERE event_object_schema='public'
      AND event_object_table IN (
        'nfl_historical_game_stats',
        'nfl_historical_game_team_stats',
        'nfl_historical_team_stats'
      )
  ) x
),
rls_status AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.table_name) AS data
  FROM (
    SELECT relname AS table_name, relrowsecurity AS rls_enabled
    FROM pg_class c
    JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname='public'
      AND relname IN (
        'nfl_historical_game_stats',
        'nfl_historical_game_team_stats',
        'nfl_historical_team_stats'
      )
  ) x
),
policy_list AS (
  SELECT jsonb_agg(to_jsonb(x) ORDER BY x.table_name, x.policy_name) AS data
  FROM (
    SELECT
      c.relname AS table_name,
      p.polname AS policy_name,
      p.polcmd  AS cmd,
      p.polroles::regrole[] AS roles
    FROM pg_policy p
    JOIN pg_class c ON c.oid=p.polrelid
    JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname='public'
      AND c.relname IN (
        'nfl_historical_game_stats',
        'nfl_historical_game_team_stats',
        'nfl_historical_team_stats'
      )
  ) x
)
SELECT jsonb_build_object(
  'schema_validation', (SELECT data FROM schema_validation),
  'key_constraints',   (SELECT data FROM key_constraints),
  'row_counts',        (SELECT data FROM row_counts),
  'deps',              (SELECT data FROM deps),
  'view_runtime',      (SELECT data FROM view_runtime_check),
  'function_runtime',  (SELECT data FROM fn_runtime_check),
  'triggers',          (SELECT data FROM triggers),
  'rls_status',        (SELECT data FROM rls_status),
  'policy_list',       (SELECT data FROM policy_list)
) AS report;

COMMIT;
