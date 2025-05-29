-- supabase/migrations/005_create_nba_snapshots_table.sql

CREATE TABLE IF NOT EXISTS public.nba_snapshots (
    game_id TEXT PRIMARY KEY,
    -- Unique identifier for the game
    headline_stats JSONB,
    -- Key comparative metrics
    bar_chart_data JSONB,
    -- Quarter-by-quarter or per-period scores
    radar_chart_data JSONB,
    -- Advanced stat comparisons between teams
    pie_chart_data JSONB,
    -- Specific insights like shot-type distributions
    last_updated TIMESTAMPTZ
        DEFAULT NOW() NOT NULL,
    game_date DATE,
    -- Use 'game_date' for NBA consistency
    season TEXT
    -- NBA season as "YYYY-YY" or start year
);

COMMENT ON TABLE public.nba_snapshots IS
    'Stores pre-generated JSON snapshots of game data and
     features for frontend display of NBA games.';

CREATE INDEX IF NOT EXISTS idx_nba_snapshots_game_date
    ON public.nba_snapshots(game_date);

CREATE INDEX IF NOT EXISTS idx_nba_snapshots_season
    ON public.nba_snapshots(season);
