-- supabase/migrations/004_create_mlb_snapshots_table.sql

CREATE TABLE IF NOT EXISTS public.mlb_snapshots (
    game_id TEXT PRIMARY KEY,
    -- Or INTEGER if your game_ids are strictly numeric
    -- and used as such
    headline_stats JSONB,
    -- Key comparative metrics
    bar_chart_data JSONB,
    -- Inning scores or average runs
    radar_chart_data JSONB,
    -- Team advanced stat comparison
    pie_chart_data JSONB,
    -- Specific insights like handedness advantage
    last_updated TIMESTAMPTZ
        DEFAULT NOW() NOT NULL,
    game_date_et DATE,
    -- Store game date for easier querying
    -- or partitioning
    season INTEGER
    -- Store season year for querying
);

COMMENT ON TABLE public.mlb_snapshots IS
    'Stores pre-generated JSON snapshots of game data and features
     for frontend display for MLB games.';

CREATE INDEX IF NOT EXISTS idx_mlb_snapshots_game_date_et
    ON public.mlb_snapshots(game_date_et);

CREATE INDEX IF NOT EXISTS idx_mlb_snapshots_season
    ON public.mlb_snapshots(season);
