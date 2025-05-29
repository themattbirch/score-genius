-- supabase/migrations/004_create_mlb_snapshots_table.sql

CREATE TABLE IF NOT EXISTS public.mlb_snapshots (
    game_id TEXT PRIMARY KEY,     -- Or INTEGER if your game_ids are strictly numeric and used as such
    headline_stats JSONB,         -- For key comparative metrics
    bar_chart_data JSONB,         -- For inning scores or avg runs
    radar_chart_data JSONB,       -- For team advanced stat comparison
    pie_chart_data JSONB,         -- For specific insights like handedness advantage
    last_updated TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    game_date_et DATE,            -- Optional: Store game date for easier querying/partitioning
    season INTEGER                -- Optional: Store season year for easier querying
);

-- Optional: Add a comment to the table
COMMENT ON TABLE public.mlb_snapshots IS 'Stores pre-generated JSON snapshots of game data and features for frontend display for MLB games.';

-- Optional: Indexes for querying snapshots, e.g., by date or season
CREATE INDEX IF NOT EXISTS idx_mlb_snapshots_game_date_et ON public.mlb_snapshots(game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_snapshots_season ON public.mlb_snapshots(season);