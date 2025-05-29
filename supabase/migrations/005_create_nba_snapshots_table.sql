-- supabase/migrations/005_create_nba_snapshots_table.sql

CREATE TABLE IF NOT EXISTS public.nba_snapshots (
    game_id TEXT PRIMARY KEY,
    headline_stats JSONB,
    bar_chart_data JSONB,
    radar_chart_data JSONB,
    pie_chart_data JSONB,
    last_updated TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    game_date DATE,    -- <<< Use 'game_date' for NBA consistency
    season TEXT        -- NBA season often "YYYY-YY" or integer start year
);

-- Optional: Add a comment to the table
COMMENT ON TABLE public.nba_snapshots IS 'Stores pre-generated JSON snapshots of game data and features for frontend display for NBA games.';

-- Optional: Indexes for querying snapshots
-- Ensure this index uses the 'game_date' column
CREATE INDEX IF NOT EXISTS idx_nba_snapshots_game_date ON public.nba_snapshots(game_date); -- <<< Corrected column name here
CREATE INDEX IF NOT EXISTS idx_nba_snapshots_season ON public.nba_snapshots(season);