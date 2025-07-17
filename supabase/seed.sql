-- supabase/seed/nfl_teams_dim_seed.sql
-- Populates / upserts the NFL teams reference dimension.
BEGIN;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (1,  'Las Vegas Raiders',        'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/1.png',  'AFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (2,  'Jacksonville Jaguars',     'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/2.png',  'AFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (3,  'New England Patriots',     'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/3.png',  'AFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (4,  'New York Giants',          'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/4.png',  'NFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (5,  'Baltimore Ravens',         'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/5.png',  'AFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (6,  'Tennessee Titans',         'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/6.png',  'AFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (7,  'Detroit Lions',            'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/7.png',  'NFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (8,  'Atlanta Falcons',          'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/8.png',  'NFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (9,  'Cleveland Browns',         'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/9.png',  'AFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (10, 'Cincinnati Bengals',       'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/10.png', 'AFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (11, 'Arizona Cardinals',        'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/11.png', 'NFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (12, 'Philadelphia Eagles',      'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/12.png', 'NFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (13, 'New York Jets',            'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/13.png', 'AFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (14, 'San Francisco 49ers',      'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/14.png', 'NFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (15, 'Green Bay Packers',        'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/15.png', 'NFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (16, 'Chicago Bears',            'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/16.png', 'NFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (17, 'Kansas City Chiefs',       'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/17.png', 'AFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (18, 'Washington Commanders',    'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/18.png', 'NFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (19, 'Carolina Panthers',        'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/19.png', 'NFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (20, 'Buffalo Bills',            'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/20.png', 'AFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (21, 'Indianapolis Colts',       'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/21.png', 'AFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (22, 'Pittsburgh Steelers',      'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/22.png', 'AFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (23, 'Seattle Seahawks',         'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/23.png', 'NFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (24, 'Tampa Bay Buccaneers',     'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/24.png', 'NFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (25, 'Miami Dolphins',           'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/25.png', 'AFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (26, 'Houston Texans',           'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/26.png', 'AFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (27, 'New Orleans Saints',       'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/27.png', 'NFC', 'South')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (28, 'Denver Broncos',           'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/28.png', 'AFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (29, 'Dallas Cowboys',           'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/29.png', 'NFC', 'East')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (30, 'Los Angeles Chargers',     'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/30.png', 'AFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (31, 'Los Angeles Rams',         'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/31.png', 'NFC', 'West')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

INSERT INTO public.nfl_teams_dim (team_id, team_name, team_logo, conference, division) VALUES
  (32, 'Minnesota Vikings',        'https://qaytaxyflvafblirxgdr.supabase.co/storage/v1/object/public/team-logos/nfl/32.png', 'NFC', 'North')
ON CONFLICT (team_id) DO UPDATE
  SET team_name  = EXCLUDED.team_name,
      team_logo  = EXCLUDED.team_logo,
      conference = EXCLUDED.conference,
      division   = EXCLUDED.division;

COMMIT;
