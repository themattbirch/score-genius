```yaml

# NFL Team Aggregation Layer (v1.3)

## Overview

Season-level performance metrics from API-Sports box scores.

## Source Tables

- nfl_historical_game_team_stats (2 rows/game/team metrics)
- nfl_historical_game_stats (scores, stage)
- nfl_teams_dim (metadata)

## Materialized View

- mv_nfl_team_boxscore_agg: all games (regular + postseason)
- Refresh nightly 06:00 UTC via pg_cron.

## Official Standings

- sync_nfl_standings_regular() rebuilds `nfl_historical_team_stats` from **Regular Season games only**.
- Cron at 06:30 UTC.

## Metrics Dictionary

(link to section; list key cols & formulas)

## Cron Jobs

(table listing jobname, schedule, purpose)

## Validation

- daily_nfl_validation (plpgsql assertions)
- daily_nfl_cron_health (alerts if failures)

## Security

- Role `nfl_stats_ro` w/ SELECT on `v_nfl_team_season_full` only.
- Grants to `anon` & `authenticated`.

## Adding a New Season

1. Ingest per-game data.
2. Run MV refresh (or wait overnight).
3. Run standings sync (or wait).
4. Run validation.
5. Done.

## Logo Hosting

Supabase Storage bucket `team-logos/nfl/<team_id>.png`.
```
