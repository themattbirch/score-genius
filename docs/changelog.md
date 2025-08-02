```yaml
# Changelog

All notable changes to this project will be documented in this file.

Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- (add upcoming changes here)

## [1.3.0] - 2025-07-17

### Added

- **Season Aggregation Layer:** Created `mv_nfl_team_boxscore_agg` to roll per‑game team box scores (2021‑2024) into season metrics (totals, averages, rates, home/away splits, advanced derived stats).
- **Public Data View:** `v_nfl_team_season_full` joining aggregated metrics with official regular‑season standings.
- **Standings Sync Function:** `sync_nfl_standings_regular()` calculates official records (regular‑season only) incl. conference/division breakdowns.
- **Validation Functions:** `validate_nfl_team_agg()` integrity checks; `check_nfl_cron_health()` job monitor; `log_refresh_mv_nfl_team_boxscore_agg()` refresh logger.
- **Team Metadata Dimension:** `nfl_teams_dim` with team_name, conference, division, hosted logo URL.
- **Seed File:** `supabase/seed/nfl_teams_dim_seed.sql` bootstrap 32 teams with Supabase‑hosted logos.

### Changed

- Normalized and cleaned `possession_time` strings → seconds.
- Parsed `passing_comp_att` (`comp/att` or `comp-att`) → numeric pass attempts.
- Added `sack_rate_per_dropback`, `win_pct`, and home/away split win % columns to aggregation MV.
- Refreshed migration structure: timestamped `nfl_team_agg_v1_3.sql` defines MV, view, funcs, cron, grants.

### Fixed

- Removed stray teams (IDs 33, 34) from dimension + storage bucket.
- Corrected initial nulls in sack rate metrics for historical seasons.

### Security

- Created role `nfl_stats_ro` (NOLOGIN) with SELECT on `v_nfl_team_season_full` only.
- Revoked all privileges on raw tables (`nfl_historical_game_stats`, `nfl_historical_game_team_stats`, `nfl_historical_team_stats`) from `anon` + `authenticated`.
- Granted `nfl_stats_ro` to `anon` + `authenticated` so clients read metrics only through the view.

### Operations

- **pg_cron jobs:**
  - `nightly_mv_nfl_team_boxscore_agg` @ 06:00 UTC – refresh aggregation MV.
  - `nightly_standings_sync` @ 06:30 UTC – rebuild official regular‑season standings.
  - `daily_nfl_cron_health` @ 12:00 UTC – job status heartbeat.
  - `daily_nfl_validation` @ 12:15 UTC – integrity assertions.
- Uploaded & rehosted all 32 team logos to Supabase Storage (`team-logos/nfl/{team_id}.png`).

### Documentation

- Updated PRD with finalized metric dictionary, refresh schedule, validation flow, and security model.
- Added scripts: `host-logos.sh` (download + upload + patch DB).

---

### Linking Versions

Tag your repo after committing the changelog:

```bash
git add CHANGELOG.md supabase/seed/nfl_teams_dim_seed.sql
git commit -m "chore: add CHANGELOG for v1.3.0 NFL aggregation"
git tag v1.3.0
git push && git push --tags
```
