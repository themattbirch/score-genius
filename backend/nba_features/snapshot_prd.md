# ScoreGenius “Snapshots” Feature PRD

**Document Version:** 1.0  
**Date:** June 9, 2025  
**Product:** ScoreGenius PWA  
**Feature:** Stat Snapshots (NBA & MLB)

---

## 1. Introduction

This document defines requirements for a “Snapshots” feature in the ScoreGenius PWA. Snapshot cards will present pre-computed visual summaries—headline stats, bar charts, radar charts, and pie charts—for individual NBA and MLB games, optimized for mobile viewing. Backend Python scripts will populate new `nba_snapshots` and `mlb_snapshots` tables in Supabase with structured JSON data.

---

## 2. Goals & Objectives

- **Quick Insights:** Enable users to grasp key game dynamics at a glance.  
- **Engagement:** Boost PWA appeal through visual summaries.  
- **Mobile-First:** Ensure snapshot cards display beautifully on smartphones.  
- **Automation:** Auto-generate and update snapshots.  
- **Consistency:** Derive snapshot data from the core feature pipeline.  
- **Scalability:** Use NBA implementation as a pattern for MLB.

---

## 3. Scope

Covers NBA and MLB:

1. **Database Schema**  
   - Define `nba_snapshots` and `mlb_snapshots` tables.  
2. **Data Generation Logic**  
   - Enhance `make_nba_snapshots.py`, create `make_mlb_snapshots.py`.  
3. **Backend API Endpoints**  
   - `/api/v1/{sport}/snapshots` for data retrieval.  
4. **Supabase RPCs**  
   - Optionally create or leverage RPCs; primary logic in Python.  
5. **Initial Data Population & Testing**  
   - Populate NBA snapshots first; verify before MLB rollout.

---

## 4. Functional Requirements

### 4.1 NBA Snapshots

#### FR.NBA.1: Snapshot Data Generation
- **Script:** `backend/nba_features/make_nba_snapshots.py` generates a JSON payload per `game_id`.  
- **Payload Fields:**  
  - `game_id` (Text)  
  - `headline_stats` (JSONB Array of `{ label, value }`)  
  - `bar_chart_data` (JSONB Array)  
  - `radar_chart_data` (JSONB Array of `{ metric, home_value, away_value }`)  
  - `pie_chart_data` (JSONB Array)  
  - `last_updated` (Timestamp with Time Zone)  
- **Data Sources:**  
  - `backend/nba_features/engine.py`  
  - Supabase tables: `nba_historical_game_stats`, `nba_historical_team_stats`  
  - RPC: `get_nba_advanced_team_stats`  
- **Normalization:** Use `normalize_team_name` for consistency.

#### FR.NBA.2: Snapshot Storage
- **Table:** `nba_snapshots`  
  ```sql
  CREATE TABLE nba_snapshots (
    game_id       TEXT PRIMARY KEY,
    headline_stats JSONB,
    bar_chart_data JSONB,
    radar_chart_data JSONB,
    pie_chart_data JSONB,
    last_updated   TIMESTAMPTZ DEFAULT now()
  );
