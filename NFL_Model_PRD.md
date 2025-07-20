# NFL Score Prediction Feature — Product Requirements Document (PRD)

**Date:** 2025-07-20  
**Author:** Matt Birch  
**Version:** 1.0

## 1. Executive Summary

ScoreGenius has existing, high-accuracy score-prediction models for MLB and NBA. This project will extend that capability to the NFL, delivering pre-game home/away score forecasts into the PWA and API. We’ll leverage our established data pipeline (Supabase tables, materialized views, REST endpoints) and modular ML stack (Ridge/SVR/XGBoost ensemble) to produce and serve NFL point-spread and total predictions with minimal lead time.

## 2. Objectives & Goals

Objective: Build, validate, and deploy an NFL score-prediction model end-to-end—from data ingestion through feature engineering, model training, and real-time prediction delivery.

**Primary Goals:**

- **Accuracy:** Achieve MAE ≤ 7.0 points on out-of-sample NFL games.
- **Reliability:** Ensure the prediction API endpoint is available 99.9% of the time with ≤200ms median latency.
- **Timeliness:** Generate and upsert predictions at least 1 hour before kickoff for every scheduled game.
- **Maintainability:** Mirror the modular structure of MLB/NBA codebases for minimal cognitive overhead.

## 3. Stakeholders

| Role / Team             | Responsibilities                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| **Product Owner: Matt** | Sets success criteria, reviews progress, signs off launch.                                    |
| **Data Engineering**    | Defines & maintains Supabase schemas, materialized views, ETL pipeline.                       |
| **ML Engineering**      | Implements `nfl_features`, model training, evaluation, and prediction scripts.                |
| **Backend/API**         | Integrates predictions into `nfl_game_schedule`, exposes REST endpoints.                      |
| **Frontend**            | Consumes new `predicted_home_score`/`predicted_away_score` fields in GameCard, UI components. |
| **DevOps/CI**           | Automates model retraining, view refresh, prediction jobs, and monitors runtime.              |
| **QA & Testing**        | Author & run unit/integration tests, perform load/performance testing.                        |

## 4. Success Metrics & KPIs

**Mean Absolute Error (MAE)**

- **Target:** ≤ 7.0 points on hold-out NFL regular-season games.

**API Uptime & Latency**

- **Availability:** ≥ 99.9% during NFL season.
- **Latency:** median response < 200ms for `/api/v1/nfl/predictions`.

**Prediction Coverage**

- **Completeness:** ≥ 99% of scheduled games receive predictions ≥ 1 hour before kickoff.

**Operational Metrics**

- **Job Success Rate:** ≥ 95% of daily refresh/model jobs complete without errors.
- **View Freshness:** Materialized views refreshed daily with < 5 min downtime.

**Adoption**

- **Frontend Display:** ≥ 80% of active users view the prediction badge on game screens.

## 5. Scope

### 5.1 In-Scope

- Data ingestion from existing Supabase tables: `nfl_historical_game_stats`, `nfl_historical_game_team_stats`, `nfl_historical_team_stats`.
- Materialized views for rolling windows and season aggregates.
- A new `backend/nfl_features/` package (season, rolling, momentum, advanced, h2h, engine).
- Model training scripts (`backend/nfl_score_prediction/train_models.py`) and predictor definitions (`models.py`) reusing the Ridge/SVR/XGBoost ensemble.
- Prediction script (`prediction.py`) that upserts into `nfl_game_schedule.predicted_*`.
- REST endpoint enhancements to surface predictions.
- Unit tests for feature modules; integration tests for end-to-end pipeline.
- CI/CD workflows to refresh views, retrain models nightly, and generate predictions.

### 5.2 Out-of-Scope

- Enriching venue data (weather, travel distance, altitude) in this phase.
- Custom deep-learning architectures or real-time play-by-play prediction.
- Mobile-only frontend redesigns—frontend will reuse existing components.

## 6. Requirements

### 6.1 Functional Requirements

**Feature Engineering**

- Load season-to-date stats from `mv_nfl_season_stats`.
- Load last-3-games rolling metrics from `mv_nfl_recent_form`.
- Compute momentum (win/loss streaks, point differential trends).
- Assemble a unified features table per upcoming game (home vs. away).

**Model Training**

- Train on historical seasons (e.g. 2015–2024).
- Target variable: `margin = home_score – away_score`.
- Use Ridge, SVR, XGBoost; optimize hyperparameters via grid search.
- Evaluate via time-series split (train on seasons n–2, validate on n–1, test on n).

**Prediction Generation**

- For each new `game_id`, load features, run ensemble, calculate point estimate.
- Compose an upsert payload:

````json
{
  "game_id": 12345,
  "predicted_home_score": 27.3,
  "predicted_away_score": 24.1
}
- Upsert into `nfl_game_schedule` via Supabase client.

**API Exposure**
- Extend existing NFL controller to include `predicted_home_score`/`predicted_away_score` in `/api/v1/nfl/schedule` and `/api/v1/nfl/games/:id`.

### 6.2 Non-Functional Requirements
- **Performance:** Model inference for a batch of 16 games must complete within 2s on standard VM.
- **Scalability:** Pipeline must handle up to 32 simultaneous games per run.
- **Reliability:** Automated retries on Supabase transient errors; alert on job failures.
- **Security:** Predictions are public data; follow existing ACLs—no new confidential fields.
- **Maintainability:** Code structure must mirror NBA/MLB packages; 80% test coverage on new modules.

### 6.3 Data Requirements

**Input Tables:**
- `nfl_historical_game_stats`
- `nfl_historical_game_team_stats`
- `nfl_historical_team_stats`

**Materialized Views:**
- `mv_nfl_season_stats` (aggregates by team-season)
- `mv_nfl_recent_form` (last-3-games rolling metrics)

**Model Artifacts:**
- Saved under `backend/models/saved/nfl_ridge.joblib`, `nfl_svr.joblib`, `selected_features_nfl.json`.

### 6.4 Testing Requirements
- **Unit Tests:** `tests/features/test_season.py`, `test_rolling.py`, `test_momentum.py`, `test_advanced.py`.
- **Integration Tests:** End-to-end smoke test for `/api/v1/nfl/predictions` returning valid JSON.
- **Performance Tests:** Benchmark inference time for a full slate of games.
- **Regression Tests:** Compare MAE on last full season against baseline.

## 7. Architecture & Design

```plaintext
backend/
├─ caching/
│    └ supabase_client.py
├─ nfl_features/
│    ├ season.py
│    ├ rolling.py
│    ├ momentum.py
│    ├ advanced.py
│    ├ h2h.py
│    └ engine.py
├─ nfl_score_prediction/
│    ├ train_models.py
│    ├ prediction.py
│    └ models.py
└─ server/
     └ controllers/nfl_controller.js  ← adds predictions to REST API

**Data Flow:**
Supabase tables → materialized views → Python feature modules → pandas DataFrame → model training/inference → Supabase upsert → API.

**CI/CD:**
- **Nightly Cron:**
    - Refresh materialized views.
    - Run `train_models.py` → push joblib artifacts.
    - Run `prediction.py` → upsert predictions.
- **GitHub Actions:**
    - On `push` to `main`: lint, unit tests, integration smoke tests.

## 8. Timeline & Milestones

| Week   | Dates               | Deliverables                                                              |
| ------ | ------------------- | ------------------------------------------------------------------------- |
| Week 1 | Jul 21 – Jul 27     | • Scaffold `nfl_features/` modules<br>• Define & deploy materialized views<br>• Unit tests template for feature modules |
| Week 2 | Jul 28 – Aug 3      | • Implement `season.py` & `rolling.py`, validate views<br>• Build `momentum.py` and `advanced.py` |
| Week 3 | Aug 4 – Aug 10      | • Finalize `engine.py`, assemble full feature DataFrame<br>• Create baseline training run; assess MAE |
| Week 4 | Aug 11 – Aug 17     | • Refine hyperparameters, finalize `models.py`<br>• Build `prediction.py`; upsert samples locally |
| Week 5 | Aug 18 – Aug 24     | • Integrate with prediction script; expose REST API<br>• Develop integration tests and benchmarks |
| Week 6 | Aug 25 – Aug 31     | • CI/CD pipeline: nightly retrain & predict jobs<br>• QA sign-off, monitoring, production rollout |

## 9. Dependencies & Assumptions
- **Existing Infrastructure:** MLB/NBA codebases, Supabase schemas, materialized-view refresh procedures.
- **Data Freshness:** Historical tables are up-to-date daily.
- **Compute Resources:** Local/VM GPU for training; CPU for inference.
- **Supabase Access:** Service role key with write permissions on `nfl_game_schedule`.

## 10. Risks & Mitigations

| Risk                             | Mitigation                                                                  |
| -------------------------------- | --------------------------------------------------------------------------- |
| Materialized view refresh fails  | Automate retries + alert; fallback to last successful snapshot.             |
| Model underperforms (MAE > 7.0)  | Experiment with sport-specific features; adjust ensemble weights.          |
| API latency spikes               | Cache recent predictions in Redis or edge CDN; add rate limiting.           |
| Schema changes in upstream tables| Version views with naming convention; CI schema validation.                 |
| Test coverage gaps               | Enforce 80% coverage threshold; require PR reviews for new feature code.    |
````
