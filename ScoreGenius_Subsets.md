| Role / Team             | Responsibilities                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| **Product Owner:** Matt | Sets success criteria, reviews progress, signs off launch.                                    |
| **Data Engineering:**   | Defines & maintains Supabase schemas, materialized views, ETL pipeline.                       |
| **ML Engineering:**     | Implements `nfl_features`, model training, evaluation, and prediction scripts.                |
| **Backend/API:**        | Integrates predictions into `nfl_game_schedule`, exposes REST endpoints.                      |
| **Frontend:**           | Consumes new `predicted_home_score`/`predicted_away_score` fields in GameCard, UI components. |
| **DevOps/CI:**          | Automates model retraining, view refresh, prediction jobs, and monitors runtime.              |
| **QA & Testing:**       | Author & run unit/integration tests, perform load/performance testing.                        |
