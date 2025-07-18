# ScoreGenius

ScoreGenius is an AI-driven live sports analytics Progressive Web App (PWA) that delivers real-time predictive analysis, natural language game recaps, and actionable betting insights for NBA games, with potential expansion to MLB and NFL. Leveraging machine learning, robust feature engineering via database views, and a modular backend, ScoreGenius provides dynamic predictions and insights.

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture & System Flow](#architecture--system-flow)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running Locally](#running-locally)
   - [Running Scripts](#running-scripts)
6. [License](#license)
7. [Contact](#contact)

## Key Features

### Predictive Analytics

- Win probability and score predictions
- Uncertainty estimation for predictions (planned)

### AI-Powered Game Recaps & Analysis

- Automated natural language game recaps (planned)
- Post-game detailed analysis with interactive visualizations (planned)

### Sports Betting & Insights

- Comparison of model predictions with market odds
- Historical trend analysis (planned)

### Robust Data Pipeline & Backend

- Automated ingestion of historical game data via Python scripts (`backend/data_pipeline/`)
- Data storage and querying using Supabase (PostgreSQL)
- Modular feature engineering pipeline (`backend/features/`)
- Database schema managed via SQL migrations (`supabase/migrations/`)

## Architecture & System Flow

### Data Ingestion & Storage

1. Historical game and team data ingested via Python scripts in `backend/data_pipeline/`
2. Data stored in a Supabase (PostgreSQL) database
3. Shared Supabase client in `backend/caching/supabase_client.py`

### Database Schema & Migrations

- SQL migration files in `supabase/migrations/`, managed with the Supabase CLI
- Materialized views (e.g., `team_rolling_20`) for complex rolling window features
- SQL linting with `sqlfluff` (planned)

### Feature Engineering

- Modular pipeline in `backend/features/` orchestrated by `engine.py`
- Modules include:
  - `rolling.py`: rolling statistics from SQL views with Python fallback
  - `h2h.py`: head-to-head matchup features
  - `rest.py`: rest days and schedule density
  - `form.py`: recent form and streaks
  - `season.py`: seasonal context
  - `momentum.py`: intra-game momentum (NBA-specific)
  - `advanced.py`: advanced stats (NBA-specific)
- Legacy code in `backend/features/legacy/` for reference

### Model Training & Prediction

- Training scripts in `backend/nba_score_prediction/train_models.py`
- Models: Ridge Regression, Support Vector Regression (SVR)
- Hyperparameter tuning with `RandomizedSearchCV` and `TimeSeriesSplit`
- Prediction module in `backend/nba_score_prediction/prediction.py`
- Ensemble and calibration methods (planned)

### Backend API

- Node.js/Express server in `backend/server/` exposing data and prediction endpoints

### Frontend PWA

- React/TypeScript application with PWA features displaying predictions and insights

## Tech Stack

### Frontend

- React 18 with TypeScript
- Vite build tool
- Tailwind CSS
- PWA support
- Data visualization libraries (Recharts, Chart.js)

### Backend & Data

- Python 3.11+
- Libraries: pandas, numpy, scikit-learn
- Supabase (PostgreSQL)
- Configuration: python-dotenv
- SQL migrations: Supabase CLI
- SQL linting: sqlfluff (planned)
- API server: Node.js with Express

### Machine Learning Models

- Ridge Regression
- Support Vector Regression (SVR)
- Feature engineering via Python modules and SQL views
- Hyperparameter tuning with RandomizedSearchCV

### Data Sources

- API-Sports
- Odds APIs (e.g., The Odds API)
- RapidAPI
- Supabase historical tables

## Project Structure

score-genius/
├── .gitignore # Should ignore: node_modules, venv*, dist, .DS_Store, *.pyc, pycache, backend/.env, reports/, remote_dump.sql, etc.
├── backend/
│ ├── **init**.py # Makes 'backend' a package
│ ├── .env # <<< Primary .env file location
│ ├── api_integration.py # <<< Mentioned file, assuming it lives here
│ ├── config.py # <<< Handles loading backend/.env, defines config vars
│ ├── caching/
│ │ ├── **init**.py
│ │ ├── supabase_client.py # <<< Initializes shared Supabase client (uses os.getenv)
│ │ └── supabase_stats.py
│ ├── data_pipeline/
│ │ ├── **init**.py
│ │ ├── config.py # <<< Does this still exist/is it needed? Maybe merged into backend/config.py?
│ │ ├── nba_game_stats_historical.py
│ │ └── nba_player_stats_historical.py
│ ├── features/ # Refactored feature engineering modules
│ │ ├── **init**.py # Exports FeatureEngine (from .legacy for now based on fixes)
│ │ ├── engine.py # New orchestrator (potentially incomplete)
│ │ ├── utils.py # Shared helpers for features
│ │ ├── momentum.py
│ │ ├── rolling.py # Handles rolling features (SQL + legacy fallback)
│ │ ├── base_windows.py # Fetches materialized view data
│ │ ├── advanced.py # NBA advanced features (updated)
│ │ ├── rest.py
│ │ ├── h2h.py
│ │ ├── season.py
│ │ ├── form.py
│ │ └── legacy/ # Original feature engineering code
│ │ ├── **init**.py # <<< Added to make legacy a package
│ │ └── feature_engineering.py # <<< Defines FeatureEngine currently being used
│ ├── mlb_features/ # NEW: MLB feature engineering modules (mirroring NBA)
│ │ ├── **init**.py
│ │ ├── advanced.py # NEW: MLB advanced features
│ │ ├── engine.py # NEW: MLB feature pipeline orchestrator (assumed)
│ │ ├── handedness_for_display.py # NEW: MLB handedness display logic
│ │ ├── make_mlb_snapshots.py # NEW: Script to generate MLB snapshots
│ │ └── utils.py # NEW: MLB specific utilities (assumed)
│ ├── nba_features/ # Existing NBA feature engineering modules
│ │ ├── **init**.py
│ │ ├── advanced.py # UPDATED: NBA advanced features
│ │ ├── engine.py # UPDATED: NBA feature pipeline orchestrator
│ │ ├── game_advanced_metrics.py # NEW: Calculates single-game advanced NBA metrics
│ │ ├── make_nba_snapshots.py # UPDATED: Script to generate NBA snapshots
│ │ └── utils.py # NBA specific utilities
│ ├── models/ # <<< Assumed location for saved model files (joblib)
│ │ └── saved/
│ │ ├── ridge_score_predictor.joblib
│ │ ├── svr_score_predictor.joblib
│ │ └── selected_features.json
│ ├── nba_score_prediction/ # ML Training, Prediction, Evaluation etc.
│ │ ├── **init**.py # <<< Added to make nba_score_prediction a package
│ │ ├── train_models.py # Main training script
│ │ ├── prediction.py # Script/module for generating predictions
│ │ ├── evaluation.py # Module for model evaluation logic/plots
│ │ ├── models.py # Defines predictor classes (RidgeScorePredictor, SVR...)
│ │ ├── simulation.py # Prediction uncertainty/simulation logic(Potential name conflict with features/utils.py)
│ │ └── dummy_modules.py # Fallback dummy implementations
│ ├── scripts/ # <<< Utility or maintenance scripts (Python)
│ │ ├── rebuild_feature_store.py # Script to regenerate features in DB
│ │ ├── run_ablation.py # Script for ablation / data leakage.
│ │ └── feature_check.py # NEW: Script for checking features
│ ├── server/ # Node.js Express API backend (Serves the PWA)
│ │ ├── node_modules/ # Node.js dependencies (...)
│ │ ├── controllers/ # API request handlers (...)
│ │ ├── routes/ # API URL definitions (...)
│ │ ├── services/ # Business logic (...)
│ │ ├── utils/ # Shared utilities for Node.js server (...)
│ │ ├── package.json
│ │ ├── package-lock.json
│ │ └── server.js
│ ├── tests/ # <<< Unit and integration tests (Python)
│ │ ├── **init**.py # Makes 'tests' a Python sub-package
│ │ └── features/ # Tests specifically for the 'features' modules
│ │ ├── **init**.py # Makes 'tests/features' a Python sub-package
│ │ ├── test_rolling.py # Unit tests for rolling.py
│ │ ├── test_h2h.py # Unit tests for h2h.py (Example)
│ │ ├── test_elo.py # Unit tests for elo.py (Example)
│ │ └── test_team_form.py # Unit tests for team_form.py (Example)
│ └── venv_pytorch/ # Python virtual environment (...)
│
├── frontend/
│ ├── app.html
│ ├── home.html
│ ├── postcss.config.js
│ ├── tailwind.config.cjs # UPDATED: For custom colors
│ ├── tsconfig.json
│ ├── tsconfig.node.json
│ ├── vite.config.ts
│ ├── public/ # Static assets (icons, manifest)
│ │ ├── favicon.ico
│ │ ├── manifest.webmanifest
│ │ ├── documentation.html # NEW: Documentation page
│ │ ├── icons/ # (all maskable/monochrome/png icons) (...)
│ │ └── logos/ # NEW: Directory for logo images (...)
│ ├── dist/ # Build output (Git-ignored) (...)
│ └── src/ # React PWA source code
│ ├── api/ # API integration modules (...)
│ ├── components/ # React components
│ │ ├── games/ # NEW: Game-specific components
│ │ │ ├── charts/ # NEW: Chart components for snapshots
│ │ │ │ ├── bar_chart_component.jsx # NEW: Recharts Bar Chart
│ │ │ │ ├── pie_chart_component.jsx # NEW: Recharts Pie Chart
│ │ │ │ └── radar_chart_component.jsx # NEW: Recharts Radar Chart
│ │ │ ├── game_card.tsx # UPDATED: Now includes SnapshotButton & WeatherBadge
│ │ │ ├── headline_grid.jsx # NEW: Displays headline stats
│ │ │ ├── snapshot_button.jsx # NEW: Button to open snapshot modal
│ │ │ ├── snapshot_modal.jsx # NEW: Full-screen snapshot modal
│ │ │ └── weather_badge.jsx # NEW: MLB weather stub badge
│ │ ├── layout/ # NEW: Layout specific components
│ │ │ └── logo_wordmark.tsx # NEW: Logo wordmark component
│ │ ├── ui/ # NEW: General UI components
│ │ │ └── skeleton_loader.jsx # NEW: Generic skeleton loader
│ │ └── (...) # Other general components
│ ├── contexts/ # React contexts (...)
│ ├── hooks/ # NEW: Custom React hooks
│ │ ├── use_network_status.ts # NEW: Hook for network status
│ │ └── use_snapshot.js # NEW: Hook to fetch snapshot data
│ ├── screens/ # Top-level screen components (...)
│ │ ├── game_detail_screen.tsx # UPDATED: Reverted snapshot logic
│ │ └── game_screen.tsx # UPDATED: To render GameCard(s)
│ ├── scripts/ # NEW: Frontend specific scripts (e.g., utility functions) (...)
│ ├── types/ # TypeScript type definitions (...)
│ │ └── index.ts # UPDATED: UnifiedGame type
│ ├── App.tsx # UPDATED: Main React application component (QueryClientProvider)
│ ├── main.tsx # Main entry point for React app
│ ├── index.css # UPDATED: Global styles/Tailwind entry (for CSS vars)
│ └── app-sw.ts # NEW: Service worker related TypeScript file
│
├── reports/ # <<< Added for generated reports (Should be gitignored)
│ └── (empty or .gitkeep)
│
├── supabase/ # <<< Added by Supabase CLI
│ ├── config.toml # <<< Supabase project config (Track in Git)
│ └── migrations/ # <<< Canonical location for DB migrations (Track in Git)
│ ├── 000_create_base_tables.sql # <<< Assumes you created this for team_box_scores
│ ├── 001_create_team_rolling_20.sql # <<< Materialized view definition
│ ├── 20250429004757_remote_schema.sql # <<< Created by db pull (might be redundant/replaced later)
│ └── ... # Other migrations for new tables (mlb_snapshots, nba_snapshots)
│
├── README.md
├── remote_dump.sql # <<< Temporary data dump (Should be gitignored)
└── package.json # Root package file (if using workspaces/root scripts)

## API Documentation

ScoreGenius exposes a fully interactive, self‑documenting Swagger UI for all NFL endpoints (and future NBA/MLB routes).

URL: http://localhost:10000/api-docs

Explore request parameters, response schemas and try out calls directly in your browser.

This makes it easy for hiring managers and recruiters to explore and validate your backend API without writing any code.

# Getting Started

## Prerequisites

- Node.js 18+ and npm (or yarn)
- Python 3.11+ and pip
- Supabase CLI (`npm install -g supabase`)
- Access to a Supabase project
- API keys for data sources (API-Sports, Odds API, RapidAPI)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:themattbirch/score-genius.git
    cd score-genius
    ```

2.  **Setup backend:**
    ```bash
    cd backend
    python -m venv venv_pytorch
    # Activate the virtual environment
    # On macOS/Linux:
    source venv_pytorch/bin/activate
    # On Windows:
    # venv_pytorch\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Setup frontend:**

    ```bash
    cd ../frontend
    npm install
    ```

4.  **Initialize Supabase:**
    ```bash
    supabase login
    supabase link --project-ref <your-project-ref>
    supabase db reset
    ```

## Running Locally

1.  **Start Supabase:**

    ```bash
    supabase start
    ```

2.  **Start backend API:**

    ```bash
    cd backend/server
    npm install
    npm start
    ```

3.  **Start frontend dev server:**
    ```bash
    cd frontend
    npm run dev
    ```

## Running Scripts

Activate the backend virtual environment first:

```bash
# On macOS/Linux:
source backend/venv_pytorch/bin/activate
# On Windows:
# backend\venv_pytorch\Scripts\activate

# Example script execution:
python -m backend.nba_score_prediction.train_models \
  --data-source supabase \
  --lookback-days 90

  ## License

Distributed under the MIT License.

## Contact

Matt Birch – [matt@optimizewebsolutions.com](mailto:matt@optimizewebsolutions.com)
```
