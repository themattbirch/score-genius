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

````text
score-genius/
├── backend/
│   ├── caching/
│   ├── data_pipeline/
│   ├── features/
│   ├── nba_score_prediction/
│   ├── server/
│   └── venv_pytorch/
├── frontend/
├── supabase/
│   └── migrations/
├── reports/
├── README.md
└── package.json

# Getting Started

## Prerequisites

-   Node.js 18+ and npm (or yarn)
-   Python 3.11+ and pip
-   Supabase CLI (`npm install -g supabase`)
-   Access to a Supabase project
-   API keys for data sources (API-Sports, Odds API, RapidAPI)

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
4.  **Setup frontend:**
    ```bash
    cd ../frontend
    npm install
    ```

5.  **Initialize Supabase:**
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
````
