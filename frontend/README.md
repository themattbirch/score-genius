# [ScoreGenius](https://sportsgenius.io)

ScoreGenius is an AI-driven, live sports analytics Progressive Web App (PWA) that delivers real-time predictive analysis, natural language game recaps, and actionable betting insights for NBA games. Leveraging state-of-the-art machine learning techniques, adaptive ensemble modeling, and robust feature engineering, ScoreGenius dynamically updates in-game predictions as a match unfolds.

## 🚀 Key Features

### Real-Time Predictive Analytics
- Live win probability, momentum shifts, and dynamic score predictions.
- Quarter-specific models integrated into an adaptive ensemble framework.
- Uncertainty estimation and confidence intervals for every prediction.

### AI-Powered Game Recaps & Analysis
- Automated, natural language game recaps powered by LLM integration.
- Post-game detailed analysis with interactive trend visualizations.

### Sports Betting & Insights
- Real-time odds analysis fused with model predictions for betting insights.
- Historical trend analysis and market comparisons for identifying value.
- Automated betting recommendations based on prediction confidence.

### Robust Data Pipeline & Monitoring
- Automated ingestion of historical and live game data via APIs and Supabase.
- Efficient caching, rate-limiting, and anomaly detection for reliable data flow.
- Comprehensive logging and performance metrics tracking across all components.

## 🛠 Architecture & System Flow

The system is designed with modularity and scalability in mind, structured into the following key layers:

### Data Ingestion & Caching
- Historical and live game data is ingested through dedicated scripts.
- Data is cached using `Supabase` and `Redis` for fast retrieval and processing.
- Shared utilities ensure duplicate columns are removed and recency weights computed consistently.

### Feature Engineering
- A centralized module generates over 100 refined features including rolling averages, season context, and advanced metrics.
- Utilities ensure robust handling of missing data and duplicate columns.
- The design enables reusability across training, tuning, and real-time prediction.

### Model Training & Ensemble Prediction
- Multiple lightweight models (`XGBoost`, `RandomForest`, `Ridge`) are trained individually.
- An adaptive ensemble framework dynamically adjusts weights based on error history and game context (time, score differential, momentum).
- Hyperparameter tuning is automated using `RandomizedSearchCV` with `TimeSeriesSplit`.
- Model saving includes comprehensive metadata for versioning and reproducibility.

### Prediction & Calibration
- Pre-game predictions are generated on-the-fly using a dedicated prediction module.
- Predictions are calibrated with betting odds when available.
- The system logs both raw and calibrated predictions along with uncertainty intervals.
- Modular functions ensure that data ingestion, feature generation, and prediction are decoupled for easier testing and maintenance.

### Visualization & Reporting
- Detailed performance plots and dashboards display model performance, prediction evolution, and betting metrics.
- Centralized reporting utilities aggregate results and highlight any anomalies.

## 🛠 Tech Stack

### Frontend
- `React 18` with `TypeScript`
- `Vite` for rapid development and bundling
- `PWA` with offline support
- Interactive data visualizations (`Recharts`, `Chart.js`)

### Backend
- `Python 3.13+`
- `FastAPI` for REST endpoints
- `Supabase` for real-time data storage and caching

### Machine Learning Models
- Adaptive ensemble weighting with quarter-specific models
- Robust feature engineering and dynamic uncertainty estimation
- Centralized Logging & Configuration for consistent system monitoring

### LLM Integration
- For narrative game recaps

### Data Sources
- Real-time sports APIs (e.g., `API-Sports`)
- Historical game and team stats data ingestion
- Real-time game data streaming

## 📁 Project Structure

```text
score-genius/
├── backend/
│   ├── caching/                 # Supabase and Redis caching utilities
│   ├── models/                  # Model definitions, training, ensemble logic, and simulation
│   │   ├── feature_engineering.py   # Unified feature engineering module
│   │   ├── models.py                # Model definitions (XGBoost, RandomForest, Ridge) and pipelines
│   │   ├── ensemble.py              # Adaptive ensemble weighting and uncertainty estimation
│   │   ├── simulation.py            # Custom loss functions and simulation utilities
│   │   └── train_models.py          # End-to-end model tuning, training, and evaluation pipeline
│   ├── prediction.py            # On-the-fly prediction module with calibration and reporting
│   ├── routers/                 # FastAPI routes for data, predictions, and recaps
│   ├── venv/                    # Python virtual environment
│   └── config.py                # Centralized configuration and environment variables
└── frontend/
    ├── src/
    │   ├── components/          # Reusable UI components
    │   ├── pages/               # Dashboard, recaps, and live updates
    │   └── services/            # API service calls and state management
    └── public/

    ## Getting Started

### Prerequisites
- `Node.js 18+`
- `Python 3.13+`
- `API-Sports` (or equivalent) credentials
- Required environment variables (see `.env.example`)

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:themattbirch/score-genius.git
    cd score-genius
    ```
2.  **Setup Frontend:**
    ```bash
    cd frontend
    npm install
    ```
3.  **Setup Backend:**
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # Use .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables:**
    ```bash
    cp .env.example .env
    # Edit .env with your API keys, database URLs, and configurations
    ```

## Running Locally

### Start the Backend:
```bash
cd backend
uvicorn main:app --reload

# License
ScoreGenius is open-source and distributed under the MIT License.

# Contact
For questions or contributions, please contact Matt Birch at matt@optimizewebsolutions.com.