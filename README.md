# ScoreGenius

ScoreGenius is a cross-platform sports analytics Progressive Web App (PWA) that surfaces game insights and predictive signals for NFL, NBA, and MLB. The system combines robust data ingestion, modular feature engineering, and lightweight model ensembles with a fast, installable UI.

---

## Table of Contents

1. [Overview](#overview)
2. [System Capabilities](#system-capabilities)
3. [Architecture & Flow](#architecture--flow)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Install & Run (Frontend)](#install--run-frontend)
   - [Optional: Local Backend](#optional-local-backend)
7. [Security & What’s Omitted](#security--whats-omitted)
8. [License](#license)
9. [Contact](#contact)

---

## Overview

- Installable PWA with fast navigation and offline shell
- Predictions + game insights from a modular analytics backend
- Snapshot generation for low-latency UI rendering
- Designed to scale across leagues (NFL, NBA, MLB)

> This public repository focuses on the **frontend and integration scaffolding**. Proprietary pipelines, model internals, and sensitive infra live in a private core.

---

## System Capabilities

- Multi-league support (NFL, NBA, MLB)
- Modular feature pipelines (rolling form, rest/schedule, matchup context, advanced metrics)
- Materialized-view backed analytics for efficient querying
- Ensemble modeling with regression-based predictors
- Snapshot generation per game (cards, charts, headline stats)
- Typed PWA (React + Vite + Tailwind), mobile-first
- CI-/cron-driven data refresh and preview ingest (private)

---

## Architecture & Flow

```mermaid
flowchart LR
    A[Data Providers] -->|ETL jobs| B[(Supabase/Postgres)]
    B --> C[Feature Pipeline<br/>(modular)]
    C --> D[Model Ensembles<br/>(regression-based)]
    D --> E[Predictions & Snapshots]
    E --> F[API Layer]
    F --> G[PWA (React/Vite)]
```

**Data & Storage:** Supabase/Postgres hosts historical and preview data.

**Features:** Modular Python/SQL pipelines compute rolling stats, form, matchup context, etc.

**Models:** Lightweight ensembles (e.g., ridge, SVR, tree-based) produce totals/margins and related signals.

**Delivery:** API serves snapshots & summaries consumed by the PWA.

> Diagram is intentionally high-level. Exact schemas, features, and model internals are private.

---

## Tech Stack

### Frontend

- React 18 + TypeScript, Vite, Tailwind CSS
- Client state/querying (e.g., React Query)
- Charts (Recharts, etc.)
- PWA (installable, offline shell)

### Backend & Data

- Node.js/Express API layer
- Supabase/Postgres (historical + preview data)
- Python 3.11+ for pipelines and training

### Modeling

- Regression-based predictors (e.g., Ridge, SVR, XGBoost/forest variants)
- Time-aware evaluation (e.g., TimeSeriesSplit)
- Snapshot emitters for UI

---

## Project Structure

```

score-genius/
├─ .gitignore # Protects secrets, models, datasets, builds
├─ backend/ # API + pipelines (private-first; ignored in public)
│ ├─ server/ # Node/Express service (entrypoint)
│ ├─ data_pipeline/ # Ingest/preview jobs
│ ├─ features/ # Modular feature engineering
│ └─ tests/ # Python tests
├─ frontend/ # React + Vite PWA
│ ├─ public/ # Icons, manifest, static
│ └─ src/ # Components, screens, hooks, contexts, api
├─ reports/ # Generated analysis (ignored)
├─ README.md # You are here
└─ package.json # Root scripts/workspaces (if used)

```

> In this public repo, sensitive subtrees (backend internals, models, migrations) are either private or git-ignored. The structure above reflects how the system is organized without exposing IP.

---

## Getting Started

### Prerequisites

- **Node.js:** v20.14.0 (nvm use reads from .nvmrc)
- **npm** or **pnpm**
- **Optional** (backend/local pipelines): Python 3.11+

You can run the frontend against any compatible API base URL via environment config.

### Install & Run (Frontend)

```bash
# from repo root
cd frontend
nvm use
npm ci
npm run dev
```

Create a minimal `.env` (or pass at build time) if you need to point to a specific API:

```bash
# frontend/.env
VITE_API_BASE_URL=https://your-api.example.com
```

Build for production:

```bash
npm run build
```

### Optional: Local Backend

If you have access to the private core (or a local API):

```bash
# Node/Express API
cd backend/server
nvm use
npm ci
npm start   # ensure PORT is respected by the server (defaults often 10000)
```

For Python pipelines/training (when present locally):

```bash
python -m venv backend/venv_pytorch
source backend/venv_pytorch/bin/activate
pip install -r backend/requirements.txt

# examples
python -m backend.data_pipeline.nfl_games_preview
python -m backend.nfl_score_prediction.train_models
```

---

## Security & What's Omitted

To protect commercial IP and operational security, the public repo intentionally excludes:

- Production feature sets and transformations, exact model weights, hyperparameters, and ensemble logic
- Full database schema, migrations, and RPCs
- CI/CD internals and cron schedules tied to production data
- Certificates/keystores, environment secrets, dumps, and build artifacts

The public repo is suitable for:

- Inspecting the frontend implementation and integration boundaries
- Understanding the system architecture and capabilities at a high level
- Running the PWA against a compatible API base

---

## License

This repository is distributed under the **Business Source License 1.1 (BUSL-1.1)**.

- You may view, clone, and modify the code for non-production and non-commercial use.
- Production/commercial use requires a separate commercial license from the author.
- After the change date specified in LICENSE, this code will be re-licensed under the GPL v2.0 or later.

See the LICENSE file for details.

---

## Contact

**Matt Birch** — matt@optimizewebsolutions.com
