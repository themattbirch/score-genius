# ScoreGenius (sportsgenius.io)

An AI-driven, live sports analytics Progressive Web App (PWA) providing real-time predictive analysis, natural language game recaps, and betting insights.

## 🚀 Features

- **Real-Time Analytics Dashboard**

  - Live win probability calculations
  - Momentum shift detection
  - Player performance projections

- **AI-Powered Game Analysis**

  - Natural language game recaps
  - Key play breakdowns
  - Post-game detailed analysis

- **Sports Betting Integration**
  - Live odds analysis
  - Predictive model insights
  - Historical trend analysis

## 🛠 Tech Stack

### Frontend

- React 18 with TypeScript
- Vite for fast bundling
- PWA capabilities with offline support
- Interactive charts (Recharts/Chart.js)

### Backend

- Python 3.13+
- FastAPI for REST endpoints
- Supabase for real-time data
- Machine Learning models for predictions
- LLM integration for narrative generation

### Data Sources

- API-Sports (Basketball/Football)
- Historical data processing
- Real-time data streaming

## 📁 Project Structure

```plaintext
score-genius/
├── backend/           # Python/FastAPI backend
│   ├── src/
│   │   └── scripts/  # Data fetching scripts
│   ├── analytics/    # ML models & analysis
│   ├── ml/          # LLM integration
│   └── utils/       # Helper functions
└── frontend/        # React/Vite frontend
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── services/
    └── public/
🚀 Getting Started
Prerequisites

Node.js 18+
Python 3.13+
API-Sports account

Installation

Clone the repository:

bashCopygit clone git@github.com:themattbirch/score-genius.git
cd score-genius

Setup Frontend:

bashCopycd frontend
npm install

Setup Backend:

bashCopycd backend
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

Configure environment variables:

bashCopycp .env.example .env
# Edit .env with your API keys and configurations
Running Locally

Start the backend:

bashCopycd backend
python app.py

Start the frontend:

bashCopycd frontend
npm run dev
🔒 Security

Protected main branch
Required PR reviews
CI/CD checks
Environment variable encryption

📝 License
This project is proprietary and confidential.
👥 Contributing
Currently accepting contributions from authorized team members only. Please contact the repository owner for access.
📫 Contact

Repository: themattbirch/score-genius
Email: [themattbirch@gmail.com]
```
