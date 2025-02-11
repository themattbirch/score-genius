#!/bin/bash
# scripts/run_local.sh
# Run backend and frontend concurrently for local development

# Start the backend (FastAPI with uvicorn)
echo "Starting backend..."
cd backend
uvicorn app:app --reload &
BACKEND_PID=$!

# Start the frontend (Vite development server)
cd ../frontend
echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!

echo "Backend running with PID $BACKEND_PID"
echo "Frontend running with PID $FRONTEND_PID"
wait
