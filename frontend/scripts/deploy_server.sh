#!/bin/bash
# scripts/deploy_server.sh
cd backend
echo "Building Docker image for backend..."
docker build -t scoregenius-backend .
echo "Deploying backend container..."
docker run -d -p 8000:8000 scoregenius-backend
