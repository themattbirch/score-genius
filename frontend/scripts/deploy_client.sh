#!/bin/bash
# scripts/deploy_client.sh
cd frontend
echo "Installing frontend dependencies..."
npm install
echo "Building the frontend..."
npm run build
echo "Deploying frontend to Vercel..."
vercel --prod
