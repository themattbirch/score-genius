// backend/server/server.js
 // Simple way to load .env - ensure it's found or configure path below

// --- OR Explicit path loading ---
// import dotenv from 'dotenv';
// import path from 'path';
// import { fileURLToPath } from 'url';
// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);
// dotenv.config({ path: path.resolve(__dirname, '../../.env') }); // Load .env from project root

import express from 'express';
import cors from 'cors';
import 'dotenv/config';

// --- Route Imports (using import, add .js extension) ---
// import nbaRoutes from './routes/nbaRoutes.js';
import mlbRoutes from './routes/mlb_routes.js';// --- End Route Imports ---

const app = express();
const PORT = process.env.PORT || 3001;

// --- Middleware ---
app.use(cors());
app.use(express.json());
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});
// --- End Middleware ---

// --- API Routes ---
// app.use('/api/v1/nba', nbaRoutes);
app.use('/api/v1/mlb', mlbRoutes); // <-- UNCOMMENT/ADD THIS

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});
// --- End API Routes ---

// --- Basic Error Handling ---
app.use((err, req, res, next) => {
  console.error("Error:", err.stack || err.message || err);
  const status = err.status || 500;
  const message = err.message || 'Internal Server Error';
  res.status(status).json({
    error: {
      message: message,
      // Only show stack in development
      stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
    },
  });
});
// --- End Error Handling ---

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});