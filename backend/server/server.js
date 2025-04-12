// backend/server/server.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import morgan from 'morgan'; // Optional: for HTTP request logging

// --- Load Environment Variables ---
// Loads variables from .env file into process.env
// Make sure to create a .env file for variables like PORT, SUPABASE_URL, SUPABASE_KEY
dotenv.config();

// --- Import Routers ---
// These files will define the specific endpoints for each sport
// We'll create basic versions of these next
import nbaRoutes from './routes/nba_routes.js';
import mlbRoutes from './routes/mlb_routes.js';

// --- Initialize Express App ---
const app = express();

// --- Middleware ---
// Enable Cross-Origin Resource Sharing for your PWA frontend
// Configure origins specifically in production for security
app.use(cors());

// Parse incoming JSON requests
app.use(express.json());

// HTTP request logger middleware (useful during development)
if (process.env.NODE_ENV !== 'production') {
  app.use(morgan('dev'));
}

// --- API Routes ---
// Mount the sport-specific routers under their base paths
app.use('/api/nba', nbaRoutes);
app.use('/api/mlb', mlbRoutes);

// --- Basic Health Check Route ---
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'UP', timestamp: new Date().toISOString() });
});

// --- Basic Root Route ---
app.get('/', (req, res) => {
  res.send('Score Genius API is running!');
});

// --- Error Handling Middleware ---
// Handle 404 Not Found errors
app.use((req, res, next) => {
  res.status(404).json({ message: 'Resource not found on this server.' });
});

// Generic error handler (catches errors from routes)
// Note: Add more specific error handling in production
app.use((err, req, res, next) => {
  console.error(err.stack); // Log error stack trace to console
  res.status(err.status || 500).json({
    message: err.message || 'An unexpected error occurred.',
    // Optionally include stack trace in development
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
  });
});

// --- Start Server ---
const PORT = process.env.PORT || 5001; // Use port from .env or default to 5001

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} in ${process.env.NODE_ENV || 'development'} mode.`);
  // In a real app, you might initialize DB connections here if needed globally
});

export default app; // Optional: export for testing frameworks