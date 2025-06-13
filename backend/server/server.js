import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";
import { createProxyMiddleware } from "http-proxy-middleware";

// Resolve environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

// Validate Supabase credentials
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}

// Initialize Supabase client
export const supabase = createClient(
  SUPABASE_URL,
  SUPABASE_SERVICE_KEY,
  { auth: { persistSession: false } }
);

// Path to built frontend
const frontEndDist = path.resolve(__dirname, "../../frontend/dist");
console.log("🔍 Serving static assets from", frontEndDist);

// Create Express app
const app = express();

// 1) CORS and body parsing
app.use(
  cors({ origin: ["https://scoregenius.io", "http://localhost:5173", "http://localhost:4173"] })
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// Cache control headers
app.disable("etag");
app.use((req, res, next) => {
  res.set({
    "Cache-Control": "private, no-store, max-age=0, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
    "Surrogate-Control": "no-store",
  });
  next();
});

// 2) Proxy API calls to backend
app.use(
  "/api/v1",
  createProxyMiddleware({ target: "https://score-genius-backend.onrender.com", changeOrigin: true, logLevel: "warn" })
);

// 3) Mount API routes (bypassed by proxy)
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// 4) Serve static frontend assets
app.use(express.static(frontEndDist));

// 5) Serve index.html at root
app.get("/", (_req, res) => {
  res.sendFile(path.join(frontEndDist, "index.html"));
});

// 6) Serve app.html at /app and /app/*
app.get(["/app", "/app/*"], (_req, res) => {
  res.sendFile(path.join(frontEndDist, "app.html"));
});

// 7) SPA fallback for other GET requests (excluding API and assets)
app.use((req, res, next) => {
  if (req.method !== 'GET') return next();
  if (req.path.startsWith('/api/v1')) return next();
  if (req.path.includes('.')) return next(); // Skip asset requests
  res.sendFile(path.join(frontEndDist, "index.html"));
});

// Health check
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// Error handling
app.use((err, _req, res, _next) => {
  console.error(err.stack || err);
  res.status(err.status || 500).json({ error: { message: err.message } });
});

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));