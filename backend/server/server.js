// backend/server/server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createClient } from "@supabase/supabase-js";
import nbaRoutes from "./routes/nba_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";

// 1) Load environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

// 2) Initialize Supabase client
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// 3) Express app setup
const app = express();

// 3a) Normalize URLs (collapse multiple slashes)
app.use((req, _res, next) => {
  req.url = req.url.replace(/\/\/{2,}/g, "/");
  next();
});

app.use(cors({ origin: ["https://scoregenius.io"] }));
app.use(express.json());
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// 4) Serve PWA static assets and SPA fallback
const staticDir = path.join(__dirname, "static");

// 4a) Serve built assets under /app
app.use("/app", express.static(staticDir, { index: false }));

// 4b) SPA fallback: any GET to /app or /app/* returns app.html
app.get(/^\/app(\/.*)?$/, (_req, res) =>
  res.sendFile(path.join(staticDir, "app.html"))
);

// 5) API routes
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// 6) Health check endpoint
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// 7) 404 handler for other routes
app.use((req, res) => res.status(404).json({ error: "Not Found" }));

// 8) Global error handler
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// 9) Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
