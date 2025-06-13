import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";
import { createProxyMiddleware } from "http-proxy-middleware";

// Resolve environment
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
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// Path to your built frontend
const frontEndDist = path.resolve(__dirname, "../../frontend/dist");
console.log("🔍 Serving static assets from", frontEndDist);

// Create Express app
const app = express();

// Proxy middleware for API calls
app.use(
  "/api/v1",
  createProxyMiddleware({
    target: "https://score-genius-backend.onrender.com",
    changeOrigin: true,
    pathRewrite: { "^/api/v1": "/api/v1" },
    logLevel: "warn",
  })
);

// Serve static frontend assets
app.use(express.static(frontEndDist));

// Mount API routes (bypassed by proxy)
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// Optional CORS and JSON parsing
app.use(
  cors({
    origin: [
      "https://scoregenius.io",
      "http://localhost:5173",
      "http://localhost:4173",
    ],
  })
);
app.use(express.json());

// SPA fallback: serve app.html for all non-API GET requests
app.get("*", (req, res, next) => {
  if (req.path.startsWith("/api/v1")) return next();
  res.sendFile(path.join(frontEndDist, "app.html"));
});

// Health check
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// Error handler
app.use((err, _req, res, _next) => {
  console.error(err.stack || err);
  res.status(err.status || 500).json({ error: { message: err.message } });
});

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
