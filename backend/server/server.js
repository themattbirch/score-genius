// backend/server/server.js
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";
import { createProxyMiddleware } from "http-proxy-middleware";

// ──────────────────────────────────────────────────────────────
// 1) Determine file paths and load environment
// ──────────────────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const envPath = path.join(__dirname, "..", ".env");
const frontEndDist = path.resolve(__dirname, "../../frontend/dist");

console.log("🔍 [DEBUG] __dirname =", __dirname);
console.log("🔍 [DEBUG] frontEndDist resolves to =", frontEndDist);
console.log(
  `🔍 [DEBUG] index.html exists?`,
  fs.existsSync(path.join(frontEndDist, "index.html"))
);
console.log(
  `🔍 [DEBUG] app.html exists?`,
  fs.existsSync(path.join(frontEndDist, "app.html"))
);

if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑  Loaded env from ${envPath}`);
} else {
  console.log("🔑  No local .env file found; assuming host provides vars");
}

// ──────────────────────────────────────────────────────────────
// 2) Validate required Supabase environment variables
// ──────────────────────────────────────────────────────────────
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error(
    "FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing in environment"
  );
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────
// 3) Initialize Supabase client
// ──────────────────────────────────────────────────────────────
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// ──────────────────────────────────────────────────────────────
// 4) Instantiate Express
// ──────────────────────────────────────────────────────────────
const app = express();

// ──────────────────────────────────────────────────────────────
// 5) Proxy all /api/v1 requests to the backend service
// ──────────────────────────────────────────────────────────────
app.use(
  "/api/v1",
  createProxyMiddleware({
    target: "https://score-genius-backend.onrender.com",
    changeOrigin: true,
    pathRewrite: { "^/api/v1": "/api/v1" },
    logLevel: "warn",
  })
);

// ──────────────────────────────────────────────────────────────
// 6) Standard middleware
// ──────────────────────────────────────────────────────────────
app.disable("etag");
app.use(
  cors({
    origin: [
      "https://scoregenius.io",
      "http://localhost:5173",
      "http://localhost:4173",
    ],
  })
);
app.use((req, res, next) => {
  res.set({
    "Cache-Control": "private, no-store, max-age=0, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
    "Surrogate-Control": "no-store",
  });
  next();
});
app.use(express.json());
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// ──────────────────────────────────────────────────────────────
// 7) Mount API routes (bypassed by the proxy)
// ──────────────────────────────────────────────────────────────
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// ──────────────────────────────────────────────────────────────
// 8) Health check endpoint
// ──────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// ──────────────────────────────────────────────────────────────
// 9) Error handling
// ──────────────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  console.error(err.stack || err);
  res.status(err.status || 500).json({
    error: {
      message: err.message ?? "Internal Server Error",
      stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    },
  });
});

// ──────────────────────────────────────────────────────────────
// 10) Start server
// ──────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
