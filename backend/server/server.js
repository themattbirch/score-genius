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
// 1) Resolve paths & load environment
// ──────────────────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑  Loaded env from ${envPath}`);
} else {
  console.log("🔑  No local .env file found; relying on host‐provided vars");
}

// location of built frontend (Vite copies public/* → dist/public)
const frontEndDistPublic = path.resolve(
  __dirname,
  "../../frontend/dist/public"
);
console.log("🔍 [DEBUG] frontEndDistPublic =", frontEndDistPublic);
console.log(
  "🔍 [DEBUG] index.html exists?",
  fs.existsSync(path.join(frontEndDistPublic, "index.html"))
);

// ──────────────────────────────────────────────────────────────
// 2) Validate Supabase env vars & create client
// ──────────────────────────────────────────────────────────────
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// ──────────────────────────────────────────────────────────────
// 3) Instantiate Express
// ──────────────────────────────────────────────────────────────
const app = express();

// ──────────────────────────────────────────────────────────────
// 4) Proxy ALL /api/v1 calls to Render backend
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
// 5) Serve static assets from built frontend
// ──────────────────────────────────────────────────────────────
app.use(express.static(frontEndDistPublic));

// ──────────────────────────────────────────────────────────────
// 6) API route mounts (bypassed when proxy matches first)
// ──────────────────────────────────────────────────────────────
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// ──────────────────────────────────────────────────────────────
// 7) Standard middleware (CORS, JSON, logging)
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
app.use(express.json());
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// ──────────────────────────────────────────────────────────────
// 8) SPA fallback – any GET not handled above returns index.html
// ──────────────────────────────────────────────────────────────
app.get("/*", (req, res) => {
  // Let proxy handle API routes
  if (req.path.startsWith("/api/v1")) return res.status(404).end();
  res.sendFile(path.join(frontEndDistPublic, "index.html")); // or app.html if that’s your entry
});

// ──────────────────────────────────────────────────────────────
// 9) Health check
// ──────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// ──────────────────────────────────────────────────────────────
// 10) Error handling
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
// 11) Start server
// ──────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
