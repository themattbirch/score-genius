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

// ─────────────── 1) Load environment ────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

// ─────────────── 2) Supabase client ────────────────
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// ─────────────── 3) Express setup ────────────────
const app = express();
// Normalize URLs (collapse multiple slashes)
app.use((req, _res, next) => {
  req.url = req.url.replace(/\/{2,}/g, "/");
  next();
});
app.use(cors({ origin: ["https://scoregenius.io", "http://localhost:10000"] }));
app.use(express.json());
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// ─────────────── 4) Directories ────────────────
const staticDir = path.join(__dirname, "static"); // Vite output root
const marketingDir = path.join(staticDir, "public"); // index.html + marketing assets
const assetsDir = path.join(staticDir, "assets"); // hashed js/css

// ─────────────── 5) Marketing site at root ───────────────
// Serve index.html for '/'
app.get("/", (_req, res) => {
  res.sendFile(path.join(marketingDir, "index.html"));
});
// Serve other marketing assets (images, css, manifest...)
app.use(express.static(marketingDir, { index: false, redirect: false }));

// ─────────────── 6) PWA assets & shell ───────────────
// Chunks available at /assets and /app/assets
app.use(
  "/assets",
  express.static(assetsDir, { index: false, redirect: false })
);
app.use(
  "/app/assets",
  express.static(assetsDir, { index: false, redirect: false })
);
// Service worker
app.get("/sw.js", (_req, res) => {
  res.sendFile(path.join(staticDir, "app-sw.js"));
});
// SPA shell under /app
app.use("/app", express.static(staticDir, { index: false, redirect: false }));
app.get(/^\/app(\/.*)?$/, (_req, res) => {
  res.sendFile(path.join(staticDir, "app.html"));
});

// ─────────────── 7) API endpoints ────────────────
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// ─────────────── 8) Health check ────────────────
app.get("/health", (_req, res) => {
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() });
});

// ─────────────── 9) 404 + error handlers ────────────────
app.use((req, res) => res.status(404).json({ error: "Not Found" }));
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// ─────────────── 10) Start server ────────────────
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
