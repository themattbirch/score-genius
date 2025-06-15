// backend/server/server.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// --- Load environment variables ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
["../.env", "../../.env"].forEach((rel) => {
  const p = path.join(__dirname, rel);
  if (fs.existsSync(p)) dotenv.config({ path: p, override: true });
});

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";
import nbaRoutes from "./routes/nba_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";

// Validate Supabase keys
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// --- Static paths ---
const staticRoot = path.join(__dirname, "static");
const marketingDir = path.join(staticRoot, "public");
const mediaDir = path.join(staticRoot, "media");
const assetsDir = path.join(staticRoot, "assets");

const app = express();

// CORS & JSON
app.use(
  cors({
    origin: [
      "https://scoregenius.io",
      "http://localhost:10000",
      "http://localhost:5173",
    ],
  })
);
app.use(express.json());

// Normalize URLs and log
app.use((req, res, next) => {
  req.url = req.url.replace(/\/\/+/g, "/");
  console.log(`${new Date().toISOString()} – ${req.method} ${req.url}`);
  next();
});

// Serve all marketing HTML (index.html + public/*.html)
app.use(
  express.static(marketingDir, {
    index: "index.html",
    extensions: ["html"],
    maxAge: "1d",
  })
);

// Explicit “pretty” routes for standalone pages
["404", "disclaimer", "documentation", "privacy", "support", "terms"].forEach(
  (page) => {
    app.get(`/${page}`, (_req, res) => {
      const file = path.join(marketingDir, `${page}.html`);
      return fs.existsSync(file)
        ? res.sendFile(file)
        : res.status(404).send("Not Found");
    });
  }
);

// Static assets
app.use("/media", express.static(mediaDir, { maxAge: "1d" }));
app.use("/assets", express.static(assetsDir, { maxAge: "1d" }));

// PWA service worker
app.get("/sw.js", (_req, res) =>
  res.sendFile(path.join(staticRoot, "app-sw.js"))
);

// SPA shell under /app
app.use("/app", express.static(staticRoot, { index: false }));
app.get(/^\/app(\/.*)?$/, (_req, res) =>
  res.sendFile(path.join(staticRoot, "app.html"))
);

// API routes
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// Health check
app.get("/health", (_req, res) =>
  res.json({ status: "OK", timestamp: new Date().toISOString() })
);

// Fallback 404 handler
app.use((req, res) => {
  const file404 = path.join(marketingDir, "404.html");
  if (fs.existsSync(file404)) {
    return res.status(404).sendFile(file404);
  }
  res.status(404).json({ error: "Not Found" });
});

// Error handler
app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// Start
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
