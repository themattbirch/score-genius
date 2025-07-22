// backend/server/server.js
// ScoreGenius backend entrypoint (Express)
// -------------------------------------------------------------
// Dependency‑free favicon support added:
//   • Detects backend/server/public/favicon.ico at startup.
//   • If present, serves /favicon.ico with long‑cache headers **before** logging middleware.
//   • Prevents noisy 404s + Workbox bad‑precaching-response errors.
// -------------------------------------------------------------

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { setupSwagger } from "./docs/swagger.js";

// --- Load environment variables -------------------------------------------------
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
import nflRoutes from "./routes/nfl_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";
import weatherRoutes from "./routes/weather_routes.js";

// -------------------------------------------------------------------------------
// Root favicon support (dependency‑free)
// -------------------------------------------------------------------------------
// You copied your favicon into backend/server/public/favicon.ico.
// We'll serve it explicitly at /favicon.ico so browsers stop generating 404s.
// We register this route BEFORE the logger to cut down on console noise.
const serverPublicDir = path.join(__dirname, "public");
const faviconPath = path.join(serverPublicDir, "favicon.ico");

// --- Validate Supabase keys -----------------------------------------------------
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// --- Static paths to built marketing / app bundle -------------------------------
const staticRoot = path.join(__dirname, "static");
const marketingDir = path.join(staticRoot, "public");
const mediaDir = path.join(staticRoot, "media");
const assetsDir = path.join(staticRoot, "assets");

const app = express();

// -------------------------------------------------------------------------------
// CORS & JSON parsing
// -------------------------------------------------------------------------------
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

// Prevent CDN or browser caching of dynamic API responses
app.use("/api/v1", (_req, res, next) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  next();
});

// -------------------------------------------------------------------------------
// Favicon route (register BEFORE logger to reduce log spam)
// -------------------------------------------------------------------------------
if (fs.existsSync(faviconPath)) {
  app.get("/favicon.ico", (req, res) => {
    // Aggressive cache; bump filename/hash if icon changes
    res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
    res.sendFile(faviconPath);
  });
}

// -------------------------------------------------------------------------------
// Normalize double slashes & request logging
// -------------------------------------------------------------------------------
app.use((req, res, next) => {
  req.url = req.url.replace(/\/\/+/g, "/");
  console.log(`${new Date().toISOString()} – ${req.method} ${req.url}`);
  next();
});

// -------------------------------------------------------------------------------
// Marketing site root (also used by Workbox precache for fallback)
// -------------------------------------------------------------------------------
// Workbox precache entry: /public/index.html
app.get("/public/index.html", (_req, res) =>
  res.sendFile(path.join(staticRoot, "public", "index.html"))
);

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

// Serve only the .well-known folder and allow dotfiles
app.use(
  "/.well-known",
  express.static(path.join(__dirname, "static/public/.well-known"), {
    dotfiles: "allow",
    maxAge: "1h",
  })
);

// Static assets
app.use("/media", express.static(mediaDir, { maxAge: "1d" }));
app.use("/assets", express.static(assetsDir, { maxAge: "1d" }));
app.use(
  "/images",
  express.static(path.join(staticRoot, "images"), { maxAge: "7d" })
);

// root-level app shell (needed by SW precache)
app.get("/app.html", (_req, res) =>
  res.sendFile(path.join(staticRoot, "app.html"))
);

// --- PWA assets at the site-root -------------------------------------------
app.get("/manifest.webmanifest", (_req, res) =>
  res.sendFile(path.join(staticRoot, "manifest.webmanifest"))
);
app.use(
  "/icons",
  express.static(path.join(staticRoot, "icons"), { maxAge: "7d" })
);
// ---------------------------------------------------------------------------

// service-worker bundle generated by VitePWA (root level)
app.get("/app-sw.js", (_req, res) =>
  res.sendFile(path.join(staticRoot, "app-sw.js"))
);

// Back-compat: /sw.js -> /app-sw.js
app.get("/sw.js", (_req, res) =>
  res.sendFile(path.join(staticRoot, "app-sw.js"))
);

// SPA shell under /app
app.use("/app", express.static(staticRoot, { index: false }));
app.get(/^\/app(\/.*)?$/, (_req, res) =>
  res.sendFile(path.join(staticRoot, "app.html"))
);

// -------------------------------------------------------------------------------
// API routes
// -------------------------------------------------------------------------------
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);
app.use("/api/v1/nfl", nflRoutes);
app.use("/api/weather", weatherRoutes);

// Serve API docs
setupSwagger(app);

// alias /docs → /api-docs
app.get("/docs", (_req, res) => res.redirect("/api-docs"));

// -------------------------------------------------------------------------------
// Health check
// -------------------------------------------------------------------------------
app.get("/health", (_req, res) =>
  res.json({ status: "OK", timestamp: new Date().toISOString() })
);

// -------------------------------------------------------------------------------
// Fallback 404 handler (after all routes)
// -------------------------------------------------------------------------------
app.use((req, res) => {
  const file404 = path.join(marketingDir, "404.html");
  if (fs.existsSync(file404)) {
    return res.status(404).sendFile(file404);
  }
  res.status(404).json({ error: "Not Found" });
});

// -------------------------------------------------------------------------------
// Error handler
// -------------------------------------------------------------------------------
app.use((err, req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// -------------------------------------------------------------------------------
// Start
// -------------------------------------------------------------------------------
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
