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

const serverPublicDir = path.join(__dirname, "public");
const faviconPath = path.join(serverPublicDir, "favicon.ico");

const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

const staticRoot = path.join(__dirname, "static");
const app = express();

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

app.use("/api/v1", (_req, res, next) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  next();
});

if (fs.existsSync(faviconPath)) {
  app.get("/favicon.ico", (req, res) => {
    res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
    res.sendFile(faviconPath);
  });
}

app.use((req, res, next) => {
  req.url = req.url.replace(/\/\/+/g, "/");
  console.log(`${new Date().toISOString()} – ${req.method} ${req.url}`);
  next();
});

// =================== ✨ SIMPLIFIED ROUTING ===================

// 1. SERVE ALL Static Files. And Force fresh fetches for SW and offline HTML. And Marketing Pages.
// This single line correctly serves EVERYTHING from your 'static' folder.
// It will handle /assets/*, /icons/*, and most importantly, /app/app-sw.js.
app.get("/app/app-sw.js", (req, res) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  res.sendFile(path.join(staticRoot, "app", "app-sw.js"));
});

app.get("/app/offline.html", (req, res) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  res.sendFile(path.join(staticRoot, "app", "offline.html"));
});

app.get("/support", (req, res) => {
  res.sendFile(path.join(staticRoot, "public", "support.html"));
});

app.get("/privacy", (req, res) => {
  res.sendFile(path.join(staticRoot, "public", "privacy.html"));
});

app.get("/", (req, res) => {
  res.sendFile(path.join(staticRoot, "public", "index.html"));
});

app.use(
  express.static(path.join(staticRoot, "public"), {
    extensions: ["html"],
    index: false,
  })
);

// 1a) .well‑known
app.use(
  "/.well-known",
  express.static(path.join(staticRoot, "public", ".well-known"), {
    dotfiles: "allow",
  })
);

app.use(express.static(staticRoot));

// 2. ✅ SERVE API ROUTES
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);
app.use("/api/v1/nfl", nflRoutes);
app.use("/api/weather", weatherRoutes);
setupSwagger(app);
app.get("/docs", (_req, res) => res.redirect("/api-docs"));
app.get("/health", (_req, res) =>
  res.json({ status: "OK", timestamp: new Date().toISOString() })
);

// 3. ✅ SPA FALLBACK (for the app)
// This specifically catches any route under /app and serves the app shell.
// It uses a Regular Expression to avoid parsing errors.
app.get(/^\/app(\/.*)?$/, (req, res) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.sendFile(path.join(staticRoot, "app.html"));
});

// 4. ✅ FINAL 404 & ERROR HANDLERS
// These will now correctly catch any request that doesn't match the above routes.
app.use((req, res, _next) => {
  const file404 = path.join(staticRoot, "public", "404.html");
  if (fs.existsSync(file404)) {
    return res.status(404).sendFile(file404);
  }
  res.status(404).json({ error: "Not Found" });
});

app.use((err, req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// Start the server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
