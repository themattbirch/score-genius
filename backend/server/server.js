// backend/server/server.js
// ScoreGenius backend entrypoint (Express)

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";

import { setupSwagger } from "./docs/swagger.js";
import nbaRoutes from "./routes/nba_routes.js";
import nflRoutes from "./routes/nfl_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";
import weatherRoutes from "./routes/weather_routes.js";

// ─── Configuration & Initialization ───────────────────────────────────────────

// Resolve __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define key static asset paths
const staticRoot = path.join(__dirname, "static");

// Load .env files from parent directories
["../.env", "../../.env"].forEach((rel) => {
  const p = path.join(__dirname, rel);
  if (fs.existsSync(p)) dotenv.config({ path: p, override: true });
});

// Initialize Supabase client
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY is not defined.");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// Initialize Express app
const app = express();

// ─── Core Middleware ──────────────────────────────────────────────────────────

// Request logger
app.use((req, res, next) => {
  req.url = req.url.replace(/\/\/+/g, "/");
  console.log(`${new Date().toISOString()} – ${req.method} ${req.url}`);
  next();
});

// Sanity log once at startup
const dalPath = path.join(
  staticRoot,
  "public",
  ".well-known",
  "assetlinks.json"
);

// 1) Load once at startup; if it fails, use inline JSON as a fallback
let dalBody = null;
try {
  dalBody = fs.readFileSync(dalPath, "utf8");
  console.log("[DAL] Loaded from disk:", dalPath, "bytes:", dalBody.length);
} catch (e) {
  console.log("[DAL] Disk load failed, using inline fallback:", e?.message);
  const fallback = [
    {
      relation: ["delegate_permission/common.handle_all_urls"],
      target: {
        namespace: "android_app",
        package_name: "io.scoregenius.app",
        sha256_cert_fingerprints: [
          "58:E7:9D:88:2B:A6:5D:DE:F6:3B:7B:4D:09:DC:B4:80:81:E2:4F:38:7F:6D:77:93:8B:91:46:1D:23:D4:94:CA",
          "5A:A0:9B:BD:E6:F4:04:F2:28:CF:75:6D:8B:64:0A:16:D3:4E:ED:8F:9E:27:7D:7E:8E:7D:35:C5:AE:D7:51:E3",
        ],
      },
    },
  ];
  dalBody = JSON.stringify(fallback);
}

// 2) Respond from memory — no filesystem at request time
app.get("/.well-known/assetlinks.json", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=86400, immutable");
  res.type("application/json");
  if (typeof res.removeHeader === "function") res.removeHeader("Vary");
  res.status(200).send(dalBody);
});

app.use(express.static(staticRoot));

// Serve the SPA shell for /app and /app/ with 200 (no redirect)
app.get(["/app", "/app/"], (req, res) => {
  res.setHeader("Cache-Control", "no-cache");
  res.sendFile(path.join(staticRoot, "app.html"));
});

// Static assets under /app/* without directory redirecting or index serving
app.use(
  "/app",
  express.static(path.join(staticRoot, "app"), {
    index: false,
    redirect: false,
    setHeaders: (res, filePath) => {
      if (filePath.endsWith(".appinstaller")) {
        res.setHeader("Content-Type", "application/appinstaller");
      } else if (
        filePath.endsWith(".appxbundle") ||
        filePath.endsWith(".msixbundle")
      ) {
        res.setHeader("Content-Type", "application/vnd.ms-appx");
      }
    },
  })
);

// CORS policy, JSON parsing
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

// Set no-cache headers for all API routes
app.use("/api/v1", (_req, res, next) => {
  res.setHeader(
    "Cache-Control",
    "no-store, no-cache, must-revalidate, proxy-revalidate"
  );
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  next();
});

// ─── Route Handlers ───────────────────────────────────────────────────────────

// 1. Explicit routes for all static marketing pages (HIGHEST PRIORITY)
app.get("/", (req, res) => {
  res.sendFile(path.join(staticRoot, "public", "index.html"));
});

app.get("/robots.txt", (req, res) => {
  res.setHeader("Content-Type", "text/plain");
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "robots.txt"));
});

// right after your robots.txt handler:
app.get("/favicon.ico", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "favicon.ico"));
});

// Add the missing handler for /support
app.get("/support", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "support.html"));
});

app.get("/privacy", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "privacy.html"));
});

app.get("/documentation", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "documentation.html"));
});

app.get("/disclaimer", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "disclaimer.html"));
});

app.get("/terms", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "terms.html"));
});

app.get("/about", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "public", "about.html"));
});

// 2. PWA-specific assets
app.get("/app/app-sw.js", (req, res) => {
  res.setHeader("Cache-Control", "no-cache");
  res.sendFile(path.join(staticRoot, "app", "app-sw.js"));
});

app.get("/app/offline.html", (req, res) => {
  res.setHeader("Cache-Control", "public, max-age=3600");
  res.sendFile(path.join(staticRoot, "app", "offline.html"));
});

// 4. API routes
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);
app.use("/api/v1/nfl", nflRoutes);
app.use("/api/weather", weatherRoutes);

// 5. Docs and health check
setupSwagger(app);
app.get("/docs", (_req, res) => res.redirect("/api-docs"));
app.get("/health", (_req, res) =>
  res.json({ status: "OK", timestamp: new Date().toISOString() })
);

// 6. SPA fallback for all /app/* routes (MUST BE NEAR THE END)
app.get(/^\/app(\/.*)?$/, (req, res) => {
  res.setHeader("Cache-Control", "no-cache");
  res.sendFile(path.join(staticRoot, "app.html"));
});

// ─── Error Handling ───────────────────────────────────────────────────────────

// 404 Handler
app.use((req, res, _next) => {
  const file404 = path.join(staticRoot, "public", "404.html");
  if (fs.existsSync(file404)) {
    return res.status(404).sendFile(file404);
  }
  res.status(404).json({ error: "Not Found" });
});

// Global error handler
app.use((err, req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// ─── Server Start ─────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`🚀 Server listening on port ${PORT}`));
