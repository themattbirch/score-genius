// backend/server/server.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// --- Load environment variables early ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Try loading .env from backend/, then project root
const envLocations = [
  path.join(__dirname, "../.env"),
  path.join(__dirname, "../../.env"),
];
let loaded = false;
envLocations.forEach((envPath) => {
  if (!loaded && fs.existsSync(envPath)) {
    dotenv.config({ path: envPath, override: true });
    console.log(`🔑 Loaded env from ${envPath}`);
    loaded = true;
  }
});
if (!loaded) console.log("🔑 No .env file found; relying on process.env");

// Import after env is set
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

// Express app
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
app.use((req, res, next) => {
  // normalize multiple slashes
  req.url = req.url.replace(/\/\/+/, "/");
  console.log(`${new Date().toISOString()} – ${req.method} ${req.url}`);
  next();
});
// Serve video and other media from frontend/dist/media
app.use(
  "/media",
  express.static(path.join(staticRoot, "media"), {
    // optional: set cache headers, e.g. 1 day
    maxAge: "1d",
  })
);

// Static paths
const staticRoot = path.join(__dirname, "static");
const marketingDir = path.join(staticRoot, "public");
const assetsDir = path.join(staticRoot, "assets");

// Marketing site at root
app.use(express.static(marketingDir, { index: false }));
app.get("/", (_req, res) =>
  res.sendFile(path.join(marketingDir, "index.html"))
);

// Serve PWA assets
app.use("/assets", express.static(assetsDir));
app.get("/sw.js", (_req, res) =>
  res.sendFile(path.join(staticRoot, "app-sw.js"))
);

// Serve application shell
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

// 404 / error handlers
app.use((req, res) => res.status(404).json({ error: "Not Found" }));
app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

// Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
