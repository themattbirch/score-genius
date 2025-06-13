import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createClient } from "@supabase/supabase-js";
import { LRUCache } from "lru-cache";
import nbaRoutes from "./routes/nba_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";

// ──────────────────────────────────────────────────────────────
// 1) Load environment variables
// ──────────────────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

// ──────────────────────────────────────────────────────────────
// 2) Validate Supabase credentials & initialize client
// ──────────────────────────────────────────────────────────────
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error(
    "FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing in environment"
  );
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// ──────────────────────────────────────────────────────────────
// 3) Create Express app and middleware
// ──────────────────────────────────────────────────────────────
const app = express();

// Enable CORS for frontend domain
app.use(cors({ origin: ["https://scoregenius.io"] }));
app.use(express.json());

// Simple request logging
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// ──────────────────────────────────────────────────────────────
// 4) In-memory LRU cache setup (optional use in routes)
// ──────────────────────────────────────────────────────────────
export const cache = new LRUCache({ max: 100, ttl: 5 * 60 * 1000 });

// ──────────────────────────────────────────────────────────────
// 5) Mount API routes
// ──────────────────────────────────────────────────────────────
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// ──────────────────────────────────────────────────────────────
// 6) Health check
// ──────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// ──────────────────────────────────────────────────────────────
// 7) Start server
// ──────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
