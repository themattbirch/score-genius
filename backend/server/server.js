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

/* ─── 1. Environment ────────────────────────────────────────────────────── */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const envPath = path.join(__dirname, "..", ".env");

if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

/* ─── 2. Supabase ───────────────────────────────────────────────────────── */
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

/* ─── 3. Front-end build location ───────────────────────────────────────── */
const FRONTEND_DIST = path.resolve(__dirname, "../../frontend/dist");
console.log("FRONTEND_DIST path:", FRONTEND_DIST);
console.log("  exists?", fs.existsSync(FRONTEND_DIST)); // should be true

/* ─── 4. Express setup ──────────────────────────────────────────────────── */
const app = express();
app.use(cors()); // relax CORS as needed
app.use(express.json());
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

/* ─── 5. Static assets ──────────────────────────────────────────────────── */
app.use(express.static(FRONTEND_DIST));

/* ─── 6. API routes ─────────────────────────────────────────────────────── */
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

/* ─── 7. Health check ───────────────────────────────────────────────────── */
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

/* ─── 8. SPA + marketing fallbacks ──────────────────────────────────────── */
// PWA shell under /app
app.get(/^\/app(\/.*)?$/, (req, res) => {
  console.log(`[SPA fallback] ${req.path}`);
  res.sendFile(path.join(FRONTEND_DIST, "app.html"));
});

// Marketing / root shell (index.html lives in dist/public/)
app.get(/^\/(?!.well-known)(?!api\/v1)(?!app)(?!.*\.).*$/, (req, res) => {
  console.log(`[Marketing fallback] ${req.path}`);
  res.sendFile(path.join(FRONTEND_DIST, "public", "index.html"));
});

/* ─── 9. Error handlers ─────────────────────────────────────────────────── */
app.use((req, res) => res.status(404).json({ error: "Not Found" }));
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server Error" });
});

/* ─── 10. Start server ──────────────────────────────────────────────────── */
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
