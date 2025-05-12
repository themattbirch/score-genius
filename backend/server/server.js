// backend/server/server.js
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

/* ──────────────────────────────────────────────────────────────
 * 1) Load /backend/.env **if it exists**; otherwise rely on host
 * ──────────────────────────────────────────────────────────── */
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

/* ----------------------------------------------------------------
 * 2) Validate that the required Supabase vars are present
 * ---------------------------------------------------------------- */
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error(
    "FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing in environment"
  );
  process.exit(1);
}

/* ──────────────────────────────────────────────────────────────
 * 3) Bring in libs that depend on those vars
 * ──────────────────────────────────────────────────────────── */
import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  { auth: { persistSession: false } }
);

/* ──────────────────────────────────────────────────────────────
 * 3) Express boilerplate (unchanged)
 * ──────────────────────────────────────────────────────────── */
const app = express();
const PORT = process.env.PORT || 3001;

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

/* ──────────────────────────────────────────────────────────────
 * 4) Dynamically import routes AFTER env + supabase are ready
 * ──────────────────────────────────────────────────────────── */
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;

app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// 5) Serve snapshots
app.use(
  "/snapshots",
  express.static(path.join(__dirname, "../../reports/snapshots"), {
    extensions: ["json"],
  })
);

/* ───────────── 6)  Serve built frontend assets ───────────── */
// NB: leave index.html to the SPA fallback ↓
app.use(
  express.static(path.resolve(__dirname, "../../frontend/dist"), {
    index: false,
  })
);

// 7a) PWA shell: all /app/* routes get app.html
app.get(
  /^\/app(?:\/.*)?$/, // any path under /app/
  (_req, res) =>
    res.sendFile(path.resolve(__dirname, "../../frontend/dist/app.html"))
);

// 7b) Marketing shell: all other non-API/non-snapshot paths get index.html
app.get(
  /^\/(?!api\/|snapshots\/).*/, // same as before
  (_req, res) =>
    res.sendFile(path.resolve(__dirname, "../../frontend/dist/index.html"))
);

/* --------------------------- Error-handling --------------------------- */
app.use((err, _req, res, _next) => {
  console.error(err.stack || err);
  res.status(err.status || 500).json({
    error: {
      message: err.message ?? "Internal Server Error",
      stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    },
  });
});

app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
