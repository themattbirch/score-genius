// backend/server/server.js
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

/* ──────────────────────────────────────────────────────────────
 * 1) Load /backend/.env BEFORE anything touches process.env
 * ──────────────────────────────────────────────────────────── */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.join(__dirname, "..", ".env");
const { parsed, error } = dotenv.config({ path: envPath });

if (error) {
  console.error(`❌  dotenv couldn't read ${envPath}`, error);
  process.exit(1);
}

if (!parsed?.SUPABASE_URL || !parsed?.SUPABASE_SERVICE_KEY) {
  console.error("❌  SUPABASE_URL or SUPABASE_SERVICE_KEY missing in .env");
  process.exit(1);
}

/* ──────────────────────────────────────────────────────────────
 * 2) Now it’s safe to import anything that relies on env vars
 *    Use top-level await for dynamic imports so we control order.
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
