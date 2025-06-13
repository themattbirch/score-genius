// backend/server/server.js
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";

// Resolve environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.join(__dirname, "..", ".env"); // Path to backend/.env
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`🔑 Loaded env from ${envPath}`);
} else {
  console.log("🔑 No local .env file found; using host-provided vars");
}

// Validate Supabase credentials
const { SUPABASE_URL, SUPABASE_SERVICE_KEY } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("FATAL: SUPABASE_URL or SUPABASE_SERVICE_KEY missing");
  process.exit(1);
}

// Initialize Supabase client
export const supabase = createClient(
  SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  {
    auth: { persistSession: false },
  }
);

// Path to your **built frontend distribution** directory (e.g., /frontend/dist)
const frontEndDist = path.resolve(__dirname, "../../frontend/dist");
console.log("🔍 Serving static assets from", frontEndDist);

// Create Express app
const app = express();

// --- 1. Essential Middleware (Order is CRUCIAL!) ---
app.use(
  cors({
    origin: [
      "https://scoregenius.io",
      "http://localhost:5173", // Your frontend dev server
      "http://localhost:4173", // Another common dev port
    ],
  })
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

app.disable("etag");
app.use((req, res, next) => {
  res.set({
    "Cache-Control": "private, no-store, max-age=0, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
    "Surrogate-Control": "no-store",
  });
  next();
});

// --- 2. Mount API Routes (CRITICAL: BEFORE serving static files) ---
// This server directly handles these API endpoints.
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// --- 3. Serve Frontend Static Assets ---
// This serves all files directly from the 'dist' folder.
// This middleware will handle requests for /assets/*, /index.html, /app.html etc.
// This MUST come AFTER API routes.
app.use(express.static(frontEndDist));

// --- 4. SPA Fallbacks for client-side routing (DEBUGGING - SIMPLIFIED) ---
// Temporarily comment out all complex client-side routing rules.
// Only keep the most basic root route for initial testing.

// Serve index.html specifically at the root path "/"
app.get("/", (_req, res) => {
  res.sendFile(path.join(frontEndDist, "index.html"));
});

// COMMENT OUT THE FOLLOWING ROUTES TEMPORARILY:
// app.get("/app", (_req, res) => {
//   res.sendFile(path.join(frontEndDist, "app.html"));
// });

// app.get("/app/*", (req, res) => {
//   res.sendFile(path.join(frontEndDist, "app.html"));
// });

// app.get("/*", (req, res, next) => {
//   if (req.path.startsWith("/api/v1")) {
//     return next();
//   }
//   res.sendFile(path.join(frontEndDist, "app.html"));
// });

// --- Health Check Endpoint ---
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// --- Error-handling Middleware ---
app.use((err, _req, res, _next) => {
  console.error(err.stack || err);
  res.status(err.status || 500).json({
    error: {
      message: err.message ?? "Internal Server Error",
      stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    },
  });
});

// --- Start server ---
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
