import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import express from "express";
import cors from "cors";
import { createClient } from "@supabase/supabase-js";
// REMOVED: import { createProxyMiddleware } from "http-proxy-middleware"; // NOT NEEDED

// Resolve environment variables (assuming this server is in backend/server/)
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
export const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

// Path to your **built frontend distribution** directory (e.g., /frontend/dist)
// This server will serve these files.
const frontEndDist = path.resolve(__dirname, "../../frontend/dist");
console.log("🔍 Serving static assets from", frontEndDist);

// Create Express app
const app = express();

// --- 1. Essential Middleware (Order matters!) ---
app.use(
  cors({
    // CORS configuration to allow your frontend domain(s)
    origin: [
      "https://scoregenius.io",
      "http://localhost:5173", // Your frontend dev server
      "http://localhost:4173", // Another common dev port
    ],
  })
);
app.use(express.json()); // To parse JSON request bodies
app.use(express.urlencoded({ extended: true })); // To parse URL-encoded request bodies

// Request logger
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} – ${req.method} ${req.path}`);
  next();
});

// Cache control headers (Optional, but good for production)
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
// These are the API routes handled directly by this server.
const nbaRoutes = (await import("./routes/nba_routes.js")).default;
const mlbRoutes = (await import("./routes/mlb_routes.js")).default;
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

// Optional: Serve /snapshots if you still need to serve static JSON files directly
// app.use("/snapshots", express.static(path.join(__dirname, "../../reports/snapshots"), { extensions: ["json"] }));

// --- 3. Serve Frontend Static Assets ---
// This serves all the files from your frontend's 'dist' folder (index.html, CSS, JS, assets)
// This MUST come AFTER API routes, so API requests are handled first.
app.use(express.static(frontEndDist));

// --- 4. SPA Fallback (Catch-all for client-side routing) ---
// For any GET request not handled by previous middleware (APIs or static files),
// send the main HTML file (your SPA entry point) to allow client-side routing.
// This ensures that refreshing deep links (e.g., /app/games) works.
app.get("/*", (req, res, next) => {
  // If the request path begins with /api/v1, let it fall through to subsequent error handlers
  // if it wasn't caught by the specific API routes above.
  if (req.path.startsWith("/api/v1")) {
    return next();
  }
  // Otherwise, assume it's a client-side route and serve app.html (or index.html)
  res.sendFile(path.join(frontEndDist, "app.html")); // Assuming your main SPA entry is app.html
});

// --- Health Check ---
app.get("/health", (_req, res) =>
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() })
);

// --- Error-handling Middleware ---
// This must be the last app.use()
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
const PORT = process.env.PORT || 10000; // Use 10000 as default, matching previous backend
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
