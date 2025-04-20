// backend/server/server.js
import express from "express";
import cors from "cors";
import "dotenv/config"; // only once

// --- Route Imports (using import, add .js extension) ---
import nbaRoutes from "./routes/nba_routes.js";
import mlbRoutes from "./routes/mlb_routes.js";

// Debug logs for env
console.log(`DEBUG server.js: Supabase URL = ${process.env.SUPABASE_URL}`);
console.log(
  `DEBUG server.js: Supabase Service Key Loaded = ${!!process.env
    .SUPABASE_SERVICE_KEY}`
);

const app = express();
const PORT = process.env.PORT || 3001;

// --- Disable ETags globally so clients always fetch fresh data ---
app.disable("etag");

// --- No‑cache headers for all responses ---
// Put this near the top of server.js, before your routes:
app.use((req, res, next) => {
  res.set({
    // max-age=0 for browsers, s-maxage=0 for CDNs, private so proxies won’t reuse
    "Cache-Control":
      "private, no-store, max-age=0, s-maxage=0, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
  });
  next();
});

// --- Middleware ---
app.use(
  cors({
    origin: ["https://scoregenius.io", "http://localhost:5173"],
  })
);
app.use(express.json());

// Simple request logger
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// --- API Routes ---
app.use("/api/v1/nba", nbaRoutes);
app.use("/api/v1/mlb", mlbRoutes);

app.get("/health", (req, res) => {
  res.status(200).json({ status: "OK", timestamp: new Date().toISOString() });
});

// --- Error Handling ---
app.use((err, req, res, next) => {
  console.error("Error:", err.stack || err.message || err);
  const status = err.status || 500;
  const message = err.message || "Internal Server Error";
  res.status(status).json({
    error: {
      message,
      stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    },
  });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
