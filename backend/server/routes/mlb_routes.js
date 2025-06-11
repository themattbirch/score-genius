// backend/server/routes/mlb_routes.js
import express from "express";
// Use snake_case filename for controller import
import {
  getMlbSchedule,
  getMlbGameHistory,
  getMlbTeamSeasonStats,
  getMlbAllTeamsSeasonStats,
  getMlbAdvancedTeamStats,
  getMlbSnapshot,
  getMlbSnapshots,
} from "../controllers/mlb_controller.js";

import { LRUCache } from "lru-cache";

// simple 5-minute LRU cache
const cache = new LRUCache({ max: 100, ttl: 5 * 60 * 1000 });
const router = express.Router();
const MLB_SNAPSHOT_TABLE = "mlb_snapshots";

// Define route: GET /api/v1/mlb/schedule
router.get("/schedule", getMlbSchedule);
router.get("/games/history", getMlbGameHistory);
// **ALL-TEAMS** season stats
router.get("/team-stats", getMlbAllTeamsSeasonStats);
// **Advanced Stats
router.get("/team-stats/advanced", getMlbAdvancedTeamStats);

// **Old: Single team** season stats
router.get("/teams/:team_id/stats/:season", getMlbTeamSeasonStats);

// ── Snapshot endpoints ──
// GET /api/v1/mlb/snapshots/:gameId (for single game, triggers generation if not found)
router.get("/snapshots/:gameId", getMlbSnapshot);

// GET /api/v1/mlb/snapshots?gameIds=1,2,3 (for multiple games, fetches pre-generated)
router.get("/snapshots", getMlbSnapshots);
export default router;
