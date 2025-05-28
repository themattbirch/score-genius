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
} from "../controllers/mlb_controller.js";

const router = express.Router();

// Define route: GET /api/v1/mlb/schedule
router.get("/schedule", getMlbSchedule);
router.get("/games/history", getMlbGameHistory);
// **ALL-TEAMS** season stats
router.get("/team-stats", getMlbAllTeamsSeasonStats);
// **Advanced Stats
router.get("/team-stats/advanced", getMlbAdvancedTeamStats);

// **Old: Single team** season stats
router.get("/teams/:team_id/stats/:season", getMlbTeamSeasonStats);

// ── Snapshot endpoint ──
// GET /api/v1/mlb/snapshots/:gameId
router.get("/snapshots/:gameId", getMlbSnapshot);

export default router;
