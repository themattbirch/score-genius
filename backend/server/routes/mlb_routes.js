// backend/server/routes/mlb_routes.js
import express from "express";
// Use snake_case filename for controller import
import {
  getMlbSchedule,
  getMlbGameHistory,
  getMlbTeamSeasonStats,
  getMlbAllTeamsSeasonStats,
} from "../controllers/mlb_controller.js";

const router = express.Router();

// Define route: GET /api/v1/mlb/schedule
router.get("/schedule", getMlbSchedule);
router.get("/games/history", getMlbGameHistory);
// **ALL-TEAMS** season stats
router.get("/team-stats", getMlbAllTeamsSeasonStats);
// **Single team** season stats
router.get("/teams/:team_id/stats/:season", getMlbTeamSeasonStats);

// Add more MLB routes here later.

export default router;
