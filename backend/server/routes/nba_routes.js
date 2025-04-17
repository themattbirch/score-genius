// backend/server/routes/nba_routes.js
import express from "express";

// Import controller functions (use snake_case filename)
import {
  getNbaSchedule,
  getNbaInjuries,
  getNbaGameHistory,
  getNbaTeamSeasonStats,
  getNbaPlayerGameHistory,
} from "../controllers/nba_controller.js";
// Import other controllers here as you create them...

const router = express.Router();

// --- Define NBA Routes ---

// GET /api/v1/nba/schedule (Fetches today/tomorrow's games)
router.get("/schedule", getNbaSchedule);
router.get("/injuries", getNbaInjuries);
router.get("/games/history", getNbaGameHistory);
router.get("/teams/:team_id/stats/:season", getNbaTeamSeasonStats);
router.get("/players/:player_id/stats/history", getNbaPlayerGameHistory);

// --- Placeholder routes ---

// Example: GET /api/v1/nba/games/upcoming?days=N

export default router;
