// backend/server/routes/nba_routes.js
import express from "express";

// Import controller functions (use snake_case filename)
import {
  getNbaSchedule,
  getNbaInjuries,
  getNbaGameHistory,
  getNbaAllTeamsSeasonStats,
  getNbaTeamSeasonStats,
  getNbaPlayerGameHistory,
} from "../controllers/nba_controller.js";

const router = express.Router();

// --- Define NBA Routes ---

// GET /api/v1/nba/schedule (Fetches today/tomorrow's games)
router.get("/schedule", getNbaSchedule);
router.get("/injuries", getNbaInjuries);
router.get("/games/history", getNbaGameHistory);
// **ALL-TEAMS** season stats
router.get("/team-stats", getNbaAllTeamsSeasonStats);
// **Single team** season stats
router.get("/teams/:team_id/stats/:season", getNbaTeamSeasonStats);
// **NBA Player season stats
router.get("/players/:player_id/stats/history", getNbaPlayerGameHistory);

export default router;
