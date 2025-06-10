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
  getNbaAllPlayersSeasonStats,
  getNbaAdvancedStats,
  getNbaSnapshot,
  getNbaSnapshots,
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
router.get("/player-stats", getNbaAllPlayersSeasonStats);
// **Advanced Stats
router.get("/advanced-stats", getNbaAdvancedStats);
// **Original Player Stats Lookup
router.get("/players/:player_id/stats/history", getNbaPlayerGameHistory);
// ── Snapshot endpoint ──
// ── Snapshot endpoints ──
// GET /api/v1/nba/snapshots/:gameId (for single game, triggers generation if not found)
router.get("/snapshots/:gameId", getNbaSnapshot); // Existing function handles single ID + generation
router.get("/snapshots", getNbaSnapshots);

export default router;
