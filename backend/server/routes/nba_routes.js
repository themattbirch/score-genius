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
} from "../controllers/nba_controller.js";


import { LRUCache } from "lru-cache";

// simple 5-minute LRU cache
const cache = new LRUCache({ max: 100, ttl: 5 * 60 * 1000 });
const router = express.Router();
const NBA_SNAPSHOT_TABLE = "nba_snapshots";

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
// GET /api/v1/nba/snapshots?gameIds=1,2,3
router.get("/snapshots", async (req, res, next) => {
  try {
    const ids = (req.query.gameIds || "").split(",");
    const { data, error } = await supabase
      .from(NBA_SNAPSHOT_TABLE)
      .select("*")
      .in("game_id", ids);
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    next(err);
  }
});
export default router;
