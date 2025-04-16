// backend/server/routes/nba_routes.js
import express from "express";

// Import controller functions (use snake_case filename)
import { getNbaSchedule } from "../controllers/nba_controller.js";
// Import other controllers here as you create them...

const router = express.Router();

// --- Define NBA Routes ---

// GET /api/v1/nba/schedule (Fetches today/tomorrow's games)
router.get("/schedule", getNbaSchedule); // <-- Added this route

// --- Your existing placeholder routes ---
// GET /api/v1/nba/predictions?date=YYYY-MM-DD
router.get("/predictions", (req, res, next) => {
  // TODO: Implement logic using nba_controller
  const { date } = req.query;
  console.log(`Placeholder: NBA predictions for date: ${date}`);
  res.json({ message: `NBA predictions for ${date || "today"}`, data: [] });
});

// GET /api/v1/nba/teams/{teamId}/stats
router.get("/teams/:teamId/stats", (req, res, next) => {
  // TODO: Implement logic using nba_controller
  const { teamId } = req.params;
  console.log(`Placeholder: NBA stats for team ID: ${teamId}`);
  res.json({ message: `NBA stats for team ${teamId}`, data: {} });
});

// GET /api/v1/nba/games/upcoming?days=N
router.get("/games/upcoming", (req, res, next) => {
  // Note: This might be redundant or could be combined with /schedule logic
  // Or it could fetch further out than just today/tomorrow
  const { days } = req.query;
  console.log(
    `Placeholder: Upcoming NBA games for next ${days || "default"} days`
  );
  res.json({ message: `Upcoming NBA games`, data: [] });
});
// --- End placeholder routes ---

// --- Add other NBA-specific routes here ---

export default router;
