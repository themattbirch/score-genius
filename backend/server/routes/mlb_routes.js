// backend/server/routes/mlb_routes.js
import express from "express";
// Use snake_case filename for controller import
import {
  getMlbSchedule,
  getMlbGamesHistory,
} from "../controllers/mlb_controller.js";

const router = express.Router();

// Define route: GET /api/v1/mlb/schedule
router.get("/schedule", getMlbSchedule);
router.get("/games/history", getMlbGamesHistory);

// Add more MLB routes here later...

export default router;
