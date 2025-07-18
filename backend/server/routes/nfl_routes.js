// backend/server/routes/nfl_routes.js
import express from "express";
import {
  isValidConference,
  isValidDivision,
  NFL_ALLOWED_CONFERENCES,
  NFL_ALLOWED_DIVISIONS,
  parseSeasonParam,
  parseCsvInts,
} from "../utils/nfl_validation.js";

import {
  getNflTeamSeasonFull,
  getNflTeamSeasonRegOnly,
  getNflSchedule,
  getNflSnapshot,
  getNflSnapshots,
  getNflDashboard,
  getNflSos,
  getNflSrs,
  getNflCronHealth,
  getNflValidation,
} from "../controllers/nfl_controller.js";

const router = express.Router();

// /api/v1/nfl/teams/:season/full
router.get("/teams/:season/full", getNflTeamSeasonFull);
// /api/v1/nfl/teams/:season/regonly
router.get("/teams/:season/regonly", getNflTeamSeasonRegOnly);

// Schedule endpoint (past vs future)
router.get("/schedule", getNflSchedule);

// SRS + SoS endpoints

router.get("/teams/:season/sos", getNflSos);
router.get("/teams/:season/srs", getNflSrs);

// dashboard route
router.get("/teams/:season/dashboard", getNflDashboard);

// Snapshots endpoints
router.get("/snapshots/:gameId", getNflSnapshot);
router.get("/snapshots", getNflSnapshots);

// Health checks
router.get("/health/cron", getNflCronHealth);
router.get("/health/validate", getNflValidation);

export default router;
