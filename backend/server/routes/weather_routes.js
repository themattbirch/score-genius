// backend/server/routes/weather_routes.js

import express from "express";

// Import the specific controller function
import { fetchWeatherForTeam } from "../controllers/weather_controller.js";

const router = express.Router();

// When a GET request is made to the root of this route,
// call the fetchWeatherForTeam function from the controller.
router.get("/", fetchWeatherForTeam);

// Export the router as the default for server.js to consume
export default router;
