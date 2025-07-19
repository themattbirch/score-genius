// backend/server/controllers/weather_controller.js

import {
  getWeatherDataForTeam,
  getVenueInfo, // ⬅️  new helper that returns the venue JSON entry
} from "../services/weather_service.js";

/**
 * Handles the API request to fetch weather (and venue meta) for a specific team.
 * @param {object} req - The Express request object.
 * @param {object} res - The Express response object.
 */
async function fetchWeatherForTeam(req, res) {
  const { sport, teamName } = req.query;

  if (!sport || !teamName) {
    return res.status(400).json({
      message:
        'Bad Request. Both "sport" and "teamName" query parameters are required.',
    });
  }

  try {
    // Fetch weather data and venue metadata in parallel
    const [weatherData, venue] = await Promise.all([
      getWeatherDataForTeam(sport, teamName),
      getVenueInfo(sport, teamName), // returns { is_indoor: true/false, ... }
    ]);

    const responsePayload = {
      ...weatherData,
      isIndoor: venue?.is_indoor ?? null, // expose the flag to the client
    };

    res.status(200).json(responsePayload);
  } catch (error) {
    console.error(
      `Weather controller error for ${sport} - ${teamName}:`,
      error.message
    );
    res
      .status(500)
      .json({ message: "An error occurred while fetching weather data." });
  }
}

export { fetchWeatherForTeam };
