// backend/server/controllers/weather_controller.js

import { getWeatherDataForTeam } from "../services/weather_service.js";

/**
 * Handles the API request to fetch weather for a specific team.
 * @param {object} req - The Express request object.
 * @param {object} res - The Express response object.
 */
async function fetchWeatherForTeam(req, res) {
  // Extract query parameters from the URL
  const { sport, teamName } = req.query;

  // 1. Validate the input
  if (!sport || !teamName) {
    return res.status(400).json({
      message:
        'Bad Request. Both "sport" and "teamName" query parameters are required.',
    });
  }

  try {
    // 2. Call the service to do the actual work
    const weatherData = await getWeatherDataForTeam(sport, teamName);

    // 3. Send the successful response
    res.status(200).json(weatherData);
  } catch (error) {
    // 4. Handle any errors that occurred during the process
    console.error(
      `Error in weather controller for ${sport} - ${teamName}:`,
      error.message
    );

    // Send a generic error message to the client
    res
      .status(500)
      .json({ message: "An error occurred while fetching weather data." });
  }
}

// Export the function so the route can use it
export { fetchWeatherForTeam };
