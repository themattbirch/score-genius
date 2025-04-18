// backend/server/controllers/nba_controller.js
// Use snake_case filename for service import
import * as nbaService from "../services/nba_service.js";
// Import Luxon if needed for date manipulation here (though likely done in service)
// import { DateTime } from 'luxon';

// Controller to handle GET /api/v1/nba/schedule
export const getNbaSchedule = async (req, res, next) => {
  try {
    // Call service function to get data for today/tomorrow ET
    const scheduleData =
      await nbaService.WorkspaceNbaScheduleForTodayAndTomorrow();

    // Send successful JSON response
    res.status(200).json({
      message: "NBA schedule fetched successfully",
      // Safely get length only if scheduleData is an array
      retrieved: Array.isArray(scheduleData) ? scheduleData.length : 0,
      // Return empty array if data is null/undefined
      data: scheduleData || [],
    });
  } catch (error) {
    // Pass errors to the global error handler in server.js
    console.error("Error in getNbaSchedule controller:", error);
    next(error); // Forward error to Express error handler
  }
};

export const getNbaInjuries = async (req, res, next) => {
  try {
    // Call the service function to fetch all current injury records
    const injuriesData = await nbaService.fetchNbaInjuries();

    res.status(200).json({
      message: "NBA injuries fetched successfully",
      retrieved: injuriesData.length,
      data: injuriesData,
    });
  } catch (error) {
    console.error("Error in getNbaInjuries controller:", error);
    next(error); // Forward error to global handler
  }
};

export const getNbaGameHistory = async (req, res, next) => {
  // <-- RENAMED HERE
  try {
    const options = {
      startDate: req.query.start_date || null,
      endDate: req.query.end_date || null,
      teamName: req.query.team_name || null,
      limit: parseInt(req.query.limit) || 20,
      page: parseInt(req.query.page) || 1,
    };
    options.limit = Math.max(1, Math.min(options.limit, 100));
    options.page = Math.max(1, options.page);

    const historicalData = await nbaService.fetchNbaGameHistory(options); // Service function name was correct
    res.status(200).json({
      message: "NBA historical game stats fetched successfully",
      options: options,
      retrieved: historicalData.length,
      data: historicalData,
    });
  } catch (error) {
    console.error("Error in getNbaGameHistory controller:", error); // Update log message if desired
    next(error);
  }
};

// Corrected and re-indented function
export const getNbaTeamSeasonStats = async (req, res, next) => {
  try {
    const { team_id, season } = req.params;

    // Validate team_id as number
    const teamIdNum = parseInt(team_id);
    if (isNaN(teamIdNum)) {
      return res
        .status(400)
        .json({ error: "Invalid Team ID. Must be numeric." });
    }

    // Validate season looks like a 4-digit year string (e.g., "2023")
    if (!/^\d{4}$/.test(season)) {
      return res.status(400).json({
        error: "Invalid Season format. Expecting a 4-digit year (e.g., 2023).",
      });
    }
    // Keep the validated season as a string, as passed in the URL
    const seasonStr = season;

    // Call the service function, passing the number ID and string season
    // Ensure fetchNbaTeamStatsBySeason service function expects the season this way
    const teamStats = await nbaService.fetchNbaTeamStatsBySeason(
      teamIdNum,
      seasonStr // Pass the validated season string
    );

    // Check the result from the service function
    if (teamStats) {
      // Service returns data object
      res.status(200).json({
        // Updated message to include team/season for clarity
        message: `NBA historical team stats for team ${teamIdNum}, season ${seasonStr} fetched successfully`,
        data: teamStats, // Send the single stats object
      });
    } else {
      // Service returns null (no record found)
      res.status(404).json({
        // *** Fixed variable name here: seasonNum -> seasonStr ***
        message: `NBA historical team stats not found for team ${teamIdNum}, season ${seasonStr}`,
        data: null,
      });
    }
  } catch (error) {
    // Handle any other unexpected errors
    console.error("Error in getNbaTeamSeasonStats controller:", error);
    next(error); // Forward other errors
  }
};

export const getNbaPlayerGameHistory = async (req, res, next) => {
  try {
    const { player_id } = req.params; // Extract player ID from URL path

    // Extract and validate query parameters for pagination
    const options = {
      limit: parseInt(req.query.limit) || 15, // Default limit 15 games
      page: parseInt(req.query.page) || 1, // Default page 1
    };
    options.limit = Math.max(1, Math.min(options.limit, 50)); // Clamp limit (e.g., max 50)
    options.page = Math.max(1, options.page);

    // Basic check if player_id was provided
    if (!player_id) {
      return res
        .status(400)
        .json({ error: "Player ID is required in the URL path." });
    }

    // Call the service function with player_id and options
    const gameLogData = await nbaService.fetchNbaPlayerGameHistory(
      player_id,
      options
    );

    res.status(200).json({
      message: `NBA historical game stats for player ${player_id} fetched successfully`,
      options: { limit: options.limit, page: options.page }, // Echo back pagination
      retrieved: gameLogData.length,
      data: gameLogData,
    });
  } catch (error) {
    console.error("Error in getNbaPlayerGameHistory controller:", error);
    next(error); // Forward error
  }
};

// Add controllers for predictions, team stats etc. later...
