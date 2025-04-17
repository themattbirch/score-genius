// backend/server/controllers/mlb_controller.js
// Use snake_case filename for service import
import * as mlbService from "../services/mlb_service.js";

// Controller function for GET /schedule
export const getMlbSchedule = async (req, res, next) => {
  try {
    // Call the service function to fetch data
    // Fetches based on current ET date for today & tomorrow inside the service
    const scheduleData =
      await mlbService.WorkspaceMlbScheduleForTodayAndTomorrow();

    // Send successful response
    res.status(200).json({
      message: "MLB schedule fetched successfully",
      // Safely get length only if scheduleData is an array
      retrieved: Array.isArray(scheduleData) ? scheduleData.length : 0,
      data: scheduleData || [], // Return empty array if null/undefined
    });
  } catch (error) {
    // Pass error to the central error handler in server.js
    console.error("Error in getMlbSchedule controller:", error);
    next(error); // Forward error to Express error handler
  }
};

export const getMlbGameHistory = async (req, res, next) => {
  try {
    // Extract and validate query parameters (provide defaults)
    const options = {
      startDate: req.query.start_date || null, // Expects YYYY-MM-DD
      endDate: req.query.end_date || null, // Expects YYYY-MM-DD
      teamName: req.query.team_name || null, // Optional: Filter by team name
      limit: parseInt(req.query.limit) || 20, // Default limit 20
      page: parseInt(req.query.page) || 1, // Default page 1
    };

    // Basic validation/sanitization for limit/page
    options.limit = Math.max(1, Math.min(options.limit, 100)); // Clamp limit between 1 and 100
    options.page = Math.max(1, options.page);

    // Call the service function with options
    const historicalData = await mlbService.fetchMlbGameHistory(options);

    res.status(200).json({
      message: "MLB historical game stats fetched successfully",
      options: options, // Echo back options used for clarity
      retrieved: historicalData.length,
      data: historicalData,
    });
  } catch (error) {
    console.error("Error in getMlbGameHistory controller:", error);
    next(error); // Forward error to global handler
  }
};

export const getMlbTeamSeasonStats = async (req, res, next) => {
  try {
    const { team_id, season } = req.params; // Extract from URL path

    // Validate parameters
    const teamIdNum = parseInt(team_id);
    // MLB Season is often stored just as the starting year (e.g., 2023)
    // Adjust validation if your service/DB expects a different format
    const seasonNum = parseInt(season);
    if (
      isNaN(teamIdNum) ||
      isNaN(seasonNum) ||
      String(seasonNum).length !== 4
    ) {
      return res.status(400).json({
        error:
          "Invalid Team ID or Season provided. Team ID must be a number, Season must be a 4-digit year.",
      });
    }

    // Call the service function (ensure service expects year number)
    const teamStats = await mlbService.fetchMlbTeamStatsBySeason(
      teamIdNum,
      seasonNum // Pass the validated year number
    );

    // Check the result from the service function
    if (teamStats) {
      // Team stats found, send successful response
      res.status(200).json({
        message: `MLB historical team stats for team ${teamIdNum}, season ${seasonNum} fetched successfully`,
        data: teamStats, // Use the actual teamStats data here
      });
    } else {
      // If service returns null or empty, send 404 Not Found
      res.status(404).json({
        message: `MLB historical team stats not found for team ${teamIdNum}, season ${seasonNum}`,
        data: null,
      });
    }
  } catch (error) {
    // Handle any errors during validation or service call
    console.error("Error in getMlbTeamSeasonStats controller:", error);
    next(error); // Forward error to the global error handler
  }
};
