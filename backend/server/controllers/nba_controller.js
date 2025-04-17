// backend/server/controllers/nba_controller.js
// Use snake_case filename for service import
import * as nbaService from "../services/nba_service.js";
// Import Luxon if needed for date manipulation here (though likely done in service)
// import { DateTime } from 'luxon';

// Controller to handle GET /api/v1/nba/schedule
export const getNbaSchedule = async (req, res, next) => {
  try {
    // Call service function to get data for today/tomorrow ET
    const scheduleData = await nbaService.fetchNbaScheduleForTodayAndTomorrow();

    // Send successful JSON response
    res.status(200).json({
      message: "NBA schedule fetched successfully",
      retrieved: scheduleData.length, // Use 'retrieved' instead of 'count' maybe
      data: scheduleData,
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

export const getNbaGamesHistory = async (req, res, next) => {
  try {
    // Extract and validate query parameters (provide defaults)
    const options = {
      startDate: req.query.start_date || null, // YYYY-MM-DD
      endDate: req.query.end_date || null, // YYYY-MM-DD
      teamName: req.query.team_name || null, // Filter by team name (optional)
      limit: parseInt(req.query.limit) || 20, // Default limit
      page: parseInt(req.query.page) || 1, // Default page
    };

    // Basic validation for limit/page
    options.limit = Math.max(1, Math.min(options.limit, 100)); // Clamp limit between 1 and 100
    options.page = Math.max(1, options.page);

    // Call the service function with options
    const historicalData = await nbaService.fetchNbaGamesHistory(options);

    res.status(200).json({
      message: "NBA historical game stats fetched successfully",
      options: options, // Echo back options used
      retrieved: historicalData.length,
      data: historicalData,
    });
  } catch (error) {
    console.error("Error in getNbaGamesHistory controller:", error);
    next(error); // Forward error
  }
};

// Add controllers for predictions, team stats etc. later...
