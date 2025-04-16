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

// Add controllers for predictions, team stats etc. later...
// export const getNbaPredictions = async (req, res, next) => { ... }
