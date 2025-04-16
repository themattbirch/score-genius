// backend/server/controllers/mlb_controller.js
// Use snake_case filename for service import
import * as mlbService from '../services/mlb_service.js';
import { DateTime } from 'luxon'; // Import Luxon here if needed, or keep in service

// Controller function for GET /schedule
export const getMlbSchedule = async (req, res, next) => {
  try {
    // Call the service function to fetch data
    // Fetches based on current ET date for today & tomorrow inside the service
    const scheduleData = await mlbService.fetchMlbScheduleForTodayAndTomorrow();

    // Send successful response
    res.status(200).json({
        message: "MLB schedule fetched successfully",
        retrieved: scheduleData.length,
        // Optional: Add current date info for context
        // date_retrieved_for_et: DateTime.now().setZone("America/New_York").toISODate(),
        data: scheduleData
    });
  } catch (error) {
    // Pass error to the central error handler in server.js
    console.error("Error in getMlbSchedule controller:", error);
    next(error); // Forward error to Express error handler
  }
};

// Add more controller functions for other MLB routes here...