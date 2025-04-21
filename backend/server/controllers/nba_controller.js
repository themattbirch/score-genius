// backend/server/controllers/nba_controller.js
// Use snake_case filename for service import
import * as nbaService from "../services/nba_service.js";
import { getSchedule } from "../services/nba_service.js";

// Controller to handle GET /api/v1/nba/schedule
// backend/server/controllers/nba_controller.js

// Controller to handle GET /api/v1/nba/schedule
export const getNbaSchedule = async (req, res, next) => {
  try {
    // 1. Get the date from the request query parameters
    const { date } = req.query;

    // 2. Validate the date format (YYYY-MM-DD)
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return res.status(400).json({ message: 'Invalid or missing date parameter. Use YYYY-MM-DD format.' });
    }

    // 3. Call the CORRECT service function that filters by date
    //    This function already handles formatting the response columns.
    const scheduleData = await getSchedule(date); // Using direct import

    // 4. Send the response (formatted like your MLB response for consistency)
    res.status(200).json({
         message: `NBA schedule fetched successfully for ${date}`,
         retrieved: scheduleData?.length ?? 0, // Use nullish check for safety
         data: scheduleData || [] // Return empty array if service returns null/undefined
     });

  } catch (error) {
    // Consistent error handling
    console.error(`Error in getNbaSchedule controller for date ${req.query.date}:`, error);
    res.status(error.status || 500).json({ message: error.message || 'Failed to fetch NBA schedule' });
    // Or use next(error); if you have middleware for it
    // next(error);
  }
};


export const getNbaInjuries = async (req, res, next) => {
  try {
    const injuriesData = await nbaService.fetchNbaInjuries();
    res.status(200).json({
      message: "NBA injuries fetched successfully",
      retrieved: injuriesData.length,
      data: injuriesData,
    });
  } catch (error) {
    console.error("Error in getNbaInjuries controller:", error);
    next(error);
  }
};

export const getNbaGameHistory = async (req, res, next) => {
  try {
    const options = {
      startDate: req.query.start_date || null,
      endDate: req.query.end_date || null,
      teamName: req.query.team_name || null,
      limit: Math.min(Math.max(parseInt(req.query.limit) || 20, 1), 100),
      page: Math.max(parseInt(req.query.page) || 1, 1),
    };

    const historicalData = await nbaService.fetchNbaGameHistory(options);
    res.status(200).json({
      message: "NBA historical game stats fetched successfully",
      options,
      retrieved: historicalData.length,
      data: historicalData,
    });
  } catch (error) {
    console.error("Error in getNbaGameHistory controller:", error);
    next(error);
  }
};

export const getNbaTeamSeasonStats = async (req, res, next) => {
  try {
    const { team_id, season } = req.params;
    const teamIdNum = parseInt(team_id, 10);
    if (isNaN(teamIdNum)) {
      return res
        .status(400)
        .json({ error: "Invalid Team ID. Must be numeric." });
    }
    if (!/^\d{4}$/.test(season)) {
      return res
        .status(400)
        .json({
          error:
            "Invalid Season format. Expecting a 4-digit year (e.g., 2023).",
        });
    }

    const teamStats = await nbaService.fetchNbaTeamStatsBySeason(
      teamIdNum,
      season
    );
    if (!teamStats) {
      return res.status(404).json({
        message: `NBA team stats not found for team ${teamIdNum}, season ${season}`,
        data: null,
      });
    }

    res.status(200).json({
      message: `NBA historical team stats for team ${teamIdNum}, season ${season} fetched successfully`,
      data: teamStats,
    });
  } catch (error) {
    console.error("Error in getNbaTeamSeasonStats controller:", error);
    next(error);
  }
};

export const getNbaPlayerGameHistory = async (req, res, next) => {
  try {
    const { player_id } = req.params;
    if (!player_id) {
      return res
        .status(400)
        .json({ error: "Player ID is required in the URL path." });
    }

    const limit = Math.min(Math.max(parseInt(req.query.limit) || 15, 1), 50);
    const page = Math.max(parseInt(req.query.page) || 1, 1);

    const gameLogData = await nbaService.fetchNbaPlayerGameHistory(player_id, {
      limit,
      page,
    });
    res.status(200).json({
      message: `NBA historical game stats for player ${player_id} fetched successfully`,
      options: { limit, page },
      retrieved: gameLogData.length,
      data: gameLogData,
    });
  } catch (error) {
    console.error("Error in getNbaPlayerGameHistory controller:", error);
    next(error);
  }
};

// If you still need a “schedule by date” endpoint, convert it to ESM too:
export const schedule = async (req, res, next) => {
  try {
    const date = req.query.date; // YYYY-MM-DD
    const sched = await getSchedule(date);
    res.status(200).json(sched);
  } catch (err) {
    console.error("Error in schedule controller:", err);
    next(err);
  }
};
