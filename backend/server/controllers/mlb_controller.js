// backend/server/controllers/mlb_controller.js
// Use snake_case filename for service import
import * as mlbService from "../services/mlb_service.js";

// Controller function for GET /schedule
export const getMlbSchedule = async (req, res, next) => {
  try {
    // 1. Get date from query
    const { date } = req.query;

    // 2. Validate date
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return res.status(400).json({
        message: "Invalid or missing date parameter. Use YYYY-MM-DD format.",
      });
    }

    // 3. Call the NEW service function that filters by date
    const scheduleData = await mlbService.getMlbScheduleByDate(date); // <-- CALL NEW FUNCTION

    // 4. Send successful response (using existing format)
    res.status(200).json({
      message: `MLB schedule fetched successfully for ${date}`,
      retrieved: scheduleData?.length ?? 0,
      data: scheduleData || [],
    });
  } catch (error) {
    console.error(
      `Error in getMlbSchedule controller for date ${req.query.date}:`,
      error
    );
    // Pass error using next() or send response directly
    res
      .status(error.status || 500)
      .json({ message: error.message || "Failed to fetch MLB schedule" });
    // next(error);
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

// Fetch GET /api/v1/mlb/team-stats?season=YYYY
export const getMlbAllTeamsSeasonStats = async (req, res, next) => {
  try {
    const { season } = req.query;
    if (!season || !/^\d{4}$/.test(season)) {
      return res
        .status(400)
        .json({ message: "Missing or invalid ?season=YYYY query param." });
    }

    console.log("📊 MLB controller: fetching all teams for season", season);
    const stats = await mlbService.fetchMlbAllTeamStatsBySeason(Number(season));

    if (!stats?.length) {
      return res.status(404).json({
        message: `No MLB team stats found for season ${season}.`,
        data: [],
      });
    }

    res.status(200).json({
      message: `MLB team stats for ${season} fetched successfully`,
      season,
      retrieved: stats.length,
      data: stats,
    });
  } catch (error) {
    console.error("🔥 Error in getMlbAllTeamsSeasonStats:", error);
    res
      .status(error.status || 500)
      .json({ error: { message: error.message, stack: error.stack } });
  }
};

export const getMlbAdvancedTeamStats = async (req, res, next) => {
  try {
    const { season } = req.query;
    if (!season || !/^\d{4}$/.test(season)) {
      return res.status(400).json({
        message: "Invalid or missing season query parameter. Use YYYY format.", // Corrected YYYY format
      });
    }

    // Call the service function using the mlbService object ***
    const advancedTeamStats = await mlbService.fetchMlbAdvancedTeamStatsFromRPC(
      season
    ); // <--- FIXED LINE

    // Send response
    res.status(200).json({
      message: `MLB advanced team stats (RPC) fetched successfully for season ${season}`,
      retrieved: advancedTeamStats?.length ?? 0,
      data: advancedTeamStats || [],
    });
  } catch (error) {
    // Log the actual error received
    console.error(
      `Error in getMlbAdvancedTeamStats controller for season ${req.query.season}:`,
      error.message || error
    );

    // Check if the error is from Supabase RPC specifically
    let errorMessage =
      "Failed to fetch/calculate MLB advanced team stats via RPC";
    if (
      error.message &&
      error.message.includes("function public.get_mlb_advanced_team_stats")
    ) {
      errorMessage = `Error calling Supabase RPC function: ${error.message}`;
    } else if (error.message) {
      errorMessage = error.message; // Use the actual error message if available
    }

    res.status(error.status || 500).json({
      message: errorMessage,
    });
    // next(error); // Consider using next(error) if you have a global error handler middleware
  }
};
