// backend/server/controllers/nba_controller.js
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
      return res.status(400).json({
        message: "Invalid or missing date parameter. Use YYYY-MM-DD format.",
      });
    }

    // 3. Call the CORRECT service function that filters by date
    //    This function already handles formatting the response columns.
    const scheduleData = await getSchedule(date); // Using direct import

    // 4. Send the response (formatted like your MLB response for consistency)
    res.status(200).json({
      message: `NBA schedule fetched successfully for ${date}`,
      retrieved: scheduleData?.length ?? 0, // Use nullish check for safety
      data: scheduleData || [], // Return empty array if service returns null/undefined
    });
  } catch (error) {
    // Consistent error handling
    console.error(
      `Error in getNbaSchedule controller for date ${req.query.date}:`,
      error
    );
    res
      .status(error.status || 500)
      .json({ message: error.message || "Failed to fetch NBA schedule" });
    // Or use next(error); if you have middleware for it
    // next(error);
  }
};

export const getNbaInjuries = async (req, res, next) => {
  const { date } = req.query;
  console.log("→ [getNbaInjuries] start, date =", date);

  try {
    const start = Date.now();
    // pass the date into your service
    const injuriesData = await nbaService.fetchNbaInjuries(date);
    console.log(
      `→ [getNbaInjuries] service returned in ${Date.now() - start}ms; count=`,
      Array.isArray(injuriesData) ? injuriesData.length : injuriesData
    );

    return res.status(200).json({
      message: "NBA injuries fetched successfully",
      retrieved: Array.isArray(injuriesData) ? injuriesData.length : 0,
      data: injuriesData,
    });
  } catch (error) {
    console.error("→ [getNbaInjuries] ERROR:", error);
    return next(error);
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
      return res.status(400).json({
        error: "Invalid Season format. Expecting a 4-digit year (e.g., 2023).",
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
      return res.status(400).json({ error: "Player ID is required." });
    }

    // Grab and validate season
    const { season } = req.query;
    if (!season || !/^\d{4}$/.test(season)) {
      return res
        .status(400)
        .json({ error: "Missing or invalid ?season=YYYY parameter." });
    }
    const seasonYear = parseInt(season, 10);

    // Pagination
    const limit = Math.min(
      Math.max(parseInt(req.query.limit, 10) || 15, 1),
      100
    );
    const page = Math.max(parseInt(req.query.page, 10) || 1, 1);

    // Call service
    const gameLogData = await nbaService.fetchNbaPlayerGameHistory(
      player_id,
      seasonYear,
      { limit, page }
    );

    // Respond
    return res.status(200).json({
      message: `Fetched ${
        gameLogData.length
      } games for player ${player_id} (${seasonYear}-${String(
        seasonYear + 1
      ).slice(-2)})`,
      options: { season: seasonYear, limit, page },
      data: gameLogData,
    });
  } catch (error) {
    console.error("Error in getNbaPlayerGameHistory:", error);
    next(error);
  }
};

/* --------------------------------------------------------------
 *  GET /api/v1/nba/player-stats
 *  Query params:
 *    season   (required) – 4-digit year, e.g. 2024 means 2024-25
 *    pos      (optional) – G | F | C
 *    search   (optional) – ilike filter on player_name
 * -------------------------------------------------------------*/
export const getNbaAllPlayersSeasonStats = async (req, res, next) => {
  try {
    const { season, search } = req.query; // NEW

    if (!season || !/^\d{4}$/.test(season)) {
      return res
        .status(400)
        .json({ message: "Missing or invalid ?season=YYYY query param." });
    }
    const seasonYear = Number(season);

    // 2. Build filter object
    // const filters = {                              // OLD
    //   search: search ? String(search) : null,      // OLD
    // };                                             // OLD
    const filters = {
      // NEW
      search: search ? String(search) : null, // NEW (only contains search now)
    }; // NEW

    // 3. Call service - This should now pass only { search: ... } in filters
    const stats = await nbaService.fetchNbaAllPlayerStatsBySeason(
      seasonYear,
      filters
    );

    res.status(200).json({
      message: `NBA player stats for ${seasonYear}-${String(
        seasonYear + 1
      ).slice(-2)} fetched successfully`,
      season: seasonYear,
      retrieved: stats.length,
      filters,
      data: stats,
    });
  } catch (error) {
    console.error("Error in getNbaAllPlayersSeasonStats controller:", error);
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

/* ----------------------------------------------
 *  GET /api/v1/nba/team-stats?season=YYYY
 *  Returns season-level metrics for **all** teams.
 * ---------------------------------------------*/
export const getNbaAllTeamsSeasonStats = async (req, res, next) => {
  try {
    const { season } = req.query;

    if (!season || !/^\d{4}$/.test(season)) {
      return res
        .status(400)
        .json({ message: "Missing or invalid ?season=YYYY query param." });
    }

    console.log("📊 Controller received request for season:", season);

    const stats = await nbaService.fetchNbaAllTeamStatsBySeason(Number(season));

    if (!stats?.length) {
      return res.status(404).json({
        message: `No NBA team stats found for season ${season}.`,
        data: [],
      });
    }

    res.status(200).json({
      message: `NBA team stats for ${season} fetched successfully`,
      season,
      retrieved: stats.length,
      data: stats,
    });
  } catch (error) {
    console.error("🔥 Controller caught error:", error);
    res.status(error.status || 500).json({
      error: { message: error.message, stack: error.stack },
    });
  }
};
/**
 * Handles GET /api/v1/nba/advanced-stats?season=YYYY
 * Fetches calculated advanced team stats for a given season.
 */
export const getNbaAdvancedStats = async (req, res, next) => {
  try {
    // 1. Get and Validate Season from query param
    const { season } = req.query;
    if (!season || !/^\d{4}$/.test(season)) {
      // Send 400 Bad Request if season is missing or not 4 digits
      return res
        .status(400)
        .json({ message: "Missing or invalid ?season=YYYY query param." });
    }
    // Convert validated season string to a number
    const seasonYear = Number(season);

    console.log(
      `📊 Controller received request for ADVANCED stats, season: ${seasonYear}`
    );

    // 2. Call the corresponding Service Function
    const advancedStatsData = await nbaService.fetchNbaAdvancedStatsBySeason(
      seasonYear
    );

    // 3. Send successful response (service function returns [] if no data)
    res.status(200).json({
      message: `NBA advanced team stats for ${seasonYear}-${String(
        seasonYear + 1
      ).slice(-2)} fetched successfully`,
      season: seasonYear,
      retrieved: advancedStatsData.length,
      data: advancedStatsData, // Send the array (could be empty)
    });
  } catch (error) {
    // 4. Handle any errors during the process
    console.error("🔥 Error in getNbaAdvancedStats controller:", error);
    // Pass the error to the central error handler in server.js
    next(error);
  }
};
