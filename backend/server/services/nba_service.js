// backend/server/services/nba_service.js

// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon";
// Import your simple in‑memory or Redis cache helper
import cache from "../utils/cache.js";

// Define constants specific to this service
const NBA_SCHEDULE_TABLE = "nba_game_schedule";
const NBA_INJURIES_TABLE = "nba_injuries";
const NBA_HISTORICAL_GAMES_TABLE = "nba_historical_game_stats";
const NBA_HISTORICAL_TEAM_STATS_TABLE = "nba_historical_team_stats";
const NBA_HISTORICAL_PLAYER_STATS_TABLE = "nba_historical_player_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

// --- Helper function for dates (ensure consistent formatting YYYY-MM-DD) ---
const getUTCDateString = (date) => date.toISOString().split("T")[0];
// --- End Helper Function ---

// Fetch today & tomorrow’s schedule
export const fetchNbaScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching NBA schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate();
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate();

  const { data, error, status } = await supabase
    .from(NBA_SCHEDULE_TABLE)
    .select(
      `
      game_id,
      game_date,
      scheduled_time,
      status,
      home_team,
      away_team,
      venue,
      predicted_home_score,
      predicted_away_score,
      moneyline_clean,
      spread_clean,
      total_clean
    `
    )
    .in("game_date", [todayStr, tomorrowStr])
    .order("scheduled_time", { ascending: true });

  if (error) {
    console.error("Supabase error fetching NBA schedule:", error);
    const dbError = new Error(error.message || "Database query failed");
    dbError.status = status || 500;
    throw dbError;
  }

  console.log(`Service: Found ${data?.length ?? 0} NBA games.`);
  return data || [];
};

// --- Unified Data Structure ---
// --- JSDoc Definition for the data structure returned by getSchedule ---
/**
 * Represents unified NBA game data. Matches frontend UnifiedGame type.
 * @typedef {object} UnifiedNBAGameData
 * @property {string} id
 * @property {string} game_date
 * @property {string} homeTeamName // Standardized name
 * @property {string} awayTeamName // Standardized name
 * @property {string | null} [gameTimeUTC] // Standardized name (from scheduled_time)
 * @property {string | null} [statusState] // Add if available
 * @property {number | null} [spreadLine] // Standardized name (from spread_clean)
 * @property {number | null} [totalLine] // Standardized name (from total_clean)
 * @property {number | null} [predictionHome]
 * @property {number | null} [predictionAway]
 * @property {number | null} [home_final_score]
 * @property {number | null} [away_final_score]
 * @property {'schedule' | 'historical'} dataType
 */

// Fetch current injuries, caching for 30m
// backend/server/services/nba_service.js

// Fetch current injuries (no cache for now)
export const fetchNbaInjuries = async () => {
  console.log("→ [fetchNbaInjuries] querying injuries table…");

  // 1) Fetch
  const { data, error } = await supabase
    .from(NBA_INJURIES_TABLE)
    .select(
      `
      injury_id,
      player_id,
      player_display_name,
      team_id,
      team_display_name,
      report_date_utc,
      injury_status,
      injury_type,
      injury_detail
      `
    )
    .order("report_date_utc", { ascending: false });

  // 2) Error handling
  if (error) {
    console.error("→ [fetchNbaInjuries] Supabase error:", error);
    return [];
  }
  if (!Array.isArray(data)) {
    console.warn("→ [fetchNbaInjuries] Unexpected data:", data);
    return [];
  }

  console.log(`→ [fetchNbaInjuries] got ${data.length} rows`);

  // 3) Normalize & return
  const normalized = data.map((inj) => ({
    id: String(inj.injury_id),
    player: inj.player_display_name,
    team_display_name: inj.team_display_name,
    status: inj.injury_status || "N/A",
    detail: inj.injury_detail || "",
    updated: inj.report_date_utc,
    type: inj.injury_type || null,
  }));

  return normalized;
};

// Fetch historical games w/ pagination & filters
export const fetchNbaGameHistory = async ({
  startDate,
  endDate,
  teamName,
  limit,
  page,
}) => {
  const cacheKey = `nba_game_history_${startDate || "null"}_${
    endDate || "null"
  }_${teamName || "null"}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying historical games...`);
  let query = supabase.from(NBA_HISTORICAL_GAMES_TABLE).select(`
      game_id, game_date, home_team, away_team, home_score, away_score,
      home_q1, home_q2, home_q3, home_q4, home_ot,
      away_q1, away_q2, away_q3, away_q4, away_ot,
      home_assists, home_steals, home_blocks, home_turnovers, home_fouls, home_total_reb,
      away_assists, away_steals, away_blocks, away_turnovers, away_fouls, away_total_reb
    `);

  if (startDate) query = query.gte("game_date", startDate);
  if (endDate) query = query.lte("game_date", endDate);
  if (teamName)
    query = query.or(
      `home_team.ilike.%${teamName}%,away_team.ilike.%${teamName}%`
    );

  query = query.order("game_date", { ascending: false });
  query = query.range((page - 1) * limit, page * limit - 1);

  const { data, error } = await query;
  if (error) {
    console.error("Supabase error fetching historical games:", error);
    return null;
  }

  console.log(`Fetched ${data.length} historical games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

// Fetch team stats for a given season
export const fetchNbaTeamStatsBySeason = async (teamId, seasonYearStr) => {
  const cacheKey = `nba_team_stats_${teamId}_${seasonYearStr}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  const startYear = parseInt(seasonYearStr, 10);
  if (isNaN(startYear)) {
    console.error("Invalid season year:", seasonYearStr);
    return null;
  }
  const seasonRange = `${startYear}-${startYear + 1}`;

  console.log(
    `CACHE MISS: ${cacheKey}. Querying team stats for ${seasonRange}...`
  );
  const { data, error } = await supabase
    .from(NBA_HISTORICAL_TEAM_STATS_TABLE)
    .select("*")
    .eq("team_id", teamId)
    .eq("season", seasonRange)
    .maybeSingle();

  if (error) {
    console.error("Supabase error fetching team stats:", error);
    return null;
  }

  console.log(
    data
      ? `Fetched stats for team ${teamId}, caching ${ttl}s.`
      : `No stats found for team ${teamId}. Caching null.`
  );
  cache.set(cacheKey, data, ttl);
  return data;
};

/* ---------------------------------------------------------
 *  ALL-TEAMS season stats (used by Stats screen)
 *  GET /team-stats?season=YYYY
 * --------------------------------------------------------*/
export const fetchNbaAllTeamStatsBySeason = async (seasonYear) => {
  const seasonRange = `${seasonYear}-${seasonYear + 1}`;
  const cacheKey = `nba_all_team_stats_${seasonRange}`;
  const ttl = 60 * 30; // 30 min

  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying all-team stats…`);
  const { data, error } = await supabase
    .from(NBA_HISTORICAL_TEAM_STATS_TABLE)
    .select(
      "team_id, team_name, season, games_played_home, games_played_away, games_played_all, wins_home_percentage, wins_away_percentage, wins_all_percentage, points_for_avg_home, points_for_avg_away, points_for_avg_all, points_against_avg_home, points_against_avg_away, points_against_avg_all, current_form"
    )
    .eq("season", seasonRange)
    // Most UIs like to see best teams first
    .order("team_name", { ascending: true });

  if (error) {
    console.error("Supabase error fetching all-team stats:", {
      message: error.message,
      hint: error.hint,
      details: error.details,
      code: error.code,
    });

    // Still throw a clean error up to the controller
    const dbError = new Error(`Supabase query failed: ${error.message}`);
    dbError.status = 500;
    throw dbError;
  }
  cache.set(cacheKey, data || [], ttl);
  return data || [];
};

// Fetch player game history
/**
 * Fetches a player’s game history for a given season (July→June),
 * optionally filtering out games below a minutes threshold.
 *
 * @param {string|number} playerId
 * @param {number} seasonYear  // e.g. 2024 for the 2024–25 season
 * @param {object} opts
 * @param {number} [opts.limit=20]
 * @param {number} [opts.page=1]
 */
export const fetchNbaPlayerGameHistory = async (
  playerId,
  seasonYear,
  { limit = 20, page = 1 } = {}
) => {
  const cacheKey = `nba_player_history_${playerId}_${seasonYear}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying player history...`);

  // Build the date-window filters for July 1 of seasonYear → July 1 of next year
  const seasonStart = `${seasonYear}-07-01`;
  const seasonEnd = `${seasonYear + 1}-07-01`;

  // Build your Supabase query
  let query = supabase
    .from("nba_historical_player_stats")
    .select(
      `
      player_id,
      player_name,
      team_name,
      game_date,
      minutes,
      points,
      rebounds,
      assists,
      steals,
      blocks,
      turnovers,
      fouls,
      fg_made,
      fg_attempted,
      three_made,
      three_attempted,
      ft_made,
      ft_attempted
    `
    )
    // filter to only this season’s games
    .gte("game_date", seasonStart)
    .lt("game_date", seasonEnd)
    // filter out sub-minMp games
    .gte("minutes")
    // pagination
    .order("game_date", { ascending: false })
    .range((page - 1) * limit, page * limit - 1);

  const { data, error } = await query;
  if (error) {
    console.error("Supabase error fetching player history:", error);
    return null;
  }

  console.log(`Fetched ${data.length} player games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

// Another schedule helper used by your controller
export const WorkspaceNbaScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "nba_schedule_today_tomorrow";
  const ttl = 1800;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying schedule...`);
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  const todayStr = getUTCDateString(today);
  const tomorrowStr = getUTCDateString(tomorrow);

  const { data, error } = await supabase
    .from(NBA_SCHEDULE_TABLE)
    .select(
      `
      game_id, game_date, home_team, away_team, scheduled_time,
      venue, status, moneyline_clean, spread_clean, total_clean,
      predicted_home_score, predicted_away_score, updated_at
    `
    )
    .in("game_date", [todayStr, tomorrowStr])
    .order("scheduled_time", { ascending: true });

  if (error) {
    console.error("Supabase error:", error);
    return null;
  }

  console.log(`Fetched ${data.length} games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

/**
 * Fetches EITHER schedule/prediction data (today/future) OR
 * historical results (past dates) for NBA games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedNBAGameData[]>} - A promise resolving to an array of game data objects.
 */
/**
 * Fetches EITHER schedule/prediction data (today/future) OR
 * historical results (past dates) for NBA games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedNBAGameData[]>} - A promise resolving to an array of game data objects.
 */
export async function getSchedule(date) {
  console.log(`[nba_service getSchedule] Received date: ${date}`);
  let isPastDate = false;
  try {
    const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
    const inputDateEt = DateTime.fromISO(date, { zone: ET_ZONE_IDENTIFIER });
    if (!inputDateEt.isValid) throw new Error(`Invalid date: ${date}`);
    isPastDate = inputDateEt.startOf("day") < nowEt.startOf("day");
    console.log(
      `[nba_service getSchedule] Date ${date}. Is Past: ${isPastDate}`
    );
  } catch (e) {
    console.error(`[nba_service getSchedule] Error parsing date: ${date}`, e);
    throw new Error("Invalid date format.");
  }

  let data = null;
  let error = null;

  try {
    if (isPastDate) {
      const HISTORICAL_TABLE = NBA_HISTORICAL_GAMES_TABLE;
      console.log(
        `[nba_service] Fetching historical NBA data for ${date} from ${HISTORICAL_TABLE}`
      );
      // Select needed columns using actual DB names for historical table
      const historicalColumns = `game_id, game_date, home_team, away_team, home_score, away_score`;
      const response = await supabase
        .from(HISTORICAL_TABLE)
        .select(historicalColumns)
        .eq("game_date", date); // Assumes game_date column exists and works for filtering
      data = response.data;
      error = response.error;
    } else {
      const SCHEDULE_TABLE = NBA_SCHEDULE_TABLE;
      console.log(
        `[nba_service] Fetching schedule NBA data for ${date} from ${SCHEDULE_TABLE}`
      );
      // Select needed columns using actual DB names for schedule table
      const scheduleColumns = `game_id, game_date, home_team, away_team, scheduled_time, spread_clean, total_clean, predicted_home_score, predicted_away_score`;
      const response = await supabase
        .from(SCHEDULE_TABLE)
        .select(scheduleColumns)
        .eq("game_date", date) // Assumes game_date column exists and works for filtering
        .order("scheduled_time", { ascending: true }); // Assumes scheduled_time exists
      data = response.data;
      error = response.error;
    }

    console.log(
      `[nba_service getSchedule] Supabase returned ${
        data?.length ?? 0
      } rows for date ${date}. Error: ${error?.message ?? "No"}`
    );
    if (error) throw error;
    if (!Array.isArray(data)) {
      console.warn(
        `[nba_service] Supabase data not array for ${date}. Data:`,
        data
      );
      return [];
    }

    // --- CORRECTED: Map results to the UNIFIED Structure ---
    const results = data.map((row) => {
      if (isPastDate) {
        /** @type {UnifiedNBAGameData} */
        const gameData = {
          id: String(row.game_id),
          game_date: row.game_date,
          homeTeamName: row.home_team, // Map DB home_team -> unified homeTeamName
          awayTeamName: row.away_team, // Map DB away_team -> unified awayTeamName
          gameTimeUTC: null, // No time in historical table?
          statusState: "Final", // Assume 'Final' for historical
          spreadLine: null, // Standardized odds field
          totalLine: null, // Standardized odds field
          predictionHome: null,
          predictionAway: null,
          home_final_score: row.home_score, // Map DB home_score
          away_final_score: row.away_score, // Map DB away_score
          dataType: "historical",
        };
        return gameData;
      } else {
        /** @type {UnifiedNBAGameData} */
        const gameData = {
          id: String(row.game_id),
          game_date: row.game_date,
          homeTeamName: row.home_team, // Map DB home_team -> unified homeTeamName
          awayTeamName: row.away_team, // Map DB away_team -> unified awayTeamName
          gameTimeUTC: row.scheduled_time, // Map DB scheduled_time -> unified gameTimeUTC
          statusState: "Scheduled", // TODO: Use actual status column if available
          // Map DB _clean odds to standardized names & parse
          spreadLine: row.spread_clean
            ? parseFloat(
                (String(row.spread_clean).match(/-?\d+(\.\d+)?/) || ["0"])[0]
              )
            : null,
          totalLine: row.total_clean
            ? parseFloat(
                (String(row.total_clean).match(/\d+(\.\d+)?/) || ["0"])[0]
              )
            : null,
          predictionHome: row.predicted_home_score,
          predictionAway: row.predicted_away_score,
          home_final_score: null,
          away_final_score: null,
          dataType: "schedule",
        };
        return gameData;
      }
    });
    return results; // Return the mapped array matching UnifiedNBAGameData structure
  } catch (err) {
    console.error(
      `[nba_service getSchedule] Error processing date ${date}:`,
      err
    );
    throw err; // Let controller handle sending error response
  }
}
/* ---------------------------------------------------------
 * ALL-PLAYERS season stats
 * GET /player-stats?season=YYYY
 * --------------------------------------------------------*/
export const fetchNbaAllPlayerStatsBySeason = async (
  seasonYear,
  { search = null } // Signature should only accept { search }
) => {
  const cacheKey = `nba_all_player_stats_agg_${seasonYear}_${search || "null"}`;
  const ttl = 1800;
  // ... cache check ...

  console.log(
    `CACHE MISS (Aggregated): ${cacheKey}. Calling RPC (no minMp)...`
  );

  // rpcParams should ONLY contain p_season_year and p_search
  const rpcParams = {
    p_season_year: seasonYear,
    p_search: search,
  };

  const { data, error } = await supabase.rpc(
    "get_nba_player_season_stats",
    rpcParams // Pass the CORRECT params object
  );

  // Handle potential errors from the RPC call
  if (error) {
    console.error("Supabase RPC error fetching aggregated player stats:", {
      message: error.message,
      details: error.details, // Might contain more SQL-specific info
      hint: error.hint,
      code: error.code, // PostgreSQL error code
    });
    // Throw a generic error to be handled by the controller
    throw new Error(
      `Database function 'get_nba_player_season_stats' failed: ${error.message}`
    );
  }

  // The RPC function is designed to return the array of aggregated player stats.
  // If 'data' is null or not an array (unexpected), return an empty array.
  const results = Array.isArray(data) ? data : [];

  console.log(
    `Workspaceed ${results.length} aggregated player season stats via RPC. Caching for ${ttl}s.`
  );
  cache.set(cacheKey, results, ttl); // Cache the results
  return results; // Return the aggregated data
};
/**
 * Fetches calculated advanced team stats (Pace, Ratings, etc.) for a given NBA season using RPC.
 * @param {number} seasonYear - The starting year of the season (e.g., 2023 for 2023-24)
 * @returns {Promise<object[]>} - Array of team objects with advanced stats.
 */
export const fetchNbaAdvancedStatsBySeason = async (seasonYear) => {
  const cacheKey = `nba_advanced_stats_${seasonYear}`;
  // Advanced stats based on full season data, cache for a decent time (e.g., 6 hours)
  const ttl = 60 * 60 * 6; // 6 hours in seconds

  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return Array.isArray(cached) ? cached : [];
  }

  console.log(
    `CACHE MISS: ${cacheKey}. Calling RPC get_nba_advanced_team_stats...`
  );

  // Prepare parameters for the RPC call
  const rpcParams = {
    p_season_year: seasonYear,
  };

  // Call the RPC function
  const { data, error } = await supabase.rpc(
    "get_nba_advanced_team_stats", // Name of the function in Supabase
    rpcParams
  );

  if (error) {
    console.error("Supabase RPC error fetching advanced team stats:", {
      message: error.message,
      details: error.details,
      hint: error.hint,
      code: error.code,
    });
    // Throw error for the controller to catch
    throw new Error(
      `Database function 'get_nba_advanced_team_stats' failed: ${error.message}`
    );
  }

  const results = Array.isArray(data) ? data : [];

  console.log(
    `Workspaceed ${results.length} teams with advanced stats via RPC. Caching for ${ttl}s.`
  );
  cache.set(cacheKey, results, ttl); // Cache the results
  return results;
};
