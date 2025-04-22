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
export const fetchNbaInjuries = async () => {
  const cacheKey = "nba_injuries_current";
  const ttl = 1800; // 30 minutes
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    // Ensure cached data is always an array
    return Array.isArray(cached) ? cached : [];
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying injuries table...`);
  let data = null;
  let error = null;

  try {
    const response = await supabase
      .from(NBA_INJURIES_TABLE)
      .select(
        `injury_id, player_id, player_display_name, team_id, team_display_name,
               report_date_utc, injury_status, injury_status_abbr, injury_type,
               injury_location, injury_detail, injury_side, return_date_est,
               short_comment, long_comment, created_at, last_api_update_time`
      )
      .order("report_date_utc", { ascending: false });

    data = response.data;
    error = response.error;
  } catch (fetchError) {
    // Catch potential errors during the fetch itself
    console.error("Supabase fetch error in fetchNbaInjuries:", fetchError);
    error = fetchError; // Assign the fetch error
  }

  // --- Handle DB Errors ---
  if (error) {
    console.error("Supabase error fetching injuries:", error);
    // Return empty array instead of null for type consistency with frontend hook
    return [];
  }

  // --- Handle Missing/Invalid Data ---
  // Add check for data being null or not an array before mapping
  if (!data || !Array.isArray(data)) {
    console.warn(
      `[fetchNbaInjuries] No data returned or data is not an array. Data:`,
      data
    );
    cache.set(cacheKey, [], ttl); // Cache empty array
    return []; // Return empty array
  }

  // --- Data Normalization & Mapping ---
  // Map raw DB data to the structure expected by the frontend Injury type
  const normalizedInjuries = data.map((inj) => {
    // TODO: Add team name normalization if values differ from schedule tables
    // const canonicalTeamName = teamNameMap[inj.team_id] || inj.team_display_name;
    return {
      id: String(inj.injury_id), // Ensure ID is string
      player: inj.player_display_name,
      team_display_name: inj.team_display_name, // Use this name for grouping in frontend
      status: inj.injury_status || "N/A", // Map status, provide fallback
      detail: inj.injury_detail || "",
      updated: inj.report_date_utc,
      injury_type: inj.injury_type || null,
      // Include other fields if the frontend Injury type needs them
    };
  });

  // --- Caching & Return ---
  console.log(
    `Workspaceed ${normalizedInjuries.length} injuries. Caching for ${ttl}s.`
  );
  cache.set(cacheKey, normalizedInjuries, ttl);
  return normalizedInjuries; // Return the processed array
}; // <-- This is the correct end of the function

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

// Fetch player game history
export const fetchNbaPlayerGameHistory = async (playerId, { limit, page }) => {
  const cacheKey = `nba_player_history_${playerId}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying player history...`);
  let query = supabase
    .from(NBA_HISTORICAL_PLAYER_STATS_TABLE)
    .select(
      `
      game_id, player_id, player_name, team_id, team_name, game_date, minutes,
      points, rebounds, assists, steals, blocks, turnovers, fouls,
      fg_made, fg_attempted, three_made, three_attempted, ft_made, ft_attempted
    `
    )
    .eq("player_id", playerId)
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
